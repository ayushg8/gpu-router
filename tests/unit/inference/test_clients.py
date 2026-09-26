"""The OpenAI-compatible client against mocked Groq, Cloudflare, Gemini and HF endpoints
(inference/clients.py). No network: httpx.MockTransport only."""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from gpu_router.inference.clients import ChatClient, classify, live_readings, parse_duration
from gpu_router.inference.errors import (
    InferAuthRequired,
    InferBadRequest,
    InferModelUnavailable,
    InferQuotaExhausted,
    InferRateLimited,
    InferUnavailable,
)
from gpu_router.inference.ledger import window_bounds
from gpu_router.inference.models import InferMessage
from tests.unit.inference.fakes import KEYS, FakeProviders, groq_headers, inference_catalog

NOW = 1_790_251_200.0  # Thu 2026-09-24 12:00 UTC
CAT = inference_catalog()
MSGS = [InferMessage(role="system", content="be brief"), InferMessage(role="user", content="2+2?")]


def keys(provider: str) -> dict[str, SecretStr]:
    if provider == "cloudflare":
        return {
            "api_key": SecretStr(KEYS["INFER_CLOUDFLARE_API_TOKEN"]),
            "account_id": SecretStr(KEYS["INFER_CLOUDFLARE_ACCOUNT_ID"]),
        }
    name = {"groq": "INFER_GROQ_API_KEY", "gemini": "INFER_GEMINI_API_KEY", "hf": "HF_TOKEN"}
    return {"api_key": SecretStr(KEYS[name[provider]])}


def chat(fake: FakeProviders, provider: str, model_id: str, **kw: Any) -> Any:
    client = ChatClient(CAT.entries[provider], transport=fake.transport())
    try:
        return client.chat(
            keys(provider),
            model_id,
            MSGS,
            max_tokens=kw.get("max_tokens"),
            temperature=kw.get("temperature"),
            now=NOW,
        )
    finally:
        client.close()


@pytest.mark.parametrize(
    ("provider", "model_id", "url"),
    [
        ("groq", "openai/gpt-oss-20b", "https://api.groq.com/openai/v1/chat/completions"),
        (
            "cloudflare",
            "@cf/meta/llama-3.1-8b-instruct-fp8-fast",
            "https://api.cloudflare.com/client/v4/accounts/acct0123456789abcdef/ai/v1/chat/completions",
        ),
        (
            "gemini",
            "gemini-3.5-flash",
            "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        ),
        ("hf", "openai/gpt-oss-20b", "https://router.huggingface.co/v1/chat/completions"),
    ],
)
def test_each_provider_gets_an_openai_chat_request(provider: str, model_id: str, url: str) -> None:
    fake = FakeProviders()
    reply = chat(fake, provider, model_id, max_tokens=64, temperature=0.2)
    [seen] = fake.calls()
    assert seen.method == "POST"
    assert seen.url == url
    assert seen.auth == f"Bearer {keys(provider)['api_key'].get_secret_value()}"
    assert seen.body == {
        "model": model_id,
        "messages": [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "2+2?"},
        ],
        "stream": False,
        "max_tokens": 64,
        "temperature": 0.2,
    }
    assert (reply.text, reply.finish_reason, reply.input_tokens, reply.output_tokens) == (
        "4",
        "stop",
        12,
        3,
    )
    assert reply.latency_s >= 0


def test_groq_headers_are_a_live_requests_per_day_reading() -> None:
    fake = FakeProviders(groq_left=871)
    reply = chat(fake, "groq", "openai/gpt-oss-20b")
    [r] = reply.live
    assert (r.scope, r.unit, r.limit, r.remaining) == ("openai/gpt-oss-20b", "requests", 1000, 870)
    assert r.resets_at == pytest.approx(NOW + 86.4)
    # other providers' headers are not interpreted
    assert live_readings(CAT.entries["gemini"], "m", httpx.Headers(groq_headers(5)), NOW) == []


def test_optional_fields_are_left_out_and_content_parts_are_joined() -> None:
    fake = FakeProviders()
    parts = {
        "choices": [{"message": {"content": [{"type": "text", "text": "a"}, {"text": "b"}]}}],
    }
    fake.queue("hf", httpx.Response(200, json=parts))
    reply = chat(fake, "hf", "openai/gpt-oss-20b")
    assert fake.calls()[0].body is not None
    assert "max_tokens" not in fake.calls()[0].body
    assert (reply.text, reply.input_tokens, reply.finish_reason) == ("ab", None, None)


def err(status: int, body: Any, headers: dict[str, str] | None = None) -> httpx.Response:
    if isinstance(body, str):
        return httpx.Response(status, text=body, headers=headers)
    return httpx.Response(status, json=body, headers=headers)


GROQ_TPM = {
    "error": {
        "message": "Rate limit reached for model `openai/gpt-oss-20b` on tokens per minute "
        "(TPM): Limit 8000, Used 7990, Requested 500. Please try again in 3.7s.",
        "type": "tokens",
        "code": "rate_limit_exceeded",
    }
}
GEMINI_DAY = [
    {
        "error": {
            "code": 429,
            "message": "You exceeded your current quota.",
            "status": "RESOURCE_EXHAUSTED",
            "details": [
                {
                    "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                    "violations": [
                        {"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}
                    ],
                },
                {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "36000s"},
            ],
        }
    }
]
GEMINI_MINUTE = [
    {
        "error": {
            "code": 429,
            "message": "Quota exceeded for metric generate_content_free_tier_requests",
            "details": [
                {
                    "violations": [
                        {"quotaId": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"}
                    ]
                },
                {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "33s"},
            ],
        }
    }
]
CF_NEURONS = {
    "success": False,
    "errors": [
        {
            "code": 4006,
            "message": "you have used up your daily free allocation of 10,000 neurons, "
            "please upgrade to Cloudflare's Workers Paid plan",
        }
    ],
}


def test_rate_limits_are_cooldowns_with_retry_after() -> None:
    fake = FakeProviders()
    fake.queue("groq", err(429, GROQ_TPM, {"retry-after": "4", **groq_headers(500)}))
    with pytest.raises(InferRateLimited) as info:
        chat(fake, "groq", "openai/gpt-oss-20b")
    assert info.value.retry_after == 4
    assert info.value.live
    assert info.value.live[0].remaining == 500
    assert "tokens per minute" in info.value.message
    fake.queue("gemini", err(429, GEMINI_MINUTE))
    with pytest.raises(InferRateLimited) as g:
        chat(fake, "gemini", "gemini-3.5-flash")
    assert g.value.retry_after == 33


def test_a_used_up_day_is_quota_exhausted_until_the_reset() -> None:
    fake = FakeProviders()
    fake.queue(
        "groq", err(429, {"error": {"message": "slow down"}}, groq_headers(0, reset="5h0m0s"))
    )
    with pytest.raises(InferQuotaExhausted) as groq:
        chat(fake, "groq", "openai/gpt-oss-20b")
    assert (groq.value.scope, groq.value.resets_at) == ("openai/gpt-oss-20b", NOW + 5 * 3600)
    fake.queue("gemini", err(429, GEMINI_DAY))
    with pytest.raises(InferQuotaExhausted) as gem:
        chat(fake, "gemini", "gemini-3.5-flash")
    assert gem.value.scope == "gemini-3.5-flash"
    assert gem.value.resets_at == NOW + 36000
    fake.queue("cloudflare", err(429, CF_NEURONS))
    with pytest.raises(InferQuotaExhausted) as cf:
        chat(fake, "cloudflare", "@cf/meta/llama-3.2-1b-instruct")
    assert cf.value.scope == ""  # neurons are account-wide
    assert cf.value.resets_at == window_bounds("daily_utc", NOW)[1]
    assert "10,000 neurons" in cf.value.message
    fake.queue("hf", err(402, {"error": "You have exceeded your monthly included credits"}))
    with pytest.raises(InferQuotaExhausted) as hf:
        chat(fake, "hf", "openai/gpt-oss-20b")
    assert hf.value.resets_at == window_bounds("monthly", NOW)[1]
    assert "never buys" in (hf.value.hint or "")


@pytest.mark.parametrize(
    ("status", "body", "cls"),
    [
        (401, {"error": {"message": "Invalid API Key"}}, InferAuthRequired),
        (403, {"error": {"message": "forbidden"}}, InferAuthRequired),
        (404, {"error": {"message": "The model `x` does not exist"}}, InferModelUnavailable),
        (400, {"error": {"message": "model x is not supported"}}, InferModelUnavailable),
        (400, {"error": {"message": "context length exceeded"}}, InferBadRequest),
        (413, "too large", InferBadRequest),
        (500, "boom", InferUnavailable),
        (503, {"error": {"message": "over capacity"}}, InferUnavailable),
    ],
)
def test_error_taxonomy(status: int, body: Any, cls: type) -> None:
    fake = FakeProviders()
    fake.queue("groq", err(status, body))
    with pytest.raises(cls) as info:
        chat(fake, "groq", "openai/gpt-oss-20b")
    assert info.value.provider == "groq"
    assert info.value.reroute is (cls is not InferBadRequest)
    if cls is InferAuthRequired:
        assert info.value.hint == "run `gpu login groq`"


def test_network_errors_and_timeouts_are_unavailable() -> None:
    fake = FakeProviders()
    fake.queue("groq", httpx.ConnectError("nope"), httpx.ReadTimeout("slow"))
    with pytest.raises(InferUnavailable, match="cannot reach groq"):
        chat(fake, "groq", "openai/gpt-oss-20b")
    with pytest.raises(InferUnavailable, match="did not answer in time"):
        chat(fake, "groq", "openai/gpt-oss-20b")
    fake.queue("groq", httpx.Response(200, text="<html>"), httpx.Response(200, json={"x": 1}))
    with pytest.raises(InferUnavailable, match="not JSON"):
        chat(fake, "groq", "openai/gpt-oss-20b")
    with pytest.raises(InferUnavailable, match="without a completion"):
        chat(fake, "groq", "openai/gpt-oss-20b")


def test_keys_never_reach_error_messages() -> None:
    fake = FakeProviders()
    key = KEYS["INFER_GROQ_API_KEY"]
    acct = KEYS["INFER_CLOUDFLARE_ACCOUNT_ID"]
    fake.queue("groq", err(401, {"error": {"message": f"bad key {key}"}}))
    with pytest.raises(InferAuthRequired) as info:
        chat(fake, "groq", "openai/gpt-oss-20b")
    assert key not in info.value.message
    fake.queue("cloudflare", httpx.ConnectError(f"https://api.cloudflare.com/accounts/{acct}"))
    with pytest.raises(InferUnavailable) as cf:
        chat(fake, "cloudflare", "@cf/meta/llama-3.2-1b-instruct")
    assert acct not in cf.value.message
    long = "x" * 1000
    e = classify(CAT.entries["groq"], "m", 500, httpx.Headers(), long, NOW)
    assert len(e.message) < 400


def test_verify_uses_the_free_check_endpoint() -> None:
    fake = FakeProviders()
    for provider, url in (
        ("groq", "https://api.groq.com/openai/v1/models"),
        ("gemini", "https://generativelanguage.googleapis.com/v1beta/openai/models"),
        (
            "cloudflare",
            "https://api.cloudflare.com/client/v4/accounts/acct0123456789abcdef/ai/models/search?per_page=1",
        ),
    ):
        client = ChatClient(CAT.entries[provider], transport=fake.transport())
        client.verify(keys(provider))
        client.close()
        assert fake.calls(provider)[-1].url == url
        assert fake.calls(provider)[-1].method == "GET"
    fake.queue("groq", httpx.Response(401))
    client = ChatClient(CAT.entries["groq"], transport=fake.transport())
    with pytest.raises(InferAuthRequired, match="rejected the key"):
        client.verify(keys("groq"))
    fake.queue("cloudflare", httpx.Response(404))
    cf = ChatClient(CAT.entries["cloudflare"], transport=fake.transport())
    with pytest.raises(InferAuthRequired, match="account id"):
        cf.verify(keys("cloudflare"))


def test_parse_duration() -> None:
    assert parse_duration("2m59.56s") == pytest.approx(179.56)
    assert parse_duration("7.66s") == pytest.approx(7.66)
    assert parse_duration("250ms") == pytest.approx(0.25)
    assert parse_duration("1h2m") == 3720
    assert parse_duration("12") == 12
    assert parse_duration("soon") is None
    assert parse_duration(None) is None


#: the Gemini API's answers to an invalid key, captured live 2026-09-25 (a made-up key)
GEMINI_BAD_KEY_MODELS = (
    '{"error": {"code": 400, "message": "Please pass a valid API key", '
    '"status": "INVALID_ARGUMENT"}}'
)
GEMINI_BAD_KEY_CHAT = f"[{GEMINI_BAD_KEY_MODELS}]"
GEMINI_BAD_KEY_NATIVE = (
    '{"error": {"code": 400, "message": "API key not valid. Please pass a valid API key.", '
    '"status": "INVALID_ARGUMENT", "details": [{"@type": '
    '"type.googleapis.com/google.rpc.ErrorInfo", "reason": "API_KEY_INVALID"}]}}'
)


@pytest.mark.parametrize(
    "body", [GEMINI_BAD_KEY_CHAT, GEMINI_BAD_KEY_MODELS, GEMINI_BAD_KEY_NATIVE]
)
def test_geminis_400_for_a_bad_key_is_auth_not_a_bad_request(body: str) -> None:
    """Review fix: Gemini answers a bad key with 400, not 401/403, so it read as
    invalid_request (no login hint, no ledger block, a batch never stopped early)."""
    err = classify(CAT.entries["gemini"], "gemini-3.5-flash", 400, httpx.Headers(), body, NOW)
    assert isinstance(err, InferAuthRequired)
    assert err.hint == "run `gpu login gemini`"
    fake = FakeProviders()
    fake.queue("gemini", httpx.Response(400, text=body))
    with pytest.raises(InferAuthRequired):
        chat(fake, "gemini", "gemini-3.5-flash")
    # verify(): `gpu login gemini` refuses the key instead of "stored anyway"
    fake.queue("gemini", httpx.Response(400, text=body))
    client = ChatClient(CAT.entries["gemini"], transport=fake.transport())
    with pytest.raises(InferAuthRequired, match="rejected the key"):
        client.verify(keys("gemini"))
    client.close()


def test_an_ordinary_400_stays_a_bad_request() -> None:
    body = '{"error": {"code": 400, "message": "max_tokens must be positive"}}'
    err = classify(CAT.entries["gemini"], "gemini-3.5-flash", 400, httpx.Headers(), body, NOW)
    assert isinstance(err, InferBadRequest)
