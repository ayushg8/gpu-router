"""InferenceService: route -> call -> record, fallbacks, waits, blocks, views
(inference/service.py), over mocked providers."""

from __future__ import annotations

import logging
from pathlib import Path

import httpx
import pytest

from gpu_router.clock import FakeClock
from gpu_router.errors import InvalidRequest
from gpu_router.inference.errors import (
    InferAuthRequired,
    InferBadRequest,
    InferQuotaExhausted,
    InferRateLimited,
    InferUnavailable,
)
from gpu_router.inference.models import InferRequest
from gpu_router.inference.service import InferenceService
from tests.unit.inference.fakes import (
    FakeProviders,
    groq_headers,
    inference_catalog,
    login,
    service,
)

NOW = 1_790_251_200.0  # Thu 2026-09-24 12:00 UTC


def req(model: str = "gpt-oss-20b", **kw: object) -> InferRequest:
    return InferRequest.model_validate({"model": model, "prompt": "What is 2+2?", **kw})


def test_answers_and_counts_the_request() -> None:
    login("groq", "cloudflare")
    fake = FakeProviders(groq_left=871)
    svc = service(fake, FakeClock(NOW))
    result = svc.infer(req())
    assert (result.provider, result.model, result.model_id, result.text) == (
        "groq",
        "gpt-oss-20b",
        "openai/gpt-oss-20b",
        "4",
    )
    assert (result.usage.input_tokens, result.usage.output_tokens) == (12, 3)
    assert result.route_reason.startswith("groq: gpt-oss-20b")
    assert (
        result.quota
        == "groq: 870/1000 requests left today for openai/gpt-oss-20b (live), resets in 12h"
    )
    assert svc.ledger.used("groq", "openai/gpt-oss-20b", "tokens") == 15
    assert svc.ledger.used("groq", "", "requests") == 1


def test_cloudflare_neurons_are_counted_from_tokens() -> None:
    login("cloudflare")
    fake = FakeProviders()
    svc = service(fake, FakeClock(NOW))
    result = svc.infer(req("llama-3.1-8b"))
    expected = (12 * 4119 + 3 * 34868) / 1e6
    assert result.usage.neurons == pytest.approx(expected, abs=1e-3)
    assert svc.ledger.used("cloudflare", "", "neurons") == pytest.approx(expected)
    [view] = [v for v in svc.quota() if v.provider == "cloudflare"]
    [counter] = [c for c in view.counters if c.unit == "neurons"]
    assert counter.source == "estimate"
    assert counter.limit == 10_000
    assert view.summary.startswith("cloudflare: 10k/10k neurons left today (est)")


def test_falls_back_to_the_next_provider_and_remembers_why() -> None:
    login("groq", "cloudflare")
    fake = FakeProviders()
    fake.queue(
        "groq",
        httpx.Response(429, json={"error": {"message": "slow"}}, headers={"retry-after": "30"}),
    )
    svc = service(fake, FakeClock(NOW))
    result = svc.infer(req())
    assert result.provider == "cloudflare"
    [fb] = result.fallbacks
    assert (fb.provider, fb.code) == ("groq", "rate_limited")
    # the cooldown is in the ledger: the next request goes straight to cloudflare
    again = svc.infer(req())
    assert again.provider == "cloudflare"
    assert again.fallbacks == []
    assert len(fake.calls("groq")) == 1


def test_a_used_up_day_blocks_the_model_until_the_reset() -> None:
    login("groq", "cloudflare")
    fake = FakeProviders()
    fake.queue("groq", httpx.Response(429, json={}, headers=groq_headers(0, reset="3h0m0s")))
    clock = FakeClock(NOW)
    svc = service(fake, clock)
    assert svc.infer(req()).provider == "cloudflare"
    [block] = svc.ledger.blocks("groq", "openai/gpt-oss-20b")
    assert (block.kind, block.until) == ("exhausted", NOW + 3 * 3600)
    assert svc.ledger.blocks("groq", "openai/gpt-oss-120b") == []  # per model on groq
    view = next(v for v in svc.quota() if v.provider == "groq")
    assert "openai/gpt-oss-20b" in view.blocked


def test_everything_used_up_is_quota_exhausted_with_the_reset() -> None:
    login("groq")
    fake = FakeProviders()
    fake.queue("groq", httpx.Response(429, json={}, headers=groq_headers(0, reset="3h0m0s")))
    svc = service(fake, FakeClock(NOW))
    with pytest.raises(InferQuotaExhausted) as info:
        svc.infer(req(provider="groq"))
    assert info.value.detail["fallbacks"][0]["provider"] == "groq"
    with pytest.raises(InferQuotaExhausted) as again:  # routed away without a call
        svc.infer(req(provider="groq"))
    assert again.value.resets_at == NOW + 3 * 3600
    assert len(fake.calls("groq")) == 1


def test_a_short_cooldown_is_waited_out_once() -> None:
    login("groq")
    fake = FakeProviders()
    fake.queue(
        "groq",
        httpx.Response(429, json={"error": {"message": "rpm"}}, headers={"retry-after": "2"}),
    )
    sleeps: list[float] = []
    svc = service(fake, FakeClock(NOW), sleeps=sleeps)
    result = svc.infer(req(provider="groq", wait_s=10))
    assert result.provider == "groq"
    assert sleeps
    assert 2 <= sleeps[0] <= 2.1
    assert len(fake.calls("groq")) == 2
    # without a wait budget the caller hears when to come back
    fake.queue("groq", httpx.Response(429, json={}, headers={"retry-after": "20"}))
    with pytest.raises(InferRateLimited) as info:
        svc.infer(req(provider="groq"))
    assert info.value.retry_after is not None


def test_a_bad_request_is_not_retried_elsewhere() -> None:
    login("groq", "cloudflare")
    fake = FakeProviders()
    fake.queue("groq", httpx.Response(400, json={"error": {"message": "context length exceeded"}}))
    svc = service(fake, FakeClock(NOW))
    with pytest.raises(InferBadRequest):
        svc.infer(req())
    assert fake.calls("cloudflare") == []


def test_missing_keys_and_unknown_models() -> None:
    fake = FakeProviders()
    svc = service(fake, FakeClock(NOW))
    with pytest.raises(InferAuthRequired) as info:
        svc.infer(req())
    assert "`gpu login groq`" in (info.value.hint or "")
    with pytest.raises(InvalidRequest, match="no free inference provider serves"):
        svc.infer(req("nope-model"))
    assert fake.calls() == []


def test_a_rejected_key_is_blocked_until_someone_logs_in_again() -> None:
    login("groq", "cloudflare")
    fake = FakeProviders()
    fake.queue("groq", httpx.Response(401, json={"error": {"message": "Invalid API Key"}}))
    svc = service(fake, FakeClock(NOW))
    assert svc.infer(req()).provider == "cloudflare"
    [block] = svc.ledger.blocks("groq")
    assert block.kind == "auth"


def test_an_outage_cools_the_provider_down() -> None:
    login("groq", "hf")
    fake = FakeProviders()
    fake.queue("groq", httpx.Response(503, text="down"))
    svc = service(fake, FakeClock(NOW))
    assert svc.infer(req()).provider == "hf"
    assert svc.ledger.blocks("groq")[0].kind == "cooldown"


def test_a_5xx_on_a_per_model_provider_cools_down_that_model_only() -> None:
    # live 2026-09-25: Gemini answered 503 "This model is currently experiencing high
    # demand" for one model while the others worked
    login("gemini")
    fake = FakeProviders()
    fake.queue(
        "gemini",
        httpx.Response(503, json={"error": {"message": "This model is currently busy."}}),
    )
    svc = service(fake, FakeClock(NOW))
    with pytest.raises(InferUnavailable, match="every free provider") as info:
        svc.infer(req("gemini-3.8-flash"))
    [block] = svc.ledger.blocks("gemini")
    assert (block.scope, block.kind) == ("gemini-3.8-flash", "cooldown")
    assert svc.infer(req("gemini-3.5-flash")).provider == "gemini"  # not blocked
    # the error shows the decision that chose gemini, not the re-route that skipped it
    route = info.value.detail["route"]
    assert isinstance(route, dict)
    assert route["outcome"] == "place"
    assert route["chosen"]["provider"] == "gemini"
    assert [f["provider"] for f in info.value.detail["fallbacks"]] == ["gemini"]


def test_no_answer_at_all_cools_the_whole_provider_down() -> None:
    login("gemini")
    fake = FakeProviders()
    fake.queue("gemini", httpx.ConnectError("connection refused"))
    svc = service(fake, FakeClock(NOW))
    with pytest.raises(InferUnavailable):
        svc.infer(req("gemini-3.8-flash"))
    [block] = svc.ledger.blocks("gemini")
    assert block.scope == ""


def test_test_mode_never_calls_a_real_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    login("groq")
    monkeypatch.delenv("GPU_ROUTER_REAL_PROVIDERS", raising=False)
    svc = InferenceService(inference_catalog(), FakeClock(NOW), ledger_file=None, test_mode=True)
    with pytest.raises(InferUnavailable, match="test mode"):
        svc.infer(req(provider="groq"))
    assert svc.ledger.blocks("groq") == []  # not the provider's fault


def test_views(tmp_path: Path) -> None:
    login("groq")
    fake = FakeProviders()
    svc = service(fake, FakeClock(NOW), ledger_file=tmp_path / "ledger.json")
    views = {v.name: v for v in svc.providers()}
    assert views["groq"].logged_in
    assert views["groq"].missing == []
    assert not views["gemini"].logged_in
    assert views["gemini"].missing == ["INFER_GEMINI_API_KEY"]
    assert views["cloudflare"].login == "cloudflare"
    assert views["hf"].login == "hf"
    assert views["hf"].missing == ["HF_TOKEN"]
    assert not views["hf_zerogpu"].routable
    assert views["groq"].models["gpt-oss-20b"] == "openai/gpt-oss-20b"
    quota = {q.provider: q for q in svc.quota()}
    assert quota["gemini"].summary == "gemini: 0 requests today, free limit unknown, resets in 19h"
    assert quota["hf"].summary.startswith("hf: $0.10/$0.10 credits left this month (est)")
    assert quota["hf_zerogpu"].summary.startswith("hf_zerogpu: listed only")


def test_prompts_and_replies_are_never_logged(caplog: pytest.LogCaptureFixture) -> None:
    login("groq")
    fake = FakeProviders()
    fake.queue(
        "groq",
        httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "SECRET-REPLY"}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2},
            },
        ),
    )
    svc = service(fake, FakeClock(NOW))
    with caplog.at_level(logging.DEBUG):
        svc.infer(InferRequest(model="gpt-oss-20b", prompt="SECRET-PROMPT"))
    text = "\n".join(f"{r.getMessage()} {getattr(r, 'fields', '')}" for r in caplog.records)
    assert "infer.call" in {getattr(r, "event", "") for r in caplog.records}
    assert "SECRET-PROMPT" not in text
    assert "SECRET-REPLY" not in text


# --------------------------------------------------------------------------- review fixes (D54)


def test_a_read_timeout_counts_an_estimate_and_the_deadline_stops_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Review fix: infer() had no overall deadline (up to 8 calls of 120 s while the
    client gave up at 420 s), and a read timeout recorded no usage although the provider
    most likely finished and billed the generation."""
    login("groq", "cloudflare")
    fake = FakeProviders()
    clock = FakeClock(NOW)

    def slow(request: httpx.Request) -> httpx.Response:
        clock.advance(206)  # the provider thinks for a long time...
        raise httpx.ReadTimeout("read timed out", request=request)  # ...past our timeout

    fake.queue("groq", slow)
    svc = service(fake, clock)
    with pytest.raises(InferUnavailable, match="within 210s") as info:
        svc.infer(req(deadline_s=210))
    assert [f["provider"] for f in info.value.detail["fallbacks"]] == ["groq"]
    assert fake.calls("cloudflare") == []  # 4 s left: no second provider call
    # the timed-out generation is counted as an estimate (tokens + the request)
    assert svc.ledger.used("groq", "openai/gpt-oss-20b", "tokens") > 0
    assert svc.ledger.used("groq", "", "requests") == 1


def test_a_connect_timeout_counts_nothing() -> None:
    login("groq", "cloudflare")
    fake = FakeProviders()
    fake.queue("groq", httpx.ConnectTimeout("no route"))
    svc = service(fake, FakeClock(NOW))
    result = svc.infer(req())
    assert result.provider == "cloudflare"
    assert svc.ledger.used("groq", "", "requests") == 0


def test_the_read_timeout_grows_with_max_tokens_and_stays_inside_the_deadline() -> None:
    login("groq")
    fake = FakeProviders()
    seen: list[dict[str, float]] = []

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(dict(request.extensions["timeout"]))
        return httpx.Response(200, json={"choices": [{"message": {"content": "4"}}]})

    fake.queue("groq", record, record, record)
    svc = service(fake, FakeClock(NOW))
    svc.infer(req())
    svc.infer(req(max_tokens=20_000))
    svc.infer(req(max_tokens=20_000, deadline_s=60))
    assert [t["read"] for t in seen] == [120.0, 230.0, 60.0]


def test_a_provider_with_every_slot_taken_is_skipped_without_a_ledger_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Review fix: a request parked on a provider's BoundedSemaphore had no timeout and
    held a daemon executor thread for minutes."""
    from gpu_router.inference import service as svc_mod

    monkeypatch.setattr(svc_mod, "SLOT_WAIT_S", 0.05)
    login("groq", "cloudflare")
    fake = FakeProviders()
    svc = service(fake, FakeClock(NOW))
    svc._client(svc.catalog.entries["groq"])
    for _ in range(svc_mod.PER_PROVIDER_CONCURRENCY):
        assert svc._slots["groq"].acquire(timeout=0)
    result = svc.infer(req())
    assert result.provider == "cloudflare"
    assert [f.code for f in result.fallbacks] == ["provider_unavailable"]
    assert "requests running" in result.fallbacks[0].message
    assert fake.calls("groq") == []
    assert svc.ledger.blocks("groq") == []  # our own limit teaches the ledger nothing
    svc.close()
