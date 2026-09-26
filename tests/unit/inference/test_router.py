"""Inference routing: model availability + remaining daily quota, one-line reasons
(inference/router.py)."""

from __future__ import annotations

from typing import Any

from gpu_router.clock import FakeClock
from gpu_router.inference.catalog import InferenceEntry
from gpu_router.inference.keys import key_status
from gpu_router.inference.ledger import InferenceLedger, LiveReading
from gpu_router.inference.models import InferRequest
from gpu_router.inference.router import estimate_need, fmt_amount, route, when
from tests.unit.inference.fakes import inference_catalog, login

NOW = 1_790_251_200.0  # Thu 2026-09-24 12:00 UTC
CAT = inference_catalog()


def req(model: str = "gpt-oss-20b", **kw: Any) -> InferRequest:
    return InferRequest(model=model, prompt=kw.pop("prompt", "What is 2+2?"), **kw)


def decide(r: InferRequest, led: InferenceLedger | None = None, clock: FakeClock | None = None):  # type: ignore[no-untyped-def]
    clock = clock or FakeClock(NOW)
    return route(r, CAT, led or InferenceLedger(None, CAT, clock), key_status, clock.now())


def test_picks_a_provider_that_serves_the_model_and_has_quota() -> None:
    login("groq", "cloudflare", "hf")
    d = decide(req())
    assert d.outcome == "place"
    assert [c.provider for c in d.candidates] == ["groq", "cloudflare", "hf"]
    chosen = d.chosen
    assert chosen is not None
    assert chosen.provider == "groq"
    assert chosen.model_id == "openai/gpt-oss-20b"
    # the binding counter is the tightest one after this request (here: tokens per day)
    assert chosen.reason.startswith("groq: gpt-oss-20b (openai/gpt-oss-20b) · ")
    assert "left today (est), resets in 12h" in chosen.reason
    assert "next: cloudflare" in d.reason
    assert "\n" not in d.reason
    # gemini does not serve it and is not even a candidate
    assert "gemini" not in {r.provider for r in d.rejected}


def test_a_provider_without_a_key_is_ruled_out_with_the_login_command() -> None:
    login("cloudflare")
    d = decide(req())
    assert d.chosen is not None
    assert d.chosen.provider == "cloudflare"
    rejected = {r.provider: r for r in d.rejected}
    assert rejected["groq"].code == "no_key"
    assert rejected["groq"].reason == "groq: needs a key (run `gpu login groq`)"
    assert rejected["hf"].reason == "hf: needs a key (run `gpu login hf`)"
    # a better-placed provider that was ruled out is named in the reason
    assert "groq: needs a key" in d.reason


def test_no_keys_at_all_is_no_fit_naming_every_login() -> None:
    d = decide(req())
    assert d.outcome == "no_fit"
    assert {r.code for r in d.rejected} == {"no_key"}
    assert "gpu login groq" in d.reason


def test_unknown_model_suggests_known_ones() -> None:
    login("groq")
    d = decide(req("gpt-oss-2b"))
    assert d.outcome == "no_fit"
    assert d.rejected == []
    assert "no free inference provider serves 'gpt-oss-2b'" in d.reason
    assert "did you mean gpt-oss-20b" in d.reason


def test_quota_left_steers_the_choice() -> None:
    login("groq", "cloudflare")
    clock = FakeClock(NOW)
    led = InferenceLedger(None, CAT, clock)
    # groq says only 1 request is left for this model today
    led.observe("groq", [LiveReading("openai/gpt-oss-20b", "requests", 1000, 1, NOW + 3600, NOW)])
    d = decide(req(), led, clock)
    assert d.chosen is not None
    assert d.chosen.provider == "cloudflare"
    # with nothing left it is ruled out with the live number and the reset
    led.observe("groq", [LiveReading("openai/gpt-oss-20b", "requests", 1000, 0, NOW + 3600, NOW)])
    d = decide(req(), led, clock)
    groq = {r.provider: r for r in d.rejected}["groq"]
    assert groq.code == "quota"
    assert groq.reason == (
        "groq: 0/1000 requests left today (live) for openai/gpt-oss-20b, resets in 1h"
    )
    assert groq.retry_at == NOW + 3600


def test_neurons_needed_by_a_big_request_are_checked() -> None:
    login("cloudflare")
    clock = FakeClock(NOW)
    led = InferenceLedger(None, CAT, clock)
    led.record("cloudflare", "x", {"neurons": 9990})
    d = decide(req("llama-3.3-70b", max_tokens=4096), led, clock)
    assert d.outcome == "wait"  # comes back at the reset
    r = {x.provider: x for x in d.rejected}["cloudflare"]
    assert r.code == "quota"
    assert "10/10k neurons left today (est)" in r.reason
    assert "needs ~" in r.reason


def test_exhausted_and_cooling_providers_wait_for_the_earliest() -> None:
    login("groq")
    clock = FakeClock(NOW)
    led = InferenceLedger(None, CAT, clock)
    led.block("groq", "openai/gpt-oss-20b", until=NOW + 20, kind="cooldown", why="429")
    d = decide(req(provider="groq"), led, clock)
    assert (d.outcome, d.retry_at) == ("wait", NOW + 20)
    assert "groq: cooling down in 1m" in d.reason
    led.block("groq", "", until=NOW + 7200, kind="exhausted", why="day")
    d = decide(req(provider="groq"), led, clock)
    assert d.rejected[0].code == "exhausted"
    assert "used up today, resets in 2h" in d.rejected[0].reason


def test_pins_unknown_providers_and_passthrough_ids() -> None:
    login("groq", "hf")
    pinned = decide(req(provider="hf"))
    assert pinned.chosen is not None
    assert pinned.chosen.provider == "hf"
    typo = decide(req(provider="grok"))
    assert typo.outcome == "no_fit"
    assert "did you mean groq" in typo.reason
    raw = decide(req("Qwen/Qwen3-8B"))
    assert raw.chosen is not None
    assert raw.chosen.provider == "hf"
    assert raw.chosen.unlisted
    assert "sent as-is" in raw.chosen.reason
    as_is = decide(req("some-new-model", provider="groq"))
    assert as_is.chosen is not None
    assert as_is.chosen.model_id == "some-new-model"
    zero = decide(req(provider="hf_zerogpu"))
    assert zero.outcome == "no_fit"
    assert "listed only" in zero.reason


def test_daily_allowances_are_spent_before_a_monthly_one() -> None:
    login("groq", "hf")
    d = decide(req())
    scores = {c.provider: c.score for c in d.candidates}
    assert scores["groq"] > scores["hf"]  # "use it or lose it": groq resets tonight


def test_unknown_limits_rank_after_known_ones_but_still_route() -> None:
    login("gemini")
    d = decide(req("gemini-3.5-flash"))
    assert d.chosen is not None
    assert d.chosen.provider == "gemini"
    assert "free limit unknown" in d.chosen.reason


def test_estimates_and_formatting() -> None:
    entry: InferenceEntry = CAT.entries["cloudflare"]
    model = entry.models["llama-3.1-8b"]
    need = estimate_need(entry, model, req("llama-3.1-8b", prompt="x" * 400, max_tokens=100))
    assert need.input_tokens == 108
    assert need.output_tokens == 100
    assert need.amounts["requests"] == 1
    assert need.amounts["tokens"] == 208
    assert need.amounts["neurons"] == (108 * 4119 + 100 * 34868) / 1e6
    assert fmt_amount(9450, "neurons") == "9450"
    assert fmt_amount(9749.5, "neurons") == "9.7k"
    assert fmt_amount(0.25, "neurons") == "0.2"
    assert fmt_amount(200_000, "tokens") == "200k"
    assert fmt_amount(0.1, "usd") == "$0.10"
    assert when(NOW + 30, NOW) == "in 1m"
    assert when(NOW + 5 * 3600, NOW) == "in 5h"
    assert when(NOW + 5 * 86400, NOW) == "in 5d"
