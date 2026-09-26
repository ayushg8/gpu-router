"""The inference section of providers.yaml, typed (inference/catalog.py)."""

from __future__ import annotations

from datetime import date

import pytest

from gpu_router.inference.catalog import load_inference_catalog, parse_entry
from gpu_router.providers.catalog import load_catalog
from tests.unit.inference.fakes import inference_catalog


def test_packaged_inference_entries_parse() -> None:
    cat = inference_catalog()
    assert cat.problems == {}
    assert [e.name for e in cat.ordered()] == ["groq", "cloudflare", "gemini", "hf", "hf_zerogpu"]
    for name in ("groq", "cloudflare", "gemini", "hf"):
        e = cat.entries[name]
        assert e.routable
        assert e.base_url
        assert e.base_url.startswith("https://")
        assert "api_key" in e.secrets
        assert e.verified_at is not None
        assert e.verified_at >= date(2026, 9, 24)
        assert e.docs
    assert not cat.entries["hf_zerogpu"].routable  # Spaces, not a chat endpoint


def test_free_limits_are_data_with_their_reset() -> None:
    cat = inference_catalog()
    cf = cat.entries["cloudflare"]
    assert (cf.limits["neurons"], cf.reset) == (10_000, "daily_utc")
    groq = cat.entries["groq"]
    m = groq.models["gpt-oss-20b"]
    assert (m.id, m.rpm, m.limits["requests"], m.limits["tokens"]) == (
        "openai/gpt-oss-20b",
        30,
        1000,
        200_000,
    )
    assert groq.quota_scope == "model"
    assert cat.entries["gemini"].reset == "daily_pacific"
    assert cat.entries["gemini"].limits == {}  # AI Studio shows them; nothing invented
    hf = cat.entries["hf"]
    assert (hf.limits["usd"], hf.reset, hf.secrets["api_key"]) == (0.10, "monthly", "HF_TOKEN")
    assert cat.entries["hf_zerogpu"].limits["gpu_seconds"] == 300


def test_resolve_alias_listed_id_and_passthrough() -> None:
    cat = inference_catalog()
    cf = cat.entries["cloudflare"]
    assert cf.resolve("llama-3.1-8b").id == "@cf/meta/llama-3.1-8b-instruct-fp8-fast"
    assert cf.resolve("@cf/openai/gpt-oss-20b").alias == "gpt-oss-20b"
    unlisted = cf.resolve("@cf/some/new-model")
    assert unlisted is not None
    assert unlisted.unlisted
    assert cf.resolve("gemini-3.5-flash") is None
    assert cat.entries["hf"].resolve("Qwen/Qwen3-8B").unlisted
    assert cat.entries["groq"].resolve("llama-3.1-8b") is None
    assert "gpt-oss-20b" in cat.aliases()
    assert "gemini-3.5-flash" in cat.aliases()


def test_limit_scope_and_rates() -> None:
    cat = inference_catalog()
    groq = cat.entries["groq"]
    m = groq.models["gpt-oss-20b"]
    assert groq.limit(m, "requests") == ("openai/gpt-oss-20b", 1000)
    assert groq.limit(m, "neurons") is None
    cf = cat.entries["cloudflare"]
    llama = cf.models["llama-3.1-8b"]
    assert cf.limit(llama, "neurons") == ("", 10_000)
    assert cf.neurons(llama, 1_000_000, 0) == pytest.approx(4119)
    assert cf.neurons(llama, 1000, 1000) == pytest.approx((4119 + 34868) / 1000)
    assert cf.neurons(cf.resolve("@cf/x/y"), 1_000_000, 0) == pytest.approx(60_000)  # dearest
    hf = cat.entries["hf"]
    assert hf.usd(hf.models["gpt-oss-20b"], 1_000_000, 0) == pytest.approx(0.5)


def test_a_broken_entry_is_a_problem_not_a_crash() -> None:
    cat = load_catalog()
    raw = {**cat.inference, "bad": {"display_name": "Bad", "base_url": "https://x"}}
    typed = load_inference_catalog(cat.model_copy(update={"inference": raw}))
    assert "bad" in typed.problems
    assert "api_key" in typed.problems["bad"]
    assert "groq" in typed.entries
    with pytest.raises(ValueError, match="models must be a mapping"):
        parse_entry("x", {"display_name": "X", "models": ["a"]})
    with pytest.raises(ValueError, match="lowercase"):
        parse_entry("Bad Name", {"display_name": "X", "client": "none"})
