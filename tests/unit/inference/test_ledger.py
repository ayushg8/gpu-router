"""The inference lane's daily-quota ledger (inference/ledger.py)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from gpu_router.clock import FakeClock
from gpu_router.inference.ledger import InferenceLedger, LiveReading, window_bounds
from tests.unit.inference.fakes import inference_catalog

# Thu 2026-09-24 12:00 UTC (05:00 PDT)
NOW = datetime(2026, 9, 24, 12, 0, tzinfo=UTC).timestamp()
MIDNIGHT_UTC = datetime(2026, 9, 25, 0, 0, tzinfo=UTC).timestamp()
MIDNIGHT_PT = datetime(2026, 9, 25, 7, 0, tzinfo=UTC).timestamp()  # 00:00 PDT
GROQ_20B = "openai/gpt-oss-20b"


def ledger(clock: FakeClock, path: Path | None = None) -> InferenceLedger:
    return InferenceLedger(path, inference_catalog(), clock)


def test_windows() -> None:
    assert window_bounds("daily_utc", NOW) == (
        datetime(2026, 9, 24, tzinfo=UTC).timestamp(),
        MIDNIGHT_UTC,
    )
    start, end = window_bounds("daily_pacific", NOW)
    assert end == MIDNIGHT_PT
    assert start == MIDNIGHT_PT - 86_400
    # the day DST ends in the US (2026-11-01) is 25 hours long in Pacific time
    nov1 = datetime(2026, 11, 1, 12, 0, tzinfo=UTC).timestamp()
    s, e = window_bounds("daily_pacific", nov1)
    assert e - s == 25 * 3600
    assert window_bounds("monthly", NOW) == (
        datetime(2026, 9, 1, tzinfo=UTC).timestamp(),
        datetime(2026, 10, 1, tzinfo=UTC).timestamp(),
    )
    assert window_bounds("monthly", datetime(2026, 12, 31, 23, tzinfo=UTC).timestamp())[1] == (
        datetime(2027, 1, 1, tzinfo=UTC).timestamp()
    )
    assert window_bounds("rolling_24h", NOW, first_use=NOW - 3600) == (NOW - 3600, NOW + 82_800)
    assert window_bounds("rolling_24h", NOW, first_use=NOW - 90_000) == (NOW, NOW + 86_400)


def test_estimates_count_our_use_per_model_and_provider() -> None:
    clock = FakeClock(NOW)
    led = ledger(clock)
    assert led.remaining("groq", GROQ_20B, "requests").remaining == 1000  # type: ignore[union-attr]
    led.record("groq", GROQ_20B, {"requests": 1, "tokens": 900})
    led.record("groq", GROQ_20B, {"requests": 1, "tokens": 100})
    rem = led.remaining("groq", GROQ_20B, "requests")
    assert rem is not None
    assert (rem.remaining, rem.limit, rem.source, rem.resets_at) == (
        998,
        1000,
        "estimate",
        MIDNIGHT_UTC,
    )
    assert led.remaining("groq", GROQ_20B, "tokens").remaining == 199_000  # type: ignore[union-attr]
    assert led.used("groq", "", "requests") == 2  # provider-wide count too
    assert led.remaining("groq", "", "requests") is None  # no provider-wide limit
    led.record(
        "cloudflare", "@cf/meta/llama-3.1-8b-instruct-fp8-fast", {"requests": 1, "neurons": 250.5}
    )
    assert led.remaining("cloudflare", "", "neurons").remaining == pytest.approx(9749.5)  # type: ignore[union-attr]
    assert led.remaining("gemini", "gemini-3.5-flash", "requests") is None  # unknown limit


def test_the_window_rolls_over_at_its_reset() -> None:
    clock = FakeClock(NOW)
    led = ledger(clock)
    led.record("cloudflare", "m", {"requests": 3, "neurons": 9000})
    led.block("cloudflare", "", until=MIDNIGHT_UTC, kind="exhausted", why="used up")
    assert led.blocks("cloudflare")
    clock.set(MIDNIGHT_UTC + 1)
    assert led.used("cloudflare", "", "neurons") == 0
    assert led.remaining("cloudflare", "", "neurons").remaining == 10_000  # type: ignore[union-attr]
    assert led.blocks("cloudflare") == []


def test_live_readings_win_while_fresh_and_absorb_our_later_use() -> None:
    clock = FakeClock(NOW)
    led = ledger(clock)
    live = LiveReading(GROQ_20B, "requests", 1000, 870, NOW + 3600, NOW)
    led.record("groq", GROQ_20B, {"requests": 1}, [live])
    rem = led.remaining("groq", GROQ_20B, "requests")
    assert rem is not None
    assert (rem.remaining, rem.source, rem.resets_at) == (870, "live", NOW + 3600)
    led.record("groq", GROQ_20B, {"requests": 1})  # no headers this time
    assert led.remaining("groq", GROQ_20B, "requests").remaining == 869  # type: ignore[union-attr]
    clock.advance(led.live_ttl_s + 1)  # stale: back to our own count, labelled est
    rem = led.remaining("groq", GROQ_20B, "requests")
    assert rem is not None
    assert (rem.source, rem.remaining) == ("estimate", 998)


def test_blocks_scope_and_kinds() -> None:
    clock = FakeClock(NOW)
    led = ledger(clock)
    led.block("groq", GROQ_20B, until=NOW + 30, kind="cooldown", why="429")
    assert [b.scope for b in led.blocks("groq", GROQ_20B)] == [GROQ_20B]
    assert led.blocks("groq", "openai/gpt-oss-120b") == []
    # a successful answer clears a cooldown on that model
    led.record("groq", GROQ_20B, {"requests": 1})
    assert led.blocks("groq", GROQ_20B) == []
    led.block("groq", "", until=NOW + 600, kind="exhausted", why="day")
    led.block("groq", "", until=NOW + 60, kind="exhausted", why="shorter")  # longer one wins
    [b] = led.blocks("groq", GROQ_20B)
    assert (b.until, b.why) == (NOW + 600, "day")
    clock.advance(601)
    assert led.blocks("groq") == []


def test_persists_atomically_and_survives_a_corrupt_file(tmp_path: Path) -> None:
    path = tmp_path / "inference" / "ledger.json"
    clock = FakeClock(NOW)
    led = ledger(clock, path)
    led.record("groq", GROQ_20B, {"requests": 2, "tokens": 50})
    assert oct(path.stat().st_mode & 0o777) == "0o600"
    doc = json.loads(path.read_text())
    assert doc["providers"]["groq"]["used"][GROQ_20B]["requests"] == 2
    assert "prompt" not in path.read_text()
    again = ledger(clock, path)
    assert again.used("groq", GROQ_20B, "requests") == 2
    path.write_text("{not json")
    fresh = ledger(clock, path)
    assert fresh.used("groq", GROQ_20B, "requests") == 0
    assert (path.parent / "ledger.json.bad").exists()
