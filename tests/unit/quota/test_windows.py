"""Quota reset windows: anchors, fixed calendar windows, rolling fallbacks, boundaries."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from gpu_router.quota.windows import DAY_S, AnchorError, current_window, parse_anchor


def ts(iso: str) -> float:
    return datetime.fromisoformat(iso).replace(tzinfo=UTC).timestamp()


SAT = ts("2026-09-26T00:00:00")  # a Saturday, Kaggle's weekly reset instant
WEEK = 7 * DAY_S


@pytest.mark.parametrize(
    ("now", "start", "resets_at"),
    [
        (ts("2026-09-24T12:00:00"), SAT - WEEK, SAT),  # Thursday: last Saturday .. next
        (SAT - 1, SAT - WEEK, SAT),  # one second before the reset: still the old week
        (SAT, SAT, SAT + WEEK),  # at the reset instant the new week has begun
        (SAT + 1, SAT, SAT + WEEK),
        (SAT + WEEK - 1, SAT, SAT + WEEK),
    ],
)
def test_kaggle_weekly_saturday_boundary(now: float, start: float, resets_at: float) -> None:
    w = current_window("weekly", "sat 00:00 UTC", now)
    assert (w.kind, w.start, w.resets_at) == ("fixed", start, resets_at)
    assert w.label == "week from Sat 00:00 UTC"
    assert w.length_s == WEEK


def test_weekly_anchor_with_time() -> None:
    now = ts("2026-09-26T03:00:00")  # Saturday 03:00, reset at 06:30 today
    w = current_window("weekly", "Saturday 06:30", now)
    assert w.resets_at == ts("2026-09-26T06:30:00")
    assert w.start == ts("2026-09-19T06:30:00")


def test_monthly_clamps_to_last_day() -> None:
    now = ts("2027-02-15T00:00:00")
    w = current_window("monthly", "day 31 00:00 UTC", now)
    assert w.start == ts("2027-01-31T00:00:00")
    assert w.resets_at == ts("2027-02-28T00:00:00")
    w2 = current_window("monthly", "31st", ts("2027-02-28T00:00:00"))
    assert w2.start == ts("2027-02-28T00:00:00")
    assert w2.resets_at == ts("2027-03-31T00:00:00")


def test_monthly_first_and_year_wrap() -> None:
    w = current_window("monthly", "1", ts("2026-12-31T23:00:00"))
    assert w.start == ts("2026-12-01T00:00:00")
    assert w.resets_at == ts("2027-01-01T00:00:00")
    assert w.label == "month from day 1 00:00 UTC"


def test_daily() -> None:
    w = current_window("daily", "07:00 UTC", ts("2026-09-24T06:59:00"))
    assert w.start == ts("2026-09-23T07:00:00")
    assert w.resets_at == ts("2026-09-24T07:00:00")


def test_rolling_and_unlimited() -> None:
    now = ts("2026-09-24T12:00:00")
    unknown = current_window("unknown", None, now, unknown_window_s=24 * 3600)
    assert (unknown.kind, unknown.start, unknown.resets_at) == ("rolling", now - DAY_S, None)
    assert "reset time unknown" in unknown.label
    monthly = current_window("monthly", None, now)  # Lightning/Modal: no known reset day
    assert (monthly.kind, monthly.start, monthly.resets_at) == ("rolling", now - 30 * DAY_S, None)
    local = current_window("none", None, now)
    assert (local.kind, local.start, local.resets_at) == ("unlimited", None, None)


def test_bad_anchor_falls_back_to_rolling() -> None:
    now = ts("2026-09-24T12:00:00")
    w = current_window("weekly", "saturday midnight PST", now)
    assert w.kind == "rolling"
    assert w.start == now - WEEK
    assert "could not read reset_anchor" in w.label


@pytest.mark.parametrize(
    ("reset", "text"),
    [
        ("weekly", "00:00 UTC"),  # no weekday
        ("weekly", "sat 25:00"),
        ("monthly", "sat"),
        ("monthly", "day 32"),
        ("daily", "sat 00:00"),
        ("weekly", ""),
        ("none", "sat"),
    ],
)
def test_parse_errors(reset: str, text: str) -> None:
    with pytest.raises(AnchorError):
        parse_anchor(reset, text)


def test_parse_forms() -> None:
    assert parse_anchor("weekly", "SAT 00:00 UTC").weekday == 5
    assert parse_anchor("weekly", "mon").hour == 0
    a = parse_anchor("monthly", "15th 12:30z")
    assert (a.day, a.hour, a.minute) == (15, 12, 30)
    assert parse_anchor("daily", "7").hour == 7
