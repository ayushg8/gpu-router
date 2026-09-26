"""Quota reset windows (phase 5): which slice of job history counts against a provider's
quota right now, and when it resets. Pure date math on epoch seconds (invariant 13: the
caller passes `now`).

`reset` / `reset_anchor` come from providers.yaml (`QuotaSpec`):

  weekly  + "sat 00:00 UTC"     fixed calendar week: resets every Saturday 00:00 UTC (Kaggle)
  monthly + "day 1 00:00 UTC"   fixed calendar month; days 29-31 clamp to the month's last day
  daily   + "07:00 UTC"         fixed day
  weekly / monthly / daily with no anchor: a rolling 7 / 30 / 1 day window, reset unknown
  unknown                       rolling window of `unknown_window_s` (Colab: limits are dynamic)
  none                          unlimited (local)

Anchor grammar (case-insensitive, whitespace separated, trailing "UTC" / "Z" optional):
  weekday  mon|monday ... sun|sunday   (weekly)
  day      "day 15" | "15" | "15th"    (monthly)
  time     "HH:MM" | "HH"              (default 00:00)
Only UTC is understood; an anchor naming another zone, or one that does not parse, falls back
to the rolling window and says so in the label (a typo must never break routing).

At the exact anchor instant the new window has begun: usage before it no longer counts.
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal

__all__ = [
    "PERIOD_S",
    "Anchor",
    "AnchorError",
    "Window",
    "current_window",
    "parse_anchor",
]

DAY_S = 86_400.0
#: Rolling window length for a known period with no anchor.
PERIOD_S: dict[str, float] = {"daily": DAY_S, "weekly": 7 * DAY_S, "monthly": 30 * DAY_S}

_WEEKDAYS = {
    "mon": 0,
    "monday": 0,
    "tue": 1,
    "tues": 1,
    "tuesday": 1,
    "wed": 2,
    "wednesday": 2,
    "thu": 3,
    "thur": 3,
    "thurs": 3,
    "thursday": 3,
    "fri": 4,
    "friday": 4,
    "sat": 5,
    "saturday": 5,
    "sun": 6,
    "sunday": 6,
}
_WEEKDAY_NAMES = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
_TIME_RE = re.compile(r"^(\d{1,2})(?::(\d{2}))?$")
_DAY_RE = re.compile(r"^(\d{1,2})(?:st|nd|rd|th)?$")


class AnchorError(ValueError):
    """A reset_anchor that cannot be understood."""


@dataclass(frozen=True, slots=True)
class Anchor:
    hour: int = 0
    minute: int = 0
    weekday: int | None = None  # 0 = Monday (weekly)
    day: int | None = None  # 1..31 (monthly)

    def label(self) -> str:
        hm = f"{self.hour:02d}:{self.minute:02d} UTC"
        if self.weekday is not None:
            return f"{_WEEKDAY_NAMES[self.weekday]} {hm}"
        if self.day is not None:
            return f"day {self.day} {hm}"
        return hm


@dataclass(frozen=True, slots=True)
class Window:
    """The quota window containing `now`."""

    kind: Literal["fixed", "rolling", "unlimited"]
    start: float | None  # usage since this instant counts (None = unlimited)
    resets_at: float | None  # next reset (fixed windows only)
    length_s: float | None
    label: str  # "week from Sat 00:00 UTC", "rolling 24h (reset time unknown)", "unlimited"

    def to_json(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "start": self.start,
            "resets_at": self.resets_at,
            "length_h": None if self.length_s is None else round(self.length_s / 3600, 3),
            "label": self.label,
        }


def parse_anchor(reset: str, text: str) -> Anchor:
    """Parse `text` for a `reset` period. Raises AnchorError with a readable message."""
    tokens = [t for t in text.strip().lower().replace(",", " ").split() if t]
    if tokens and tokens[-1] in ("utc", "z", "gmt"):
        tokens = tokens[:-1]
    # "12:30z" -> "12:30" (ISO-style Zulu suffix on the time)
    tokens = [t[:-1] if t.endswith("z") and _TIME_RE.match(t[:-1]) else t for t in tokens]
    if not tokens:
        raise AnchorError(f"empty reset_anchor for a {reset} reset")
    weekday: int | None = None
    day: int | None = None
    hour = minute = 0
    time_seen = False
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in _WEEKDAYS and weekday is None:
            weekday = _WEEKDAYS[tok]
        elif tok == "day" and i + 1 < len(tokens) and day is None:
            i += 1
            day = _parse_day(tokens[i], text)
        elif ":" not in tok and reset == "monthly" and day is None and _DAY_RE.match(tok):
            day = _parse_day(tok, text)  # "1", "15th": the first bare number is the day
        elif _TIME_RE.match(tok) and not time_seen:
            hour, minute = _parse_time(tok, text)
            time_seen = True
        else:
            raise AnchorError(f"cannot read {tok!r} in reset_anchor {text!r} (only UTC is known)")
        i += 1
    if reset == "weekly":
        if weekday is None or day is not None:
            raise AnchorError(
                f"a weekly reset_anchor names a weekday, e.g. 'sat 00:00 UTC': {text!r}"
            )
    elif reset == "monthly":
        if day is None or weekday is not None:
            raise AnchorError(
                f"a monthly reset_anchor names a day, e.g. 'day 1 00:00 UTC': {text!r}"
            )
    elif reset == "daily":
        if day is not None or weekday is not None:
            raise AnchorError(f"a daily reset_anchor is only a time, e.g. '00:00 UTC': {text!r}")
    else:
        raise AnchorError(f"reset {reset!r} takes no anchor")
    return Anchor(hour=hour, minute=minute, weekday=weekday, day=day)


def _parse_time(tok: str, text: str) -> tuple[int, int]:
    m = _TIME_RE.match(tok)
    if m is None:
        raise AnchorError(f"cannot read time {tok!r} in reset_anchor {text!r}")
    hour, minute = int(m.group(1)), int(m.group(2) or 0)
    if hour > 23 or minute > 59:
        raise AnchorError(f"time {tok!r} out of range in reset_anchor {text!r}")
    return hour, minute


def _parse_day(tok: str, text: str) -> int:
    m = _DAY_RE.match(tok)
    if m is None or not 1 <= int(m.group(1)) <= 31:
        raise AnchorError(f"cannot read day {tok!r} in reset_anchor {text!r}")
    return int(m.group(1))


def _month_anchor(year: int, month: int, a: Anchor) -> datetime:
    last = calendar.monthrange(year, month)[1]
    assert a.day is not None
    return datetime(year, month, min(a.day, last), a.hour, a.minute, tzinfo=UTC)


def _shift_month(year: int, month: int, delta: int) -> tuple[int, int]:
    idx = year * 12 + (month - 1) + delta
    return idx // 12, idx % 12 + 1


def _fixed(reset: str, a: Anchor, now: float) -> tuple[float, float]:
    """(start, resets_at) of the fixed window containing `now` (start <= now < resets_at)."""
    dt = datetime.fromtimestamp(now, tz=UTC)
    if reset == "daily":
        start = dt.replace(hour=a.hour, minute=a.minute, second=0, microsecond=0)
        if start > dt:
            start -= timedelta(days=1)
        return start.timestamp(), (start + timedelta(days=1)).timestamp()
    if reset == "weekly":
        assert a.weekday is not None
        start = dt.replace(hour=a.hour, minute=a.minute, second=0, microsecond=0)
        start -= timedelta(days=(dt.weekday() - a.weekday) % 7)
        if start > dt:
            start -= timedelta(days=7)
        return start.timestamp(), (start + timedelta(days=7)).timestamp()
    # monthly
    start = _month_anchor(dt.year, dt.month, a)
    if start > dt:
        y, m = _shift_month(dt.year, dt.month, -1)
        start = _month_anchor(y, m, a)
    y, m = _shift_month(start.year, start.month, 1)
    return start.timestamp(), _month_anchor(y, m, a).timestamp()


def _hours(seconds: float) -> str:
    h = seconds / 3600
    if h < 48:
        return f"{h:g}h"
    return f"{h / 24:g}d"


def current_window(
    reset: str, anchor: str | None, now: float, *, unknown_window_s: float = DAY_S
) -> Window:
    """The window containing `now` for a provider's `reset` / `reset_anchor`."""
    if reset == "none":
        return Window(
            kind="unlimited", start=None, resets_at=None, length_s=None, label="unlimited"
        )
    if reset not in PERIOD_S:  # "unknown" (and anything newer we do not know)
        return Window(
            kind="rolling",
            start=now - unknown_window_s,
            resets_at=None,
            length_s=unknown_window_s,
            label=f"rolling {_hours(unknown_window_s)} (reset time unknown)",
        )
    period = PERIOD_S[reset]
    note = "reset time unknown"
    if anchor:
        try:
            a = parse_anchor(reset, anchor)
        except AnchorError as exc:
            note = f"could not read reset_anchor: {exc}"
        else:
            start, resets_at = _fixed(reset, a, now)
            unit = {"daily": "day", "weekly": "week", "monthly": "month"}[reset]
            return Window(
                kind="fixed",
                start=start,
                resets_at=resets_at,
                length_s=resets_at - start,
                label=f"{unit} from {a.label()}",
            )
    return Window(
        kind="rolling",
        start=now - period,
        resets_at=None,
        length_s=period,
        label=f"rolling {_hours(period)} ({note})",
    )
