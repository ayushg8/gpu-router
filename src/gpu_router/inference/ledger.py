"""The inference lane's daily-quota ledger (phase 7b).

Owned by the daemon's `InferenceService` (invariant 2): one JSON file,
`<home>/inference/ledger.json` (0600), rewritten atomically after every change and read
back at start, so counts survive restarts. It is not in gpu.db: the counters are per
window, small, separate from job state, and are written from the service's worker threads
(the Store is event-loop-thread only, invariant 9); a JSON file behind one lock needs no
schema migration either.

Per provider it keeps, for the current reset window only:
  used     scope -> unit -> amount we spent (scope "" = all models, else a model id)
  live     scope -> unit -> the provider's own reading (limit, remaining, resets_at,
           observed_at), from rate-limit headers; decremented by what we spend after it
  blocked  scope -> {until, kind: exhausted | cooldown | auth, why}

Remaining for (scope, unit) = a live reading while it is fresh (observed within
`live_ttl_s` and before its own reset) and labelled `live`; else the catalog limit minus
what we counted this window, labelled `estimate`; None when no limit is known.

Windows (`reset`): daily_utc (00:00 UTC), daily_pacific (00:00 America/Los_Angeles, the
Gemini API), monthly (1st, 00:00 UTC; HF credits), rolling_24h (24 h from the first use
in the window; ZeroGPU). Time comes from the injected Clock (invariant 13).
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import threading
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

if TYPE_CHECKING:
    from gpu_router.clock import Clock
    from gpu_router.inference.catalog import InferenceCatalog, InferenceEntry

__all__ = ["Block", "InferenceLedger", "LiveReading", "Remaining", "window_bounds"]

_logger = logging.getLogger("gpu_router.inference.ledger")

LEDGER_VERSION = 1
PACIFIC = ZoneInfo("America/Los_Angeles")
DEFAULT_LIVE_TTL_S = 900.0


def window_bounds(reset: str, now: float, first_use: float | None = None) -> tuple[float, float]:
    """(start, end) epoch seconds of the reset window containing `now`."""
    if reset == "daily_pacific":
        local = datetime.fromtimestamp(now, PACIFIC)
        start = local.replace(hour=0, minute=0, second=0, microsecond=0)
        # add a calendar day in local time (23h / 25h days around DST are right this way)
        nxt = (start + timedelta(days=1, hours=2)).replace(hour=0)
        return start.timestamp(), nxt.timestamp()
    if reset == "monthly":
        utc = datetime.fromtimestamp(now, UTC)
        start = utc.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        nxt = (start + timedelta(days=32)).replace(day=1)
        return start.timestamp(), nxt.timestamp()
    if reset == "rolling_24h":
        if first_use is not None and now < first_use + 86_400:
            return first_use, first_use + 86_400
        return now, now + 86_400
    utc = datetime.fromtimestamp(now, UTC)  # daily_utc and anything unknown
    start = utc.replace(hour=0, minute=0, second=0, microsecond=0)
    return start.timestamp(), (start + timedelta(days=1)).timestamp()


@dataclass(frozen=True)
class LiveReading:
    """What a provider said about one counter (e.g. Groq's requests-per-day headers)."""

    scope: str  # model id, or "" for provider-wide
    unit: str
    limit: float | None
    remaining: float
    resets_at: float | None
    observed_at: float


@dataclass(frozen=True)
class Remaining:
    scope: str
    unit: str
    remaining: float
    limit: float | None
    source: str  # live | estimate
    resets_at: float | None


@dataclass(frozen=True)
class Block:
    scope: str
    until: float
    kind: str  # exhausted | cooldown | auth
    why: str


def _num(v: Any) -> float | None:
    return float(v) if isinstance(v, int | float) and not isinstance(v, bool) else None


class InferenceLedger:
    """Thread-safe; every public method takes the lock. `path=None` keeps it in memory."""

    def __init__(
        self,
        path: Path | None,
        catalog: InferenceCatalog,
        clock: Clock,
        *,
        live_ttl_s: float = DEFAULT_LIVE_TTL_S,
    ) -> None:
        self.path = path
        self.catalog = catalog
        self.clock = clock
        self.live_ttl_s = live_ttl_s
        self._lock = threading.RLock()
        self._data: dict[str, Any] = {"version": LEDGER_VERSION, "providers": {}}
        self._load()

    # ------------------------------------------------------------------ persistence

    def _load(self) -> None:
        if self.path is None or not self.path.is_file():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict) or not isinstance(raw.get("providers"), dict):
                raise ValueError("not a ledger document")
            self._data = {"version": LEDGER_VERSION, "providers": raw["providers"]}
        except (OSError, ValueError) as exc:
            # keep the bad file for a human, start counting again (counts are estimates)
            _logger.warning("inference ledger unreadable (%s); starting empty", exc)
            with contextlib.suppress(OSError):
                self.path.replace(self.path.with_suffix(".json.bad"))

    def _save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        tmp = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(self._data, fh, separators=(",", ":"), sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self.path)

    # ------------------------------------------------------------------ windows

    def _entry(self, provider: str) -> InferenceEntry | None:
        return self.catalog.get(provider)

    def _book(self, provider: str, now: float) -> dict[str, Any]:
        """The provider's record for the window containing `now` (rolled over if needed)."""
        providers: dict[str, Any] = self._data.setdefault("providers", {})
        book = providers.get(provider)
        entry = self._entry(provider)
        reset = entry.reset if entry is not None else "daily_utc"
        if not isinstance(book, dict):
            book = {}
        end = _num(book.get("end"))
        if end is None or now >= end:
            first = _num(book.get("first_use")) if reset == "rolling_24h" else None
            start, new_end = window_bounds(reset, now, first)
            book = {
                "start": start,
                "end": new_end,
                "used": {},
                "live": self._still_valid(book.get("live"), now),
                "blocked": self._still_blocked(book.get("blocked"), now),
            }
            providers[provider] = book
        for key in ("used", "live", "blocked"):
            if not isinstance(book.get(key), dict):
                book[key] = {}
        return book

    @staticmethod
    def _still_valid(live: Any, now: float) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if not isinstance(live, dict):
            return out
        for scope, units in live.items():
            if not isinstance(units, dict):
                continue
            keep = {
                u: r
                for u, r in units.items()
                if isinstance(r, dict) and (_num(r.get("resets_at")) or 0) > now
            }
            if keep:
                out[scope] = keep
        return out

    @staticmethod
    def _still_blocked(blocked: Any, now: float) -> dict[str, Any]:
        if not isinstance(blocked, dict):
            return {}
        return {
            s: b
            for s, b in blocked.items()
            if isinstance(b, dict) and (_num(b.get("until")) or 0) > now
        }

    def window(self, provider: str) -> tuple[float, float]:
        with self._lock:
            book = self._book(provider, self.clock.now())
            return float(book["start"]), float(book["end"])

    # ------------------------------------------------------------------ writes

    def record(
        self,
        provider: str,
        model_id: str,
        amounts: dict[str, float],
        live: list[LiveReading] | tuple[LiveReading, ...] = (),
    ) -> None:
        """Count one answered request (amounts per unit, e.g. {"requests": 1, "tokens":
        812}) against the provider and the model, apply live readings, clear cooldowns."""
        with self._lock:
            now = self.clock.now()
            book = self._book(provider, now)
            if book.get("first_use") is None or _num(book.get("first_use")) is None:
                book["first_use"] = now
            used: dict[str, Any] = book["used"]
            for scope in ("", model_id):
                bucket = used.setdefault(scope, {})
                for unit, amount in amounts.items():
                    if amount:
                        bucket[unit] = float(bucket.get(unit, 0.0)) + float(amount)
                # what we spent after a live reading was taken comes off it
                for unit, amount in amounts.items():
                    reading = book["live"].get(scope, {}).get(unit)
                    if isinstance(reading, dict) and amount:
                        left = (_num(reading.get("remaining")) or 0.0) - float(amount)
                        reading["remaining"] = max(0.0, left)
            for reading in live:
                self._apply_live(book, reading)
            blocked: dict[str, Any] = book["blocked"]
            for scope in ("", model_id):
                b = blocked.get(scope)
                if isinstance(b, dict) and b.get("kind") in ("cooldown", "auth"):
                    blocked.pop(scope, None)
            self._save()

    def _apply_live(self, book: dict[str, Any], r: LiveReading) -> None:
        book["live"].setdefault(r.scope, {})[r.unit] = {
            "limit": r.limit,
            "remaining": r.remaining,
            "resets_at": r.resets_at,
            "observed_at": r.observed_at,
        }

    def observe(self, provider: str, readings: list[LiveReading]) -> None:
        """Live readings without a counted request (e.g. headers on a 429)."""
        if not readings:
            return
        with self._lock:
            book = self._book(provider, self.clock.now())
            for r in readings:
                self._apply_live(book, r)
            self._save()

    def block(self, provider: str, scope: str, *, until: float, kind: str, why: str) -> None:
        """Keep `scope` ("" = the whole provider) out of routing until `until`."""
        with self._lock:
            book = self._book(provider, self.clock.now())
            current = book["blocked"].get(scope)
            if (
                isinstance(current, dict)
                and (_num(current.get("until")) or 0) > until
                and current.get("kind") == kind
            ):
                return  # an existing longer block of the same kind wins
            book["blocked"][scope] = {"until": until, "kind": kind, "why": why}
            self._save()

    # ------------------------------------------------------------------ reads

    def blocks(self, provider: str, model_id: str | None = None) -> list[Block]:
        """Active blocks for the provider ("") and, when given, the model."""
        with self._lock:
            now = self.clock.now()
            book = self._book(provider, now)
            out: list[Block] = []
            for scope, b in book["blocked"].items():
                if scope not in ("", model_id) and model_id is not None:
                    continue
                until = _num(b.get("until")) if isinstance(b, dict) else None
                if until is None or until <= now:
                    continue
                out.append(Block(scope, until, str(b.get("kind")), str(b.get("why", ""))))
            return sorted(out, key=lambda b: -b.until)

    def used(self, provider: str, scope: str, unit: str) -> float:
        with self._lock:
            book = self._book(provider, self.clock.now())
            return float(book["used"].get(scope, {}).get(unit, 0.0))

    def remaining(self, provider: str, scope: str, unit: str) -> Remaining | None:
        """Left in (scope, unit): live when a fresh reading exists, else the catalog limit
        minus our count; None when no limit is known."""
        with self._lock:
            now = self.clock.now()
            book = self._book(provider, now)
            reading = book["live"].get(scope, {}).get(unit)
            if isinstance(reading, dict):
                observed = _num(reading.get("observed_at")) or 0.0
                resets = _num(reading.get("resets_at"))
                if now - observed <= self.live_ttl_s and (resets is None or resets > now):
                    return Remaining(
                        scope=scope,
                        unit=unit,
                        remaining=max(0.0, _num(reading.get("remaining")) or 0.0),
                        limit=_num(reading.get("limit")),
                        source="live",
                        resets_at=resets,
                    )
            entry = self._entry(provider)
            if entry is None:
                return None
            limit: float | None
            if scope == "":
                limit = entry.limits.get(unit)  # type: ignore[call-overload]
            else:
                model = entry.resolve(scope)
                limit = model.limits.get(unit) if model is not None else None  # type: ignore[call-overload]
            if limit is None:
                return None
            spent = float(book["used"].get(scope, {}).get(unit, 0.0))
            return Remaining(
                scope=scope,
                unit=unit,
                remaining=max(0.0, float(limit) - spent),
                limit=float(limit),
                source="estimate",
                resets_at=float(book["end"]),
            )

    def snapshot(self, provider: str) -> dict[str, Any]:
        """A deep copy of the provider's current-window record (for views and tests)."""
        with self._lock:
            book = self._book(provider, self.clock.now())
            copy: dict[str, Any] = json.loads(json.dumps(book))
            return copy
