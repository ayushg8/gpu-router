"""Quota ledger view (phase 5, spec "Quota ledger"): per provider used / limit / resets_at /
source, live where the provider exposes it, otherwise estimated from job history, always
labelled which.

Inputs are all local (the store and the catalog), so the view is cheap and routing never
waits on a provider (the daemon's `QuotaService` refreshes live readings in the background):

  * `quota_snapshots`: the latest live reading of a provider whose adapter reports live
    quota (`capabilities.live_quota`, e.g. Kaggle). Adapter estimates (Colab) are ignored:
    the ledger's own history estimate replaces them.
  * `attempts` via `Store.usage_seconds`: GPU time of our own runs (started_at..ended_at,
    running attempts up to now; D33 stamps a start on runs no poll saw running).
  * `provider_state.exhausted_until`: the provider refused for quota (QuotaExhausted).
  * the catalog's `quota` (unit, limit, reset, reset_anchor) -> `windows.current_window`.

Per provider, first match wins:

  unlimited  reset: none (local): used = GPU hours in the last 7 days, limit None, source
             live (the "unlimited" fact is exact).
  live       a live reading younger than ttl_s and not past its own resets_at: returned as
             observed, ledger keys added to `detail`.
  live+hist  an older live reading in the same window: used = reading + our GPU time since
             it was taken; source estimate.
  reset      a live reading from before its own resets_at: used = our GPU time since that
             reset; next reset from the catalog window; source estimate.
  history    no usable reading: used = our GPU time in the current window (fixed calendar
             window when the catalog has a reset_anchor, else rolling); source estimate.

Units: history is GPU hours. For a `credits` / `usd` provider the hours are converted with
the catalog option `quota_per_gpu_hour` when present; otherwise the limit is reported as
unknown (None) instead of pretending nothing was used. `quota_per_gpu_hour_by_gpu` ({GPU
name: rate}, e.g. Lightning's L4 at 1.68 vs T4 at 0.68) prices one offer differently; the
router converts with the rate of the offer it would place on, history keeps the default.
An exhausted provider reports used = limit (nothing left) until `exhausted_until`.

`detail` keys (additive, JSON-safe): basis, note (one human line), window {kind, start,
resets_at, length_h, label}, remaining, history_h, live_used, live_observed_at,
exhausted_until, catalog_limit.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from gpu_router.models import QuotaSnapshot, QuotaUnit
from gpu_router.quota.settings import QuotaSettings
from gpu_router.quota.windows import DAY_S, Window, current_window

if TYPE_CHECKING:
    from gpu_router.adapters.base import Capabilities
    from gpu_router.adapters.registry import AdapterRegistry
    from gpu_router.models import ProviderState
    from gpu_router.providers.catalog import ProviderEntry
    from gpu_router.store import Store

__all__ = [
    "LedgerInput",
    "compute_view",
    "compute_views",
    "fmt_hours",
    "gpu_name",
    "ledger_views",
    "remaining",
    "to_gpu_hours",
]

UNLIMITED_HISTORY_S = 7 * DAY_S
RATE_OPTION = "quota_per_gpu_hour"
RATE_BY_GPU_OPTION = "quota_per_gpu_hour_by_gpu"


@dataclass(frozen=True, slots=True)
class LedgerInput:
    name: str
    entry: ProviderEntry
    live_capable: bool
    live: QuotaSnapshot | None  # latest live reading, if any
    state: ProviderState | None


def fmt_hours(h: float) -> str:
    """0.5h, 12h, 1.25h -> '0.5h', '12h', '1.2h'."""
    r = round(h, 1)
    return f"{int(r)}h" if r == int(r) else f"{r:.1f}h"


def _ago(seconds: float) -> str:
    s = max(0, round(seconds))
    if s < 90:
        return f"{s}s ago" if s < 60 else "1m ago"
    m = s // 60
    if m < 90:
        return f"{m}m ago"
    h = m / 60
    return f"{h:.0f}h ago" if h < 48 else f"{h / 24:.0f}d ago"


def _clock(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=UTC).strftime("%a %H:%M UTC")


def _positive(raw: object) -> float | None:
    if isinstance(raw, int | float) and not isinstance(raw, bool) and raw > 0:
        return float(raw)
    return None


def gpu_name(gpu: str | None) -> str | None:
    """'2xT4' -> 'T4'; 'L4' -> 'L4' (offer labels carry the count, rates do not)."""
    if not gpu:
        return None
    count, sep, name = gpu.partition("x")
    return name if sep and count.isdigit() and name else gpu


def _rate(entry: ProviderEntry, gpu: str | None = None) -> float | None:
    """Quota units per GPU hour: the offer's own rate when the catalog prices it, else
    the provider's `quota_per_gpu_hour`."""
    name = gpu_name(gpu)
    by_gpu = entry.options.get(RATE_BY_GPU_OPTION)
    if name is not None and isinstance(by_gpu, dict):
        own = _positive(by_gpu.get(name))
        if own is not None:
            return own
    return _positive(entry.options.get(RATE_OPTION))


def to_gpu_hours(entry: ProviderEntry, amount: float, gpu: str | None = None) -> float | None:
    """Convert an amount in the provider's quota unit to GPU hours (None = unknown).

    `gpu` (an offer name or label) picks that offer's rate from `quota_per_gpu_hour_by_gpu`:
    5 credits are ~7h of Lightning T4 but only ~3h of L4."""
    if entry.quota.unit is QuotaUnit.GPU_HOURS:
        return amount
    rate = _rate(entry, gpu)
    return None if rate is None else amount / rate


def _from_hours(entry: ProviderEntry, hours: float) -> float | None:
    if entry.quota.unit is QuotaUnit.GPU_HOURS:
        return hours
    rate = _rate(entry)
    return None if rate is None else hours * rate


def remaining(q: QuotaSnapshot | None) -> float | None:
    """Quota left in the snapshot's unit; None = unknown or unlimited."""
    if q is None or q.limit is None:
        return None
    left = q.detail.get("remaining")
    if isinstance(left, int | float) and not isinstance(left, bool):
        return max(0.0, float(left))
    return max(0.0, q.limit - q.used)


def compute_view(
    inp: LedgerInput,
    *,
    usage_h: Callable[[float, float], float],
    now: float,
    settings: QuotaSettings,
) -> QuotaSnapshot:
    """The ledger's view of one provider. `usage_h(since, until)` = our GPU hours on it."""
    entry = inp.entry
    spec = entry.quota
    window = current_window(
        spec.reset, spec.reset_anchor, now, unknown_window_s=settings.unknown_window_s
    )
    detail: dict[str, Any] = {"window": window.to_json()}
    if spec.limit is not None:
        detail["catalog_limit"] = spec.limit

    if window.kind == "unlimited":
        hist = usage_h(now - UNLIMITED_HISTORY_S, now)
        detail.update(
            basis="unlimited",
            unlimited=True,
            history_h=round(hist, 4),
            remaining=None,
            note=f"unlimited; {fmt_hours(hist)} used in the last 7 days",
        )
        return QuotaSnapshot(
            provider=inp.name,
            used=hist,
            limit=None,
            unit=QuotaUnit.GPU_HOURS,
            resets_at=None,
            source="live",
            detail=detail,
            observed_at=now,
        )

    live = inp.live if inp.live_capable else None
    view: QuotaSnapshot | None = None
    if live is not None:
        view = _from_live(inp, live, window, detail, usage_h=usage_h, now=now, settings=settings)
    if view is None:
        view = _from_history(inp, window, detail, usage_h=usage_h, now=now)
    return _apply_exhausted(inp, view, now)


def _from_live(
    inp: LedgerInput,
    live: QuotaSnapshot,
    window: Window,
    detail: dict[str, Any],
    *,
    usage_h: Callable[[float, float], float],
    now: float,
    settings: QuotaSettings,
) -> QuotaSnapshot | None:
    age = now - live.observed_at
    detail["live_used"] = live.used
    detail["live_observed_at"] = live.observed_at
    reset_passed = live.resets_at is not None and live.resets_at <= now
    if not reset_passed and 0 <= age < settings.ttl_s:
        merged = {**live.detail, **detail}
        merged.update(basis="live", note=f"live from {inp.name}, checked {_ago(age)}")
        left = None if live.limit is None else max(0.0, live.limit - live.used)
        merged["remaining"] = left
        return live.model_copy(update={"detail": merged})
    if reset_passed:
        assert live.resets_at is not None
        since = live.resets_at
        hist = usage_h(since, now)
        used = _from_hours(inp.entry, hist)
        if used is None:
            return None
        resets_at = window.resets_at
        if resets_at is None and window.length_s is not None:
            # the provider's own cadence: next reset one period after the observed one
            resets_at = since + window.length_s
            while resets_at <= now:
                resets_at += window.length_s
        detail.update(
            basis="reset",
            history_h=round(hist, 4),
            note=(
                f"quota reset at {_clock(since)} since the last live reading; "
                f"{fmt_hours(hist)} of runs since"
            ),
        )
        limit = live.limit
        detail["remaining"] = None if limit is None else max(0.0, limit - used)
        return QuotaSnapshot(
            provider=inp.name,
            used=used,
            limit=limit,
            unit=live.unit,
            resets_at=resets_at,
            source="estimate",
            detail=detail,
            observed_at=now,
        )
    # stale reading in the same window: reading + our GPU time since it was taken
    hist = usage_h(live.observed_at, now) if age > 0 else 0.0
    extra = _from_hours(inp.entry, hist)
    if extra is None:
        return None
    used = live.used + extra
    detail.update(
        basis="live+history",
        history_h=round(hist, 4),
        note=f"last live reading {_ago(age)} + {fmt_hours(hist)} of runs since",
    )
    detail["remaining"] = None if live.limit is None else max(0.0, live.limit - used)
    return QuotaSnapshot(
        provider=inp.name,
        used=used,
        limit=live.limit,
        unit=live.unit,
        resets_at=live.resets_at,
        source="estimate",
        detail=detail,
        observed_at=now,
    )


def _from_history(
    inp: LedgerInput,
    window: Window,
    detail: dict[str, Any],
    *,
    usage_h: Callable[[float, float], float],
    now: float,
) -> QuotaSnapshot:
    entry = inp.entry
    start = window.start if window.start is not None else now
    hist = usage_h(start, now)
    used = _from_hours(entry, hist)
    limit = entry.quota.limit
    where = f"since {_clock(start)}" if window.kind == "fixed" else f"in the {window.label}"
    note = f"estimated from your jobs {where}"
    if used is None:
        # history is GPU hours; this provider meters something else and has no rate
        used, limit = hist, None
        note += f" ({fmt_hours(hist)} GPU; {entry.quota.unit} not convertible)"
    detail.update(
        basis="history",
        history_h=round(hist, 4),
        remaining=None if limit is None else max(0.0, limit - used),
        note=note,
    )
    return QuotaSnapshot(
        provider=inp.name,
        used=used,
        limit=limit,
        unit=entry.quota.unit if limit is not None else QuotaUnit.GPU_HOURS,
        resets_at=window.resets_at,
        source="estimate",
        detail=detail,
        observed_at=now,
    )


def _apply_exhausted(inp: LedgerInput, view: QuotaSnapshot, now: float) -> QuotaSnapshot:
    until = inp.state.exhausted_until if inp.state is not None else None
    if until is None or until <= now:
        return view
    detail = dict(view.detail)
    detail["exhausted_until"] = until
    detail["remaining"] = 0.0
    detail["note"] = f"{detail.get('note', '')}; {inp.name} refused for quota until {_clock(until)}"
    detail["note"] = str(detail["note"]).lstrip("; ")
    update: dict[str, Any] = {"detail": detail}
    if view.limit is not None:
        update["used"] = max(view.used, view.limit)
    if view.resets_at is None or view.resets_at < until:
        update["resets_at"] = until
    return view.model_copy(update=update)


def compute_views(
    store: Store,
    providers: Sequence[tuple[str, ProviderEntry, Capabilities]],
    now: float,
    settings: QuotaSettings,
) -> dict[str, QuotaSnapshot]:
    """Ledger views for `providers` (name, catalog entry, capabilities), from the store."""
    latest = store.latest_quota_snapshots()
    states = store.all_provider_states()
    out: dict[str, QuotaSnapshot] = {}
    for name, entry, caps in providers:
        snap = latest.get(name)
        live = snap if snap is not None and snap.source == "live" else None

        def usage_h(since: float, until: float, _name: str = name) -> float:
            return store.usage_seconds(_name, since, until) / 3600

        out[name] = compute_view(
            LedgerInput(
                name=name,
                entry=entry,
                live_capable=caps.live_quota,
                live=live,
                state=states.get(name),
            ),
            usage_h=usage_h,
            now=now,
            settings=settings,
        )
    return out


def ledger_views(
    store: Store, registry: AdapterRegistry, now: float, settings: QuotaSettings
) -> dict[str, QuotaSnapshot]:
    """Ledger views for every registered provider, in registry order."""
    providers = [(n, registry.entry(n), registry.get(n).capabilities) for n in registry.names()]
    return compute_views(store, providers, now, settings)
