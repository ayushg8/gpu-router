"""Quota ledger views: live vs estimate, reset windows, history, exhaustion, units."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import pytest

from gpu_router.adapters.base import Capabilities
from gpu_router.clock import FakeClock
from gpu_router.models import (
    AttemptPatch,
    AttemptState,
    JobSpec,
    ProviderState,
    QuotaSnapshot,
    QuotaUnit,
)
from gpu_router.providers.catalog import ProviderEntry, load_catalog
from gpu_router.quota.ledger import (
    LedgerInput,
    compute_view,
    compute_views,
    remaining,
    to_gpu_hours,
)
from gpu_router.quota.settings import QuotaSettings
from gpu_router.statemachine import JobState, Reason
from gpu_router.store import Store


def ts(iso: str) -> float:
    return datetime.fromisoformat(iso).replace(tzinfo=UTC).timestamp()


SAT = ts("2026-09-26T00:00:00")
THU = ts("2026-09-24T12:00:00")
H = 3600.0
CAT = load_catalog()
KAGGLE = CAT.get("kaggle")
COLAB = CAT.get("colab")
LOCAL = CAT.get("local")
LIGHTNING = CAT.get("lightning")
SETTINGS = QuotaSettings()


def usage(spans: list[tuple[float, float | None]]) -> Callable[[float, float], float]:
    """usage_h over explicit (start, end|None=running) spans, like Store.usage_seconds."""

    def fn(since: float, until: float) -> float:
        total = 0.0
        for start, end in spans:
            stop = until if end is None else min(end, until)
            total += max(0.0, stop - max(start, since))
        return total / H

    return fn


def live(
    used: float, *, at: float, resets_at: float | None = SAT, limit: float = 30
) -> QuotaSnapshot:
    return QuotaSnapshot(
        provider="kaggle",
        used=used,
        limit=limit,
        unit=QuotaUnit.GPU_HOURS,
        resets_at=resets_at,
        source="live",
        detail={"remaining_h": limit - used},
        observed_at=at,
    )


def view(
    entry: ProviderEntry,
    now: float,
    spans: list[tuple[float, float | None]] | None = None,
    *,
    live_capable: bool = False,
    snap: QuotaSnapshot | None = None,
    state: ProviderState | None = None,
) -> QuotaSnapshot:
    return compute_view(
        LedgerInput(
            name=entry.name, entry=entry, live_capable=live_capable, live=snap, state=state
        ),
        usage_h=usage(spans or []),
        now=now,
        settings=SETTINGS,
    )


def test_fresh_live_reading_is_used_as_is() -> None:
    v = view(KAGGLE, THU, live_capable=True, snap=live(22, at=THU - 60))
    assert v.source == "live"
    assert (v.used, v.limit, v.resets_at) == (22, 30, SAT)
    assert v.observed_at == THU - 60
    assert v.detail["basis"] == "live"
    assert v.detail["remaining_h"] == 8  # the adapter's own detail survives
    assert remaining(v) == 8
    assert "checked 1m ago" in v.detail["note"]


def test_stale_live_reading_adds_history_since() -> None:
    at = THU - 2 * H  # older than the 15 min TTL
    v = view(KAGGLE, THU, [(THU - 1.5 * H, THU - 0.5 * H)], live_capable=True, snap=live(20, at=at))
    assert v.source == "estimate"
    assert v.used == pytest.approx(21.0)
    assert v.resets_at == SAT
    assert v.detail["basis"] == "live+history"
    assert remaining(v) == pytest.approx(9.0)


def test_live_reading_from_before_the_saturday_reset() -> None:
    now = SAT + 3 * H
    v = view(
        KAGGLE,
        now,
        [(SAT - 2 * H, SAT + 1 * H)],  # a run across the reset: only the hour after counts
        live_capable=True,
        snap=live(29, at=SAT - 1 * H, resets_at=SAT),
    )
    assert v.source == "estimate"
    assert v.detail["basis"] == "reset"
    assert v.used == pytest.approx(1.0)
    assert v.resets_at == SAT + 7 * 24 * H  # next Saturday, from the catalog anchor
    assert remaining(v) == pytest.approx(29.0)


def test_history_estimate_counts_only_the_current_kaggle_week() -> None:
    spans = [
        (SAT - 5 * H, SAT - 1 * H),  # last week: does not count
        (SAT - 1 * H, SAT + 2 * H),  # across the boundary: 2h count
        (SAT + 10 * H, None),  # running now: counts up to now
    ]
    now = SAT + 12 * H
    v = view(KAGGLE, now, spans)  # no live reading at all
    assert v.source == "estimate"
    assert v.used == pytest.approx(4.0)
    assert (v.limit, v.resets_at) == (30, SAT + 7 * 24 * H)
    assert v.detail["basis"] == "history"
    assert v.detail["window"]["kind"] == "fixed"
    assert "since Sat 00:00 UTC" in v.detail["note"]


def test_live_reading_ignored_when_adapter_is_not_live() -> None:
    # Colab's own estimate snapshot is not a live reading: the ledger estimates itself
    snap = live(99, at=THU).model_copy(update={"provider": "colab", "source": "estimate"})
    v = view(COLAB, THU, [(THU - 30 * H, THU - 22 * H), (THU - 3 * H, THU - 1 * H)], snap=snap)
    assert v.source == "estimate"
    assert v.used == pytest.approx(4.0)  # rolling 24h window: 2h + 2h of the older run
    assert v.limit is None
    assert remaining(v) is None
    assert v.resets_at is None
    assert "rolling 24h" in v.detail["note"]


def test_local_is_unlimited() -> None:
    v = view(LOCAL, THU, [(THU - 2 * H, THU - 1 * H)])
    assert v.source == "live"
    assert v.limit is None
    assert remaining(v) is None
    assert v.used == pytest.approx(1.0)
    assert v.detail["unlimited"] is True


def test_exhausted_provider_reports_nothing_left() -> None:
    state = ProviderState(provider="kaggle", exhausted_until=THU + 5 * H, updated_at=THU)
    v = view(KAGGLE, THU, [], state=state)
    assert remaining(v) == 0
    assert v.used == 30
    assert v.detail["exhausted_until"] == THU + 5 * H
    colab_state = ProviderState(provider="colab", exhausted_until=THU + 20 * H, updated_at=THU)
    c = view(COLAB, THU, [], state=colab_state)
    assert c.resets_at == THU + 20 * H  # unknown reset -> the refusal's retry time
    assert "refused for quota" in c.detail["note"]


def test_credits_without_a_rate_are_not_faked() -> None:
    no_rate = LIGHTNING.model_copy(update={"options": {}})  # the catalog has one since 7a
    v = view(no_rate, THU, [(THU - 2 * H, THU)])
    assert v.limit is None  # 15 credits known, but hours cannot be converted
    assert v.unit is QuotaUnit.GPU_HOURS
    assert v.used == pytest.approx(2.0)
    assert v.detail["catalog_limit"] == 15
    assert "not convertible" in v.detail["note"]


def test_credits_with_a_rate() -> None:
    entry = LIGHTNING.model_copy(update={"options": {"quota_per_gpu_hour": 0.5}})
    v = view(entry, THU, [(THU - 4 * H, THU)])
    assert (v.used, v.limit, v.unit) == (2.0, 15, QuotaUnit.CREDITS)
    assert remaining(v) == 13
    assert to_gpu_hours(entry, 13) == 26
    assert to_gpu_hours(LIGHTNING.model_copy(update={"options": {}}), 13) is None
    assert to_gpu_hours(LIGHTNING, 0.68) == pytest.approx(1.0)  # catalog: 0.68 credits/T4-hour
    assert to_gpu_hours(KAGGLE, 13) == 13


def _run(store: Store, clock: FakeClock, provider: str, start: float, end: float | None) -> None:
    job, _ = store.create_job(JobSpec(project_dir="/p", script="t.py"), actor="api")
    store.transition(
        job.id,
        from_state=JobState.QUEUED,
        to_state=JobState.ROUTING,
        reason=Reason.ROUTING_STARTED,
        message="routing",
        actor="engine",
    )
    _, att = store.place(
        job.id,
        from_state=JobState.ROUTING,
        provider=provider,
        gpu="T4",
        route_reason="x",
        message="placed",
        detail={},
    )
    store.record_submission(att.id, remote_id="r", remote_url=None, remote_meta={})
    store.update_attempt(att.id, AttemptPatch(state=AttemptState.RUNNING, started_at=start))
    if end is not None:
        clock.set(end)  # terminal attempt states stamp ended_at = now
        store.update_attempt(att.id, AttemptPatch(state=AttemptState.SUCCEEDED))


def test_compute_views_from_a_real_store() -> None:
    clock = FakeClock(SAT - 10 * H)
    store = Store.open_memory(clock)
    _run(store, clock, "kaggle", SAT - 3 * H, SAT - 1 * H)  # last week
    _run(store, clock, "colab", SAT + 2 * H, None)  # still running
    _run(store, clock, "kaggle", SAT + 1 * H, SAT + 3 * H)  # this week: 2h
    clock.set(SAT + 4 * H)
    caps: dict[str, Any] = {"kaggle": Capabilities(live_quota=True), "colab": Capabilities()}
    views = compute_views(
        store,
        [("kaggle", KAGGLE, caps["kaggle"]), ("colab", COLAB, caps["colab"])],
        clock.now(),
        SETTINGS,
    )
    assert views["kaggle"].used == pytest.approx(2.0)
    assert views["kaggle"].source == "estimate"
    assert views["colab"].used == pytest.approx(2.0)
    # a fresh live reading replaces the estimate
    store.record_quota_snapshot(live(7, at=clock.now() - 10, resets_at=SAT + 7 * 24 * H))
    views = compute_views(store, [("kaggle", KAGGLE, caps["kaggle"])], clock.now(), SETTINGS)
    assert (views["kaggle"].used, views["kaggle"].source) == (7, "live")
    store.close()
