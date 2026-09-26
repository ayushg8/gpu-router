"""Phase 4/5 review regressions in the engine (D44): waits for capacity vs quota resets,
the queue budget during a quota wait, and invalid-job classification with the reserved Mac.
Engine harness (FakeClock, inline adapter calls) with the phase-5 scoring router."""

from __future__ import annotations

import pytest

from gpu_router.clock import settle
from gpu_router.config import Config
from gpu_router.models import FailureKind, JobState, QuotaSnapshot, QuotaUnit, Reason
from gpu_router.router.base import (
    Candidate,
    RejectCode,
    Rejection,
    RouteDecision,
    RouteOutcome,
    RoutingContext,
)
from gpu_router.router.scoring import ScoringRouter
from tests.unit.engine.conftest import Engine, engine_config

H = 3600.0


@pytest.fixture
def engine_cfg() -> Config:
    # long backoffs: a job that is not woken sleeps 10 min, far past the test's window
    return engine_config(backoff_base_s=600, backoff_cap_s=3600)


def _scoring(eng: Engine) -> None:
    object.__setattr__(eng.supervisor.deps, "router", ScoringRouter())


def _used_up(eng: Engine, provider: str, resets_in: float) -> None:
    now = eng.clock.now()
    eng.store.record_quota_snapshot(
        QuotaSnapshot(
            provider=provider,
            used=30,
            limit=30,
            unit=QuotaUnit.GPU_HOURS,
            resets_at=now + resets_in,
            source="live",
            observed_at=now,
        )
    )


async def test_a_freed_slot_places_a_job_that_would_otherwise_wait_for_a_quota_reset(
    eng: Engine,
) -> None:
    """Finding: fake's quota is used up until a reset in 48h and fake-b is busy (4/4): the
    job slept until the reset although fake-b freed up a minute later."""
    _scoring(eng)
    _used_up(eng, "fake", 48 * H)
    blockers = [
        await eng.submit(
            provider="fake-b",
            options={"fake-b": {"duration": 60, "steps": 3}},
        )
        for _ in range(4)
    ]
    await eng.run_until(
        lambda: all(eng.job(b.id).state is JobState.RUNNING for b in blockers), max_s=30
    )
    job = await eng.submit(hours=1, options={"fake-b": {"duration": 5, "steps": 2}})
    queued = await eng.until_state(job.id, JobState.QUEUED, max_s=10)
    assert "fake: quota used up" in (queued.message or "")
    assert "fake-b: busy (4/4 running)" in (queued.message or "")
    now = eng.clock.now()
    assert queued.not_before is not None
    assert queued.not_before - now <= 600 + 1  # the engine's backoff, not the 48h reset
    await eng.run_until(
        lambda: all(eng.job(b.id).state is JobState.DONE for b in blockers), max_s=120
    )
    freed_at = eng.clock.now()
    placed = await eng.until_state(
        job.id, JobState.PROVISIONING, JobState.RUNNING, JobState.DONE, max_s=30
    )
    assert placed.provider == "fake-b"
    assert eng.clock.now() - freed_at < 30  # woken when the slot freed, not at the backoff
    done = await eng.until_terminal(job.id, max_s=120)
    assert done.state is JobState.DONE


async def test_waiting_for_a_quota_reset_does_not_spend_the_queue_budget(eng: Engine) -> None:
    """Finding: a 2-day quota wait always hit max_queue_wait_s (6h) and failed with gave_up
    right after the reset instead of running."""
    _scoring(eng)
    _used_up(eng, "fake", 48 * H)
    _used_up(eng, "fake-b", 48 * H)
    job = await eng.submit(
        hours=1,
        options={"fake": {"duration": 5, "steps": 2}, "fake-b": {"duration": 5, "steps": 2}},
    )
    queued = await eng.until_state(job.id, JobState.QUEUED, max_s=10)
    assert "retrying in 2d" in (queued.message or "")
    assert queued.waiting_since is not None
    assert queued.waiting_since >= eng.clock.now() + 47 * H  # the wait counts from the reset
    eng.clock.advance(48 * H + 10)
    await settle(30)
    done = await eng.until_terminal(job.id, max_s=300)
    assert done.state is JobState.DONE, done.message
    assert Reason.GAVE_UP not in eng.reasons(job.id)


async def test_a_capacity_wait_still_counts_against_the_queue_budget(eng: Engine) -> None:
    """Only a wait that is purely for quota resets is exempt; one mixed with a busy
    provider keeps max_queue_wait_s."""
    _scoring(eng)
    _used_up(eng, "fake", 48 * H)
    blockers = [
        await eng.submit(provider="fake-b", options={"fake-b": {"duration": 36_000}})
        for _ in range(4)
    ]
    await eng.run_until(
        lambda: all(eng.job(b.id).state is JobState.RUNNING for b in blockers), max_s=30
    )
    job = await eng.submit(hours=1)
    queued = await eng.until_state(job.id, JobState.QUEUED, max_s=10)
    assert queued.waiting_since is not None
    assert queued.waiting_since <= eng.clock.now()


class _InvalidThenNoFit:
    """PLACE on fake once (the fake refuses the job as invalid), then NO_FIT with fake
    excluded and a RESERVED Mac, like the scoring router over local/kaggle/colab."""

    name = "stub"

    def __init__(self) -> None:
        self.calls = 0

    def route(self, ctx: RoutingContext) -> RouteDecision:
        self.calls += 1
        if "fake" not in ctx.excluded:
            cand = Candidate(provider="fake", gpu="T4", vram_gb=16, reason="fake: fits 16GB")
            return RouteDecision(
                outcome=RouteOutcome.PLACE, chosen=cand, candidates=[cand], reason=cand.reason
            )
        rejected = [
            Rejection(provider="fake", code=RejectCode.EXCLUDED, reason="fake: rejected"),
            Rejection(
                provider="local",
                code=RejectCode.RESERVED,
                reason="local: kept for smoke tests (use --smoke or --provider local)",
            ),
        ]
        return RouteDecision(
            outcome=RouteOutcome.NO_FIT, rejected=rejected, reason="no provider fits: ..."
        )


async def test_a_job_every_cloud_provider_refused_is_an_invalid_job(eng: Engine) -> None:
    """Finding: the reserved Mac's rejection made the driver classify a job every cloud
    provider refused as no_provider instead of invalid_job."""
    object.__setattr__(eng.supervisor.deps, "router", _InvalidThenNoFit())
    job = await eng.submit(fake={"invalid": True})
    done = await eng.until_terminal(job.id, max_s=60)
    assert done.state is JobState.FAILED
    assert done.failure_kind is FailureKind.INVALID_JOB
