"""Declared hours with checkpoint storage (D48): an agent job that runs past its hours is
asked for a checkpoint first, then stopped for approval; approving resumes it from that
checkpoint. Without an answer it is stopped anyway once the request times out."""

from __future__ import annotations

from gpu_router.models import JobSpec, JobState, Reason, Source
from gpu_router.policy import overrun_limit_s
from gpu_router.statemachine import AttemptState, is_terminal
from tests.unit.checkpoint.test_engine_storage import (
    HOUR,
    CEngine,
    _ack_with_checkpoint,
    _control,
    _running_attempt,
    ceng,  # noqa: F401 - the fixture
    hub_kw,  # noqa: F401 - the fixture
)


def agent(ce: CEngine, **fake: object) -> JobSpec:
    spec = ce.spec(fake=dict(fake), provider="fake", hours=0.5)
    return JobSpec.model_validate({**spec.model_dump(), "source": Source.AGENT})


async def test_overrun_asks_for_a_checkpoint_then_waits_for_approval(ceng: CEngine) -> None:  # noqa: F811
    job = await ceng.submit(
        agent(ceng, duration=3 * HOUR, attempts={"2": {"duration": 60, "checkpoint_every": 0}})
    )
    await ceng.run_until(lambda: _control(ceng, job.id, 1) is not None, max_s=2 * HOUR, step=30)
    ran = ceng.clock.now() - (ceng.store.attempts_for(job.id)[0].started_at or 0)
    assert ran >= overrun_limit_s(0.5)
    (asked,) = ceng.notes(job.id, Reason.HANDOFF_REQUESTED)
    assert asked.detail["why"] == "overrun"
    assert "ran past its declared 30m" in asked.message
    _ack_with_checkpoint(ceng, job.id, 1, seq=7, step=70)
    await ceng.run_until(lambda: ceng.job(job.id).state is JobState.AWAITING_APPROVAL, step=5)
    assert ("running", "migrating", Reason.HOURS_EXCEEDED) in ceng.transitions(job.id)
    stop = ceng.notes(job.id, Reason.HOURS_EXCEEDED)[0]
    assert "saved a checkpoint" in stop.message
    assert "resume from checkpoint 7" in stop.message
    assert ceng.store.attempts_for(job.id)[0].state is AttemptState.CANCELLED

    await ceng.supervisor.approve(job.id, actor="user:cli")
    await ceng.run_until(lambda: _running_attempt(ceng, job.id, 2), step=1)
    second = ceng.store.attempts_for(job.id)[1]
    assert second.resume_checkpoint_id == f"{job.id}.c7"
    await ceng.run_until(lambda: is_terminal(ceng.job(job.id).state), step=5)
    assert ceng.job(job.id).state is JobState.DONE


async def test_overrun_without_an_answer_is_stopped_anyway(ceng: CEngine) -> None:  # noqa: F811
    job = await ceng.submit(agent(ceng, duration=3 * HOUR))
    await ceng.run_until(
        lambda: ceng.job(job.id).state is JobState.AWAITING_APPROVAL, max_s=3 * HOUR, step=30
    )
    (skipped,) = ceng.notes(job.id, Reason.HANDOFF_SKIPPED)
    assert "stopped anyway" in skipped.message
    assert Reason.HOURS_EXCEEDED in [t[2] for t in ceng.transitions(job.id)]
    assert len(ceng.store.attempts_for(job.id)) == 1
