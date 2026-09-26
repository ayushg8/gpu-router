"""Daemon-restart recovery (statemachine.RECOVERY) against the in-process engine."""

from __future__ import annotations

from gpu_router.adapters.base import AttemptContext
from gpu_router.engine.capture import read_log_lines
from gpu_router.models import AttemptState, JobState, Reason
from gpu_router.statemachine import RecoveryAction
from tests.unit.engine.conftest import Engine


async def _stop_driving(eng: Engine) -> None:
    await eng.supervisor.stop()


async def test_restart_while_running_reattaches_without_duplicate_logs(eng: Engine) -> None:
    job = await eng.submit(fake={"duration": 20, "steps": 20})
    await eng.until_state(job.id, JobState.RUNNING)
    eng.clock.advance(3)
    await eng.run_until(lambda: eng.store.attempts_for(job.id)[0].log_lines > 2)
    await eng.restart()
    notes = [e for e in eng.store.events_for(job.id) if e.reason == Reason.RECOVERED]
    assert notes
    assert notes[-1].detail["action"] == RecoveryAction.REATTACH
    done = await eng.until_terminal(job.id)
    assert done.state is JobState.DONE
    (attempt,) = eng.store.attempts_for(job.id)
    lines = [
        text for _, text in read_log_lines(eng.paths.job_log(job.id, 1), include_protocol=True)
    ]
    assert len(lines) == attempt.log_lines
    assert len(lines) == len(set(lines))  # every fake log line is unique: no duplicates
    assert len(eng.fake().all_runs()) == 1


async def test_crash_after_place_commit_submits_once(eng: Engine) -> None:
    """Attempt row committed, submit never called: recovery looks the key up, finds
    nothing, and submits under the same key."""
    await _stop_driving(eng)
    spec = eng.spec(fake={"duration": 3})
    job, _ = eng.store.create_job(spec, actor="api")
    job = eng.store.transition(
        job.id,
        from_state=JobState.QUEUED,
        to_state=JobState.ROUTING,
        reason=Reason.ROUTING_STARTED,
        message="x",
        actor="engine",
    )
    _, attempt = eng.store.place(
        job.id,
        from_state=JobState.ROUTING,
        provider="fake",
        gpu="T4",
        route_reason="fake: first fit",
        message="placed",
    )
    await eng.restart()
    done = await eng.until_terminal(job.id)
    assert done.state is JobState.DONE
    (only,) = eng.store.attempts_for(job.id)
    assert only.id == attempt.id
    assert only.state is AttemptState.SUCCEEDED
    assert len(eng.fake().all_runs()) == 1


async def test_crash_after_submit_return_finds_run_by_key(eng: Engine) -> None:
    """Remote run exists but record_submission never committed: recovery must adopt it,
    never start a second run."""
    await _stop_driving(eng)
    spec = eng.spec(fake={"duration": 3})
    job, _ = eng.store.create_job(spec, actor="api")
    job = eng.store.transition(
        job.id,
        from_state=JobState.QUEUED,
        to_state=JobState.ROUTING,
        reason=Reason.ROUTING_STARTED,
        message="x",
        actor="engine",
    )
    job, attempt = eng.store.place(
        job.id,
        from_state=JobState.ROUTING,
        provider="fake",
        gpu="T4",
        route_reason="fake: first fit",
        message="placed",
    )
    eng.fake().submit(
        job, AttemptContext(attempt_id=attempt.id, attempt_key=attempt.attempt_key, n=1)
    )
    await eng.restart()
    done = await eng.until_terminal(job.id)
    assert done.state is JobState.DONE
    assert len(eng.fake().all_runs()) == 1
    assert Reason.SUBMIT_CONFIRMED in eng.reasons(job.id)


async def test_restart_while_cancelling_finishes_cancel(eng: Engine) -> None:
    job = await eng.submit(fake={"duration": 100})
    await eng.until_state(job.id, JobState.RUNNING)
    # user cancel lands, then the daemon dies before the driver acts on it
    await eng.supervisor.stop()
    eng.store.transition(
        job.id,
        from_state=JobState.RUNNING,
        to_state=JobState.CANCELLING,
        reason=Reason.USER_CANCEL,
        message="cancel",
        actor="user:cli",
    )
    await eng.restart()
    done = await eng.until_terminal(job.id)
    assert done.state is JobState.CANCELLED
    run = eng.fake().all_runs()[0]
    assert run.cancelled_at is not None


async def test_restart_while_awaiting_approval_keeps_waiting(eng: Engine) -> None:
    job = await eng.submit(requires_approval=True)
    await eng.until_state(job.id, JobState.AWAITING_APPROVAL)
    await eng.restart()
    await eng.run_until(lambda: True)
    assert eng.job(job.id).state is JobState.AWAITING_APPROVAL
    await eng.supervisor.approve(job.id, actor="user:cli")
    assert (await eng.until_terminal(job.id)).state is JobState.DONE


async def test_restart_while_migrating_places_again(eng: Engine) -> None:
    job = await eng.submit(fake={"duration": 20, "attempts": {"1": {"die_after": 2}}})
    await eng.until_state(job.id, JobState.RUNNING)
    await eng.supervisor.stop()
    eng.clock.advance(5)  # session dies while the daemon is down
    await eng.restart()
    done = await eng.until_terminal(job.id)
    assert done.state is JobState.DONE
    assert [a.state for a in eng.store.attempts_for(job.id)] == [
        AttemptState.LOST,
        AttemptState.SUCCEEDED,
    ]


async def test_restart_with_queued_job_resumes(eng: Engine) -> None:
    await eng.supervisor.stop()
    job, _ = eng.store.create_job(eng.spec(), actor="api")
    await eng.restart()
    assert (await eng.until_terminal(job.id)).state is JobState.DONE


async def test_restart_while_routing_reroutes(eng: Engine) -> None:
    await eng.supervisor.stop()
    job, _ = eng.store.create_job(eng.spec(), actor="api")
    eng.store.transition(
        job.id,
        from_state=JobState.QUEUED,
        to_state=JobState.ROUTING,
        reason=Reason.ROUTING_STARTED,
        message="x",
        actor="engine",
    )
    await eng.restart()
    assert (await eng.until_terminal(job.id)).state is JobState.DONE
