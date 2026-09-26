"""End-to-end engine flows: Supervisor + JobDriver + FakeAdapter + in-memory Store."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from gpu_router.models import AttemptState, FailureKind, JobState, ProviderHealth, Reason
from tests.unit.engine.conftest import Engine, engine_config

# --------------------------------------------------------------------------- happy path


async def test_job_runs_to_done_and_fetches_outputs(eng: Engine) -> None:
    job = await eng.submit(fake={"duration": 5, "steps": 5})
    done = await eng.until_terminal(job.id)

    assert done.state is JobState.DONE
    assert done.provider == "fake"
    assert done.exit_code == 0
    assert done.outputs_fetched
    assert done.outputs_dir == f"{eng.project}/runs/{job.id[:4]}"
    result = json.loads((Path(done.outputs_dir) / "result.json").read_text())
    assert result["job_id"] == job.id
    assert [t[1] for t in eng.transitions(job.id)] == [
        "queued",
        "routing",
        "provisioning",
        "running",
        "done",
    ]
    assert Reason.FETCHED in eng.reasons(job.id)
    (attempt,) = eng.store.attempts_for(job.id)
    assert attempt.state is AttemptState.SUCCEEDED
    assert attempt.remote_id is not None
    assert attempt.log_lines > 0
    # the helper protocol reported progress and metrics
    assert done.progress.source == "helper"
    assert done.progress.total == 5
    assert done.progress.step == 5
    assert "loss" in done.last_metrics
    assert eng.paths.job_metrics(job.id).exists()
    assert "finished on fake" in done.message
    assert len(eng.fake().all_runs()) == 1


@pytest.mark.parametrize("exit_code", [0, 3])
async def test_run_finished_between_polls_gets_a_start_time(
    eng: Engine, monkeypatch: pytest.MonkeyPatch, exit_code: int
) -> None:
    """D33: a run that ends before the first poll sees it running (Kaggle polls every 60 s)
    and whose provider reports no start time is charged from submit, so the job and the
    attempt still get started_at (duration in `gpu status`, phase-5 usage ledger)."""
    fake = eng.fake()
    real_status = fake.status

    def status_without_start(ref: Any) -> Any:
        return real_status(ref).model_copy(update={"started_at": None})

    monkeypatch.setattr(fake, "status", status_without_start)
    job = await eng.submit(fake={"duration": 0, "steps": 1, "exit_code": exit_code})
    done = await eng.until_terminal(job.id)
    assert done.state is (JobState.DONE if exit_code == 0 else JobState.FAILED)
    assert "running" not in [t[1] for t in eng.transitions(job.id)]
    (attempt,) = eng.store.attempts_for(job.id)
    assert attempt.submitted_at is not None
    assert attempt.started_at == attempt.submitted_at
    assert done.started_at == attempt.submitted_at
    if exit_code == 0:
        assert "of submit" in done.message


async def test_script_failure_fails_with_exit_code(eng: Engine) -> None:
    job = await eng.submit(fake={"exit_code": 3})
    done = await eng.until_terminal(job.id)
    assert done.state is JobState.FAILED
    assert done.failure_kind is FailureKind.USER_ERROR
    assert done.exit_code == 3
    assert eng.reasons(job.id)[-1] == Reason.SCRIPT_FAILED
    assert "code 3" in done.message
    assert eng.store.attempts_for(job.id)[0].state is AttemptState.FAILED


# --------------------------------------------------------------------------- reroutes


async def test_rate_limit_cools_provider_down_and_reroutes(eng: Engine) -> None:
    spec = eng.spec(options={"fake": {"rate_limit_n": 1, "duration": 3}, "fake-b": {"duration": 3}})
    job = await eng.submit(spec)
    done = await eng.until_terminal(job.id)
    assert done.state is JobState.DONE
    assert done.provider == "fake-b"
    a1, a2 = eng.store.attempts_for(job.id)
    assert (a1.provider, a1.state, a1.error_kind) == ("fake", AttemptState.REJECTED, "RateLimited")
    assert a2.provider == "fake-b"
    assert Reason.RATE_LIMITED in eng.reasons(job.id)
    state = eng.store.get_provider_state("fake")
    assert state.consecutive_failures == 1
    assert state.cooldown_until is not None
    # nothing was created on the rate-limited provider
    assert eng.fake("fake").all_runs() == []


async def test_ambiguous_submit_is_resolved_by_key_before_rerouting(eng: Engine) -> None:
    spec = eng.spec(
        options={"fake": {"unavailable_n": 1, "duration": 3}, "fake-b": {"duration": 3}}
    )
    job = await eng.submit(spec)
    done = await eng.until_terminal(job.id)
    assert done.state is JobState.DONE
    reasons = eng.reasons(job.id)
    assert Reason.SUBMIT_AMBIGUOUS in reasons
    a1 = eng.store.attempts_for(job.id)[0]
    assert a1.state is AttemptState.REJECTED
    assert a1.error_kind == "Unavailable"
    assert sum(len(eng.fake(n).all_runs()) for n in ("fake", "fake-b")) == 1


async def test_quota_exhaustion_mid_run_migrates_to_other_provider(eng: Engine) -> None:
    spec = eng.spec(options={"fake": {"quota_limit": 2, "duration": 10}, "fake-b": {"duration": 3}})
    job = await eng.submit(spec)
    done = await eng.until_terminal(job.id)
    assert done.state is JobState.DONE
    assert done.provider == "fake-b"
    assert ("running", "migrating", "quota_exhausted") in eng.transitions(job.id)
    a1 = eng.store.attempts_for(job.id)[0]
    assert a1.state is AttemptState.LOST
    state = eng.store.get_provider_state("fake")
    assert state.exhausted_until is not None
    assert state.exhausted_until > eng.clock.now()


async def test_session_loss_migrates_and_resumes_from_latest_checkpoint(eng: Engine) -> None:
    spec = eng.spec(
        fake={"duration": 20, "checkpoint_every": 4, "attempts": {"1": {"die_after": 10}}}
    )
    job = await eng.submit(spec)
    done = await eng.until_terminal(job.id)
    assert done.state is JobState.DONE
    trans = eng.transitions(job.id)
    assert ("running", "checkpointing", "checkpoint_begin") in trans
    assert ("checkpointing", "running", "checkpoint_end") in trans
    assert any(t[1] == "migrating" and t[2] == "session_lost" for t in trans)
    a1, a2 = eng.store.attempts_for(job.id)
    assert a1.state is AttemptState.LOST
    latest_before = [c for c in eng.store.checkpoints_for(job.id) if c.attempt_id == a1.id]
    assert latest_before
    assert a2.resume_checkpoint_id == latest_before[-1].id
    assert eng.fake().run_record(a2.remote_id or "").resume_seq == latest_before[-1].seq
    assert done.checkpoint_count >= len(latest_before)


async def test_interactive_job_is_not_migrated(eng: Engine) -> None:
    # the fakes cannot host interactive jobs, so give the router a provider that can
    eng.fake().capabilities = eng.fake().capabilities.model_copy(update={"interactive": True})
    job = await eng.submit(fake={"duration": 20, "die_after": 3}, interactive=True)
    done = await eng.until_terminal(job.id)
    assert done.state is JobState.FAILED
    assert eng.reasons(job.id)[-1] == Reason.INTERACTIVE_LOST


async def test_attempt_budget_ends_in_gave_up(eng: Engine) -> None:
    job = await eng.submit(fake={"duration": 20, "die_after": 2}, max_attempts=1)
    done = await eng.until_terminal(job.id)
    assert done.state is JobState.FAILED
    assert done.failure_kind is FailureKind.NO_PROVIDER
    assert eng.reasons(job.id)[-1] == Reason.GAVE_UP


async def test_invalid_job_excludes_provider(eng: Engine) -> None:
    spec = eng.spec(options={"fake": {"invalid": True}, "fake-b": {"duration": 2}})
    job = await eng.submit(spec)
    done = await eng.until_terminal(job.id)
    assert done.state is JobState.DONE
    assert done.provider == "fake-b"
    assert eng.store.excluded_providers(job.id) == {"fake"}
    assert Reason.PROVIDER_EXCLUDED in eng.reasons(job.id)


async def test_invalid_everywhere_fails_invalid_job(eng: Engine) -> None:
    job = await eng.submit(fake={"invalid": True})  # "fake" key applies to both fakes
    done = await eng.until_terminal(job.id)
    assert done.state is JobState.FAILED
    assert done.failure_kind is FailureKind.INVALID_JOB
    assert eng.reasons(job.id)[-1] == Reason.NO_PROVIDER_FITS


async def test_permanent_error_fails_the_job(eng: Engine) -> None:
    job = await eng.submit(fake={"permanent": True})
    done = await eng.until_terminal(job.id)
    assert done.state is JobState.FAILED
    assert done.failure_kind is FailureKind.PROVIDER_ERROR
    assert eng.reasons(job.id)[-1] == Reason.PROVIDER_PERMANENT


async def test_auth_required_marks_provider_and_reroutes(eng: Engine) -> None:
    spec = eng.spec(options={"fake": {"auth_required": True}, "fake-b": {"duration": 2}})
    job = await eng.submit(spec)
    done = await eng.until_terminal(job.id)
    assert done.state is JobState.DONE
    assert done.provider == "fake-b"
    assert eng.store.get_provider_state("fake").health is ProviderHealth.AUTH_REQUIRED
    assert Reason.AUTH_REQUIRED in eng.reasons(job.id)


async def test_no_fit_fails_immediately(eng: Engine) -> None:
    job = await eng.submit(vram_gb=80)
    done = await eng.until_terminal(job.id)
    assert done.state is JobState.FAILED
    assert done.failure_kind is FailureKind.NO_PROVIDER
    assert "80GB" in done.message
    assert "fake-b" in done.message


async def test_waits_while_all_providers_cool_down(eng: Engine) -> None:
    now = eng.clock.now()
    for name in ("fake", "fake-b"):
        eng.store.upsert_provider_state(name, cooldown_until=now + 30)
    job = await eng.submit()
    queued = await eng.until_state(job.id, JobState.QUEUED)
    await eng.run_until(lambda: eng.job(job.id).not_before is not None)
    assert eng.job(job.id).not_before == pytest.approx(now + 30)
    assert Reason.NO_CAPACITY in eng.reasons(job.id)
    assert queued.state is JobState.QUEUED
    done = await eng.until_terminal(job.id)
    assert done.state is JobState.DONE
    assert eng.store.attempts_for(job.id)[0].created_at >= now + 30


@pytest.mark.parametrize("engine_cfg", [engine_config(provision_timeout_s=10)])
async def test_provision_timeout_cancels_and_reroutes(eng: Engine) -> None:
    spec = eng.spec(options={"fake": {"pending_s": 1000}, "fake-b": {"duration": 2}})
    job = await eng.submit(spec)
    done = await eng.until_terminal(job.id)
    assert done.state is JobState.DONE
    assert done.provider == "fake-b"
    assert ("provisioning", "queued", "provision_timeout") in eng.transitions(job.id)
    a1 = eng.store.attempts_for(job.id)[0]
    assert a1.state is AttemptState.CANCELLED
    assert eng.fake().run_record(a1.remote_id or "").cancelled_at is not None


# --------------------------------------------------------------------------- cancel


async def test_cancel_running_job_stops_remote(eng: Engine) -> None:
    job = await eng.submit(fake={"duration": 100})
    await eng.until_state(job.id, JobState.RUNNING)
    cancelled = await eng.supervisor.cancel(job.id[:4], actor="user:cli")
    assert cancelled.state is JobState.CANCELLING
    done = await eng.until_terminal(job.id)
    assert done.state is JobState.CANCELLED
    assert eng.reasons(job.id)[-1] == Reason.CANCELLED
    (attempt,) = eng.store.attempts_for(job.id)
    assert attempt.state is AttemptState.CANCELLED
    assert eng.fake().run_record(attempt.remote_id or "").cancelled_at is not None
    # idempotent
    again = await eng.supervisor.cancel(job.id, actor="user:cli")
    assert again.state is JobState.CANCELLED


async def test_cancel_queued_job_is_immediate(eng: Engine) -> None:
    now = eng.clock.now()
    for name in ("fake", "fake-b"):
        eng.store.upsert_provider_state(name, cooldown_until=now + 3600)
    job = await eng.submit()
    await eng.run_until(lambda: eng.job(job.id).not_before is not None)
    cancelled = await eng.supervisor.cancel(job.id, actor="user:cli")
    assert cancelled.state is JobState.CANCELLED
    assert eng.store.attempts_for(job.id) == []


@pytest.mark.parametrize("engine_cfg", [engine_config(cancel_timeout_s=20)])
async def test_cancel_unconfirmed_after_timeout(eng: Engine) -> None:
    job = await eng.submit(fake={"duration": 1000, "ignore_cancel": True})
    await eng.until_state(job.id, JobState.RUNNING)
    await eng.supervisor.cancel(job.id, actor="user:cli")
    done = await eng.until_terminal(job.id)
    assert done.state is JobState.CANCELLED
    assert eng.reasons(job.id)[-1] == Reason.CANCEL_UNCONFIRMED
    assert eng.store.attempts_for(job.id)[0].state is AttemptState.ABANDONED


async def test_cancel_after_remote_succeeded_keeps_outputs(eng: Engine) -> None:
    job = await eng.submit(fake={"duration": 2})
    await eng.until_state(job.id, JobState.RUNNING)
    eng.clock.advance(5)  # the remote finishes before the driver polls again
    await eng.supervisor.cancel(job.id, actor="user:cli")
    done = await eng.until_terminal(job.id)
    assert done.state is JobState.CANCELLED
    assert Reason.OUTPUTS_KEPT in eng.reasons(job.id)
    assert (Path(done.outputs_dir or "") / "result.json").exists()


# --------------------------------------------------------------------------- approval


async def test_approval_then_runs(eng: Engine) -> None:
    job = await eng.submit(requires_approval=True)
    waiting = await eng.until_state(job.id, JobState.AWAITING_APPROVAL)
    assert waiting.approval_reason
    assert "fake" in waiting.approval_reason
    assert waiting.provider == "fake"
    await eng.supervisor.approve(job.id, actor="user:cli")
    done = await eng.until_terminal(job.id)
    assert done.state is JobState.DONE
    assert done.approved_by == "user:cli"
    assert ("awaiting_approval", "provisioning", "placed") in eng.transitions(job.id)


async def test_deny(eng: Engine) -> None:
    job = await eng.submit(requires_approval=True)
    await eng.until_state(job.id, JobState.AWAITING_APPROVAL)
    denied = await eng.supervisor.deny(job.id, actor="user:cli", reason="too big")
    assert denied.state is JobState.DENIED
    assert "too big" in denied.message
    assert eng.store.attempts_for(job.id) == []


@pytest.mark.parametrize("engine_cfg", [engine_config(approval_timeout_s=60)])
async def test_approval_expires(eng: Engine) -> None:
    job = await eng.submit(requires_approval=True)
    done = await eng.until_terminal(job.id, max_s=120)
    assert done.state is JobState.DENIED
    assert eng.reasons(job.id)[-1] == Reason.APPROVAL_EXPIRED


async def test_approve_running_job_is_invalid(eng: Engine) -> None:
    from gpu_router.errors import InvalidTransition

    job = await eng.submit(fake={"duration": 100})
    await eng.until_state(job.id, JobState.RUNNING)
    with pytest.raises(InvalidTransition):
        await eng.supervisor.approve(job.id, actor="user:cli")
    with pytest.raises(InvalidTransition):
        await eng.supervisor.deny(job.id, actor="user:cli")


# --------------------------------------------------------------------------- misc


async def test_internal_errors_retry_then_fail(eng: Engine) -> None:
    class Boom:
        name = "boom"

        def route(self, ctx: object) -> object:
            raise RuntimeError("router exploded")

    object.__setattr__(eng.supervisor.deps, "router", Boom())
    job = await eng.submit()
    done = await eng.until_terminal(job.id)
    assert done.state is JobState.FAILED
    assert done.failure_kind is FailureKind.INTERNAL
    notes = [r for r in eng.reasons(job.id) if r == Reason.INTERNAL_ERROR]
    assert len(notes) == eng.config.engine.internal_error_limit  # limit-1 notes + transition


async def test_request_fetch_refetches_outputs(eng: Engine) -> None:
    job = await eng.submit(fake={"duration": 2})
    done = await eng.until_terminal(job.id)
    out = Path(done.outputs_dir or "")
    (out / "model.txt").unlink()
    await eng.supervisor.request_fetch(job.id, actor="user:cli")
    await eng.run_until(lambda: (out / "model.txt").exists())
    assert eng.reasons(job.id)[-2:] == [Reason.FETCH_REQUESTED, Reason.FETCHED]


async def test_request_fetch_rejects_unfinished_job(eng: Engine) -> None:
    from gpu_router.errors import InvalidTransition

    job = await eng.submit(fake={"duration": 100})
    await eng.until_state(job.id, JobState.RUNNING)
    with pytest.raises(InvalidTransition):
        await eng.supervisor.request_fetch(job.id, actor="user:cli")


async def test_dry_route_writes_nothing(eng: Engine) -> None:
    decision = eng.supervisor.dry_route(eng.spec(vram_gb=24))
    assert decision.outcome == "place"
    assert decision.chosen is not None
    assert decision.chosen.provider == "fake-b"
    assert eng.store.list_jobs() == []


async def test_provider_views_and_healthcheck(eng: Engine) -> None:
    views = {v.name: v for v in eng.supervisor.provider_views()}
    assert views["fake"].enabled
    assert views["fake"].health is ProviderHealth.OK
    assert not views["kaggle"].enabled
    assert views["kaggle"].health is ProviderHealth.DISABLED
    eng.fake("fake-b").set_health("unavailable", "simulated outage")
    view = await eng.supervisor.healthcheck("fake-b")
    assert view.health is ProviderHealth.UNAVAILABLE
    assert view.health_reason == "simulated outage"


async def test_unhealthy_provider_is_skipped_until_it_recovers(eng: Engine) -> None:
    for name in ("fake", "fake-b"):
        eng.fake(name).set_health("unavailable", "down")
        await eng.supervisor.healthcheck(name)
    job = await eng.submit()
    await eng.run_until(lambda: eng.job(job.id).not_before is not None)
    assert eng.job(job.id).state is JobState.QUEUED
    eng.fake("fake").set_health(None)
    await eng.supervisor.healthcheck("fake")  # back to OK wakes queued drivers
    done = await eng.until_terminal(job.id)
    assert done.state is JobState.DONE
    assert done.provider == "fake"


async def test_quotas_are_recorded(eng: Engine) -> None:
    snaps = await eng.supervisor.quotas()
    assert {s.provider for s in snaps} == {"fake", "fake-b"}
    assert set(eng.store.latest_quota_snapshots()) == {"fake", "fake-b"}


async def test_missing_secret_fails_with_hint(eng: Engine) -> None:
    job = await eng.submit(secrets=["WANDB_KEY"])
    done = await eng.until_terminal(job.id)
    assert done.state is JobState.FAILED
    assert done.failure_kind is FailureKind.USER_ERROR
    assert "gpu secrets set WANDB_KEY" in done.message
    assert eng.fake().all_runs() == []


async def test_secret_is_passed_to_adapter(eng: Engine) -> None:
    from gpu_router import secrets

    secrets.set_secret("WANDB_KEY", "wandb-secret-value-123")
    seen: list[object] = []
    fake = eng.fake()
    original = fake.submit

    def spy(job: object, ctx: object) -> object:
        seen.append(ctx)
        return original(job, ctx)  # type: ignore[arg-type]

    object.__setattr__(fake, "submit", spy)
    job = await eng.submit(secrets=["WANDB_KEY"])
    done = await eng.until_terminal(job.id)
    assert done.state is JobState.DONE
    ctx = seen[0]
    assert ctx.secrets["WANDB_KEY"].get_secret_value() == "wandb-secret-value-123"  # type: ignore[attr-defined]
    assert ctx.env["GPU_ROUTER_JOB_ID"] == job.id  # type: ignore[attr-defined]
    assert "wandb-secret-value-123" not in json.dumps(
        [e.model_dump(mode="json") for e in eng.store.events_for(job.id)]
    )
