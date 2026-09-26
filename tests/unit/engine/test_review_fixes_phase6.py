"""Phase-6 review fixes in the engine (D48): reserved secret names, agent jobs that read
secrets ask first, and declared hours are enforced for agent jobs."""

from __future__ import annotations

import pytest

from gpu_router import secrets
from gpu_router.errors import InvalidSpec
from gpu_router.models import JobSpec, Source
from gpu_router.policy import PolicyConfig, PolicyRules, RulesPolicy, overrun_limit_s
from gpu_router.statemachine import AttemptState, JobState, Reason
from tests.unit.engine.conftest import Engine


def agent_spec(eng: Engine, **fields: object) -> JobSpec:
    spec = eng.spec(**fields)  # type: ignore[arg-type]
    return JobSpec.model_validate({**spec.model_dump(), "source": Source.AGENT})


# ------------------------------------------------------------ finding 1: reserved secrets


@pytest.mark.parametrize(
    "name", ["HF_TOKEN", "hf_token", "kaggle", "KAGGLE_API_TOKEN", "HF_TOKEN_REMOTE"]
)
async def test_reserved_secret_names_are_refused_at_intake(eng: Engine, name: str) -> None:
    with pytest.raises(InvalidSpec) as err:
        await eng.supervisor.submit(eng.spec(secrets=[name]), actor="api")
    assert "gpu-router's own provider credentials" in err.value.message
    assert err.value.detail == {"field": "secrets"}
    assert eng.store.non_terminal_jobs() == []


async def test_a_stored_spec_naming_a_reserved_secret_never_ships_it(eng: Engine) -> None:
    """A job created before the intake rule (the store is not re-validated) fails at
    submit instead of exporting the admin token."""
    secrets.set_secret("HF_TOKEN", "hf_" + "a" * 36)
    job, _ = eng.store.create_job(eng.spec(secrets=["HF_TOKEN"]), actor="api")
    eng.supervisor._start_driver(job.id)
    done = await eng.until_terminal(job.id)
    assert done.state is JobState.FAILED
    assert "own credentials" in done.message
    attempt = eng.store.attempts_for(job.id)[0]
    assert attempt.state is AttemptState.REJECTED
    assert attempt.error_kind == "ReservedSecret"


async def test_agent_job_reading_secrets_waits_for_approval(eng: Engine) -> None:
    secrets.set_secret("WANDB_API_KEY", "w" * 40)
    job = await eng.submit(agent_spec(eng, hours=0.25, secrets=["WANDB_API_KEY"]))
    waiting = await eng.until_state(job.id, JobState.AWAITING_APPROVAL)
    assert waiting.approval_reason is not None
    assert "reads Keychain secrets WANDB_API_KEY" in waiting.approval_reason
    # the same job from the user (cli/api) runs without asking
    user = await eng.submit(hours=0.25, secrets=["WANDB_API_KEY"])
    assert (await eng.until_terminal(user.id)).state is JobState.DONE


# ------------------------------------------------------------ finding 3: declared hours


async def test_agent_job_past_its_declared_hours_is_stopped_for_approval(eng: Engine) -> None:
    hours = 0.01  # 36 s declared -> stopped after max(54 s, 36 s + 15 min) of running
    limit = overrun_limit_s(hours)
    job = await eng.submit(agent_spec(eng, hours=hours, fake={"duration": 5000, "steps": 50}))
    await eng.until_state(job.id, JobState.RUNNING)
    await eng.run_until(
        lambda: eng.job(job.id).state is JobState.AWAITING_APPROVAL, max_s=limit + 120, step=5
    )
    waiting = eng.job(job.id)
    assert ("running", "migrating", Reason.HOURS_EXCEEDED) in eng.transitions(job.id)
    assert ("migrating", "awaiting_approval", Reason.APPROVAL_REQUIRED) in eng.transitions(job.id)
    assert waiting.approval_reason is not None
    assert waiting.approval_reason.startswith("ran past its declared 36s")
    stop = next(e for e in eng.store.events_for(job.id) if e.reason == Reason.HOURS_EXCEEDED)
    assert stop.detail["ran_s"] >= limit
    assert "asking you before it runs longer" in stop.message
    first = eng.store.attempts_for(job.id)[0]
    assert first.state is AttemptState.CANCELLED  # the remote session was stopped
    # nothing is placed again without the user
    eng.clock.advance(600)
    await eng.run_until(lambda: True)
    assert eng.job(job.id).state is JobState.AWAITING_APPROVAL
    assert len(eng.store.attempts_for(job.id)) == 1

    # approving lets it finish: no second stop
    await eng.supervisor.approve(job.id, actor="user:cli")
    await eng.until_state(job.id, JobState.RUNNING, max_s=120)
    done = await eng.until_terminal(job.id, max_s=6000)
    assert done.state is JobState.DONE
    assert eng.reasons(job.id).count(Reason.HOURS_EXCEEDED) == 1


async def test_user_jobs_and_exempt_providers_are_not_stopped(eng: Engine) -> None:
    user = await eng.submit(hours=0.01, fake={"duration": 1100, "steps": 10})
    assert (await eng.until_terminal(user.id, max_s=3000)).state is JobState.DONE
    assert Reason.HOURS_EXCEEDED not in eng.reasons(user.id)

    rules = PolicyRules(exempt_providers=("fake", "fake-b"), ask_secrets=True, enforce_hours=True)
    object.__setattr__(eng.supervisor.deps, "policy", RulesPolicy(PolicyConfig(agent=rules)))
    agent = await eng.submit(agent_spec(eng, hours=0.01, fake={"duration": 1100, "steps": 10}))
    assert (await eng.until_terminal(agent.id, max_s=3000)).state is JobState.DONE
    assert Reason.HOURS_EXCEEDED not in eng.reasons(agent.id)


async def test_hours_enforcement_can_be_turned_off(eng: Engine) -> None:
    rules = PolicyRules(enforce_hours=False, ask_secrets=True)
    object.__setattr__(eng.supervisor.deps, "policy", RulesPolicy(PolicyConfig(agent=rules)))
    agent = await eng.submit(agent_spec(eng, hours=0.01, fake={"duration": 1100, "steps": 10}))
    assert (await eng.until_terminal(agent.id, max_s=3000)).state is JobState.DONE
    assert Reason.HOURS_EXCEEDED not in eng.reasons(agent.id)


def test_overrun_limit() -> None:
    assert overrun_limit_s(0.5) == 45 * 60  # max(45m, 30m + 15m)
    assert overrun_limit_s(12) == 18 * 3600
    assert overrun_limit_s(0.1) == 6 * 60 + 15 * 60
