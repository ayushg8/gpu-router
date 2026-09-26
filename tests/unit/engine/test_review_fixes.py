"""Regression tests for the phase-1 review findings (driver / caller robustness)."""

from __future__ import annotations

import asyncio
import concurrent.futures
from collections.abc import Callable
from typing import Any

import pytest

from gpu_router import secrets
from gpu_router.adapters.fake import FakeAdapter
from gpu_router.clock import settle
from gpu_router.config import CallTimeouts, Config, ProviderSettings
from gpu_router.engine import crashpoints
from gpu_router.engine.calls import AdapterCaller
from gpu_router.engine.capture import read_log_lines
from gpu_router.engine.deps import EngineDeps
from gpu_router.engine.supervisor import Supervisor
from gpu_router.errors import InvalidTransition, RateLimited, Unavailable
from gpu_router.models import AttemptState, JobState, Reason
from gpu_router.policy import ApprovalDecision, default_policy
from gpu_router.router.simple import SimpleRouter
from gpu_router.statemachine import AttemptState as AS
from tests.unit.engine.conftest import Engine, engine_config


def _cfg(*, poll: float | None = None, **engine: Any) -> Config:
    base = engine_config(**engine)
    if poll is None:
        return base
    return base.model_copy(
        update={
            "providers": {
                "fake": ProviderSettings(poll_interval_s=poll),
                "fake-b": ProviderSettings(poll_interval_s=poll),
            }
        }
    )


def _raise_once(monkeypatch: pytest.MonkeyPatch, point: str) -> list[str]:
    hits: list[str] = []
    original = crashpoints.crashpoint

    def fake_crashpoint(name: str) -> None:
        if name == point and not hits:
            hits.append(name)
            raise RuntimeError(f"simulated crash at {name}")
        original(name)

    monkeypatch.setattr(crashpoints, "crashpoint", fake_crashpoint)
    return hits


def _assert_log_consistent(eng: Engine, job_id: str) -> None:
    (attempt, *_) = eng.store.attempts_for(job_id)
    lines = [t for _, t in read_log_lines(eng.paths.job_log(job_id, 1), include_protocol=True)]
    assert len(lines) == attempt.log_lines
    assert len(lines) == len(set(lines))  # fake lines are unique: nothing duplicated


# ---------------------------------------------------------------- 1/8: cursor after facts


@pytest.mark.parametrize("engine_cfg", [_cfg(poll=10)])
async def test_checkpoint_not_lost_when_step_fails_mid_checkpoint(
    eng: Engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    hits = _raise_once(monkeypatch, "mid_checkpoint")
    job = await eng.submit(fake={"duration": 30, "steps": 6, "checkpoint_every": 4})
    done = await eng.until_terminal(job.id)
    assert hits == ["mid_checkpoint"]
    assert done.state is JobState.DONE
    seqs = [c.seq for c in eng.store.checkpoints_for(job.id)]
    assert seqs == list(range(1, len(seqs) + 1))
    assert seqs[0] == 1
    _assert_log_consistent(eng, job.id)


@pytest.mark.parametrize("engine_cfg", [_cfg(poll=10)])
async def test_checkpoint_not_lost_when_step_fails_after_running(
    eng: Engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    hits = _raise_once(monkeypatch, "after_running")
    # poll 1 (t=0) sees PENDING; poll 2 (t=10) sees RUNNING plus checkpoints 1-4 at once
    job = await eng.submit(fake={"pending_s": 1, "duration": 30, "steps": 6, "checkpoint_every": 2})
    done = await eng.until_terminal(job.id)
    assert hits == ["after_running"]
    assert done.state is JobState.DONE
    assert [c.seq for c in eng.store.checkpoints_for(job.id)][:2] == [1, 2]
    _assert_log_consistent(eng, job.id)


# ---------------------------------------------------------------- 2: unreachability clock


async def test_downtime_does_not_count_as_unreachable(
    eng: Engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = await eng.submit(fake={"duration": 5 * 3600, "steps": 10})
    await eng.until_state(job.id, JobState.RUNNING)
    await eng.supervisor.stop()
    eng.clock.advance(4 * 3600)  # daemon down / laptop asleep overnight

    calls = {"n": 0}
    original = FakeAdapter.status

    def flaky_status(self: FakeAdapter, ref: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 1:
            raise Unavailable("wi-fi not up yet", provider="fake")
        return original(self, ref)

    monkeypatch.setattr(FakeAdapter, "status", flaky_status)
    await eng.restart()
    await eng.run_until(lambda: calls["n"] >= 3)
    assert eng.job(job.id).state is JobState.RUNNING
    (attempt,) = eng.store.attempts_for(job.id)
    assert attempt.state is AttemptState.RUNNING
    assert Reason.STATUS_LOST not in eng.reasons(job.id)


def _ambiguous_attempt(eng: Engine, **fake: Any) -> tuple[str, str]:
    spec = eng.spec(fake={"duration": 3, **fake})
    job, _ = eng.store.create_job(spec, actor="api")
    eng.store.transition(
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
    from gpu_router.models import AttemptPatch

    eng.store.update_attempt(attempt.id, AttemptPatch(error_kind="ambiguous:Unavailable"))
    return job.id, attempt.id


async def test_old_ambiguous_attempt_not_abandoned_on_first_lookup_error(
    eng: Engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    await eng.supervisor.stop()
    job_id, attempt_id = _ambiguous_attempt(eng)
    eng.clock.advance(4 * 3600)
    calls = {"n": 0}
    original = FakeAdapter.lookup_by_key

    def flaky_lookup(self: FakeAdapter, key: str) -> Any:
        calls["n"] += 1
        if calls["n"] == 1:
            raise Unavailable("down", provider="fake")
        return original(self, key)

    monkeypatch.setattr(FakeAdapter, "lookup_by_key", flaky_lookup)
    await eng.restart()
    done = await eng.until_terminal(job_id)
    assert done.state is JobState.DONE
    first = eng.store.get_attempt(attempt_id)
    assert first.state is not AttemptState.ABANDONED
    assert eng.store.excluded_providers(job_id) == set()


# ---------------------------------------------------------------- 9: outage never excludes


@pytest.mark.parametrize("engine_cfg", [engine_config(unreachable_lost_after_s=100)])
async def test_lookup_outage_does_not_exclude_provider(
    eng: Engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    await eng.supervisor.stop()
    job_id, attempt_id = _ambiguous_attempt(eng)
    state = {"down": True}
    original = FakeAdapter.lookup_by_key

    def lookup(self: FakeAdapter, key: str) -> Any:
        if state["down"]:
            raise Unavailable("down", provider="fake")
        return original(self, key)

    monkeypatch.setattr(FakeAdapter, "lookup_by_key", lookup)
    await eng.restart()
    await eng.run_until(lambda: eng.store.get_attempt(attempt_id).state is not AS.SUBMITTING)
    state["down"] = False
    first = eng.store.get_attempt(attempt_id)
    assert first.state is AttemptState.ABANDONED
    assert first.error_kind == "Unreachable"
    assert eng.store.excluded_providers(job_id) == set()
    done = await eng.until_terminal(job_id)
    assert done.state is JobState.DONE


# ---------------------------------------------------------------- 3/11: orphaned submits


class HoldingExecutor(concurrent.futures.Executor):
    """Inline, except that calls while `hold` is set are parked until `release()`."""

    def __init__(self) -> None:
        self.hold = False
        self.parked: list[tuple[Callable[[], Any], concurrent.futures.Future[Any]]] = []

    def submit(
        self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any
    ) -> concurrent.futures.Future[Any]:
        fut: concurrent.futures.Future[Any] = concurrent.futures.Future()
        if self.hold and ".submit." in getattr(fn, "__qualname__", ""):
            self.hold = False
            self.parked.append((lambda: fn(*args, **kwargs), fut))
            return fut
        try:
            fut.set_result(fn(*args, **kwargs))
        except BaseException as exc:
            fut.set_exception(exc)
        return fut

    def release(self) -> None:
        for fn, fut in self.parked:
            try:
                fut.set_result(fn())
            except BaseException as exc:
                fut.set_exception(exc)
        self.parked.clear()


def _holding_engine(eng: Engine, executor: HoldingExecutor, config: Config) -> Supervisor:
    from gpu_router.adapters.registry import AdapterRegistry

    registry = AdapterRegistry.build(
        config=config, catalog=eng.registry.catalog, paths=eng.paths, clock=eng.clock
    )
    caller = AdapterCaller(registry, config.engine, executor=executor)
    deps = EngineDeps(
        store=eng.store,
        registry=registry,
        router=SimpleRouter(),
        policy=default_policy(),
        caller=caller,
        clock=eng.clock,
        config=config,
        paths=eng.paths,
    )
    return Supervisor(deps)


def _held_submit_config() -> Config:
    base = engine_config()
    engine = base.engine.model_copy(
        update={"timeouts": CallTimeouts(submit=0.05, status=5, logs=5, cancel=5, fetch=5)}
    )
    return base.model_copy(update={"engine": engine})


async def _swap_supervisor(eng: Engine, sup: Supervisor) -> None:
    await eng.supervisor.stop()
    eng.supervisor = sup
    eng.registry = sup.deps.registry
    await sup.start()


async def _until_parked(eng: Engine, executor: HoldingExecutor) -> None:
    await eng.run_until(lambda: bool(executor.parked))
    await asyncio.sleep(0.15)  # real time: the submit timeout fires
    await settle(30)


async def test_timed_out_submit_is_adopted_not_duplicated(eng: Engine) -> None:
    executor = HoldingExecutor()
    config = _held_submit_config()
    await _swap_supervisor(eng, _holding_engine(eng, executor, config))
    executor.hold = True
    job = await eng.submit(fake={"duration": 3})
    await _until_parked(eng, executor)
    assert Reason.SUBMIT_AMBIGUOUS in eng.reasons(job.id)
    # lookups run meanwhile would find nothing: the driver must keep waiting
    eng.clock.advance(60)
    await settle(30)
    eng.clock.advance(60)
    await settle(30)
    (attempt,) = eng.store.attempts_for(job.id)
    assert attempt.state is AttemptState.SUBMITTING
    executor.release()  # the slow upload finishes and creates the remote run
    done = await eng.until_terminal(job.id)
    assert done.state is JobState.DONE
    (attempt,) = eng.store.attempts_for(job.id)
    assert attempt.state is AttemptState.SUCCEEDED
    assert len(eng.fake().all_runs()) == 1


async def test_orphan_run_from_late_submit_is_cancelled(eng: Engine) -> None:
    executor = HoldingExecutor()
    config = _held_submit_config().model_copy(
        update={"engine": _held_submit_config().engine.model_copy(update={"cancel_timeout_s": 20})}
    )
    await _swap_supervisor(eng, _holding_engine(eng, executor, config))
    executor.hold = True
    job = await eng.submit(fake={"duration": 1000})
    await _until_parked(eng, executor)
    await eng.supervisor.cancel(job.id, actor="user:cli")
    await eng.run_until(lambda: eng.job(job.id).state is JobState.CANCELLED, max_s=120)
    (attempt,) = eng.store.attempts_for(job.id)
    assert attempt.state is AttemptState.ABANDONED
    executor.release()
    await eng.run_until(lambda: Reason.ORPHAN_CANCELLED in eng.reasons(job.id), max_s=60)
    (run,) = eng.fake().all_runs()
    assert run.cancelled_at is not None


# ---------------------------------------------------------------- 4/12: provision cancel


@pytest.mark.parametrize("engine_cfg", [engine_config(provision_timeout_s=10)])
async def test_failed_provision_cancel_keeps_attempt_live(
    eng: Engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    def cancel(self: FakeAdapter, ref: Any) -> None:
        raise RateLimited("slow down", provider="fake")

    monkeypatch.setattr(FakeAdapter, "cancel", cancel)
    job = await eng.submit(fake={"pending_s": 30, "duration": 2})
    done = await eng.until_terminal(job.id)
    assert done.state is JobState.DONE
    (attempt,) = eng.store.attempts_for(job.id)  # never re-placed
    assert attempt.state is AttemptState.SUCCEEDED
    assert len(eng.fake().all_runs()) == 1


@pytest.mark.parametrize("engine_cfg", [engine_config(provision_timeout_s=10, cancel_timeout_s=20)])
async def test_unconfirmed_provision_cancel_is_abandoned_not_cancelled(
    eng: Engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    def cancel(self: FakeAdapter, ref: Any) -> None:
        raise RateLimited("slow down", provider="fake")

    monkeypatch.setattr(FakeAdapter, "cancel", cancel)
    spec = eng.spec(options={"fake": {"pending_s": 10_000}, "fake-b": {"duration": 2}})
    job = await eng.submit(spec)
    done = await eng.until_terminal(job.id)
    assert done.state is JobState.DONE
    assert done.provider == "fake-b"
    first = eng.store.attempts_for(job.id)[0]
    assert first.state is AttemptState.ABANDONED
    assert Reason.ATTEMPT_ABANDONED in eng.reasons(job.id)


# ---------------------------------------------------------------- 5: cancel during fetch


async def test_cancel_during_fetch_is_not_an_internal_error(
    eng: Engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = FakeAdapter.fetch
    fetches = {"n": 0}
    job_ref: dict[str, str] = {}

    # make the cancel land before the success transition: run it synchronously
    def fetch_sync(self: FakeAdapter, ref: Any, dest: Any) -> Any:
        fetches["n"] += 1
        job = eng.store.get_job(job_ref["id"])
        eng.store.transition(
            job.id,
            from_state=job.state,
            to_state=JobState.CANCELLING,
            reason=Reason.USER_CANCEL,
            message="cancel requested by you",
            actor="user:cli",
        )
        return original(self, ref, dest)

    monkeypatch.setattr(FakeAdapter, "fetch", fetch_sync)
    job = await eng.submit(fake={"duration": 2})
    job_ref["id"] = job.id
    done = await eng.until_terminal(job.id)
    assert done.state is JobState.CANCELLED
    assert Reason.INTERNAL_ERROR not in eng.reasons(job.id)
    assert Reason.OUTPUTS_KEPT in eng.reasons(job.id)
    assert fetches["n"] == 1  # not downloaded twice


async def test_invalid_transition_from_moved_state_is_not_a_bug(eng: Engine) -> None:
    job = await eng.submit(fake={"duration": 1000})
    await eng.until_state(job.id, JobState.RUNNING)
    driver = eng.supervisor._drivers[job.id]
    stale = eng.job(job.id)
    await eng.supervisor.cancel(job.id, actor="user:cli")
    assert driver._state_moved(stale)
    with pytest.raises(InvalidTransition):
        eng.store.transition(
            job.id,
            from_state=JobState.CANCELLING,
            to_state=JobState.DONE,
            reason=Reason.COMPLETED,
            message="x",
            actor="engine",
        )


# ---------------------------------------------------------------- 6: stale approval


class AlwaysAsk:
    name = "always"

    def evaluate(self, job: Any, decision: Any, candidate: Any) -> ApprovalDecision:
        return ApprovalDecision(required=True, reason="always ask", rule="always")


async def test_new_approval_request_clears_old_approval(eng: Engine) -> None:
    object.__setattr__(eng.supervisor.deps, "policy", AlwaysAsk())
    job = await eng.submit(fake={"duration": 100, "die_after": 5})
    await eng.until_state(job.id, JobState.AWAITING_APPROVAL)
    await eng.supervisor.approve(job.id, actor="user:cli")
    await eng.until_state(job.id, JobState.RUNNING)
    # the session dies -> migrating -> policy asks again: must wait for a NEW approval
    await eng.run_until(
        lambda: (
            [e.to_state for e in eng.store.events_for(job.id, limit=10_000)].count(
                JobState.AWAITING_APPROVAL
            )
            == 2
        )
    )
    eng.clock.advance(30)
    await settle(30)
    current = eng.job(job.id)
    assert current.state is JobState.AWAITING_APPROVAL
    assert current.approved_at is None
    assert len(eng.store.attempts_for(job.id)) == 1


# ---------------------------------------------------------------- 7: secrets after restart


async def test_secrets_registered_for_redaction_after_restart(eng: Engine) -> None:
    value = "0123456789abcdef0123456789abcdef01234567"
    secrets.set_secret("WANDB_API_KEY", value)
    job = await eng.submit(fake={"duration": 1000}, secrets=["WANDB_API_KEY"])
    await eng.until_state(job.id, JobState.RUNNING)
    await eng.supervisor.stop()
    secrets._reset_redaction_for_tests()  # a new daemon process knows no secret values
    assert value in secrets.redact(f"key={value}")
    await eng.restart()
    eng.clock.advance(2)
    await eng.run_until(lambda: value not in secrets.redact(f"key={value}"), max_s=10)
    assert value not in secrets.redact(f"key={value}")
