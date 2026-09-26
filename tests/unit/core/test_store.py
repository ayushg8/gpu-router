"""Store: creation, idempotency, refs, transactional transitions, attempts, checkpoints."""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from pathlib import Path

import pytest

from gpu_router.clock import FakeClock
from gpu_router.errors import (
    AmbiguousJobRef,
    InvalidRequest,
    InvalidTransition,
    JobNotFound,
    StaleState,
)
from gpu_router.lock import InstanceLock
from gpu_router.models import (
    AttemptPatch,
    FailureKind,
    JobEvent,
    JobPatch,
    JobSpec,
    ProviderHealth,
    QuotaSnapshot,
    QuotaUnit,
)
from gpu_router.paths import Paths
from gpu_router.statemachine import AttemptState, JobState, Reason
from gpu_router.store import AttemptChange, Store

J = JobState


class Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[JobEvent]]] = []

    def job_changed(self, job_id: str, events: Sequence[JobEvent]) -> None:
        self.calls.append((job_id, list(events)))


def spec(**over: object) -> JobSpec:
    data: dict[str, object] = {"project_dir": "/work/proj", "script": "train.py"}
    data.update(over)
    return JobSpec.model_validate(data)


def to_provisioning(store: Store, job_id: str, provider: str = "fake") -> str:
    store.transition(
        job_id,
        from_state=J.QUEUED,
        to_state=J.ROUTING,
        reason=Reason.ROUTING_STARTED,
        message="routing",
        actor="engine",
    )
    _, att = store.place(
        job_id,
        from_state=J.ROUTING,
        provider=provider,
        gpu="T4",
        route_reason=f"{provider}: fits",
        message=f"placed on {provider}",
        detail={"candidates": [provider]},
    )
    return att.id


# --------------------------------------------------------------------------- lifecycle


def test_open_requires_held_lock(paths: Paths, clock: FakeClock) -> None:
    unheld = InstanceLock(paths)
    with pytest.raises(RuntimeError, match="InstanceLock"):
        Store.open(paths.db, lock=unheld, clock=clock)
    with InstanceLock.acquire(paths) as lock:
        s = Store.open(paths.db, lock=lock, clock=clock)
        job, _ = s.create_job(spec(), actor="api")
        s.close()
        s2 = Store.open(paths.db, lock=lock, clock=clock)  # reopen: data persisted, no re-migrate
        assert s2.get_job(job.id).id == job.id
        s2.close()
    assert not list(Path(paths.home).glob("gpu.db.bak-*"))


# --------------------------------------------------------------------------- create / read


def test_create_job_fields(store: Store, clock: FakeClock) -> None:
    job, created = store.create_job(spec(name="bert"), actor="user:cli")
    assert created
    assert len(job.id) == 12
    assert job.short_id == job.id[:4]
    assert job.state is J.QUEUED
    assert job.name == "bert"
    assert job.source == "cli"
    assert job.project_dir == "/work/proj"
    assert job.outputs_dir == f"/work/proj/runs/{job.id[:4]}"
    assert job.created_at == job.updated_at == job.waiting_since == clock.now()
    assert job.spec == spec(name="bert")
    assert len(job.spec_hash) == 64
    assert job.version == 0
    [event] = store.events_for(job.id)
    assert event.kind == "transition"
    assert event.from_state is None
    assert event.to_state is J.QUEUED
    assert event.reason == Reason.SUBMITTED
    assert event.actor == "user:cli"


def test_create_job_name_defaults(store: Store) -> None:
    assert store.create_job(spec(), actor="api")[0].name == "train"
    job, _ = store.create_job(spec(script=None, command=["bash", "scripts/run.sh"]), actor="api")
    assert job.name == "bash"


def test_create_job_idempotent_on_request_id(store: Store) -> None:
    rec = Recorder()
    store.set_listener(rec)
    a, created_a = store.create_job(spec(), actor="api", request_id="req-1")
    b, created_b = store.create_job(spec(name="other"), actor="api", request_id="req-1")
    assert created_a
    assert not created_b
    assert a.id == b.id
    assert b.name == "train"
    assert len(rec.calls) == 1
    assert len(store.list_jobs()) == 1
    c, created_c = store.create_job(spec(), actor="api", request_id="req-2")
    assert created_c
    assert c.id != a.id


def test_create_job_avoids_taken_prefixes(store: Store, monkeypatch: pytest.MonkeyPatch) -> None:
    ids_iter = iter(["aaaa00000001", "aaaa00000002", "bbbb00000003"])
    monkeypatch.setattr("gpu_router.ids.secrets.token_hex", lambda _n: next(ids_iter))
    first, _ = store.create_job(spec(), actor="api")
    assert first.id == "aaaa00000001"
    second, _ = store.create_job(spec(), actor="api")
    assert second.id == "bbbb00000003"  # aaaa00000002 skipped: 4-char prefix taken


def test_short_id_grows_when_prefix_shared(store: Store, monkeypatch: pytest.MonkeyPatch) -> None:
    # Every candidate shares the prefix, so new_job_id gives up and returns the last one.
    counter = iter(range(1, 100))
    monkeypatch.setattr("gpu_router.ids.secrets.token_hex", lambda _n: f"abcd{next(counter):08x}")
    a, _ = store.create_job(spec(), actor="api")
    b, _ = store.create_job(spec(), actor="api")
    assert a.id[:4] == b.id[:4] == "abcd"
    a2 = store.get_job(a.id)
    b2 = store.get_job(b.id)
    assert a2.short_id != b2.short_id
    assert len(a2.short_id) > 4
    assert a2.outputs_dir == "/work/proj/runs/abcd"  # fixed at creation


def test_get_job_not_found(store: Store) -> None:
    with pytest.raises(JobNotFound):
        store.get_job("0123456789ab")


def test_resolve_ref_prefixes(store: Store, monkeypatch: pytest.MonkeyPatch) -> None:
    ids_iter = iter(["a7f2c19e0b3d", "a7f9000000aa", "b0000000000c"])
    monkeypatch.setattr("gpu_router.ids.secrets.token_hex", lambda _n: next(ids_iter))
    a, _ = store.create_job(spec(), actor="api")
    b, _ = store.create_job(spec(), actor="api")
    c, _ = store.create_job(spec(), actor="api")

    assert store.resolve_ref("a7f2").id == a.id
    assert store.resolve_ref("A7F2C19E0B3D").id == a.id
    assert store.resolve_ref("  a7f9 ").id == b.id
    assert store.resolve_ref("b").id == c.id
    assert store.resolve_ref(a.id).id == a.id

    with pytest.raises(AmbiguousJobRef) as info:
        store.resolve_ref("a7f")
    assert info.value.detail["matches"] == [a.id, b.id]
    with pytest.raises(AmbiguousJobRef):
        store.resolve_ref("a")
    with pytest.raises(JobNotFound):
        store.resolve_ref("ffff")
    with pytest.raises(InvalidRequest):
        store.resolve_ref("not-hex")
    with pytest.raises(InvalidRequest):
        store.resolve_ref("")


def test_resolve_ref_ambiguous_lists_at_most_ten(
    store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    counter = iter(range(100))
    monkeypatch.setattr(
        "gpu_router.ids.secrets.token_hex", lambda _n: f"cc{next(counter):02x}00000000"
    )
    for _ in range(12):
        store.create_job(spec(), actor="api")
    with pytest.raises(AmbiguousJobRef) as info:
        store.resolve_ref("cc")
    assert len(info.value.detail["matches"]) == 10


def test_list_jobs_filters_and_paging(store: Store, clock: FakeClock) -> None:
    made = []
    for i in range(5):
        job, _ = store.create_job(spec(project_dir=f"/p{i % 2}"), actor="api")
        made.append(job)
        clock.advance(1)
    assert [j.id for j in store.list_jobs()] == [j.id for j in reversed(made)]
    assert [j.id for j in store.list_jobs(limit=2)] == [made[4].id, made[3].id]
    page2 = store.list_jobs(limit=2, before=made[3].created_at)
    assert [j.id for j in page2] == [made[2].id, made[1].id]
    assert {j.id for j in store.list_jobs(project_dir="/p1")} == {made[1].id, made[3].id}
    assert {j.id for j in store.list_jobs(project_dir="/p1/")} == {made[1].id, made[3].id}
    store.transition(
        made[0].id,
        from_state=J.QUEUED,
        to_state=J.CANCELLED,
        reason=Reason.USER_CANCEL,
        message="cancelled",
        actor="user:cli",
    )
    assert [j.id for j in store.list_jobs(states=[J.CANCELLED])] == [made[0].id]
    assert store.list_jobs(states=[]) == []
    assert [j.id for j in store.non_terminal_jobs()] == [j.id for j in made[1:]]
    assert store.count_by_state() == {J.QUEUED: 4, J.CANCELLED: 1}


def test_recent_finished(store: Store, clock: FakeClock) -> None:
    a, _ = store.create_job(spec(), actor="api")
    b, _ = store.create_job(spec(), actor="api")
    for job in (a, b):
        store.transition(
            job.id,
            from_state=J.QUEUED,
            to_state=J.CANCELLED,
            reason=Reason.USER_CANCEL,
            message="x",
            actor="api",
        )
        clock.advance(100)
    assert [j.id for j in store.recent_finished(clock.now() - 150)] == [b.id]
    assert [j.id for j in store.recent_finished(0)] == [b.id, a.id]


# --------------------------------------------------------------------------- transition


def test_transition_writes_job_and_event_atomically(store: Store, clock: FakeClock) -> None:
    rec = Recorder()
    store.set_listener(rec)
    job, _ = store.create_job(spec(), actor="api")
    clock.advance(5)
    updated = store.transition(
        job.id,
        from_state=J.QUEUED,
        to_state=J.ROUTING,
        reason=Reason.ROUTING_STARTED,
        message="choosing a provider",
        actor="engine",
        detail={"k": 1},
        patch=JobPatch(not_before=None, route_reason="thinking"),
    )
    assert updated.state is J.ROUTING
    assert updated.message == "choosing a provider"
    assert updated.route_reason == "thinking"
    assert updated.updated_at == clock.now()
    assert updated.version == job.version + 1
    events = store.events_for(job.id)
    assert [e.reason for e in events] == ["submitted", "routing_started"]
    ev = events[-1]
    assert (ev.from_state, ev.to_state, ev.detail, ev.actor) == (
        J.QUEUED,
        J.ROUTING,
        {"k": 1},
        "engine",
    )
    assert rec.calls[-1] == (job.id, [ev])


def test_transition_patch_message_overrides(store: Store) -> None:
    job, _ = store.create_job(spec(), actor="api")
    updated = store.transition(
        job.id,
        from_state=J.QUEUED,
        to_state=J.ROUTING,
        reason=Reason.ROUTING_STARTED,
        message="event text",
        actor="engine",
        patch=JobPatch(message="status text"),
    )
    assert updated.message == "status text"
    assert store.events_for(job.id)[-1].message == "event text"


def test_transition_invalid_before_touching_db(store: Store) -> None:
    rec = Recorder()
    job, _ = store.create_job(spec(), actor="api")
    store.set_listener(rec)
    with pytest.raises(InvalidTransition):
        store.transition(
            job.id,
            from_state=J.QUEUED,
            to_state=J.RUNNING,
            reason=Reason.STARTED,
            message="x",
            actor="engine",
        )
    # Even for a job that does not exist: the table check comes first.
    with pytest.raises(InvalidTransition):
        store.transition(
            "0123456789ab",
            from_state=J.DONE,
            to_state=J.QUEUED,
            reason=Reason.SUBMITTED,
            message="x",
            actor="engine",
        )
    assert store.get_job(job.id).version == 0
    assert len(store.events_for(job.id)) == 1
    assert rec.calls == []


def test_transition_stale_state(store: Store) -> None:
    job, _ = store.create_job(spec(), actor="api")
    store.transition(
        job.id,
        from_state=J.QUEUED,
        to_state=J.ROUTING,
        reason=Reason.ROUTING_STARTED,
        message="x",
        actor="engine",
    )
    with pytest.raises(StaleState) as info:
        store.transition(
            job.id,
            from_state=J.QUEUED,
            to_state=J.CANCELLED,
            reason=Reason.USER_CANCEL,
            message="x",
            actor="user:cli",
            patch=JobPatch(route_reason="should not be written"),
        )
    assert info.value.expected == "queued"
    assert info.value.actual == "routing"
    after = store.get_job(job.id)
    assert after.state is J.ROUTING
    assert after.route_reason is None
    assert len(store.events_for(job.id)) == 2


def test_transition_missing_job(store: Store) -> None:
    with pytest.raises(JobNotFound):
        store.transition(
            "0123456789ab",
            from_state=J.QUEUED,
            to_state=J.ROUTING,
            reason=Reason.ROUTING_STARTED,
            message="x",
            actor="engine",
        )


def test_transition_rolls_back_when_attempt_change_invalid(store: Store) -> None:
    job, _ = store.create_job(spec(), actor="api")
    att_id = to_provisioning(store, job.id)
    # The attempt is already terminal, so the attempt half of the change is illegal.
    store.update_attempt(att_id, AttemptPatch(state=AttemptState.REJECTED))
    before = store.get_job(job.id)
    n_events = len(store.events_for(job.id))
    with pytest.raises(InvalidTransition):
        store.transition(
            job.id,
            from_state=J.PROVISIONING,
            to_state=J.RUNNING,
            reason=Reason.STARTED,
            message="running",
            actor="engine",
            attempt=AttemptChange(att_id, AttemptPatch(state=AttemptState.RUNNING)),
        )
    after = store.get_job(job.id)
    assert after.state is J.PROVISIONING
    assert after.version == before.version
    assert len(store.events_for(job.id)) == n_events


def test_transition_to_failed_requires_failure_kind(store: Store) -> None:
    job, _ = store.create_job(spec(), actor="api")
    with pytest.raises(ValueError, match="failure_kind"):
        store.transition(
            job.id,
            from_state=J.QUEUED,
            to_state=J.FAILED,
            reason=Reason.GAVE_UP,
            message="x",
            actor="engine",
        )
    with pytest.raises(ValueError, match="failure_kind"):
        store.transition(
            job.id,
            from_state=J.QUEUED,
            to_state=J.ROUTING,
            reason=Reason.ROUTING_STARTED,
            message="x",
            actor="engine",
            patch=JobPatch(failure_kind=FailureKind.INTERNAL),
        )
    failed = store.transition(
        job.id,
        from_state=J.QUEUED,
        to_state=J.FAILED,
        reason=Reason.GAVE_UP,
        message="gave up after 6h",
        actor="engine",
        patch=JobPatch(failure_kind=FailureKind.NO_PROVIDER),
    )
    assert failed.failure_kind is FailureKind.NO_PROVIDER
    assert failed.finished_at is not None


def test_started_at_and_finished_at_stamps(store: Store, clock: FakeClock) -> None:
    job, _ = store.create_job(spec(), actor="api")
    att_id = to_provisioning(store, job.id)
    store.record_submission(att_id, remote_id="r1", remote_url=None, remote_meta={})
    clock.advance(10)
    t_run = clock.now()
    running = store.transition(
        job.id,
        from_state=J.PROVISIONING,
        to_state=J.RUNNING,
        reason=Reason.STARTED,
        message="running",
        actor="engine",
        attempt=AttemptChange(att_id, AttemptPatch(state=AttemptState.RUNNING)),
    )
    assert running.started_at == t_run
    assert running.finished_at is None
    assert store.get_attempt(att_id).started_at == t_run
    clock.advance(10)
    store.transition(
        job.id,
        from_state=J.RUNNING,
        to_state=J.CHECKPOINTING,
        reason=Reason.CHECKPOINT_BEGIN,
        message="ckpt",
        actor="engine",
    )
    back = store.transition(
        job.id,
        from_state=J.CHECKPOINTING,
        to_state=J.RUNNING,
        reason=Reason.CHECKPOINT_END,
        message="ckpt done",
        actor="engine",
    )
    assert back.started_at == t_run  # only the first entry stamps it
    clock.advance(10)
    done = store.transition(
        job.id,
        from_state=J.RUNNING,
        to_state=J.DONE,
        reason=Reason.COMPLETED,
        message="done",
        actor="engine",
        patch=JobPatch(exit_code=0, outputs_fetched=True),
        attempt=AttemptChange(att_id, AttemptPatch(state=AttemptState.SUCCEEDED, exit_code=0)),
    )
    assert done.finished_at == clock.now()
    assert done.outputs_fetched is True
    att = store.get_attempt(att_id)
    assert att.state is AttemptState.SUCCEEDED
    assert att.ended_at == clock.now()
    assert store.events_for(job.id)[-1].attempt_id == att_id


def test_listener_errors_do_not_undo_writes(store: Store) -> None:
    class Boom:
        def job_changed(self, job_id: str, events: Sequence[JobEvent]) -> None:
            raise RuntimeError("listener bug")

    store.set_listener(Boom())
    job, _ = store.create_job(spec(), actor="api")
    assert store.get_job(job.id).state is J.QUEUED


def test_provider_state_notifies_optional_hook(store: Store) -> None:
    seen: list[str] = []

    class WithHook:
        def job_changed(self, job_id: str, events: Sequence[JobEvent]) -> None:
            pass

        def provider_changed(self, provider: str) -> None:
            seen.append(provider)

    store.set_listener(WithHook())
    store.upsert_provider_state("fake", consecutive_failures=2)
    assert seen == ["fake"]

    class HookBoom(WithHook):
        def provider_changed(self, provider: str) -> None:
            raise RuntimeError("listener bug")

    store.set_listener(HookBoom())
    assert store.upsert_provider_state("fake", consecutive_failures=3).consecutive_failures == 3

    class NoHook:
        def job_changed(self, job_id: str, events: Sequence[JobEvent]) -> None:
            pass

    store.set_listener(NoHook())
    store.upsert_provider_state("fake", consecutive_failures=4)


# --------------------------------------------------------------------------- notes / updates


def test_update_job_and_add_note(store: Store, clock: FakeClock) -> None:
    rec = Recorder()
    store.set_listener(rec)
    job, _ = store.create_job(spec(), actor="api")
    updated = store.update_job(
        job.id,
        JobPatch(
            progress_step=10,
            progress_total=100,
            progress_source="helper",
            last_metrics={"loss": 0.5},
        ),
    )
    assert updated.progress.step == 10
    assert updated.progress.fraction == 0.1
    assert updated.last_metrics == {"loss": 0.5}
    assert rec.calls[-1] == (job.id, [])
    assert len(store.events_for(job.id)) == 1

    cleared = store.update_job(job.id, JobPatch(last_metrics=None))
    assert cleared.last_metrics == {}
    assert cleared.progress.step == 10  # untouched: not in fields_set

    clock.advance(3)
    note = store.add_note(
        job.id,
        reason=Reason.RETRY_SCHEDULED,
        message="retrying in 30s",
        actor="engine",
        detail={"delay": 30},
        patch=JobPatch(not_before=clock.now() + 30),
    )
    assert note.kind == "note"
    assert note.to_state is None
    assert store.get_job(job.id).not_before == clock.now() + 30
    assert rec.calls[-1] == (job.id, [note])
    with pytest.raises(JobNotFound):
        store.add_note("0123456789ab", reason=Reason.RECOVERED, message="x", actor="recovery")
    with pytest.raises(JobNotFound):
        store.update_job("0123456789ab", JobPatch(message="x"))


def test_record_approval(store: Store, clock: FakeClock) -> None:
    job, _ = store.create_job(spec(requires_approval=True), actor="agent")
    with pytest.raises(InvalidTransition):
        store.record_approval(job.id, actor="user:cli")
    store.transition(
        job.id,
        from_state=J.QUEUED,
        to_state=J.ROUTING,
        reason=Reason.ROUTING_STARTED,
        message="x",
        actor="engine",
    )
    store.transition(
        job.id,
        from_state=J.ROUTING,
        to_state=J.AWAITING_APPROVAL,
        reason=Reason.APPROVAL_REQUIRED,
        message="needs approval",
        actor="engine",
        patch=JobPatch(approval_reason="submitter asked", provider="fake"),
    )
    approved = store.record_approval(job.id, actor="user:cli")
    assert approved.state is J.AWAITING_APPROVAL
    assert approved.approved_at == clock.now()
    assert approved.approved_by == "user:cli"
    assert store.events_for(job.id)[-1].reason == Reason.APPROVED
    with pytest.raises(JobNotFound):
        store.record_approval("0123456789ab", actor="user:cli")


# --------------------------------------------------------------------------- attempts


def test_place_creates_attempt_and_transitions(store: Store, clock: FakeClock) -> None:
    rec = Recorder()
    store.set_listener(rec)
    job, _ = store.create_job(spec(), actor="api")
    store.update_job(job.id, JobPatch(not_before=clock.now() + 5))
    store.transition(
        job.id,
        from_state=J.QUEUED,
        to_state=J.ROUTING,
        reason=Reason.ROUTING_STARTED,
        message="x",
        actor="engine",
    )
    placed, att = store.place(
        job.id,
        from_state=J.ROUTING,
        provider="fake",
        gpu="T4",
        route_reason="fake: fits 16GB",
        message="placed on fake",
        detail={"candidates": ["fake"]},
    )
    assert placed.state is J.PROVISIONING
    assert placed.provider == "fake"
    assert placed.gpu == "T4"
    assert placed.current_attempt_id == att.id == f"{job.id}.1"
    assert placed.attempt_count == 1
    assert placed.waiting_since is None
    assert placed.not_before is None
    assert placed.route_reason == "fake: fits 16GB"
    assert att.state is AttemptState.SUBMITTING
    assert att.attempt_key == f"gpu-{job.id}-1"
    assert att.created_at == clock.now()
    ev = store.events_for(job.id)[-1]
    assert ev.reason == Reason.PLACED
    assert ev.attempt_id == att.id
    assert ev.detail == {"candidates": ["fake"]}
    assert rec.calls[-1] == (job.id, [ev])
    assert store.current_attempt(placed) == att
    assert store.live_attempts_by_provider() == {"fake": 1}


def test_place_stale_and_invalid(store: Store) -> None:
    job, _ = store.create_job(spec(), actor="api")
    with pytest.raises(InvalidTransition):
        store.place(
            job.id, from_state=J.QUEUED, provider="fake", gpu=None, route_reason="r", message="m"
        )
    with pytest.raises(StaleState):
        store.place(
            job.id, from_state=J.ROUTING, provider="fake", gpu=None, route_reason="r", message="m"
        )
    assert store.attempts_for(job.id) == []


def test_second_live_attempt_is_a_bug(store: Store) -> None:
    job, _ = store.create_job(spec(), actor="api")
    to_provisioning(store, job.id)
    # provisioning -> queued without ending the attempt, then placing again: index fires.
    store.transition(
        job.id,
        from_state=J.PROVISIONING,
        to_state=J.QUEUED,
        reason=Reason.RATE_LIMITED,
        message="x",
        actor="engine",
    )
    store.transition(
        job.id,
        from_state=J.QUEUED,
        to_state=J.ROUTING,
        reason=Reason.ROUTING_STARTED,
        message="x",
        actor="engine",
    )
    with pytest.raises(sqlite3.IntegrityError):
        store.place(
            job.id, from_state=J.ROUTING, provider="fake", gpu=None, route_reason="r", message="m"
        )
    assert store.get_job(job.id).state is J.ROUTING
    assert len(store.attempts_for(job.id)) == 1


def test_record_submission(store: Store) -> None:
    job, _ = store.create_job(spec(), actor="api")
    att_id = to_provisioning(store, job.id)
    att = store.record_submission(
        att_id, remote_id="run-1", remote_url="https://x/1", remote_meta={"slug": "s"}
    )
    assert att.state is AttemptState.SUBMITTED
    assert (att.remote_id, att.remote_url, att.remote_meta) == (
        "run-1",
        "https://x/1",
        {"slug": "s"},
    )
    assert att.submitted_at is not None
    assert store.get_job(job.id).accepted_attempts == 1
    assert store.events_for(job.id)[-1].reason == Reason.SUBMIT_CONFIRMED
    n_events = len(store.events_for(job.id))
    again = store.record_submission(att_id, remote_id="run-1", remote_url=None, remote_meta={})
    assert again == att
    assert store.get_job(job.id).accepted_attempts == 1
    assert len(store.events_for(job.id)) == n_events
    with pytest.raises(InvalidTransition):
        store.record_submission(att_id, remote_id="run-2", remote_url=None, remote_meta={})
    with pytest.raises(JobNotFound):
        store.record_submission("nope.1", remote_id="x", remote_url=None, remote_meta={})


def test_update_attempt_rules(store: Store, clock: FakeClock) -> None:
    rec = Recorder()
    job, _ = store.create_job(spec(), actor="api")
    att_id = to_provisioning(store, job.id)
    store.set_listener(rec)
    same = store.update_attempt(att_id, AttemptPatch(state=AttemptState.SUBMITTING))
    assert same.state is AttemptState.SUBMITTING
    store.update_attempt(att_id, AttemptPatch(log_cursor="c1", log_lines=3, remote_message="hi"))
    att = store.get_attempt(att_id)
    assert (att.log_cursor, att.log_lines, att.remote_message) == ("c1", 3, "hi")
    assert rec.calls[-1] == (job.id, [])
    clock.advance(1)
    rej = store.update_attempt(
        att_id,
        AttemptPatch(state=AttemptState.REJECTED, error_kind="InvalidJob", error_message="no"),
    )
    assert rej.ended_at == clock.now()
    with pytest.raises(InvalidTransition):
        store.update_attempt(att_id, AttemptPatch(state=AttemptState.RUNNING))
    assert store.excluded_providers(job.id) == {"fake"}
    assert store.live_attempts_by_provider() == {}
    assert [a.id for a in store.attempts_in_state([AttemptState.REJECTED])] == [att_id]
    assert store.attempts_in_state([]) == []


def test_excluded_providers_includes_abandoned(store: Store) -> None:
    job, _ = store.create_job(spec(), actor="api")
    att_id = to_provisioning(store, job.id, provider="colab")
    store.update_attempt(att_id, AttemptPatch(state=AttemptState.ABANDONED))
    assert store.excluded_providers(job.id) == {"colab"}


def test_usage_seconds(store: Store, clock: FakeClock) -> None:
    t0 = clock.now()
    job, _ = store.create_job(spec(), actor="api")
    att_id = to_provisioning(store, job.id)
    store.record_submission(att_id, remote_id="r", remote_url=None, remote_meta={})
    store.update_attempt(att_id, AttemptPatch(state=AttemptState.RUNNING, started_at=t0 + 100))
    # Still running: counted up to `until`.
    assert store.usage_seconds("fake", t0, t0 + 400) == 300
    clock.set(t0 + 500)
    store.update_attempt(att_id, AttemptPatch(state=AttemptState.SUCCEEDED))
    assert store.usage_seconds("fake", t0, t0 + 1000) == 400
    assert store.usage_seconds("fake", t0 + 200, t0 + 300) == 100
    assert store.usage_seconds("fake", t0 + 600, t0 + 700) == 0
    assert store.usage_seconds("other", t0, t0 + 1000) == 0
    assert store.usage_seconds("fake", t0 + 10, t0) == 0


# --------------------------------------------------------------------------- events


def test_event_feeds(store: Store) -> None:
    assert store.max_event_seq() == 0
    a, _ = store.create_job(spec(), actor="api")
    b, _ = store.create_job(spec(), actor="api")
    store.add_note(a.id, reason=Reason.RECOVERED, message="recovered", actor="recovery")
    feed = store.events_after(0)
    assert [e.job_id for e in feed] == [a.id, b.id, a.id]
    assert [e.seq for e in feed] == sorted(e.seq for e in feed)
    assert store.max_event_seq() == feed[-1].seq
    assert store.events_after(feed[0].seq, limit=1) == [feed[1]]
    assert [e.seq for e in store.events_for(a.id, after_seq=feed[0].seq)] == [feed[2].seq]


# --------------------------------------------------------------------------- checkpoints


def test_checkpoints(store: Store, clock: FakeClock) -> None:
    job, _ = store.create_job(spec(), actor="api")
    att_id = to_provisioning(store, job.id)
    assert store.latest_checkpoint(job.id) is None
    c1 = store.record_checkpoint(
        job.id, att_id, seq=1, uri="fake://c1", step=10, size_bytes=5, sha256="a", created_at=1.0
    )
    assert c1.id == f"{job.id}.c1"
    assert c1.recorded_at == clock.now()
    again = store.record_checkpoint(
        job.id, att_id, seq=1, uri="fake://c1", step=10, size_bytes=5, sha256="a", created_at=9.0
    )
    assert again == c1
    with pytest.raises(ValueError, match="different"):
        store.record_checkpoint(
            job.id,
            att_id,
            seq=1,
            uri="fake://other",
            step=10,
            size_bytes=5,
            sha256="a",
            created_at=1.0,
        )
    store.record_checkpoint(
        job.id,
        att_id,
        seq=2,
        uri="fake://c2",
        step=20,
        size_bytes=None,
        sha256=None,
        created_at=2.0,
    )
    got = store.get_job(job.id)
    assert got.checkpoint_count == 2
    assert got.last_checkpoint_at == 2.0
    latest = store.latest_checkpoint(job.id)
    assert latest is not None
    assert latest.seq == 2
    assert [c.seq for c in store.checkpoints_for(job.id)] == [1, 2]


# --------------------------------------------------------------------------- providers, quota, meta


def test_provider_state(store: Store, clock: FakeClock) -> None:
    default = store.get_provider_state("kaggle")
    assert default.health is ProviderHealth.UNKNOWN
    assert store.all_provider_states() == {}
    s1 = store.upsert_provider_state("kaggle", health=ProviderHealth.OK, consecutive_failures=0)
    assert s1.health is ProviderHealth.OK
    assert s1.updated_at == clock.now()
    clock.advance(5)
    s2 = store.upsert_provider_state("kaggle", cooldown_until=clock.now() + 60)
    assert s2.health is ProviderHealth.OK  # untouched column kept
    assert s2.cooldown_until == clock.now() + 60
    assert s2.updated_at == clock.now()
    assert set(store.all_provider_states()) == {"kaggle"}
    with pytest.raises(ValueError, match="unknown"):
        store.upsert_provider_state("kaggle", bogus=1)


def test_quota_snapshots(store: Store, clock: FakeClock) -> None:
    def snap(provider: str, used: float, at: float) -> QuotaSnapshot:
        return QuotaSnapshot(
            provider=provider,
            used=used,
            limit=30,
            unit=QuotaUnit.GPU_HOURS,
            source="live",
            detail={"week": 1},
            observed_at=at,
        )

    store.record_quota_snapshot(snap("kaggle", 1, 10))
    store.record_quota_snapshot(snap("kaggle", 2, 20))
    store.record_quota_snapshot(snap("colab", 5, 15))
    latest = store.latest_quota_snapshots()
    assert latest["kaggle"].used == 2
    assert latest["kaggle"].detail == {"week": 1}
    assert latest["colab"].used == 5


def test_quota_snapshots_are_pruned_per_provider(store: Store) -> None:
    def snap(provider: str, used: float, at: float) -> QuotaSnapshot:
        return QuotaSnapshot(
            provider=provider,
            used=used,
            limit=30,
            unit=QuotaUnit.GPU_HOURS,
            source="live",
            observed_at=at,
        )

    store.record_quota_snapshot(snap("colab", 1, 5), keep=3)
    for i in range(10):
        store.record_quota_snapshot(snap("kaggle", i, 10 + i), keep=3)
    rows = store._conn.execute(
        "SELECT provider, used FROM quota_snapshots ORDER BY provider, observed_at"
    ).fetchall()
    assert [(r["provider"], r["used"]) for r in rows] == [
        ("colab", 1),
        ("kaggle", 7),
        ("kaggle", 8),
        ("kaggle", 9),
    ]
    assert store.latest_quota_snapshots()["kaggle"].used == 9


def test_meta(store: Store) -> None:
    assert store.get_meta("instance_id") is None
    store.set_meta("instance_id", "abc")
    store.set_meta("instance_id", "def")
    assert store.get_meta("instance_id") == "def"


def test_close_is_idempotent(clock: FakeClock) -> None:
    s = Store.open_memory(clock)
    s.close()
    s.close()
