"""FakeAdapter behaviour beyond the contract suite: directives, disk persistence (D9),
timeline, log determinism, quota, health overrides. Owner: group B."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from gpu_router.adapters.base import RemotePhase, RemoteRef
from gpu_router.adapters.fake import FakeAdapter, FakeDirectives
from gpu_router.clock import FakeClock
from gpu_router.errors import (
    AuthRequired,
    InvalidJob,
    NotFound,
    QuotaExhausted,
    RateLimited,
    Unavailable,
)
from gpu_router.models import Checkpoint, ProviderHealth
from gpu_router.protocol import parse_line
from tests.contract.harness import make_ctx
from tests.unit.adapters.conftest import MakeFake, MakeJob

# --------------------------------------------------------------------------- directives


def test_directives_default_and_provider_fallback() -> None:
    assert FakeDirectives.for_attempt({}, "fake", 1) == FakeDirectives()
    d = FakeDirectives.for_attempt({"fake": {"duration": 3}}, "fake-b", 1)
    assert d.duration == 3
    d = FakeDirectives.for_attempt(
        {"fake": {"duration": 3}, "fake-b": {"duration": 7}}, "fake-b", 1
    )
    assert d.duration == 7


def test_directives_per_attempt_override() -> None:
    opts = {"fake": {"duration": 5, "die_after": 1, "attempts": {"2": {"die_after": None}}}}
    assert FakeDirectives.for_attempt(opts, "fake", 1).die_after == 1
    second = FakeDirectives.for_attempt(opts, "fake", 2)
    assert second.die_after is None
    assert second.duration == 5


def test_unknown_directive_is_invalid_job() -> None:
    with pytest.raises(InvalidJob, match="bogus"):
        FakeDirectives.for_attempt({"fake": {"bogus": 1}}, "fake", 1)


def test_bad_directive_type_is_invalid_job() -> None:
    with pytest.raises(InvalidJob):
        FakeDirectives.for_attempt({"fake": {"duration": -1}}, "fake", 1)


# --------------------------------------------------------------------------- timeline


def test_pending_then_running_then_succeeded(
    fake: FakeAdapter, make_fake_job: MakeJob, clock: FakeClock
) -> None:
    job = make_fake_job(pending_s=2, duration=3)
    ref = fake.submit(job, make_ctx(job))
    st = fake.status(ref)
    assert st.phase is RemotePhase.PENDING
    assert st.started_at is None
    clock.advance(2)
    st = fake.status(ref)
    assert st.phase is RemotePhase.RUNNING
    assert st.started_at == clock.now()
    clock.advance(3)
    st = fake.status(ref)
    assert st.phase is RemotePhase.SUCCEEDED
    assert st.exit_code == 0
    assert st.ended_at == clock.now()
    assert st.gpu == "T4"


def test_fail_at_exits_with_code_one(
    fake: FakeAdapter, make_fake_job: MakeJob, clock: FakeClock
) -> None:
    job = make_fake_job(duration=10, fail_at=2)
    ref = fake.submit(job, make_ctx(job))
    clock.advance(1.9)
    assert fake.status(ref).phase is RemotePhase.RUNNING
    clock.advance(0.1)
    st = fake.status(ref)
    assert st.phase is RemotePhase.FAILED
    assert st.exit_code == 1


def test_die_after_is_lost_with_reason(
    fake: FakeAdapter, make_fake_job: MakeJob, clock: FakeClock
) -> None:
    job = make_fake_job(duration=10, die_after=4)
    ref = fake.submit(job, make_ctx(job))
    clock.advance(4)
    st = fake.status(ref)
    assert st.phase is RemotePhase.LOST
    assert st.lost_reason == "session limit"
    assert not st.quota_exhausted


def test_per_attempt_directives_let_retry_succeed(
    fake: FakeAdapter, make_fake_job: MakeJob, clock: FakeClock
) -> None:
    job = make_fake_job(duration=5, die_after=1, attempts={"2": {"die_after": None}})
    first = fake.submit(job, make_ctx(job, 1))
    clock.advance(1)
    assert fake.status(first).phase is RemotePhase.LOST
    second = fake.submit(job, make_ctx(job, 2))
    clock.advance(5)
    assert fake.status(second).phase is RemotePhase.SUCCEEDED


# --------------------------------------------------------------------------- persistence


def test_state_survives_a_new_adapter_instance(
    make_fake: MakeFake, make_fake_job: MakeJob, clock: FakeClock
) -> None:
    a = make_fake()
    job = make_fake_job(duration=4)
    ctx = make_ctx(job)
    ref = a.submit(job, ctx)
    clock.advance(2)
    b = make_fake()  # "daemon restarted"
    assert b.status(ref).phase is RemotePhase.RUNNING
    found = b.lookup_by_key(ctx.attempt_key)
    assert found == ref
    assert b.submit(job, ctx) == ref
    clock.advance(2)
    assert b.status(ref).phase is RemotePhase.SUCCEEDED
    assert len(b.all_runs()) == 1


def test_rate_limit_counter_is_per_job_and_persists(
    make_fake: MakeFake, make_fake_job: MakeJob
) -> None:
    job = make_fake_job(rate_limit_n=2, duration=1)
    with pytest.raises(RateLimited) as info:
        make_fake().submit(job, make_ctx(job, 1))
    assert info.value.retry_after == 1
    with pytest.raises(RateLimited):
        make_fake().submit(job, make_ctx(job, 2))
    ref = make_fake().submit(job, make_ctx(job, 3))
    assert ref.remote_id.endswith("-3")
    other = make_fake_job(rate_limit_n=1, duration=1)
    with pytest.raises(RateLimited):
        make_fake().submit(other, make_ctx(other, 1))


def test_unavailable_n_then_success(fake: FakeAdapter, make_fake_job: MakeJob) -> None:
    job = make_fake_job(unavailable_n=1)
    ctx = make_ctx(job)
    with pytest.raises(Unavailable):
        fake.submit(job, ctx)
    assert fake.lookup_by_key(ctx.attempt_key) is None
    assert fake.submit(job, make_ctx(job, 2)).remote_id


def test_concurrent_submits_with_one_key_create_one_run(
    make_fake: MakeFake, make_fake_job: MakeJob
) -> None:
    job = make_fake_job(duration=1)
    ctx = make_ctx(job)
    refs: list[RemoteRef] = []
    lock = threading.Lock()

    def go() -> None:
        ref = make_fake().submit(job, ctx)
        with lock:
            refs.append(ref)

    threads = [threading.Thread(target=go) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len({r.remote_id for r in refs}) == 1
    assert len(make_fake().all_runs()) == 1


def test_unsafe_remote_ids_are_not_found(fake: FakeAdapter, tmp_path: Path) -> None:
    for rid in ("../../etc", "a/b", ".hidden", ""):
        with pytest.raises(NotFound):
            fake.run_record(rid)
    fake.cancel(RemoteRef(remote_id="../x"))
    assert fake.lookup_by_key("../x") is None


def test_runs_are_isolated_per_provider_name(make_fake: MakeFake, make_fake_job: MakeJob) -> None:
    a, b = make_fake("fake"), make_fake("fake-b")
    job = make_fake_job()
    ref = a.submit(job, make_ctx(job))
    with pytest.raises(NotFound):
        b.status(ref)
    assert b.all_runs() == []


# --------------------------------------------------------------------------- cancel


def test_cancel_while_pending_never_starts(
    fake: FakeAdapter, make_fake_job: MakeJob, clock: FakeClock
) -> None:
    job = make_fake_job(pending_s=5, duration=5)
    ref = fake.submit(job, make_ctx(job))
    clock.advance(1)
    fake.cancel(ref)
    st = fake.status(ref)
    assert st.phase is RemotePhase.CANCELLED
    assert st.started_at is None
    assert fake.quota().used == 0


def test_cancel_keeps_already_returned_log_lines(
    fake: FakeAdapter, make_fake_job: MakeJob, clock: FakeClock
) -> None:
    job = make_fake_job(duration=10, steps=10)
    ref = fake.submit(job, make_ctx(job))
    clock.advance(3)
    before = [line for c in fake.logs(ref) for line in c.lines]
    fake.cancel(ref)
    clock.advance(1)
    after = [line for c in fake.logs(ref) for line in c.lines]
    assert after[: len(before)] == before
    assert after[-1] == "fake: cancelled"


def test_ignore_cancel_keeps_running(
    fake: FakeAdapter, make_fake_job: MakeJob, clock: FakeClock
) -> None:
    job = make_fake_job(duration=4, ignore_cancel=True)
    ref = fake.submit(job, make_ctx(job))
    fake.cancel(ref)
    assert fake.status(ref).phase is RemotePhase.RUNNING
    clock.advance(4)
    assert fake.status(ref).phase is RemotePhase.SUCCEEDED


def test_cancel_after_finish_is_a_noop(
    fake: FakeAdapter, make_fake_job: MakeJob, clock: FakeClock
) -> None:
    job = make_fake_job(duration=1)
    ref = fake.submit(job, make_ctx(job))
    clock.advance(2)
    fake.cancel(ref)
    assert fake.status(ref).phase is RemotePhase.SUCCEEDED
    assert fake.run_record(ref.remote_id).cancelled_at is None


# --------------------------------------------------------------------------- logs


def test_log_lines_carry_metrics_and_checkpoints(
    fake: FakeAdapter, make_fake_job: MakeJob, clock: FakeClock
) -> None:
    job = make_fake_job(duration=10, steps=10, checkpoint_every=4)
    ref = fake.submit(job, make_ctx(job))
    clock.advance(10)
    lines = [line for c in fake.logs(ref) for line in c.lines]
    events = [e for line in lines if (e := parse_line(line)) is not None]
    assert events[0].t == "hello"
    totals = [e for e in events if e.t == "total"]
    assert totals[0].total == 10
    metrics = [e for e in events if e.t == "metric"]
    assert [m.step for m in metrics] == list(range(1, 11))
    assert metrics[0].metrics["loss"] == pytest.approx(2.0 * 0.97)
    ends = [e for e in events if e.t == "ckpt_end"]
    assert [e.seq for e in ends] == [1, 2]
    assert ends[0].uri == f"fake://fake/{ref.remote_id}/ckpt-1"
    assert ends[0].step == 4
    assert events[-1].t == "exit"
    assert events[-1].code == 0
    assert "step 10/10 loss=" in "\n".join(lines)


def test_resume_logs_first_and_continues_checkpoint_seq(
    fake: FakeAdapter, make_fake_job: MakeJob, clock: FakeClock
) -> None:
    job = make_fake_job(duration=5, steps=5, checkpoint_every=2)
    ckpt = Checkpoint(
        id=f"{job.id}.c3",
        job_id=job.id,
        attempt_id=f"{job.id}.1",
        seq=3,
        uri="fake://fake/x/ckpt-3",
        created_at=clock.now(),
        recorded_at=clock.now(),
    )
    ref = fake.submit(job, make_ctx(job, 2, resume_from=ckpt))
    clock.advance(5)
    lines = [line for c in fake.logs(ref) for line in c.lines]
    assert lines[1] == "resuming from checkpoint 3"
    seqs = [e.seq for line in lines if (e := parse_line(line)) and e.t == "ckpt_end"]
    assert seqs == [4, 5]


def test_logs_follow_cursor_and_chunking(
    fake: FakeAdapter, make_fake_job: MakeJob, clock: FakeClock
) -> None:
    job = make_fake_job(duration=1, steps=1500)
    ref = fake.submit(job, make_ctx(job))
    clock.advance(1)
    chunks = list(fake.logs(ref))
    assert len(chunks) >= 3
    assert all(len(c.lines) <= 1000 for c in chunks)
    assert [c.eof for c in chunks] == [False] * (len(chunks) - 1) + [True]
    mid = chunks[0].cursor
    rest = [line for c in fake.logs(ref, since=mid) for line in c.lines]
    everything = [line for c in chunks for line in c.lines]
    assert rest == everything[int(mid) :]
    assert next(iter(fake.logs(ref, follow=True, since=chunks[-1].cursor))).lines == []


def test_failed_run_logs_traceback_and_exit_code(
    fake: FakeAdapter, make_fake_job: MakeJob, clock: FakeClock
) -> None:
    job = make_fake_job(duration=2, exit_code=7)
    ref = fake.submit(job, make_ctx(job))
    clock.advance(2)
    lines = [line for c in fake.logs(ref) for line in c.lines]
    assert any(line.startswith("Traceback") for line in lines)
    last = parse_line(lines[-1])
    assert last is not None
    assert last.code == 7


# --------------------------------------------------------------------------- fetch


def test_fetch_requires_success_then_writes_result(
    fake: FakeAdapter, make_fake_job: MakeJob, clock: FakeClock, tmp_path: Path
) -> None:
    job = make_fake_job(duration=2, steps=20)
    ref = fake.submit(job, make_ctx(job))
    with pytest.raises(NotFound):
        fake.fetch(ref, tmp_path / "out")
    clock.advance(2)
    res = fake.fetch(ref, tmp_path / "out")
    assert res.files == 2
    assert res.bytes > 0
    data = json.loads((tmp_path / "out" / "result.json").read_text())
    assert data["job_id"] == job.id
    assert data["steps"] == 20
    assert data["final_loss"] == pytest.approx(2.0 * 0.97**20, rel=1e-5)
    assert (tmp_path / "out" / "model.txt").is_file()


# --------------------------------------------------------------------------- quota


def test_quota_counts_running_hours(
    fake: FakeAdapter, make_fake_job: MakeJob, clock: FakeClock
) -> None:
    q = fake.quota()
    assert q.used == 0
    assert q.limit == 30
    assert q.source == "live"
    assert q.resets_at is not None
    job = make_fake_job(duration=7200, pending_s=100)
    fake.submit(job, make_ctx(job))
    clock.advance(100 + 1800)
    assert fake.quota().used == pytest.approx(0.5)
    clock.advance(10_000)
    assert fake.quota().used == pytest.approx(2.0)


def test_quota_limit_cuts_off_runs_and_blocks_submits(
    fake: FakeAdapter, make_fake_job: MakeJob, clock: FakeClock
) -> None:
    a = make_fake_job(duration=4, quota_limit=10)
    ra = fake.submit(a, make_ctx(a))
    clock.advance(4)
    assert fake.status(ra).phase is RemotePhase.SUCCEEDED
    b = make_fake_job(duration=60, quota_limit=10)
    rb = fake.submit(b, make_ctx(b))
    clock.advance(6)
    st = fake.status(rb)
    assert st.phase is RemotePhase.LOST
    assert st.quota_exhausted
    assert st.lost_reason == "quota exhausted"
    q = fake.quota()
    assert q.limit == pytest.approx(10 / 3600)
    assert q.used == pytest.approx(10 / 3600)
    c = make_fake_job(duration=1, quota_limit=10)
    with pytest.raises(QuotaExhausted) as info:
        fake.submit(c, make_ctx(c))
    assert info.value.resets_at == clock.now() + 3600


# --------------------------------------------------------------------------- health


def test_health_override_round_trip(fake: FakeAdapter, make_fake_job: MakeJob) -> None:
    assert fake.healthcheck().ok
    job = make_fake_job(duration=100)
    ref = fake.submit(job, make_ctx(job))

    fake.set_health("auth_required", "kaggle token expired")
    h = fake.healthcheck()
    assert h.health is ProviderHealth.AUTH_REQUIRED
    assert h.reason == "kaggle token expired"
    assert h.hint
    with pytest.raises(AuthRequired):
        fake.submit(job, make_ctx(job, 2))
    with pytest.raises(AuthRequired):
        fake.status(ref)

    fake.set_health("unavailable")
    h = fake.healthcheck()
    assert h.health is ProviderHealth.UNAVAILABLE
    assert h.reason
    with pytest.raises(Unavailable):
        fake.status(ref)

    fake.set_health(None)
    assert fake.healthcheck().ok
    assert fake.status(ref).phase is RemotePhase.RUNNING


def test_set_health_rejects_unknown_value(fake: FakeAdapter) -> None:
    with pytest.raises(ValueError, match="sideways"):
        fake.set_health("sideways")


def test_idempotent_submit_ignores_later_errors(fake: FakeAdapter, make_fake_job: MakeJob) -> None:
    """A retried submit of a run that exists returns it even if the provider is now down."""
    job = make_fake_job(duration=100)
    ctx = make_ctx(job)
    ref = fake.submit(job, ctx)
    fake.set_health("unavailable")
    assert fake.submit(job, ctx) == ref


def test_subsecond_checkpoints_still_finish(
    fake: FakeAdapter, make_fake_job: MakeJob, clock: FakeClock
) -> None:
    job = make_fake_job(duration=0.5, steps=2, checkpoint_every=0.2)
    ref = fake.submit(job, make_ctx(job))
    clock.advance(0.5)
    lines = [line for c in fake.logs(ref) for line in c.lines]
    seqs = [e.seq for line in lines if (e := parse_line(line)) and e.t == "ckpt_end"]
    assert seqs == [1, 2]
