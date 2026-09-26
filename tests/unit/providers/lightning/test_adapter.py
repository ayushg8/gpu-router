"""LightningAdapter over SimLightning: uploads, naming, idempotency, status judging,
logs, fetch, cancel, quota, health, credentials, the wall-clock backstop."""

from __future__ import annotations

import io
import json
import os
import tarfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr

from gpu_router import protocol, secrets
from gpu_router.adapters.base import AttemptContext, RemotePhase, RemoteRef
from gpu_router.clock import FakeClock
from gpu_router.errors import (
    DEFINITIVE_SUBMIT_ERRORS,
    AuthRequired,
    InvalidJob,
    NotFound,
    QuotaExhausted,
    RateLimited,
    Unavailable,
)
from gpu_router.models import Checkpoint, Job, ProviderHealth
from gpu_router.paths import Paths
from gpu_router.providers.lightning import adapter as lmod
from gpu_router.providers.lightning.adapter import (
    BACKSTOP_GRACE_S,
    DRIVE_ROOT,
    EMPTY_LOG_GRACE_S,
    LightningAdapter,
    job_command,
    month_bounds,
    name_for_key,
    normalize,
)
from tests.contract.harness import ContractTarget, make_ctx, make_job
from tests.contract.lightning.sim import SimLightning
from tests.contract.lightning.targets import (
    SIM_API_KEY,
    SIM_USER_ID,
    SimLightningAdapter,
    lightning_deps,
)
from tests.unit.providers.lightning.conftest import NOW

TERMINAL = {p for p in RemotePhase if p.terminal}


def _target(clock: FakeClock, **directives: Any) -> ContractTarget:
    return ContractTarget(
        name="lightning",
        build=lambda: None,  # type: ignore[arg-type,return-value]
        clock=clock,
        options={"lightning": {"duration": 5, "steps": 4, **directives}},
    )


def _job(clock: FakeClock, **directives: Any) -> Job:
    return make_job(_target(clock, **directives))


def _run(
    adapter: SimLightningAdapter, clock: FakeClock, ref: RemoteRef, *, limit: int = 400
) -> Any:
    for _ in range(limit):
        st = adapter.status(ref)
        if st.phase.terminal:
            return st
        clock.advance(0.5)
    raise AssertionError("never finished")


def _uploads(sim: SimLightning, name: str) -> dict[str, bytes]:
    prefix = f"{DRIVE_ROOT}/{name}/"
    return {k[len(prefix) :]: v for k, v in sim.uploads.items() if k.startswith(prefix)}


# ---------------------------------------------------------------------- naming, command


def test_job_names_follow_the_attempt_key() -> None:
    assert name_for_key("gpu-0123456789ab-2") == "gr-0123456789ab-2"
    assert name_for_key("not-a-key") is None
    assert name_for_key("gpu-0123456789ab-0") is None


def test_the_command_runs_launch_py_from_the_mount_or_a_downloaded_copy() -> None:
    cmd = job_command("gr-0123456789ab-1", "me/default", ["launch.py", "bundle.tar.gz"])
    assert "D=/teamspace/uploads/gpu-router/gr-0123456789ab-1" in cmd
    assert 'exec "$PY" -u "$D/launch.py"' in cmd
    assert "me/default uploads/gpu-router/gr-0123456789ab-1" in cmd
    assert "gpu-router-dl/gr-0123456789ab-1" in cmd


def test_month_bounds_are_calendar_months_utc() -> None:
    start, nxt = month_bounds(NOW)
    assert start == 1_788_220_800.0  # 2026-09-01 00:00 UTC
    assert nxt == 1_790_812_800.0  # 2026-10-01 00:00 UTC
    dec = month_bounds(datetime(2026, 12, 31, 23, 59, tzinfo=UTC).timestamp())
    assert dec == (
        datetime(2026, 12, 1, tzinfo=UTC).timestamp(),
        datetime(2027, 1, 1, tzinfo=UTC).timestamp(),
    )


def test_protocol_lines_behind_a_platform_prefix_are_recovered() -> None:
    line = protocol.exit_line(0)
    assert normalize(f"[rank 0] {line}") == line
    assert normalize("plain text") == "plain text"


# ---------------------------------------------------------------------- submit


def test_submit_uploads_the_launcher_bundle_and_parameters(
    adapter: SimLightningAdapter, sim: SimLightning, fclock: FakeClock, tmp_path: Path
) -> None:
    job = _job(fclock)
    archive = tmp_path / "bundle.tar.gz"
    archive.write_bytes(b"bundle bytes")
    ctx = make_ctx(
        job,
        bundle_archive=archive,
        env={"GPU_ROUTER_JOB_ID": job.id, "GPU_STORAGE": "hf://buckets/me/gpu-router"},
        checkpoint_interval_min=15,
    )
    ref = adapter.submit(job, ctx)
    name = f"gr-{job.id}-1"
    assert ref.remote_id == name
    assert ref.meta["teamspace"] == "simuser/default"
    assert ref.meta["gpu"] == "T4"
    assert ref.meta["drive_dir"] == f"{DRIVE_ROOT}/{name}"
    assert int(ref.meta["session_s"]) == 4 * 3600 - lmod.SESSION_MARGIN_S
    assert ref.url
    assert name in ref.url
    files = _uploads(sim, name)
    assert set(files) == {"launch.py", "launch.json", "bundle.tar.gz"}
    assert files["bundle.tar.gz"] == b"bundle bytes"
    assert files["launch.py"] == Path(lmod.launch.__file__).read_bytes()
    cfg = json.loads(files["launch.json"])
    assert cfg["env"]["GPU_STORAGE"] == "hf://buckets/me/gpu-router"
    assert cfg["ckpt_seq_start"] == 1
    assert cfg["checkpoint_interval_min"] == 15
    assert cfg["wall_clock_s"] == int(ref.meta["session_s"])
    assert cfg["teamspace"] == "simuser/default"
    assert cfg["secrets"] is False
    submit = next(p for op, p in sim.calls if op == "submit")
    assert submit["machine"] == "T4"
    assert submit["studio"] == "gpu-router"
    assert submit["interruptible"] is False
    assert submit["env"] == {"GPU_ROUTER_ATTEMPT_KEY": ctx.attempt_key}
    assert f"/teamspace/{DRIVE_ROOT}/{name}" in submit["command"]
    assert not any((adapter.scratch_dir / "staging").iterdir())


def test_secrets_ride_in_a_file_that_is_gone_locally_after_the_call(
    adapter: SimLightningAdapter, sim: SimLightning, fclock: FakeClock
) -> None:
    job = _job(fclock)
    secret = "s3cr3t-value-for-the-job"
    token = "hf_" + "x" * 34
    ctx = make_ctx(
        job,
        secrets={"WANDB_API_KEY": SecretStr(secret), "GPU_STORAGE_TOKEN": SecretStr(token)},
    )
    adapter.submit(job, ctx)
    name = f"gr-{job.id}-1"
    files = _uploads(sim, name)
    assert json.loads(files["secrets.json"])["values"] == {
        "WANDB_API_KEY": secret,
        "GPU_STORAGE_TOKEN": token,
    }
    assert json.loads(files["launch.json"])["secrets"] is True
    submit = next(p for op, p in sim.calls if op == "submit")
    blob = (
        json.dumps({k: v for k, v in submit.items() if k != "files"})
        + files["launch.json"].decode()
    )
    assert secret not in blob
    assert token not in blob
    assert not any((adapter.scratch_dir / "staging").iterdir())


def test_submit_is_idempotent_without_a_second_create_call(
    adapter: SimLightningAdapter, sim: SimLightning, fclock: FakeClock
) -> None:
    job = _job(fclock)
    ctx = make_ctx(job)
    first = adapter.submit(job, ctx)
    second = adapter.submit(job, ctx)
    assert first == second
    assert sim.ops().count("submit") == 1


def test_a_lost_local_record_still_finds_the_existing_job(
    adapter: SimLightningAdapter, sim: SimLightning, fclock: FakeClock, paths: Paths
) -> None:
    job = _job(fclock)
    ctx = make_ctx(job)
    first = adapter.submit(job, ctx)
    for path in (adapter.scratch_dir / "runs").iterdir():
        path.unlink()
    again = adapter.submit(job, ctx)
    assert again.remote_id == first.remote_id
    assert len(sim.jobs) == 1
    assert adapter.lookup_by_key(ctx.attempt_key) is not None


def test_a_submit_in_flight_blocks_a_second_one(
    adapter: SimLightningAdapter, sim: SimLightning, fclock: FakeClock
) -> None:
    job = _job(fclock)
    ctx = make_ctx(job)
    name = f"gr-{job.id}-1"
    intent = adapter._dir("runs") / f"{name}.submitting"
    intent.write_text(json.dumps({"started_at": fclock.now()}))
    with pytest.raises(Unavailable, match="may still be running"):
        adapter.submit(job, ctx)
    with pytest.raises(Unavailable):
        adapter.lookup_by_key(ctx.attempt_key)
    fclock.advance(lmod.T_SUBMIT + lmod.SUBMIT_ORPHAN_GRACE_S + 1)
    assert adapter.lookup_by_key(ctx.attempt_key) is None
    assert adapter.submit(job, ctx).remote_id == name
    assert not intent.exists()


def test_a_submit_that_times_out_is_ambiguous(
    adapter: SimLightningAdapter, sim: SimLightning, fclock: FakeClock
) -> None:
    job = _job(fclock)
    ctx = make_ctx(job)
    adapter.teamspace()  # resolve the account first, so the hang hits the submit call
    sim.hang_next["submit"] = True
    with pytest.raises(Unavailable) as info:
        adapter.submit(job, ctx)
    assert not isinstance(info.value, DEFINITIVE_SUBMIT_ERRORS)
    assert not (adapter.scratch_dir / "runs" / f"gr-{job.id}-1.submitting").exists()
    assert adapter.lookup_by_key(ctx.attempt_key) is None


@pytest.mark.parametrize(
    ("failure", "error"),
    [
        ({"kind": "auth", "error": "401", "stage": "pre", "status": 401}, AuthRequired),
        ({"kind": "verify", "error": "phone", "stage": "pre", "status": 403}, AuthRequired),
        ({"kind": "config", "error": "studio", "stage": "pre"}, AuthRequired),
        ({"kind": "quota", "error": "no credits", "stage": "pre"}, QuotaExhausted),
        ({"kind": "rate", "error": "429", "stage": "run", "status": 429}, RateLimited),
        ({"kind": "invalid", "error": "bad", "stage": "run", "status": 400}, InvalidJob),
        ({"kind": "unavailable", "error": "reset", "stage": "run"}, Unavailable),
        ({"kind": "sdk", "error": "boom", "stage": "run"}, Unavailable),
        ({"kind": "invalid", "error": "500 in disguise", "stage": "run"}, Unavailable),
        ({"kind": "rate", "error": "429 w/o stage"}, Unavailable),
        # review fix: a failure after Job.run was called may have created the job
        ({"kind": "rate", "error": "429 after", "stage": "post", "status": 429}, Unavailable),
        ({"kind": "invalid", "error": "404 after", "stage": "post", "status": 404}, Unavailable),
    ],
)
def test_submit_failures_are_definitive_only_when_nothing_was_created(
    adapter: SimLightningAdapter,
    sim: SimLightning,
    fclock: FakeClock,
    failure: dict[str, Any],
    error: type[Exception],
) -> None:
    job = _job(fclock)
    adapter.teamspace()
    sim.fail_next["submit"] = failure
    with pytest.raises(error) as info:
        adapter.submit(job, make_ctx(job))
    if error is QuotaExhausted:
        assert info.value.resets_at == month_bounds(fclock.now())[1]  # type: ignore[attr-defined]


def test_machines_options_and_limits(
    adapter: SimLightningAdapter, sim: SimLightning, fclock: FakeClock, tmp_path: Path
) -> None:
    job = _job(fclock)
    adapter.submit(job, make_ctx(job, gpu="L4"))
    assert [p for op, p in sim.calls if op == "submit"][-1]["machine"] == "L4"
    job2 = _job(fclock, machine="t4_small", timeout_s=600)
    ref = adapter.submit(job2, make_ctx(job2))
    assert [p for op, p in sim.calls if op == "submit"][-1]["machine"] == "T4_SMALL"
    assert ref.meta["session_s"] == "600"
    job3 = _job(fclock)
    with pytest.raises(InvalidJob, match="no A100"):
        adapter.submit(job3, make_ctx(job3, gpu="A100"))
    bad = _job(fclock)
    bad = bad.model_copy(
        update={"spec": bad.spec.model_copy(update={"provider_options": {"lightning": {"x": 1}}})}
    )
    with pytest.raises(InvalidJob, match="unknown provider_options"):
        LightningAdapter.submit(adapter, bad, make_ctx(bad, bundle_archive=tmp_path / "none"))


def test_a_bundle_over_the_limit_is_refused_before_any_call(
    sim: SimLightning, paths: Paths, fclock: FakeClock, tmp_path: Path
) -> None:
    from tests.contract.lightning.targets import store_sim_credentials

    store_sim_credentials()
    a = SimLightningAdapter(sim, paths, max_bundle_mb=0.001)
    big = tmp_path / "big.tar.gz"
    big.write_bytes(b"x" * 5000)
    job = _job(fclock)
    with pytest.raises(InvalidJob, match="bundle is"):
        a.submit(job, make_ctx(job, bundle_archive=big))
    assert sim.calls == []


def test_resume_from_a_checkpoint_file_on_this_mac(
    adapter: SimLightningAdapter, sim: SimLightning, fclock: FakeClock, tmp_path: Path
) -> None:
    ckpt_file = tmp_path / "ckpt-0003.tar.gz"
    ckpt_file.write_bytes(b"checkpoint")
    job = _job(fclock)
    ckpt = Checkpoint(
        id=f"{job.id}.c3",
        job_id=job.id,
        attempt_id=f"{job.id}.1",
        seq=3,
        uri=ckpt_file.as_uri(),
        created_at=fclock.now(),
        recorded_at=fclock.now(),
    )
    adapter.submit(job, make_ctx(job, 2, resume_from=ckpt))
    files = _uploads(sim, f"gr-{job.id}-2")
    assert files["resume.tar.gz"] == b"checkpoint"
    cfg = json.loads(files["launch.json"])
    assert cfg["ckpt_seq_start"] == 4
    assert cfg["resume_sha256"]
    # hf:// resumes are the runner's job; unreachable ones start fresh with a note
    remote = ckpt.model_copy(update={"uri": "hf://buckets/me/gpu-router/jobs/x/ckpt-0003"})
    adapter.submit(job, make_ctx(job, 3, resume_from=remote, env={"GPU_RESUME_URI": remote.uri}))
    files = _uploads(sim, f"gr-{job.id}-3")
    assert "resume.tar.gz" not in files
    assert json.loads(files["launch.json"])["resume_note"] is None
    gone = ckpt.model_copy(update={"uri": "file:///nowhere/ckpt.tar.gz"})
    adapter.submit(job, make_ctx(job, 4, resume_from=gone))
    note = json.loads(_uploads(sim, f"gr-{job.id}-4")["launch.json"])["resume_note"]
    assert "starting fresh" in note


# ---------------------------------------------------------------------- status


def test_lifecycle_pending_running_succeeded_and_cached(
    adapter: SimLightningAdapter, sim: SimLightning, fclock: FakeClock
) -> None:
    job = _job(fclock, pending_s=3, duration=4)
    ref = adapter.submit(job, make_ctx(job))
    assert adapter.status(ref).phase is RemotePhase.PENDING
    fclock.advance(4)
    st = adapter.status(ref)
    assert st.phase is RemotePhase.RUNNING
    assert st.started_at == pytest.approx(NOW + 3)
    st = _run(adapter, fclock, ref)
    assert (st.phase, st.exit_code) == (RemotePhase.SUCCEEDED, 0)
    assert st.ended_at == pytest.approx(NOW + 7)
    calls = len(sim.calls)
    assert adapter.status(ref).phase is RemotePhase.SUCCEEDED
    assert len(sim.calls) == calls  # the verdict is cached


@pytest.mark.parametrize(
    ("directives", "phase", "code", "reason"),
    [
        ({"exit_code": 3}, RemotePhase.FAILED, 3, None),
        ({"exit_code": 90}, RemotePhase.LOST, 90, "dependency install failed"),
        ({"die_after": 2}, RemotePhase.LOST, None, "the machine was lost"),
        ({"wall": 2}, RemotePhase.LOST, 143, "wall-clock limit"),
    ],
)
def test_terminal_outcomes_are_judged_from_the_log(
    adapter: SimLightningAdapter,
    fclock: FakeClock,
    directives: dict[str, Any],
    phase: RemotePhase,
    code: int | None,
    reason: str | None,
) -> None:
    job = _job(fclock, **directives)
    ref = adapter.submit(job, make_ctx(job))
    st = _run(adapter, fclock, ref)
    assert st.phase is phase
    assert st.exit_code == code
    if reason:
        assert reason in (st.lost_reason or "")


def test_credits_running_out_is_lost_with_quota_exhausted(
    adapter: SimLightningAdapter, fclock: FakeClock
) -> None:
    job = _job(fclock, duration=30, quota_limit=5)
    ref = adapter.submit(job, make_ctx(job))
    st = _run(adapter, fclock, ref)
    assert st.phase is RemotePhase.LOST
    assert st.quota_exhausted
    assert "credits ran out" in (st.lost_reason or "")


def test_a_job_stopped_outside_gpu_router_is_cancelled(
    adapter: SimLightningAdapter, sim: SimLightning, fclock: FakeClock
) -> None:
    job = _job(fclock, duration=60)
    ref = adapter.submit(job, make_ctx(job))
    sim.jobs[ref.remote_id].stop_at = fclock.now()
    sim.jobs[ref.remote_id].stop_done_at = fclock.now() + 1
    fclock.advance(2)
    st = adapter.status(ref)
    assert st.phase is RemotePhase.CANCELLED
    assert st.message == "stopped outside gpu-router"


def test_an_empty_final_log_waits_before_a_verdict(
    adapter: SimLightningAdapter,
    sim: SimLightning,
    fclock: FakeClock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = _job(fclock, duration=1, exit_code=4)
    ref = adapter.submit(job, make_ctx(job))
    fclock.advance(5)
    monkeypatch.setattr(sim, "_log", lambda j: {"lines": []})
    with pytest.raises(Unavailable, match="has not published its log"):
        adapter.status(ref)
    fclock.advance(EMPTY_LOG_GRACE_S + 1)
    st = adapter.status(ref)
    assert st.phase is RemotePhase.LOST  # Failed with no exit line after the grace


def test_final_logs_are_cached_redacted(
    adapter: SimLightningAdapter,
    sim: SimLightning,
    fclock: FakeClock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = _job(fclock, duration=1)
    ref = adapter.submit(job, make_ctx(job))
    fclock.advance(5)
    leaked = "tok-" + "z" * 24
    secrets.register_for_redaction(leaked)
    real = sim._log
    monkeypatch.setattr(sim, "_log", lambda j: {"lines": [*real(j)["lines"], f"key={leaked}"]})
    assert adapter.status(ref).phase is RemotePhase.SUCCEEDED
    cached = (adapter.scratch_dir / "final" / f"{ref.remote_id}.json").read_text()
    assert leaked not in cached


def test_the_backstop_stops_a_job_that_outlived_its_wall_clock(
    adapter: SimLightningAdapter, sim: SimLightning, fclock: FakeClock
) -> None:
    job = _job(fclock, duration=10 * 3600, timeout_s=600)
    ref = adapter.submit(job, make_ctx(job))
    fclock.advance(600 + BACKSTOP_GRACE_S - 10)
    assert adapter.status(ref).phase is RemotePhase.RUNNING
    fclock.advance(20)
    st = adapter.status(ref)
    assert st.phase is RemotePhase.LOST
    assert "wall-clock" in (st.lost_reason or "")
    assert sim.ops()[-1] == "stop"
    assert sim.jobs[ref.remote_id].stop_at is not None
    fclock.advance(5)
    assert adapter.status(ref).phase is RemotePhase.LOST  # Stopped + wall stop = lost


def test_status_of_a_foreign_name_is_not_found_without_a_call(
    adapter: SimLightningAdapter, sim: SimLightning
) -> None:
    with pytest.raises(NotFound):
        adapter.status(RemoteRef(remote_id="someone-elses-job"))
    assert sim.calls == []


# ---------------------------------------------------------------------- logs


def test_running_logs_resume_from_the_cursor(
    adapter: SimLightningAdapter, sim: SimLightning, fclock: FakeClock
) -> None:
    job = _job(fclock, duration=10, steps=8)
    ref = adapter.submit(job, make_ctx(job))
    fclock.advance(3)
    first = list(adapter.logs(ref))
    lines = [x for c in first for x in c.lines]
    assert lines
    assert not first[-1].eof
    cursor = first[-1].cursor
    assert cursor == str(len(lines))
    again = list(adapter.logs(ref, since=cursor))
    assert all(not c.lines for c in again) or again[0].lines[0] not in lines[-1:]
    _run(adapter, fclock, ref)
    rest = [x for c in adapter.logs(ref, since=cursor) for x in c.lines]
    full = [x for c in adapter.logs(ref) for x in c.lines]
    assert full[: len(lines)] == lines
    assert full[len(lines) :] == rest
    assert protocol.exit_line(0) in full


def test_logs_of_a_pending_job_are_an_empty_chunk(
    adapter: SimLightningAdapter, fclock: FakeClock
) -> None:
    job = _job(fclock, pending_s=100)
    ref = adapter.submit(job, make_ctx(job))
    chunks = list(adapter.logs(ref, since="0"))
    assert [(c.lines, c.cursor, c.eof) for c in chunks] == [([], "0", False)]


# ---------------------------------------------------------------------- fetch


def test_fetch_extracts_into_dest_and_keeps_other_files(
    adapter: SimLightningAdapter, fclock: FakeClock, tmp_path: Path
) -> None:
    job = _job(fclock, duration=1)
    ref = adapter.submit(job, make_ctx(job))
    with pytest.raises(NotFound, match="still"):
        adapter.fetch(ref, tmp_path / "early")
    _run(adapter, fclock, ref)
    dest = tmp_path / "out"
    dest.mkdir()
    (dest / "mine.txt").write_text("keep")
    res = adapter.fetch(ref, dest)
    assert (res.files, res.partial) == (2, False)
    assert (dest / "model.txt").read_text() == "trained\n"
    assert (dest / "metrics" / "result.json").exists()
    assert (dest / "mine.txt").read_text() == "keep"
    assert not any((adapter.scratch_dir / "fetch").iterdir())


def test_extract_refuses_paths_outside_dest(adapter: SimLightningAdapter, tmp_path: Path) -> None:
    archive = tmp_path / "evil.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for name in ("../escape.txt", "/abs.txt", "ok/fine.txt"):
            info = tarfile.TarInfo(name)
            info.size = 2
            tar.addfile(info, io.BytesIO(b"hi"))
        link = tarfile.TarInfo("ok/link")
        link.type = tarfile.SYMTYPE
        link.linkname = "/etc/passwd"
        tar.addfile(link)
    dest = tmp_path / "dest"
    files, _ = adapter._extract(archive, tmp_path / "work", dest)
    assert files == 1
    assert (dest / "ok" / "fine.txt").exists()
    assert not (tmp_path / "escape.txt").exists()
    assert not (dest / "ok" / "link").exists()


def test_a_success_without_an_archive_is_a_partial_fetch(
    adapter: SimLightningAdapter,
    sim: SimLightning,
    fclock: FakeClock,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = _job(fclock, duration=1)
    ref = adapter.submit(job, make_ctx(job))
    _run(adapter, fclock, ref)
    monkeypatch.setattr(
        sim, "_op_fetch", lambda p: {"source": None, "archive": None, "notes": ["drive: 404"]}
    )
    res = adapter.fetch(ref, tmp_path / "o")
    assert (res.files, res.partial) == (0, True)
    assert "drive: 404" in (res.message or "")


def test_finished_jobs_get_their_uploaded_secrets_swept(
    adapter: SimLightningAdapter, sim: SimLightning, fclock: FakeClock, tmp_path: Path
) -> None:
    job = _job(fclock, duration=1)
    ref = adapter.submit(job, make_ctx(job, secrets={"A_SECRET": SecretStr("value-123456")}))
    path = f"{DRIVE_ROOT}/{ref.remote_id}/secrets.json"
    assert path in sim.uploads
    _run(adapter, fclock, ref)
    adapter.fetch(ref, tmp_path / "o")
    assert path in sim.removed
    calls = sim.ops().count("cleanup")
    job2 = _job(fclock)
    adapter.submit(job2, make_ctx(job2))
    assert sim.ops().count("cleanup") == calls  # already swept, not again


# ---------------------------------------------------------------------- cancel


def test_cancel_stops_and_reads_as_cancelled(
    adapter: SimLightningAdapter, sim: SimLightning, fclock: FakeClock
) -> None:
    job = _job(fclock, duration=600)
    ref = adapter.submit(job, make_ctx(job))
    adapter.cancel(ref)
    stop = [p for op, p in sim.calls if op == "stop"][-1]
    assert stop["name"] == ref.remote_id
    assert stop["wait_s"] == 20
    assert adapter.status(ref).phase is RemotePhase.RUNNING  # Stopping
    fclock.advance(3)
    assert adapter.status(ref).phase is RemotePhase.CANCELLED
    calls = len(sim.calls)
    adapter.cancel(ref)
    adapter.cancel(RemoteRef(remote_id="nope"))
    assert len(sim.calls) == calls


def test_cancel_of_a_job_lightning_forgot_is_quiet(
    adapter: SimLightningAdapter, sim: SimLightning, fclock: FakeClock
) -> None:
    job = _job(fclock, duration=600)
    ref = adapter.submit(job, make_ctx(job))
    del sim.jobs[ref.remote_id]
    adapter.cancel(ref)
    assert adapter.status(ref).phase is RemotePhase.CANCELLED


# ---------------------------------------------------------------------- quota, health


def test_quota_is_the_live_credit_balance(
    adapter: SimLightningAdapter, sim: SimLightning, fclock: FakeClock
) -> None:
    job = _job(fclock, duration=1800)
    adapter.submit(job, make_ctx(job))
    fclock.advance(1800)
    q = adapter.quota()
    assert q.source == "live"
    assert q.unit == "credits"
    assert q.limit == 15
    assert q.used == pytest.approx(0.5)
    assert q.detail["balance"] == pytest.approx(14.5)
    assert q.detail["rates"]["T4"]["cost"] == 0.68
    assert q.resets_at == month_bounds(fclock.now())[1]
    quota_call = [p for op, p in sim.calls if op == "quota"][-1]
    assert quota_call["since"] == month_bounds(fclock.now())[0]


def test_quota_limit_is_what_is_left_plus_the_months_jobs_not_the_catalog(
    adapter: SimLightningAdapter, sim: SimLightning, fclock: FakeClock
) -> None:
    """The live account had 5.0 credits, not the catalog's 15 [3P] (2026-09-25): flooring
    the limit at 15 showed "10 of 15 used" on an untouched account."""
    sim.balance_limit = 5 * 3600  # 5 credits at the sim's 1 credit per job-hour
    q = adapter.quota()
    assert (q.used, q.limit) == (0.0, 5.0)
    assert q.detail["remaining"] == 5.0
    assert q.detail["catalog_limit"] == 15
    job = _job(fclock, duration=1800)
    adapter.submit(job, make_ctx(job))
    fclock.advance(1800)
    q = adapter.quota()
    assert q.used == pytest.approx(0.5)
    assert q.limit == pytest.approx(5.0)
    assert q.detail["balance"] == pytest.approx(4.5)
    assert "left" in q.detail["note"]


def test_quota_without_a_balance_api_reports_job_costs(
    adapter: SimLightningAdapter, sim: SimLightning, fclock: FakeClock
) -> None:
    sim.balance_api = False
    job = _job(fclock, duration=3600)
    adapter.submit(job, make_ctx(job))
    fclock.advance(3600)
    q = adapter.quota()
    assert q.source == "live"
    assert q.used == pytest.approx(1.0)
    assert q.detail["basis"] == "job costs"
    assert "Studio time" in q.detail["note"]


def test_healthcheck_ok_names_the_account(adapter: SimLightningAdapter) -> None:
    h = adapter.healthcheck()
    assert h.ok
    assert h.detail["user"] == "simuser"
    assert h.detail["teamspace"] == "simuser/default"
    assert h.detail["credentials"] == "keychain"


def test_not_logged_in_is_explained_without_touching_the_sdk(
    sim: SimLightning, paths: Paths, fclock: FakeClock
) -> None:
    a = SimLightningAdapter(sim, paths)
    h = a.healthcheck()
    assert h.health is ProviderHealth.AUTH_REQUIRED
    assert "gpu login lightning" in (h.hint or "")
    job = _job(fclock)
    with pytest.raises(AuthRequired) as info:
        a.submit(job, make_ctx(job))
    assert isinstance(info.value, DEFINITIVE_SUBMIT_ERRORS)
    assert sim.calls == []


def test_the_login_source_setting_survives_config_yaml(
    sim: SimLightning, paths: Paths, fclock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """config.yaml refuses `providers.lightning.credentials` (it looks like a secret), so the
    mode is read from `login_source`; env mode ignores the Keychain."""
    from gpu_router.config import load_config
    from tests.contract.lightning.targets import SIM_API_KEY, SIM_USER_ID, store_sim_credentials

    paths.home.mkdir(parents=True, exist_ok=True)
    paths.config.write_text("providers:\n  lightning:\n    login_source: env\n")
    settings = load_config(paths, {}).providers["lightning"]
    extra = dict(settings.model_extra or {})
    assert extra["login_source"] == "env"
    store_sim_credentials()
    a = SimLightningAdapter(sim, paths, **extra)
    monkeypatch.delenv("LIGHTNING_USER_ID", raising=False)
    monkeypatch.delenv("LIGHTNING_API_KEY", raising=False)
    assert a.healthcheck().health is ProviderHealth.AUTH_REQUIRED
    monkeypatch.setenv("LIGHTNING_USER_ID", SIM_USER_ID)
    monkeypatch.setenv("LIGHTNING_API_KEY", SIM_API_KEY)
    fclock.advance(lmod.CRED_TTL_S + 1)
    h = a.healthcheck()
    assert h.ok
    assert h.detail["credentials"] == "env"


def test_several_teamspaces_need_a_choice(
    sim: SimLightning, paths: Paths, fclock: FakeClock
) -> None:
    from tests.contract.lightning.targets import store_sim_credentials

    store_sim_credentials()
    sim.teamspaces = ["simuser/default", "acme/research"]
    a = SimLightningAdapter(sim, paths)
    h = a.healthcheck()
    assert h.health is ProviderHealth.AUTH_REQUIRED
    assert "acme/research" in (h.hint or "")
    job = _job(fclock)
    with pytest.raises(AuthRequired, match="teamspace"):
        a.submit(job, make_ctx(job))
    chosen = SimLightningAdapter(sim, paths, teamspace="acme/research")
    ref = chosen.submit(job, make_ctx(job))
    assert ref.meta["teamspace"] == "acme/research"


def test_the_sdk_child_gets_credentials_only_through_a_scrubbed_environment(
    adapter: SimLightningAdapter,
    sim: SimLightning,
    fclock: FakeClock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LIGHTNING_TEAMSPACE", "someone/else")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-never-reach-the-sdk-000000")
    job = _job(fclock)
    adapter.submit(job, make_ctx(job))
    env = sim.envs[-1]
    assert env["LIGHTNING_USER_ID"] == SIM_USER_ID
    assert env["LIGHTNING_API_KEY"] == SIM_API_KEY
    assert "LIGHTNING_TEAMSPACE" not in env
    assert "OPENAI_API_KEY" not in env
    home = Path(env["GR_LIGHTNING_HOME"])
    assert home == adapter.scratch_dir / "sdk-home"
    assert Path(env["LIGHTNING_CREDENTIAL_PATH"]).is_relative_to(home)
    assert env["LIGHTNING_DISABLE_VERSION_CHECK"] == "1"
    assert oct(home.stat().st_mode & 0o777) == "0o700"


def test_test_mode_keeps_the_real_sdk_offline(paths: Paths, fclock: FakeClock) -> None:
    from tests.contract.lightning.targets import store_sim_credentials

    store_sim_credentials()
    a = LightningAdapter(lightning_deps(paths, fclock, test_mode=True), interpreter=["x"])
    assert a.healthcheck().health is ProviderHealth.DISABLED
    job = make_job(_target(fclock), directives={})
    with pytest.raises(AuthRequired, match="test mode"):
        a.submit(job, make_ctx(job, bundle_archive=Path(__file__)))
    assert a.lookup_by_key("gpu-0123456789ab-1") is None


def test_capabilities_follow_the_catalog(adapter: SimLightningAdapter) -> None:
    caps = adapter.capabilities
    assert caps.lookup_by_key
    assert caps.resume
    assert caps.live_quota
    assert caps.cancel_confirms
    assert caps.max_session_hours == 4
    assert caps.max_concurrency == 1
    assert caps.poll_interval_s == 60


def test_registry_builds_the_lightning_adapter() -> None:
    from gpu_router.adapters.registry import adapter_class

    assert adapter_class("lightning") is LightningAdapter


def test_ctx_env_never_holds_secrets_by_construction() -> None:
    # the engine passes secrets separately; the adapter must not merge them into env
    ctx = AttemptContext(attempt_id="a.1", attempt_key="gpu-0123456789ab-1", n=1)
    assert dict(ctx.env) == {}
    assert os.environ.get("LIGHTNING_API_KEY") is None


# ---------------------------------------------------------------------- review fixes (D54)


def test_the_backstop_reports_lost_only_after_the_stop_took_effect(
    adapter: SimLightningAdapter, sim: SimLightning, fclock: FakeClock
) -> None:
    """Review fix: a stop that timed out used to be swallowed and the job reported lost
    ('stopped by gpu-router') while it kept burning credits untracked."""
    job = _job(fclock, duration=10 * 3600, timeout_s=600)
    ref = adapter.submit(job, make_ctx(job))
    fclock.advance(600 + BACKSTOP_GRACE_S + 10)
    sim.hang_next["stop"] = True
    st = adapter.status(ref)
    assert st.phase is RemotePhase.RUNNING
    assert "stopping it" in st.message
    rec = json.loads((adapter.scratch_dir / "runs" / f"{ref.remote_id}.json").read_text())
    assert rec["stop_pending"] is True
    # healthcheck retries the pending stop even if nobody polls the job any more
    adapter.healthcheck()
    assert sim.jobs[ref.remote_id].stop_at is not None
    rec = json.loads((adapter.scratch_dir / "runs" / f"{ref.remote_id}.json").read_text())
    assert rec["stop_pending"] is False
    assert adapter.status(ref).message == "lightning is stopping the job"
    fclock.advance(3)
    st = adapter.status(ref)
    assert st.phase is RemotePhase.LOST
    assert "wall-clock" in (st.lost_reason or "")


def test_a_stop_the_job_ignores_keeps_the_backstop_retrying(
    adapter: SimLightningAdapter, sim: SimLightning, fclock: FakeClock
) -> None:
    job = _job(fclock, duration=10 * 3600, timeout_s=600, ignore_cancel=True)
    ref = adapter.submit(job, make_ctx(job))
    fclock.advance(600 + BACKSTOP_GRACE_S + 10)
    assert adapter.status(ref).phase is RemotePhase.RUNNING  # stop sent, job still Running
    stops = sim.ops().count("stop")
    adapter.quota()  # the pending stop is retried from quota() too
    assert sim.ops().count("stop") == stops + 1
    assert adapter.status(ref).phase is RemotePhase.RUNNING
    assert sim.ops().count("stop") == stops + 2


def test_a_slow_final_log_read_is_never_judged_as_an_empty_log(
    adapter: SimLightningAdapter,
    sim: SimLightning,
    fclock: FakeClock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Review fix: a whole-log read that timed out came back as no lines; after the
    empty-log grace a failed script (exit 1) was cached as lost and re-run elsewhere."""
    job = _job(fclock, duration=1, exit_code=1)
    ref = adapter.submit(job, make_ctx(job))
    fclock.advance(5)
    real = sim._log
    monkeypatch.setattr(sim, "_log", lambda j: {"lines": [], "timeout": True})
    with pytest.raises(Unavailable, match="timed out"):
        adapter.status(ref)
    fclock.advance(EMPTY_LOG_GRACE_S + 1)
    with pytest.raises(Unavailable, match="timed out"):
        adapter.status(ref)  # still no verdict: a timeout never starts the empty-log clock
    assert not (adapter.scratch_dir / "final" / f"{ref.remote_id}.empty-log").exists()
    status_call = [p for op, p in sim.calls if op == "status"][-1]
    assert status_call["log_tail"] == lmod.VERDICT_TAIL
    # the driver's tail fallback: judged from the tail, cached as partial
    tail = real(sim.jobs[ref.remote_id])["lines"][-3:]
    monkeypatch.setattr(sim, "_log", lambda j: {"lines": tail, "timeout": True, "tail": True})
    st = adapter.status(ref)
    assert (st.phase, st.exit_code) == (RemotePhase.FAILED, 1)
    final = json.loads((adapter.scratch_dir / "final" / f"{ref.remote_id}.json").read_text())
    assert final["lines_partial"] is True
    # logs() serves nothing from a tail (its line numbers are not the log's) ...
    monkeypatch.setattr(sim, "_log", lambda j: {"lines": [], "timeout": True})
    chunks = list(adapter.logs(ref, since="7"))
    assert [(c.lines, c.cursor, c.eof) for c in chunks] == [([], "7", False)]
    # ... and upgrades the cache once a whole read works
    monkeypatch.setattr(sim, "_log", real)
    chunks = list(adapter.logs(ref))
    lines = [line for c in chunks for line in c.lines]
    assert lines == real(sim.jobs[ref.remote_id])["lines"]
    assert chunks[-1].eof
    final = json.loads((adapter.scratch_dir / "final" / f"{ref.remote_id}.json").read_text())
    assert "lines_partial" not in final


def test_cancel_raises_when_lightning_did_not_confirm_the_stop(
    adapter: SimLightningAdapter, sim: SimLightning, fclock: FakeClock
) -> None:
    """Review fix: cancel() dropped op_stop's answer, so a stop that never landed let the
    engine mark the attempt cancelled and start the job a second time elsewhere."""
    job = _job(fclock, duration=600, ignore_cancel=True)
    ref = adapter.submit(job, make_ctx(job))
    with pytest.raises(Unavailable, match="not confirmed the stop"):
        adapter.cancel(ref)
    sim.fail_next["stop"] = {"kind": "unavailable", "error": "reset"}
    with pytest.raises(Unavailable):
        adapter.cancel(ref)
    job2 = _job(fclock, duration=600)
    ref2 = adapter.submit(job2, make_ctx(job2))
    adapter.cancel(ref2)  # Stopping = the stop took: returns normally


def test_stale_staging_dirs_with_secrets_are_swept(
    sim: SimLightning, paths: Paths, fclock: FakeClock
) -> None:
    """Review fix: a daemon killed mid-submit left plaintext secrets.json in staging/."""
    from tests.contract.lightning.targets import store_sim_credentials

    store_sim_credentials()
    first = SimLightningAdapter(sim, paths)
    staging = first.scratch_dir / "staging"
    stale = staging / "gr-0123456789ab-1-abcd"
    live = staging / "gr-0123456789ab-2-efgh"
    for d in (stale, live):
        d.mkdir(parents=True)
        (d / "secrets.json").write_text('{"v": 1}')
    (first.scratch_dir / "runs").mkdir(parents=True, exist_ok=True)
    (first.scratch_dir / "runs" / "gr-0123456789ab-2.submitting").write_text(
        json.dumps({"started_at": fclock.now()})
    )
    first.close()
    second = SimLightningAdapter(sim, paths)  # adapter init sweeps
    assert not stale.exists()
    assert live.exists()  # an orphaned driver may still upload from it
    fclock.advance(lmod.T_SUBMIT + lmod.SUBMIT_ORPHAN_GRACE_S + 1)
    job = _job(fclock)
    second.submit(job, make_ctx(job))  # each submit sweeps too
    assert not live.exists()
    assert list(staging.iterdir()) == []
    second.close()


def test_an_upload_that_cannot_fit_the_submit_budget_is_refused(
    sim: SimLightning, paths: Paths, fclock: FakeClock, tmp_path: Path
) -> None:
    """Review fix: T_SUBMIT is fixed, so a bundle + resume the old caps accepted timed
    out on every placement. Now it is InvalidJob (the engine places the job elsewhere)."""
    from tests.contract.lightning.targets import store_sim_credentials

    store_sim_credentials()
    a = SimLightningAdapter(sim, paths, upload_mbps=0.01)  # floored at 1 MB per submit
    assert a.capabilities.max_bundle_mb == pytest.approx(a._max_upload_mb)
    job = _job(fclock)
    ckpt_file = tmp_path / "ckpt-0001.tar.gz"
    ckpt_file.write_bytes(b"c" * 2_000_000)
    ckpt = Checkpoint(
        id=f"{job.id}.c1",
        job_id=job.id,
        attempt_id=f"{job.id}.1",
        seq=1,
        uri=ckpt_file.as_uri(),
        created_at=fclock.now(),
        recorded_at=fclock.now(),
    )
    with pytest.raises(InvalidJob, match="bundle \\+ checkpoint") as info:
        a.submit(job, make_ctx(job, 2, resume_from=ckpt))
    assert "upload_mbps" in (info.value.hint or "")
    assert [op for op in sim.ops() if op == "submit"] == []
    a.close()
    fast = SimLightningAdapter(sim, paths, upload_mbps=100)
    fast.submit(job, make_ctx(job, 2, resume_from=ckpt))
    assert _uploads(sim, f"gr-{job.id}-2")["resume.tar.gz"] == b"c" * 2_000_000
    fast.close()


@pytest.mark.parametrize("machine", ["L40S", "T4_X_4", "L4_X_2", "A100"])
def test_an_explicit_machine_must_be_one_the_router_can_price(
    adapter: SimLightningAdapter, fclock: FakeClock, machine: str
) -> None:
    """Review fix: machines were labelled by prefix (L40S -> L4, T4_X_4 -> T4) and billed
    at the router's T4/L4 rate, past the approval policy."""
    job = _job(fclock, machine=machine)
    with pytest.raises(InvalidJob, match="not a machine gpu-router can price"):
        adapter.submit(job, make_ctx(job))


def test_an_explicit_machine_must_match_the_placed_gpu(
    adapter: SimLightningAdapter, sim: SimLightning, fclock: FakeClock
) -> None:
    job = _job(fclock, machine="L4")
    with pytest.raises(InvalidJob, match="placed on a T4") as info:
        adapter.submit(job, make_ctx(job, gpu="T4"))
    assert "gpu: L4" in (info.value.hint or "")
    ref = adapter.submit(job, make_ctx(job, gpu="L4"))
    assert ref.meta["gpu"] == "L4"
    assert [p for op, p in sim.calls if op == "submit"][-1]["machine"] == "L4"


def test_a_403_for_l4_is_the_plan_refusing_the_gpu_not_a_login_problem(
    adapter: SimLightningAdapter, sim: SimLightning, fclock: FakeClock
) -> None:
    """D56, live 2026-09-25 (job 1253): the free account's L4 create got
    `jobs_service_create_job_with_http_info ... response: 403` while T4 creates worked. It
    was AuthRequired ("needs login"), which cooled Lightning down for every job and made a
    pinned L4 job retry forever; now it is an InvalidJob that names the GPU."""
    msg = (
        "lightning refused access: The jobs_service_create_job_with_http_info request "
        "failed to reach the server, response: 403."
    )
    job = _job(fclock)
    adapter.teamspace()
    sim.fail_next["submit"] = {"kind": "auth", "error": msg, "stage": "run", "status": 403}
    with pytest.raises(InvalidJob) as info:
        adapter.submit(job, make_ctx(job, gpu="L4"))
    assert isinstance(info.value, DEFINITIVE_SUBMIT_ERRORS)
    assert "refused to create a L4 job (403 Forbidden)" in info.value.message
    assert "free tier runs T4 only" in info.value.message
    assert info.value.hint is not None
    assert "--gpu T4" in info.value.hint
    # the same 403 for the free T4 is still an access problem, and a 403 after the create
    # call (the job may exist) is still ambiguous
    sim.fail_next["submit"] = {"kind": "auth", "error": msg, "stage": "run", "status": 403}
    job2 = _job(fclock)
    with pytest.raises(AuthRequired):
        adapter.submit(job2, make_ctx(job2, gpu="T4"))
    sim.fail_next["submit"] = {"kind": "auth", "error": msg, "stage": "post", "status": 403}
    job3 = _job(fclock)
    with pytest.raises(Unavailable):
        adapter.submit(job3, make_ctx(job3, gpu="L4"))


def test_the_packaged_catalog_offers_lightning_t4_only() -> None:
    """D56: the free tier refuses L4, so the catalog stops offering a free 24GB L4; the L4
    rate stays for a user catalog that re-adds it."""
    from gpu_router.providers.catalog import load_catalog

    entry = load_catalog().get("lightning")
    assert [g.name for g in entry.gpus] == ["T4"]
    assert entry.max_vram_gb == 16
    assert entry.options["quota_per_gpu_hour_by_gpu"] == {"L4": 1.68}
