"""ColabAdapter end to end against the simulated colab CLI (fake_colab.py).

The runner is the real `gpu_runner/bootstrap.py` from a real bundle; only the CLI and the
VM are simulated. Covers the submit pattern (new -> prepare -> upload -> detached launch),
harvest-then-stop teardown, cancel, crash recovery, failure mapping and invariant 20.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr

from gpu_router.adapters.base import AttemptContext, RemotePhase, RemoteRef, RemoteStatus
from gpu_router.clock import FakeClock, SystemClock
from gpu_router.config import ProviderSettings
from gpu_router.errors import (
    DEFINITIVE_SUBMIT_ERRORS,
    AuthRequired,
    InvalidJob,
    NotFound,
    QuotaExhausted,
    Unavailable,
)
from gpu_router.models import Checkpoint, Job, ProviderHealth
from gpu_router.paths import Paths
from gpu_router.protocol import parse_line
from gpu_router.providers.colab import adapter as colab_mod
from gpu_router.providers.colab.adapter import ColabAdapter, session_name, short_gpu
from gpu_router.providers.colab.cli import ColabCli
from gpu_router.providers.colab.state import RunRecord, RunState
from tests.contract.harness import ContractTarget, make_ctx, make_job
from tests.unit.providers.colab.helpers import (
    ColabSim,
    all_lines,
    make_adapter,
    make_bundle,
    secret_sha,
    with_bundle,
)


def _target(adapter: ColabAdapter) -> ContractTarget:
    return ContractTarget(name="colab", build=lambda: adapter, clock=SystemClock())


def wait_terminal(adapter: ColabAdapter, ref: RemoteRef, timeout: float = 60) -> RemoteStatus:
    deadline = time.monotonic() + timeout
    while True:
        st = adapter.status(ref)
        if st.phase.terminal:
            return st
        assert time.monotonic() < deadline, f"still {st.phase} after {timeout}s"
        time.sleep(0.25)


def wait_running(adapter: ColabAdapter, ref: RemoteRef, timeout: float = 30) -> None:
    deadline = time.monotonic() + timeout
    while adapter.status(ref).phase is not RemotePhase.RUNNING:
        assert time.monotonic() < deadline
        time.sleep(0.2)


def submit(
    adapter: ColabAdapter,
    paths: Paths,
    tmp_path: Path,
    *,
    n: int = 1,
    job: Job | None = None,
    ctx_fields: dict[str, Any] | None = None,
    **cfg: Any,
) -> tuple[Job, AttemptContext, RemoteRef]:
    cfg.setdefault("seconds", 0.5)
    archive = make_bundle(paths, tmp_path / "projects", **cfg)
    job = job or make_job(_target(adapter))
    ctx = with_bundle(make_ctx(job, n), archive, **(ctx_fields or {}))
    return job, ctx, adapter.submit(job, ctx)


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


# --------------------------------------------------------------------------- pure helpers


def test_session_name_is_derived_from_the_attempt_key() -> None:
    assert session_name("gpu-0123456789ab-2") == "gr-0123456789ab-2"
    odd = session_name("weird key/with spaces")
    assert odd.startswith("gr-")
    assert odd == session_name("weird key/with spaces")


@pytest.mark.parametrize(
    ("raw", "short"),
    [
        ("Tesla T4, 15360 MiB", "T4"),
        ("NVIDIA A100-SXM4-40GB, 40960 MiB", "A100"),
        ("NVIDIA L4", "L4"),
        ("NVIDIA L40S", "NVIDIA L40S"),
        (None, None),
    ],
)
def test_short_gpu(raw: str | None, short: str | None) -> None:
    assert short_gpu(raw) == short


# --------------------------------------------------------------------------- lifecycle


def test_every_call_uses_adc_and_the_private_session_file(
    colab: ColabAdapter, sim: ColabSim, paths: Paths, tmp_path: Path
) -> None:
    _, _, ref = submit(colab, paths, tmp_path)
    wait_terminal(colab, ref)
    calls = sim.calls()
    assert calls
    for argv in calls:
        assert "--auth=adc" in argv
        assert argv[argv.index("--config") + 1] == str(colab.config_file)
    assert paths.home in colab.config_file.parents
    assert Path.home() / ".config" / "colab-cli" not in colab.config_file.parents


def test_success_harvests_log_and_outputs_then_stops_the_session(
    colab: ColabAdapter, sim: ColabSim, paths: Paths, tmp_path: Path
) -> None:
    _, _, ref = submit(colab, paths, tmp_path, extra_outputs=True)
    assert ref.remote_id in sim.sessions()
    st = wait_terminal(colab, ref)
    assert st.phase is RemotePhase.SUCCEEDED
    assert st.exit_code == 0
    assert st.gpu == "T4"
    assert st.started_at is not None
    assert st.ended_at is not None
    rec = colab.store.load(ref.remote_id)
    assert rec is not None
    assert rec.stopped
    assert rec.log_cached
    assert rec.outputs_cached
    assert ref.remote_id not in sim.sessions(), "the session must be stopped after the run"
    assert sim.commands()[-1] == "stop"

    # after the VM is gone, logs and outputs still come from the harvested copies
    chunks = list(colab.logs(ref))
    lines = all_lines(chunks)
    assert "demo done" in lines
    assert chunks[-1].eof
    kinds = {e.t for line in lines if (e := parse_line(line)) is not None}
    assert {"hello", "total", "metric", "exit"} <= kinds
    dest = tmp_path / "out"
    first = colab.fetch(ref, dest)
    assert first.files == 2
    assert json.loads((dest / "result.json").read_text()) == {"steps": 5}
    assert (dest / "sub" / "b.txt").read_text() == "b"
    (dest / "mine.txt").write_text("keep")
    assert colab.fetch(ref, dest).files == 2
    assert (dest / "mine.txt").read_text() == "keep"


def test_submit_is_idempotent_per_attempt_key(
    colab: ColabAdapter, sim: ColabSim, paths: Paths, tmp_path: Path
) -> None:
    job, ctx, ref = submit(colab, paths, tmp_path, seconds=3)
    assert ref.remote_id == session_name(ctx.attempt_key) == f"gr-{job.id}-1"
    again = colab.submit(job, ctx)
    assert again.remote_id == ref.remote_id
    assert sim.commands().count("new") == 1
    colab.cancel(ref)


def test_secrets_travel_as_a_file_that_is_deleted(
    colab: ColabAdapter, sim: ColabSim, paths: Paths, tmp_path: Path
) -> None:
    value = "hf_super_secret_value_123"
    _, _, ref = submit(
        colab,
        paths,
        tmp_path,
        secret_name="HF_TOKEN",
        secret_sha256=secret_sha(value),
        ctx_fields={"secrets": {"HF_TOKEN": SecretStr(value)}},
    )
    wait_terminal(colab, ref)
    assert "secret_ok=True" in all_lines(colab.logs(ref))
    assert not (sim.run_dir(ref.remote_id) / ".secrets.json").exists()
    assert all(value not in " ".join(argv) for argv in sim.calls())
    assert not any((colab.scratch_dir / "tmp").iterdir()), "local temp files must be gone"
    record_text = (colab.store.run_dir(ref.remote_id) / "record.json").read_text()
    assert value not in record_text
    assert value not in json.dumps(ref.meta)


def test_the_storage_token_never_sits_in_a_long_lived_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review finding (D44): the launcher put GPU_STORAGE_TOKEN in the environment of the
    `bash -c` wrapper that lives all run (readable in /proc/<pid>/environ); it now goes to
    bootstrap in a 0600 file (bootstrap deletes it after reading). Other secrets are the
    job's own and still reach its environment."""
    import io
    import subprocess
    import tarfile

    from gpu_router.providers.colab import remote

    run = tmp_path / "run"
    run.mkdir()
    with tarfile.open(run / "bundle.tar.gz", "w:gz") as tf:
        data = b"print('boot')\n"
        info = tarfile.TarInfo("gpu_runner/bootstrap.py")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
    token = "hf_" + "c" * 34
    (run / ".secrets.json").write_text(json.dumps({"GPU_STORAGE_TOKEN": token, "WANDB": "w"}))
    seen: dict[str, Any] = {}

    class FakePopen:
        pid = 4242

        def __init__(self, argv: Any, **kw: Any) -> None:
            seen["env"] = kw["env"]

    monkeypatch.setattr(subprocess, "Popen", FakePopen)
    ns: dict[str, Any] = {}
    exec(remote.LAUNCH, ns)
    ns["_gr_main"]({"run_dir": str(run), "env": {}})
    env = seen["env"]
    assert "GPU_STORAGE_TOKEN" not in env
    assert all(token not in str(v) for v in env.values())
    assert env["WANDB"] == "w"
    path = Path(env["GPU_STORAGE_TOKEN_FILE"])
    assert path.read_text() == token
    assert path.stat().st_mode & 0o777 == 0o600
    assert not (run / ".secrets.json").exists()


def test_failed_script_is_failed_with_its_exit_code_and_has_no_outputs(
    colab: ColabAdapter, sim: ColabSim, paths: Paths, tmp_path: Path
) -> None:
    _, _, ref = submit(colab, paths, tmp_path, exit=3)
    st = wait_terminal(colab, ref)
    assert st.phase is RemotePhase.FAILED
    assert st.exit_code == 3
    assert ref.remote_id not in sim.sessions()
    assert "demo done" in all_lines(colab.logs(ref))
    with pytest.raises(NotFound):
        colab.fetch(ref, tmp_path / "out")


@pytest.mark.parametrize(("code", "why"), [(90, "dependency install"), (137, "out of RAM")])
def test_environment_failures_are_lost_so_the_engine_reroutes(
    colab: ColabAdapter, paths: Paths, tmp_path: Path, code: int, why: str
) -> None:
    _, _, ref = submit(colab, paths, tmp_path, exit=code)
    st = wait_terminal(colab, ref)
    assert st.phase is RemotePhase.LOST
    assert st.lost_reason is not None
    assert why in st.lost_reason


def test_reclaimed_session_is_lost(
    colab: ColabAdapter, sim: ColabSim, paths: Paths, tmp_path: Path
) -> None:
    _, _, ref = submit(colab, paths, tmp_path, seconds=30)
    wait_running(colab, ref)
    sim.control(lose=[ref.remote_id])
    st = colab.status(ref)
    assert st.phase is RemotePhase.LOST
    assert st.lost_reason is not None
    assert "reclaimed" in st.lost_reason
    assert not st.quota_exhausted
    rec = colab.store.load(ref.remote_id)
    assert rec is not None
    assert rec.stopped
    assert colab.status(ref).phase is RemotePhase.LOST  # stable, no more CLI calls
    assert list(colab.logs(ref))[-1].eof


def test_cancel_stops_the_session_and_the_runner(
    colab: ColabAdapter, sim: ColabSim, paths: Paths, tmp_path: Path
) -> None:
    _, _, ref = submit(colab, paths, tmp_path, seconds=60)
    wait_running(colab, ref)
    pid = int(json.loads((sim.run_dir(ref.remote_id) / "launched.json").read_text())["pid"])
    colab.cancel(ref)
    assert colab.status(ref).phase is RemotePhase.CANCELLED
    assert ref.remote_id not in sim.sessions()
    deadline = time.monotonic() + 10
    while pid_alive(pid):
        assert time.monotonic() < deadline, "the runner outlived its session"
        time.sleep(0.1)
    colab.cancel(ref)  # idempotent
    colab.cancel(RemoteRef(remote_id="gr-unknown-1"))  # unknown: no-op (A5)
    assert colab.status(ref).phase is RemotePhase.CANCELLED


def test_cancel_after_success_keeps_the_outputs(
    colab: ColabAdapter, paths: Paths, tmp_path: Path
) -> None:
    _, _, ref = submit(colab, paths, tmp_path)
    assert wait_terminal(colab, ref).phase is RemotePhase.SUCCEEDED
    colab.cancel(ref)
    assert colab.status(ref).phase is RemotePhase.SUCCEEDED
    assert colab.fetch(ref, tmp_path / "out").files == 1


# --------------------------------------------------------------------------- submit failures


def test_quota_refusal_is_definitive_and_says_when(
    colab: ColabAdapter, sim: ColabSim, paths: Paths, tmp_path: Path
) -> None:
    sim.control(new="quota")
    job = make_job(_target(colab))
    ctx = with_bundle(make_ctx(job), make_bundle(paths, tmp_path))
    with pytest.raises(QuotaExhausted) as info:
        colab.submit(job, ctx)
    assert isinstance(info.value, DEFINITIVE_SUBMIT_ERRORS)
    assert info.value.resets_at is not None
    assert info.value.resets_at > SystemClock().now()
    assert info.value.provider == "colab"
    assert colab.lookup_by_key(ctx.attempt_key) is None
    q = colab.quota()
    assert q.resets_at == pytest.approx(info.value.resets_at, abs=5)
    assert "refused_at" in q.detail
    assert "stop" not in sim.commands()  # nothing was created, nothing to stop


def test_capacity_refusal_is_ambiguous_but_resolves_to_no_run(
    colab: ColabAdapter, sim: ColabSim, paths: Paths, tmp_path: Path
) -> None:
    sim.control(new="busy")
    job = make_job(_target(colab))
    ctx = with_bundle(make_ctx(job), make_bundle(paths, tmp_path))
    with pytest.raises(Unavailable) as info:
        colab.submit(job, ctx)
    assert not isinstance(info.value, DEFINITIVE_SUBMIT_ERRORS)
    assert "too many active sessions" in info.value.message
    assert colab.lookup_by_key(ctx.attempt_key) is None


@pytest.mark.parametrize("mode", ["scope", "auth"])
def test_credential_problems_are_auth_required(
    colab: ColabAdapter, sim: ColabSim, paths: Paths, tmp_path: Path, mode: str
) -> None:
    sim.control(new=mode)
    job = make_job(_target(colab))
    ctx = with_bundle(make_ctx(job), make_bundle(paths, tmp_path))
    with pytest.raises(AuthRequired) as info:
        colab.submit(job, ctx)
    assert info.value.hint is not None
    assert "colaboratory" in info.value.hint
    assert colab.lookup_by_key(ctx.attempt_key) is None


@pytest.mark.parametrize("mode", ["hang", "crash"])
def test_unclear_new_failure_stops_the_name_and_is_ambiguous(
    colab: ColabAdapter,
    sim: ColabSim,
    paths: Paths,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    monkeypatch.setattr(colab_mod, "NEW_TIMEOUT_S", 2.0)
    sim.control(new=mode)
    job = make_job(_target(colab))
    ctx = with_bundle(make_ctx(job), make_bundle(paths, tmp_path))
    with pytest.raises(Unavailable):
        colab.submit(job, ctx)
    assert "stop" in sim.commands()
    assert session_name(ctx.attempt_key) not in sim.sessions()
    assert colab.lookup_by_key(ctx.attempt_key) is None
    rec = colab.store.load(session_name(ctx.attempt_key))
    assert rec is not None
    assert rec.state is RunState.ABANDONED
    assert rec.stopped


def test_setup_failure_after_new_stops_the_session(
    colab: ColabAdapter, sim: ColabSim, paths: Paths, tmp_path: Path
) -> None:
    sim.control(exec_fail=5)
    job = make_job(_target(colab))
    ctx = with_bundle(make_ctx(job), make_bundle(paths, tmp_path))
    with pytest.raises(Unavailable) as info:
        colab.submit(job, ctx)
    assert "it was stopped" in info.value.message
    assert session_name(ctx.attempt_key) not in sim.sessions()
    assert colab.lookup_by_key(ctx.attempt_key) is None


def test_gpu_is_validated_before_any_cli_call(
    colab: ColabAdapter, sim: ColabSim, paths: Paths, tmp_path: Path
) -> None:
    archive = make_bundle(paths, tmp_path)
    job = make_job(_target(colab), gpu="A100")
    with pytest.raises(InvalidJob, match="A100"):
        colab.submit(job, with_bundle(make_ctx(job), archive))
    job = make_job(_target(colab))
    with pytest.raises(InvalidJob):  # a typo would silently become an A100 in the CLI
        colab.submit(job, with_bundle(make_ctx(job), archive, gpu="T5"))
    assert sim.calls() == []


def test_missing_or_oversized_bundle_is_invalid(
    paths: Paths, sim: ColabSim, tmp_path: Path
) -> None:
    colab = make_adapter(paths, SystemClock(), sim)
    job = make_job(_target(colab))
    with pytest.raises(InvalidJob, match="bundle"):
        colab.submit(job, make_ctx(job))
    small = make_adapter(paths, SystemClock(), sim, max_bundle_mb=0.0001)
    with pytest.raises(InvalidJob, match="MB"):
        small.submit(job, with_bundle(make_ctx(job), make_bundle(paths, tmp_path)))
    assert sim.calls() == []


def test_test_mode_without_opt_in_never_runs_the_cli(
    paths: Paths, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def boom(*a: object, **k: object) -> None:
        raise AssertionError("the real colab CLI must not run in tests (invariant 20)")

    monkeypatch.setattr(subprocess, "Popen", boom)
    monkeypatch.delenv("GPU_ROUTER_REAL_PROVIDERS", raising=False)
    inert = make_adapter(paths, SystemClock(), None, test_mode=True)
    h = inert.healthcheck()
    assert h.health is ProviderHealth.DISABLED
    assert h.reason
    job = make_job(_target(inert))
    with pytest.raises(InvalidJob, match="test mode"):
        inert.submit(job, with_bundle(make_ctx(job), make_bundle(paths, tmp_path)))
    assert inert.lookup_by_key("gpu-000000000000-1") is None
    inert.cancel(RemoteRef(remote_id="gr-000000000000-1"))
    assert inert.quota().source == "estimate"
    monkeypatch.setenv("GPU_ROUTER_REAL_PROVIDERS", "kaggle,colab")
    assert not make_adapter(paths, SystemClock(), None, test_mode=True)._inert
    assert not make_adapter(paths, SystemClock(), None, test_mode=False)._inert


# --------------------------------------------------------------------------- crash recovery


def test_a_restarted_daemon_reattaches_by_attempt_key(
    colab: ColabAdapter, sim: ColabSim, paths: Paths, tmp_path: Path
) -> None:
    _, ctx, ref = submit(colab, paths, tmp_path, seconds=2)
    fresh = make_adapter(paths, SystemClock(), sim)  # same home, new process
    found = fresh.lookup_by_key(ctx.attempt_key)
    assert found is not None
    assert found.remote_id == ref.remote_id
    assert wait_terminal(fresh, found).phase is RemotePhase.SUCCEEDED
    assert fresh.fetch(found, tmp_path / "out").files == 1


def _dead_setup_record(adapter: ColabAdapter, key: str) -> RunRecord:
    return adapter.store.save(
        RunRecord(
            session=session_name(key),
            attempt_key=key,
            job_id=key.split("-")[1],
            attempt_n=1,
            state=RunState.SETUP,
            owner="a-daemon-that-died",
            gpu_requested="T4",
            created_at=SystemClock().now(),
        )
    )


def test_crash_mid_setup_is_lost_and_the_session_is_stopped(
    colab: ColabAdapter, sim: ColabSim
) -> None:
    key = "gpu-deadbeef0001-1"
    rec = _dead_setup_record(colab, key)
    cli = ColabCli([*colab._cli().prefix], colab.config_file, home=colab.cli_home)
    assert cli.run(["new", "--gpu", "T4", "-s", rec.session], timeout=20).ok
    ref = colab.lookup_by_key(key)
    assert ref is not None
    st = colab.status(ref)
    assert st.phase is RemotePhase.LOST
    assert st.lost_reason is not None
    assert "interrupted" in st.lost_reason
    assert rec.session not in sim.sessions()


def test_crash_after_launch_but_before_the_record_is_recovered(
    colab: ColabAdapter, paths: Paths, tmp_path: Path
) -> None:
    _, ctx, ref = submit(colab, paths, tmp_path, seconds=1)
    rec = colab.store.load(ref.remote_id)
    assert rec is not None
    colab.store.save(rec.model_copy(update={"state": RunState.SETUP, "owner": "dead"}))
    fresh = make_adapter(paths, SystemClock(), None, settings=colab.settings)
    found = fresh.lookup_by_key(ctx.attempt_key)
    assert found is not None
    assert wait_terminal(fresh, found).phase is RemotePhase.SUCCEEDED


def test_reaper_stops_finished_sessions_whose_stop_failed(
    colab: ColabAdapter, sim: ColabSim, paths: Paths, tmp_path: Path
) -> None:
    sim.control(stop_fail=100)
    _, _, first = submit(colab, paths, tmp_path)
    assert wait_terminal(colab, first).phase is RemotePhase.SUCCEEDED
    rec = colab.store.load(first.remote_id)
    assert rec is not None
    assert not rec.stopped
    assert rec.stop_error
    assert first.remote_id in sim.sessions()
    sim.control(stop_fail=0)
    _, _, second = submit(colab, paths, tmp_path, seconds=3)
    assert first.remote_id not in sim.sessions()
    assert second.remote_id in sim.sessions()
    colab.cancel(second)


def test_foreign_sessions_are_never_stopped(
    colab: ColabAdapter, sim: ColabSim, paths: Paths, tmp_path: Path
) -> None:
    sim.control(others=[["?", "T4"]])
    h = colab.healthcheck()
    assert h.health is ProviderHealth.DEGRADED
    assert h.reason
    assert "other colab" in h.reason
    _, _, ref = submit(colab, paths, tmp_path)
    wait_terminal(colab, ref)
    stopped = [a[a.index("-s") + 1] for a in sim.calls() if "stop" in a]
    assert stopped == [ref.remote_id]


# --------------------------------------------------------------------------- logs


def test_log_cursor_never_repeats_across_remote_and_harvested_reads(
    colab: ColabAdapter, paths: Paths, tmp_path: Path
) -> None:
    # ~420 KB of log (more than one remote read) and a line longer than a whole read,
    # printed while the run is still up, so the first reads come from the VM
    _, _, ref = submit(
        colab, paths, tmp_path, lines=6000, width=60, long_line=300_000, sleep_after=4
    )
    seen: list[str] = []
    cursor: str | None = None
    remote_reads = 0
    for _ in range(300):
        rec = colab.store.load(ref.remote_id)
        assert rec is not None
        remote_reads += 0 if rec.log_cached else 1
        chunks = list(colab.logs(ref, since=cursor))
        for c in chunks:
            seen.extend(c.lines)
            cursor = c.cursor
        if colab.status(ref).phase.terminal and chunks[-1].eof:
            break
        time.sleep(0.3)
    full = all_lines(colab.logs(ref))
    assert seen == full
    assert remote_reads > 0
    assert sum(1 for line in full if line.startswith("line ")) == 6000
    pieces = [line for line in full if line and set(line) == {"L"}]
    assert "".join(pieces) == "L" * 300_000  # cut at 64 KiB, the same way on both paths
    assert max(len(p) for p in pieces) == 64 * 1024
    assert all(not c.lines for c in colab.logs(ref, since=cursor))


def test_a_trailing_partial_line_arrives_once(
    colab: ColabAdapter, paths: Paths, tmp_path: Path
) -> None:
    _, _, ref = submit(colab, paths, tmp_path, partial_tail=True)
    wait_terminal(colab, ref)
    lines = all_lines(colab.logs(ref))
    assert sum("tail without newline" in line for line in lines) == 1


# --------------------------------------------------------------------------- checkpoints


def _ckpt_uri(adapter: ColabAdapter, ref: RemoteRef) -> tuple[int, str]:
    for line in all_lines(adapter.logs(ref)):
        ev = parse_line(line)
        if ev is not None and ev.t == "ckpt_end":
            data = json.loads(line.split(" ", 1)[1])
            return int(data["seq"]), str(data["uri"])
    raise AssertionError("no ckpt_end line")


def test_checkpoints_are_mirrored_and_a_new_vm_resumes_from_them(
    colab: ColabAdapter, sim: ColabSim, paths: Paths, tmp_path: Path
) -> None:
    # the first attempt dies (137 = OOM -> lost -> the engine migrates and resumes); its
    # final checkpoint is older than bootstrap's 10 s unsafe window, so it is archived
    job, _, first = submit(
        colab, paths, tmp_path, checkpoint=True, sleep_after_checkpoint=10.2, exit=137
    )
    assert wait_terminal(colab, first).phase is RemotePhase.LOST
    rec = colab.store.load(first.remote_id)
    assert rec is not None
    assert rec.ckpt_mirrored
    seq, uri = _ckpt_uri(colab, first)
    assert uri.endswith(rec.ckpt_mirrored)
    import shutil

    shutil.rmtree(sim.run_dir(first.remote_id))  # the first VM is gone
    now = SystemClock().now()
    ckpt = Checkpoint(
        id=f"{job.id}.c{seq}",
        job_id=job.id,
        attempt_id=f"{job.id}.1",
        seq=seq,
        uri=uri,
        created_at=now,
        recorded_at=now,
    )
    _, _, second = submit(
        colab,
        paths,
        tmp_path,
        n=2,
        job=job,
        resume_check=True,
        checkpoint=True,
        ctx_fields={"resume_from": ckpt},
    )
    assert wait_terminal(colab, second).phase is RemotePhase.SUCCEEDED
    lines = all_lines(colab.logs(second))
    assert "resumed=True" in lines
    assert "resume_files=['state.txt']" in lines
    rec2 = colab.store.load(second.remote_id)
    assert rec2 is not None
    assert rec2.resume == "uploaded"
    assert _ckpt_uri(colab, second)[0] == seq + 1  # seq stays monotonic per job


def test_unreachable_checkpoint_starts_fresh(
    colab: ColabAdapter, paths: Paths, tmp_path: Path
) -> None:
    job = make_job(_target(colab))
    now = SystemClock().now()
    ckpt = Checkpoint(
        id=f"{job.id}.c1",
        job_id=job.id,
        attempt_id=f"{job.id}.1",
        seq=1,
        uri="hf://datasets/someone/ckpts/c1.tar.gz",
        created_at=now,
        recorded_at=now,
    )
    _, _, ref = submit(
        colab, paths, tmp_path, n=2, job=job, resume_check=True, ctx_fields={"resume_from": ckpt}
    )
    wait_terminal(colab, ref)
    assert "resumed=False" in all_lines(colab.logs(ref))
    rec = colab.store.load(ref.remote_id)
    assert rec is not None
    assert rec.resume == "unavailable"


# --------------------------------------------------------------------------- quota + health


def test_quota_is_an_estimate_of_gpu_time_in_the_last_day(paths: Paths, sim: ColabSim) -> None:
    clock = FakeClock()
    colab = make_adapter(paths, clock, sim)
    now = clock.now()

    def rec(name: str, start: float, end: float | None) -> None:
        colab.store.save(
            RunRecord(
                session=name,
                attempt_key=f"gpu-{name}-1",
                job_id=name,
                attempt_n=1,
                state=RunState.EXITED if end else RunState.LAUNCHED,
                owner="t",
                gpu_requested="T4",
                created_at=start,
                launched_at=start,
                ended_at=end,
            )
        )

    rec("gr-old", now - 3 * 86400, now - 3 * 86400 + 3600)  # outside the window
    rec("gr-a", now - 7200, now - 3600)  # 1 h
    rec("gr-b", now - 1800, None)  # running: 0.5 h so far
    q = colab.quota()
    assert q.source == "estimate"
    assert q.used == pytest.approx(1.5)
    assert q.limit is None
    assert q.resets_at is None
    assert q.detail["runs"] == 2


def test_healthcheck_reports_auth_and_network_problems(colab: ColabAdapter, sim: ColabSim) -> None:
    assert colab.healthcheck().health is ProviderHealth.OK
    sim.control(sessions_mode="auth")
    h = colab.healthcheck()
    assert h.health is ProviderHealth.AUTH_REQUIRED
    assert h.hint
    assert "gcloud" in h.hint
    sim.control(sessions_mode="network")
    h = colab.healthcheck()
    assert h.health is ProviderHealth.UNAVAILABLE
    assert h.reason


def test_healthcheck_without_the_cli(paths: Paths, tmp_path: Path) -> None:
    missing = make_adapter(
        paths,
        SystemClock(),
        None,
        test_mode=False,
        settings=ProviderSettings(cli=str(tmp_path / "no-such-colab")),
    )
    h = missing.healthcheck()
    assert h.health is ProviderHealth.UNAVAILABLE
    assert h.hint is not None
    assert "uv tool install" in h.hint


def test_old_local_run_copies_are_dropped_at_the_next_submit(
    colab: ColabAdapter, paths: Paths, tmp_path: Path
) -> None:
    now = SystemClock().now()
    old = colab.store.save(
        RunRecord(
            session="gr-0ld000000000-1",
            attempt_key="gpu-0ld000000000-1",
            job_id="0ld000000000",
            attempt_n=1,
            state=RunState.EXITED,
            exit_code=0,
            owner="t",
            gpu_requested="T4",
            created_at=now - 9 * 86400,
            ended_at=now - 8 * 86400,
            stopped=True,
        )
    )
    recent = colab.store.save(
        old.model_copy(update={"session": "gr-new000000000-1", "ended_at": now - 3600})
    )
    _, _, ref = submit(colab, paths, tmp_path, seconds=3)
    assert colab.store.load(old.session) is None
    assert colab.store.load(recent.session) is not None
    colab.cancel(ref)
