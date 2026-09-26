"""LocalAdapter: cancel, external kills, daemon restarts, interrupted submits."""

from __future__ import annotations

import json
import os
import signal
import time
from pathlib import Path

import pytest

from gpu_router.adapters.base import RemotePhase, RemoteRef
from gpu_router.paths import Paths
from gpu_router.providers.local import adapter as local_adapter
from tests.unit.providers.local.helpers import (
    all_lines,
    build_project,
    make_adapter,
    make_bundle,
    make_ctx,
    make_job,
    pid_alive,
    read_pid,
    run_dir,
    wait_for_line,
    wait_phase,
)

SLEEPER = """
import os, time, gpu
(gpu.output_dir() / "child.pid").write_text(str(os.getpid()))
print("ready", flush=True)
time.sleep(60)
"""

STUBBORN = """
import os, signal, time, gpu
signal.signal(signal.SIGTERM, signal.SIG_IGN)
(gpu.output_dir() / "child.pid").write_text(str(os.getpid()))
print("ready", flush=True)
while True:
    time.sleep(0.1)
"""


def _child_pid(adapter: local_adapter.LocalAdapter, ref: RemoteRef) -> int:
    return int((run_dir(adapter, ref) / "work" / "outputs" / "child.pid").read_text())


def _gone(pid: int, timeout_s: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while pid_alive(pid):
        if time.monotonic() > deadline:
            return False
        time.sleep(0.05)
    return True


def _start(
    paths: Paths, tmp_path: Path, script: str
) -> tuple[local_adapter.LocalAdapter, RemoteRef]:
    adapter = make_adapter(paths)
    project = build_project(tmp_path, script)
    job = make_job(project)
    ref = adapter.submit(job, make_ctx(job, make_bundle(paths, project)))
    return adapter, ref


def test_cancel_running_job_is_cancelled_and_kills_the_script(paths: Paths, tmp_path: Path) -> None:
    adapter, ref = _start(paths, tmp_path, SLEEPER)
    wait_for_line(adapter, ref, "ready")
    assert adapter.status(ref).phase is RemotePhase.RUNNING
    child = _child_pid(adapter, ref)
    adapter.cancel(ref)
    st = wait_phase(adapter, ref)
    assert st.phase is RemotePhase.CANCELLED
    assert _gone(child)
    assert not pid_alive(read_pid(adapter, ref))
    adapter.cancel(ref)  # idempotent (A5)
    assert adapter.status(ref).phase is RemotePhase.CANCELLED
    assert all_lines(adapter, ref)[-1].startswith("::gpu:: ")  # bootstrap still wrote exit


def test_cancel_right_after_submit_stops_the_setup(paths: Paths, tmp_path: Path) -> None:
    adapter, ref = _start(paths, tmp_path, SLEEPER)
    adapter.cancel(ref)
    assert wait_phase(adapter, ref).phase is RemotePhase.CANCELLED


def test_cancel_escalates_to_sigkill_when_sigterm_is_ignored(
    paths: Paths, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(local_adapter, "CANCEL_GRACE_S", 0.5)
    adapter, ref = _start(paths, tmp_path, STUBBORN)
    wait_for_line(adapter, ref, "ready")
    child = _child_pid(adapter, ref)
    adapter.cancel(ref)
    assert wait_phase(adapter, ref, timeout_s=10).phase is RemotePhase.CANCELLED
    assert _gone(child), "the entrypoint's own process group must be killed too"


def test_cancel_after_success_keeps_succeeded(paths: Paths, tmp_path: Path) -> None:
    adapter, ref = _start(paths, tmp_path, "print('quick')\n")
    wait_phase(adapter, ref, {RemotePhase.SUCCEEDED})
    adapter.cancel(ref)
    assert adapter.status(ref).phase is RemotePhase.SUCCEEDED
    assert not (run_dir(adapter, ref) / "cancel.json").exists()


def test_exit_zero_wins_over_a_cancel_that_raced_the_finish(paths: Paths, tmp_path: Path) -> None:
    adapter, ref = _start(paths, tmp_path, "print('quick')\n")
    wait_phase(adapter, ref, {RemotePhase.SUCCEEDED})
    (run_dir(adapter, ref) / "cancel.json").write_text("{}")
    assert adapter.status(ref).phase is RemotePhase.SUCCEEDED


def test_runner_killed_from_outside_is_lost(paths: Paths, tmp_path: Path) -> None:
    adapter, ref = _start(paths, tmp_path, SLEEPER)
    wait_for_line(adapter, ref, "ready")
    child = _child_pid(adapter, ref)
    os.kill(read_pid(adapter, ref), signal.SIGKILL)  # bootstrap dies, no EXIT
    try:
        st = wait_phase(adapter, ref, timeout_s=10)
        assert st.phase is RemotePhase.LOST
        assert st.lost_reason is not None
        assert "without an exit code" in st.lost_reason
        chunks = list(adapter.logs(ref))
        assert chunks[-1].eof
    finally:
        os.killpg(child, signal.SIGKILL)  # the orphaned entrypoint (its own group)


def test_sigterm_from_outside_is_lost_not_failed(paths: Paths, tmp_path: Path) -> None:
    adapter, ref = _start(paths, tmp_path, SLEEPER)
    wait_for_line(adapter, ref, "ready")
    os.killpg(read_pid(adapter, ref), signal.SIGTERM)  # e.g. macOS shutting down
    st = wait_phase(adapter, ref, timeout_s=15)
    assert st.phase is RemotePhase.LOST
    assert st.exit_code == 128 + signal.SIGTERM
    assert st.lost_reason is not None
    assert "SIGTERM" in st.lost_reason


def test_new_adapter_instance_reattaches_to_a_running_job(paths: Paths, tmp_path: Path) -> None:
    script = "import time\nprint('ready', flush=True)\ntime.sleep(1.0)\nprint('finished')\n"
    first, ref = _start(paths, tmp_path, script)
    wait_for_line(first, ref, "ready")
    chunks = list(first.logs(ref))
    cursor = chunks[-1].cursor
    before = [line for c in chunks for line in c.lines]
    del first  # the daemon went away; the run did not (invariant 11)

    second = make_adapter(paths)
    found = second.lookup_by_key(ref.remote_id)
    assert found is not None
    assert found.remote_id == ref.remote_id
    assert second.status(ref).phase is RemotePhase.RUNNING
    assert wait_phase(second, ref).phase is RemotePhase.SUCCEEDED
    after = [line for c in second.logs(ref, since=cursor) for line in c.lines]
    assert before + after == all_lines(second, ref)
    assert "finished" in after


def test_submit_interrupted_before_the_launcher_started_is_lost(
    paths: Paths, tmp_path: Path
) -> None:
    """run.json committed, then the daemon died before starting the launcher."""
    adapter = make_adapter(paths)
    project = build_project(tmp_path, "print(1)\n")
    job = make_job(project)
    ctx = make_ctx(job, make_bundle(paths, project))
    rd = adapter.runs_root / ctx.attempt_key
    rd.mkdir(parents=True)
    (rd / "run.json").write_text(json.dumps({"remote_id": ctx.attempt_key, "env": "system"}))
    (rd / "alive.lock").touch()
    found = adapter.lookup_by_key(ctx.attempt_key)
    assert found is not None
    st = adapter.status(found)
    assert st.phase is RemotePhase.LOST
    assert st.lost_reason is not None
    assert "never started" in st.lost_reason
    again = adapter.submit(job, ctx)  # A4: the key already has a run; no second launch
    assert again.remote_id == found.remote_id
    assert adapter.status(again).phase is RemotePhase.LOST
    adapter.cancel(again)  # dead: no-op


def test_leftover_dir_without_a_record_is_reclaimed(paths: Paths, tmp_path: Path) -> None:
    adapter = make_adapter(paths)
    project = build_project(tmp_path, "print('reclaimed')\n")
    job = make_job(project)
    ctx = make_ctx(job, make_bundle(paths, project))
    rd = adapter.runs_root / ctx.attempt_key
    rd.mkdir(parents=True)
    (rd / "launch.json").write_text("{half written")
    assert adapter.lookup_by_key(ctx.attempt_key) is None
    ref = adapter.submit(job, ctx)
    assert wait_phase(adapter, ref).phase is RemotePhase.SUCCEEDED
    assert "reclaimed" in all_lines(adapter, ref)


def test_status_never_changes_the_run(paths: Paths, tmp_path: Path) -> None:
    adapter, ref = _start(paths, tmp_path, "print('x')\n")
    wait_phase(adapter, ref)
    rd = run_dir(adapter, ref)
    before = sorted((p.name, p.stat().st_mtime_ns) for p in rd.iterdir())
    for _ in range(3):
        adapter.status(ref)
        list(adapter.logs(ref))
        adapter.lookup_by_key(ref.remote_id)
    assert sorted((p.name, p.stat().st_mtime_ns) for p in rd.iterdir()) == before
