"""Phase-3 review regressions for the Colab adapter (D34-D37 in CLAUDE.md), against the
simulated colab CLI (fake_colab.py)."""

from __future__ import annotations

import contextlib
import json
import os
import signal
import stat
import subprocess
import time
from pathlib import Path
from typing import Any

import pytest

from gpu_router.adapters.base import RemotePhase, RemoteRef
from gpu_router.clock import FakeClock, SystemClock
from gpu_router.errors import DEFINITIVE_SUBMIT_ERRORS, RateLimited, Unavailable
from gpu_router.paths import Paths
from gpu_router.providers.colab import adapter as colab_mod
from gpu_router.providers.colab.adapter import ColabAdapter, session_name
from gpu_router.providers.colab.cli import CliResult, classify
from gpu_router.providers.colab.state import RunRecord, RunState
from tests.contract.harness import ContractTarget, make_ctx, make_job
from tests.unit.providers.colab.helpers import (
    ColabSim,
    all_lines,
    make_adapter,
    make_bundle,
    with_bundle,
)
from tests.unit.providers.colab.test_adapter import submit, wait_running, wait_terminal

JOB_429 = "a4290c12de34"


def _target(adapter: ColabAdapter) -> ContractTarget:
    return ContractTarget(name="colab", build=lambda: adapter, clock=SystemClock())


# --------------------------------------------------------------------------- 429 in names


def test_new_output_naming_a_429_session_is_not_a_rate_limit() -> None:
    res = CliResult(
        ("new",),
        1,
        f"[colab] Creating session 'gr-{JOB_429}-1'...\n",
        "Traceback (most recent call last):\nKeyError: 'endpoint'\n",
    )
    err = classify(res, provider="colab", op="new")
    assert type(err) is Unavailable


@pytest.mark.parametrize(
    "text",
    [
        "429 Client Error: Too Many Requests for url: https://colab.research.google.com/x",
        '{"error": {"code": 429, "status": "RESOURCE_EXHAUSTED"}}',
    ],
)
def test_real_throttling_is_still_rate_limited(text: str) -> None:
    res = CliResult(("new",), 1, f"[colab] Creating session 'gr-{JOB_429}-1'...\n", text)
    assert isinstance(classify(res, provider="colab", op="new"), RateLimited)


def test_unclear_new_failure_of_a_429_job_is_ambiguous_and_stops_the_name(
    colab: ColabAdapter,
    sim: ColabSim,
    paths: Paths,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(colab_mod, "NEW_TIMEOUT_S", 5.0)
    sim.control(new="crash")
    job = make_job(_target(colab)).model_copy(update={"id": JOB_429})
    ctx = with_bundle(make_ctx(job), make_bundle(paths, tmp_path))
    with pytest.raises(Unavailable) as info:
        colab.submit(job, ctx)
    assert not isinstance(info.value, DEFINITIVE_SUBMIT_ERRORS)
    assert "stop" in sim.commands(), "a session the CLI may have created must be stopped"
    rec = colab.store.load(session_name(ctx.attempt_key))
    assert rec is not None
    assert rec.state is RunState.ABANDONED


# --------------------------------------------------------------------------- helpers


def _rec_path_mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def _wait(cond: Any, timeout: float = 30.0, what: str = "condition") -> None:
    deadline = time.monotonic() + timeout
    while not cond():
        assert time.monotonic() < deadline, f"timed out waiting for {what}"
        time.sleep(0.1)


def _submit(adapter: ColabAdapter, paths: Paths, tmp_path: Path, **cfg: Any) -> RemoteRef:
    return submit(adapter, paths, tmp_path, **cfg)[2]


def _record(adapter: ColabAdapter, session: str) -> RunRecord:
    rec = adapter.store.load(session)
    assert rec is not None
    return rec


# --------------------------------------------------------------------------- teardown (D36)


def test_logs_retries_a_stop_that_failed_inside_status(
    colab: ColabAdapter, sim: ColabSim, paths: Paths, tmp_path: Path
) -> None:
    """Engine order on a finished run: status() (terminal) -> logs() -> fetch()."""
    sim.control(stop_fail=1)  # one transient `colab stop` failure
    ref = _submit(colab, paths, tmp_path)
    assert wait_terminal(colab, ref).phase is RemotePhase.SUCCEEDED
    assert not _record(colab, ref.remote_id).stopped  # the stop inside status failed
    assert "demo done" in all_lines(colab.logs(ref, since=None))
    assert _record(colab, ref.remote_id).stopped
    assert ref.remote_id not in sim.sessions()
    assert colab.fetch(ref, tmp_path / "out").files == 1


def test_fetch_retries_a_stop_even_when_outputs_were_cached(
    colab: ColabAdapter, sim: ColabSim, paths: Paths, tmp_path: Path
) -> None:
    sim.control(stop_fail=1)
    ref = _submit(colab, paths, tmp_path)
    assert wait_terminal(colab, ref).phase is RemotePhase.SUCCEEDED
    rec = _record(colab, ref.remote_id)
    assert rec.outputs_cached
    assert rec.log_cached
    assert not rec.stopped
    assert colab.fetch(ref, tmp_path / "out").files == 1
    assert _record(colab, ref.remote_id).stopped
    assert ref.remote_id not in sim.sessions()


def test_failed_run_whose_stop_failed_is_stopped_by_logs(
    colab: ColabAdapter, sim: ColabSim, paths: Paths, tmp_path: Path
) -> None:
    sim.control(stop_fail=1)
    ref = _submit(colab, paths, tmp_path, exit=3)
    assert wait_terminal(colab, ref).phase is RemotePhase.FAILED
    list(colab.logs(ref))
    assert ref.remote_id not in sim.sessions()


def test_a_status_that_runs_out_of_budget_leaves_the_rest_to_logs_and_fetch(
    colab: ColabAdapter,
    sim: ColabSim,
    paths: Paths,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ref = _submit(colab, paths, tmp_path, extra_outputs=True)
    _wait(lambda: (sim.run_dir(ref.remote_id) / "RC").exists(), what="the run to end on the VM")
    monkeypatch.setattr(colab_mod, "STATUS_BUDGET_S", 20.0)  # poll only; no harvest fits
    assert colab.status(ref).phase is RemotePhase.SUCCEEDED
    rec = _record(colab, ref.remote_id)
    assert not rec.stopped
    assert rec.settle_tries == 0, "a harvest that never started must not use up a try"
    monkeypatch.setattr(colab_mod, "STATUS_BUDGET_S", 52.0)
    lines = all_lines(colab.logs(ref))
    assert "demo done" in lines
    assert colab.fetch(ref, tmp_path / "out").files == 2
    rec = _record(colab, ref.remote_id)
    assert rec.stopped
    assert ref.remote_id not in sim.sessions()


def test_settle_never_starts_a_harvest_without_room_for_the_stop(
    colab: ColabAdapter, paths: Paths
) -> None:
    rec = RunRecord(
        session="gr-000000000001-1",
        attempt_key="gpu-000000000001-1",
        job_id="000000000001",
        attempt_n=1,
        state=RunState.EXITED,
        exit_code=3,
        owner="t",
        gpu_requested="T4",
        created_at=SystemClock().now(),
    )
    budget = colab_mod._Budget(colab, colab_mod.MIN_STEP_S + colab_mod.STOP_RESERVE_S - 1)
    calls: list[list[str]] = []
    monkey = colab._cli()
    orig = type(monkey).run

    def spy(self: Any, args: Any, **kw: Any) -> Any:
        calls.append(list(args))
        return orig(self, args, **kw)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(type(monkey), "run", spy)
        colab._settle(rec, budget, packed=None)
    assert calls == [] or calls[0][0] == "stop"
    assert rec.settle_tries == 0


def test_a_success_does_not_spend_its_teardown_budget_on_the_final_checkpoint(
    colab: ColabAdapter, sim: ColabSim, paths: Paths, tmp_path: Path
) -> None:
    ref = _submit(colab, paths, tmp_path, checkpoint=True)
    assert wait_terminal(colab, ref).phase is RemotePhase.SUCCEEDED
    rec = _record(colab, ref.remote_id)
    assert rec.ckpt_mirrored is None  # a finished job is never resumed
    assert rec.stopped
    assert rec.outputs_cached


def test_janitor_stops_a_finished_session_without_another_submit(
    colab: ColabAdapter,
    sim: ColabSim,
    paths: Paths,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(colab_mod, "JANITOR_ENABLED", True)
    monkeypatch.setattr(colab_mod, "JANITOR_INTERVAL_S", 0.3)
    monkeypatch.setattr(colab_mod, "RAW_LOG_GRACE_S", 0.5)  # the raw log purge is its job too
    sim.control(stop_fail=1000)  # colab cannot be asked to stop anything for a while
    ref = _submit(colab, paths, tmp_path, exit=3)
    assert wait_terminal(colab, ref).phase is RemotePhase.FAILED
    list(colab.logs(ref))
    assert ref.remote_id in sim.sessions()
    assert colab._janitor is not None
    sim.control(stop_fail=0)  # colab is reachable again; nobody calls the adapter
    _wait(lambda: ref.remote_id not in sim.sessions(), what="the janitor to stop the session")
    _wait(lambda: _record(colab, ref.remote_id).stopped, what="the record to say stopped")
    _wait(lambda: _record(colab, ref.remote_id).log_purged, what="the raw log purge")
    _wait(lambda: colab._janitor is None, what="the janitor to finish")


def test_healthcheck_after_a_restart_starts_the_janitor_for_leftovers(
    colab: ColabAdapter,
    sim: ColabSim,
    paths: Paths,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sim.control(stop_fail=1000)
    ref = _submit(colab, paths, tmp_path)
    wait_terminal(colab, ref)
    colab.close()
    sim.control(stop_fail=0)
    monkeypatch.setattr(colab_mod, "JANITOR_ENABLED", True)
    monkeypatch.setattr(colab_mod, "JANITOR_INTERVAL_S", 0.3)
    fresh = make_adapter(paths, SystemClock(), None, settings=colab.settings)  # new daemon
    try:
        assert fresh.healthcheck().ok
        _wait(lambda: ref.remote_id not in sim.sessions(), what="the janitor to stop it")
    finally:
        fresh.close()


# --------------------------------------------------------------------------- cancel (F5)


def test_cancel_of_a_run_that_already_finished_keeps_its_outputs_and_log(
    colab: ColabAdapter, sim: ColabSim, paths: Paths, tmp_path: Path
) -> None:
    ref = _submit(colab, paths, tmp_path)
    _wait(lambda: (sim.run_dir(ref.remote_id) / "RC").exists(), what="the run to end on the VM")
    colab.cancel(ref)  # no status() has seen the exit yet
    st = colab.status(ref)
    assert st.phase is RemotePhase.SUCCEEDED
    assert colab.fetch(ref, tmp_path / "out").files == 1
    assert "demo done" in all_lines(colab.logs(ref))
    assert ref.remote_id not in sim.sessions()


def test_cancel_of_a_running_run_still_stops_it(
    colab: ColabAdapter, sim: ColabSim, paths: Paths, tmp_path: Path
) -> None:
    ref = _submit(colab, paths, tmp_path, seconds=60)
    wait_running(colab, ref)
    colab.cancel(ref)
    assert colab.status(ref).phase is RemotePhase.CANCELLED
    assert ref.remote_id not in sim.sessions()


# --------------------------------------------------------------------------- orphaned `new` (F3)


def _creating_record(
    adapter: ColabAdapter, key: str, *, created_at: float, pid: int | None
) -> RunRecord:
    return adapter.store.save(
        RunRecord(
            session=session_name(key),
            attempt_key=key,
            job_id=key.split("-")[1],
            attempt_n=1,
            state=RunState.CREATING,
            owner="a-daemon-that-died",
            gpu_requested="T4",
            created_at=created_at,
            cli_pid=pid,
        )
    )


def _orphan_new(adapter: ColabAdapter, session: str) -> subprocess.Popen[bytes]:
    """A `colab new` that outlived its daemon: own session, stdout to nowhere."""
    cli = adapter._cli()
    return subprocess.Popen(
        [*cli.base_argv(), "new", "--gpu", "T4", "-s", session],
        env=cli.env(),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def test_an_orphaned_new_keeps_the_attempt_pending_then_its_session_is_stopped(
    colab: ColabAdapter, sim: ColabSim
) -> None:
    key = "gpu-0rphan000001-1"
    sim.control(new="slow", new_delay=2.5)
    orphan = _orphan_new(colab, session_name(key))
    try:
        _creating_record(colab, key, created_at=SystemClock().now(), pid=orphan.pid)
        fresh = make_adapter(colab.paths, SystemClock(), None, settings=colab.settings)
        ref = fresh.lookup_by_key(key)
        assert ref is not None
        assert fresh.status(ref).phase is RemotePhase.PENDING  # the `new` is still running
        assert orphan.wait(timeout=30) == 0
        assert session_name(key) in sim.sessions()  # registered after the daemon "died"
        st = fresh.status(ref)
        assert st.phase is RemotePhase.LOST
        assert session_name(key) not in sim.sessions(), "the late session must be stopped"
        assert _record(fresh, session_name(key)).stopped
    finally:
        with contextlib.suppress(OSError):
            os.killpg(orphan.pid, signal.SIGKILL)


def test_without_a_pid_the_attempt_waits_out_the_new_timeout_then_stops_the_name(
    paths: Paths, sim: ColabSim
) -> None:
    clock = FakeClock(SystemClock().now())
    adapter = make_adapter(paths, clock, sim)
    key = "gpu-0rphan000002-1"
    _creating_record(adapter, key, created_at=clock.now(), pid=None)
    ref = adapter.lookup_by_key(key)
    assert ref is not None
    assert adapter.status(ref).phase is RemotePhase.PENDING
    assert "exec" not in sim.commands()
    clock.advance(colab_mod.NEW_TIMEOUT_S + colab_mod.ORPHAN_GRACE_S + 1)
    assert adapter.status(ref).phase is RemotePhase.LOST
    assert "stop" in sim.commands(), "a 'not found' name is still stopped (idempotent)"


def test_an_orphan_still_running_past_the_stale_window_is_killed(
    paths: Paths, sim: ColabSim
) -> None:
    clock = FakeClock(SystemClock().now())
    adapter = make_adapter(paths, clock, sim)
    key = "gpu-0rphan000003-1"
    sim.control(new="slow", new_delay=120)
    orphan = _orphan_new(adapter, session_name(key))
    try:
        _creating_record(adapter, key, created_at=clock.now(), pid=orphan.pid)
        ref = adapter.lookup_by_key(key)
        assert ref is not None
        assert adapter.status(ref).phase is RemotePhase.PENDING
        clock.advance(colab_mod.STALE_START_S + 1)
        assert adapter.status(ref).phase is RemotePhase.LOST
        assert orphan.wait(timeout=10) != 0  # killed, never got to register the session
        assert session_name(key) not in sim.sessions()
    finally:
        with contextlib.suppress(OSError):
            os.killpg(orphan.pid, signal.SIGKILL)


def test_a_retried_submit_never_races_an_orphaned_new(
    colab: ColabAdapter, sim: ColabSim, paths: Paths, tmp_path: Path
) -> None:
    job = make_job(_target(colab)).model_copy(update={"id": "0rphan000004"})
    ctx = with_bundle(make_ctx(job), make_bundle(paths, tmp_path))
    sim.control(new="slow", new_delay=30)
    orphan = _orphan_new(colab, session_name(ctx.attempt_key))
    try:
        _creating_record(colab, ctx.attempt_key, created_at=SystemClock().now(), pid=orphan.pid)
        _wait(lambda: "new" in sim.commands(), what="the orphan to reach the CLI")
        with pytest.raises(Unavailable, match="may still be creating"):
            colab.submit(job, ctx)
        assert sim.commands().count("new") == 1  # only the orphan's own call
    finally:
        with contextlib.suppress(OSError):
            os.killpg(orphan.pid, signal.SIGKILL)


def test_the_pid_of_new_is_recorded_while_it_runs(
    colab: ColabAdapter, sim: ColabSim, paths: Paths, tmp_path: Path
) -> None:
    seen: list[int | None] = []
    orig = colab._note_cli_pid

    def spy(rec: RunRecord, pid: int) -> None:
        orig(rec, pid)
        saved = colab.store.load(rec.session)
        seen.append(saved.cli_pid if saved else None)

    colab._note_cli_pid = spy  # type: ignore[method-assign]
    ref = _submit(colab, paths, tmp_path, seconds=3)
    assert seen
    assert seen[0] is not None
    assert _record(colab, ref.remote_id).cli_pid is None  # cleared once `new` returned
    colab.cancel(ref)


# --------------------------------------------------------------------------- CLI home (F7/F9)


def test_the_cli_runs_with_a_private_home_and_forgets_our_history(
    colab: ColabAdapter, sim: ColabSim, paths: Paths, tmp_path: Path
) -> None:
    ref = _submit(colab, paths, tmp_path, seconds=3)
    home = colab.cli_home
    history = home / ".config" / "colab-cli" / "history" / f"{ref.remote_id}.jsonl"
    assert history.is_file(), "the CLI's exec history goes to the private home"
    assert _rec_path_mode(home) == 0o700
    assert paths.home in home.parents
    wait_terminal(colab, ref)
    list(colab.logs(ref))
    assert not history.exists(), "a stopped session's history (exec output) is deleted"
    log = home / ".config" / "colab-cli" / "colab.log"
    assert "colab-runtime-proxy-token" in log.read_text()  # stays inside the data dir
    calls = [
        json.loads(line)
        for line in (colab.config_file.parent / "sim-calls.jsonl").read_text().splitlines()
    ]
    real_gcloud = str(Path(os.path.expanduser("~")) / ".config" / "gcloud")
    for call in calls:
        assert call["home"] == str(home)
        assert call["cloudsdk_config"] == os.environ.get("CLOUDSDK_CONFIG", real_gcloud)


def test_an_oversized_private_cli_log_is_truncated(colab: ColabAdapter) -> None:
    log = colab.cli_home / ".config" / "colab-cli" / "colab.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_bytes(b"x" * (colab_mod.CLI_LOG_MAX_BYTES + 1))
    colab.healthcheck()
    assert log.stat().st_size < 1000  # truncated (the healthcheck's own call adds a line)


# --------------------------------------------------------------------------- tmp secrets (F8)


def test_leftover_secret_files_are_swept_at_start_and_at_submit(
    colab: ColabAdapter, sim: ColabSim, paths: Paths, tmp_path: Path
) -> None:
    tmp = colab.scratch_dir / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    left = tmp / ".secrets-ab12cd.json"
    left.write_text('{"HF_TOKEN": "hf_left_behind_by_a_killed_daemon"}')
    (tmp / "poll-xyz.py").write_text("print(1)")
    make_adapter(paths, SystemClock(), None, settings=colab.settings)  # a restarted daemon
    assert not left.exists()
    assert not any(tmp.iterdir())
    left.write_text("{}")  # appears while the daemon runs (a crash of another thread)
    ref = _submit(colab, paths, tmp_path, seconds=2)
    assert not left.exists()
    colab.cancel(ref)


# --------------------------------------------------------------------------- raw log (F10)


def test_the_raw_harvested_log_is_purged_after_it_was_served(
    colab: ColabAdapter,
    sim: ColabSim,
    paths: Paths,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ref = _submit(colab, paths, tmp_path)
    wait_terminal(colab, ref)
    raw = colab.store.run_dir(ref.remote_id) / "job.log"
    assert raw.is_file()
    chunks = list(colab.logs(ref))
    assert chunks[-1].eof
    cursor = chunks[-1].cursor
    assert _record(colab, ref.remote_id).log_served_at is not None
    colab.healthcheck()
    assert raw.is_file(), "kept for RAW_LOG_GRACE_S after it was served"
    monkeypatch.setattr(colab_mod, "RAW_LOG_GRACE_S", 0.0)
    colab.healthcheck()
    assert not raw.exists()
    assert _record(colab, ref.remote_id).log_purged
    after = list(colab.logs(ref, since=cursor))
    assert after[-1].eof
    assert all(not c.lines for c in after)  # A7: nothing repeats past the cursor
