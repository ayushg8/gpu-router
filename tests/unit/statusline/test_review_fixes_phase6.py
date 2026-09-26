"""Phase-6 review fixes on the status line (D48): one install record per settings file,
backups never overwritten, a command that degrades to the original, OS errors reported
cleanly, a bounded wait for gpu in the wrapper, control characters stripped, a per-attempt
ETA, and a wall-clock heartbeat."""

from __future__ import annotations

import asyncio
import io
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import pytest

from gpu_router.clock import FakeClock
from gpu_router.models import AttemptPatch, JobPatch, JobSpec
from gpu_router.statefile import StateFileWriter, StateSnapshot, build_snapshot
from gpu_router.statemachine import AttemptState, JobState, Reason
from gpu_router.statusline import fast, install
from gpu_router.store import AttemptChange, Store

REPO = Path(__file__).parents[3]
WRAPPER = REPO / "plugin" / "statusline" / "gpu-statusline.sh"

# ------------------------------------------------------------ installer fixtures


@pytest.fixture
def two(tmp_path: Path) -> dict[str, Any]:
    """One gpu-router home, two Claude config dirs (A and B) with their own status lines."""
    user = tmp_path / "user"
    out: dict[str, Any] = {"user": user, "home": tmp_path / "gpuhome", "tmp": tmp_path}
    for name in ("A", "B"):
        settings = user / f"claude-{name}" / "settings.json"
        settings.parent.mkdir(parents=True)
        line = {"type": "command", "command": f"echo ACCOUNT-{name}", "refreshInterval": 2}
        settings.write_text(install.dumps({"statusLine": line, "theme": name}))
        out[name] = settings
    gpu = tmp_path / "bin" / "gpu"
    gpu.parent.mkdir()
    gpu.write_text("#!/bin/sh\ncat >/dev/null\nexit 0\n")
    gpu.chmod(0o755)
    out["gpu"] = gpu
    return out


def _inst(t: dict[str, Any], which: str, home: Path | None = None) -> tuple[int, str]:
    buf = io.StringIO()
    code = install.run_install(
        t[which], home or t["home"], yes=True, out=buf, gpu_bin=t["gpu"], user_home=t["user"]
    )
    return code, buf.getvalue()


def _uninst(t: dict[str, Any], which: str, home: Path | None = None) -> tuple[int, str]:
    buf = io.StringIO()
    code = install.run_uninstall(
        t[which], home or t["home"], yes=True, out=buf, user_home=t["user"]
    )
    return code, buf.getvalue()


def _run_line(t: dict[str, Any], which: str) -> subprocess.CompletedProcess[str]:
    """Run the settings file's statusLine command the way Claude Code does."""
    command = json.loads(t[which].read_text())["statusLine"]["command"]
    env = {"PATH": os.environ["PATH"], "HOME": str(t["user"])}
    return subprocess.run(
        ["/bin/sh", "-c", command],
        input='{"workspace": {"current_dir": "/x"}}',
        capture_output=True,
        text=True,
        timeout=20,
        env=env,
        check=False,
    )


# ------------------------------------------------------------ finding 11: records


def test_two_settings_files_keep_their_own_originals(two: dict[str, Any]) -> None:
    assert _inst(two, "A")[0] == 0
    assert _inst(two, "B")[0] == 0
    assert install.record_dir(two["home"], two["A"]) != install.record_dir(two["home"], two["B"])
    assert _run_line(two, "A").stdout == "ACCOUNT-A"  # not B's line
    assert _run_line(two, "B").stdout == "ACCOUNT-B"
    code, out = _uninst(two, "A")
    assert code == 0, out
    assert json.loads(two["A"].read_text())["statusLine"]["command"] == "echo ACCOUNT-A"
    # B still works and can still be uninstalled
    b = _run_line(two, "B")
    assert (b.returncode, b.stdout) == (0, "ACCOUNT-B")
    assert install.record_dir(two["home"], two["B"]).is_dir()
    assert _uninst(two, "B")[0] == 0
    assert json.loads(two["B"].read_text())["statusLine"]["command"] == "echo ACCOUNT-B"
    assert not (two["home"] / "statusline").exists()


def test_uninstall_from_another_home_is_refused_and_names_it(two: dict[str, Any]) -> None:
    assert _inst(two, "A")[0] == 0
    installed = two["A"].read_bytes()
    sandbox = two["tmp"] / "sandbox-home"
    code, out = _uninst(two, "A", home=sandbox)
    assert code == install.EXIT_ERROR
    assert str(two["home"]) in out
    assert "GPU_ROUTER_HOME=" in out
    assert two["A"].read_bytes() == installed
    code, out = _inst(two, "A", home=sandbox)  # never wraps the wrapper
    assert code == install.EXIT_ERROR
    assert "already runs gpu-router's wrapper" in out
    assert two["A"].read_bytes() == installed


def test_a_copied_settings_file_is_not_restored_from_the_wrong_record(
    two: dict[str, Any],
) -> None:
    assert _inst(two, "A")[0] == 0
    two["B"].write_bytes(two["A"].read_bytes())  # e.g. a dotfiles copy
    before = two["B"].read_bytes()
    code, out = _uninst(two, "B")
    assert code == install.EXIT_ERROR
    assert "another settings file" in out
    assert two["B"].read_bytes() == before


# ------------------------------------------------------------ finding 14: backups


def test_backups_in_the_same_second_never_overwrite(
    two: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    original = two["A"].read_bytes()
    monkeypatch.setattr(install, "_stamp", lambda: "20260924-061738")
    assert _inst(two, "A")[0] == 0
    assert _uninst(two, "A")[0] == 0
    backups = sorted(two["A"].parent.glob("settings.json.gpu-router-*.bak"))
    assert [b.name for b in backups] == [
        "settings.json.gpu-router-20260924-061738-2.bak",
        "settings.json.gpu-router-20260924-061738.bak",
    ]
    assert (two["A"].parent / "settings.json.gpu-router-20260924-061738.bak").read_bytes() == (
        original
    )


# ------------------------------------------------------------ finding 15: self-degrading


def test_deleting_the_data_dir_keeps_the_users_own_line(two: dict[str, Any]) -> None:
    assert _inst(two, "A")[0] == 0
    shutil.rmtree(two["home"])
    res = _run_line(two, "A")  # the original runs directly (its own trailing newline)
    assert (res.returncode, res.stdout) == (0, "ACCOUNT-A\n")
    code, out = _uninst(two, "A")
    assert code == 0, out
    assert "restoring the original command kept in settings.json" in out
    line = json.loads(two["A"].read_text())["statusLine"]
    assert line == {"type": "command", "command": "echo ACCOUNT-A", "refreshInterval": 2}


def test_embedded_original_round_trips_awkward_commands(tmp_path: Path) -> None:
    for original in (
        'bash "$HOME/.claude/statusline.sh"',
        "printf 'a\\nb' # comment ; fi",
        'echo "it\'s" $(date)',
        None,
    ):
        cmd = install.wrapper_command(
            tmp_path / "w" / install.WRAPPER_NAME, user_home=tmp_path, original=original
        )
        assert install.embedded_original(cmd) == original
        assert install.wrapper_path_of(cmd, user_home=tmp_path) == (
            tmp_path / "w" / install.WRAPPER_NAME
        )
    res = subprocess.run(
        [
            "/bin/sh",
            "-c",
            install.wrapper_command(tmp_path / "gone.sh", original="printf 'a\\nb' # c; fi"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert (res.returncode, res.stdout) == (0, "a\nb")


# ------------------------------------------------------------ finding 16: OS errors


def test_a_read_only_settings_target_is_reported_cleanly(two: dict[str, Any]) -> None:
    ro = two["tmp"] / "nix-store"
    ro.mkdir()
    real = ro / "settings.json"
    real.write_bytes(two["A"].read_bytes())
    two["A"].unlink()
    two["A"].symlink_to(real)
    ro.chmod(0o555)
    try:
        code, out = _inst(two, "A")
    finally:
        ro.chmod(0o755)
    assert code == install.EXIT_ERROR
    assert "\ngpu statusline: cannot write to" in out
    assert "Traceback" not in out
    assert not (two["home"] / "statusline").exists()  # nothing left behind


def test_an_os_error_mid_install_rolls_the_record_back(
    two: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    real = install._atomic_write

    def flaky(path: Path, data: bytes, *, mode: int) -> None:
        if path.name == "settings.json":
            raise PermissionError(13, "Permission denied", str(path))
        real(path, data, mode=mode)

    monkeypatch.setattr(install, "_atomic_write", flaky)
    before = two["A"].read_bytes()
    code, out = _inst(two, "A")
    assert code == install.EXIT_ERROR
    assert "cannot write" in out
    assert "nothing changed in settings.json" in out
    assert two["A"].read_bytes() == before
    assert not install.record_dir(two["home"], two["A"]).exists()


# ------------------------------------------------------------ finding 13: bounded wait


def _wrapper(tmp_path: Path, gpu_script: str) -> tuple[subprocess.CompletedProcess[str], float]:
    gpu = tmp_path / "slowgpu"
    gpu.write_text(gpu_script)
    gpu.chmod(0o755)
    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(tmp_path),
        "GPU_ROUTER_BIN": str(gpu),
        "GPU_ROUTER_HOME": str(tmp_path / "h"),
        "GPU_STATUSLINE_ORIGINAL": 'printf "user line 1\\nuser line 2"',
    }
    t0 = time.monotonic()
    res = subprocess.run(
        ["/bin/bash", str(WRAPPER)],
        input="{}",
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
        check=False,
    )
    return res, time.monotonic() - t0


def test_a_stuck_gpu_never_holds_the_users_lines(tmp_path: Path) -> None:
    marker = tmp_path / "still-running"
    res, took = _wrapper(
        tmp_path,
        f"#!/bin/sh\ncat >/dev/null\nsleep 5\ntouch {marker}\nprintf 'gpu row'\n",
    )
    assert res.returncode == 0
    assert res.stdout == "user line 1\nuser line 2"
    assert took < 3.0, took
    time.sleep(5)
    assert not marker.exists()  # the stuck gpu was stopped, not left running


def test_a_fast_gpu_still_adds_its_rows(tmp_path: Path) -> None:
    res, took = _wrapper(tmp_path, "#!/bin/sh\ncat >/dev/null\nprintf 'gpu row 1\\ngpu row 2\\n'\n")
    assert res.stdout == "user line 1\nuser line 2\ngpu row 1\ngpu row 2"
    assert took < 2.0


# ------------------------------------------------------------ finding 6: control chars


def test_control_characters_never_reach_the_terminal(store: Store, clock: FakeClock) -> None:
    evil = "x\x1b]52;c;aGVsbG8=\x07pwn"
    job, _ = store.create_job(
        JobSpec(project_dir="/proj", script="train.py", name=evil), actor="agent"
    )
    assert "\x1b" not in job.name  # dropped at the spec
    att = _running(store, job.id)
    store.update_job(
        job.id,
        JobPatch(last_metrics={"\x1b]0;evil\x07m": 1.0}, progress_step=5, progress_total=10),
    )
    del att
    snap = build_snapshot(
        store=store, provider_summaries=[], session_caps={}, now=clock.now(), daemon_pid=1
    )
    text = "\n".join(fast.rows_text(snap, now=clock.now(), pid_alive=lambda _p: True))
    assert "\x1b]" not in text
    assert "\x07" not in text
    # the renderer strips them too (a file from an older daemon)
    raw: dict[str, Any] = json.loads(json.dumps(snap))
    raw["active"][0]["name"] = evil
    raw["active"][0]["metric"] = {"name": "\x1b]0;t\x07", "value": 1.0, "trend": None}
    text = "\n".join(fast.rows_text(raw, now=clock.now(), pid_alive=lambda _p: True))
    assert "\x1b]" not in text
    assert "\x07" not in text


# ------------------------------------------------------------ finding 12: resumed ETA


def _running(
    store: Store, job_id: str, *, from_state: JobState = JobState.QUEUED, **place: Any
) -> str:
    if from_state is JobState.QUEUED:
        store.transition(
            job_id,
            from_state=JobState.QUEUED,
            to_state=JobState.ROUTING,
            reason=Reason.ROUTING_STARTED,
            message="routing",
            actor="engine",
        )
        from_state = JobState.ROUTING
    _, att = store.place(
        job_id,
        from_state=from_state,
        provider="kaggle",
        gpu="T4",
        route_reason="r",
        message="placed",
        **place,
    )
    store.record_submission(att.id, remote_id=f"r{att.n}", remote_url=None, remote_meta={})
    store.transition(
        job_id,
        from_state=JobState.PROVISIONING,
        to_state=JobState.RUNNING,
        reason=Reason.STARTED,
        message="running",
        actor="engine",
        attempt=AttemptChange(att.id, AttemptPatch(state=AttemptState.RUNNING)),
    )
    return att.id


def test_eta_of_a_resumed_attempt_uses_its_own_rate(
    store: Store, clock: FakeClock, tmp_path: Path
) -> None:
    metrics = tmp_path / "metrics.jsonl"

    def snap_row() -> dict[str, Any]:
        s = build_snapshot(
            store=store,
            provider_summaries=[],
            session_caps={},
            now=clock.now(),
            daemon_pid=1,
            metrics_path=lambda _j: metrics,
        )
        return dict(s["active"][0])

    def point(attempt: int, step: int) -> None:
        with metrics.open("a") as fh:
            fh.write(
                json.dumps(
                    {"ts": clock.now(), "attempt": attempt, "step": step, "metrics": {"loss": 1}}
                )
                + "\n"
            )

    job, _ = store.create_job(JobSpec(project_dir="/p", script="t.py"), actor="api")
    att1 = _running(store, job.id)
    for step in range(0, 5001, 500):  # 10 s per step
        store.update_job(job.id, JobPatch(progress_step=step, progress_total=10_000))
        point(1, step)
        if step < 5000:
            clock.advance(5000)
    first = snap_row()
    assert first["eta_s"] == pytest.approx(50_000, rel=0.01)
    store.record_checkpoint(
        job.id,
        att1,
        seq=4,
        uri="x://c4",
        step=5000,
        size_bytes=None,
        sha256=None,
        created_at=clock.now(),
    )
    store.transition(
        job.id,
        from_state=JobState.RUNNING,
        to_state=JobState.MIGRATING,
        reason=Reason.SESSION_LOST,
        message="lost",
        actor="engine",
        attempt=AttemptChange(att1, AttemptPatch(state=AttemptState.LOST)),
    )
    _running(store, job.id, from_state=JobState.MIGRATING, resume_checkpoint_id=f"{job.id}.c4")
    clock.advance(30)
    # still at job-level step 5000 from attempt 1: no ETA rather than "0:01 left"
    assert snap_row()["eta_s"] is None
    for _ in range(36):  # one hour at the same 10 s/step
        clock.advance(100)
        step = store.get_job(job.id).progress.step or 0
        store.update_job(job.id, JobPatch(progress_step=step + 10))
        point(2, step + 10)
    row = snap_row()
    assert store.get_job(job.id).progress.step == 5360
    assert row["eta_s"] == pytest.approx((10_000 - 5360) * 10, rel=0.02)  # ~12.9 h
    text = fast.rows_text(
        {
            **build_snapshot(
                store=store,
                provider_summaries=[],
                session_caps={},
                now=clock.now(),
                daemon_pid=1,
                metrics_path=lambda _j: metrics,
            )
        },
        now=clock.now(),
        color=False,
        pid_alive=lambda _p: True,
    )[0]
    assert " 12:54 " in text or " 12:53 " in text  # 10h+ drops the word "left"


# ------------------------------------------------------------ finding 17: wall heartbeat


async def test_heartbeat_follows_wall_time_after_a_sleep(tmp_path: Path) -> None:
    """The loop clock stops while a Mac sleeps; the writer rewrites once WALL time says
    the heartbeat is due, within one short tick."""
    wall = [1000.0]

    def build() -> StateSnapshot:
        return StateSnapshot(
            schema=1,
            written_at=wall[0],
            daemon_pid=1,
            active=[{"id": "x"}],  # type: ignore[list-item]
            recent=[],
            counts={},
            providers=[],
        )

    writer = StateFileWriter(
        tmp_path / "state.json", build, min_interval_s=0.0, heartbeat_s=60.0, wall=lambda: wall[0]
    )
    import gpu_router.statefile as sf

    old_tick = sf.HEARTBEAT_TICK_S
    sf.HEARTBEAT_TICK_S = 0.05
    task = asyncio.create_task(writer.run())
    try:
        writer.mark_dirty()
        await asyncio.sleep(0.2)
        first = writer.writes
        assert first == 1  # wall time has not moved: no heartbeat yet
        wall[0] += 8 * 3600  # the lid was closed overnight (loop time barely moved)
        await asyncio.sleep(0.2)
        assert writer.writes == first + 1
    finally:
        sf.HEARTBEAT_TICK_S = old_tick
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
