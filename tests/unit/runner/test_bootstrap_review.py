"""Phase-2 review fixes for bootstrap.py: checkpoint safety (no hard links, no torn or
unrestorable archives, bounded disk), protocol lines behind tqdm bars, leftover processes,
reused workdirs, relative dirs and an EXIT file for every outcome."""

from __future__ import annotations

import gzip
import os
import signal
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path

import pytest

from gpu_router import protocol
from gpu_router.packaging import build_bundle
from gpu_router.paths import Paths
from gpu_router.runner import bootstrap
from tests.unit.runner.test_bootstrap import BOOTSTRAP, _events, _run


def _script(tmp_path: Path, paths: Paths, body: str, name: str = "proj") -> Path:
    from gpu_router.models import JobSpec
    from tests.unit.packaging.helpers import make_project

    project = make_project(tmp_path / name, {"train.py": body})
    spec = JobSpec(project_dir=str(project), script="train.py")
    return build_bundle(project, spec, paths=paths).archive


# --------------------------------------------------------------------------- resume copies


def test_resume_never_hard_links_the_source_checkpoint(tmp_path: Path, paths: Paths) -> None:
    """Review finding: resume seeding hard-linked prev/last.pt -> resume/ -> checkpoints/,
    so an in-place torch.save killed mid-write truncated every copy, source included."""
    prev = tmp_path / "prev_ckpt"
    prev.mkdir()
    (prev / "last.pt").write_text("GOOD step 5000")
    body = (
        "import os, signal, gpu\n"
        "f = open(gpu.checkpoint_dir() / 'last.pt', 'wb')\n"
        "f.write(b'half')\n"
        "f.flush()\n"
        "os.kill(os.getpid(), signal.SIGKILL)\n"
    )
    archive = _script(tmp_path, paths, body)
    work = tmp_path / "work"
    proc = _run(
        "--bundle", str(archive), "--workdir", str(work), "--skip-install",
        "--resume", str(prev),
    )  # fmt: skip
    assert proc.returncode == 137, proc.stdout
    assert (prev / "last.pt").read_text() == "GOOD step 5000"
    assert (work / "resume" / "last.pt").read_text() == "GOOD step 5000"
    assert (work / "checkpoints" / "last.pt").read_bytes() == b"half"
    assert os.stat(prev / "last.pt").st_nlink == 1


# --------------------------------------------------------------------------- torn checkpoints


def test_final_sync_skipped_after_kill_mid_save(tmp_path: Path, paths: Paths) -> None:
    """Review finding: the exit-time sync archived a 0-byte last.pt after an OOM kill and
    published it as the newest checkpoint."""
    body = (
        "import os, signal, gpu\n"
        "f = open(gpu.checkpoint_dir() / 'last.pt', 'wb')\n"
        "f.flush()\n"
        "os.kill(os.getpid(), signal.SIGKILL)\n"
    )
    archive = _script(tmp_path, paths, body)
    work = tmp_path / "work"
    sync = tmp_path / "sync"
    proc = _run(
        "--bundle", str(archive), "--workdir", str(work), "--skip-install",
        "--checkpoint-sync-dir", str(sync), "--ckpt-seq-start", "4",
    )  # fmt: skip
    assert proc.returncode == 137
    assert not [e for e in _events(proc.stdout) if e.t in ("ckpt_begin", "ckpt_end")]
    assert "may be half-written" in proc.stdout
    assert not list(sync.glob("ckpt-*")) if sync.exists() else True


def _syncer(tmp_path: Path) -> tuple[bootstrap.CheckpointSyncer, Path, Path, list[str]]:
    ckpt = tmp_path / "ck"
    ckpt.mkdir()
    sync = tmp_path / "sync"
    lines: list[str] = []

    class Tee:
        def event(self, t: str, **fields: object) -> None:
            lines.append(t)

        def say(self, text: str) -> None:
            lines.append("say: " + text)

    return bootstrap.CheckpointSyncer(Tee(), ckpt, sync, 1, 60.0), ckpt, sync, lines


def _age(path: Path, seconds: float) -> None:
    t = path.stat().st_mtime - seconds  # the file was just written: its mtime is ~now
    os.utime(path, (t, t))


def test_sync_waits_for_files_to_settle(tmp_path: Path) -> None:
    syncer, ckpt, _sync, lines = _syncer(tmp_path)
    (ckpt / "last.pt").write_text("being written")
    assert syncer.sync() is None  # fresh file: postponed, no seq used
    assert syncer.due()  # retried on the next tick
    assert lines == []
    _age(ckpt / "last.pt", 5)
    assert syncer.sync() == 1
    assert lines == ["ckpt_begin", "ckpt_end"]
    assert not syncer.due()


def test_sync_drops_archive_when_files_change_while_archiving(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    syncer, ckpt, sync, lines = _syncer(tmp_path)
    (ckpt / "last.pt").write_text("v1")
    _age(ckpt / "last.pt", 5)
    real = bootstrap._fingerprint
    calls = {"n": 0}

    def changing(path: Path) -> tuple[tuple[str, int, int], ...]:
        calls["n"] += 1
        fp = real(path)
        return fp if calls["n"] == 1 else (*fp, ("new.pt", 1, 1))

    monkeypatch.setattr(bootstrap, "_fingerprint", changing)
    assert syncer.sync() is None
    assert lines == []  # no ckpt_begin for an archive that is thrown away
    assert not list(sync.iterdir())  # tmp removed
    assert syncer.seq == 0  # seq not consumed


def test_sync_keeps_newest_archives_and_compresses_fast(tmp_path: Path) -> None:
    syncer, ckpt, sync, _lines = _syncer(tmp_path)
    for i in range(6):
        (ckpt / "last.pt").write_bytes(os.urandom(1000) + bytes([i]))
        _age(ckpt / "last.pt", 5 + i)
        assert syncer.sync() == i + 1
    names = sorted(p.name for p in sync.iterdir())
    assert names == ["ckpt-0004.tar.gz", "ckpt-0005.tar.gz", "ckpt-0006.tar.gz"]
    raw = (sync / "ckpt-0006.tar.gz").read_bytes()
    assert raw[8] == 4  # gzip XFL: fastest compression (level 1), not -9
    with gzip.open(sync / "ckpt-0006.tar.gz") as fh:
        assert fh.read(1)


def test_periodic_sync_runs_off_the_tick_thread(tmp_path: Path) -> None:
    syncer, _ckpt, _sync, _lines = _syncer(tmp_path)
    seen: list[str] = []
    done = threading.Event()

    def slow_sync(final: bool = False, min_age_s: float | None = None) -> None:
        seen.append(threading.current_thread().name)
        done.wait(5)

    syncer.sync = slow_sync  # type: ignore[method-assign]
    start = time.monotonic()
    syncer.start_background()
    syncer.start_background()  # no second sync while one runs
    assert time.monotonic() - start < 1  # the caller (tick thread) is not blocked
    done.set()
    syncer.wait(5)
    assert seen == ["gpu-ckpt-sync"]


def test_symlinked_checkpoint_is_archived_as_a_file_and_restores(
    tmp_path: Path, paths: Paths
) -> None:
    """Review finding: `last.pt -> /abs/epoch3.pt` was archived as a link and every later
    restore failed with 'unsafe link in archive'."""
    body = (
        "import os, gpu\n"
        "d = gpu.checkpoint_dir()\n"
        "(d / 'epoch3.pt').write_text('weights 3')\n"
        "os.symlink(str((d / 'epoch3.pt').resolve()), str(d / 'last.pt'))\n"
    )
    archive = _script(tmp_path, paths, body)
    work = tmp_path / "work"
    sync = tmp_path / "sync"
    proc = _run(
        "--bundle", str(archive), "--workdir", str(work), "--skip-install",
        "--checkpoint-sync-dir", str(sync),
    )  # fmt: skip
    assert proc.returncode == 0, proc.stdout
    restored = tmp_path / "restored"
    bootstrap.extract(sync / "ckpt-0001.tar.gz", restored)
    assert not (restored / "last.pt").is_symlink()
    assert (restored / "last.pt").read_text() == "weights 3"


# --------------------------------------------------------------------------- tqdm glue


def test_protocol_lines_behind_a_tqdm_bar_are_parsed(tmp_path: Path, paths: Paths) -> None:
    """Review finding: `\\r<bar>` on stderr + `::gpu::` on stdout became one line
    `<bar>::gpu:: {...}`, so no metric was ever parsed."""
    body = (
        "import sys, gpu\n"
        "gpu.total_steps(5)\n"
        "for i in range(1, 6):\n"
        "    sys.stderr.write('\\r%3d%%|###   | %d/5' % (i * 20, i))\n"
        "    sys.stderr.flush()\n"
        "    gpu.log(step=i, loss=1.0 / i)\n"
        "sys.stderr.write('\\n')\n"
    )
    archive = _script(tmp_path, paths, body)
    proc = _run("--bundle", str(archive), "--workdir", str(tmp_path / "w"), "--skip-install")
    assert proc.returncode == 0, proc.stdout
    metrics = [e for e in _events(proc.stdout) if e.t == "metric"]
    assert [m.step for m in metrics] == [1, 2, 3, 4, 5]
    assert not [ln for ln in proc.stdout.splitlines() if protocol.PREFIX in ln[1:]]


# --------------------------------------------------------------------------- process group


def test_leftover_grandchild_does_not_hold_the_run_open(tmp_path: Path, paths: Paths) -> None:
    pid_file = tmp_path / "grandchild.pid"
    body = (
        "import subprocess\n"
        f"p = subprocess.Popen(['sleep', '30'])\n"
        f"open({str(pid_file)!r}, 'w').write(str(p.pid))\n"
        "print('main done')\n"
    )
    archive = _script(tmp_path, paths, body)
    start = time.monotonic()
    proc = _run("--bundle", str(archive), "--workdir", str(tmp_path / "w"), "--skip-install")
    took = time.monotonic() - start
    assert proc.returncode == 0, proc.stdout
    assert took < bootstrap.DRAIN_GRACE_S + 6, took
    assert "stopped processes the job left running" in proc.stdout
    pid = int(pid_file.read_text())
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        pytest.fail(f"grandchild {pid} still alive")


def test_sigterm_reaches_the_whole_group_and_writes_exit(tmp_path: Path, paths: Paths) -> None:
    pid_file = tmp_path / "grandchild.pid"
    body = (
        "import subprocess, time\n"
        "p = subprocess.Popen(['sleep', '60'])\n"
        f"open({str(pid_file)!r}, 'w').write(str(p.pid))\n"
        "print('ready', flush=True)\n"
        "time.sleep(60)\n"
    )
    archive = _script(tmp_path, paths, body)
    work = tmp_path / "w"
    env = {k: v for k, v in os.environ.items() if not k.startswith(("GPU_", "PYTHONPATH"))}
    runner = subprocess.Popen(
        [sys.executable, str(BOOTSTRAP), "--bundle", str(archive), "--workdir", str(work),
         "--skip-install"],
        stdout=subprocess.PIPE, text=True, env=env,
    )  # fmt: skip
    assert runner.stdout is not None
    for line in runner.stdout:
        if line.strip() == "ready":
            break
    runner.send_signal(signal.SIGTERM)
    runner.communicate(timeout=30)
    assert runner.returncode == 128 + signal.SIGTERM
    assert (work / "EXIT").read_text() == f"{128 + signal.SIGTERM}\n"
    pid = int(pid_file.read_text())
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        pytest.fail(f"grandchild {pid} survived SIGTERM")


# --------------------------------------------------------------------------- reused workdir


def test_reused_workdir_runs_the_new_code_and_forgets_the_old_job(
    tmp_path: Path, paths: Paths
) -> None:
    """Review finding: bundle B in job A's workdir ran A's code, saw A's EXIT and loaded
    A's checkpoint."""
    body_a = (
        "import gpu\n"
        "print('running code version A')\n"
        "(gpu.checkpoint_dir() / 'a.pt').write_text('A')\n"
        "(gpu.output_dir() / 'a.txt').write_text('A')\n"
    )
    body_b = (
        "import os, gpu\n"
        "print('running code version B')\n"
        "print('stale EXIT=%s' % os.path.exists(os.path.join(WORKDIR, 'EXIT')))\n"
        "print('latest=%s' % gpu.latest_checkpoint())\n"
        "print('outputs=%s' % sorted(p.name for p in gpu.output_dir().iterdir()))\n"
    )
    work = tmp_path / "work"
    a = _script(tmp_path, paths, body_a, "proj_a")
    b = _script(tmp_path, paths, body_b.replace("WORKDIR", repr(str(work))), "proj_b")
    first = _run(
        "--bundle", str(a), "--workdir", str(work), "--skip-install",
        env={"GPU_ROUTER_JOB_ID": "aaaaaaaaaaaa"},
    )  # fmt: skip
    assert first.returncode == 0, first.stdout
    second = _run(
        "--bundle", str(b), "--workdir", str(work), "--skip-install",
        env={"GPU_ROUTER_JOB_ID": "bbbbbbbbbbbb"},
    )  # fmt: skip
    assert second.returncode == 0, second.stdout
    assert "running code version B" in second.stdout
    assert "stale EXIT=False" in second.stdout
    assert "latest=None" in second.stdout
    assert "outputs=[]" in second.stdout
    log = (work / "job.log").read_text()
    assert "version A" not in log
    assert "version B" in log
    assert "version A" in (work / "job.log.prev").read_text()
    # same job again (a restart on the same machine) keeps its checkpoints
    third = _run(
        "--bundle", str(a), "--workdir", str(work), "--skip-install",
        env={"GPU_ROUTER_JOB_ID": "aaaaaaaaaaaa"},
    )  # fmt: skip
    assert third.returncode == 0
    assert "running code version A" in third.stdout


# --------------------------------------------------------------------------- env handling


def test_relative_dirs_resolve_against_the_runner_cwd(tmp_path: Path, paths: Paths) -> None:
    body = (
        "import gpu\n"
        "(gpu.output_dir() / 'result.txt').write_text('ok')\n"
        "(gpu.checkpoint_dir() / 'c.pt').write_text('c')\n"
    )
    archive = _script(tmp_path, paths, body)
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    env = {k: v for k, v in os.environ.items() if not k.startswith(("GPU_", "PYTHONPATH"))}
    env.update({"GPU_OUTPUT_DIR": "out", "GPU_CHECKPOINT_DIR": "ck"})
    proc = subprocess.run(
        [sys.executable, str(BOOTSTRAP), "--bundle", str(archive), "--workdir", "w",
         "--skip-install", "--checkpoint-sync-dir", "sync"],
        cwd=cwd, env=env, capture_output=True, text=True, timeout=120, check=False,
    )  # fmt: skip
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (cwd / "out" / "result.txt").read_text() == "ok"
    assert (cwd / "ck" / "c.pt").read_text() == "c"
    with tarfile.open(cwd / "sync" / "ckpt-0001.tar.gz") as tar:
        assert tar.getnames() == ["c.pt"]
    assert (cwd / "w" / "EXIT").read_text() == "0\n"


def test_bad_env_numbers_fall_back_to_defaults(tmp_path: Path, paths: Paths) -> None:
    archive = _script(tmp_path, paths, "print('hi')\n")
    work = tmp_path / "w"
    proc = _run(
        "--bundle", str(archive), "--workdir", str(work), "--skip-install",
        env={"GPU_CKPT_SEQ_START": "", "GPU_HEARTBEAT_S": "soon"},
    )  # fmt: skip
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "ignored GPU_HEARTBEAT_S='soon'" in proc.stdout
    assert (work / "EXIT").read_text() == "0\n"


def test_bad_arguments_still_write_exit(tmp_path: Path) -> None:
    work = tmp_path / "w"
    proc = _run("--workdir", str(work), "--heartbeat-s", "abc")
    assert proc.returncode == 2
    assert (work / "EXIT").read_text() == "2\n"


def test_sigterm_outside_the_entrypoint_writes_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def interrupted(args: object, workdir: Path, tee: object) -> Path:
        raise bootstrap.Terminated(signal.SIGTERM)

    monkeypatch.setattr(bootstrap, "_prepare_bundle", interrupted)
    work = tmp_path / "w"
    code = bootstrap.run(["--bundle", str(tmp_path / "x.tar.gz"), "--workdir", str(work)])
    assert code == 128 + signal.SIGTERM
    assert (work / "EXIT").read_text() == f"{128 + signal.SIGTERM}\n"
    assert "stopped by signal 15" in (work / "job.log").read_text()
