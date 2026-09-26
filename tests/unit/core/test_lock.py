from __future__ import annotations

import os
import subprocess
import sys

import pytest

from gpu_router.errors import DaemonAlreadyRunning
from gpu_router.lock import InstanceLock, holder_pid
from gpu_router.paths import Paths


def test_acquire_release_reacquire(paths: Paths) -> None:
    lock = InstanceLock.acquire(paths)
    assert lock.held
    assert holder_pid(paths) == os.getpid()
    with pytest.raises(DaemonAlreadyRunning) as info:
        InstanceLock.acquire(paths)  # flock is per open file description: conflicts in-process
    assert str(os.getpid()) in info.value.message
    lock.release()
    lock.release()  # idempotent
    assert not lock.held
    assert paths.lock.exists()  # never deleted
    with InstanceLock.acquire(paths) as again:
        assert again.held
    assert not again.held


def test_lock_blocks_other_process_and_dies_with_it(paths: Paths) -> None:
    code = (
        "import sys, time\n"
        "from gpu_router.lock import InstanceLock\n"
        "from gpu_router.paths import Paths\n"
        "l = InstanceLock.acquire(Paths.from_env())\n"
        "print('locked', flush=True)\n"
        "time.sleep(60)\n"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", code], stdout=subprocess.PIPE, text=True, env=os.environ.copy()
    )
    try:
        assert proc.stdout is not None
        assert proc.stdout.readline().strip() == "locked"
        with pytest.raises(DaemonAlreadyRunning, match=f"pid {proc.pid}"):
            InstanceLock.acquire(paths)
        assert holder_pid(paths) == proc.pid
    finally:
        proc.kill()  # SIGKILL: the kernel releases the flock
        proc.wait()
    InstanceLock.acquire(paths).release()


def test_acquire_creates_home(gpu_home: object) -> None:
    p = Paths.from_env()
    assert not p.home.exists()
    with InstanceLock.acquire(p):
        assert p.lock.exists()
        assert p.lock.stat().st_mode & 0o777 == 0o600


def test_holder_pid_unreadable(paths: Paths) -> None:
    assert holder_pid(paths) is None
    paths.lock.write_text("garbage")
    assert holder_pid(paths) is None
