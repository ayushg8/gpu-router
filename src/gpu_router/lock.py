"""Single-instance daemon lock (phase 1; owner: group A).

`fcntl.flock(LOCK_EX | LOCK_NB)` on `<home>/daemon.lock`, held for the daemon's lifetime.
The kernel releases it when the process dies for any reason (including SIGKILL), so there
is no stale-pidfile problem. The file's content (the holder's pid) is informational only.

The lock is acquired BEFORE the database is opened, and `Store.open` requires a held
`InstanceLock` (invariant 1): two daemons can never share gpu.db, and no client can open
it by accident.
"""

from __future__ import annotations

import contextlib
import fcntl
import os
from types import TracebackType

from gpu_router.errors import DaemonAlreadyRunning
from gpu_router.paths import Paths


class InstanceLock:
    """Holds the daemon lock. Use `InstanceLock.acquire(paths)` or as a context manager."""

    def __init__(self, paths: Paths) -> None:
        self.paths = paths
        self._fd: int | None = None

    @classmethod
    def acquire(cls, paths: Paths) -> InstanceLock:
        """Take the lock or raise DaemonAlreadyRunning (message includes the holder's pid
        from the lock file when readable). Creates home via paths.ensure()."""
        paths.ensure()
        fd = os.open(paths.lock, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            pid = holder_pid(paths)
            who = f" (pid {pid})" if pid is not None else ""
            raise DaemonAlreadyRunning(
                f"another gpu-router daemon is already running{who}",
                hint="use `gpu daemon status`, or `gpu daemon stop` to stop it",
                detail={"pid": pid} if pid is not None else None,
            ) from None
        # Informational content only; the flock is the truth.
        with contextlib.suppress(OSError):
            os.ftruncate(fd, 0)
            os.lseek(fd, 0, os.SEEK_SET)
            os.write(fd, f"{os.getpid()}\n".encode())
            os.fsync(fd)
        lock = cls(paths)
        lock._fd = fd
        return lock

    @property
    def held(self) -> bool:
        return self._fd is not None

    def release(self) -> None:
        """Unlock and close. Idempotent. Does not delete the file (deleting would race)."""
        fd, self._fd = self._fd, None
        if fd is None:
            return
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    def __enter__(self) -> InstanceLock:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.release()


def holder_pid(paths: Paths) -> int | None:
    """Best-effort read of the pid recorded in daemon.lock (for messages and `daemon status`)."""
    try:
        text = paths.lock.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    first = text.split()[0] if text else ""
    return int(first) if first.isdigit() else None
