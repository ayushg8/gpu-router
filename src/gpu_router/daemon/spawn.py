"""Start the daemon in the background and wait for it (phase 2).

Used by `gpu daemon start` and by every CLI command that needs the daemon when none is
running (auto-start). Lightweight imports only (client, paths); no typer.

How it starts: when the launchd agent is installed AND this process uses the default data
dir (no GPU_ROUTER_HOME), `launchctl kickstart` the agent so launchd owns the process;
otherwise spawn `python -m gpu_router daemon run` detached (new session, stdin closed,
stdout/stderr appended to logs/launchd.log) with this process's environment, so the child
uses the same GPU_ROUTER_HOME / GPU_ROUTER_PORT / GPU_ROUTER_TEST_MODE. The daemon's
single-instance lock makes a racing double start harmless.

The spawned child is watched while waiting: if it exits with "already running" (exit 3,
another CLI's daemon won the race) we keep waiting for that daemon and report
started=False; any other early exit (port taken, bad config) fails right away with the
`gpu: ...` line it wrote to launchd.log. `started` is True only when the daemon that
answers is the process this call spawned (or launchd was kickstarted).

Set GPU_ROUTER_NO_AUTOSTART=1 to make CLI commands fail with DaemonUnavailable instead
of starting one (scripts that must not spawn processes, some tests).
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass

from gpu_router.client import DEFAULT_TIMEOUT_S, GpuClient, read_runtime_info
from gpu_router.errors import DaemonUnavailable, GpuRouterError, NotReady
from gpu_router.paths import ENV_HOME, Paths

ENV_NO_AUTOSTART = "GPU_ROUTER_NO_AUTOSTART"
DEFAULT_START_WAIT_S = 20.0


@dataclass(frozen=True)
class Connected:
    client: GpuClient
    started: bool  # True if this call started the daemon
    pid: int
    port: int


def autostart_enabled() -> bool:
    return os.environ.get(ENV_NO_AUTOSTART, "").strip() not in {"1", "true", "yes"}


def _try_connect(paths: Paths) -> tuple[GpuClient | None, bool]:
    """(ready client or None, whether a daemon process answered at all)."""
    try:
        client = GpuClient.from_env(paths, timeout_s=2.0)
    except DaemonUnavailable:
        return None, False
    try:
        health = client.health()
    except GpuRouterError:
        client.close()
        return None, False
    if health.ready:
        client.close()
        client.timeout_s = DEFAULT_TIMEOUT_S
        return client, True
    client.close()
    return None, True


#: `gpu daemon run` exit code when another daemon already holds the lock (daemon/__main__).
_EXIT_ALREADY_RUNNING = 3


@dataclass
class Spawned:
    how: str  # "launchd" or "process"
    proc: subprocess.Popen[bytes] | None = None
    log_offset: int = 0  # launchd.log size before the child started


def spawn(paths: Paths) -> Spawned:
    """Start the daemon without waiting. Returns how, plus the child process to watch."""
    from gpu_router.daemon import launchd

    if ENV_HOME not in os.environ and launchd.is_installed() and launchd.kickstart():
        return Spawned("launchd")
    paths.home.mkdir(mode=0o700, parents=True, exist_ok=True)
    paths.logs_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    with open(paths.launchd_log, "ab") as log:
        offset = log.tell()
        proc = subprocess.Popen(
            [sys.executable, "-m", "gpu_router", "daemon", "run"],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            start_new_session=True,
            close_fds=True,
        )
    return Spawned("process", proc, offset)


def _early_exit_reason(paths: Paths, offset: int) -> str | None:
    """The first `gpu: ...` line the child wrote to launchd.log (after `offset`)."""
    try:
        with open(paths.launchd_log, "rb") as fh:
            fh.seek(offset)
            text = fh.read(64 * 1024).decode("utf-8", "replace")
    except OSError:
        return None
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for ln in lines:
        if ln.startswith("gpu: "):
            return ln.removeprefix("gpu: ")
    return lines[-1] if lines else None


def connect(
    paths: Paths | None = None,
    *,
    start: bool = True,
    wait_s: float = DEFAULT_START_WAIT_S,
    on_start: Callable[[], None] | None = None,
) -> Connected:
    """A client for a ready daemon, starting one if needed (and allowed).

    `on_start` is called once, right before spawning, so the CLI can tell the user.
    A daemon that is up but still recovering jobs is waited for (up to wait_s).
    Raises DaemonUnavailable (not running and not allowed to start, or not ready in time).
    """
    p = paths or Paths.from_env()
    client, alive = _try_connect(p)
    spawned: Spawned | None = None
    if client is None:
        if not alive:
            if not start or not autostart_enabled():
                raise DaemonUnavailable(
                    "the gpu-router daemon is not running",
                    hint="start it with `gpu daemon start` (or `gpu daemon install-launchd` "
                    "to run it at login)",
                )
            if on_start is not None:
                on_start()
            spawned = spawn(p)
        deadline = time.monotonic() + wait_s
        child_done = False
        while client is None and time.monotonic() < deadline:
            time.sleep(0.1)
            client, alive = _try_connect(p)
            proc = spawned.proc if spawned is not None else None
            if client is not None or proc is None or child_done:
                continue
            rc = proc.poll()
            if rc is None:
                continue
            child_done = True
            if rc != _EXIT_ALREADY_RUNNING:
                reason = _early_exit_reason(p, spawned.log_offset) if spawned else None
                raise DaemonUnavailable(
                    f"the daemon exited right away (exit {rc})" + (f": {reason}" if reason else ""),
                    hint=f"see {p.launchd_log}",
                    detail={"exit_code": rc},
                )
            # another CLI's daemon holds the lock: keep waiting for it to become ready
        if client is None:
            if alive:
                raise NotReady(
                    f"the daemon is still recovering jobs after {wait_s:g}s",
                    hint="try again in a moment; `gpu daemon status` shows when it is ready",
                )
            raise DaemonUnavailable(
                f"started the daemon but it did not answer within {wait_s:g}s",
                hint=f"see {p.launchd_log} and {p.daemon_log}",
            )
    info = read_runtime_info(p)
    pid = info.pid if info else 0
    port = info.port if info else int(client.base_url.rsplit(":", 1)[1])
    started = False
    if spawned is not None:
        # launchd owns its child (pid unknown here); a spawned process must be the one
        # answering, else another CLI started it first
        started = spawned.proc is None or spawned.proc.pid == pid
    return Connected(client=client, started=started, pid=pid, port=port)
