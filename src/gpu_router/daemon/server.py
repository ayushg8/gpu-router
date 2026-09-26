"""Run the daemon (phase 1; owner: group C).

`run_daemon(paths, config, *, foreground, port)`:
1. DaemonRuntime.create(...) (DaemonAlreadyRunning -> exit code 3 in __main__).
2. Bind a socket to 127.0.0.1:<port> ourselves (port 0 = ephemeral) so the real port is
   known before uvicorn starts; runtime.port = bound port.
3. uvicorn.Server(Config(app, log_config=None, access_log=False, lifespan="on")) serving on
   that socket; lifespan startup awaits runtime.start(), shutdown awaits runtime.stop().
4. After startup completes, write daemon.json (api.RuntimeInfo, atomic, 0644); remove it on
   clean exit (a stale file after SIGKILL is fine: clients verify with GET /v1/health and
   the pid).
5. SIGTERM/SIGINT -> graceful stop within config.daemon.shutdown_grace_s. Remote runs are
   never touched (invariant 11). POST /v1/daemon/shutdown triggers the same path.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import socket
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gpu_router.config import Config
    from gpu_router.paths import Paths

HOST = "127.0.0.1"


def _bind(port: int) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((HOST, port))
    except OSError:
        sock.close()
        raise
    sock.listen(128)
    sock.setblocking(False)
    return sock


def run_daemon(
    paths: Paths, config: Config, *, foreground: bool = False, port: int | None = None
) -> int:
    """Blocking. Returns a process exit code."""
    import uvicorn

    from gpu_router.clock import SystemClock
    from gpu_router.daemon.app import create_app
    from gpu_router.daemon.runtime import DaemonRuntime

    runtime = DaemonRuntime.create(paths, config, SystemClock(), foreground=foreground)
    want = config.daemon.port if port is None else port
    try:
        sock = _bind(want)
    except OSError as exc:
        with contextlib.suppress(Exception):
            runtime.store.close()
        runtime.lock.release()
        raise OSError(f"cannot listen on {HOST}:{want}: {exc.strerror or exc}") from exc
    runtime.port = int(sock.getsockname()[1])
    app = create_app(runtime)

    @contextlib.asynccontextmanager
    async def lifespan(_app: Any) -> AsyncIterator[None]:
        await runtime.start()
        write_runtime_info(
            paths,
            pid=os.getpid(),
            port=runtime.port,
            started_at=runtime.started_at,
            test_mode=config.test_mode,
        )
        try:
            yield
        finally:
            try:
                await runtime.stop()
            finally:
                remove_runtime_info(paths)  # last step: "gone" means fully stopped

    app.router.lifespan_context = lifespan
    uv_config = uvicorn.Config(
        app,
        log_config=None,
        access_log=False,
        lifespan="on",
        timeout_graceful_shutdown=int(max(1, config.daemon.shutdown_grace_s)),
        server_header=False,
    )
    server = uvicorn.Server(uv_config)
    runtime.shutdown_hook = lambda: setattr(server, "should_exit", True)
    try:
        asyncio.run(server.serve(sockets=[sock]))
    finally:
        sock.close()
        # If startup failed before the lifespan cleanup ran, still release everything.
        if not runtime._stopped:
            with contextlib.suppress(Exception):
                asyncio.run(runtime.stop())
        remove_runtime_info(paths)
    return 0 if server.started else 1


def write_runtime_info(
    paths: Paths, *, pid: int, port: int, started_at: float, test_mode: bool
) -> None:
    from gpu_router import __version__
    from gpu_router.api import RuntimeInfo

    info = RuntimeInfo(
        pid=pid, port=port, version=__version__, started_at=started_at, test_mode=test_mode
    )
    data = info.model_dump_json().encode("utf-8") + b"\n"
    tmp = paths.runtime.with_name(f".{paths.runtime.name}.{os.getpid()}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        os.write(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)
    os.chmod(tmp, 0o644)
    os.replace(tmp, paths.runtime)


def remove_runtime_info(paths: Paths) -> None:
    """Remove daemon.json if it belongs to this process (never another daemon's file)."""
    try:
        from gpu_router.api import RuntimeInfo

        info = RuntimeInfo.model_validate_json(paths.runtime.read_bytes())
    except FileNotFoundError:
        return
    except Exception:  # unreadable/corrupt: it cannot help any client, drop it
        info = None
    if info is None or info.pid == os.getpid():
        with contextlib.suppress(FileNotFoundError):
            paths.runtime.unlink()
