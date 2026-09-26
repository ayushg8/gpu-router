"""Shell test fixtures: a real daemon running IN this process (uvicorn on its own thread and
event loop, a real DaemonRuntime on the fake providers, real HTTP on 127.0.0.1:<port 0>),
and a GpuShell driven by Textual's pilot against it.

The daemon writes daemon.json with this process's pid, so GpuClient.from_env and the
shell's spawn.connect find it exactly as they find a launchd daemon. Logging is not
configured (DaemonRuntime.create(configure_logging=False)), so nothing leaks into other
tests. Auto-start is disabled for every shell test (GPU_ROUTER_NO_AUTOSTART=1): a test
that sees "daemon down" must not spawn a real one.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import threading
import time
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from gpu_router.client import GpuClient
from gpu_router.models import JobSpec, Source
from gpu_router.paths import ENV_HOME, Paths
from tests.crash.harness import FAST_CONFIG, FAST_PROVIDERS


@dataclass
class InProcDaemon:
    home: Path
    providers_yaml: str = FAST_PROVIDERS
    config_yaml: str = FAST_CONFIG
    server: Any = None
    thread: threading.Thread | None = None
    runtime: Any = None
    errors: list[BaseException] = field(default_factory=list)

    @property
    def paths(self) -> Paths:
        return Paths(self.home)

    def start(self, timeout_s: float = 20.0) -> None:
        """Build the runtime ON the daemon thread (the Store is bound to the thread that
        opened it, invariant 9) and serve until stop()."""
        self.home.mkdir(parents=True, exist_ok=True)
        (self.home / "providers.yaml").write_text(self.providers_yaml)
        (self.home / "config.yaml").write_text(self.config_yaml)
        self.thread = threading.Thread(target=self._serve, name="inproc-daemon", daemon=True)
        self.thread.start()
        paths = self.paths
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.errors:
                raise AssertionError(f"in-process daemon failed: {self.errors[0]!r}")
            server = self.server
            if server is not None and server.started and paths.runtime.exists():
                with GpuClient.from_env(paths, timeout_s=5) as c:
                    if c.health().ready:
                        return
            time.sleep(0.02)
        raise AssertionError("in-process daemon did not become ready")

    def _serve(self) -> None:
        import uvicorn

        from gpu_router.clock import SystemClock
        from gpu_router.config import load_config
        from gpu_router.daemon.app import create_app
        from gpu_router.daemon.runtime import DaemonRuntime
        from gpu_router.daemon.server import _bind, remove_runtime_info, write_runtime_info

        paths = self.paths
        try:
            env = {"GPU_ROUTER_TEST_MODE": "1", ENV_HOME: str(self.home)}
            config = load_config(paths, environ=env)
            runtime = DaemonRuntime.create(
                paths, config, SystemClock(), configure_logging=False, foreground=True
            )
            sock = _bind(0)
        except BaseException as exc:
            self.errors.append(exc)
            return
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
                test_mode=True,
            )
            try:
                yield
            finally:
                try:
                    await runtime.stop()
                finally:
                    remove_runtime_info(paths)

        app.router.lifespan_context = lifespan
        server = uvicorn.Server(
            uvicorn.Config(
                app,
                log_config=None,
                access_log=False,
                lifespan="on",
                timeout_graceful_shutdown=1,
                server_header=False,
            )
        )
        runtime.shutdown_hook = lambda: setattr(server, "should_exit", True)
        self.server, self.runtime = server, runtime
        try:
            asyncio.run(server.serve(sockets=[sock]))
        except BaseException as exc:
            self.errors.append(exc)
        finally:
            sock.close()
            if not runtime._stopped:
                with contextlib.suppress(Exception):
                    asyncio.run(runtime.stop())
            remove_runtime_info(paths)

    def stop(self) -> None:
        if self.server is not None:
            self.server.should_exit = True
        if self.thread is not None:
            self.thread.join(15)

    def client(self) -> GpuClient:
        return GpuClient.from_env(self.paths, timeout_s=10)

    def submit(self, project: Path, fake: dict[str, Any], **fields: Any) -> Any:
        fields.setdefault("source", Source.API)
        spec = JobSpec(
            project_dir=str(project),
            script="train.py",
            provider_options={"fake": fake},
            **fields,
        )
        with self.client() as c:
            return c.submit(spec)

    def job(self, ref: str) -> Any:
        with self.client() as c:
            return c.job(ref).job


def wait_for(pred: Callable[[], bool], *, timeout_s: float = 20, what: str = "condition") -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if pred():
            return
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {what}")


async def settle(
    pilot: Any, pred: Callable[[], bool], *, timeout_s: float = 20, what: str = "condition"
) -> None:
    """Let the app run (pilot.pause) until `pred()` holds."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if pred():
            return
        await pilot.pause(0.05)
    raise AssertionError(f"timed out waiting for {what}")


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    proj = tmp_path / "project"
    proj.mkdir()
    (proj / "train.py").write_text("print('hello from train.py')\n")
    (proj / "eval.py").write_text("print('eval')\n")
    (proj / "models").mkdir()
    (proj / "models" / "yolo.py").write_text("x = 1\n")
    monkeypatch.chdir(proj)
    return proj


@pytest.fixture
def daemon(gpu_home: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[InProcDaemon]:
    """A fresh in-process daemon on this test's tmp GPU_ROUTER_HOME (no jobs yet)."""
    monkeypatch.setenv("GPU_ROUTER_NO_AUTOSTART", "1")
    d = InProcDaemon(home=gpu_home)
    d.start()
    try:
        yield d
    finally:
        d.stop()


@pytest.fixture
def no_daemon(gpu_home: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """No daemon at all, and auto-start off: the shell must show the down state."""
    monkeypatch.setenv("GPU_ROUTER_NO_AUTOSTART", "1")
    gpu_home.mkdir(parents=True, exist_ok=True)
    return gpu_home


def write_gpu_yaml(project: Path, **fake: Any) -> None:
    body = ", ".join(f"{k}: {json.dumps(v)}" for k, v in fake.items())
    (project / "gpu.yaml").write_text(
        f"version: 1\nscript: train.py\nprovider_options:\n  fake: {{{body}}}\n"
    )
