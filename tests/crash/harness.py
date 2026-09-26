"""Crash-recovery harness (owner: group C).

Spawns a real `python -m gpu_router daemon run --port 0` on a private GPU_ROUTER_HOME with
GPU_ROUTER_TEST_MODE=1, talks to it through GpuClient, and can arm crash points
(GPU_ROUTER_CRASH_AT) so the daemon dies with exit code 137 mid-flight. The fake provider's
"remote" lives under <home>/fake/, outside the daemon, so it survives the kill.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gpu_router.client import GpuClient, read_runtime_info
from gpu_router.errors import DaemonUnavailable
from gpu_router.models import JobSpec, Source
from gpu_router.paths import Paths

FAST_PROVIDERS = """\
providers:
  fake: {poll_interval_s: 0.05}
  fake-b: {poll_interval_s: 0.05}
"""

FAST_CONFIG = """\
version: 1
engine:
  backoff_base_s: 0.1
  backoff_cap_s: 0.5
  health_recheck_s: 5
  cancel_timeout_s: 20
daemon:
  shutdown_grace_s: 2
"""


@dataclass
class DaemonProc:
    home: Path
    project: Path
    proc: subprocess.Popen[bytes] | None = None
    logs: list[Path] = field(default_factory=list)

    @property
    def paths(self) -> Paths:
        return Paths(self.home)

    def prepare(self) -> None:
        self.home.mkdir(parents=True, exist_ok=True)
        (self.home / "providers.yaml").write_text(FAST_PROVIDERS)
        (self.home / "config.yaml").write_text(FAST_CONFIG)
        self.project.mkdir(parents=True, exist_ok=True)

    def start(self, *, crash_at: str | None = None, timeout_s: float = 20) -> GpuClient:
        env = {k: v for k, v in os.environ.items() if not k.startswith("GPU_ROUTER_")}
        env.update({"GPU_ROUTER_HOME": str(self.home), "GPU_ROUTER_TEST_MODE": "1"})
        if crash_at:
            env["GPU_ROUTER_CRASH_AT"] = crash_at
        log = self.home / f"daemon-{len(self.logs)}.out"
        self.logs.append(log)
        with log.open("wb") as fh:
            self.proc = subprocess.Popen(
                [sys.executable, "-m", "gpu_router", "daemon", "run", "--port", "0"],
                env=env,
                stdout=fh,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise AssertionError(
                    f"daemon exited early ({self.proc.returncode}):\n" + log.read_text()
                )
            info = read_runtime_info(self.paths)
            if info is not None and info.pid == self.proc.pid:
                try:
                    client = GpuClient.from_env(self.paths, timeout_s=5)
                    if client.health().ready:
                        return client
                    client.close()
                except DaemonUnavailable:
                    pass
            time.sleep(0.05)
        raise AssertionError("daemon did not become ready:\n" + log.read_text())

    def wait_exit(self, timeout_s: float = 20) -> int:
        assert self.proc is not None
        try:
            return self.proc.wait(timeout_s)
        except subprocess.TimeoutExpired:
            raise AssertionError("daemon did not exit:\n" + self.logs[-1].read_text()) from None

    def kill(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait(10)

    def spec(self, fake: dict[str, Any], **fields: Any) -> JobSpec:
        return JobSpec(
            project_dir=str(self.project),
            script="train.py",
            source=Source.API,
            provider_options={"fake": fake},
            **fields,
        )

    def fake_runs(self, provider: str = "fake") -> list[dict[str, Any]]:
        runs = self.home / "fake" / provider / "runs"
        if not runs.exists():
            return []
        return [json.loads(p.read_text()) for p in sorted(runs.glob("*/run.json"))]


def wait_for(pred: Callable[[], bool], *, timeout_s: float = 30, what: str = "condition") -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if pred():
            return
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {what}")
