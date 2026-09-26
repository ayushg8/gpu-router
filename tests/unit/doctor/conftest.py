"""Doctor test harness (phase 8a): a ProbeEnv over tmp dirs with a scripted subprocess
runner, a fake `which`, a fake daemon client and a fake HF whoami. Nothing here touches a
real provider, the real Keychain, ~/.claude or the user's data dir (invariant 20)."""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from gpu_router.api import HealthView
from gpu_router.clock import FakeClock
from gpu_router.doctor.probe import CmdResult, DaemonInfo, ProbeEnv
from gpu_router.paths import Paths

WHOAMI_OK = """Auth provider: adc
Email:         me@example.com
Audience:      123.apps.googleusercontent.com
Expires in:    59m
Scopes:
  - https://www.googleapis.com/auth/cloud-platform
  - https://www.googleapis.com/auth/colaboratory
  - https://www.googleapis.com/auth/userinfo.email
  - openid
"""

COLAB_NEW_HELP = """
 Usage: colab new [OPTIONS]
╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --session   -s      <str>  Session name                                      │
│ --tpu               <str>  TPU accelerator variant. Supported: v5e1, v6e1.   │
│ --gpu               <str>  GPU accelerator variant. Supported: T4, L4, G4,   │
│                            H100, A100.                                       │
│                            If omitted (along with --tpu), a CPU runtime is   │
╰──────────────────────────────────────────────────────────────────────────────╯
"""


@dataclass
class FakeRun:
    """Scripted subprocesses: the first rule whose regex matches ' '.join(argv) answers."""

    rules: list[tuple[str, CmdResult | Callable[[list[str]], CmdResult]]] = field(
        default_factory=list
    )
    calls: list[list[str]] = field(default_factory=list)
    envs: list[dict[str, str] | None] = field(default_factory=list)

    def on(self, pattern: str, result: CmdResult | Callable[[list[str]], CmdResult]) -> FakeRun:
        self.rules.insert(0, (pattern, result))
        return self

    def __call__(
        self,
        argv: list[str],
        timeout: float,
        *,
        env: Any = None,
        cwd: str | None = None,
    ) -> CmdResult:
        self.calls.append(list(argv))
        self.envs.append(dict(env) if env is not None else None)
        line = " ".join(argv)
        for pattern, result in self.rules:
            if re.search(pattern, line):
                return result(argv) if callable(result) else result
        return CmdResult(None, "", "", error=f"{argv[0]} not found")


def ok(out: str = "", err: str = "") -> CmdResult:
    return CmdResult(0, out, err)


def default_run() -> FakeRun:
    return (
        FakeRun()
        .on(r"kaggle --version$", ok("Kaggle API 2.2.4"))
        .on(r"colab version$", ok("Version: 0.7.2"))
        .on(r"colab --auth=adc --config \S+ whoami$", ok(WHOAMI_OK))
        .on(r"colab --auth=adc --config \S+ new --help$", ok(COLAB_NEW_HELP))
        .on(r"git --version$", ok("git version 2.54.0"))
        .on(r"uv --version$", ok("uv 0.11.29 (Homebrew)"))
    )


class FakeClient:
    """Answers GpuClient.request like the daemon's /v1 endpoints (dicts)."""

    base_url = "http://127.0.0.1:47291"

    def __init__(
        self,
        providers: list[dict[str, Any]] | None = None,
        quota: list[dict[str, Any]] | None = None,
        health: dict[str, dict[str, Any]] | None = None,
        errors: dict[str, Exception] | None = None,
        infer_quota: list[dict[str, Any]] | None = None,
    ) -> None:
        self._providers = providers or []
        self._infer_quota = infer_quota or []
        self._quota = quota or []
        self._health = health or {}
        self._errors = errors or {}
        self.calls: list[tuple[str, str]] = []

    def request(self, method: str, path: str, **_kw: Any) -> Any:
        self.calls.append((method, path))
        if path in self._errors:
            raise self._errors[path]
        if path == "/providers":
            return self._providers
        if path == "/quota":
            return self._quota
        if path == "/infer/quota":
            return self._infer_quota
        m = re.match(r"/providers/([^/]+)/healthcheck", path)
        if m:
            name = m.group(1)
            base = next(p for p in self._providers if p["name"] == name)
            return {**base, **self._health.get(name, {})}
        raise AssertionError(f"unexpected request {method} {path}")

    def close(self) -> None:
        pass


def provider(name: str, kind: str | None = None, **fields: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "name": name,
        "display_name": name.title(),
        "kind": kind or name,
        "enabled": True,
        "health": "ok",
        "capabilities": {},
        "gpus": ["T4"],
    }
    body.update(fields)
    return body


def daemon_up(
    client: FakeClient, *, pid: int = 4242, test_mode: bool = False, **kw: Any
) -> DaemonInfo:
    health = HealthView(
        version=kw.pop("version", "0.1.0"),
        ready=kw.pop("ready", True),
        pid=pid,
        started_at=kw.pop("started_at", 1_000.0),
        test_mode=test_mode,
    )
    return DaemonInfo(client=client, health=health)  # type: ignore[arg-type]


@pytest.fixture
def user_home(tmp_path: Path) -> Path:
    home = tmp_path / "user"
    home.mkdir()
    return home


@pytest.fixture
def make_env(paths: Paths, user_home: Path) -> Callable[..., ProbeEnv]:
    def build(
        *,
        run: FakeRun | None = None,
        which: dict[str, str] | None = None,
        environ: dict[str, str] | None = None,
        daemon: DaemonInfo | None = None,
        whoami: tuple[str | None, str | None] = ("me", None),
        role: str | None = "write",
        clock: FakeClock | None = None,
        deadline_s: float = 10.0,
    ) -> ProbeEnv:
        tools = (
            which
            if which is not None
            else {
                "kaggle": "/fake/bin/kaggle",
                "colab": "/fake/bin/colab",
                "git": "/usr/bin/git",
                "uv": "/fake/bin/uv",
                "gpu": "/fake/bin/gpu",
            }
        )
        env_vars = {"HOME": str(user_home), "PATH": "/usr/bin:/bin"}
        if environ:
            env_vars.update(environ)
        return ProbeEnv(
            paths=paths,
            clock=clock or FakeClock(),
            environ=env_vars,
            user_home=user_home,
            run=run or default_run(),
            which=tools.get,
            hf_whoami=lambda _token: whoami,
            hf_role=lambda _token: role,
            launchd_plist=user_home / "Library" / "LaunchAgents" / "dev.gpu-router.daemon.plist",
            deadline_s=deadline_s,
            daemon=daemon or DaemonInfo(error="the gpu-router daemon is not running"),
        )

    return build


def write(path: Path, text: str = "{}", mode: int = 0o600) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    os.chmod(path, mode)
    return path
