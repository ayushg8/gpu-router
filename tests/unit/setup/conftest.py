"""Setup-wizard test harness (phase 8b).

Everything the wizard touches is sandboxed: a tmp user home (so ~/.kaggle, ~/.lightning,
~/.claude, ~/.codex, ~/Library/LaunchAgents and ~/.local/bin are all under tmp_path), a
scripted subprocess runner (launchctl, uv, claude, kaggle, colab, git are fakes that record
their argv), a fake `which`, a scripted UI, a fake HF whoami and Lightning bridge, and the
in-memory keyring from tests/conftest.py. No test runs a real launchctl / uv / claude /
codex / gcloud / provider, and the real ~/.claude, ~/.codex and ~/Library are never
written (`real_home_guard` checks the files that matter did not change).
"""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from gpu_router.clock import FakeClock
from gpu_router.doctor.probe import CmdResult
from gpu_router.errors import DaemonUnavailable
from gpu_router.paths import Paths
from gpu_router.providers.lightning.sdk import CallResult
from gpu_router.setup.base import Ctx, Options
from gpu_router.setup.context import SetupEnv
from gpu_router.setup.state import SetupState

WHOAMI_OK = """Auth provider: adc
Email:         me@example.com
Scopes:
  - https://www.googleapis.com/auth/cloud-platform
  - https://www.googleapis.com/auth/colaboratory
  - openid
"""
WHOAMI_NO_SCOPE = """Auth provider: adc
Email:         me@example.com
Scopes:
  - https://www.googleapis.com/auth/cloud-platform
  - openid
"""
LIGHTNING_USER = "user-0123456789"
LIGHTNING_KEY = "key-abcdef0123456789"
HF_TOKEN = "hf_" + "a" * 34
KAGGLE_KEY = "0123456789abcdef0123456789abcdef"


def ok(out: str = "", err: str = "") -> CmdResult:
    return CmdResult(0, out, err)


def fail(code: int = 1, out: str = "", err: str = "") -> CmdResult:
    return CmdResult(code, out, err)


@dataclass
class FakeRun:
    """Scripted subprocesses: the most recently added rule whose regex matches
    ' '.join(argv) answers (a CmdResult or a callable(argv, env) -> CmdResult)."""

    rules: list[tuple[str, Any]] = field(default_factory=list)
    calls: list[list[str]] = field(default_factory=list)
    envs: list[dict[str, str] | None] = field(default_factory=list)

    def on(self, pattern: str, result: Any) -> FakeRun:
        self.rules.insert(0, (pattern, result))
        return self

    def __call__(
        self, argv: list[str], timeout: float, *, env: Any = None, cwd: str | None = None
    ) -> CmdResult:
        self.calls.append(list(argv))
        self.envs.append(dict(env) if env is not None else None)
        line = " ".join(argv)
        for pattern, result in self.rules:
            if re.search(pattern, line):
                return result(argv, env) if callable(result) else result
        return CmdResult(None, "", "", error=f"{argv[0]} not found")

    def ran(self, pattern: str) -> list[list[str]]:
        return [c for c in self.calls if re.search(pattern, " ".join(c))]


class ScriptedUi:
    """Answers questions from a script; records every line. An unexpected question fails
    the test (the wizard must never ask something the test did not plan for)."""

    def __init__(self, answers: list[Any] | None = None, *, interactive: bool = True) -> None:
        self.answers = list(answers or [])
        self.interactive = interactive
        self.lines: list[str] = []
        self.questions: list[str] = []
        self.secrets_asked: list[str] = []

    def _next(self, what: str) -> Any:
        if not self.answers:
            raise AssertionError(f"unexpected {what}; output so far:\n" + self.text)
        return self.answers.pop(0)

    @property
    def text(self) -> str:
        return "\n".join(self.lines)

    def title(self, text: str, sub: str = "") -> None:
        self.lines.append(f"{text}  {sub}")

    def step(self, n: int, total: int, title: str, sub: str = "") -> None:
        self.lines.append(f"{n}/{total} {title}  {sub}")

    def item(self, mark: tuple[str, str], text: str, fix: str | None = None) -> None:
        self.lines.append(f"{mark[0]} {text}")
        if fix:
            self.lines.append(f"$ {fix}")

    def say(self, text: str, style: str = "") -> None:
        self.lines.append(f"  {text}")

    def command(self, argv_text: str) -> None:
        self.lines.append(f"$ {argv_text}")

    def block(self, text: str) -> None:
        self.lines.extend(text.splitlines())

    def write(self, text: str) -> None:
        self.lines.extend(text.splitlines())

    def ask(self, question: str, default: bool) -> bool:
        self.questions.append(question)
        self.lines.append(f"? {question}")
        answer = self._next(f"question {question!r}")
        if answer is None:
            return default
        assert isinstance(answer, bool), f"answer for {question!r} must be a bool: {answer!r}"
        return answer

    def secret(self, prompt: str) -> str:
        self.secrets_asked.append(prompt)
        answer = self._next(f"secret prompt {prompt!r}")
        assert isinstance(answer, str)
        return answer

    def choose(self, question: str, options: list[tuple[str, str]], default: str) -> str:
        self.questions.append(question)
        answer = self._next(f"choice {question!r}")
        assert isinstance(answer, str)
        assert answer in {k for k, _ in options}
        return answer


class FakeBridge:
    def __init__(self, result: CallResult | Exception) -> None:
        self.result = result

    def call(self, op: str, params: Any, *, timeout: float) -> CallResult:
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


LIGHTNING_OK = CallResult(
    ok=True, result={"user": "demo", "teamspace": "demo/default", "teamspaces": ["demo/default"]}
)


@dataclass
class Sandbox:
    """One wizard world. `env` is rebuilt on access so changes to `tools` show up."""

    paths: Paths
    user_home: Path
    run: FakeRun
    tools: dict[str, str]
    clock: FakeClock
    environ: dict[str, str]
    whoami: tuple[str | None, str | None] = ("demo", None)
    hf_role: str | None = "write"
    lightning: CallResult | Exception = LIGHTNING_OK
    browser_pair: tuple[str, str] | None = (LIGHTNING_USER, LIGHTNING_KEY)
    attached: list[list[str]] = field(default_factory=list)
    on_attached: Callable[[list[str]], int] | None = None
    connect: Callable[[], Any] | None = None
    gpu_bin: Path | None = None

    def _attached(self, argv: list[str], env: Any = None) -> int:
        self.attached.append(list(argv))
        return self.on_attached(argv) if self.on_attached else 0

    def _connect(self) -> Any:
        if self.connect is None:
            raise DaemonUnavailable("the gpu-router daemon is not running")
        return self.connect()

    @property
    def env(self) -> SetupEnv:
        return SetupEnv(
            paths=self.paths,
            clock=self.clock,
            environ=self.environ,
            user_home=self.user_home,
            run=self.run,
            run_attached=self._attached,
            which=self.tools.get,
            hf_whoami=lambda _t: self.whoami,
            hf_role=lambda _t: self.hf_role,
            lightning_bridge=lambda _env: FakeBridge(self.lightning),
            lightning_browser=lambda: self.browser_pair,
            connect=self._connect,
            gpu_bin=self.gpu_bin,
            smoke_poll_s=0.05,
        )

    def ctx(self, ui: ScriptedUi, **opts: Any) -> Ctx:
        from gpu_router.setup.wizard import expand_only

        only = opts.pop("only", None)
        options = Options(only=expand_only(only or []), **opts)
        state = SetupState.load(self.paths.home)
        if options.only:
            state.begin_subset()
        return Ctx(env=self.env, ui=ui, state=state, opts=options)

    def install_tool(self, name: str) -> Path:
        """A fake executable in ~/.local/bin (what `uv tool install` leaves behind)."""
        exe = self.user_home / ".local" / "bin" / name
        exe.parent.mkdir(parents=True, exist_ok=True)
        exe.write_text("#!/bin/sh\nexit 0\n")
        exe.chmod(0o755)
        self.tools[name] = str(exe)
        return exe

    def write(self, rel: str, text: str = "{}", mode: int = 0o600) -> Path:
        path = self.user_home / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        os.chmod(path, mode)
        return path


def default_run() -> FakeRun:
    return (
        FakeRun()
        .on(r"kaggle --version$", ok("Kaggle API 2.2.4"))
        .on(r"colab version$", ok("Version: 0.7.2"))
        .on(r"colab --auth=adc --config \S+ whoami$", ok(WHOAMI_OK))
        .on(r"colab --auth=adc --config \S+ new --help$", ok("--gpu  T4, L4"))
        .on(r"git --version$", ok("git version 2.54.0"))
        .on(r"uv --version$", ok("uv 0.11.29"))
        .on(r"hf --version$", ok("huggingface_hub version: 1.32.0"))
        .on(r"kaggle -W quota --format json$", ok("[]"))
        .on(r"^/bin/launchctl print ", fail(113, err="Could not find service"))
    )


@pytest.fixture
def user_home(tmp_path: Path) -> Path:
    home = tmp_path / "user"
    home.mkdir()
    return home


@pytest.fixture
def sandbox(paths: Paths, user_home: Path, monkeypatch: pytest.MonkeyPatch) -> Sandbox:
    # the launchd agent "serves this data dir" only when it is the default one
    monkeypatch.setattr("gpu_router.paths.DEFAULT_HOME", paths.home)
    gpu = user_home / ".local" / "bin" / "gpu"
    gpu.parent.mkdir(parents=True, exist_ok=True)
    gpu.write_text("#!/bin/sh\nexit 0\n")
    gpu.chmod(0o755)
    tools = {
        "uv": "/fake/bin/uv",
        "git": "/usr/bin/git",
        "gpu": str(gpu),
        "kaggle": "/fake/bin/kaggle",
        "colab": "/fake/bin/colab",
    }
    return Sandbox(
        paths=paths,
        user_home=user_home,
        run=default_run(),
        tools=tools,
        clock=FakeClock(),
        environ={"HOME": str(user_home), "PATH": "/usr/bin:/bin"},
        gpu_bin=gpu,
    )


# --------------------------------------------------------------------------- real-home guard

_REAL = Path(os.path.expanduser("~"))
_WATCHED = (
    _REAL / ".claude" / "settings.json",
    _REAL / ".claude" / "plugins" / "installed_plugins.json",
    _REAL / ".claude" / "plugins" / "known_marketplaces.json",
    _REAL / ".codex" / "config.toml",
    _REAL / "Library" / "LaunchAgents" / "dev.gpu-router.daemon.plist",
)


def _fingerprint() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for p in _WATCHED:
        try:
            out[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
        except OSError:
            out[str(p)] = None
    out["colab-skill"] = str((_REAL / ".claude" / "skills" / "colab").exists())
    return out


@pytest.fixture(autouse=True)
def real_home_guard() -> Iterator[None]:
    before = _fingerprint()
    yield
    assert _fingerprint() == before, "a setup test changed a file in the real home directory"
