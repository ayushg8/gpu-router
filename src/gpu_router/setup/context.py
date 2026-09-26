"""Everything the wizard touches, behind one injectable object (phase 8b).

`SetupEnv` holds the data dir, the user's home (so ~/.kaggle, ~/.lightning, ~/.claude,
~/.codex and ~/Library/LaunchAgents all resolve under it), the environment, a bounded
subprocess runner (also used for `/bin/launchctl`, `uv tool install`, `claude plugin`,
`kaggle`), an interactive runner (gcloud's browser sign-in, attached to the terminal),
`which`, the HF whoami check, the Lightning SDK bridge / browser sign-in, and how to reach
the daemon. Tests build one over tmp dirs with fakes (`tests/unit/setup/conftest.py`), so
no test runs launchctl, uv, claude, codex, gcloud or a provider for real, and nothing in
the real ~/.claude, ~/.codex or ~/Library/LaunchAgents changes. `SetupEnv.real()` is the
one the CLI uses.

`probe()` returns a doctor `ProbeEnv` over the same fakes: the wizard reuses doctor's
checks to decide what is already done, so `gpu setup` and `gpu doctor` never disagree.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from gpu_router.doctor.probe import CmdResult, DaemonInfo, ProbeEnv, run_cmd

if TYPE_CHECKING:
    from gpu_router.client import GpuClient
    from gpu_router.clock import Clock
    from gpu_router.paths import Paths

__all__ = ["LAUNCHCTL", "SetupEnv", "run_attached"]

LAUNCHCTL = "/bin/launchctl"

Runner = Callable[..., CmdResult]
AttachedRunner = Callable[[list[str], Mapping[str, str] | None], int]


def run_attached(argv: list[str], env: Mapping[str, str] | None = None) -> int:
    """Run argv on the user's terminal (stdin/stdout/stderr inherited, no timeout: a browser
    sign-in waits for the human). Returns the exit code; 127 when it cannot start."""
    try:
        return subprocess.run(
            argv, env=dict(env) if env is not None else None, check=False
        ).returncode
    except FileNotFoundError:
        return 127
    except OSError:
        return 126


def _default_whoami(token: str) -> tuple[str | None, str | None]:
    from gpu_router.cli.login import _whoami

    return _whoami(token)


def _default_role(token: str) -> str | None:
    from gpu_router.cli.login import _role

    return _role(token)


def _default_connect(paths: Paths) -> GpuClient:
    from gpu_router.daemon.spawn import connect

    return connect(paths).client


@dataclass
class SetupEnv:
    paths: Paths
    clock: Clock
    environ: dict[str, str] = field(default_factory=lambda: dict(os.environ))
    user_home: Path = field(default_factory=Path.home)
    run: Runner = run_cmd
    run_attached: AttachedRunner = run_attached
    which: Callable[[str], str | None] = shutil.which
    hf_whoami: Callable[[str], tuple[str | None, str | None]] = _default_whoami
    #: the token's role, "write" / "read" / None (unknown); checkpoints need write
    hf_role: Callable[[str], str | None] = _default_role
    #: Lightning: SdkBridge factory for the credential check (None = the real bridge)
    lightning_bridge: Callable[[dict[str, str]], Any] | None = None
    #: Lightning browser sign-in: returns (user_id, api_key) or None (None = the real one)
    lightning_browser: Callable[[], tuple[str, str] | None] | None = None
    #: a ready daemon client (auto-starts one like the CLI); tests pass an in-process one
    connect: Callable[[], GpuClient] | None = None
    #: the `gpu` executable for launchd / the status-line wrapper (None = find it)
    gpu_bin: Path | None = None
    smoke_poll_s: float = 3.0

    @classmethod
    def real(cls, paths: Paths) -> SetupEnv:
        from gpu_router.clock import SystemClock

        return cls(paths=paths, clock=SystemClock())

    # ------------------------------------------------------------------ places

    @property
    def claude_dir(self) -> Path:
        base = self.environ.get("CLAUDE_CONFIG_DIR")
        return Path(base).expanduser() if base else self.user_home / ".claude"

    @property
    def codex_dir(self) -> Path:
        base = self.environ.get("CODEX_HOME")
        return Path(base).expanduser() if base else self.user_home / ".codex"

    @property
    def claude_settings(self) -> Path:
        return self.claude_dir / "settings.json"

    @property
    def launchd_plist(self) -> Path:
        from gpu_router.daemon.launchd import plist_path

        return plist_path(self.user_home)

    @property
    def kaggle_dir(self) -> Path:
        base = self.environ.get("KAGGLE_CONFIG_DIR")
        return Path(base).expanduser() if base else self.user_home / ".kaggle"

    @property
    def lightning_file(self) -> Path:
        return self.user_home / ".lightning" / "credentials.json"

    def hf_environ(self) -> dict[str, str]:
        """environ for `checkpoint.tokens.external_token`, with the hf CLI's token file under
        THIS user home when HF_HOME / HF_TOKEN_PATH are not set."""
        env = dict(self.environ)
        if not env.get("HF_TOKEN_PATH") and not env.get("HF_HOME"):
            env["HF_HOME"] = str(self.user_home / ".cache" / "huggingface")
        return env

    # ------------------------------------------------------------------ programs

    def find(self, exe: str, *extra: Path) -> str | None:
        """`exe` on PATH, else ~/.local/bin (uv tool installs), else the extra places (all
        under the user home: tests stay hermetic; the wizard runs from a terminal whose PATH
        already has Homebrew)."""
        found = self.which(exe)
        if found:
            return found
        for cand in (self.user_home / ".local" / "bin" / exe, *extra):
            if cand.is_file() and os.access(cand, os.X_OK):
                return str(cand)
        return None

    def uv(self) -> str | None:
        return self.find("uv", self.user_home / ".cargo" / "bin" / "uv")

    def gpu_executable(self) -> Path | None:
        """The `gpu` launchd and the status line should run: the one on PATH (a uv tool
        install survives repo moves), else the console script next to this interpreter."""
        if self.gpu_bin is not None:
            return self.gpu_bin
        found = self.find("gpu")
        if found:
            return Path(found)
        beside = Path(sys.executable).with_name("gpu")
        if beside.is_file() and os.access(beside, os.X_OK):
            return beside
        return None

    def launchctl(self, *args: str) -> CmdResult:
        return self.run([LAUNCHCTL, *args], 30.0)

    # ------------------------------------------------------------------ doctor / daemon

    def probe(self, *, daemon: DaemonInfo | None = None, deadline_s: float = 20.0) -> ProbeEnv:
        return ProbeEnv(
            paths=self.paths,
            clock=self.clock,
            environ=self.environ,
            user_home=self.user_home,
            run=self.run,
            which=self.which,
            hf_whoami=self.hf_whoami,
            hf_role=self.hf_role,
            launchd_plist=self.launchd_plist,
            deadline_s=deadline_s,
            daemon=daemon or DaemonInfo(error="not connected"),
        )

    def client(self) -> GpuClient:
        """A ready daemon client (starts the daemon like any CLI command)."""
        if self.connect is not None:
            return self.connect()
        return _default_connect(self.paths)

    def gpu_router_repo(self) -> Path | None:
        return self.probe().gpu_router_repo()
