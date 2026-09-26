"""What `gpu doctor` looks at, behind one injectable object (phase 8a).

`ProbeEnv` holds the data dir, environment, the user's home, a bounded subprocess runner,
`which`, the daemon connection (never auto-started unless `gpu doctor --start`), an HF
whoami function and the overall deadline. Tests build one over tmp dirs with fakes; the
CLI and the shell build the real one with `ProbeEnv.real()`.

Rules: nothing here prints or stores a credential value. Credential FILES are only
stat()ed (existence, owner, mode), never opened; Keychain secrets are listed by NAME from
secrets.index; the one value read is the HF token for a whoami call, in this process only.
Provider accounts are touched through the daemon (healthcheck, quota ledger), so the
ledger stays the one owner of quota (invariant 2); doctor itself runs only local,
quota-free probes: tool versions, `colab whoami` (token scopes), `colab new --help`.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gpu_router.api import HealthView, ProviderView
    from gpu_router.client import GpuClient
    from gpu_router.clock import Clock
    from gpu_router.config import Config
    from gpu_router.models import QuotaSnapshot
    from gpu_router.paths import Paths
    from gpu_router.providers.catalog import Catalog

__all__ = [
    "CmdResult",
    "DaemonInfo",
    "Lazy",
    "ProbeEnv",
    "ToolInfo",
    "parse_version",
    "run_cmd",
    "tool_info",
    "version_tuple",
]

_VERSION_RE = re.compile(r"(\d+\.\d+(?:\.\d+)?(?:\.post\d+)?)")


@dataclass(frozen=True, slots=True)
class CmdResult:
    returncode: int | None  # None = timed out (killed) or could not start
    stdout: str
    stderr: str
    error: str | None = None  # why it could not run

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    @property
    def text(self) -> str:
        return f"{self.stdout}\n{self.stderr}".strip()


Runner = Callable[..., CmdResult]


def run_cmd(
    argv: list[str],
    timeout: float,
    *,
    env: Mapping[str, str] | None = None,
    cwd: str | None = None,
) -> CmdResult:
    """Run argv with stdin closed, capture output, kill the process group at `timeout`."""
    import signal

    try:
        proc = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            env=dict(env) if env is not None else None,
            cwd=cwd,
            start_new_session=True,
        )
    except FileNotFoundError:
        return CmdResult(None, "", "", error=f"{argv[0]} not found")
    except OSError as exc:
        return CmdResult(None, "", "", error=f"cannot run {argv[0]}: {exc.strerror or exc}")
    try:
        out, err = proc.communicate(timeout=max(0.5, timeout))
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            proc.kill()
        try:
            out, err = proc.communicate(timeout=3)
        except subprocess.TimeoutExpired:
            out, err = "", ""
        return CmdResult(None, out or "", err or "", error=f"timed out after {timeout:.0f}s")
    return CmdResult(proc.returncode, out or "", err or "")


def parse_version(text: str) -> str | None:
    m = _VERSION_RE.search(text or "")
    return m.group(1) if m else None


def version_tuple(v: str | None) -> tuple[int, ...]:
    if not v:
        return ()
    nums: list[int] = []
    for part in v.split("."):
        digits = re.match(r"\d+", part)
        if digits is None:
            break
        nums.append(int(digits.group()))
    return tuple(nums)


class Lazy[T]:
    """A value computed once, on first use, by whichever check thread asks first; the
    others wait for it (thread-safe)."""

    def __init__(self, fn: Callable[[], T]) -> None:
        self._fn = fn
        self._lock = threading.Lock()
        self._done = False
        self._value: T | None = None
        self._error: BaseException | None = None

    def get(self) -> T:
        with self._lock:
            if not self._done:
                try:
                    self._value = self._fn()
                except BaseException as exc:
                    self._error = exc
                self._done = True
        if self._error is not None:
            raise self._error
        return self._value  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class ToolInfo:
    exe: str  # command name
    path: str | None  # resolved executable, None = not installed
    version: str | None
    how: str  # where the version came from: dist-info | --version | unknown


def _dist_version(exe_path: str, dist: str) -> str | None:
    """Version from the tool's own venv (uv tool envs): <venv>/lib/python*/site-packages/
    <dist>-<version>.dist-info. No subprocess, no import of the tool."""
    try:
        real = Path(os.path.realpath(exe_path))
    except OSError:
        return None
    venv = real.parent.parent
    if not (venv / "pyvenv.cfg").is_file():
        return None
    norm = re.sub(r"[-_.]+", "_", dist).lower()
    for site in venv.glob("lib/python*/site-packages"):
        try:
            entries = list(site.iterdir())
        except OSError:
            continue
        for entry in entries:
            name = entry.name
            if not name.endswith(".dist-info"):
                continue
            stem = name[: -len(".dist-info")]
            pkg, _, ver = stem.rpartition("-")
            if re.sub(r"[-_.]+", "_", pkg).lower() == norm and ver:
                return ver
    return None


def tool_info(
    exe: str,
    dist: str,
    *,
    which: Callable[[str], str | None],
    run: Runner,
    user_home: Path,
    version_args: tuple[str, ...] = ("--version",),
    timeout: float = 10.0,
) -> ToolInfo:
    """Find `exe` (PATH, then ~/.local/bin like the adapters) and its version."""
    path = which(exe)
    if path is None:
        fallback = user_home / ".local" / "bin" / exe
        if fallback.is_file() and os.access(fallback, os.X_OK):
            path = str(fallback)
    if path is None:
        return ToolInfo(exe=exe, path=None, version=None, how="unknown")
    ver = _dist_version(path, dist)
    if ver is not None:
        return ToolInfo(exe=exe, path=path, version=ver, how="dist-info")
    env = {**os.environ, "NO_COLOR": "1", "COLUMNS": "200"}
    res = run([path, *version_args], timeout, env=env)
    ver = parse_version(res.text) if res.returncode == 0 else None
    return ToolInfo(exe=exe, path=path, version=ver, how="--version" if ver else "unknown")


@dataclass
class DaemonInfo:
    client: GpuClient | None = None
    health: HealthView | None = None
    error: str | None = None
    hint: str | None = None
    started: bool = False  # `gpu doctor --start` started it

    @property
    def up(self) -> bool:
        return self.client is not None and self.health is not None


def _default_whoami(token: str) -> tuple[str | None, str | None]:
    from gpu_router.cli.login import _whoami

    return _whoami(token)


def _default_role(token: str) -> str | None:
    from gpu_router.cli.login import _role

    return _role(token)


@dataclass
class ProbeEnv:
    paths: Paths
    clock: Clock
    environ: Mapping[str, str] = field(default_factory=lambda: dict(os.environ))
    user_home: Path = field(default_factory=Path.home)
    run: Runner = run_cmd
    which: Callable[[str], str | None] = shutil.which
    hf_whoami: Callable[[str], tuple[str | None, str | None]] = _default_whoami
    #: the token's role, "write" / "read" / None (unknown); checkpoints need write
    hf_role: Callable[[str], str | None] = _default_role
    launchd_plist: Path | None = None  # None = ~/Library/LaunchAgents/<label>.plist
    version: str = ""
    deadline_s: float = 20.0
    daemon: DaemonInfo = field(default_factory=DaemonInfo)
    started_mono: float = field(default_factory=time.monotonic)

    def __post_init__(self) -> None:
        if not self.version:
            from gpu_router import __version__

            self.version = __version__
        self._config: Lazy[Config] = Lazy(self._load_config)
        self._catalog: Lazy[Catalog] = Lazy(self._load_catalog)
        self._providers: Lazy[list[ProviderView]] = Lazy(self._load_providers)
        self._quotas: Lazy[dict[str, QuotaSnapshot]] = Lazy(self._load_quotas)
        self._health: dict[str, Lazy[ProviderView]] = {}
        self._health_lock = threading.Lock()

    # ------------------------------------------------------------------ construction

    @classmethod
    def real(
        cls,
        paths: Paths,
        *,
        deadline_s: float = 20.0,
        client: GpuClient | None = None,
    ) -> ProbeEnv:
        from gpu_router.clock import SystemClock

        env = cls(paths=paths, clock=SystemClock(), deadline_s=deadline_s)
        env.daemon = probe_daemon(paths) if client is None else daemon_from_client(client)
        return env

    # ------------------------------------------------------------------ time

    def remaining(self, cap: float | None = None) -> float:
        """Seconds left before the doctor's deadline (at least 0.5), capped at `cap`."""
        left = self.deadline_s - (time.monotonic() - self.started_mono)
        left = max(0.5, left)
        return min(left, cap) if cap is not None else left

    # ------------------------------------------------------------------ shared data

    @property
    def claude_dir(self) -> Path:
        base = self.environ.get("CLAUDE_CONFIG_DIR")
        return Path(base).expanduser() if base else self.user_home / ".claude"

    @property
    def codex_dir(self) -> Path:
        base = self.environ.get("CODEX_HOME")
        return Path(base).expanduser() if base else self.user_home / ".codex"

    def config(self) -> Config:
        """config.yaml as the CLI reads it (raises ConfigError)."""
        return self._config.get()

    def catalog(self) -> Catalog:
        """providers.yaml (packaged + <home>/providers.yaml; raises ConfigError)."""
        return self._catalog.get()

    def providers(self) -> list[ProviderView]:
        """The daemon's provider list ([] when the daemon is down)."""
        return self._providers.get()

    def quotas(self) -> dict[str, QuotaSnapshot]:
        """The daemon's quota ledger views by provider ({} when down or on error)."""
        return self._quotas.get()

    def healthcheck(self, name: str) -> ProviderView:
        """A live healthcheck through the daemon (once per provider per doctor run).
        Raises GpuRouterError (DaemonUnavailable when the daemon is down)."""
        with self._health_lock:
            lazy = self._health.get(name)
            if lazy is None:
                lazy = self._health[name] = Lazy(lambda: self._healthcheck(name))
        return lazy.get()

    def _load_config(self) -> Config:
        from gpu_router.config import load_config

        return load_config(self.paths, self.environ)

    def _load_catalog(self) -> Catalog:
        from gpu_router.providers.catalog import load_catalog

        return load_catalog(self.paths.user_providers)

    def _load_providers(self) -> list[ProviderView]:
        from gpu_router.api import ProviderView

        client = self.daemon.client
        if not self.daemon.up or client is None:
            return []
        data = client.request("GET", "/providers", timeout_s=self.remaining(10))
        return [ProviderView.model_validate(p) for p in data or []]

    def _load_quotas(self) -> dict[str, QuotaSnapshot]:
        from gpu_router.errors import GpuRouterError
        from gpu_router.models import QuotaSnapshot

        client = self.daemon.client
        if not self.daemon.up or client is None:
            return {}
        try:
            data = client.request("GET", "/quota", timeout_s=self.remaining(20))
        except GpuRouterError:
            return {}
        out: dict[str, QuotaSnapshot] = {}
        for raw in data or []:
            snap = QuotaSnapshot.model_validate(raw)
            out[snap.provider] = snap
        return out

    def _healthcheck(self, name: str) -> ProviderView:
        from gpu_router.api import ProviderView
        from gpu_router.errors import DaemonUnavailable

        client = self.daemon.client
        if not self.daemon.up or client is None:
            raise DaemonUnavailable("the gpu-router daemon is not running")
        data = client.request(
            "POST", f"/providers/{name}/healthcheck", timeout_s=self.remaining(45)
        )
        return ProviderView.model_validate(data)

    def secret_names(self) -> set[str]:
        """Keychain secret NAMES gpu-router stored (secrets.index; values never read)."""
        from gpu_router import secrets

        try:
            return set(secrets.secret_names())
        except OSError:
            return set()

    def gpu_router_repo(self) -> Path | None:
        """The source checkout this gpu-router runs from (an editable install), if any."""
        import gpu_router

        try:
            root = Path(gpu_router.__file__).resolve().parents[2]
        except (IndexError, OSError):
            return None
        return root if (root / "pyproject.toml").is_file() and (root / "plugin").is_dir() else None


def daemon_from_client(client: GpuClient) -> DaemonInfo:
    from gpu_router.errors import GpuRouterError

    try:
        health = client.health()
    except GpuRouterError as exc:
        return DaemonInfo(error=exc.message, hint=exc.hint)
    return DaemonInfo(client=client, health=health)


def probe_daemon(paths: Paths, *, timeout_s: float = 3.0) -> DaemonInfo:
    """Connect to a running daemon (never starts one) and read /v1/health."""
    from gpu_router.client import GpuClient
    from gpu_router.errors import GpuRouterError

    try:
        client = GpuClient.from_env(paths, client_name="cli", timeout_s=timeout_s)
    except GpuRouterError as exc:
        return DaemonInfo(error=exc.message, hint=exc.hint)
    info = daemon_from_client(client)
    if not info.up:
        client.close()
    return info


def stat_mode(path: Path) -> tuple[int, int] | None:
    """(permission bits, owner uid) or None when missing. Metadata only: the file is
    never opened (credential files are judged by their mode, never read)."""
    try:
        st = path.stat()
    except OSError:
        return None
    return st.st_mode & 0o777, st.st_uid


def fmt_mode(mode: int) -> str:
    return f"{mode:04o}"


def home_label(path: Path | str, user_home: Path) -> str:
    """~/... for paths under the user's home (display only)."""
    text = str(path)
    home = str(user_home).rstrip("/")
    if home and (text == home or text.startswith(home + "/")):
        return "~" + text[len(home) :]
    return text


def detail_safe(value: Any) -> Any:
    """Drop anything that is not plain JSON (detail dicts stay small and printable)."""
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, dict):
        return {str(k): detail_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [detail_safe(v) for v in value]
    return str(value)
