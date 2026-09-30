"""The individual `gpu doctor` checks (phase 8a).

Each check takes a `ProbeEnv` and returns CheckResult(s). Checks never raise on purpose (the
runner turns a crash into a warn row), never print, and bound every subprocess by
`env.remaining(...)`. Fix strings are exact commands; paths in them are shell-quoted.
"""

from __future__ import annotations

import atexit
import json
import os
import re
import shlex
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

from gpu_router.doctor.model import CheckResult, DriftItem, Status
from gpu_router.doctor.probe import (
    ProbeEnv,
    fmt_mode,
    home_label,
    stat_mode,
    tool_info,
    version_tuple,
)
from gpu_router.errors import ConfigError, GpuRouterError, NotReady

if TYPE_CHECKING:
    from gpu_router.api import ProviderView
    from gpu_router.config import Config
    from gpu_router.providers.catalog import Catalog, ProviderEntry

__all__ = ["EXCLUDED", "TOOLS", "Task", "default_tasks"]

OK, WARN, FAIL, SKIP = Status.OK, Status.WARN, Status.FAIL, Status.SKIP

RESTART = "gpu daemon stop && gpu daemon start"
EDIT = "${EDITOR:-nano} "
COLAB_SCOPE = "https://www.googleapis.com/auth/colaboratory"
ADC_LOGIN = (
    "gcloud auth application-default login --scopes=openid,"
    "https://www.googleapis.com/auth/cloud-platform,"
    "https://www.googleapis.com/auth/userinfo.email,"
    "https://www.googleapis.com/auth/colaboratory"
)
KAGGLE_FILE_FIX = (
    "mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json"
)
PLUGIN_KEY = "gpu-router@gpu-router-local"  # the checkout's own marketplace (plugin/)
#: the repo-root marketplace: `claude plugin marketplace add ayushg8/gpu-router`
GITHUB_MARKETPLACE = "ayushg8/gpu-router"
GITHUB_PLUGIN_KEY = "gpu-router@gpu-router"
DISK_WARN_GB = 5.0
DISK_FAIL_GB = 1.0
HF_HUB_MIN = "1.32"

#: Fallback when providers.yaml has no `excluded:` section (catalogs before phase 7):
#: services dropped on the provider's own words. The catalog's section wins when present.
EXCLUDED: dict[str, str] = {
    "modal": "needs a card: “Note that you must have a payment method on file in order to "
    "use Modal.” (https://modal.com/docs/guide/billing)",
}


@dataclass(frozen=True, slots=True)
class ToolSpec:
    exe: str
    dist: str
    install: str
    upgrade: str
    tested: str | None = None  # version the adapter was verified against
    version_args: tuple[str, ...] = ("--version",)


TOOLS: dict[str, ToolSpec] = {
    "kaggle": ToolSpec(
        "kaggle", "kaggle", "uv tool install kaggle", "uv tool upgrade kaggle", tested="2.2.4"
    ),
    "colab": ToolSpec(
        "colab",
        "google-colab-cli",
        "uv tool install google-colab-cli",
        "uv tool upgrade google-colab-cli",
        tested="0.7.2",
        version_args=("version",),
    ),
}

LIVE_OK = {
    "kaggle": "kaggle answered (credentials + `kaggle quota` work)",
    "colab": "colab answered (`colab sessions` works)",
    "lightning": "lightning answered",
    "local": "ready: jobs run on this Mac",
}


@dataclass(frozen=True, slots=True)
class Task:
    id: str
    group: str
    title: str
    fn: Callable[[ProbeEnv], CheckResult | list[CheckResult]]


def _r(
    task_id: str,
    group: str,
    title: str,
    status: Status,
    summary: str,
    fix: str | None = None,
    **detail: Any,
) -> CheckResult:
    return CheckResult(
        id=task_id,
        group=group,
        title=title,
        status=status,
        summary=summary,
        fix=fix,
        detail={k: v for k, v in detail.items() if v is not None},
    )


def _q(path: Path | str) -> str:
    return shlex.quote(str(path))


def _dur(seconds: float) -> str:
    s = max(0, int(seconds))
    if s < 60:
        return f"{s}s"
    m = s // 60
    if m < 60:
        return f"{m}m"
    h, m = divmod(m, 60)
    if h < 48:
        return f"{h}h{m:02d}m"
    return f"{h // 24}d{h % 24}h"


def _num(v: float) -> str:
    return f"{v:.2f}".rstrip("0").rstrip(".") if v != int(v) else str(int(v))


def _config(env: ProbeEnv) -> Config | None:
    try:
        return env.config()
    except (ConfigError, OSError):
        return None


def _catalog(env: ProbeEnv) -> Catalog | None:
    try:
        return env.catalog()
    except (ConfigError, OSError):
        return None


def _enabled(env: ProbeEnv, entry: ProviderEntry) -> bool:
    """The daemon's word when it is up, else config + catalog like the registry."""
    if env.daemon.up:
        try:
            for view in env.providers():
                if view.name == entry.name:
                    return bool(view.enabled)
        except GpuRouterError:
            pass
    from gpu_router.adapters.registry import is_enabled

    config = _config(env)
    if config is None:
        return entry.enabled_by_default and not entry.test_only
    return is_enabled(
        entry, config.providers.get(entry.name), test_mode=config.test_mode, environ=env.environ
    )


def _not_enabled(t: tuple[str, str, str], entry: ProviderEntry) -> CheckResult:
    """A provider that is not enabled is not probed (no subprocess, no network)."""
    return _r(*t, SKIP, f"{entry.name} is not enabled; not checked")


# =========================================================================== daemon


def _state_active(env: ProbeEnv) -> int:
    """Active jobs listed in state.json (0 when absent or unreadable)."""
    try:
        snap = json.loads(env.paths.state.read_bytes())
        return len(snap.get("active") or [])
    except (OSError, ValueError, AttributeError, TypeError):
        return 0


def _launchd_serves_here(env: ProbeEnv) -> bool:
    import plistlib

    from gpu_router.daemon.launchd import plist_path
    from gpu_router.paths import DEFAULT_HOME, ENV_HOME

    plist = env.launchd_plist or plist_path(env.user_home)
    try:
        agent = plistlib.loads(plist.read_bytes())
    except (OSError, ValueError, plistlib.InvalidFileException):
        return False
    raw = (agent.get("EnvironmentVariables") or {}).get(ENV_HOME)
    home = Path(raw).expanduser().resolve() if raw else DEFAULT_HOME.resolve()
    return home == env.paths.home.resolve()


def _launchd_last_exit(env: ProbeEnv) -> int | None:
    """The agent's last exit code from `launchctl print` (None: unknown or never exited)."""
    from gpu_router.daemon.launchd import LABEL

    res = env.run(["/bin/launchctl", "print", f"gui/{os.getuid()}/{LABEL}"], env.remaining(5))
    if not res.ok:
        return None
    m = re.search(r"^\s*last exit code = (-?\d+)", res.stdout, re.M)
    return int(m.group(1)) if m else None


def check_daemon_running(env: ProbeEnv) -> CheckResult:
    d = env.daemon
    if not d.up or d.health is None or d.client is None:
        why = d.error or "the gpu-router daemon is not running"
        active = _state_active(env)
        if active:
            return _r(
                "daemon.running",
                "daemon",
                "running",
                FAIL,
                f"{why}: {active} job(s) were active when it stopped and nothing watches them "
                "(remote runs keep going; it reattaches on start)",
                "gpu daemon start",
            )
        if _launchd_serves_here(env):
            # KeepAlive {SuccessfulExit: false}: a clean `gpu daemon stop` stays stopped until
            # the next login, so only an abnormal last exit is a failure (review fix)
            code = _launchd_last_exit(env)
            if code is not None and code != 0:
                return _r(
                    "daemon.running",
                    "daemon",
                    "running",
                    FAIL,
                    f"{why}; its last run under launchd exited with code {code} (see "
                    "logs/launchd.log)",
                    "gpu daemon start",
                )
            return _r(
                "daemon.running",
                "daemon",
                "running",
                WARN,
                f"{why}: stopped (launchd starts it again at login; nothing is active)",
                "gpu daemon start",
            )
        return _r(
            "daemon.running",
            "daemon",
            "running",
            WARN,
            f"{why} (any gpu command starts it; nothing is active)",
            "gpu daemon start",
        )
    h = d.health
    port = d.client.base_url.rsplit(":", 1)[-1]
    parts = [f"pid {h.pid}", f"port {port}", f"up {_dur(env.clock.now() - h.started_at)}"]
    if h.test_mode:
        parts.append("test mode")
    if d.started:
        parts.append("started by this check")
    summary = " · ".join(parts)
    if not h.ready:
        return _r(
            "daemon.running",
            "daemon",
            "running",
            WARN,
            summary + " · still recovering jobs; actions wait until it is ready",
            pid=h.pid,
        )
    return _r("daemon.running", "daemon", "running", OK, summary, pid=h.pid, port=port)


def check_daemon_version(env: ProbeEnv) -> CheckResult:
    from gpu_router.api import API_VERSION

    h = env.daemon.health
    if not env.daemon.up or h is None:
        return _r("daemon.version", "daemon", "version", SKIP, "daemon not running")
    if h.version != env.version:
        return _r(
            "daemon.version",
            "daemon",
            "version",
            WARN,
            f"the daemon runs gpu-router {h.version}, this gpu is {env.version}",
            RESTART,
            daemon=h.version,
            cli=env.version,
        )
    if h.api_version != API_VERSION:
        return _r(
            "daemon.version",
            "daemon",
            "version",
            WARN,
            f"daemon API v{h.api_version}, this gpu speaks v{API_VERSION}",
            RESTART,
        )
    return _r(
        "daemon.version",
        "daemon",
        "version",
        OK,
        f"daemon and CLI are both {env.version} (api v{h.api_version})",
    )


def check_launchd(env: ProbeEnv) -> CheckResult:
    import plistlib

    from gpu_router.daemon.launchd import LABEL, plist_path
    from gpu_router.paths import DEFAULT_HOME, ENV_HOME

    t = ("daemon.launchd", "daemon", "launchd")
    plist = env.launchd_plist or plist_path(env.user_home)
    here = env.paths.home.resolve()
    custom = bool(env.environ.get(ENV_HOME)) and here != DEFAULT_HOME.resolve()
    if not plist.exists():
        if custom:
            return _r(
                *t,
                SKIP,
                "no launchd agent; this data dir (GPU_ROUTER_HOME) starts on demand",
            )
        return _r(
            *t,
            WARN,
            "not installed: the daemon starts on demand, not at login",
            "gpu daemon install-launchd",
        )
    try:
        agent = plistlib.loads(plist.read_bytes())
    except (OSError, ValueError, plistlib.InvalidFileException) as exc:
        if custom:  # the one global agent is the default data dir's business, not this one's
            return _r(
                *t,
                SKIP,
                f"cannot read {home_label(plist, env.user_home)} ({exc}); the launchd agent "
                "belongs to the default data dir, this one (GPU_ROUTER_HOME) starts on demand",
            )
        return _r(
            *t,
            FAIL,
            f"cannot read {home_label(plist, env.user_home)}: {exc}",
            "gpu daemon install-launchd",
        )
    args = agent.get("ProgramArguments") or []
    program = str(args[0]) if args else ""
    agent_home_raw = (agent.get("EnvironmentVariables") or {}).get(ENV_HOME)
    agent_home = (
        Path(agent_home_raw).expanduser().resolve() if agent_home_raw else DEFAULT_HOME.resolve()
    )
    if agent_home != here:
        # checked first (review fix): an agent serving another data dir is not this one's
        # to repair, even when its program is gone (its fix would hand it to this dir)
        return _r(
            *t,
            SKIP,
            f"the launchd agent serves {home_label(agent_home, env.user_home)}, not this data dir",
        )
    if not program or not os.path.exists(program):
        return _r(
            *t,
            FAIL,
            f"the agent runs {program or '(nothing)'}, which no longer exists",
            "gpu daemon install-launchd",
        )
    uid = os.getuid()
    res = env.run(["/bin/launchctl", "print", f"gui/{uid}/{LABEL}"], env.remaining(5))
    if not res.ok:
        return _r(
            *t,
            WARN,
            "installed but not loaded: it will not start at login",
            f"launchctl bootstrap gui/{uid} {_q(plist)}",
        )
    state = re.search(r"^\s*state = (\S+)", res.stdout, re.M)
    pid_m = re.search(r"^\s*pid = (\d+)", res.stdout, re.M)
    pid = int(pid_m.group(1)) if pid_m else None
    h = env.daemon.health
    if pid is not None and h is not None and pid == h.pid:
        return _r(*t, OK, f"loaded; launchd runs this daemon (pid {pid}) and starts it at login")
    if state and state.group(1) == "running":
        return _r(*t, OK, f"loaded and running (pid {pid})")
    if not env.daemon.up:
        return _r(
            *t,
            WARN,
            "loaded but not running (see logs/launchd.log)",
            f"launchctl kickstart gui/{uid}/{LABEL}",
        )
    return _r(*t, OK, "loaded; starts the daemon at login")


def check_state_file(env: ProbeEnv) -> CheckResult:
    t = ("daemon.state", "daemon", "state.json")
    path = env.paths.state
    up = env.daemon.up
    h = env.daemon.health
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        if up:
            return _r(*t, WARN, "state.json is missing: the status line shows nothing", RESTART)
        return _r(*t, SKIP, "no state.json yet (the daemon writes it)")
    except OSError as exc:
        return _r(*t, WARN, f"cannot read state.json: {exc.strerror or exc}", RESTART)
    try:
        snap = json.loads(raw)
        written = float(snap["written_at"])
        pid = int(snap.get("daemon_pid") or 0)
        active = len(snap.get("active") or [])
        beat = snap.get("heartbeat_s")
    except (ValueError, KeyError, TypeError):
        return _r(*t, WARN, "state.json is unreadable", RESTART)
    age = max(0.0, env.clock.now() - written)
    if not up or h is None:
        return _r(
            *t,
            SKIP,
            f"written {_dur(age)} ago by a daemon that is not running (the status line ignores it)",
        )
    if pid and pid != h.pid:
        return _r(
            *t,
            WARN,
            f"state.json was written by pid {pid}, not the running daemon (pid {h.pid})",
            RESTART,
        )
    limit = max(5 * float(beat or 60), 300.0)
    if active and age > limit:
        return _r(
            *t,
            WARN,
            f"not rewritten for {_dur(age)} while {active} job(s) are active: the status "
            "line shows 'daemon not responding'",
            RESTART,
            age_s=round(age, 1),
        )
    if written < h.started_at - 1 and not active:
        return _r(*t, OK, f"written {_dur(age)} ago (before this daemon started; nothing active)")
    return _r(
        *t,
        OK,
        f"written {_dur(age)} ago · {active} active job{'s' if active != 1 else ''}",
        age_s=round(age, 1),
    )


# =========================================================================== providers


def _tool(env: ProbeEnv, spec: ToolSpec) -> Any:
    return tool_info(
        spec.exe,
        spec.dist,
        which=env.which,
        run=env.run,
        user_home=env.user_home,
        version_args=spec.version_args,
        timeout=env.remaining(10),
    )


def check_provider_cli(env: ProbeEnv, entry: ProviderEntry) -> CheckResult:
    spec = TOOLS[entry.kind]
    t = (f"provider.{entry.name}.cli", "providers", f"{entry.name} cli")
    if not _enabled(env, entry):
        return _not_enabled(t, entry)
    info = _tool(env, spec)
    if info.path is None:
        return _r(*t, FAIL, f"{spec.exe} is not installed ({spec.dist})", spec.install)
    ver = info.version or "version unknown"
    where = home_label(info.path, env.user_home)
    if spec.tested and info.version and version_tuple(info.version) < version_tuple(spec.tested):
        return _r(
            *t,
            WARN,
            f"{spec.dist} {ver} is older than {spec.tested}, the version gpu-router was "
            f"verified with",
            spec.upgrade,
            version=info.version,
            path=info.path,
        )
    return _r(*t, OK, f"{spec.dist} {ver} · {where}", version=info.version, path=info.path)


LIGHTNING_INSTALL = "uv tool install lightning-sdk"


def _uv_path(env: ProbeEnv) -> str | None:
    found = env.which("uv")
    if found:
        return found
    for cand in (
        env.user_home / ".local" / "bin" / "uv",
        Path("/opt/homebrew/bin/uv"),
        Path("/usr/local/bin/uv"),
    ):
        if cand.is_file():
            return str(cand)
    return None


def check_lightning_sdk(env: ProbeEnv, entry: ProviderEntry) -> CheckResult:
    """The adapter runs lightning-sdk in its own interpreter: providers.lightning.python,
    else the `uv tool install lightning-sdk` env, else `uv run --with lightning-sdk==<pin>`
    (providers/lightning/sdk.py). No `lightning` executable is needed."""
    from gpu_router.doctor.probe import _dist_version

    t = (f"provider.{entry.name}.cli", "providers", f"{entry.name} sdk")
    if not _enabled(env, entry):
        return _not_enabled(t, entry)
    pin: str | None
    try:
        from gpu_router.providers.lightning.sdk import DEFAULT_SDK_VERSION

        pin = DEFAULT_SDK_VERSION
    except ImportError:
        pin = None
    config = _config(env)
    settings = config.providers.get(entry.name) if config is not None else None
    python = (settings.model_extra or {}).get("python") if settings is not None else None
    if python:
        ok = os.path.exists(str(python))
        return _r(
            *t,
            OK if ok else FAIL,
            f"providers.{entry.name}.python = {python}" + ("" if ok else " (does not exist)"),
            None if ok else EDIT + _q(env.paths.config),
        )
    base = env.environ.get("UV_TOOL_DIR")
    root = Path(base).expanduser() if base else env.user_home / ".local" / "share" / "uv" / "tools"
    tool_python = root / "lightning-sdk" / "bin" / "python"
    if tool_python.is_file():
        ver = _dist_version(str(tool_python), "lightning-sdk")
        note = f" (gpu-router was built against {pin})" if pin and ver and ver != pin else ""
        return _r(
            *t,
            OK,
            f"lightning-sdk {ver or '(version unknown)'} · uv tool env{note}",
            version=ver,
        )
    uv = _uv_path(env)
    if uv is not None:
        return _r(
            *t,
            OK,
            f"no uv tool install; gpu-router runs lightning-sdk{'==' + pin if pin else ''} "
            "through `uv run` (the first call downloads it)",
        )
    return _r(
        *t,
        FAIL,
        "lightning-sdk is not installed and uv is missing",
        LIGHTNING_INSTALL,
    )


def _mode_problems(files: list[tuple[Path, int]]) -> list[Path]:
    return [p for p, mode in files if mode & 0o077]


def _chmod_fix(paths: list[Path], env: ProbeEnv) -> str:
    return "chmod 600 " + " ".join(_q(p) for p in paths)


#: credentials doctor sees only as environment variables of its own shell
ENV_ONLY_NOTE = "the daemon sees these only if it was started from this shell; launchd does not"


def check_kaggle_login(env: ProbeEnv, entry: ProviderEntry) -> CheckResult:
    t = (f"provider.{entry.name}.login", "providers", f"{entry.name} login")
    if not _enabled(env, entry):
        return _not_enabled(t, entry)
    config = _config(env)
    settings = config.providers.get(entry.name) if config is not None else None
    mode = str((settings.model_extra or {}).get("credentials", "auto")) if settings else "auto"
    names = env.secret_names()
    keychain = [n for n in ("kaggle", "KAGGLE_API_TOKEN") if n in names]
    cfg_dir = Path(env.environ.get("KAGGLE_CONFIG_DIR") or env.user_home / ".kaggle")
    files: list[tuple[Path, int]] = []
    for name in ("kaggle.json", "access_token"):
        st = stat_mode(cfg_dir / name)
        if st is not None:
            files.append((cfg_dir / name, st[0]))
    from_env = bool(
        (env.environ.get("KAGGLE_USERNAME") and env.environ.get("KAGGLE_KEY"))
        or env.environ.get("KAGGLE_API_TOKEN")
    )
    sources: list[str] = []
    if keychain:
        sources.append(f"Keychain ({', '.join(keychain)})")
    sources += [f"{home_label(p, env.user_home)} ({fmt_mode(m)})" for p, m in files]
    if from_env:
        sources.append("environment variables")
    if mode == "keychain" and not keychain:
        return _r(
            *t,
            FAIL,
            "providers.kaggle.credentials is `keychain` but the Keychain has no kaggle secret",
            "gpu login kaggle",
        )
    if not sources:
        return _r(
            *t,
            FAIL,
            "no kaggle credentials: create a token at kaggle.com → Settings → API → Create New "
            "Token, then move it into place",
            KAGGLE_FILE_FIX,
        )
    loose = _mode_problems(files)
    if loose:
        return _r(
            *t,
            WARN,
            f"{', '.join(home_label(p, env.user_home) for p in loose)} can be read by other users",
            _chmod_fix(loose, env),
            sources=sources,
        )
    used = "the Keychain copy" if keychain and mode != "cli" else None
    summary = "credentials: " + ", ".join(sources)
    if used and len(sources) > 1:
        summary += f" · gpu-router uses {used}"
    if from_env and not keychain and not files:
        # review fix: this is doctor's shell, not the daemon's environment
        return _r(
            *t,
            WARN,
            summary + f" · {ENV_ONLY_NOTE}",
            "gpu login kaggle",
            sources=sources,
            mode=mode,
            uses="env",
        )
    return _r(*t, OK, summary, sources=sources, mode=mode)


def _adc_path(env: ProbeEnv) -> Path:
    explicit = env.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if explicit:
        return Path(explicit).expanduser()
    return _gcloud_dir(env) / "application_default_credentials.json"


def _gcloud_dir(env: ProbeEnv) -> Path:
    explicit = env.environ.get("CLOUDSDK_CONFIG")
    return Path(explicit).expanduser() if explicit else env.user_home / ".config" / "gcloud"


def _colab_run(env: ProbeEnv, colab: str, args: list[str], cap: float) -> Any:
    """Run the colab CLI like the adapter does (ADC, a private HOME so nothing lands in the
    user's ~/.config/colab-cli, a session file only this probe uses), in a throwaway dir."""
    from gpu_router.providers.colab.cli import redact

    tmp = Path(tempfile.mkdtemp(prefix="gpu-doctor-colab-"))
    _COLAB_TMP.add(tmp)
    try:
        home = tmp / "home"
        home.mkdir(mode=0o700)
        child_env = {
            **env.environ,
            "HOME": str(home),
            "CLOUDSDK_CONFIG": str(_gcloud_dir(env)),
            "NO_COLOR": "1",
            "COLUMNS": "200",
            "PYTHONUNBUFFERED": "1",
        }
        argv = [colab, "--auth=adc", "--config", str(tmp / "sessions.json"), *args]
        # a little under what is left of the deadline (review fix): the subprocess is killed
        # and this thread's cleanup runs before the runner gives up on the row and exits
        budget = max(0.5, min(cap, env.remaining() - COLAB_CLEANUP_S))
        res = env.run(argv, budget, env=child_env)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        _COLAB_TMP.discard(tmp)
    return res, redact


#: seconds of the deadline kept for killing the colab CLI and removing its throwaway HOME
COLAB_CLEANUP_S = 1.0
#: throwaway colab HOMEs still in use; an atexit sweep removes any a timed-out probe left
_COLAB_TMP: set[Path] = set()


def _sweep_colab_tmp() -> None:
    for tmp in list(_COLAB_TMP):
        shutil.rmtree(tmp, ignore_errors=True)
    _COLAB_TMP.clear()


atexit.register(_sweep_colab_tmp)


def check_colab_login(env: ProbeEnv, entry: ProviderEntry) -> CheckResult:
    t = (f"provider.{entry.name}.login", "providers", f"{entry.name} login")
    if not _enabled(env, entry):
        return _not_enabled(t, entry)
    adc = _adc_path(env)
    st = stat_mode(adc)
    if st is None:
        return _r(
            *t,
            FAIL,
            "no Google application-default credentials (colab needs them, with the "
            "colaboratory scope)",
            ADC_LOGIN,
        )
    loose = _mode_problems([(adc, st[0])])
    info = _tool(env, TOOLS["colab"])
    if info.path is None:
        return _r(
            *t,
            SKIP,
            f"credentials at {home_label(adc, env.user_home)}; install the colab CLI to "
            "check their scopes",
        )
    res, redact = _colab_run(env, info.path, ["whoami"], 25)
    text = redact(res.text)
    if res.returncode is None:
        return _r(
            *t,
            WARN,
            f"`colab whoami` did not answer ({res.error or 'timed out'}); credentials at "
            f"{home_label(adc, env.user_home)}",
        )
    if not res.ok:
        last = next((ln for ln in reversed(text.splitlines()) if ln.strip()), "")
        rejected = any(
            k in text
            for k in ("invalid_grant", "RefreshError", "reauth", "invalid_rapt", "expired")
        )
        return _r(
            *t,
            FAIL if rejected else WARN,
            (
                "Google rejected the application-default credentials: "
                if rejected
                else "`colab whoami` failed: "
            )
            + last[:200],
            ADC_LOGIN if rejected else None,
        )
    scopes = [ln.strip()[2:].strip() for ln in text.splitlines() if ln.strip().startswith("- ")]
    email_m = re.search(r"^Email:\s*(\S+)", text, re.M)
    who = email_m.group(1) if email_m and "@" in email_m.group(1) else "your account"
    if COLAB_SCOPE not in scopes:
        return _r(
            *t,
            FAIL,
            f"the credentials for {who} lack the colaboratory scope: every colab session "
            "would get a 403",
            ADC_LOGIN,
            scopes=scopes,
        )
    if loose:
        return _r(
            *t,
            WARN,
            f"{home_label(adc, env.user_home)} can be read by other users",
            _chmod_fix(loose, env),
        )
    return _r(*t, OK, f"credentials for {who} have the colaboratory scope", account=who)


def _lightning_login_fix() -> str:
    try:
        from gpu_router.cli.login import login_app

        names = {
            (c.name or (c.callback.__name__ if c.callback else "")).replace("_", "-")
            for c in login_app.registered_commands
        }
        if "lightning" in names or "login-lightning" in names:
            return "gpu login lightning"
    except Exception:  # an older/newer login app: fall back to secrets
        return "gpu secrets set LIGHTNING_USER_ID && gpu secrets set LIGHTNING_API_KEY"
    return "gpu secrets set LIGHTNING_USER_ID && gpu secrets set LIGHTNING_API_KEY"


LIGHTNING_KEYS = ("LIGHTNING_USER_ID", "LIGHTNING_API_KEY")
LIGHTNING_MODES = ("auto", "keychain", "env", "file")


def check_lightning_login(env: ProbeEnv, entry: ProviderEntry) -> CheckResult:
    """Mirrors providers/lightning/credentials.resolve: `login_source` auto = Keychain, else
    the daemon's LIGHTNING_USER_ID + LIGHTNING_API_KEY, else ~/.lightning/credentials.json
    (stat only; the file is never opened here). LIGHTNING_AUTH_TOKEN is not a source: the
    adapter clears it from the SDK's environment."""
    t = (f"provider.{entry.name}.login", "providers", f"{entry.name} login")
    if not _enabled(env, entry):
        return _not_enabled(t, entry)
    config = _config(env)
    settings = config.providers.get(entry.name) if config is not None else None
    extra = (settings.model_extra or {}) if settings is not None else {}
    mode = str(extra.get("login_source") or extra.get("credentials") or "auto")
    login = _lightning_login_fix()
    if mode not in LIGHTNING_MODES:
        return _r(
            *t,
            FAIL,
            f"providers.{entry.name}.login_source is {mode!r}; use auto, keychain, env or file",
            EDIT + _q(env.paths.config),
        )
    names = env.secret_names()
    keychain = [n for n in LIGHTNING_KEYS if n in names]
    env_ok = all(env.environ.get(n, "").strip() for n in LIGHTNING_KEYS)
    cred = env.user_home / ".lightning" / "credentials.json"  # the `lightning login` file
    st = stat_mode(cred)
    found: dict[str, str] = {}
    if len(keychain) == 2:
        found["keychain"] = "Keychain (LIGHTNING_USER_ID, LIGHTNING_API_KEY)"
    if env_ok:
        found["env"] = "environment variables"
    if st is not None:
        found["file"] = f"{home_label(cred, env.user_home)} ({fmt_mode(st[0])})"
    sources = list(found.values())
    allowed = ("keychain", "env", "file") if mode == "auto" else (mode,)
    usable = [k for k in allowed if k in found]
    if len(keychain) == 1 and "keychain" in allowed and not usable:
        missing = "LIGHTNING_API_KEY" if keychain[0] == "LIGHTNING_USER_ID" else "LIGHTNING_USER_ID"
        return _r(*t, WARN, f"only {keychain[0]} is in the Keychain", f"gpu secrets set {missing}")
    if mode == "env" and not usable:
        return _r(
            *t,
            WARN,
            f"providers.{entry.name}.login_source is `env`: the daemon needs LIGHTNING_USER_ID "
            "and LIGHTNING_API_KEY in its environment (this shell has neither; the live check "
            "shows what the daemon sees)",
            login,
            mode=mode,
        )
    if mode != "auto" and not usable:
        where = {
            "keychain": "the Keychain has no Lightning keys",
            "file": f"there is no {home_label(cred, env.user_home)}",
        }[mode]
        other = f" (found: {', '.join(sources)})" if sources else ""
        return _r(
            *t,
            FAIL,
            f"providers.{entry.name}.login_source is `{mode}` but {where}{other}",
            login if mode == "keychain" else "lightning login",
            mode=mode,
        )
    if not usable:
        return _r(
            *t,
            FAIL,
            "no Lightning credentials yet (sign up at lightning.ai and verify your phone; no "
            "card), then log in",
            login,
        )
    if st is not None and st[0] & 0o077 and "file" in usable:
        return _r(
            *t,
            WARN,
            f"{home_label(cred, env.user_home)} can be read by other users",
            _chmod_fix([cred], env),
            sources=sources,
        )
    summary = "credentials: " + ", ".join(sources)
    if len(sources) > 1:
        summary += f" · gpu-router uses the {usable[0]} copy"
    if usable[0] == "file":
        summary += " (content is checked by the live check)"
    if usable[0] == "env":
        # review fix: this is doctor's shell, not the daemon's environment
        return _r(
            *t,
            WARN,
            summary + f" · {ENV_ONLY_NOTE}",
            login,
            sources=sources,
            mode=mode,
            uses="env",
        )
    return _r(*t, OK, summary, sources=sources, mode=mode, uses=usable[0])


def _login_fix(kind: str) -> str | None:
    return {
        "kaggle": "gpu login kaggle",
        "colab": ADC_LOGIN,
        "lightning": _lightning_login_fix(),
    }.get(kind)


def _quota_line(q: Any) -> str | None:
    if q is None or q.source != "live" or q.limit is None:
        return None
    unit = {"gpu_hours": "h", "credits": " credits", "usd": " USD"}.get(str(q.unit), "")
    return f"{_num(q.used)} of {_num(q.limit)}{unit} used"


def check_provider_live(env: ProbeEnv, entry: ProviderEntry) -> CheckResult:
    t = (f"provider.{entry.name}.live", "providers", f"{entry.name} live")
    if not env.daemon.up:
        return _r(*t, SKIP, "not tested live: the daemon is not running")
    try:
        views = {p.name: p for p in env.providers()}
    except GpuRouterError as exc:
        return _r(*t, WARN, f"cannot list providers: {exc.message}")
    view = views.get(entry.name)
    if view is None or not view.enabled:
        test = env.daemon.health is not None and env.daemon.health.test_mode
        why = " (test-mode daemon: real providers are off)" if test else ""
        return _r(*t, SKIP, f"{entry.name} is not enabled in the daemon{why}")
    try:
        checked = env.healthcheck(entry.name)
    except NotReady:
        return _r(*t, SKIP, "the daemon is still recovering jobs; try again in a moment")
    except GpuRouterError as exc:
        return _r(*t, WARN, f"healthcheck failed: {exc.message}")
    return _live_result(env, entry, checked, t)


def _live_result(
    env: ProbeEnv, entry: ProviderEntry, view: ProviderView, t: tuple[str, str, str]
) -> CheckResult:
    from gpu_router.models import ProviderHealth

    reason = view.health_reason or ""
    h = view.health
    if h is ProviderHealth.OK:
        summary = LIVE_OK.get(entry.kind, "healthcheck ok")
        try:
            quota = _quota_line(env.quotas().get(entry.name))
        except GpuRouterError:
            quota = None
        if quota:
            summary += f" · {quota}"
        return _r(*t, OK, summary, health=str(h))
    if h is ProviderHealth.DEGRADED:
        return _r(*t, WARN, reason or "degraded", health=str(h))
    if h is ProviderHealth.AUTH_REQUIRED:
        return _r(*t, FAIL, reason or "not logged in", _login_fix(entry.kind), health=str(h))
    if h is ProviderHealth.DISABLED:
        return _r(*t, SKIP, reason or "disabled", health=str(h))
    if "not installed" in reason:
        spec = TOOLS.get(entry.kind)
        return _r(*t, FAIL, reason, spec.install if spec else None, health=str(h))
    return _r(*t, WARN, reason or f"health {h}", health=str(h))


def _excluded(env: ProbeEnv) -> dict[str, str]:
    """name -> why, for the excluded services whose decision rests on the provider's own
    words (a `quote`): the ones a user might expect to see (Modal). The spec's paid-only
    list stays in `gpu providers`."""
    catalog = _catalog(env)
    listed = getattr(catalog, "excluded", None) if catalog is not None else None
    if not listed:
        return dict(EXCLUDED)
    out: dict[str, str] = {}
    for name, ex in listed.items():
        if not getattr(ex, "quote", None):
            continue
        why = f"{ex.reason}: “{ex.quote}”"
        if ex.source:
            why += f" ({ex.source})"
        out[name] = why
    return out


def check_excluded(env: ProbeEnv, name: str, why: str) -> CheckResult:
    t = (f"provider.{name}.excluded", "providers", f"{name} excluded")
    in_daemon = False
    if env.daemon.up:
        try:
            in_daemon = any(p.name == name and p.enabled for p in env.providers())
        except GpuRouterError:
            in_daemon = False
    if in_daemon:
        return _r(
            *t,
            FAIL,
            f"the daemon has {name} enabled, but it is excluded: {why}",
            RESTART,
        )
    config = _config(env)
    if config is not None and name in config.providers:
        return _r(
            *t,
            WARN,
            f"config.yaml still has settings for {name}, which is never used: {why}",
            EDIT + _q(env.paths.config),
        )
    stale = sorted(n for n in env.secret_names() if n.upper().startswith(f"{name.upper()}_"))
    if stale:
        return _r(
            *t,
            WARN,
            f"the Keychain still holds {', '.join(stale)} for {name}, which is never used",
            " && ".join(f"gpu secrets rm {n}" for n in stale),
        )
    return _r(*t, OK, f"never used: {why}{_largest_free(env)}")


def _largest_free(env: ProbeEnv) -> str:
    """'; no free provider fits jobs over 16GB VRAM (largest: kaggle)', from the catalog
    (data-driven like the router's no-fit reason); '' when the catalog cannot be read."""
    catalog = _catalog(env)
    entries = [e for e in catalog.listed("gpu") if e.kind != "local"] if catalog else []
    if not entries:
        return ""
    best = max(entries, key=lambda e: (e.max_vram_gb, -e.priority))
    return f"; no free provider fits jobs over {best.max_vram_gb:g}GB VRAM (largest: {best.name})"


def check_verify_lane(env: ProbeEnv, names: tuple[str, ...]) -> CheckResult:
    return _r(
        "provider.verify_at_signup",
        "providers",
        "verify at signup",
        SKIP,
        f"{', '.join(names)}: listed, never routed until a human checks them at signup "
        "(gpu providers shows what to check)",
    )


# =========================================================================== storage


def check_hf_hub(env: ProbeEnv) -> CheckResult:
    from importlib import metadata

    t = ("storage.huggingface_hub", "storage", "huggingface_hub")
    repo = env.gpu_router_repo()
    fix = (
        f"uv tool install --force --editable {_q(repo)}"
        if repo
        else "uv tool install --force gpu-router"
    )
    try:
        ver = metadata.version("huggingface_hub")
    except metadata.PackageNotFoundError:
        return _r(*t, FAIL, "huggingface_hub is missing from gpu-router's environment", fix)
    if version_tuple(ver) < version_tuple(HF_HUB_MIN):
        return _r(
            *t, FAIL, f"huggingface_hub {ver} is older than {HF_HUB_MIN} (storage buckets)", fix
        )
    return _r(*t, OK, f"huggingface_hub {ver}", version=ver)


def _storage_backend(env: ProbeEnv) -> str | None:
    config = _config(env)
    return None if config is None else config.checkpoint.backend


def check_hf_token(env: ProbeEnv) -> CheckResult:
    from gpu_router.errors import SecretsError

    t = ("storage.hf_token", "storage", "HF token")
    backend = _storage_backend(env)
    if backend in ("off", "local"):
        return _r(*t, SKIP, f"checkpoint.backend is {backend}: no Hugging Face storage")
    if "HF_TOKEN" not in env.secret_names():
        return _r(
            *t,
            WARN,
            "no Hugging Face token: checkpoints stay on this Mac, so a job on kaggle or colab "
            "cannot resume on another provider",
            "gpu login hf",
        )
    from gpu_router.checkpoint.tokens import admin_token

    try:
        token = admin_token()
    except SecretsError as exc:
        return _r(*t, WARN, f"cannot read HF_TOKEN: {exc.message}", "security unlock-keychain")
    if token is None:
        return _r(
            *t, WARN, "secrets.index lists HF_TOKEN but the Keychain has none", "gpu login hf"
        )
    name, problem = env.hf_whoami(token.get_secret_value())
    if problem:
        return _r(*t, FAIL, f"{problem} (HF_TOKEN)", "gpu login hf")
    if name is None:
        return _r(*t, WARN, "could not reach Hugging Face to check HF_TOKEN (network?)")
    if env.hf_role(token.get_secret_value()) == "read":
        # review fix: a bare whoami said "works" for a read-only token
        return _r(
            *t,
            WARN,
            f"HF_TOKEN (user {name}) is read-only: checkpoints need write access, so saving "
            "them fails",
            "gpu login hf",
            user=name,
        )
    return _r(*t, OK, f"HF_TOKEN works (user {name})", user=name)


def check_hf_remote(env: ProbeEnv) -> CheckResult:
    from gpu_router.errors import SecretsError

    t = ("storage.hf_remote", "storage", "HF remote token")
    backend = _storage_backend(env)
    if backend in ("off", "local"):
        return _r(*t, SKIP, f"checkpoint.backend is {backend}: remote runs get no storage")
    if "HF_TOKEN_REMOTE" in env.secret_names():
        # review fix: the Keychain name alone said "can save and resume"; check the token
        from gpu_router import secrets as secret_store

        fix = "gpu login hf --remote"
        try:
            value = secret_store.get_secret("HF_TOKEN_REMOTE")
        except SecretsError as exc:
            return _r(*t, WARN, f"cannot read HF_TOKEN_REMOTE: {exc.message}", fix)
        if value is None:
            return _r(
                *t, WARN, "secrets.index lists HF_TOKEN_REMOTE but the Keychain has none", fix
            )
        name, problem = env.hf_whoami(value)
        if problem:
            return _r(*t, FAIL, f"{problem} (HF_TOKEN_REMOTE)", fix)
        if name is None:
            return _r(
                *t,
                WARN,
                "HF_TOKEN_REMOTE is stored (not verified: could not reach Hugging Face)",
            )
        if env.hf_role(value) == "read":
            return _r(
                *t,
                WARN,
                f"HF_TOKEN_REMOTE (user {name}) is read-only: remote runs cannot save checkpoints",
                fix,
            )
        return _r(
            *t,
            OK,
            f"HF_TOKEN_REMOTE works (user {name}): remote runs can save and resume checkpoints",
        )
    return _r(
        *t,
        WARN,
        "no HF_TOKEN_REMOTE: runs on kaggle and colab get no checkpoint storage (a fine-grained "
        "token that can only write your buckets)",
        "gpu login hf --remote",
    )


# =========================================================================== inference


def _infer_rejected(env: ProbeEnv) -> dict[str, str]:
    """provider -> "key rejected until ..." from the daemon's inference ledger (a 401/403
    the daemon saw); {} when the daemon is down or predates the inference lane."""
    client = env.daemon.client
    if not env.daemon.up or client is None:
        return {}
    try:
        data = client.request("GET", "/infer/quota", timeout_s=env.remaining(5))
    except GpuRouterError:
        return {}
    out: dict[str, str] = {}
    for raw in data or []:
        if not isinstance(raw, dict):
            continue
        for text in (raw.get("blocked") or {}).values():
            if str(text).startswith("key rejected"):
                out[str(raw.get("provider"))] = str(text)
                break
    return out


def check_inference(env: ProbeEnv) -> CheckResult:
    """The inference lane (D50: Groq, Cloudflare, Gemini, HF) is optional: keys by Keychain
    NAME only (secrets.index), never read or sent anywhere from here."""
    from gpu_router.inference.catalog import load_inference_catalog

    t = ("inference.keys", "inference", "inference keys")
    catalog = _catalog(env)
    if catalog is None:
        return _r(*t, SKIP, "providers.yaml is invalid (see the config check)")
    lane = load_inference_catalog(catalog)
    if lane.problems:
        name, why = next(iter(sorted(lane.problems.items())))
        more = f" (+{len(lane.problems) - 1} more)" if len(lane.problems) > 1 else ""
        return _r(
            *t,
            WARN,
            f"inference.{name} in providers.yaml is skipped: {why}{more}",
            EDIT + _q(env.paths.user_providers),
            problems=lane.problems,
        )
    entries = sorted((e for e in lane.entries.values() if e.routable), key=lambda e: e.priority)
    if not entries:
        return _r(*t, SKIP, "no inference providers in providers.yaml")
    names = env.secret_names()
    ready: list[str] = []
    partial: list[tuple[str, str, list[str]]] = []
    unset: list[tuple[str, str]] = []
    for e in entries:
        need = list(e.secrets.values())
        missing = [n for n in need if n not in names]
        if not missing:
            ready.append(e.name)
        elif len(missing) < len(need):
            partial.append((e.name, e.login_name, missing))
        else:
            unset.append((e.name, e.login_name))
    rejected = _infer_rejected(env)
    if rejected:
        name = sorted(rejected)[0]
        login = lane.entries[name].login_name if name in lane.entries else name
        return _r(
            *t,
            WARN,
            f"{name}: {rejected[name]} (the provider refused the stored key)",
            f"gpu login {login}",
            rejected=rejected,
        )
    if partial:
        name, login, missing = partial[0]
        return _r(
            *t,
            WARN,
            f"{name} is half set up: {', '.join(missing)} is not in the Keychain",
            f"gpu login {login}",
        )
    if not ready:
        return _r(
            *t,
            SKIP,
            "optional: no inference keys yet (" + ", ".join(n for n, _ in unset) + "); "
            "free LLM calls for evals, `gpu providers` lists the limits",
            f"gpu login {unset[0][1]}",
        )
    summary = "keys for " + ", ".join(ready)
    if unset:
        summary += " · not set: " + ", ".join(f"{n} (gpu login {login})" for n, login in unset)
    return _r(*t, OK, summary, ready=ready, unset=[n for n, _ in unset])


# =========================================================================== limits


_WEEKDAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")


def live_anchor(reset: str, resets_at: float) -> str | None:
    """providers.yaml anchor text for a live reset instant ("sat 00:00 UTC")."""
    dt = datetime.fromtimestamp(round(resets_at / 60) * 60, tz=UTC)
    hm = f"{dt.hour:02d}:{dt.minute:02d} UTC"
    if reset == "weekly":
        return f"{_WEEKDAYS[dt.weekday()]} {hm}"
    if reset == "monthly":
        return f"day {dt.day} {hm}"
    if reset == "daily":
        return hm
    return None


def check_limits(env: ProbeEnv, entry: ProviderEntry) -> CheckResult:
    from gpu_router.quota.windows import current_window

    t = (f"limits.{entry.name}", "limits", f"{entry.name} limits")
    if not env.daemon.up:
        return _r(*t, SKIP, "needs the daemon (live quota readings)")
    try:
        views = {p.name: p for p in env.providers()}
    except GpuRouterError as exc:
        return _r(*t, WARN, f"cannot list providers: {exc.message}")
    if entry.name not in views or not views[entry.name].enabled:
        return _r(*t, SKIP, f"{entry.name} is not enabled in the daemon")
    q = env.quotas().get(entry.name)
    basis = str(q.detail.get("basis") or "") if q is not None else ""
    if q is None or q.source != "live" or basis == "unlimited":
        what = "no live reading yet"
        if entry.quota.reset == "unknown":
            what = f"{entry.name} does not publish its limits"
        return _r(*t, SKIP, f"{what}; providers.yaml keeps its numbers")
    # the reset instant is the provider's own only while the reading is current (basis
    # live / live+history); after its reset the ledger derives it from providers.yaml
    own_reset = basis in ("live", "live+history", "")
    drift: list[DriftItem] = []
    spec = entry.quota
    unit = str(q.unit)
    if str(spec.unit) != unit:
        drift.append(
            DriftItem(
                provider=entry.name,
                key="quota.unit",
                catalog=str(spec.unit),
                live=unit,
                note=f"{entry.name} reports its quota in {unit}, providers.yaml says {spec.unit}",
            )
        )
    elif q.limit is not None and (spec.limit is None or abs(spec.limit - q.limit) > 0.01):
        drift.append(
            DriftItem(
                provider=entry.name,
                key="quota.limit",
                catalog=spec.limit,
                live=int(q.limit) if float(q.limit).is_integer() else round(q.limit, 2),
                note=f"limit: providers.yaml "
                f"{_num(spec.limit) if spec.limit is not None else 'unknown'}, "
                f"{entry.name} reports {_num(q.limit)} ({unit})",
            )
        )
    anchor_live = live_anchor(spec.reset, q.resets_at) if q.resets_at and own_reset else None
    if anchor_live is not None and q.resets_at is not None:
        window = current_window(spec.reset, spec.reset_anchor, q.observed_at)
        expected = window.resets_at
        if expected is None or abs(expected - q.resets_at) > 3600:
            drift.append(
                DriftItem(
                    provider=entry.name,
                    key="quota.reset_anchor",
                    catalog=spec.reset_anchor,
                    live=anchor_live,
                    note=f"reset: providers.yaml {spec.reset_anchor or 'unknown'}, "
                    f"{entry.name} resets {anchor_live}",
                )
            )
    per = {"weekly": "/week", "monthly": "/month", "daily": "/day"}.get(spec.reset, "")
    live_text = f"{_num(q.limit)}{per}" if q.limit is not None else "no limit reported"
    if drift:
        return _r(
            *t,
            WARN,
            "drift from providers.yaml: " + "; ".join(d.note for d in drift),
            "gpu doctor --update-catalog",
            drift=[d.model_dump(mode="json") for d in drift],
        )
    reset = f", resets {anchor_live}" if anchor_live else ""
    return _r(*t, OK, f"live {live_text} ({unit}){reset} matches providers.yaml")


_BOX = re.compile(r"[│╭╮╰╯─┃━┏┓┗┛]")


def colab_gpus_from_help(text: str) -> list[str] | None:
    flat = " ".join(_BOX.sub(" ", text).split())
    m = re.search(r"GPU accelerator variant\.\s*Supported:\s*([A-Za-z0-9][A-Za-z0-9 ,-]*?)\.", flat)
    if not m:
        return None
    return [g.strip() for g in m.group(1).split(",") if g.strip()]


def check_colab_gpus(env: ProbeEnv, entry: ProviderEntry) -> CheckResult:
    t = (f"limits.{entry.name}.gpus", "limits", f"{entry.name} GPUs")
    if not _enabled(env, entry):
        return _not_enabled(t, entry)
    info = _tool(env, TOOLS["colab"])
    if info.path is None:
        return _r(*t, SKIP, "the colab CLI is not installed")
    res, _ = _colab_run(env, info.path, ["new", "--help"], 15)
    offered = colab_gpus_from_help(res.text) if res.ok else None
    if offered is None:
        return _r(*t, WARN, "could not read the GPU list from `colab new --help`")
    listed = [g.name for g in entry.gpus]
    upper = {g.upper() for g in offered}
    gone = [g for g in listed if g.upper() not in upper]
    extra = [g for g in offered if g.upper() not in {x.upper() for x in listed}]
    if gone:
        return _r(
            *t,
            WARN,
            f"providers.yaml lists {', '.join(gone)} for {entry.name}, but the colab CLI no "
            f"longer offers it (offers {', '.join(offered)})",
            EDIT + _q(env.paths.user_providers),
            offered=offered,
        )
    note = f"; the CLI also offers {', '.join(extra)} (paid tiers)" if extra else ""
    return _r(
        *t,
        OK,
        f"providers.yaml lists {', '.join(listed)}, which the colab CLI offers{note}",
        offered=offered,
    )


# =========================================================================== local


#: Must be 0600 even inside the 0700 data dir (defense in depth): the bearer token, the job
#: database, secret names, and config files that may name private paths.
_SECRET_FILES = (
    "daemon.token",
    "gpu.db",
    "gpu.db-wal",
    "gpu.db-shm",
    "secrets.index",
    "config.yaml",
    "providers.yaml",
)


def check_perms(env: ProbeEnv) -> CheckResult:
    """The data dir is 0700 and owned by you; secret files are 0600. Other files may be
    0644 by design (daemon.json, state.json: the 0700 dir already keeps other users out),
    but never group/other-writable."""
    t = ("local.perms", "local", "data dir")
    home = env.paths.home
    if not home.exists():
        return _r(*t, SKIP, "no data dir yet: the daemon creates it (0700) on first start")
    me = os.getuid()
    candidates = [home]
    try:
        candidates += sorted(home.iterdir())
    except OSError as exc:
        return _r(*t, FAIL, f"cannot list {home}: {exc.strerror or exc}")
    foreign: list[Path] = []
    bad_dirs: list[Path] = []
    bad_secret: list[Path] = []
    writable: list[Path] = []
    checked = 0
    for p in candidates:
        try:
            st = p.lstat()
        except OSError:
            continue
        if p.is_symlink():
            continue
        checked += 1
        if st.st_uid != me:
            foreign.append(p)
        mode = st.st_mode & 0o777
        if p.is_dir():
            if mode & 0o077:
                bad_dirs.append(p)
        elif p.name in _SECRET_FILES and mode & 0o077:
            bad_secret.append(p)
        elif mode & 0o022:
            writable.append(p)
    label = home_label(home, env.user_home)
    if foreign:
        return _r(
            *t,
            FAIL,
            f"{len(foreign)} item(s) in {label} belong to another user",
            f'sudo chown -R "$(id -un)" {_q(home)}',
        )
    if not (bad_dirs or bad_secret or writable):
        return _r(*t, OK, f"{label} is private (0700; token and database 0600)")
    parts: list[str] = []
    if bad_dirs:
        parts.append("chmod 700 " + " ".join(_q(p) for p in bad_dirs))
    if bad_secret or writable:
        parts.append("chmod 600 " + " ".join(_q(p) for p in (*bad_secret, *writable)))
    what: list[str] = []
    if home in bad_dirs:
        what.append(f"other users can open {label}")
    others = [p.name for p in bad_dirs if p != home] + [p.name for p in bad_secret]
    if others:
        what.append(f"other users can read {', '.join(others[:6])}")
    if writable:
        what.append(f"other users can write {', '.join(p.name for p in writable[:6])}")
    severe = home in bad_dirs or bool(bad_secret) or bool(writable)
    summary = "; ".join(what)
    if home in bad_dirs or bad_secret:
        summary += " (the daemon token and job database live here)"
    return _r(*t, FAIL if severe else WARN, summary, " && ".join(parts))


def check_disk(env: ProbeEnv) -> CheckResult:
    t = ("local.disk", "local", "disk space")
    target = env.paths.home
    while not target.exists() and target != target.parent:
        target = target.parent
    try:
        usage = shutil.disk_usage(target)
    except OSError as exc:
        return _r(*t, WARN, f"cannot read free space: {exc.strerror or exc}")
    free = usage.free / 1e9
    fix = f"du -sh {_q(env.paths.home)}/* | sort -h"
    if free < DISK_FAIL_GB:
        return _r(*t, FAIL, f"only {free:.1f} GB free: bundles, venvs and outputs will fail", fix)
    if free < DISK_WARN_GB:
        return _r(*t, WARN, f"{free:.1f} GB free on the data dir's volume", fix)
    return _r(*t, OK, f"{free:.0f} GB free on the data dir's volume", free_gb=round(free, 1))


def check_config(env: ProbeEnv) -> CheckResult:
    t = ("local.config", "local", "config")
    try:
        config = env.config()
    except ConfigError as exc:
        return _r(*t, FAIL, exc.message, EDIT + _q(env.paths.config))
    problems: list[str] = []
    try:
        from gpu_router.notify.settings import notify_settings

        notify_settings(config.notifications)
    except ConfigError as exc:
        problems.append(exc.message)
    try:
        from gpu_router.router.settings import routing_settings

        routing_settings(config.routing)
    except (ConfigError, ValueError) as exc:
        problems.append(getattr(exc, "message", str(exc)))
    try:
        from gpu_router.policy import default_policy

        default_policy(config)
    except (ConfigError, ValueError) as exc:
        problems.append(getattr(exc, "message", str(exc)))
    if problems:
        return _r(*t, FAIL, "; ".join(problems)[:300], EDIT + _q(env.paths.config))
    try:
        env.catalog()
    except ConfigError as exc:
        return _r(*t, FAIL, exc.message, EDIT + _q(env.paths.user_providers))
    have = [p.name for p in (env.paths.config, env.paths.user_providers) if p.exists()]
    what = " and ".join(have) + " valid" if have else "defaults (no config.yaml yet)"
    return _r(*t, OK, what)


def check_git(env: ProbeEnv) -> CheckResult:
    t = ("local.git", "local", "git")
    path = env.which("git")
    if path is None:
        return _r(
            *t, FAIL, "git is missing: gpu run packages git-tracked files", "xcode-select --install"
        )
    res = env.run([path, "--version"], env.remaining(5))
    from gpu_router.doctor.probe import parse_version

    ver = parse_version(res.text) if res.ok else None
    return _r(*t, OK, f"git {ver or '(version unknown)'}", version=ver)


def check_uv(env: ProbeEnv) -> CheckResult:
    t = ("local.uv", "local", "uv")
    path = env.which("uv")
    if path is None:
        for cand in (
            env.user_home / ".local" / "bin" / "uv",
            env.user_home / ".cargo" / "bin" / "uv",
        ):
            if cand.is_file() and os.access(cand, os.X_OK):
                path = str(cand)
                break
    if path is None:
        return _r(
            *t,
            WARN,
            "uv is missing: local runs fall back to venv + pip (slower)",
            "curl -LsSf https://astral.sh/uv/install.sh | sh",
        )
    res = env.run([path, "--version"], env.remaining(5))
    from gpu_router.doctor.probe import parse_version

    ver = parse_version(res.text) if res.ok else None
    return _r(*t, OK, f"uv {ver or '(version unknown)'}", version=ver)


# =========================================================================== integration


def _install_cmd(env: ProbeEnv) -> str:
    repo = env.gpu_router_repo()
    return f"uv tool install --editable {_q(repo)}" if repo else "uv tool install gpu-router"


def check_gpu_on_path(env: ProbeEnv) -> CheckResult:
    t = ("integration.gpu_path", "integration", "gpu on PATH")
    found = env.which("gpu")
    if found is None:
        installed = _uv_tool_bin(env) / "gpu"
        if os.path.exists(installed):
            # review fix: installed as a uv tool, its bin dir is just not on PATH; installing
            # again answers "already installed" and changes nothing
            return _r(
                *t,
                WARN,
                f"`gpu` is installed ({home_label(installed, env.user_home)}) but that dir is "
                "not on PATH: the plugin's MCP server, the status line and Codex need it "
                "(open a new terminal after the fix)",
                "uv tool update-shell",
                path=str(installed),
            )
        return _r(
            *t,
            WARN,
            "`gpu` is not on PATH: the plugin's MCP server, the status line and Codex need it",
            _install_cmd(env),
        )
    return _r(*t, OK, home_label(found, env.user_home), path=found)


def _uv_tool_bin(env: ProbeEnv) -> Path:
    """Where `uv tool install` puts executables: UV_TOOL_BIN_DIR, else $XDG_BIN_HOME, else
    ~/.local/bin."""
    for var in ("UV_TOOL_BIN_DIR", "XDG_BIN_HOME"):
        raw = env.environ.get(var)
        if raw:
            return Path(raw).expanduser()
    return env.user_home / ".local" / "bin"


def check_statusline(env: ProbeEnv) -> CheckResult:
    from gpu_router.statusline import install

    t = ("integration.statusline", "integration", "status line")
    settings = env.claude_dir / "settings.json"
    label = home_label(settings, env.user_home)
    if not settings.exists():
        return _r(
            *t,
            WARN,
            f"no {label}: gpu rows are not in the Claude Code status line",
            "gpu statusline install",
        )
    info = install.status(settings, env.paths.home, user_home=env.user_home)
    if info.get("error"):
        return _r(*t, WARN, f"cannot read {label}: {info['error']}")
    if not info.get("installed"):
        return _r(
            *t,
            WARN,
            "gpu rows are not in the Claude Code status line (install shows the settings.json "
            "diff and asks first)",
            "gpu statusline install",
        )
    if info.get("other_home"):
        other = info["other_home"]
        return _r(
            *t,
            WARN,
            "the status line runs the wrapper of another data dir "
            f"({home_label(other, env.user_home)})",
            f"GPU_ROUTER_HOME={_q(other)} gpu statusline uninstall && gpu statusline install",
        )
    if not info.get("wrapper_exists"):
        return _r(
            *t,
            WARN,
            "installed, but the wrapper file is gone: Claude Code shows only your own lines",
            "gpu statusline install",
        )
    return _r(*t, OK, "installed: gpu rows appear under your status line while GPU work is active")


def _packaged_plugin_version(env: ProbeEnv) -> str | None:
    repo = env.gpu_router_repo()
    if repo is None:
        return env.version
    try:
        data = json.loads((repo / "plugin" / ".claude-plugin" / "plugin.json").read_text())
    except (OSError, ValueError):
        return env.version
    v = data.get("version")
    return str(v) if v else env.version


def check_plugin(env: ProbeEnv) -> CheckResult:
    t = ("integration.plugin", "integration", "Claude Code plugin")
    repo = env.gpu_router_repo()
    install = (
        f"claude plugin marketplace add {_q(repo / 'plugin')} && claude plugin install {PLUGIN_KEY}"
        if repo
        else f"claude plugin marketplace add {GITHUB_MARKETPLACE} && "
        f"claude plugin install {GITHUB_PLUGIN_KEY}"
    )
    f = env.claude_dir / "plugins" / "installed_plugins.json"
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
    except FileNotFoundError:
        data = {}
    except (OSError, ValueError) as exc:
        return _r(*t, WARN, f"cannot read {home_label(f, env.user_home)}: {exc}")
    plugins = data.get("plugins") if isinstance(data, dict) else None
    plugins = plugins if isinstance(plugins, dict) else {}
    key = (
        PLUGIN_KEY
        if PLUGIN_KEY in plugins
        else next((k for k in plugins if str(k).startswith("gpu-router@")), None)
    )
    if key is None:
        return _r(
            *t,
            WARN,
            "not installed: Claude Code has no gpu_* MCP tools, gpu-router skill or /gpu-* "
            "commands",
            install,
        )
    entries = plugins.get(key) or []
    first = (
        entries[0] if isinstance(entries, list) and entries and isinstance(entries[0], dict) else {}
    )
    version = str(first.get("version") or "")
    try:
        settings = json.loads((env.claude_dir / "settings.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        settings = {}
    enabled = (
        (settings.get("enabledPlugins") or {}).get(key) if isinstance(settings, dict) else None
    )
    if enabled is False:
        return _r(*t, WARN, f"{key} is installed but disabled", f"claude plugin enable {key}")
    want = _packaged_plugin_version(env)
    if version and want and version != want:
        return _r(
            *t,
            WARN,
            f"installed {version}, this gpu-router ships {want}",
            f"claude plugin marketplace update {str(key).partition('@')[2]} && "
            f"claude plugin update {key}",
        )
    return _r(*t, OK, f"{key} installed" + (f" ({version})" if version else ""))


def check_colab_skill(env: ProbeEnv) -> CheckResult:
    t = ("integration.colab_skill", "integration", "colab skill")
    skill = env.claude_dir / "skills" / "colab"
    if not skill.exists():
        return _r(*t, OK, "no global colab skill competing for “run this on a GPU”")
    # review fix: the old `read -r -p` fix was bash-only (zsh: "-p: no coprocess"); the
    # wizard's own step shows what moves where and asks first, in any shell
    fix = "gpu setup --only integration.colab_skill"
    return _r(
        *t,
        WARN,
        f"{home_label(skill, env.user_home)} also claims “run this on a GPU” and drives colab "
        "directly, past the quota ledger (it is kept, just moved)",
        fix,
    )


def check_codex(env: ProbeEnv) -> CheckResult:
    import tomllib

    t = ("integration.codex", "integration", "Codex")
    if env.which("codex") is None:
        return _r(*t, SKIP, "Codex is not installed")
    cfg = env.codex_dir / "config.toml"
    try:
        data = tomllib.loads(cfg.read_text(encoding="utf-8"))
    except FileNotFoundError:
        data = {}
    except (OSError, ValueError) as exc:
        return _r(*t, WARN, f"cannot read {home_label(cfg, env.user_home)}: {exc}")
    servers = data.get("mcp_servers") or {}
    if isinstance(servers, dict) and "gpu-router" in servers:
        return _r(*t, OK, "Codex has the gpu-router MCP server")
    return _r(
        *t,
        WARN,
        "Codex does not have the gpu-router MCP server (docs/codex/AGENTS-section.md has the "
        "config.toml entry with tool_timeout_sec = 330)",
        "codex mcp add gpu-router -- gpu mcp",
    )


def check_notifications(env: ProbeEnv) -> CheckResult:
    from gpu_router.notify.backends import NullBackend, choose_backend
    from gpu_router.notify.settings import notify_settings

    t = ("integration.notifications", "integration", "notifications")
    config = _config(env)
    if config is None:
        return _r(*t, SKIP, "config.yaml is invalid (see the config check)")
    try:
        settings = notify_settings(config.notifications)
    except ConfigError as exc:
        return _r(*t, FAIL, exc.message, EDIT + _q(env.paths.config))
    kinds = settings.enabled_kinds()
    health = env.daemon.health if env.daemon.up else None
    running = health.notifications if health is not None else None
    if running is not None:
        # the running daemon's own pick: a launchd daemon's PATH is not this shell's
        if running.startswith("off: "):
            why = running.removeprefix("off: ")
            if "not installed" in why:
                return _r(*t, WARN, f"the daemon: {why}", "brew install terminal-notifier")
            return _r(*t, SKIP, f"off in the daemon: {why}")
        if not kinds:
            return _r(*t, SKIP, "every event type is switched off in config.yaml")
        return _r(
            *t,
            OK,
            f"on via {running} (the daemon's) for {', '.join(kinds)} · gpu notify test shows one",
            backend=running,
            events=kinds,
        )
    backend = choose_backend(
        settings, test_mode=config.test_mode, environ=env.environ, which=env.which
    )
    if isinstance(backend, NullBackend):
        if "not installed" in backend.why:
            return _r(*t, WARN, backend.why, "brew install terminal-notifier")
        return _r(*t, SKIP, f"off: {backend.why}")
    if not kinds:
        return _r(*t, SKIP, "every event type is switched off in config.yaml")
    return _r(
        *t,
        OK,
        f"on via {backend.name} for {', '.join(kinds)} · gpu notify test shows one",
        backend=backend.name,
        events=kinds,
    )


# =========================================================================== task list


def _entry_tasks(entry: ProviderEntry) -> list[Task]:
    n = entry.name
    out: list[Task] = []
    cli: Callable[..., CheckResult] | None = None
    if entry.kind in TOOLS:
        cli = check_provider_cli
    elif entry.kind == "lightning":
        cli = check_lightning_sdk
    if cli is not None:
        out.append(Task(f"provider.{n}.cli", "providers", f"{n} cli", partial(cli, entry=entry)))
    login = {
        "kaggle": check_kaggle_login,
        "colab": check_colab_login,
        "lightning": check_lightning_login,
    }.get(entry.kind)
    if login is not None:
        out.append(
            Task(f"provider.{n}.login", "providers", f"{n} login", partial(login, entry=entry))
        )
    out.append(
        Task(
            f"provider.{n}.live",
            "providers",
            f"{n} live",
            partial(check_provider_live, entry=entry),
        )
    )
    return out


def default_tasks(env: ProbeEnv) -> list[Task]:
    tasks = [
        Task("daemon.running", "daemon", "running", check_daemon_running),
        Task("daemon.version", "daemon", "version", check_daemon_version),
        Task("daemon.launchd", "daemon", "launchd", check_launchd),
        Task("daemon.state", "daemon", "state.json", check_state_file),
    ]
    catalog = _catalog(env)
    excluded = _excluded(env)
    entries: list[ProviderEntry] = []
    verify: list[str] = []
    if catalog is not None:
        for entry in catalog.ordered():
            lane = getattr(entry, "lane", "manual" if entry.manual_only else "gpu")
            if lane == "verify":
                verify.append(entry.name)
                continue
            if lane != "gpu" or entry.name in excluded:
                continue
            if entry.test_only:
                # the fakes only matter when a (test-mode) daemon registered them
                health = env.daemon.health
                if health is None or not health.test_mode:
                    continue
            entries.append(entry)
    for entry in entries:
        tasks += _entry_tasks(entry)
    for name, why in excluded.items():
        tasks.append(
            Task(
                f"provider.{name}.excluded",
                "providers",
                f"{name} excluded",
                partial(check_excluded, name=name, why=why),
            )
        )
    if verify:
        tasks.append(
            Task(
                "provider.verify_at_signup",
                "providers",
                "verify at signup",
                partial(check_verify_lane, names=tuple(verify)),
            )
        )
    tasks += [
        Task("storage.huggingface_hub", "storage", "huggingface_hub", check_hf_hub),
        Task("storage.hf_token", "storage", "HF token", check_hf_token),
        Task("storage.hf_remote", "storage", "HF remote token", check_hf_remote),
        Task("inference.keys", "inference", "inference keys", check_inference),
    ]
    for entry in entries:
        if entry.test_only or entry.kind == "local":
            continue
        tasks.append(
            Task(
                f"limits.{entry.name}",
                "limits",
                f"{entry.name} limits",
                partial(check_limits, entry=entry),
            )
        )
        if entry.kind == "colab":
            tasks.append(
                Task(
                    f"limits.{entry.name}.gpus",
                    "limits",
                    f"{entry.name} GPUs",
                    partial(check_colab_gpus, entry=entry),
                )
            )
    tasks += [
        Task("local.config", "local", "config", check_config),
        Task("local.perms", "local", "data dir", check_perms),
        Task("local.disk", "local", "disk space", check_disk),
        Task("local.git", "local", "git", check_git),
        Task("local.uv", "local", "uv", check_uv),
        Task("integration.gpu_path", "integration", "gpu on PATH", check_gpu_on_path),
        Task("integration.statusline", "integration", "status line", check_statusline),
        Task("integration.plugin", "integration", "Claude Code plugin", check_plugin),
        Task("integration.colab_skill", "integration", "colab skill", check_colab_skill),
        Task("integration.codex", "integration", "Codex", check_codex),
        Task("integration.notifications", "integration", "notifications", check_notifications),
    ]
    return tasks
