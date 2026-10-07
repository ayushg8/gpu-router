"""launchd integration (phase 1 writer; `gpu setup` in phase 8 calls it; owner: group C).

Label `dev.gpu-router.daemon`; plist at ~/Library/LaunchAgents/dev.gpu-router.daemon.plist:
ProgramArguments = [<abs path of the `gpu` executable>, "daemon", "run", "--launchd"],
RunAtLoad true, KeepAlive {SuccessfulExit: false} (restart after crashes, not after
`gpu daemon stop`; `--launchd` makes "already running" / "port taken" exit 0 so they do not
respawn in a loop), ThrottleInterval 30,
StandardOutPath/StandardErrorPath = paths.launchd_log, EnvironmentVariables with
GPU_ROUTER_HOME only when it is set (never secrets). Load with
`launchctl bootstrap gui/<uid> <plist>`, unload with `launchctl bootout gui/<uid>/<label>`.
"""

from __future__ import annotations

import os
import plistlib
import subprocess
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from gpu_router.paths import ENV_HOME, Paths

LABEL = "dev.gpu-router.daemon"
LAUNCHCTL = "/bin/launchctl"
_LAUNCHCTL_TIMEOUT_S = 30
BOOTSTRAP_BUSY = 5  # launchctl bootstrap's code while the booted-out instance still exits
BOOTSTRAP_RETRY_S = (2.0, 4.0, 8.0, 16.0)


def plist_path(home: Path | None = None) -> Path:
    """~/Library/LaunchAgents/<LABEL>.plist (home = user's home dir; tests pass a tmp dir)."""
    base = Path.home() if home is None else home
    return base / "Library" / "LaunchAgents" / f"{LABEL}.plist"


def render_plist(*, executable: Path, paths: Paths, env_home: str | None) -> bytes:
    """plistlib.dumps of the agent definition described above."""
    agent: dict[str, Any] = {
        "Label": LABEL,
        "ProgramArguments": [str(executable), "daemon", "run", "--launchd"],
        "RunAtLoad": True,
        "KeepAlive": {"SuccessfulExit": False},
        "ThrottleInterval": 30,
        "StandardOutPath": str(paths.launchd_log),
        "StandardErrorPath": str(paths.launchd_log),
        # Standard, not Background (2026-10-06): background QoS let macOS starve the daemon
        # and every provider CLI it starts whenever the Mac was busy (1 s of CPU in 14 min
        # at load 150), which is exactly when agents send work to cloud GPUs. The daemon
        # idles between polls, so normal priority costs the foreground nothing.
        "ProcessType": "Standard",
    }
    if env_home:
        agent["EnvironmentVariables"] = {ENV_HOME: env_home}
    return plistlib.dumps(agent, sort_keys=True)


def _domain() -> str:
    return f"gui/{os.getuid()}"


def _launchctl(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [LAUNCHCTL, *args],
        capture_output=True,
        text=True,
        timeout=_LAUNCHCTL_TIMEOUT_S,
        check=False,
    )


def is_installed(home: Path | None = None) -> bool:
    return plist_path(home).exists()


def install(
    paths: Paths,
    *,
    executable: Path,
    load: bool = True,
    home: Path | None = None,
    launchctl: Callable[..., Any] | None = None,
    environ: Mapping[str, str] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> Path:
    """Write the plist (atomic, 0644) and, if `load`, bootstrap it. Idempotent: an existing
    agent is booted out first. Returns the plist path. `launchctl(*args)` (returns an object
    with returncode/stdout/stderr) and `environ` (GPU_ROUTER_HOME) are injectable for the
    setup wizard's tests (phase 8b); the defaults run /bin/launchctl and read os.environ."""
    run = launchctl or _launchctl
    env = os.environ if environ is None else environ
    target = plist_path(home)
    target.parent.mkdir(parents=True, exist_ok=True)
    paths.logs_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    data = render_plist(executable=executable.resolve(), paths=paths, env_home=env.get(ENV_HOME))
    tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    tmp.write_bytes(data)
    os.chmod(tmp, 0o644)
    if load and target.exists():
        run("bootout", f"{_domain()}/{LABEL}")  # ignore "not loaded"
    os.replace(tmp, target)
    if load:
        proc = run("bootstrap", _domain(), str(target))
        # right after a bootout, launchd answers 5 (Input/output error) until the old
        # daemon has exited, which can take a while on a busy Mac (2026-10-06)
        for pause in BOOTSTRAP_RETRY_S:
            if proc.returncode != BOOTSTRAP_BUSY:
                break
            sleep(pause)
            proc = run("bootstrap", _domain(), str(target))
        if proc.returncode != 0:
            raise RuntimeError(
                f"launchctl bootstrap failed ({proc.returncode}): "
                f"{(proc.stderr or proc.stdout).strip()}"
            )
    return target


def uninstall(*, unload: bool = True, home: Path | None = None) -> bool:
    """Boot out and delete the plist. Returns False if it was not installed."""
    target = plist_path(home)
    if not target.exists():
        return False
    if unload:
        _launchctl("bootout", f"{_domain()}/{LABEL}")
    target.unlink(missing_ok=True)
    return True


def kickstart() -> bool:
    """Ask launchd to start the (installed) agent now. True if launchctl accepted it."""
    return _launchctl("kickstart", f"{_domain()}/{LABEL}").returncode == 0
