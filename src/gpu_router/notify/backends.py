"""How a notification reaches macOS (phase 8a).

- `terminal-notifier` when installed (`-group` per job: a newer notification replaces the
  older one in Notification Center). Found on PATH, else in Homebrew's own bin dirs
  (KNOWN_NOTIFIER_PATHS): a launchd daemon's PATH is /usr/bin:/bin:/usr/sbin:/sbin, which
  never has Homebrew (review fix: `backend: terminal-notifier` silently became "off").
- `osascript` otherwise (always on macOS). The text travels as argv to an `on run argv`
  script, never spliced into AppleScript source, so a job name cannot inject code.
- `NullBackend` records what would have been sent (test mode, pytest, `backend: off`).

Backends are blocking and run on the notifier's worker thread only; each call is bounded
by `timeout_s` (the process group is killed after it).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Protocol

from gpu_router.notify.format import Notification
from gpu_router.notify.settings import NotifySettings

__all__ = [
    "ENV_REAL",
    "Backend",
    "NotifyError",
    "NullBackend",
    "OsaScriptBackend",
    "TerminalNotifierBackend",
    "choose_backend",
    "describe",
]

#: set to 1 to let a pytest run send real notifications (manual checks only)
ENV_REAL = "GPU_ROUTER_NOTIFY_REAL"
OSASCRIPT = "/usr/bin/osascript"
#: where Homebrew installs terminal-notifier (Apple Silicon, Intel)
KNOWN_NOTIFIER_PATHS = ("/opt/homebrew/bin/terminal-notifier", "/usr/local/bin/terminal-notifier")


def describe(backend: Backend) -> str:
    """One line for logs, /v1/health and doctor: the backend's name, or why it is off."""
    if isinstance(backend, NullBackend):
        return f"off: {backend.why}"
    return backend.name


# `display notification` with every part passed as argv: item 1 body, 2 title, 3 subtitle,
# 4 sound name ("" = silent). AppleScript never parses the values.
_APPLESCRIPT = (
    "on run argv",
    "set b to item 1 of argv",
    "set t to item 2 of argv",
    "set s to item 3 of argv",
    "set snd to item 4 of argv",
    'if snd is "" then',
    "display notification b with title t subtitle s",
    "else",
    "display notification b with title t subtitle s sound name snd",
    "end if",
    "end run",
)

Runner = Callable[[list[str], float], "subprocess.CompletedProcess[str]"]


class NotifyError(Exception):
    """A notification could not be shown (message says why; never contains job output)."""


class Backend(Protocol):
    name: str

    def send(self, n: Notification) -> None: ...


def _run(argv: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=timeout,
        stdin=subprocess.DEVNULL,
        check=False,
        start_new_session=True,
    )


def _check(proc: subprocess.CompletedProcess[str], what: str) -> None:
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip().splitlines()
        raise NotifyError(f"{what} exited {proc.returncode}: {err[-1][:200] if err else ''}")


@dataclass
class OsaScriptBackend:
    path: str = OSASCRIPT
    timeout_s: float = 10
    run: Runner = _run
    name: str = "osascript"

    def argv(self, n: Notification) -> list[str]:
        script: list[str] = []
        for line in _APPLESCRIPT:
            script += ["-e", line]
        return [self.path, *script, "--", n.body, n.title, n.subtitle, n.sound or ""]

    def send(self, n: Notification) -> None:
        try:
            proc = self.run(self.argv(n), self.timeout_s)
        except subprocess.TimeoutExpired:
            raise NotifyError(f"osascript did not finish in {self.timeout_s:g}s") from None
        except OSError as exc:
            raise NotifyError(f"cannot run osascript: {exc.strerror or exc}") from None
        _check(proc, "osascript")


@dataclass
class TerminalNotifierBackend:
    path: str
    timeout_s: float = 10
    run: Runner = _run
    name: str = "terminal-notifier"

    def argv(self, n: Notification) -> list[str]:
        argv = [self.path, "-title", n.title, "-subtitle", n.subtitle, "-message", n.body]
        if n.group:
            argv += ["-group", n.group]
        if n.sound:
            argv += ["-sound", n.sound]
        return argv

    def send(self, n: Notification) -> None:
        # terminal-notifier reads a message that starts with "-" or "[" as an option/stdin
        # marker: a leading space keeps the text literal
        if n.body[:1] in ("-", "["):
            n = Notification(
                kind=n.kind,
                title=n.title,
                subtitle=n.subtitle,
                body=" " + n.body,
                job_id=n.job_id,
                sound=n.sound,
                group=n.group,
            )
        try:
            proc = self.run(self.argv(n), self.timeout_s)
        except subprocess.TimeoutExpired:
            raise NotifyError(f"terminal-notifier did not finish in {self.timeout_s:g}s") from None
        except OSError as exc:
            raise NotifyError(f"cannot run terminal-notifier: {exc.strerror or exc}") from None
        _check(proc, "terminal-notifier")


@dataclass
class NullBackend:
    """Records instead of showing (`why` says why nothing appears)."""

    why: str = "off"
    sent: list[Notification] = field(default_factory=list)
    name: str = "none"

    def send(self, n: Notification) -> None:
        self.sent.append(n)


def _under_pytest(env: Mapping[str, str]) -> bool:
    return "PYTEST_CURRENT_TEST" in env and env.get(ENV_REAL, "").strip() not in ("1", "true")


def choose_backend(
    settings: NotifySettings,
    *,
    test_mode: bool,
    environ: Mapping[str, str] | None = None,
    which: Callable[[str], str | None] = shutil.which,
    known_paths: tuple[str, ...] = KNOWN_NOTIFIER_PATHS,
) -> Backend:
    """The backend `settings.backend` asks for on this Mac (see module docstring)."""
    env = os.environ if environ is None else environ
    if not settings.enabled or settings.backend == "off":
        return NullBackend(why="notifications are off in config.yaml")
    if _under_pytest(env):
        return NullBackend(why="running under pytest")
    if settings.backend == "auto" and test_mode:
        return NullBackend(why="test mode (set notifications.backend to force one)")
    if settings.backend in ("auto", "terminal-notifier"):
        # which() of an absolute path answers only for that file, if it is executable
        found = which("terminal-notifier") or next((p for p in known_paths if which(p)), None)
        if found:
            return TerminalNotifierBackend(path=found, timeout_s=settings.timeout_s)
        if settings.backend == "terminal-notifier":
            return NullBackend(
                why="terminal-notifier is not installed (brew install terminal-notifier)"
            )
    if os.path.exists(OSASCRIPT) or which("osascript"):
        path = OSASCRIPT if os.path.exists(OSASCRIPT) else (which("osascript") or OSASCRIPT)
        return OsaScriptBackend(path=path, timeout_s=settings.timeout_s)
    return NullBackend(why="neither terminal-notifier nor osascript is available")
