"""Bounded calls to the kaggle CLI (D3: provider SDKs are not project dependencies).

`KaggleCli.run()` executes `kaggle -W <args>` through an injectable `Runner` (tests pass a
simulated Kaggle), always with a timeout below the engine's per-call budget (rule A2), and
`classify()` turns a failed call into the adapter error taxonomy (rule A3). Output snippets
that end up in exception messages are redacted and truncated (A8).

Credentials (invariant 12) come from `credentials.resolve()`: Keychain values reach the
subprocess only through its environment, never argv, logs or files.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from gpu_router.errors import (
    AdapterError,
    AuthRequired,
    NotFound,
    RateLimited,
    Unavailable,
)
from gpu_router.secrets import redact

__all__ = [
    "CliMissing",
    "CliResult",
    "CliTimeout",
    "KaggleCli",
    "Runner",
    "SubprocessRunner",
    "classify",
    "find_executable",
    "snippet",
]

INSTALL_HINT = "install the kaggle CLI with `uv tool install kaggle`"
LOGIN_HINT = (
    "check ~/.kaggle/kaggle.json, or run `gpu login kaggle` (checks the token, then keeps it "
    "in the Keychain)"
)
PHONE_HINT = "verify your phone number at https://www.kaggle.com/settings to use GPUs"
SNIPPET_CHARS = 300


class CliMissing(Exception):
    """The kaggle executable is not installed / not on PATH."""


class CliTimeout(Exception):
    """The call ran past its timeout and was killed."""


@dataclass(frozen=True, slots=True)
class CliResult:
    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str

    @property
    def output(self) -> str:
        return f"{self.stdout}\n{self.stderr}" if self.stderr else self.stdout

    @property
    def ok(self) -> bool:
        return self.returncode == 0


class Runner(Protocol):
    """Runs one command. Raises CliMissing / CliTimeout; never raises for exit codes."""

    def __call__(
        self,
        argv: Sequence[str],
        *,
        timeout: float,
        env: Mapping[str, str],
        cwd: Path | None = None,
    ) -> CliResult: ...


class SubprocessRunner:
    """The real runner: `subprocess.run` with capture, timeout and no stdin."""

    def __call__(
        self,
        argv: Sequence[str],
        *,
        timeout: float,
        env: Mapping[str, str],
        cwd: Path | None = None,
    ) -> CliResult:
        try:
            proc = subprocess.run(  # argv is built by us, never a shell string
                list(argv),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                env=dict(env),
                cwd=str(cwd) if cwd is not None else None,
                stdin=subprocess.DEVNULL,
                check=False,
            )
        except FileNotFoundError as exc:
            raise CliMissing(str(argv[0])) from exc
        except subprocess.TimeoutExpired as exc:
            raise CliTimeout(f"{' '.join(argv[:3])} timed out after {timeout:g}s") from exc
        return CliResult(tuple(argv), proc.returncode, proc.stdout or "", proc.stderr or "")


def find_executable(configured: str | None = None) -> str | None:
    """settings `cli_path`, else PATH, else ~/.local/bin/kaggle (where `uv tool install`
    puts it; a launchd daemon's PATH often lacks it)."""
    if configured:
        return configured
    found = shutil.which("kaggle")
    if found:
        return found
    fallback = Path.home() / ".local" / "bin" / "kaggle"
    return str(fallback) if fallback.is_file() else None


def snippet(text: str, limit: int = SNIPPET_CHARS) -> str:
    """One redacted, whitespace-collapsed, truncated line for messages (A8)."""
    flat = " ".join(redact(text).split())
    return flat if len(flat) <= limit else flat[: limit - 3] + "..."


class KaggleCli:
    """Thin, thread-safe wrapper: builds argv + env, runs, returns CliResult.

    `env_factory` returns (inherited env var names to drop, extra environment) for each
    call, so a Keychain change is picked up without restarting the daemon.
    """

    def __init__(
        self,
        provider: str,
        *,
        runner: Runner,
        executable: Callable[[], str | None],
        env_factory: Callable[[], tuple[Sequence[str], Mapping[str, str]]] = lambda: ((), {}),
    ) -> None:
        self.provider = provider
        self._runner = runner
        self._executable = executable
        self._env_factory = env_factory

    def _env(self) -> dict[str, str]:
        env = dict(os.environ)
        drop, extra = self._env_factory()
        for key in drop:
            env.pop(key, None)
        env.update(extra)
        env["PYTHONIOENCODING"] = "utf-8"
        env.setdefault("PYTHONUNBUFFERED", "1")
        return env

    def run(self, args: Sequence[str], *, timeout: float, cwd: Path | None = None) -> CliResult:
        """Run `kaggle -W <args>`. Raises AuthRequired (CLI missing) or Unavailable
        (timeout); a non-zero exit is returned, not raised: callers classify."""
        exe = self._executable()
        if exe is None:
            raise AuthRequired(
                "the kaggle CLI is not installed", provider=self.provider, hint=INSTALL_HINT
            )
        argv = [exe, "-W", *args]
        try:
            return self._runner(argv, timeout=timeout, env=self._env(), cwd=cwd)
        except CliMissing:
            raise AuthRequired(
                f"the kaggle CLI was not found at {exe}", provider=self.provider, hint=INSTALL_HINT
            ) from None
        except CliTimeout:
            raise Unavailable(
                f"kaggle {' '.join(args[:2])} did not answer within {timeout:g}s",
                provider=self.provider,
                hint="kaggle may be slow or unreachable; gpu-router retries",
            ) from None


# --------------------------------------------------------------------------- classification

_AUTH_MARKERS = (
    "authentication required to call the kaggle api",
    "401 client error",
    "unauthorized",
    "invalid credentials",
    "could not find kaggle.json",
)
#: Throttling signatures. Never a bare "429": the searched text carries the kernel ref
#: (`<user>/gpu-router-<job_id>-<n>`), and about 1 hex job id in 400 contains "429".
_RATE_MARKERS = (
    "429 client error",
    "too many requests",
    "rate limit",
    "ratelimit",
    "rate-limit",
    "http 429",
    "http error 429",
    "status 429",
    "status: 429",
    "status code 429",
    "status_code=429",
    "error 429",
    "code 429",
    "(429)",
)
_MISSING_MARKERS = ("cannot access kernel", "404 client error", "not found for url")
#: Kernel refs / slugs and `userName=` query values: names we chose or the account's, never
#: part of an error signature, so they are blanked before matching.
_NAME_NOISE = re.compile(
    r"(?:[A-Za-z0-9][A-Za-z0-9_.-]{0,63}/)?gpu-router-[0-9a-f]{6,32}-[0-9]{1,4}"
    r"|(?i:(?:user_?name|owner_?slug|kernel_?slug)=)[^&\s'\"]*"
)
_TRANSIENT_MARKERS = (
    "server error",
    "connectionerror",
    "connection aborted",
    "connection reset",
    "max retries exceeded",
    "timed out",
    "timeout",
    "temporarily unavailable",
    "service unavailable",
    "bad gateway",
    "name or service not known",
    "nodename nor servname",
    "ssl",
)


def classify(provider: str, op: str, result: CliResult) -> AdapterError:
    """Map a failed kaggle call to the taxonomy. `op` names the call for the message.

    Kernel-scoped 401/403 become "Cannot access kernel ..." (a ValueError inside the CLI),
    which is how a missing kernel looks too: callers that must tell the two apart confirm
    auth separately (KaggleAdapter._confirm_missing)."""
    text = result.output
    low = _NAME_NOISE.sub("<name>", text).lower()
    short = snippet(text) or f"exit code {result.returncode}"
    if any(m in low for m in _AUTH_MARKERS):
        return AuthRequired(
            f"kaggle is not logged in ({op}): {short}", provider=provider, hint=LOGIN_HINT
        )
    if any(m in low for m in _MISSING_MARKERS):
        return NotFound(f"kaggle has no such kernel ({op})", provider=provider)
    if any(m in low for m in _RATE_MARKERS):
        return RateLimited(f"kaggle is rate limiting ({op})", provider=provider, retry_after=60)
    if any(m in low for m in _TRANSIENT_MARKERS):
        return Unavailable(f"kaggle is unreachable ({op}): {short}", provider=provider)
    return Unavailable(f"kaggle {op} failed: {short}", provider=provider)
