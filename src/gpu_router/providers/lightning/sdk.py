"""Bounded calls into the lightning-sdk, which lives in its own environment (D3).

The SDK is never imported by gpu-router. `SdkBridge.call(op, params)` runs driver.py with
an interpreter that has `lightning-sdk` installed and reads the one `@@GRL:` result line
(protocol in driver.py). Interpreter, in order: settings `python`, the `uv tool install
lightning-sdk` env (`<uv tool dir>/lightning-sdk/bin/python`), else `uv run --no-project
--python 3.12 --with lightning-sdk==<pin> python` (uv caches the env; ~0.3 s once cached).

Every call (rule A2):
- runs in a new session with a timeout below the engine's budget; on timeout the whole
  process group is killed (uv's child interpreter included), so no SDK call outlives it;
- gets a scrubbed environment: an allowlist of the daemon's variables (PATH, locale, CA
  bundles, proxies, uv's own), no inherited LIGHTNING_* identity, and our credentials
  (invariant 12: environment only, never argv or files);
- points the SDK's HOME / credential / settings paths at `<home>/providers/<name>/sdk-home`
  (0700), so ~/.lightning is never read or written, and turns its version check off.

`to_error()` maps a driver failure kind to the adapter taxonomy (rule A3); messages are
redacted and truncated (A8).
"""

from __future__ import annotations

import base64
import binascii
import json
import os
import shutil
import signal
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from gpu_router.errors import (
    AdapterError,
    AuthRequired,
    InvalidJob,
    NotFound,
    Permanent,
    QuotaExhausted,
    RateLimited,
    Unavailable,
)
from gpu_router.providers.lightning.credentials import LOGIN_HINT, SDK_ENV_VARS
from gpu_router.secrets import redact

__all__ = [
    "DEFAULT_SDK_VERSION",
    "DRIVER_PATH",
    "MARKER",
    "CallResult",
    "ProcResult",
    "Runner",
    "SdkBridge",
    "SubprocessRunner",
    "find_uv",
    "resolve_interpreter",
    "snippet",
    "to_error",
]

MARKER = "@@GRL:"
DRIVER_PATH = Path(__file__).with_name("driver.py")
#: verified in NOTES.md (2026-09-23); settings `sdk_version` overrides ("latest" = unpinned)
DEFAULT_SDK_VERSION = "2026.9.18.post1"
SDK_PYTHON = "3.12"
SNIPPET_CHARS = 300
INSTALL_HINT = (
    "install the lightning SDK with `uv tool install lightning-sdk` (or install uv: "
    "gpu-router then runs it through `uv run --with lightning-sdk`)"
)
PHONE_HINT = "verify your phone number on lightning.ai to unlock the free credits"

#: daemon environment variables the SDK child may inherit (everything else is dropped).
_ENV_ALLOW = {
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "SHELL",
    "TMPDIR",
    "TZ",
    "LANG",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "REQUESTS_CA_BUNDLE",
    "CURL_CA_BUNDLE",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "no_proxy",
    "all_proxy",
    "XDG_CACHE_HOME",
    "XDG_DATA_HOME",
    "XDG_BIN_HOME",
}
_ENV_ALLOW_PREFIX = ("LC_", "UV_")


class SdkMissing(Exception):
    """The interpreter (or uv) could not be executed."""


class CallTimeout(Exception):
    """The call ran past its timeout; its process group was killed."""


@dataclass(frozen=True, slots=True)
class ProcResult:
    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


class Runner(Protocol):
    """Runs one command with `stdin` text. Raises SdkMissing / CallTimeout; never raises
    for exit codes."""

    def __call__(
        self, argv: Sequence[str], *, stdin: str, timeout: float, env: Mapping[str, str]
    ) -> ProcResult: ...


class SubprocessRunner:
    """The real runner: its own session, so a timeout kills uv AND the interpreter."""

    def __call__(
        self, argv: Sequence[str], *, stdin: str, timeout: float, env: Mapping[str, str]
    ) -> ProcResult:
        try:
            proc = subprocess.Popen(  # argv built by us, never a shell string
                list(argv),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=dict(env),
                text=True,
                encoding="utf-8",
                errors="replace",
                start_new_session=True,
            )
        except (FileNotFoundError, PermissionError) as exc:
            raise SdkMissing(str(argv[0])) from exc
        try:
            out, err = proc.communicate(stdin, timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_group(proc)
            proc.communicate()
            raise CallTimeout(f"timed out after {timeout:g}s") from None
        except BaseException:
            _kill_group(proc)
            proc.communicate()
            raise
        return ProcResult(tuple(argv), proc.returncode, out or "", err or "")


def _kill_group(proc: subprocess.Popen[str]) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except OSError:
        proc.kill()


def find_uv(configured: str | None = None) -> str | None:
    if configured:
        return configured
    found = shutil.which("uv")
    if found:
        return found
    for candidate in (
        Path.home() / ".local" / "bin" / "uv",
        Path("/opt/homebrew/bin/uv"),
        Path("/usr/local/bin/uv"),
    ):
        if candidate.is_file():
            return str(candidate)
    return None


def _uv_tool_python() -> Path:
    base = os.environ.get("UV_TOOL_DIR")
    root = Path(base) if base else Path.home() / ".local" / "share" / "uv" / "tools"
    return root / "lightning-sdk" / "bin" / "python"


def resolve_interpreter(
    *,
    python: str | None = None,
    sdk_version: str | None = DEFAULT_SDK_VERSION,
    uv: str | None = None,
    tool_python: Path | None = None,
) -> list[str] | None:
    """The command prefix that runs a Python with lightning-sdk importable, or None."""
    if python:
        return [python]
    tool = tool_python or _uv_tool_python()
    if tool.is_file():
        return [str(tool)]
    uv_path = find_uv(uv)
    if uv_path is None:
        return None
    pinned = sdk_version not in (None, "", "latest")
    req = f"lightning-sdk=={sdk_version}" if pinned else "lightning-sdk"
    uv_run = [uv_path, "run", "--no-project", "--quiet", "--python", SDK_PYTHON]
    return [*uv_run, "--with", req, "python"]


def snippet(text: str, limit: int = SNIPPET_CHARS) -> str:
    """One redacted, whitespace-collapsed, truncated line for messages (A8)."""
    flat = " ".join(redact(text).split())
    return flat if len(flat) <= limit else flat[: limit - 3] + "..."


@dataclass(frozen=True, slots=True)
class CallResult:
    ok: bool
    result: dict[str, Any]
    kind: str | None = None
    error: str | None = None
    stage: str | None = None
    status: int | None = None


def parse_result(stdout: str) -> dict[str, Any] | None:
    """The last `@@GRL:` line's JSON, or None."""
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line.startswith(MARKER):
            continue
        try:
            doc = json.loads(base64.b64decode(line[len(MARKER) :]).decode("utf-8"))
        except (ValueError, binascii.Error, UnicodeDecodeError):
            return None
        return doc if isinstance(doc, dict) else None
    return None


def to_error(
    provider: str, op: str, kind: str | None, message: str, *, resets_at: float | None = None
) -> AdapterError:
    """Driver failure kind -> taxonomy class (rule A3)."""
    text = snippet(message) or f"lightning {op} failed"
    if kind == "auth":
        return AuthRequired(text, provider=provider, hint=LOGIN_HINT)
    if kind == "verify":
        return AuthRequired(text, provider=provider, hint=PHONE_HINT)
    if kind == "config":
        return AuthRequired(text, provider=provider, hint="fix providers.lightning in config.yaml")
    if kind == "not_found":
        return NotFound(text, provider=provider)
    if kind == "rate":
        return RateLimited(text, provider=provider, retry_after=60)
    if kind == "quota":
        return QuotaExhausted(
            text,
            provider=provider,
            resets_at=resets_at,
            hint="gpu-router uses other providers until the monthly credits come back",
        )
    if kind == "invalid":
        return InvalidJob(text, provider=provider)
    if kind == "permanent":
        return Permanent(text, provider=provider)
    return Unavailable(text, provider=provider)


class SdkBridge:
    """Thread-safe: every call is its own process. `env_factory` returns the credential
    environment for each call (so a Keychain change needs no daemon restart)."""

    def __init__(
        self,
        provider: str,
        *,
        runner: Runner,
        interpreter: Callable[[], list[str] | None],
        env_factory: Callable[[], Mapping[str, str]],
        home: Callable[[], Path],
        driver: Path = DRIVER_PATH,
        extra_env: Mapping[str, str] | None = None,
    ) -> None:
        self.provider = provider
        self._runner = runner
        self._interpreter = interpreter
        self._env_factory = env_factory
        self._home = home
        self._driver = driver
        self._extra_env = dict(extra_env or {})

    def _env(self) -> dict[str, str]:
        env = {
            k: v
            for k, v in os.environ.items()
            if (k in _ENV_ALLOW or k.startswith(_ENV_ALLOW_PREFIX)) and k not in SDK_ENV_VARS
        }
        home = self._home()
        home.mkdir(mode=0o700, parents=True, exist_ok=True)
        dot = home / ".lightning"
        env.update(
            {
                "GR_LIGHTNING_HOME": str(home),
                "LIGHTNING_CREDENTIAL_PATH": str(dot / "credentials.json"),
                "LIGHTNING_SETTINGS_PATH": str(dot / "settings.json"),
                "LIGHTNING_DISABLE_VERSION_CHECK": "1",
                "BROWSER": "true",
                "TQDM_DISABLE": "1",
                "PYTHONIOENCODING": "utf-8",
                "PYTHONUNBUFFERED": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
            }
        )
        env.update(self._extra_env)
        env.update(self._env_factory())
        return env

    def call(self, op: str, params: Mapping[str, Any], *, timeout: float) -> CallResult:
        """Run one driver op. Raises AuthRequired (no SDK interpreter) or Unavailable
        (timeout, crash without a result line); a classified failure is returned."""
        prefix = self._interpreter()
        if prefix is None:
            raise AuthRequired(
                "the lightning SDK is not installed and uv was not found",
                provider=self.provider,
                hint=INSTALL_HINT,
            )
        argv = [*prefix, "-s", "-u", str(self._driver)]
        request = json.dumps({"op": op, "params": dict(params)}, default=str)
        try:
            res = self._runner(argv, stdin=request, timeout=timeout, env=self._env())
        except SdkMissing:
            raise AuthRequired(
                f"could not run the lightning SDK interpreter {prefix[0]}",
                provider=self.provider,
                hint=INSTALL_HINT,
            ) from None
        except CallTimeout:
            raise Unavailable(
                f"lightning {op} did not answer within {timeout:g}s",
                provider=self.provider,
                hint="lightning.ai may be slow or unreachable; gpu-router retries",
            ) from None
        doc = parse_result(res.stdout)
        if doc is None:
            text = res.stderr or res.stdout
            low = text.lower()
            if "no module named 'lightning_sdk'" in low or "no module named lightning_sdk" in low:
                raise AuthRequired(
                    "the lightning SDK is not installed in its interpreter",
                    provider=self.provider,
                    hint=INSTALL_HINT,
                )
            if res.returncode != 0 and prefix[0].endswith("uv") and "error:" in low:
                raise Unavailable(
                    f"uv could not prepare the lightning SDK: {snippet(text)}",
                    provider=self.provider,
                    hint=INSTALL_HINT,
                )
            raise Unavailable(
                f"lightning {op} ended without a result (exit {res.returncode}): "
                f"{snippet(text) or 'no output'}",
                provider=self.provider,
            )
        if doc.get("ok"):
            result = doc.get("result")
            return CallResult(ok=True, result=result if isinstance(result, dict) else {})
        status = doc.get("status")
        return CallResult(
            ok=False,
            result={},
            kind=str(doc.get("kind") or "sdk"),
            error=snippet(str(doc.get("error") or "")),
            stage=str(doc["stage"]) if doc.get("stage") else None,
            status=status if isinstance(status, int) else None,
        )
