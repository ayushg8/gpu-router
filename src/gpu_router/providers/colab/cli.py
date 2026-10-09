"""Bounded calls to the `colab` CLI and the mapping of its failures (phase 3).

Every call is `colab --auth=adc --config <home>/providers/colab/colab-cli/sessions.json ...`:

- `--auth=adc` always (the CLI's own default is oauth2; NOTES.md "Login and auth").
- `--config` points at a session file only gpu-router uses, so `colab sessions` / `stop`
  never see another tool's or a human's sessions as ours. That file holds runtime proxy tokens:
  this module never reads it.
- `HOME` is a private dir under the 0700 data dir (`<home>/providers/colab/cli-home/`, D37).
  The CLI writes `~/.config/colab-cli/colab.log` (urllib3 DEBUG: every contents-API URL
  with its `colab-runtime-proxy-token`) and `history/<session>.jsonl` (every exec's code
  AND output, e.g. live log reads) with default permissions; under the user's real home
  those were 0644 files next to other tools' sessions. `CLOUDSDK_CONFIG` keeps pointing at the real
  gcloud dir so application-default credentials still resolve.

Output is redacted before it can reach an exception, a log or `RemoteRef.meta` (A8): the
CLI prints request URLs on some failures, and those carry `colab-runtime-proxy-token=...`.
"""

from __future__ import annotations

import contextlib
import os
import re
import shutil
import signal
import socket
import subprocess
import threading
import time
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gpu_router.errors import (
    AdapterError,
    AuthRequired,
    QuotaExhausted,
    RateLimited,
    Unavailable,
)

#: Lines the CLI prints about its own updates; never part of a result.
NOISE = ("new version of Colab", "colab update", "enable_update_check")

ADC_HINT = (
    "run: gcloud auth application-default login --scopes=openid,"
    "https://www.googleapis.com/auth/cloud-platform,"
    "https://www.googleapis.com/auth/userinfo.email,"
    "https://www.googleapis.com/auth/colaboratory"
)
INSTALL_HINT = "install it with `uv tool install google-colab-cli`"

_REDACTIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?i)((?:colab-runtime-proxy-)?token|access_token|key)=[^&\s'\"]+"), r"\1=***"),
    (re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+"), r"\1***"),
    (re.compile(r"ya29\.[A-Za-z0-9._-]+"), "***"),
    (re.compile(r"(?i)(\"(?:token|access_token|refresh_token)\"\s*:\s*\")[^\"]*"), r"\1***"),
)


def redact(text: str) -> str:
    """Strip tokens from CLI output, then apply gpu-router's global redaction."""
    for pattern, repl in _REDACTIONS:
        text = pattern.sub(repl, text)
    from gpu_router import secrets

    return secrets.redact(text)


def clean(text: str) -> str:
    """Drop update-nag lines and blank lines."""
    return "\n".join(
        line for line in text.splitlines() if line.strip() and not any(n in line for n in NOISE)
    )


@dataclass(frozen=True, slots=True)
class CliResult:
    argv: tuple[str, ...]  # without the global flags; safe to show
    returncode: int | None  # None = timed out (process group killed)
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    @property
    def timed_out(self) -> bool:
        return self.returncode is None

    @property
    def text(self) -> str:
        """stdout + stderr, update noise removed, redacted."""
        return redact(clean(f"{self.stdout}\n{self.stderr}"))

    def tail(self, n: int = 600) -> str:
        t = self.text.strip()
        return t if len(t) <= n else "..." + t[-n:]


class SessionGone(Exception):
    """The session no longer exists (VM reclaimed, pruned, or never created)."""


def resolve_cli(configured: object | None) -> list[str] | None:
    """argv prefix for the CLI: the `cli` provider setting (string or list), else `colab`
    on PATH, else ~/.local/bin/colab. None when nothing is installed."""
    if isinstance(configured, str) and configured.strip():
        return [configured]
    if isinstance(configured, list | tuple) and configured:
        return [str(p) for p in configured]
    found = shutil.which("colab")
    if found:
        return [found]
    fallback = Path.home() / ".local" / "bin" / "colab"
    if fallback.is_file() and os.access(fallback, os.X_OK):
        return [str(fallback)]
    return None


def real_gcloud_config_dir() -> Path:
    """Where gcloud / google-auth look for application-default credentials for THIS user
    (`$CLOUDSDK_CONFIG`, else `~/.config/gcloud` of the daemon's real home)."""
    explicit = os.environ.get("CLOUDSDK_CONFIG")
    if explicit:
        return Path(explicit)
    return Path(os.path.expanduser("~")) / ".config" / "gcloud"


class ColabCli:
    """Runs one CLI command at a time with a hard timeout (A2).

    `home`: the HOME the CLI runs with (its colab.log, history/ and settings live under
    `<home>/.config/colab-cli/`). None keeps the caller's HOME (tests of the runner only).
    """

    def __init__(
        self, prefix: Sequence[str], config_file: Path, *, home: Path | None = None
    ) -> None:
        self.prefix = list(prefix)
        self.config_file = config_file
        self.home = home

    def base_argv(self) -> list[str]:
        return [*self.prefix, "--auth=adc", "--config", str(self.config_file)]

    def env(self) -> dict[str, str]:
        env = dict(os.environ)
        env.setdefault("NO_COLOR", "1")
        env["PYTHONUNBUFFERED"] = "1"
        if self.home is not None:
            env.setdefault("CLOUDSDK_CONFIG", str(real_gcloud_config_dir()))
            env["HOME"] = str(self.home)
        return env

    def history_file(self, session: str) -> Path | None:
        """The CLI's per-session history file under the private HOME (None without one)."""
        if self.home is None:
            return None
        return self.home / ".config" / "colab-cli" / "history" / f"{session}.jsonl"

    def run(
        self,
        args: Sequence[str],
        *,
        timeout: float,
        on_spawn: Callable[[int], None] | None = None,
    ) -> CliResult:
        """`on_spawn(pid)` runs right after the child starts (before waiting): the CLI runs
        in its own session, so it outlives a daemon crash and callers may record its pid."""
        self.config_file.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if self.home is not None:
            d = self.home
            for part in ("", ".config", "colab-cli"):
                d = d / part if part else d
                d.mkdir(mode=0o700, parents=True, exist_ok=True)
            with contextlib.suppress(OSError):
                os.chmod(self.home, 0o700)
        argv = [*self.base_argv(), *args]
        env = self.env()
        try:
            proc = subprocess.Popen(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                errors="replace",
                env=env,
                start_new_session=True,  # a timeout kills the CLI and its children only
            )
        except FileNotFoundError:
            return CliResult(tuple(args), 127, "", f"colab CLI not found: {self.prefix[0]}")
        except OSError as exc:
            return CliResult(tuple(args), 126, "", f"cannot run colab CLI: {exc}")
        if on_spawn is not None:
            try:
                on_spawn(proc.pid)
            except Exception:  # bookkeeping must never orphan the child we just started
                _kill_group(proc)
                proc.communicate()
                raise
        try:
            out, err = proc.communicate(timeout=max(1.0, timeout))
            return CliResult(tuple(args), proc.returncode, out or "", err or "")
        except subprocess.TimeoutExpired:
            _kill_group(proc)
            try:
                out, err = proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                out, err = "", ""
            return CliResult(tuple(args), None, out or "", err or "")


def _kill_group(proc: subprocess.Popen[str]) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    except OSError:
        proc.kill()


# --------------------------------------------------------------------------- classification

_LOST = (
    "appears to be lost",
    "session_not_found",
)
_NOT_FOUND = re.compile(r"Session '[^']*' not found")
_AUTH = (
    "No valid default credentials",
    "DefaultCredentialsError",
    "RefreshError",
    "invalid_grant",
    "Reauthentication is needed",
    "reauth",
    "invalid_rapt",
    "UNAUTHENTICATED",
)
_SCOPE = ("missing an OAuth scope", "SCOPE_NOT_PERMITTED", "insufficient authentication scopes")
_NETWORK = (
    "ConnectionError",
    "Max retries exceeded",
    "NameResolutionError",
    "Temporary failure in name resolution",
    "nodename nor servname",
    "Connection reset",
    "Connection aborted",
    "Network is unreachable",
    "ReadTimeout",
    "ConnectTimeout",
    "TimeoutError",
    "timed out",
    "502 Server Error",
    "503 Server Error",
    "504 Server Error",
    "Bad Gateway",
    "Service Unavailable",
)
#: Throttling signatures. Never a bare "429": `colab new` echoes the session name
#: (`gr-<job>-<n>`, about 1 hex job id in 400 contains "429") and exec output is arbitrary.
_RATE = (
    "429 Client Error",
    "Too Many Requests",
    "RESOURCE_EXHAUSTED",
    "rateLimitExceeded",
    "HTTP 429",
    "HTTP Error 429",
    "status 429",
    "status: 429",
    "status code 429",
    "Error 429",
    "code 429",
    "(429)",
)
#: Session names we chose (`gr-...`): blanked before matching failure signatures.
_OUR_NAMES = re.compile(r"\bgr-[A-Za-z0-9_-]{1,62}")
#: Google's token endpoint, where the CLI refreshes application-default credentials.
GOOGLE_TOKEN_HOST = ("oauth2.googleapis.com", 443)
REACH_TIMEOUT_S = 4.0
#: Addresses tried per probe (IPv4 first: a dead IPv6 route must not hide a working IPv4).
REACH_MAX_ADDRS = 4
_AddrInfo = tuple[socket.AddressFamily, socket.SocketKind, int, str, Any]


def _behind_proxy(host: str) -> bool:
    """Whether `requests` (the CLI's HTTP stack) would go through a proxy for `host`: env
    vars, and on macOS the System Settings proxy (`urllib.request.getproxies`), minus
    NO_PROXY / the bypass list."""
    proxies = urllib.request.getproxies()
    if not (proxies.get("https") or proxies.get("all")):
        return False
    return not urllib.request.proxy_bypass(host)


def connect_order(infos: Sequence[_AddrInfo]) -> list[_AddrInfo]:
    """getaddrinfo results to try: IPv4 first (stable), at most REACH_MAX_ADDRS."""
    return sorted(infos, key=lambda info: info[0] != socket.AF_INET)[:REACH_MAX_ADDRS]


def google_reachable(timeout: float = REACH_TIMEOUT_S) -> bool:
    """Whether this Mac can open a connection to Google's token endpoint. The colab CLI
    reports a failed credential refresh as "No valid default credentials found" whether the
    sign-in expired or the network is down (it even exits 0 for `sessions`; reproduced with
    an unreachable proxy, 2026-10-08, after the daemon marked colab "login needed" on wakes
    with no network), so an auth-looking failure is a login problem only while Google
    answers. Runs on a thread because getaddrinfo ignores socket timeouts; a hang counts as
    unreachable. Behind a proxy a direct connection proves nothing: True."""
    host, port = GOOGLE_TOKEN_HOST
    if _behind_proxy(host):
        return True
    answer: list[bool] = []

    def probe() -> None:
        deadline = time.monotonic() + timeout
        try:
            infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        except OSError:
            answer.append(False)
            return
        infos = connect_order(infos)
        for n, (family, kind, proto, _name, addr) in enumerate(infos):
            left = deadline - time.monotonic()
            if left <= 0:
                break
            try:
                with socket.socket(family, kind, proto) as sock:
                    sock.settimeout(max(0.5, left / (len(infos) - n)))
                    sock.connect(addr)
            except OSError:
                continue
            answer.append(True)
            return
        answer.append(False)

    worker = threading.Thread(target=probe, name="colab-reach", daemon=True)
    worker.start()
    worker.join(timeout + 1.0)
    return bool(answer) and answer[0]


def session_gone(res: CliResult) -> bool:
    text = res.text
    return any(s in text for s in _LOST) or bool(_NOT_FOUND.search(text))


def classify(
    res: CliResult,
    *,
    provider: str,
    op: str,
    gpu: str | None = None,
    now: float | None = None,
    quota_reset_s: float = 24 * 3600,
    reachable: Callable[[], bool] | None = None,
) -> AdapterError:
    """Map a failed CLI call to the adapter taxonomy (A3). Callers check `session_gone`
    first where a vanished session has its own meaning (status -> lost, cancel -> done).

    - GPU refused with 400 ("Backend rejected accelerator"): QuotaExhausted for the free T4
      (Colab's dynamic usage limit; resets in roughly a day), with `resets_at` when `now`
      is known.
    - 412 ("Allocation refused (precondition failed)"): Unavailable. It means too many
      active sessions (another tool may hold the account's one free GPU) or a temporary
      capacity limit.
    - missing scope / no or expired ADC / missing CLI: AuthRequired with the fix as hint;
      ADC trouble while `reachable()` says Google cannot be reached is Unavailable.
    - 429-style throttling: RateLimited. Network trouble, timeouts, anything else:
      Unavailable (transient; the engine backs off).
    """
    tail = res.tail()
    what = f"colab {op}"
    if res.returncode == 127:
        return AuthRequired(
            f"{what}: the colab CLI is not installed", provider=provider, hint=INSTALL_HINT
        )
    if res.timed_out:
        return Unavailable(f"{what} timed out", provider=provider)
    text = _OUR_NAMES.sub("gr-<session>", res.text)
    if "Backend rejected accelerator" in text:
        resets_at = None if now is None else now + quota_reset_s
        return QuotaExhausted(
            f"colab refused a {gpu or 'GPU'} runtime: the free GPU quota is used up for now",
            resets_at=resets_at,
            provider=provider,
            hint="colab's free limit is dynamic and usually resets within a day",
            detail={"op": op, "gpu": gpu},
        )
    if "Allocation refused" in text or "precondition failed" in text:
        return Unavailable(
            f"colab refused the session (too many active sessions or a temporary limit): {tail}",
            provider=provider,
            hint="another tool may be using the account's free GPU; `colab --auth=adc "
            "sessions` lists what is running",
        )
    if any(s in text for s in _SCOPE):
        return AuthRequired(
            f"{what}: the Google credentials lack the colaboratory scope",
            provider=provider,
            hint=ADC_HINT,
        )
    if any(s in text for s in _AUTH):
        if any(s in text for s in _NETWORK) or (reachable is not None and not reachable()):
            return Unavailable(
                f"{what} could not refresh the Google sign-in: oauth2.googleapis.com is "
                "unreachable, so the network looks down (the login itself may be fine)",
                provider=provider,
                hint=f"if the network is fine, the sign-in expired; {ADC_HINT}",
            )
        return AuthRequired(
            f"{what}: Google application-default credentials are missing or expired",
            provider=provider,
            hint=ADC_HINT,
        )
    if any(s in text for s in _RATE):
        return RateLimited(f"{what} was throttled: {tail}", retry_after=300, provider=provider)
    if any(s in text for s in _NETWORK):
        return Unavailable(f"{what} could not reach colab: {tail}", provider=provider)
    return Unavailable(f"{what} failed (exit {res.returncode}): {tail}", provider=provider)
