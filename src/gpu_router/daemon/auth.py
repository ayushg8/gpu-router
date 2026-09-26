"""Daemon auth and request guards (phase 1; owner: group C). Invariant 18.

Token: 32 random bytes, urlsafe base64, stored in paths.token with mode 0600, created by the
daemon at start if missing (kept across restarts so running clients keep working).
Comparison with hmac.compare_digest.

Guard, for every request:
- `Host` must be 127.0.0.1:<port> or localhost:<port>, else 403 forbidden. While the bound
  port is not known yet (port 0: in-process tests), any port on those two hosts is accepted.
- Any `Origin` header -> 403 forbidden (browsers send it; our clients never do).
- Missing/invalid `Authorization: Bearer <token>` -> 401 unauthorized, except
  GET /v1/health.
"""

from __future__ import annotations

import base64
import contextlib
import hmac
import os
import secrets

from gpu_router.errors import Forbidden, Unauthorized
from gpu_router.paths import Paths

TOKEN_BYTES = 32
PUBLIC_PATHS: frozenset[tuple[str, str]] = frozenset({("GET", "/v1/health")})
LOOPBACK_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "localhost"})


def _new_token() -> str:
    return base64.urlsafe_b64encode(secrets.token_bytes(TOKEN_BYTES)).decode("ascii").rstrip("=")


def ensure_token(paths: Paths) -> str:
    """Return the existing token, or create one (0600, atomic) and return it."""
    existing = read_token(paths)
    if existing:
        with contextlib.suppress(OSError):
            os.chmod(paths.token, 0o600)
        return existing
    paths.home.mkdir(mode=0o700, parents=True, exist_ok=True)
    token = _new_token()
    tmp = paths.token.with_name(f".{paths.token.name}.{os.getpid()}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, token.encode("ascii") + b"\n")
        os.fsync(fd)
    finally:
        os.close(fd)
    os.chmod(tmp, 0o600)
    os.replace(tmp, paths.token)
    return token


def read_token(paths: Paths) -> str | None:
    """Client side: the token, or None if the file is missing/unreadable."""
    try:
        value = paths.token.read_text(encoding="ascii").strip()
    except (OSError, UnicodeDecodeError):
        return None
    return value or None


def _host_ok(host: str, port: int) -> bool:
    host = host.strip().lower()
    if not host:
        return False
    name, sep, port_s = host.rpartition(":")
    if not sep:
        name, port_s = host, ""
    if name not in LOOPBACK_HOSTS:
        return False
    if port == 0:
        return port_s == "" or port_s.isdigit()
    if port_s == "":
        return port == 80
    return port_s == str(port)


def check_request(
    *, method: str, path: str, headers: dict[str, str], token: str, port: int
) -> None:
    """Raise errors.Forbidden / errors.Unauthorized per the module docstring.
    `headers` keys are lowercase."""
    if not _host_ok(headers.get("host", ""), port):
        raise Forbidden(
            "requests must be addressed to 127.0.0.1 or localhost",
            hint="the daemon only serves local clients",
        )
    if "origin" in headers:
        raise Forbidden(
            "browser requests are not allowed",
            hint="use the gpu CLI, shell or MCP server",
        )
    if (method.upper(), path) in PUBLIC_PATHS:
        return
    auth = headers.get("authorization", "")
    scheme, _, supplied = auth.partition(" ")
    supplied = supplied.strip()
    if (
        scheme.lower() != "bearer"
        or not supplied
        or not hmac.compare_digest(supplied.encode("utf-8"), token.encode("utf-8"))
    ):
        raise Unauthorized(
            "missing or invalid bearer token",
            hint="clients read the token from daemon.token in the gpu-router data dir",
        )
