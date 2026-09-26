"""Which Claude Code session a job came from (D56).

`gpu run` and the MCP server tag every job they submit with two labels taken from their own
environment: `claude_session` (CLAUDE_CODE_SESSION_ID) and `claude_pid` (CLAUDE_PID, the
claude process; the MCP server, a direct child of claude, falls back to its parent pid).
The status line shows a job only in the session it came from (fast.py). Jobs from a plain
terminal or the `gpu` shell carry no origin. The origin is display routing only: it is not
secret and never used for auth.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

    from gpu_router.models import JobSpec

LABEL_SESSION = "claude_session"
LABEL_PID = "claude_pid"
_SESSION = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


def claude_origin(environ: Mapping[str, str], *, fallback_pid: int | None = None) -> dict[str, str]:
    """The origin labels for a job submitted from this environment ({} outside Claude Code).
    `fallback_pid` is used when CLAUDE_PID is missing but the env is a Claude Code child."""
    out: dict[str, str] = {}
    session = environ.get("CLAUDE_CODE_SESSION_ID", "")
    if _SESSION.match(session):
        out[LABEL_SESSION] = session
    pid = environ.get("CLAUDE_PID", "")
    if pid.isdigit() and int(pid) > 1:
        out[LABEL_PID] = str(int(pid))
    elif fallback_pid and fallback_pid > 1 and (out or environ.get("CLAUDECODE") == "1"):
        out[LABEL_PID] = str(fallback_pid)
    return out


def with_origin(
    spec: JobSpec, environ: Mapping[str, str], *, fallback_pid: int | None = None
) -> JobSpec:
    """`spec` with the origin labels of `environ` (unchanged outside Claude Code)."""
    origin = claude_origin(environ, fallback_pid=fallback_pid)
    if not origin:
        return spec
    from gpu_router.models import JobSpec

    return JobSpec.model_validate({**spec.model_dump(), "labels": {**spec.labels, **origin}})
