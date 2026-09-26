"""Pure parsers for kaggle CLI 2.2.x output. No I/O, no clock; every shape is in NOTES.md.

The CLI prints human text (and exits 0 on some server-side errors), so the adapter never
trusts the exit code alone: it parses stdout/stderr with the functions here.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum

from gpu_router import protocol

__all__ = [
    "KernelStatus",
    "PushOutcome",
    "QuotaRow",
    "StatusOutcome",
    "last_exit_code",
    "parse_config_username",
    "parse_log",
    "parse_push",
    "parse_quota",
    "parse_status",
    "parse_version",
]


class KernelStatus(StrEnum):
    """kagglesdk KernelWorkerStatus names (verified in kagglesdk 0.1.37)."""

    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"
    ERROR = "ERROR"
    CANCEL_REQUESTED = "CANCEL_REQUESTED"
    CANCEL_ACKNOWLEDGED = "CANCEL_ACKNOWLEDGED"
    NEW_SCRIPT = "NEW_SCRIPT"

    @property
    def finished(self) -> bool:
        return self in (KernelStatus.COMPLETE, KernelStatus.ERROR)

    @property
    def cancelled(self) -> bool:
        return self in (KernelStatus.CANCEL_REQUESTED, KernelStatus.CANCEL_ACKNOWLEDGED)


# `<ref> has status "KernelWorkerStatus.RUNNING"` (+ optional `Failure message: "..."`).
_STATUS_RE = re.compile(r'has status "(?:KernelWorkerStatus\.)?([A-Za-z_]+)"')
_FAILURE_RE = re.compile(r'Failure message: "(.*)"\s*\Z', re.S)


@dataclass(frozen=True, slots=True)
class StatusOutcome:
    status: KernelStatus
    failure_message: str | None = None


def parse_status(text: str) -> StatusOutcome | None:
    """Parse `kaggle kernels status` stdout; None when the text has no status line."""
    m = _STATUS_RE.search(text)
    if m is None:
        return None
    try:
        status = KernelStatus(m.group(1).upper())
    except ValueError:
        return None
    failure: str | None = None
    fm = _FAILURE_RE.search(text[m.end() :])
    if fm is not None:
        failure = fm.group(1).strip() or None
    return StatusOutcome(status=status, failure_message=failure)


# `Kernel version 3 successfully pushed.  Please check progress at https://...`
_PUSH_OK_RE = re.compile(
    r"Kernel version\s*(\d+)?\s*successfully pushed\.\s+Please check progress at (\S+)"
)
_PUSH_ERR_RE = re.compile(r"Kernel push error:\s*(.*)")
_PUSH_WARN_PREFIXES = (
    "The following are not valid",
    "Your kernel title does not resolve",
)


@dataclass(frozen=True, slots=True)
class PushOutcome:
    ok: bool
    version: int | None = None
    url: str | None = None
    error: str | None = None  # server-side "Kernel push error: <msg>"
    warnings: list[str] = field(default_factory=list)


def parse_push(text: str) -> PushOutcome | None:
    """Parse `kaggle kernels push` output. None when neither success nor a push error is
    recognisable (a crash, a network traceback: the caller decides)."""
    warnings = [
        line.strip() for line in text.splitlines() if line.strip().startswith(_PUSH_WARN_PREFIXES)
    ]
    ok = _PUSH_OK_RE.search(text)
    if ok is not None:
        version = int(ok.group(1)) if ok.group(1) else None
        return PushOutcome(ok=True, version=version, url=ok.group(2), warnings=warnings)
    err = _PUSH_ERR_RE.search(text)
    if err is not None:
        return PushOutcome(ok=False, error=err.group(1).strip() or "unknown", warnings=warnings)
    return None


@dataclass(frozen=True, slots=True)
class QuotaRow:
    resource: str  # "GPU" | "TPU"
    used_h: float
    total_h: float
    remaining_h: float
    refresh_at: float | None  # epoch seconds (refreshAt has no tz suffix; UTC assumed)


def _hours(value: object) -> float:
    text = str(value).strip().lower().removesuffix("h").strip()
    return float(text)


def _refresh(value: object) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.timestamp()


def parse_quota(text: str) -> dict[str, QuotaRow]:
    """Parse `kaggle quota --format json` -> {"GPU": QuotaRow, "TPU": ...}.
    Raises ValueError on anything else (the caller maps it to Unavailable)."""
    start = text.find("[")
    if start < 0:
        raise ValueError("no JSON array in kaggle quota output")
    rows = json.loads(text[start:])
    if not isinstance(rows, list):
        raise ValueError("kaggle quota output is not a list")
    out: dict[str, QuotaRow] = {}
    for row in rows:
        if not isinstance(row, dict) or "resource" not in row:
            continue
        name = str(row["resource"]).upper()
        used = _hours(row.get("used", "0"))
        total = _hours(row.get("total", "0"))
        remaining = _hours(row["remaining"]) if "remaining" in row else max(0.0, total - used)
        out[name] = QuotaRow(
            resource=name,
            used_h=used,
            total_h=total,
            remaining_h=remaining,
            refresh_at=_refresh(row.get("refreshAt") or row.get("refresh_at")),
        )
    return out


def parse_log(text: str) -> list[str]:
    """Kernel session log -> lines. `kaggle kernels logs` prints a JSON array of
    {stream_name, time, data} events (data usually ends with a newline); anything else is
    treated as plain text. A `\\r` inside a line keeps only what a terminal would show."""
    body = text.strip()
    if not body:
        return []
    joined: str
    try:
        events = json.loads(body)
    except ValueError:
        joined = text
    else:
        if isinstance(events, dict):
            events = [events]
        if not isinstance(events, list):
            joined = text
        else:
            joined = "".join(
                str(e.get("data", "")) for e in events if isinstance(e, dict) and e.get("data")
            )
    lines = joined.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    out: list[str] = []
    for line in lines:
        line = line.rstrip("\r")
        if "\r" in line:
            line = line.rsplit("\r", 1)[1]
        out.append(line)
    return out


def trim_post_run(lines: list[str]) -> list[str]:
    """Drop what Kaggle appends after the runner's last `::gpu:: exit` line (nbconvert
    rendering `__results__.html`: `[NbConvertApp] ...` and SyntaxWarnings from its own
    packages, seen live at integration). Nothing of the job runs after that line. A log
    without an exit line (time limit, crash) is returned whole."""
    for i in range(len(lines) - 1, -1, -1):
        line = lines[i]
        if not line.startswith(protocol.PREFIX):
            continue
        event = protocol.parse_line(line)
        if event is not None and event.t == "exit":
            return lines[: i + 1]
    return lines


def last_exit_code(lines: list[str]) -> int | None:
    """Exit code from the last `::gpu:: {"t":"exit",...}` line (bootstrap or the kaggle
    runner wrapper prints it), or None when the runner never reported one."""
    for line in reversed(lines):
        if not line.startswith(protocol.PREFIX):
            continue
        event = protocol.parse_line(line)
        if event is not None and event.t == "exit" and event.code is not None:
            return event.code
    return None


_USERNAME_RE = re.compile(r"^-\s*username:\s*(\S+)\s*$", re.M)


def parse_config_username(text: str) -> str | None:
    """`kaggle config view` prints `- username: <name>` (never the key)."""
    m = _USERNAME_RE.search(text)
    if m is None or m.group(1) == "None":
        return None
    return m.group(1)


_VERSION_RE = re.compile(r"Kaggle CLI\s+(\S+)")


def parse_version(text: str) -> str | None:
    m = _VERSION_RE.search(text)
    return m.group(1) if m else None
