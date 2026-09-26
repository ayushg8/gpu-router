"""Structured logging (phase 1; owner: group A).

Named `log` (not `logging`) to avoid shadowing the stdlib. The daemon logs JSON lines to
`<home>/logs/daemon.jsonl` via a RotatingFileHandler (config.logging.max_bytes, backups);
in foreground mode it also writes a compact human format to stderr.

Record shape (one JSON object per line, keys in this order, absent keys omitted):

    {"ts": "2026-09-23T12:00:00.000Z", "level": "info", "logger": "gpu_router.engine",
     "event": "job.transition", "msg": "a7f2 provisioning -> running",
     "job_id": "a7f2c19e0b3d", "attempt_id": "a7f2c19e0b3d.1", "provider": "fake",
     "from": "provisioning", "to": "running", "reason": "started", ...extra fields}

An extra field whose name collides with a key above is written as `field_<name>`; a
traceback (exc_info) is written as `exc`.

Event names are dotted and stable: job.transition, job.note, attempt.submit, attempt.poll,
adapter.call, adapter.timeout, adapter.bug, provider.health, daemon.start, daemon.stop,
daemon.recovery, api.request (DEBUG only), statefile.write (DEBUG only).

Every handler gets `RedactingFilter` (secrets.redact on msg and all string fields).
Uvicorn's loggers are routed through the same handlers (access log disabled; api.request
at DEBUG instead).
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import sys
from datetime import UTC, datetime
from typing import Any

from gpu_router.paths import Paths
from gpu_router.secrets import redact

ROOT_LOGGER = "gpu_router"
UVICORN_LOGGERS: tuple[str, ...] = ("uvicorn", "uvicorn.error", "uvicorn.access")
#: Marker attribute on handlers installed by setup_daemon_logging (for idempotency).
_HANDLER_MARK = "_gpu_router_handler"
#: Keys the formatter owns; extra fields with these names are written as "field_<name>".
_RESERVED = frozenset({"ts", "level", "logger", "event", "msg", "exc"})


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _jsonable(value: Any) -> Any:
    """Best-effort conversion so an odd field value never breaks logging."""
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple | set | frozenset):
        return [_jsonable(v) for v in value]
    return str(value)


def _redact_value(value: Any) -> Any:
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, dict):
        return {k: _redact_value(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_redact_value(v) for v in value]
    return value


class JsonFormatter(logging.Formatter):
    """Formats a LogRecord as one JSON line in the shape documented above.
    Extra fields come from `record.fields` (set by `log_event`)."""

    def format(self, record: logging.LogRecord) -> str:
        out: dict[str, Any] = {
            "ts": _iso(record.created),
            "level": record.levelname.lower(),
            "logger": record.name,
        }
        event = getattr(record, "event", None)
        if event:
            out["event"] = str(event)
        out["msg"] = record.getMessage()
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            for key, value in fields.items():
                name = f"field_{key}" if key in _RESERVED else str(key)
                out[name] = _jsonable(value)
        if record.exc_info:
            out["exc"] = redact(self.formatException(record.exc_info))
        elif record.exc_text:
            out["exc"] = redact(record.exc_text)
        return json.dumps(out, ensure_ascii=False, separators=(",", ":"))


class HumanFormatter(logging.Formatter):
    """Compact stderr format for `gpu daemon run --foreground`, e.g.
    `12:00:00 info  job.transition a7f2 provisioning -> running provider=fake`."""

    def format(self, record: logging.LogRecord) -> str:
        stamp = datetime.fromtimestamp(record.created, UTC).strftime("%H:%M:%S")
        parts = [stamp, f"{record.levelname.lower():<5}"]
        event = getattr(record, "event", None)
        if event:
            parts.append(str(event))
        parts.append(record.getMessage())
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict) and fields:
            parts.append(" ".join(f"{k}={_jsonable(v)}" for k, v in fields.items()))
        line = " ".join(parts)
        if record.exc_info:
            line += "\n" + redact(self.formatException(record.exc_info))
        return line


class RedactingFilter(logging.Filter):
    """Applies gpu_router.secrets.redact to record.msg/args and every str in record.fields."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:  # malformed %-args: keep the raw msg rather than dropping the record
            message = str(record.msg)
        record.msg = redact(message)
        record.args = None
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            record.fields = {k: _redact_value(v) for k, v in fields.items()}
        return True


def setup_daemon_logging(
    paths: Paths,
    *,
    level: str = "INFO",
    max_bytes: int = 10 * 1024 * 1024,
    backups: int = 5,
    foreground: bool = False,
) -> None:
    """Configure the root `gpu_router` logger and uvicorn loggers for the daemon process.
    Idempotent (safe to call twice in tests)."""
    paths.ensure()
    lvl = level.upper()
    root = logging.getLogger(ROOT_LOGGER)
    root.setLevel(lvl)
    root.propagate = False
    for handler in list(root.handlers):
        if getattr(handler, _HANDLER_MARK, False):
            root.removeHandler(handler)
            handler.close()

    handlers: list[logging.Handler] = []
    file_handler = logging.handlers.RotatingFileHandler(
        paths.daemon_log, maxBytes=max_bytes, backupCount=backups, encoding="utf-8"
    )
    file_handler.setFormatter(JsonFormatter())
    handlers.append(file_handler)
    if foreground:
        stream = logging.StreamHandler(sys.stderr)
        stream.setFormatter(HumanFormatter())
        handlers.append(stream)
    for handler in handlers:
        handler.addFilter(RedactingFilter())
        setattr(handler, _HANDLER_MARK, True)
        root.addHandler(handler)

    for name in UVICORN_LOGGERS:
        uv = logging.getLogger(name)
        for handler in list(uv.handlers):
            uv.removeHandler(handler)
        uv.propagate = False
        if name == "uvicorn.access":
            uv.disabled = True  # api.request at DEBUG replaces the access log
            continue
        uv.setLevel(lvl)
        for handler in handlers:
            uv.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """`logging.getLogger(name)`; name should be a module path under gpu_router."""
    return logging.getLogger(name)


def log_event(
    logger: logging.Logger,
    event: str,
    msg: str,
    /,
    *,
    level: int = logging.INFO,
    exc_info: bool | BaseException = False,
    **fields: Any,
) -> None:
    """Emit one structured record: `event` + human `msg` + JSON-serialisable `fields`."""
    if not logger.isEnabledFor(level):
        return
    logger.log(
        level,
        "%s",
        msg,
        exc_info=exc_info,
        extra={"event": event, "fields": fields},
        stacklevel=2,
    )
