"""Engine logging helper (owner: group C).

`emit()` wraps gpu_router.log.log_event so that a logging failure (full disk, a bug in a
formatter) can never take down a job driver: observability is best effort, the Store is
the source of truth.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any

from gpu_router.log import get_logger, log_event

logger = get_logger("gpu_router.engine")


def emit(
    event: str,
    msg: str,
    /,
    *,
    level: int = logging.INFO,
    exc_info: bool | BaseException = False,
    log: logging.Logger | None = None,
    **fields: Any,
) -> None:
    """Emit one structured record; swallow any error raised while logging."""
    try:
        log_event(log or logger, event, msg, level=level, exc_info=exc_info, **fields)
    except Exception:  # logging must never break the engine
        with contextlib.suppress(Exception):
            (log or logger).log(level, "%s: %s", event, msg, exc_info=exc_info)


def fmt_duration(seconds: float) -> str:
    """Compact human duration: 45s, 4m, 1h05m, 2d3h."""
    s = max(0, round(seconds))
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m"
    h, m = divmod(m, 60)
    if h < 48:
        return f"{h}h{m:02d}m" if m else f"{h}h"
    d, h = divmod(h, 24)
    return f"{d}d{h}h" if h else f"{d}d"
