"""Test-mode crash injection (phase 1; owner: group C).

The crash harness (tests/crash/) starts the daemon with GPU_ROUTER_TEST_MODE=1 and
GPU_ROUTER_CRASH_AT=<name>[,<name>...]. When execution reaches `crashpoint(name)` for a listed
name, the process dies immediately with os._exit(137) (no cleanup, like SIGKILL). Outside
test mode `crashpoint` is a no-op costing one set lookup.

Named points (keep this list in sync with the harness):
    after_place_commit     attempt row committed, adapter.submit not yet called
    after_submit_return    adapter.submit returned, record_submission not yet committed
    after_running          job transitioned to running
    mid_checkpoint         job transitioned to checkpointing
    before_fetch           remote succeeded, outputs not fetched yet
    after_fetch            outputs fetched, job not yet done
    during_cancel          cancelling, adapter.cancel returned, not yet confirmed
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping

ENV_CRASH_AT = "GPU_ROUTER_CRASH_AT"
EXIT_CODE = 137

CRASH_POINTS: frozenset[str] = frozenset(
    {
        "after_place_commit",
        "after_submit_return",
        "after_running",
        "mid_checkpoint",
        "before_fetch",
        "after_fetch",
        "during_cancel",
    }
)

_armed: frozenset[str] = frozenset()


def configure(*, test_mode: bool, environ: Mapping[str, str] | None = None) -> None:
    """Read ENV_CRASH_AT once at daemon start (only honoured when test_mode). Unknown names
    raise ValueError so a typo in a test fails loudly."""
    global _armed
    env = os.environ if environ is None else environ
    raw = env.get(ENV_CRASH_AT, "")
    names = frozenset(n.strip() for n in raw.split(",") if n.strip())
    unknown = names - CRASH_POINTS
    if unknown:
        raise ValueError(
            f"unknown crash point(s) in {ENV_CRASH_AT}: {', '.join(sorted(unknown))}; "
            f"known: {', '.join(sorted(CRASH_POINTS))}"
        )
    _armed = names if test_mode else frozenset()


def armed() -> frozenset[str]:
    """The crash points currently armed (tests and diagnostics)."""
    return _armed


def crashpoint(name: str) -> None:
    """Die with os._exit(137) if `name` is armed; otherwise return immediately."""
    if name in _armed:
        sys.stderr.write(f"gpu-router: crash point {name} reached; exiting {EXIT_CODE}\n")
        sys.stderr.flush()
        os._exit(EXIT_CODE)
