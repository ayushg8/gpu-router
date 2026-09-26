"""Does bare `gpu` (on a terminal) launch the setup wizard first? (phase 8b)

STDLIB ONLY (invariant 14): entry.py calls `first_run_mode()` before the shell starts, so
this module must stay as cheap as `paths.py`.

The wizard is offered when `<home>/setup.json` is missing (never set up: "new") or records
a run that was interrupted before it finished and was never completed ("resume"). It is
never offered again once a run completed or the user answered "not now" (`dismissed_at`),
in test mode (`GPU_ROUTER_TEST_MODE`), or with `GPU_ROUTER_NO_SETUP=1`. `gpu setup` always
works, whatever this says.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any

from gpu_router.paths import home_from_env

__all__ = ["ENV_NO_SETUP", "SETUP_FILE", "first_run_mode", "read_raw", "setup_file"]

SETUP_FILE = "setup.json"
ENV_NO_SETUP = "GPU_ROUTER_NO_SETUP"
_TRUE = {"1", "true", "yes", "on"}


def setup_file(home: str) -> str:
    return os.path.join(home, SETUP_FILE)


def read_raw(home: str) -> dict[str, Any] | None:
    """setup.json as a dict; None when missing. An unreadable or broken file is `{}` (the
    wizard rewrites it), so a damaged file never loops the first-run prompt forever."""
    try:
        with open(setup_file(home), encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def first_run_mode(environ: Mapping[str, str] | None = None) -> str | None:
    """ "new" | "resume" | None (open the shell straight away)."""
    env = os.environ if environ is None else environ
    if env.get("GPU_ROUTER_TEST_MODE", "").strip().lower() in _TRUE:
        return None
    if env.get(ENV_NO_SETUP, "").strip().lower() in _TRUE:
        return None
    raw = read_raw(home_from_env(env))
    if raw is None:
        return "new"
    if raw.get("completed_at") or raw.get("dismissed_at"):
        return None
    run = raw.get("run")
    if isinstance(run, dict) and run.get("started_at") and not run.get("finished_at"):
        return "resume"
    return None
