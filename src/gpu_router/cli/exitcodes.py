"""CLI exit codes (phase 2). Stable: scripts and agents branch on them (docs/cli.md).

| code | name          | when                                                              |
|------|---------------|-------------------------------------------------------------------|
| 0    | OK            | success; `run --wait` / `logs --follow` ended in `done`           |
| 1    | ERROR         | unexpected error, internal daemon error, fetch failed             |
| 2    | USAGE         | bad flags, invalid gpu.yaml or job spec, invalid request          |
| 3    | DAEMON        | daemon not running / could not be started / still recovering      |
| 4    | NOT_FOUND     | no such job (or ambiguous id prefix) or provider                  |
| 5    | CONFLICT      | action not allowed in the job's state (approve a running job, ...)|
| 10   | JOB_FAILED    | waited-for job ended `failed`                                     |
| 11   | JOB_STOPPED   | waited-for job ended `cancelled` or `denied`                      |
| 12   | NO_FIT        | `route` / `run --dry-run`: no provider can ever run this job      |
| 130  | INTERRUPTED   | Ctrl-C (a job being waited on keeps running)                      |
"""

from __future__ import annotations

from gpu_router.errors import (
    AmbiguousJobRef,
    ApiError,
    Conflict,
    DaemonUnavailable,
    GpuRouterError,
    InvalidRequest,
    InvalidSpec,
    InvalidTransition,
    JobNotFound,
    NotReady,
    ProviderNotFound,
)
from gpu_router.statemachine import JobState

OK = 0
ERROR = 1
USAGE = 2
DAEMON = 3
NOT_FOUND = 4
CONFLICT = 5
JOB_FAILED = 10
JOB_STOPPED = 11
NO_FIT = 12
INTERRUPTED = 130


def for_error(exc: GpuRouterError) -> int:
    if isinstance(exc, InvalidSpec | InvalidRequest):
        return USAGE
    if isinstance(exc, DaemonUnavailable | NotReady):
        return DAEMON
    if isinstance(exc, JobNotFound | AmbiguousJobRef | ProviderNotFound):
        return NOT_FOUND
    if isinstance(exc, Conflict | InvalidTransition):
        return CONFLICT
    if isinstance(exc, ApiError) and exc.raw_code == "invalid_transition":
        return CONFLICT
    return ERROR


def for_state(state: JobState) -> int:
    """Exit code for a job that was waited on until it reached `state`."""
    if state is JobState.DONE:
        return OK
    if state is JobState.FAILED:
        return JOB_FAILED
    if state in (JobState.CANCELLED, JobState.DENIED):
        return JOB_STOPPED
    return OK
