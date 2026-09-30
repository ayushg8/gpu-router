"""HTTP API v1 models, shared by the daemon (routes) and the client (phase 1; real code;
owner: group C). Additive-only (invariant 17): add optional fields, never rename/remove.

Imports pydantic and gpu_router models only (no fastapi, no httpx) so the CLI, shell and
MCP server can import it cheaply. Timestamps serialize as ISO-8601 `Z` (models.Timestamp).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from gpu_router.adapters.base import Capabilities
from gpu_router.models import (
    Attempt,
    Checkpoint,
    Job,
    JobEvent,
    JobSpec,
    JobState,
    ProviderHealth,
    ProviderState,
    QuotaSnapshot,
    Timestamp,
)
from gpu_router.policy import PolicyConfig
from gpu_router.router.base import RouteDecision

API_VERSION = 1
API_PREFIX = "/v1"
VERSION_HEADER = "X-Gpu-Router-Version"
IDEMPOTENCY_HEADER = "Idempotency-Key"
NDJSON = "application/x-ndjson"
LONG_POLL_MAX_S = 30.0
LOG_HEARTBEAT_S = 15.0

__all__ = [
    "API_PREFIX",
    "API_VERSION",
    "DecisionRequest",
    "ErrorBody",
    "ErrorEnvelope",
    "EventList",
    "HealthView",
    "JobDetail",
    "JobList",
    "JobView",
    "LogRecord",
    "PolicyView",
    "ProviderView",
    "RouteDecision",
    "RuntimeInfo",
    "StatusView",
    "SubmitRequest",
]


class _Api(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")  # tolerate newer daemons


# --------------------------------------------------------------------------- requests


class SubmitRequest(BaseModel):
    """POST /v1/jobs and POST /v1/route. The client builds the JobSpec (phase 2 merges
    gpu.yaml + flags); the daemon re-validates it."""

    model_config = ConfigDict(extra="forbid")

    spec: JobSpec


class DecisionRequest(BaseModel):
    """POST /v1/jobs/{ref}/approve|deny."""

    model_config = ConfigDict(extra="forbid")

    reason: str | None = Field(default=None, max_length=500)


# --------------------------------------------------------------------------- responses


class HealthView(_Api):
    ok: bool = True
    version: str
    api_version: int = API_VERSION
    ready: bool  # False while recovery runs (mutations get 503 not_ready)
    pid: int
    started_at: Timestamp
    test_mode: bool = False
    #: the notification backend this daemon uses ("osascript", "terminal-notifier" or
    #: "off: <why>"); doctor reports it instead of guessing from its own PATH (phase 8
    #: review fix). None from a daemon without a notifier.
    notifications: str | None = None


class JobView(Job):
    """A Job as the API returns it (same fields; room for computed extras)."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    @classmethod
    def of(cls, job: Job) -> JobView:
        return cls.model_validate(job.model_dump())


class JobList(_Api):
    jobs: list[JobView]
    next_before: Timestamp | None = None  # pass as ?before= for the next page


class JobDetail(_Api):
    job: JobView
    attempts: list[Attempt]
    checkpoints: list[Checkpoint]
    events: list[JobEvent]  # last 50, oldest first
    route: RouteDecision | None = None  # decision recorded with the latest placement


class EventList(_Api):
    events: list[JobEvent]
    next: int  # pass as ?after= to continue (max seq seen)


class LogRecord(_Api):
    """One NDJSON line of GET /v1/jobs/{ref}/logs. Exactly one of the shapes:
    {attempt, offset, line} | {heartbeat: true} | {eof: true, state}."""

    attempt: int | None = None
    offset: int | None = None  # 0-based line index within that attempt's log
    line: str | None = None
    heartbeat: bool | None = None
    eof: bool | None = None
    state: JobState | None = None


class ProviderView(_Api):
    name: str
    display_name: str
    kind: str
    enabled: bool
    health: ProviderHealth
    health_reason: str | None = None  # an unhealthy one ends with "; re-checking in 40s"
    state: ProviderState | None = None
    # when the health loop re-checks an unhealthy provider (backoff; None while healthy)
    next_healthcheck_at: Timestamp | None = None
    capabilities: Capabilities
    gpus: list[str]  # labels, e.g. ["P100", "2xT4"]
    session_hours: float | None = None
    live_attempts: int = 0
    quota: QuotaSnapshot | None = None


class StatusView(_Api):
    ready: bool
    counts: dict[str, int]  # non-terminal state -> count
    active: list[JobView]  # non-terminal jobs, same order as state.json
    recent: list[JobView]  # finished within the last 10 min (routes.RECENT_FINISHED_S)
    providers: list[ProviderView]


class PolicyView(_Api):
    """GET/PUT /v1/policy (phase 5): the approval rules in force. PUT takes a full
    `PolicyConfig` document and persists it to config.yaml `policy:`."""

    name: str  # policy implementation ("rules"; "spec_only" in engine tests)
    editable: bool  # False when the daemon runs a policy without rules
    policy: PolicyConfig | None = None  # rules in force (None when not editable)
    defaults: PolicyConfig = Field(default_factory=PolicyConfig)
    config_path: str | None = None  # where `gpu policy set` writes


class ErrorBody(_Api):
    code: str  # errors.ErrorCode value (unknown codes tolerated)
    message: str
    hint: str | None = None
    detail: dict[str, Any] = Field(default_factory=dict)


class ErrorEnvelope(_Api):
    error: ErrorBody


# --------------------------------------------------------------------------- daemon.json


class RuntimeInfo(_Api):
    """Contents of <home>/daemon.json, written by the daemon once it is listening and
    removed on clean shutdown. Clients use it to find the port. The token is NOT here; it is
    in daemon.token (0600)."""

    pid: int
    port: int
    host: Literal["127.0.0.1"] = "127.0.0.1"
    version: str
    api_version: int = API_VERSION
    started_at: Timestamp
    test_mode: bool = False

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"
