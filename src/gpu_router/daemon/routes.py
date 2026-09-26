"""/v1 endpoints (phase 1; owner: group C). Contract: CLAUDE.md "Daemon HTTP API"; models in
gpu_router.api. Handlers are thin: resolve the runtime, call the Supervisor or Store (on the
event loop, invariant 9), convert to api models. Actor strings: the `X-Gpu-Router-Client`
request header ("cli", "shell", "mcp"...) maps to actor "user:<client>" ("agent" for mcp),
default "api".

Log streaming: `offset` applies to the attempt named by `attempt`, or to the first attempt
streamed when `attempt` is omitted; `protocol=true` includes `::gpu::` lines.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
from collections.abc import AsyncIterator
from typing import Annotated, Any

from fastapi import APIRouter, Body, Depends, Header, Query, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from gpu_router import __version__
from gpu_router.api import (
    IDEMPOTENCY_HEADER,
    LOG_HEARTBEAT_S,
    LONG_POLL_MAX_S,
    NDJSON,
    DecisionRequest,
    EventList,
    HealthView,
    JobDetail,
    JobList,
    JobView,
    LogRecord,
    PolicyView,
    ProviderView,
    RouteDecision,
    StatusView,
    SubmitRequest,
)
from gpu_router.daemon.runtime import DaemonRuntime
from gpu_router.errors import Conflict, InvalidRequest
from gpu_router.models import JobSpec, QuotaSnapshot
from gpu_router.policy import PolicyConfig
from gpu_router.router.base import JobEstimate
from gpu_router.statemachine import TERMINAL_STATES, JobState, Reason, is_terminal

CLIENT_HEADER = "X-Gpu-Router-Client"
#: Seconds between heartbeat lines on an idle log follow (tests shorten it).
HEARTBEAT_S = LOG_HEARTBEAT_S
MAX_IDEMPOTENCY_KEY = 200
_CLIENT_RE = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")


#: How long finished jobs stay in /v1/status `recent` (the shell panel, `gpu status`).
RECENT_FINISHED_S = 600.0


def actor_for(client: str | None) -> str:
    """Map the X-Gpu-Router-Client header to an event actor."""
    if not client:
        return "api"
    c = client.strip().lower()
    if c in ("mcp", "agent"):
        return "agent"
    if not _CLIENT_RE.match(c):
        return "api"
    return f"user:{c}"


def _parse_states(raw: list[str] | None) -> list[JobState] | None:
    if not raw:
        return None
    out: list[JobState] = []
    for item in raw:
        for part in item.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                out.append(JobState(part))
            except ValueError:
                raise InvalidRequest(
                    f"unknown job state {part!r}",
                    hint="states: " + ", ".join(s.value for s in JobState),
                ) from None
    return out or None


def _parse_ts(raw: str | None) -> float | None:
    if raw is None or raw == "":
        return None
    try:
        return float(raw)
    except ValueError:
        pass
    from datetime import datetime

    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        raise InvalidRequest(
            f"not a timestamp: {raw!r}", hint="use epoch seconds or ISO-8601 with Z"
        ) from None
    if dt.tzinfo is None:
        raise InvalidRequest(f"timestamp {raw!r} has no timezone", hint="append Z for UTC")
    return dt.timestamp()


def get_runtime(request: Request) -> DaemonRuntime:
    """FastAPI dependency: the runtime stored on app.state by create_app."""
    runtime: DaemonRuntime = request.app.state.runtime
    return runtime


Runtime = Annotated[DaemonRuntime, Depends(get_runtime)]
Client = Annotated[str | None, Header(alias=CLIENT_HEADER)]


def build_router() -> APIRouter:
    """Return an APIRouter (no prefix; app.py mounts it at /v1) with every endpoint:

    GET  /health                       -> HealthView           (public)
    GET  /status                       -> StatusView
    POST /jobs                         -> 201/200 JobView      (Idempotency-Key header)
    GET  /jobs                         -> JobList
    GET  /jobs/{ref}                   -> JobDetail
    GET  /jobs/{ref}/events            -> EventList
    GET  /jobs/{ref}/logs              -> NDJSON LogRecord stream (StreamingResponse)
    POST /jobs/{ref}/cancel            -> JobView
    POST /jobs/{ref}/approve           -> JobView
    POST /jobs/{ref}/deny              -> JobView
    POST /jobs/{ref}/fetch             -> JobView
    POST /route                        -> RouteDecision
    GET  /providers                    -> list[ProviderView]
    GET  /providers/{name}             -> ProviderView
    POST /providers/{name}/healthcheck -> ProviderView
    GET  /quota                        -> list[QuotaSnapshot]
    GET  /events                       -> EventList            (long poll, timeout<=30s)
    POST /daemon/shutdown              -> 202
    """
    router = APIRouter()

    # ------------------------------------------------------------------ daemon

    @router.get("/health", response_model=HealthView)
    async def health(rt: Runtime) -> HealthView:
        from gpu_router.notify.backends import describe

        notifier = getattr(rt, "notifier", None)
        return HealthView(
            version=__version__,
            ready=rt.ready,
            pid=os.getpid(),
            started_at=rt.started_at,
            test_mode=rt.config.test_mode,
            notifications=describe(notifier.backend) if notifier is not None else None,
        )

    @router.get("/status", response_model=StatusView)
    async def status(rt: Runtime) -> StatusView:
        store = rt.store
        now = rt.clock.now()
        rank = {
            JobState.RUNNING: 0,
            JobState.CHECKPOINTING: 0,
            JobState.AWAITING_APPROVAL: 1,
            JobState.PROVISIONING: 2,
            JobState.MIGRATING: 2,
            JobState.CANCELLING: 3,
        }
        active = sorted(
            store.non_terminal_jobs(), key=lambda j: (rank.get(j.state, 4), j.created_at, j.id)
        )
        counts = {
            str(s): n
            for s, n in store.count_by_state().items()
            if s not in TERMINAL_STATES and n > 0
        }
        # The shell panel and `gpu status` keep finished jobs for 10 min; the Claude status
        # line has its own window (statusline.finished_visible_s, 0 by default, D55).
        recent = store.recent_finished(now - RECENT_FINISHED_S)
        return StatusView(
            ready=rt.ready,
            counts=counts,
            active=[JobView.of(j) for j in active],
            recent=[JobView.of(j) for j in recent],
            providers=rt.supervisor.provider_views(),
        )

    @router.post("/daemon/shutdown", status_code=202)
    async def shutdown(rt: Runtime) -> Response:
        rt.request_shutdown()
        return JSONResponse(
            {"ok": True, "message": "shutting down; remote runs keep going"}, status_code=202
        )

    # ------------------------------------------------------------------ jobs

    @router.post("/jobs", response_model=JobView, status_code=201)
    async def submit(
        rt: Runtime,
        response: Response,
        client: Client = None,
        body: SubmitRequest = Body(...),  # noqa: B008
        idempotency_key: Annotated[str | None, Header(alias=IDEMPOTENCY_HEADER)] = None,
    ) -> JobView:
        key = idempotency_key.strip() if idempotency_key else None
        if key is not None and (not key or len(key) > MAX_IDEMPOTENCY_KEY):
            raise InvalidRequest(f"{IDEMPOTENCY_HEADER} must be 1-{MAX_IDEMPOTENCY_KEY} characters")
        job, created = await rt.supervisor.submit(
            body.spec, actor=actor_for(client), request_id=key
        )
        if not created:
            spec_hash = hashlib.sha256(body.spec.model_dump_json().encode()).hexdigest()
            if spec_hash != job.spec_hash:
                raise Conflict(
                    f"{IDEMPOTENCY_HEADER} {key!r} was already used for a different job "
                    f"({job.short_id})",
                    hint="use a new key for a new job",
                    detail={"job_id": job.id},
                )
            response.status_code = 200
        return JobView.of(job)

    @router.get("/jobs", response_model=JobList)
    async def list_jobs(
        rt: Runtime,
        state: Annotated[list[str] | None, Query()] = None,
        project_dir: str | None = None,
        limit: Annotated[int, Query(ge=1, le=500)] = 50,
        before: str | None = None,
    ) -> JobList:
        jobs = rt.store.list_jobs(
            states=_parse_states(state),
            project_dir=project_dir,
            limit=limit,
            before=_parse_ts(before),
        )
        next_before = jobs[-1].created_at if len(jobs) == limit else None
        return JobList(jobs=[JobView.of(j) for j in jobs], next_before=next_before)

    @router.get("/jobs/{ref}", response_model=JobDetail)
    async def job_detail(rt: Runtime, ref: str) -> JobDetail:
        store = rt.store
        job = store.resolve_ref(ref)
        events = store.events_for(job.id, limit=100_000)
        route: RouteDecision | None = None
        for ev in reversed(events):
            if ev.kind == "transition" and ev.reason == Reason.PLACED and ev.detail:
                try:
                    route = RouteDecision.model_validate(ev.detail)
                except ValueError:
                    route = None
                break
        return JobDetail(
            job=JobView.of(job),
            attempts=store.attempts_for(job.id),
            checkpoints=store.checkpoints_for(job.id),
            events=events[-50:],
            route=route,
        )

    @router.get("/jobs/{ref}/events", response_model=EventList)
    async def job_events(
        rt: Runtime, ref: str, after: Annotated[int, Query(ge=0)] = 0
    ) -> EventList:
        job = rt.store.resolve_ref(ref)
        events = rt.store.events_for(job.id, after_seq=after)
        return EventList(events=events, next=events[-1].seq if events else after)

    @router.get("/jobs/{ref}/logs")
    async def job_logs(
        rt: Runtime,
        ref: str,
        attempt: Annotated[int | None, Query(ge=1)] = None,
        offset: Annotated[int, Query(ge=0)] = 0,
        follow: bool = False,
        protocol: bool = False,
    ) -> StreamingResponse:
        job = rt.store.resolve_ref(ref)
        if attempt is not None and attempt not in {a.n for a in rt.store.attempts_for(job.id)}:
            raise InvalidRequest(f"job {job.short_id} has no attempt {attempt}")
        return StreamingResponse(
            _stream_logs(
                rt, job.id, attempt=attempt, offset=offset, follow=follow, include_protocol=protocol
            ),
            media_type=NDJSON,
        )

    @router.post("/jobs/{ref}/cancel", response_model=JobView)
    async def cancel(rt: Runtime, ref: str, client: Client = None) -> JobView:
        return JobView.of(await rt.supervisor.cancel(ref, actor=actor_for(client)))

    @router.post("/jobs/{ref}/approve", response_model=JobView)
    async def approve(
        rt: Runtime, ref: str, client: Client = None, body: DecisionRequest | None = None
    ) -> JobView:
        reason = body.reason if body else None
        return JobView.of(await rt.supervisor.approve(ref, actor=actor_for(client), reason=reason))

    @router.post("/jobs/{ref}/deny", response_model=JobView)
    async def deny(
        rt: Runtime, ref: str, client: Client = None, body: DecisionRequest | None = None
    ) -> JobView:
        reason = body.reason if body else None
        return JobView.of(await rt.supervisor.deny(ref, actor=actor_for(client), reason=reason))

    @router.post("/jobs/{ref}/fetch", response_model=JobView)
    async def fetch(rt: Runtime, ref: str, client: Client = None) -> JobView:
        return JobView.of(await rt.supervisor.request_fetch(ref, actor=actor_for(client)))

    # ------------------------------------------------------------------ routing / providers

    @router.post("/route", response_model=RouteDecision)
    async def route(rt: Runtime, body: SubmitRequest) -> RouteDecision:
        # phase 5: the project's VRAM/runtime estimate, like a real submit's bundle has
        estimate = await _dry_estimate(body.spec)
        return rt.supervisor.dry_route(body.spec, estimate=estimate)

    @router.get("/providers", response_model=list[ProviderView])
    async def providers(rt: Runtime) -> list[ProviderView]:
        return rt.supervisor.provider_views()

    @router.get("/providers/{name}", response_model=ProviderView)
    async def provider(rt: Runtime, name: str) -> ProviderView:
        return rt.supervisor.provider_view(name)

    @router.post("/providers/{name}/healthcheck", response_model=ProviderView)
    async def healthcheck(rt: Runtime, name: str) -> ProviderView:
        return await rt.supervisor.healthcheck(name)

    @router.get("/quota", response_model=list[QuotaSnapshot])
    async def quota(rt: Runtime, refresh: bool = False) -> list[QuotaSnapshot]:
        """Quota ledger views (phase 5). Stale live readings are refreshed first, waiting
        at most `routing.quota.wait_s`; `refresh=true` re-reads every live provider."""
        if rt.quota is None:
            return await rt.supervisor.quotas()
        await rt.quota.refresh(force=refresh, wait_s=rt.quota.settings.wait_s)
        return rt.quota.snapshots()

    @router.get("/policy", response_model=PolicyView)
    async def get_policy(rt: Runtime) -> PolicyView:
        return _policy_view(rt)

    @router.put("/policy", response_model=PolicyView)
    async def put_policy(rt: Runtime, body: PolicyConfig, client: Client = None) -> PolicyView:
        """Replace the approval rules and persist them to config.yaml `policy:`."""
        from gpu_router.config import set_config_section
        from gpu_router.engine._obs import emit
        from gpu_router.policy import RulesPolicy

        policy = rt.supervisor.deps.policy
        if not isinstance(policy, RulesPolicy):
            raise Conflict(
                f"this daemon's approval policy ({policy.name}) has no editable rules",
                hint="restart the daemon to load the rules from config.yaml",
            )
        doc = body.model_dump(mode="json")
        await asyncio.to_thread(set_config_section, rt.paths, "policy", doc)
        policy.update(body)
        rt.config.policy = doc
        emit(
            "policy.update",
            f"approval rules updated by {actor_for(client)}",
            actor=actor_for(client),
            policy=doc,
        )
        return _policy_view(rt)

    @router.get("/events", response_model=EventList)
    async def events(
        rt: Runtime,
        after: Annotated[int, Query(ge=0)] = 0,
        timeout: Annotated[float, Query(ge=0)] = 0.0,  # noqa: ASYNC109
    ) -> EventList:
        found = rt.store.events_after(after)
        if not found and timeout > 0:
            await rt.bus.wait_for_events(after, min(timeout, LONG_POLL_MAX_S))
            found = rt.store.events_after(after)
        return EventList(events=found, next=found[-1].seq if found else after)

    return router


def _line(record: LogRecord) -> bytes:
    return (record.model_dump_json(exclude_none=True) + "\n").encode("utf-8")


async def _stream_logs(
    rt: DaemonRuntime,
    job_id: str,
    *,
    attempt: int | None,
    offset: int,
    follow: bool,
    include_protocol: bool,
) -> AsyncIterator[bytes]:
    """NDJSON generator for GET /v1/jobs/{ref}/logs (CLAUDE.md "Log streaming")."""
    from gpu_router.engine.capture import read_log_lines

    store = rt.store
    positions: dict[int, int] = {}  # attempt n -> next line index to read

    def attempt_numbers() -> list[int]:
        ns = [a.n for a in store.attempts_for(job_id)]
        return [attempt] if attempt is not None else ns

    first = True

    def drain() -> list[bytes]:
        nonlocal first
        out: list[bytes] = []
        for n in attempt_numbers():
            if n not in positions:
                positions[n] = offset if first else 0
                first = False
            path = rt.paths.job_log(job_id, n)
            for idx, text in read_log_lines(
                path, offset=positions[n], include_protocol=include_protocol
            ):
                out.append(_line(LogRecord(attempt=n, offset=idx, line=text)))
                positions[n] = idx + 1
            # protocol lines are skipped in the output but still advance the position
            total = _count_lines(path)
            if total > positions[n]:
                positions[n] = total
        return out

    for chunk in drain():
        yield chunk
    if not follow:
        return
    while True:
        job = store.get_job(job_id)
        if is_terminal(job.state):
            for chunk in drain():
                yield chunk
            yield _line(LogRecord(eof=True, state=job.state))
            return
        changed = await rt.bus.wait_for_job_change(job_id, HEARTBEAT_S)
        emitted = drain()
        for chunk in emitted:
            yield chunk
        if not changed and not emitted:
            yield _line(LogRecord(heartbeat=True))


def _count_lines(path: Any) -> int:
    """Complete lines in a captured log (a partial trailing line is not counted)."""
    try:
        with open(path, "rb") as fh:
            return sum(1 for raw in fh if raw.endswith(b"\n"))
    except FileNotFoundError:
        return 0


DRY_ESTIMATE_TIMEOUT_S = 10.0


async def _dry_estimate(spec: JobSpec) -> JobEstimate | None:
    """The project's estimate for a dry run, off the event loop and bounded in time."""
    from gpu_router.engine.context import project_estimate

    try:
        return await asyncio.wait_for(
            asyncio.to_thread(project_estimate, spec), timeout=DRY_ESTIMATE_TIMEOUT_S
        )
    except Exception:  # includes TimeoutError: a dry run without an estimate is fine
        return None


def _policy_view(rt: DaemonRuntime) -> PolicyView:
    from gpu_router.policy import RulesPolicy

    policy = rt.supervisor.deps.policy
    editable = isinstance(policy, RulesPolicy)
    return PolicyView(
        name=policy.name,
        editable=editable,
        policy=policy.config if isinstance(policy, RulesPolicy) else None,
        config_path=str(rt.paths.config),
    )
