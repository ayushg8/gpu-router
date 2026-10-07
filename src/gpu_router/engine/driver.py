"""JobDriver: one asyncio task per non-terminal job (phase 1; owner: group C).

The driver is the state machine in motion; CLAUDE.md "Driver semantics" is its spec and the
transition table in statemachine.py bounds what it may do. Structure: `run()` loops

    job = store.get_job(id)
    if terminal: return
    if job.not_before and now < job.not_before: sleep until then (or until woken)
    step = STEP_FOR_STATE[job.state]; await step(job)

Each step performs at most one state transition (or one adapter round trip plus the
resulting writes) and returns, so the loop re-reads the job and a crash between steps is
always covered by statemachine.RECOVERY. All Store calls happen on the event-loop thread;
adapter calls go through deps.caller (invariant 9). A StaleState from the store means
someone else (a user action) changed the job: reload and continue, never overwrite.

Wake-ups: `wake()` interrupts any sleep (user cancel/approve/deny, provider back to healthy).
Sleeps use deps.clock.sleep so FakeClock drives tests.

Unexpected exceptions inside a step: add note internal_error (message + exception class,
traceback in the daemon log as `engine.bug`), back off, retry; after
config.engine.internal_error_limit consecutive failures move the job to failed
(failure_kind=internal) unless it is cancelling.

Recovery (daemon restart): every RecoveryAction maps onto the normal step for the job's
state, because each step is written to be re-entrant from persisted facts alone:
RESOLVE_ATTEMPT = a `submitting` attempt this process did not create is resolved by
attempt key before anything else happens to the job (invariant 6); REATTACH = the log
capture reopens at the persisted line count and polling resumes; REMIGRATE / RECANCEL /
REROUTE / WAIT / RESUME just re-run their step.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import SecretStr

from gpu_router.adapters.base import AttemptContext, RemotePhase, RemoteRef, RemoteStatus
from gpu_router.checkpoint.hub import BULK_TIMEOUT_S
from gpu_router.checkpoint.storage import StorageError
from gpu_router.engine import crashpoints
from gpu_router.engine._obs import emit, fmt_duration
from gpu_router.engine.backoff import backoff_s, cooldown_until
from gpu_router.engine.capture import CaptureResult, LogCapture, gpu_mismatch
from gpu_router.engine.context import build_routing_context, quota_left_hours
from gpu_router.errors import (
    DEFINITIVE_SUBMIT_ERRORS,
    AdapterContractViolation,
    AdapterError,
    AuthRequired,
    InvalidJob,
    InvalidTransition,
    JobNotFound,
    NotFound,
    Permanent,
    ProviderNotFound,
    QuotaExhausted,
    RateLimited,
    SecretsError,
    StaleState,
    Unavailable,
)
from gpu_router.models import (
    Attempt,
    AttemptPatch,
    Checkpoint,
    DataRef,
    FailureKind,
    Job,
    JobEvent,
    JobPatch,
    ProviderHealth,
)
from gpu_router.policy import ApprovalDecision, hours_limit_s
from gpu_router.protocol import merge_metrics
from gpu_router.router.base import (
    TEMPORARY_REJECTIONS,
    Candidate,
    RejectCode,
    RouteDecision,
    RouteOutcome,
)
from gpu_router.statemachine import (
    LIVE_ATTEMPT_STATES,
    AttemptState,
    JobState,
    Reason,
    is_terminal,
)
from gpu_router.store import UNREACHABLE_ERROR_KIND, AttemptChange

if TYPE_CHECKING:
    from gpu_router.checkpoint.hub import CheckpointHub, StoredCheckpoint
    from gpu_router.engine.deps import EngineDeps
    from gpu_router.statemachine import RecoveryAction

ACTOR = "engine"
#: `_Handoff.code` of a checkpoint request made because the job ran past its hours (D48)
OVERRUN = "overrun"

#: Exceptions an adapter round trip may legitimately end with (anything else is a bug).
ADAPTER_FAILURES: tuple[type[Exception], ...] = (
    AdapterError,
    AdapterContractViolation,
    ProviderNotFound,
)

_logger = logging.getLogger("gpu_router.engine.driver")

#: Marker stored in attempts.error_kind while a submit outcome is ambiguous (invariant 6).
AMBIGUOUS_PREFIX = "ambiguous:"

#: Job states whose current attempt may hold a provider slot.
_HOLDS_ATTEMPT = frozenset(
    {
        JobState.PROVISIONING,
        JobState.RUNNING,
        JobState.CHECKPOINTING,
        JobState.MIGRATING,
        JobState.CANCELLING,
    }
)
#: Rejections a WAIT can be waiting on (router/base.py TEMPORARY_REJECTIONS + QUOTA).
_WAIT_CODES = TEMPORARY_REJECTIONS | {RejectCode.QUOTA}
#: ... of which these clear at a known quota reset.
_RESET_CODES = frozenset({RejectCode.QUOTA, RejectCode.EXHAUSTED})


def remote_ref(attempt: Attempt) -> RemoteRef:
    assert attempt.remote_id is not None
    return RemoteRef(
        remote_id=attempt.remote_id, url=attempt.remote_url, meta=dict(attempt.remote_meta)
    )


def _err_text(exc: BaseException) -> str:
    """An adapter error's text for events and attempts, redacted again (invariant 12:
    adapters redact their snippets, this catches one that forgot)."""
    from gpu_router.secrets import redact

    msg = getattr(exc, "message", None) or str(exc) or type(exc).__name__
    return redact(str(msg))


class JobDriver:
    def __init__(
        self,
        job_id: str,
        deps: EngineDeps,
        *,
        recovery: RecoveryAction | None = None,
        on_attempt_ended: Callable[[str], None] | None = None,
    ) -> None:
        """`recovery` is set when the supervisor starts the driver during daemon startup;
        the first loop iteration then performs that RecoveryAction before normal steps.
        `on_attempt_ended(job_id)` is called when a step ended this job's live attempt
        (its provider has a free slot again: the supervisor wakes queued jobs, D44)."""
        self.job_id = job_id
        self.deps = deps
        self.recovery = recovery
        self._on_attempt_ended = on_attempt_ended
        self._wake = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._consecutive_errors = 0
        # attempts placed by THIS process whose submit has not been tried yet
        self._fresh_attempts: set[str] = set()
        self._retry_k = 0  # consecutive transient failures (sleep backoff)
        self._wait_k = 0  # consecutive WAIT decisions (queue backoff)
        self._capture: LogCapture | None = None
        self._capture_attempt: str | None = None
        self._stale_noted: set[str] = set()
        self._ckpt_began_at: float | None = None
        # attempt id -> (first failure, latest failure) of the current run of failed
        # status/lookup calls made BY THIS PROCESS. Unreachability is measured from here,
        # never from persisted timestamps: time the daemon was down or the Mac was asleep
        # is not evidence that the provider is unreachable.
        self._unreachable: dict[str, tuple[float, float]] = {}
        self._secrets_registered = False
        # phase 5 (checkpoint storage): planned handoffs per attempt id, the attempt whose
        # checkpoints were last reconciled with storage, and one-time notes
        self._handoffs: dict[str, _Handoff] = {}
        self._handoff_checked: set[str] = set()
        self._reconciled: str | None = None
        self._noted: set[str] = set()
        self._route_now = False  # a queued job was woken: route before not_before
        # key -> first failure of the current run of "storage cannot be reached" (D44)
        self._storage_trouble: dict[str, float] = {}

    # ------------------------------------------------------------------ lifecycle

    def start(self) -> asyncio.Task[None]:
        """Create and return the task running `run()` (named "job-<short id>")."""
        if self._task is None or self._task.done():
            self._task = asyncio.get_running_loop().create_task(
                self.run(), name=f"job-{self.job_id[:4]}"
            )
        return self._task

    @property
    def task(self) -> asyncio.Task[None] | None:
        return self._task

    async def run(self) -> None:
        """Main loop (module docstring). Returns when the job is terminal. Never raises
        except CancelledError (daemon shutdown: leave remote runs alone, invariant 11)."""
        try:
            while True:
                try:
                    job = self.deps.store.get_job(self.job_id)
                except JobNotFound:
                    return
                if is_terminal(job.state):
                    await self._sweep_orphans(wait=True)
                    await self._cleanup_storage(job)
                    return
                await self._sweep_orphans(wait=False)
                if self.recovery is not None:
                    emit(
                        "job.recover",
                        f"{job.short_id} {job.state}: {self.recovery}",
                        log=_logger,
                        job_id=job.id,
                        action=str(self.recovery),
                    )
                    self.recovery = None
                now = self.deps.clock.now()
                if (
                    job.state is JobState.QUEUED
                    and job.not_before is not None
                    and now < job.not_before
                    and not self._route_now
                ):
                    # woken early (a provider slot freed, a provider is healthy again):
                    # route now instead of sleeping out the rest of the wait (D44)
                    self._route_now = await self.sleep(job.not_before - now)
                    continue
                self._route_now = False
                step = self._step_for(job.state)
                held = self._live_attempt_id(job)
                try:
                    await step(job)
                    self._consecutive_errors = 0
                except StaleState:
                    continue
                except InvalidTransition as exc:
                    if self._state_moved(job):
                        continue  # a concurrent user action changed the job: like StaleState
                    await self._internal_error(job, exc)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    await self._internal_error(job, exc)
                finally:
                    if held is not None:
                        self._check_attempt_ended(held)
        finally:
            self._close_capture()

    def _live_attempt_id(self, job: Job) -> str | None:
        """The job's current attempt id when it is live (it holds a provider slot)."""
        if job.current_attempt_id is None or job.state not in _HOLDS_ATTEMPT:
            return None
        try:
            attempt = self.deps.store.get_attempt(job.current_attempt_id)
        except Exception:
            return None
        return attempt.id if attempt.state in LIVE_ATTEMPT_STATES else None

    def _check_attempt_ended(self, attempt_id: str) -> None:
        """A step ended the live attempt: its provider slot is free (capacity waits have no
        known end, so queued jobs are woken instead of sleeping out their backoff)."""
        if self._on_attempt_ended is None:
            return
        try:
            attempt = self.deps.store.get_attempt(attempt_id)
        except Exception:
            return
        if attempt.state not in LIVE_ATTEMPT_STATES:
            with contextlib.suppress(Exception):
                self._on_attempt_ended(self.job_id)

    def _state_moved(self, job: Job) -> bool:
        try:
            current = self.deps.store.get_job(job.id)
        except JobNotFound:
            return True
        return current.state is not job.state

    def wake(self) -> None:
        self._wake.set()

    @property
    def done(self) -> bool:
        return self._task is not None and self._task.done()

    def _step_for(self, state: JobState) -> Callable[[Job], Awaitable[None]]:
        steps: dict[JobState, Callable[[Job], Awaitable[None]]] = {
            JobState.QUEUED: self.step_queued,
            JobState.ROUTING: self.step_routing,
            JobState.AWAITING_APPROVAL: self.step_awaiting_approval,
            JobState.PROVISIONING: self.step_provisioning,
            JobState.RUNNING: self.step_running,
            JobState.CHECKPOINTING: self.step_running,
            JobState.MIGRATING: self.step_migrating,
            JobState.CANCELLING: self.step_cancelling,
        }
        return steps[state]

    # ------------------------------------------------------------------ small helpers

    @property
    def _now(self) -> float:
        return self.deps.clock.now()

    def _backoff(self, k: int) -> float:
        cfg = self.deps.config.engine
        return backoff_s(k, base_s=cfg.backoff_base_s, cap_s=cfg.backoff_cap_s)

    def _poll_interval(self, provider: str) -> float:
        try:
            return self.deps.registry.poll_interval_s(
                provider, self.deps.config.providers.get(provider)
            )
        except ProviderNotFound:
            return self.deps.config.engine.default_poll_interval_s

    async def _transient_pause(self) -> None:
        self._retry_k += 1
        await self.sleep(self._backoff(self._retry_k))

    def _max_attempts(self, job: Job) -> int:
        return job.spec.max_attempts or self.deps.config.engine.max_attempts

    def _progress_attempts(self, job: Job) -> int:
        """Attempts that saved a checkpoint beyond the one they resumed from (phase 5).
        The runner publishes only when the checkpoint dir changed after the restore, so a
        new seq means the job wrote new state: a multi-day job moved at every 12 h
        session cap is progressing, not retrying, and those attempts do not count
        against max_attempts (max_placements still bounds everything)."""
        if job.checkpoint_count == 0:
            return 0
        ckpts = self.deps.store.checkpoints_for(job.id)
        by_id = {c.id: c for c in ckpts}
        count = 0
        for att in self.deps.store.attempts_for(job.id):
            start = by_id.get(att.resume_checkpoint_id) if att.resume_checkpoint_id else None
            floor = start.seq if start is not None else 0
            if any(c.attempt_id == att.id and c.seq > floor for c in ckpts):
                count += 1
        return count

    def _budget_exhausted(self, job: Job) -> str | None:
        cfg = self.deps.config.engine
        max_attempts = self._max_attempts(job)
        progressed = self._progress_attempts(job) if job.accepted_attempts >= max_attempts else 0
        if job.accepted_attempts - progressed >= max_attempts:
            if progressed:
                return (
                    f"gave up after {job.accepted_attempts - progressed} attempts that made no "
                    f"progress ({job.accepted_attempts} in total, limit {max_attempts})"
                )
            return f"gave up after {job.accepted_attempts} attempts (limit {max_attempts})"
        if job.attempt_count >= cfg.max_placements:
            return f"gave up after {job.attempt_count} placements (limit {cfg.max_placements})"
        if job.waiting_since is not None and self._now - job.waiting_since >= cfg.max_queue_wait_s:
            return (
                f"gave up after waiting {fmt_duration(self._now - job.waiting_since)} "
                f"for a provider"
            )
        return None

    def _gave_up(self, job: Job, why: str) -> Job:
        return self.deps.store.transition(
            job.id,
            from_state=job.state,
            to_state=JobState.FAILED,
            reason=Reason.GAVE_UP,
            message=f"{why}; see `gpu status {job.short_id}` for each attempt",
            actor=ACTOR,
            patch=JobPatch(failure_kind=FailureKind.NO_PROVIDER),
        )

    def _react_provider(self, provider: str, exc: BaseException) -> str:
        """Apply the errors.py provider reaction; return a short phrase for messages."""
        store = self.deps.store
        cfg = self.deps.config.engine
        now = self._now
        if isinstance(exc, QuotaExhausted):
            until = exc.resets_at or now + cfg.unknown_quota_reset_s
            store.upsert_provider_state(provider, exhausted_until=until)
            return f"{provider} quota is used up (resets in {fmt_duration(until - now)})"
        if isinstance(exc, AuthRequired):
            store.upsert_provider_state(
                provider,
                health=ProviderHealth.AUTH_REQUIRED,
                health_reason=_err_text(exc),
                last_healthcheck_at=now,
            )
            return f"{provider} needs login (run `gpu login {provider}`)"
        if isinstance(exc, InvalidJob):
            return f"{provider} cannot run this job ({_err_text(exc)})"
        if isinstance(exc, RateLimited | AdapterError | AdapterContractViolation):
            state = store.get_provider_state(provider)
            k = state.consecutive_failures + 1
            retry_after = exc.retry_after if isinstance(exc, RateLimited) else None
            until = cooldown_until(
                now,
                k=k,
                base_s=cfg.backoff_base_s,
                cap_s=cfg.backoff_cap_s,
                retry_after=retry_after,
            )
            store.upsert_provider_state(provider, consecutive_failures=k, cooldown_until=until)
            what = "rate limited us" if isinstance(exc, RateLimited) else "is unavailable"
            return f"{provider} {what} (cooling down {fmt_duration(until - now)})"
        return f"{provider}: {_err_text(exc)}"

    def _unreachable_for(self, attempt: Attempt) -> float:
        """Record one failed status/lookup call; return how long this process has seen the
        provider fail continuously. A gap longer than the longest sleep between two checks
        (daemon down, laptop asleep) restarts the clock."""
        now = self._now
        cfg = self.deps.config.engine
        gap_limit = 2 * max(self._poll_interval(attempt.provider), cfg.backoff_cap_s) + 60
        first, last = self._unreachable.get(attempt.id, (now, now))
        if now - last > gap_limit:
            first = now
        self._unreachable[attempt.id] = (first, now)
        return now - first

    def _reachable(self, attempt: Attempt) -> None:
        self._unreachable.pop(attempt.id, None)

    def _provider_ok(self, provider: str) -> None:
        state = self.deps.store.get_provider_state(provider)
        if state.consecutive_failures:
            self.deps.store.upsert_provider_state(provider, consecutive_failures=0)

    # ------------------------------------------------------------------ steps (one per state)

    async def step_queued(self, job: Job) -> None:
        """Budget checks (gave_up), else queued -> routing (routing_started)."""
        why = self._budget_exhausted(job)
        if why is not None:
            self._gave_up(job, why)
            return
        patch = JobPatch(not_before=None)
        if job.waiting_since is None:
            patch = JobPatch(not_before=None, waiting_since=self._now)
        self.deps.store.transition(
            job.id,
            from_state=JobState.QUEUED,
            to_state=JobState.ROUTING,
            reason=Reason.ROUTING_STARTED,
            message="choosing a provider",
            actor=ACTOR,
            patch=patch,
        )

    async def step_routing(self, job: Job) -> None:
        """Build RoutingContext, call router; PLACE -> policy -> awaiting_approval or
        store.place (-> provisioning); WAIT -> queued with not_before (no_capacity);
        NO_FIT -> failed (no_provider_fits, failure_kind=no_provider)."""
        decision = self.deps.router.route(build_routing_context(self.deps, job))
        await self._apply_decision(job, decision, ask_policy=True)

    async def step_awaiting_approval(self, job: Job) -> None:
        """approved_at set -> re-route and place (D10); the policy is asked again but only
        a decision marked `always` makes the job wait for a new answer (D43: the quota rule
        when the re-route landed on another provider). approval_timeout_s since the latest
        request -> denied (approval_expired). Else sleep until woken or timeout."""
        if job.approved_at is not None:
            decision = self.deps.router.route(build_routing_context(self.deps, job))
            await self._apply_decision(job, decision, ask_policy=True, after_approval=True)
            return
        entered = self._approval_asked_at(job)
        deadline = entered + self.deps.config.engine.approval_timeout_s
        now = self._now
        if now >= deadline:
            self.deps.store.transition(
                job.id,
                from_state=JobState.AWAITING_APPROVAL,
                to_state=JobState.DENIED,
                reason=Reason.APPROVAL_EXPIRED,
                message=(
                    f"approval request expired after "
                    f"{fmt_duration(self.deps.config.engine.approval_timeout_s)}; "
                    f"nothing ran. resubmit to try again"
                ),
                actor=ACTOR,
            )
            return
        await self.sleep(deadline - now)

    async def step_provisioning(self, job: Job) -> None:
        """Attempt `submitting` without remote_id -> submit (or resolve by key when recovering
        or after an ambiguous error). `submitted` -> poll; pending longer than
        provision_timeout_s -> cancel + queued (provision_timeout)."""
        attempt = self.deps.store.current_attempt(job)
        if attempt is None or attempt.state not in LIVE_ATTEMPT_STATES:
            # Cannot happen through the store's composite writes; recover rather than wedge.
            self.deps.store.transition(
                job.id,
                from_state=JobState.PROVISIONING,
                to_state=JobState.QUEUED,
                reason=Reason.PROVIDER_UNAVAILABLE,
                message="lost track of the placement; choosing a provider again",
                actor=ACTOR,
                patch=JobPatch(current_attempt_id=None),
            )
            return
        if attempt.state is AttemptState.SUBMITTING and attempt.remote_id is None:
            if attempt.id in self._fresh_attempts:
                self._fresh_attempts.discard(attempt.id)
                await self.submit_attempt(job, attempt)
            else:
                await self.resolve_ambiguous(job, attempt)
            return
        await self._poll(job, attempt)

    async def step_running(self, job: Job) -> None:
        """Poll status + capture logs; apply checkpoint/progress results; map remote phase
        to transitions (completed / script_failed / session_lost / remote_cancelled ...).
        Also used for `checkpointing` (checkpoint_end / checkpoint_stalled)."""
        attempt = self.deps.store.current_attempt(job)
        if attempt is None or attempt.remote_id is None:
            self.deps.store.transition(
                job.id,
                from_state=job.state,
                to_state=JobState.MIGRATING,
                reason=Reason.STATUS_LOST,
                message="lost track of the remote run; placing the job again",
                actor=ACTOR,
            )
            return
        if attempt.state not in LIVE_ATTEMPT_STATES:
            # The attempt ended but the job did not follow (e.g. a cancel_unconfirmed
            # abandon raced a restart): migrate so the job is placed again.
            self.deps.store.transition(
                job.id,
                from_state=job.state,
                to_state=JobState.MIGRATING,
                reason=Reason.STATUS_LOST,
                message=f"attempt {attempt.n} already ended ({attempt.state}); "
                f"placing the job again",
                actor=ACTOR,
            )
            return
        if await self._enforce_hours(job, attempt):
            return
        if await self._handoff(job, attempt):
            return
        await self._poll(job, attempt)

    async def step_migrating(self, job: Job) -> None:
        """Ensure the previous attempt is terminal (cancel + confirm if needed), then route
        again resuming from store.latest_checkpoint(job.id)."""
        attempt = self.deps.store.current_attempt(job)
        if attempt is not None and attempt.state in LIVE_ATTEMPT_STATES:
            if attempt.remote_id is None:
                ref = await self._resolve_for_migration(job, attempt)
                if ref is None:
                    return
                attempt = self.deps.store.record_submission(
                    attempt.id, remote_id=ref.remote_id, remote_url=ref.url, remote_meta=ref.meta
                )
            try:
                await self.deps.caller.cancel(attempt.provider, remote_ref(attempt))
            except NotFound:
                pass
            except ADAPTER_FAILURES as exc:
                emit(
                    "attempt.cancel",
                    f"{attempt.id}: cancel before migration failed: {_err_text(exc)}",
                    level=logging.WARNING,
                    log=_logger,
                    job_id=job.id,
                    attempt_id=attempt.id,
                )
                await self._transient_pause()
                return
            self.deps.store.update_attempt(
                attempt.id,
                AttemptPatch(state=AttemptState.CANCELLED, lost_reason="stopped for a migration"),
            )
            return
        if not await self._reconcile_checkpoints(job):
            return
        job = self.deps.store.get_job(job.id)
        why = self._budget_exhausted(job)
        if why is not None:
            self._gave_up(job, why)
            return
        decision = self.deps.router.route(build_routing_context(self.deps, job))
        await self._apply_decision(job, decision, ask_policy=True)

    async def _resolve_for_migration(self, job: Job, attempt: Attempt) -> RemoteRef | None:
        """A live attempt without remote_id while migrating: find out whether it started
        (so it can be stopped) or end it. Returns the run to stop, or None when this step
        is over (attempt ended or retry later)."""
        store = self.deps.store
        provider = attempt.provider
        orphan = self.deps.caller.orphan_submit(attempt.attempt_key)
        if orphan is not None:
            if not orphan.done:
                await self.sleep(self._poll_interval(provider))
                return None
            self.deps.caller.forget_orphan_submit(attempt.attempt_key)
            if orphan.ref is not None:
                return orphan.ref
            if isinstance(orphan.error, DEFINITIVE_SUBMIT_ERRORS):
                store.update_attempt(
                    attempt.id,
                    AttemptPatch(
                        state=AttemptState.REJECTED,
                        error_kind=type(orphan.error).__name__,
                        error_message=_err_text(orphan.error),
                    ),
                )
                return None
        try:
            ref = await self._lookup(attempt)
        except _NoLookup:
            store.update_attempt(
                attempt.id,
                AttemptPatch(state=AttemptState.ABANDONED, lost_reason="superseded by a migration"),
            )
            return None
        except ADAPTER_FAILURES as exc:
            if self._unreachable_for(attempt) >= self.deps.config.engine.unreachable_lost_after_s:
                self._reachable(attempt)
                self._react_provider(provider, exc)
                store.update_attempt(
                    attempt.id,
                    AttemptPatch(
                        state=AttemptState.ABANDONED,
                        error_kind=UNREACHABLE_ERROR_KIND,
                        error_message=_err_text(exc),
                        lost_reason="superseded by a migration (provider unreachable)",
                    ),
                )
                return None
            await self._transient_pause()
            return None
        self._reachable(attempt)
        if ref is None:
            store.update_attempt(
                attempt.id,
                AttemptPatch(state=AttemptState.CANCELLED, lost_reason="superseded by a migration"),
            )
        return ref

    async def _sweep_orphans(self, *, wait: bool) -> None:
        """Stop remote runs created by timed-out submits whose attempt has already ended
        (the job moved on). With `wait`, block until such submits return (job terminal)."""
        caller = self.deps.caller
        keys = caller.orphan_keys()
        if not keys:
            return
        mine = {a.attempt_key: a for a in self.deps.store.attempts_for(self.job_id)}
        for key in keys:
            attempt = mine.get(key)
            if attempt is None or attempt.state in LIVE_ATTEMPT_STATES:
                continue
            orphan = await caller.wait_orphan_submit(key) if wait else caller.orphan_submit(key)
            if orphan is None or not orphan.done:
                continue
            caller.forget_orphan_submit(key)
            if orphan.ref is None or orphan.ref.remote_id == attempt.remote_id:
                continue
            stopped = True
            try:
                await caller.cancel(orphan.provider, orphan.ref)
            except NotFound:
                pass
            except ADAPTER_FAILURES:
                stopped = False
            where = f" ({orphan.ref.url})" if orphan.ref.url else ""
            self.deps.store.add_note(
                self.job_id,
                reason=Reason.ORPHAN_CANCELLED,
                actor=ACTOR,
                attempt_id=attempt.id,
                message=(
                    f"a late submit of attempt {attempt.n} started {orphan.ref.remote_id} on "
                    f"{orphan.provider}{where}; stopped it"
                    if stopped
                    else f"a late submit of attempt {attempt.n} started "
                    f"{orphan.ref.remote_id} on {orphan.provider}{where} and it could not be "
                    f"stopped; stop it in the provider's console"
                ),
                detail={"remote_id": orphan.ref.remote_id, "stopped": stopped},
            )

    async def step_cancelling(self, job: Job) -> None:
        """adapter.cancel (idempotent), confirm via status, -> cancelled; after
        cancel_timeout_s -> cancelled (cancel_unconfirmed). Remote already succeeded ->
        fetch outputs, note outputs_kept, -> cancelled."""
        store = self.deps.store
        attempt = store.current_attempt(job)
        if attempt is None or attempt.state not in LIVE_ATTEMPT_STATES:
            self._finish_cancel(job, None, None, "cancelled; nothing was running")
            return
        provider = attempt.provider
        if attempt.remote_id is None:
            ref: RemoteRef | None = None
            orphan = self.deps.caller.orphan_submit(attempt.attempt_key)
            if orphan is not None and not orphan.done:
                if self._cancel_timed_out(job):
                    # the orphan sweep stops the run if the submit still creates one
                    self._finish_cancel(
                        job,
                        attempt,
                        AttemptPatch(
                            state=AttemptState.ABANDONED,
                            lost_reason="cancelled while the submit was still in progress",
                        ),
                        f"cancelled; the submit to {provider} was still in progress, "
                        f"gpu-router stops the run if it starts",
                        reason=Reason.CANCEL_UNCONFIRMED,
                    )
                    return
                await self.sleep(self._poll_interval(provider))
                return
            if orphan is not None:
                self.deps.caller.forget_orphan_submit(attempt.attempt_key)
            if attempt.id in self._fresh_attempts:
                self._fresh_attempts.discard(attempt.id)
            elif orphan is not None and orphan.ref is not None:
                ref = orphan.ref
            elif orphan is not None and isinstance(orphan.error, DEFINITIVE_SUBMIT_ERRORS):
                ref = None  # the provider refused it: nothing started
            else:
                try:
                    ref = await self._lookup(attempt)
                except _NoLookup:
                    self._finish_cancel(
                        job,
                        attempt,
                        AttemptPatch(
                            state=AttemptState.ABANDONED,
                            lost_reason="cancelled while the submit outcome was unknown",
                        ),
                        f"cancelled; {provider} may still have started attempt "
                        f"{attempt.n} (it cannot be looked up), check its console",
                    )
                    return
                except ADAPTER_FAILURES:
                    if self._cancel_timed_out(job):
                        self._finish_cancel(
                            job,
                            attempt,
                            AttemptPatch(
                                state=AttemptState.ABANDONED, lost_reason="cancel not confirmed"
                            ),
                            f"cancelled, but {provider} never confirmed; check its console",
                            reason=Reason.CANCEL_UNCONFIRMED,
                        )
                        return
                    await self._transient_pause()
                    return
            if ref is None:
                self._finish_cancel(
                    job,
                    attempt,
                    AttemptPatch(state=AttemptState.CANCELLED),
                    "cancelled before it started",
                )
                return
            attempt = store.record_submission(
                attempt.id, remote_id=ref.remote_id, remote_url=ref.url, remote_meta=ref.meta
            )
        ref = remote_ref(attempt)
        try:
            await self.deps.caller.cancel(provider, ref)
        except NotFound:
            self._finish_cancel(
                job,
                attempt,
                AttemptPatch(state=AttemptState.CANCELLED),
                f"cancelled; {provider} no longer had the run",
            )
            return
        except ADAPTER_FAILURES as exc:
            if self._cancel_timed_out(job):
                self._finish_cancel(
                    job,
                    attempt,
                    AttemptPatch(state=AttemptState.ABANDONED, lost_reason="cancel not confirmed"),
                    f"cancelled, but {provider} never confirmed ({_err_text(exc)}); "
                    f"check its console",
                    reason=Reason.CANCEL_UNCONFIRMED,
                )
                return
            await self._transient_pause()
            return
        crashpoints.crashpoint("during_cancel")
        try:
            st = await self.deps.caller.status(provider, ref)
        except NotFound:
            self._finish_cancel(
                job, attempt, AttemptPatch(state=AttemptState.CANCELLED), "cancelled"
            )
            return
        except ADAPTER_FAILURES:
            st = None
        caps_confirm = (
            provider in self.deps.registry
            and self.deps.registry.get(provider).capabilities.cancel_confirms
        )
        if st is not None and st.phase is RemotePhase.SUCCEEDED:
            # outputs_fetched: the success path already fetched them before the cancel landed
            fetched = job.outputs_fetched or await self.fetch_outputs(job, attempt)
            store.add_note(
                job.id,
                reason=Reason.OUTPUTS_KEPT,
                actor=ACTOR,
                attempt_id=attempt.id,
                message=(f"the run had already finished; outputs kept in {job.outputs_dir}")
                if fetched
                else "the run had already finished before the cancel",
            )
            self._finish_cancel(
                job,
                attempt,
                AttemptPatch(state=AttemptState.SUCCEEDED, exit_code=st.exit_code or 0),
                "cancelled after the run had already finished",
            )
            return
        if st is not None and st.phase.terminal:
            patch = AttemptPatch(
                state={
                    RemotePhase.FAILED: AttemptState.FAILED,
                    RemotePhase.LOST: AttemptState.LOST,
                }.get(st.phase, AttemptState.CANCELLED),
                exit_code=st.exit_code,
                lost_reason=st.lost_reason,
            )
            self._finish_cancel(job, attempt, patch, f"cancelled; stopped on {provider}")
            return
        if st is not None and not caps_confirm:
            self._finish_cancel(
                job,
                attempt,
                AttemptPatch(state=AttemptState.CANCELLED),
                f"cancelled; {provider} accepted the stop request",
            )
            return
        if self._cancel_timed_out(job):
            self._finish_cancel(
                job,
                attempt,
                AttemptPatch(state=AttemptState.ABANDONED, lost_reason="cancel not confirmed"),
                f"cancelled, but {provider} never confirmed the run stopped; check its console",
                reason=Reason.CANCEL_UNCONFIRMED,
            )
            return
        await self.sleep(self._poll_interval(provider))

    # ------------------------------------------------------------------ routing helpers

    def _approval_asked_at(self, job: Job) -> float:
        """When the current approval request was made: the latest approval_required event
        (the transition into awaiting_approval, or a re-ask note after an approval, D43)."""
        events = self.deps.store.events_for(job.id, limit=10_000)
        for ev in reversed(events):
            if ev.reason == Reason.APPROVAL_REQUIRED:
                return ev.ts
            if ev.kind == "transition" and ev.to_state is job.state:
                return ev.ts
        return job.updated_at

    def _entered_state_at(self, job: Job) -> float:
        """Timestamp of the transition that put the job in its current state."""
        events = self.deps.store.events_for(job.id, limit=10_000)
        for ev in reversed(events):
            if ev.kind == "transition" and ev.to_state is job.state:
                return ev.ts
        return job.updated_at

    def _ask_approval(
        self, job: Job, decision: RouteDecision, cand: Candidate, approval: ApprovalDecision
    ) -> None:
        """-> awaiting_approval. A job already waiting (a re-ask after an approval whose
        re-route landed on another provider, D43) cannot transition to the same state, so
        its approval fields are reset in place with an approval_required note instead."""
        store = self.deps.store
        reason = approval.reason or "approval required"
        # a fresh request needs a fresh answer: an older approval (from a previous
        # placement, or for another provider) must not auto-approve this one
        patch = JobPatch(
            approval_reason=reason,
            approved_at=None,
            approved_by=None,
            provider=cand.provider,
            gpu=cand.gpu,
            route_reason=cand.reason,
        )
        detail = {**decision.detail(), "rule": approval.rule}
        answer = f"run `gpu approve {job.short_id}` or `gpu deny {job.short_id}`"
        if job.state is JobState.AWAITING_APPROVAL:
            was = f" for {job.provider}" if job.provider and job.provider != cand.provider else ""
            message = (
                f"the approval{was} does not cover this placement; waiting for your approval "
                f"again ({reason}); {answer}"
            )
            store.add_note(
                job.id,
                reason=Reason.APPROVAL_REQUIRED,
                actor=ACTOR,
                message=message,
                detail={**detail, "reask": True},
                patch=patch.model_copy(update={"message": message}),
            )
            return
        store.transition(
            job.id,
            from_state=job.state,
            to_state=JobState.AWAITING_APPROVAL,
            reason=Reason.APPROVAL_REQUIRED,
            actor=ACTOR,
            message=f"waiting for your approval ({reason}); {answer}",
            detail=detail,
            patch=patch,
        )

    async def _apply_decision(
        self,
        job: Job,
        decision: RouteDecision,
        *,
        ask_policy: bool,
        after_approval: bool = False,
    ) -> None:
        """PLACE (after the policy), WAIT or NO_FIT. `after_approval`: the job was just
        approved, so only a policy decision marked `always` asks again (D43)."""
        store = self.deps.store
        if decision.outcome is RouteOutcome.PLACE and decision.chosen is not None:
            self._wait_k = 0
            cand = decision.chosen
            if ask_policy:
                overrun = self._overrun_pending(job)
                if overrun is not None:  # stopped past its declared hours (D48)
                    self._ask_approval(job, decision, cand, overrun)
                    return
                approval = self.deps.policy.evaluate(job, decision, cand)
                if approval.required and (approval.always or not after_approval):
                    self._ask_approval(job, decision, cand, approval)
                    return
            latest = store.latest_checkpoint(job.id)
            resume = f"; resuming from checkpoint {latest.seq}" if latest else ""
            gpu = f" {cand.gpu}" if cand.gpu else ""
            _, attempt = store.place(
                job.id,
                from_state=job.state,
                provider=cand.provider,
                gpu=cand.gpu,
                route_reason=cand.reason,
                detail=decision.detail(),
                message=f"placed on {cand.provider}{gpu} ({cand.reason}){resume}; submitting",
                resume_checkpoint_id=latest.id if latest else None,
                actor=ACTOR,
            )
            self._fresh_attempts.add(attempt.id)
            return
        if decision.outcome is RouteOutcome.WAIT:
            self._wait_k += 1
            now = self._now
            waits = [r for r in decision.rejected if r.code in _WAIT_CODES]
            backoff_at = now + max(self._backoff(self._wait_k), 0.001)
            retry_at = decision.retry_at
            if retry_at is None or retry_at <= now:
                retry_at = backoff_at
            elif any(r.until is None for r in waits):
                # a busy provider (or a login) may clear any minute: never sleep until a
                # far-off quota reset because another provider happens to have one (D44)
                retry_at = min(retry_at, backoff_at)
            since = job.waiting_since or now
            if (
                decision.retry_at is not None
                and waits
                and all(r.code in _RESET_CODES and r.until is not None for r in waits)
            ):
                # only quota resets hold it back (D41: the job waits for the reset): that
                # wait is known and bounded, so max_queue_wait_s counts from the reset (D44)
                since = max(since, retry_at)
            store.transition(
                job.id,
                from_state=job.state,
                to_state=JobState.QUEUED,
                reason=Reason.NO_CAPACITY,
                actor=ACTOR,
                detail=decision.detail(),
                message=f"{decision.reason}; retrying in {fmt_duration(retry_at - now)}",
                patch=JobPatch(not_before=retry_at, waiting_since=since),
            )
            return
        failure = FailureKind.NO_PROVIDER
        # RESERVED (the Mac kept for smoke tests) and OVERRIDE never explain the failure
        core = [r for r in decision.rejected if r.code not in ("override", "reserved")]
        if core and all(r.code == "excluded" for r in core):
            invalid = [a for a in store.attempts_for(job.id) if a.error_kind == "InvalidJob"]
            if invalid:
                failure = FailureKind.INVALID_JOB
        store.transition(
            job.id,
            from_state=job.state,
            to_state=JobState.FAILED,
            reason=Reason.NO_PROVIDER_FITS,
            actor=ACTOR,
            detail=decision.detail(),
            message=f"{decision.reason}; change the job's requirements and resubmit",
            patch=JobPatch(failure_kind=failure),
        )

    # ------------------------------------------------------------------ submit

    async def submit_attempt(self, job: Job, attempt: Attempt) -> None:
        """Resolve secrets, build AttemptContext, crashpoint("after_place_commit"), call
        submit, crashpoint("after_submit_return"), record_submission. Error handling per the
        errors.py table and invariant 6."""
        store = self.deps.store
        provider = attempt.provider
        trouble = f"storage:{attempt.id}"
        degrade = self._storage_waited(trouble) >= self.deps.config.checkpoint.storage_wait_s
        try:
            extras = await self._storage_extras(job, attempt, degrade=degrade)
        except _DataProblem as exc:
            if exc.reroute:
                self._submit_rejected(
                    job,
                    attempt,
                    exc.error or InvalidJob(exc.message, provider=provider, hint=exc.hint),
                )
                return
            store.transition(
                job.id,
                from_state=JobState.PROVISIONING,
                to_state=JobState.FAILED,
                reason=Reason.PROVIDER_PERMANENT,
                actor=ACTOR,
                message=f"{exc.message}; fix the job's data and resubmit",
                patch=JobPatch(failure_kind=FailureKind.USER_ERROR),
                attempt=AttemptChange(
                    attempt.id,
                    AttemptPatch(
                        state=AttemptState.REJECTED,
                        error_kind="DataUnavailable",
                        error_message=exc.message,
                    ),
                ),
            )
            return
        except StorageError as exc:
            # only retryable trouble gets here (the hub turns refusals into "no storage"),
            # and only until checkpoint.storage_wait_s: then the attempt goes ahead without
            self._fresh_attempts.add(attempt.id)  # nothing was sent: still safe to submit
            if trouble not in self._storage_trouble:
                self._storage_trouble[trouble] = self._now
                wait = fmt_duration(self.deps.config.checkpoint.storage_wait_s)
                store.add_note(
                    job.id,
                    reason=Reason.RETRY_SCHEDULED,
                    actor=ACTOR,
                    attempt_id=attempt.id,
                    message=(
                        f"{exc.message}; retrying for up to {wait} before trying another provider"
                        if isinstance(exc, _StagePending)
                        else f"checkpoint storage cannot be reached before submitting "
                        f"({exc.message}); waiting up to {wait} for it before the attempt "
                        f"goes ahead without it"
                    ),
                    detail={"error": "StorageError"},
                )
            await self._transient_pause()
            return
        self._storage_trouble.pop(trouble, None)
        try:
            ctx = self._attempt_context(job, attempt, extras)
        except _MissingSecret as exc:
            reserved = isinstance(exc, _ReservedSecret)
            store.transition(
                job.id,
                from_state=JobState.PROVISIONING,
                to_state=JobState.FAILED,
                reason=Reason.PROVIDER_PERMANENT,
                actor=ACTOR,
                message=(
                    f"secret {exc.name!r} is one of gpu-router's own credentials and is "
                    "never sent to a job; store the job's own token under another name "
                    "and resubmit"
                    if reserved
                    else f"secret {exc.name!r} is not set; store it with "
                    f"`gpu secrets set {exc.name}` and resubmit"
                ),
                patch=JobPatch(failure_kind=FailureKind.USER_ERROR),
                attempt=AttemptChange(
                    attempt.id,
                    AttemptPatch(
                        state=AttemptState.REJECTED,
                        error_kind="ReservedSecret" if reserved else "MissingSecret",
                        error_message=f"secret {exc.name!r} is "
                        + ("reserved" if reserved else "not set"),
                    ),
                ),
            )
            return
        except SecretsError as exc:
            self._fresh_attempts.add(attempt.id)  # nothing was sent: still safe to submit
            store.add_note(
                job.id,
                reason=Reason.RETRY_SCHEDULED,
                actor=ACTOR,
                attempt_id=attempt.id,
                message=f"could not read secrets ({_err_text(exc)}); retrying",
            )
            await self._transient_pause()
            return

        crashpoints.crashpoint("after_place_commit")
        try:
            ref = await self.deps.caller.submit(provider, job, ctx)
        except DEFINITIVE_SUBMIT_ERRORS as exc:
            self._submit_rejected(job, attempt, exc)
            return
        except ADAPTER_FAILURES as exc:
            # Ambiguous (invariant 6): the request may have reached the provider.
            phrase = self._react_provider(provider, exc)
            store.update_attempt(
                attempt.id,
                AttemptPatch(
                    error_kind=f"{AMBIGUOUS_PREFIX}{type(exc).__name__}",
                    error_message=_err_text(exc),
                ),
            )
            store.add_note(
                job.id,
                reason=Reason.SUBMIT_AMBIGUOUS,
                actor=ACTOR,
                attempt_id=attempt.id,
                message=(
                    f"submit to {provider} did not complete ({phrase}); "
                    f"checking whether it started before trying elsewhere"
                ),
                detail={"error": type(exc).__name__},
            )
            await self._transient_pause()
            return
        crashpoints.crashpoint("after_submit_return")
        store.record_submission(
            attempt.id, remote_id=ref.remote_id, remote_url=ref.url, remote_meta=ref.meta
        )
        self._provider_ok(provider)
        self._retry_k = 0
        emit(
            "attempt.submit",
            f"{attempt.id} submitted to {provider} as {ref.remote_id}",
            log=_logger,
            job_id=job.id,
            attempt_id=attempt.id,
            provider=provider,
        )

    def _attempt_context(
        self, job: Job, attempt: Attempt, extras: _Extras | None = None
    ) -> AttemptContext:
        from gpu_router import secrets as secret_store
        from gpu_router.models import is_reserved_secret

        resolved: dict[str, SecretStr] = {}
        for name in job.spec.secrets:
            if is_reserved_secret(name):  # gpu-router's own credentials never ship (D48)
                raise _ReservedSecret(name)
            value = secret_store.get_secret(name)
            if value is None:
                raise _MissingSecret(name)
            resolved[name] = SecretStr(value)
        resume: Checkpoint | None = None
        if attempt.resume_checkpoint_id is not None:
            for ckpt in self.deps.store.checkpoints_for(job.id):
                if ckpt.id == attempt.resume_checkpoint_id:
                    resume = ckpt
                    break
        env = dict(job.spec.env)
        env.update(
            {
                "GPU_ROUTER_JOB_ID": job.id,
                "GPU_ROUTER_ATTEMPT": str(attempt.n),
                "GPU_ROUTER_PROTOCOL": "1",
            }
        )
        if extras is not None:
            env.update(extras.env)
            for secret_name, secret in extras.secrets.items():
                resolved.setdefault(secret_name, secret)
        bundle_dir = self.deps.paths.job_bundle_dir(job.id)
        bundle_archive = self.deps.paths.job_bundle_archive(job.id)
        if (
            self.deps.bundler is not None
            and job.bundle_sha256 is not None
            and not (bundle_dir / "manifest.json").is_file()
        ):
            # Crash between create_job and materialize (Supervisor.submit): heal from cache.
            self.deps.bundler.materialize(job.bundle_sha256, job.id)
        return AttemptContext(
            attempt_id=attempt.id,
            attempt_key=attempt.attempt_key,
            n=attempt.n,
            bundle_dir=bundle_dir if bundle_dir.exists() else None,
            bundle_archive=bundle_archive if bundle_archive.exists() else None,
            resume_from=resume,
            env=env,
            secrets=resolved,
            gpu=attempt.gpu,
            checkpoint_interval_min=job.spec.checkpoint_interval_min,
        )

    def _submit_rejected(self, job: Job, attempt: Attempt, exc: AdapterError) -> None:
        store = self.deps.store
        provider = attempt.provider
        att_patch = AttemptPatch(
            state=AttemptState.REJECTED, error_kind=type(exc).__name__, error_message=_err_text(exc)
        )
        if isinstance(exc, Permanent):
            store.transition(
                job.id,
                from_state=JobState.PROVISIONING,
                to_state=JobState.FAILED,
                reason=Reason.PROVIDER_PERMANENT,
                actor=ACTOR,
                message=f"{provider} refused the job permanently: {_err_text(exc)}",
                patch=JobPatch(failure_kind=FailureKind.PROVIDER_ERROR),
                attempt=AttemptChange(attempt.id, att_patch),
            )
            return
        phrase = self._react_provider(provider, exc)
        reason = Reason.PROVIDER_UNAVAILABLE
        for klass, r in (
            (RateLimited, Reason.RATE_LIMITED),
            (QuotaExhausted, Reason.QUOTA_EXHAUSTED),
            (AuthRequired, Reason.AUTH_REQUIRED),
            (InvalidJob, Reason.INVALID_FOR_PROVIDER),
        ):
            if isinstance(exc, klass):
                reason = r
                break
        if isinstance(exc, InvalidJob):
            store.add_note(
                job.id,
                reason=Reason.PROVIDER_EXCLUDED,
                actor=ACTOR,
                attempt_id=attempt.id,
                message=f"not using {provider} again for this job: {_err_text(exc)}",
            )
        store.transition(
            job.id,
            from_state=JobState.PROVISIONING,
            to_state=JobState.QUEUED,
            reason=reason,
            actor=ACTOR,
            detail={"error": type(exc).__name__},
            message=f"{phrase}; choosing another provider",
            patch=JobPatch(current_attempt_id=None, not_before=None),
            attempt=AttemptChange(attempt.id, att_patch),
        )

    async def _lookup(self, attempt: Attempt) -> RemoteRef | None:
        provider = attempt.provider
        if (
            provider not in self.deps.registry
            or not self.deps.registry.get(provider).capabilities.lookup_by_key
        ):
            raise _NoLookup
        return await self.deps.caller.lookup_by_key(provider, attempt.attempt_key)

    async def resolve_ambiguous(self, job: Job, attempt: Attempt) -> None:
        """lookup_by_key until found (record_submission) or NotFound (attempt rejected,
        job -> queued). Adapter without lookup_by_key -> attempt abandoned, provider
        excluded for this job, note attempt_abandoned with a warning for the user.

        A `submitting` attempt with no ambiguity marker is a crash-recovery case (the daemon
        died between committing the attempt and hearing back from submit): when the lookup
        proves nothing exists, it is submitted now under the same attempt key."""
        store = self.deps.store
        provider = attempt.provider
        cfg = self.deps.config.engine
        orphan = self.deps.caller.orphan_submit(attempt.attempt_key)
        if orphan is not None:
            if not orphan.done:
                # The timed-out submit is still running in its worker thread: a lookup now
                # could miss the run it is about to create (invariant 6). Wait for it.
                await self.sleep(self._poll_interval(provider))
                return
            self.deps.caller.forget_orphan_submit(attempt.attempt_key)
            if orphan.ref is not None:
                store.record_submission(
                    attempt.id,
                    remote_id=orphan.ref.remote_id,
                    remote_url=orphan.ref.url,
                    remote_meta=orphan.ref.meta,
                )
                self._provider_ok(provider)
                self._retry_k = 0
                return
            if isinstance(orphan.error, DEFINITIVE_SUBMIT_ERRORS):
                self._submit_rejected(job, attempt, orphan.error)
                return
            # still ambiguous: the lookup below is authoritative now the worker is done
        try:
            ref = await self._lookup(attempt)
        except _NoLookup:
            store.add_note(
                job.id,
                reason=Reason.ATTEMPT_ABANDONED,
                actor=ACTOR,
                attempt_id=attempt.id,
                message=(
                    f"could not tell whether {provider} started attempt "
                    f"{attempt.n}; it may still be running there (check its "
                    f"console). not using {provider} again for this job"
                ),
            )
            store.transition(
                job.id,
                from_state=JobState.PROVISIONING,
                to_state=JobState.QUEUED,
                reason=Reason.PROVIDER_UNAVAILABLE,
                actor=ACTOR,
                message=f"abandoned the uncertain submit to {provider}; choosing another provider",
                patch=JobPatch(current_attempt_id=None),
                attempt=AttemptChange(
                    attempt.id,
                    AttemptPatch(
                        state=AttemptState.ABANDONED,
                        lost_reason="submit outcome unknown and cannot be looked up",
                    ),
                ),
            )
            return
        except ADAPTER_FAILURES as exc:
            unreachable = self._unreachable_for(attempt)
            if unreachable >= cfg.unreachable_lost_after_s:
                # An outage, not a provider that cannot run the job: the abandon carries
                # UNREACHABLE_ERROR_KIND so excluded_providers does not ban the provider for
                # this job (invariant 8); it is cooled down and routing waits for it.
                self._reachable(attempt)
                phrase = self._react_provider(provider, exc)
                store.add_note(
                    job.id,
                    reason=Reason.ATTEMPT_ABANDONED,
                    actor=ACTOR,
                    attempt_id=attempt.id,
                    message=(
                        f"could not reach {provider} for {fmt_duration(unreachable)} to "
                        f"confirm attempt {attempt.n}; it may have started there (check its "
                        f"console)"
                    ),
                )
                store.transition(
                    job.id,
                    from_state=JobState.PROVISIONING,
                    to_state=JobState.QUEUED,
                    reason=Reason.PROVIDER_UNAVAILABLE,
                    actor=ACTOR,
                    message=(
                        f"gave up confirming attempt {attempt.n} on {provider} ({phrase}); "
                        f"choosing a provider again"
                    ),
                    patch=JobPatch(current_attempt_id=None),
                    attempt=AttemptChange(
                        attempt.id,
                        AttemptPatch(
                            state=AttemptState.ABANDONED,
                            error_kind=UNREACHABLE_ERROR_KIND,
                            error_message=_err_text(exc),
                            lost_reason="submit outcome unknown (provider unreachable)",
                        ),
                    ),
                )
                return
            await self._transient_pause()
            return
        self._reachable(attempt)
        if ref is not None:
            store.record_submission(
                attempt.id, remote_id=ref.remote_id, remote_url=ref.url, remote_meta=ref.meta
            )
            self._retry_k = 0
            return
        ambiguous = (attempt.error_kind or "").startswith(AMBIGUOUS_PREFIX)
        if not ambiguous:
            await self.submit_attempt(job, attempt)
            return
        store.transition(
            job.id,
            from_state=JobState.PROVISIONING,
            to_state=JobState.QUEUED,
            reason=Reason.PROVIDER_UNAVAILABLE,
            actor=ACTOR,
            message=f"{provider} never started attempt {attempt.n}; choosing a provider again",
            patch=JobPatch(current_attempt_id=None),
            attempt=AttemptChange(
                attempt.id,
                AttemptPatch(
                    state=AttemptState.REJECTED,
                    error_kind=(attempt.error_kind or "")[len(AMBIGUOUS_PREFIX) :] or "Unavailable",
                ),
            ),
        )

    # ------------------------------------------------------------------ polling

    def _register_secrets(self, job: Job) -> None:
        """Register the job's secret values for redaction before any log line is captured.
        The redaction set lives in process memory, so after a daemon restart a reattached
        job's secrets are unknown until this runs (submit_attempt is not called again).
        Missing or unreadable secrets are skipped and retried on the next capture open."""
        if self._secrets_registered or not job.spec.secrets:
            self._secrets_registered = True
            return
        from gpu_router import secrets as secret_store

        ok = True
        for name in job.spec.secrets:
            try:
                secret_store.get_secret(name)
            except SecretsError as exc:
                ok = False
                emit(
                    "secrets.unavailable",
                    f"{job.short_id}: could not read secret {name!r} for log redaction: "
                    f"{_err_text(exc)}",
                    level=logging.WARNING,
                    log=_logger,
                    job_id=job.id,
                )
        self._secrets_registered = ok

    def _open_capture(self, job: Job, attempt: Attempt) -> LogCapture:
        self._register_secrets(job)
        if self._capture is not None and self._capture_attempt == attempt.id:
            return self._capture
        self._close_capture()
        cap = LogCapture(
            job.id,
            attempt.n,
            self.deps.paths.job_log(job.id, attempt.n),
            self.deps.paths.job_metrics(job.id),
            helper_seen=job.progress.source == "helper",
        )
        cap.open(attempt.log_lines)
        self._capture = cap
        self._capture_attempt = attempt.id
        return cap

    def _close_capture(self) -> None:
        if self._capture is not None:
            with contextlib.suppress(OSError):
                self._capture.close()
        self._capture = None
        self._capture_attempt = None

    async def _capture_logs(self, job: Job, attempt: Attempt) -> CaptureResult:
        """Fetch and ingest new log lines (appended to the log file). Nothing is persisted
        here: the caller commits the cursor with `_commit_capture` only AFTER the facts
        parsed from these lines (checkpoints, running transition) are written, so a crash
        in between re-reads the same lines instead of losing them (record_checkpoint is
        idempotent on (job_id, seq))."""
        cap = self._open_capture(job, attempt)
        try:
            chunks = await self.deps.caller.logs(
                attempt.provider, remote_ref(attempt), since=attempt.log_cursor
            )
        except ADAPTER_FAILURES as exc:
            emit(
                "attempt.logs",
                f"{attempt.id}: log fetch failed: {_err_text(exc)}",
                level=logging.DEBUG,
                log=_logger,
                job_id=job.id,
                attempt_id=attempt.id,
            )
            chunks = []
        return cap.ingest(chunks, now=self._now)

    def _cursor_patch_fields(self, attempt: Attempt, result: CaptureResult) -> dict[str, object]:
        fields: dict[str, object] = {"last_seen_at": self._now}
        cap = self._capture
        if cap is None:
            return fields
        if result.cursor is not None and result.cursor != attempt.log_cursor:
            fields["log_cursor"] = result.cursor
            fields["log_lines"] = cap.lines
        elif result.lines_added:
            fields["log_lines"] = cap.lines
        return fields

    def _commit_progress(self, job: Job, result: CaptureResult) -> None:
        if not result.has_progress:
            return
        current = self.deps.store.get_job(job.id)
        fields: dict[str, object] = {}
        if result.step is not None:
            fields["progress_step"] = result.step
        if result.total is not None:
            fields["progress_total"] = result.total
        if result.progress_source is not None:
            fields["progress_source"] = result.progress_source
        if result.metrics:  # bounded: a job can report thousands of names (D48)
            fields["last_metrics"] = merge_metrics(current.last_metrics, result.metrics)
        self.deps.store.update_job(job.id, JobPatch.model_validate(fields))

    def _commit_capture(self, job: Job, attempt: Attempt, result: CaptureResult) -> Attempt:
        """Persist progress/metrics, then the log cursor + line count (the last of a poll's
        derived writes)."""
        self._commit_progress(job, result)
        return self.deps.store.update_attempt(
            attempt.id, AttemptPatch.model_validate(self._cursor_patch_fields(attempt, result))
        )

    async def _poll(self, job: Job, attempt: Attempt) -> None:
        store = self.deps.store
        provider = attempt.provider
        try:
            st = await self.deps.caller.status(provider, remote_ref(attempt))
        except NotFound:
            await self._handle_lost(
                job, attempt, "the provider no longer has this run", status_lost=False, quota=False
            )
            return
        except ADAPTER_FAILURES as exc:
            await self._status_trouble(job, attempt, exc)
            return
        self._reachable(attempt)
        if attempt.id in self._stale_noted:
            self._stale_noted.discard(attempt.id)
            store.add_note(
                job.id,
                reason=Reason.STATUS_RECOVERED,
                actor=ACTOR,
                attempt_id=attempt.id,
                message=f"{provider} is reachable again",
            )
        self._retry_k = 0
        try:
            result = await self._capture_logs(job, attempt)
            job = store.get_job(job.id)
            if (
                job.state
                not in (
                    JobState.PROVISIONING,
                    JobState.RUNNING,
                    JobState.CHECKPOINTING,
                )
                or job.current_attempt_id != attempt.id
            ):
                # A user action (cancel) landed during the adapter calls: leave the job to
                # its new state's step. Nothing parsed was committed, so drop the capture;
                # it reopens at the persisted line count.
                self._close_capture()
                return
            ended = st.phase not in (RemotePhase.PENDING, RemotePhase.RUNNING)
            if st.message != attempt.remote_message and (st.message or ended):
                # an ended run without a final message clears the running one ("setting up
                # the python env" stayed on a finished attempt, 2026-10-04 field test)
                attempt = store.update_attempt(attempt.id, AttemptPatch(remote_message=st.message))
            if result.devices:
                self._check_gpu(job, attempt, result.devices)

            if job.state is JobState.PROVISIONING and (
                st.phase is RemotePhase.RUNNING
                or (st.phase is not RemotePhase.PENDING and result.checkpoint_events)
            ):
                job = self._mark_running(job, attempt, st)
                attempt = store.get_attempt(attempt.id)

            if job.state in (JobState.RUNNING, JobState.CHECKPOINTING):
                job = self._apply_checkpoints(job, attempt, result)
            elif result.checkpoint_events:
                self._record_checkpoints(job, attempt, result)

            if st.phase is RemotePhase.FAILED:
                # the cursor commits inside the failed transition below (with the exit code)
                self._commit_progress(job, result)
            else:
                attempt = self._commit_capture(job, attempt, result)
        except BaseException:
            # Whatever was ingested but not committed is re-read next time; reopening the
            # capture truncates the file back to the persisted line count.
            self._close_capture()
            raise

        phase = st.phase
        if phase is RemotePhase.PENDING:
            await self._pending(job, attempt, st)
            return
        if phase is RemotePhase.RUNNING:
            await self.sleep(self._poll_interval(provider))
            return
        if phase is RemotePhase.SUCCEEDED:
            await self._succeeded(job, attempt, st)
            return
        if phase is RemotePhase.FAILED:
            code = st.exit_code if st.exit_code is not None else result.exit_code
            code_txt = f"code {code}" if code is not None else "an error"
            started, _ = self._run_start(attempt, st)
            store.transition(
                job.id,
                from_state=job.state,
                to_state=JobState.FAILED,
                reason=Reason.SCRIPT_FAILED,
                actor=ACTOR,
                message=(
                    f"script exited with {code_txt} on {provider}; see `gpu logs {job.short_id}`"
                ),
                patch=JobPatch.model_validate(
                    {
                        **self._start_stamp(job, attempt, started, job_level=True),
                        "failure_kind": FailureKind.USER_ERROR,
                        "exit_code": code,
                    }
                ),
                attempt=AttemptChange(
                    attempt.id,
                    AttemptPatch.model_validate(
                        {
                            **self._cursor_patch_fields(attempt, result),
                            **self._start_stamp(job, attempt, started, job_level=False),
                            "state": AttemptState.FAILED,
                            "exit_code": code,
                        }
                    ),
                ),
            )
            return
        if phase is RemotePhase.CANCELLED:
            store.transition(
                job.id,
                from_state=job.state,
                to_state=JobState.CANCELLED,
                reason=Reason.REMOTE_CANCELLED,
                actor=ACTOR,
                message=f"the run was stopped outside gpu-router on {provider}",
                attempt=AttemptChange(attempt.id, AttemptPatch(state=AttemptState.CANCELLED)),
            )
            return
        await self._handle_lost(
            job,
            attempt,
            st.lost_reason or "session ended",
            status_lost=False,
            quota=st.quota_exhausted,
        )

    def _session_deadline(self, attempt: Attempt, st: RemoteStatus) -> float | None:
        """When the provider ends this session, conservatively (D44): from the run's own
        start when the adapter knows it, else from the submit (Kaggle reports no start;
        a daemon that slept through the start would otherwise anchor it at wake-up), and
        the shorter of the catalog cap and the limit the adapter set on the run
        (`remote_meta["session_s"]`, e.g. Kaggle's `-t`)."""
        cap: float | None = None
        with contextlib.suppress(ProviderNotFound):
            cap = self.deps.registry.entry(attempt.provider).session_cap_s
        raw = attempt.remote_meta.get("session_s")
        try:
            limit = float(raw) if raw is not None else None
        except (TypeError, ValueError):
            limit = None
        if limit is not None and limit > 0:
            cap = limit if cap is None else min(cap, limit)
        if not cap:
            return None
        anchor = st.started_at or attempt.submitted_at or attempt.created_at or self._now
        return anchor + cap

    def _mark_running(self, job: Job, attempt: Attempt, st: RemoteStatus) -> Job:
        now = self._now
        started = st.started_at or now
        deadline = self._session_deadline(attempt, st)
        gpu = st.gpu or attempt.gpu
        gpu_txt = f" {gpu}" if gpu else ""
        att = AttemptPatch(
            state=AttemptState.RUNNING,
            started_at=started,
            last_seen_at=now,
            session_deadline=deadline,
        )
        if st.gpu:
            att = AttemptPatch(
                state=AttemptState.RUNNING,
                started_at=started,
                last_seen_at=now,
                gpu=st.gpu,
                session_deadline=deadline,
            )
        patch = JobPatch(gpu=gpu) if st.gpu else JobPatch()
        job = self.deps.store.transition(
            job.id,
            from_state=JobState.PROVISIONING,
            to_state=JobState.RUNNING,
            reason=Reason.STARTED,
            actor=ACTOR,
            patch=patch,
            message=f"running on {attempt.provider}{gpu_txt}",
            attempt=AttemptChange(attempt.id, att),
        )
        crashpoints.crashpoint("after_running")
        return job

    def _record_checkpoints(self, job: Job, attempt: Attempt, result: CaptureResult) -> None:
        for ev in result.checkpoint_events:
            if ev.t == "ckpt_end" and ev.seq is not None and ev.uri is not None:
                self.deps.store.record_checkpoint(
                    job.id,
                    attempt.id,
                    seq=ev.seq,
                    uri=ev.uri,
                    step=ev.step,
                    size_bytes=ev.size,
                    sha256=ev.sha256,
                    created_at=self._now,
                )

    def _apply_checkpoints(self, job: Job, attempt: Attempt, result: CaptureResult) -> Job:
        store = self.deps.store
        for ev in result.checkpoint_events:
            if ev.t == "ckpt_begin" and job.state is JobState.RUNNING:
                job = store.transition(
                    job.id,
                    from_state=JobState.RUNNING,
                    to_state=JobState.CHECKPOINTING,
                    reason=Reason.CHECKPOINT_BEGIN,
                    actor=ACTOR,
                    message=f"saving checkpoint {ev.seq}",
                    detail={"seq": ev.seq},
                    attempt=None,
                )
                self._ckpt_began_at = self._now
                crashpoints.crashpoint("mid_checkpoint")
            elif ev.t == "ckpt_end" and ev.seq is not None and ev.uri is not None:
                store.record_checkpoint(
                    job.id,
                    attempt.id,
                    seq=ev.seq,
                    uri=ev.uri,
                    step=ev.step,
                    size_bytes=ev.size,
                    sha256=ev.sha256,
                    created_at=self._now,
                )
                if job.state is JobState.CHECKPOINTING:
                    job = store.transition(
                        job.id,
                        from_state=JobState.CHECKPOINTING,
                        to_state=JobState.RUNNING,
                        reason=Reason.CHECKPOINT_END,
                        actor=ACTOR,
                        message=f"checkpoint {ev.seq} saved; running on {attempt.provider}",
                        detail={"seq": ev.seq, "uri": ev.uri, "step": ev.step},
                    )
                    self._ckpt_began_at = None
        if job.state is JobState.CHECKPOINTING:
            began = self._ckpt_began_at
            if began is None:
                began = self._ckpt_began_at = self._now
            stall = self.deps.config.engine.checkpoint_stall_s
            if self._now - began >= stall:
                job = store.transition(
                    job.id,
                    from_state=JobState.CHECKPOINTING,
                    to_state=JobState.RUNNING,
                    reason=Reason.CHECKPOINT_STALLED,
                    actor=ACTOR,
                    message=(
                        f"checkpoint did not finish within {fmt_duration(stall)}; "
                        f"still running, the previous checkpoint stays the resume point"
                    ),
                )
                self._ckpt_began_at = None
        return job

    async def _pending(self, job: Job, attempt: Attempt, st: RemoteStatus) -> None:
        cfg = self.deps.config.engine
        provider = attempt.provider
        if job.state is JobState.PROVISIONING:
            since = attempt.submitted_at or attempt.created_at
            if self._now - since >= cfg.provision_timeout_s:
                try:
                    await self.deps.caller.cancel(provider, remote_ref(attempt))
                except NotFound:
                    pass
                except ADAPTER_FAILURES as exc:
                    # Never record "cancelled" for a run that may still start: keep the
                    # attempt live (the next poll may even find it running) and retry.
                    await self._provision_cancel_failed(job, attempt, since, exc)
                    return
                timeout_exc = _ProvisionTimeout(f"{provider} found no GPU in time")
                phrase = self._react_provider(provider, timeout_exc)
                self.deps.store.transition(
                    job.id,
                    from_state=JobState.PROVISIONING,
                    to_state=JobState.QUEUED,
                    reason=Reason.PROVISION_TIMEOUT,
                    actor=ACTOR,
                    message=(
                        f"no GPU on {provider} after "
                        f"{fmt_duration(self._now - since)} ({phrase}); cancelled it, "
                        f"choosing another provider"
                    ),
                    patch=JobPatch(current_attempt_id=None),
                    attempt=AttemptChange(
                        attempt.id,
                        AttemptPatch(state=AttemptState.CANCELLED, lost_reason="provision timeout"),
                    ),
                )
                return
        await self.sleep(self._poll_interval(provider))

    async def _provision_cancel_failed(
        self, job: Job, attempt: Attempt, since: float, exc: BaseException
    ) -> None:
        cfg = self.deps.config.engine
        provider = attempt.provider
        waited = self._now - since
        emit(
            "attempt.cancel",
            f"{attempt.id}: cancel after provision timeout failed: {_err_text(exc)}",
            level=logging.WARNING,
            log=_logger,
            job_id=job.id,
            attempt_id=attempt.id,
        )
        if waited < cfg.provision_timeout_s + cfg.cancel_timeout_s:
            await self._transient_pause()
            return
        phrase = self._react_provider(provider, exc)
        self.deps.store.add_note(
            job.id,
            reason=Reason.ATTEMPT_ABANDONED,
            actor=ACTOR,
            attempt_id=attempt.id,
            message=(
                f"could not stop the pending run on {provider} ({phrase}); it may still "
                f"start there (check its console). not using {provider} again for this job"
            ),
        )
        self.deps.store.transition(
            job.id,
            from_state=JobState.PROVISIONING,
            to_state=JobState.QUEUED,
            reason=Reason.PROVISION_TIMEOUT,
            actor=ACTOR,
            message=(
                f"no GPU on {provider} after {fmt_duration(waited)} and the run could not be "
                f"cancelled; choosing another provider"
            ),
            patch=JobPatch(current_attempt_id=None),
            attempt=AttemptChange(
                attempt.id,
                AttemptPatch(
                    state=AttemptState.ABANDONED,
                    error_message=_err_text(exc),
                    lost_reason="provision timeout; cancel not confirmed",
                ),
            ),
        )

    async def _succeeded(self, job: Job, attempt: Attempt, st: RemoteStatus) -> None:
        fetched = await self.fetch_outputs(job, attempt)
        crashpoints.crashpoint("after_fetch")
        job = self.deps.store.get_job(job.id)
        if job.state not in (JobState.PROVISIONING, JobState.RUNNING, JobState.CHECKPOINTING):
            return  # a user cancel landed during the fetch: step_cancelling finishes it
        took = ""
        started, from_submit = self._run_start(attempt, st)
        if started is not None:
            dur = fmt_duration((st.ended_at or self._now) - started)
            took = f" within {dur} of submit" if from_submit else f" in {dur}"
        where = (
            f"; outputs in {job.outputs_dir}"
            if fetched
            else f"; outputs could not be fetched (retry with `gpu fetch {job.short_id}`)"
        )
        seen = self._gpu_seen_text(job, attempt)
        self.deps.store.transition(
            job.id,
            from_state=job.state,
            to_state=JobState.DONE,
            reason=Reason.COMPLETED,
            actor=ACTOR,
            message=f"finished on {attempt.provider}{took}{seen}{where}",
            patch=JobPatch.model_validate(
                {
                    **self._start_stamp(job, attempt, started, job_level=True),
                    "exit_code": st.exit_code if st.exit_code is not None else 0,
                }
            ),
            attempt=AttemptChange(
                attempt.id,
                AttemptPatch.model_validate(
                    {
                        **self._start_stamp(job, attempt, started, job_level=False),
                        "state": AttemptState.SUCCEEDED,
                        "exit_code": st.exit_code if st.exit_code is not None else 0,
                    }
                ),
            ),
        )

    @staticmethod
    def _run_start(attempt: Attempt, st: RemoteStatus) -> tuple[float | None, bool]:
        """When the run started, and whether that is only the submit time: a run that
        finishes between two polls (Kaggle polls every 60 s) is never seen `running`, so
        without this the job has no started_at ("took -") and the phase-5 usage ledger
        (attempts with started_at) would skip it. Charging from submit over-counts the queue
        wait, the safe side for a quota ledger (phase-3 integration, D33)."""
        started = attempt.started_at or st.started_at
        if started is not None:
            return started, False
        return attempt.submitted_at, attempt.submitted_at is not None

    @staticmethod
    def _start_stamp(
        job: Job, attempt: Attempt, started: float | None, *, job_level: bool
    ) -> dict[str, float]:
        """`started_at` patch fields for a run that ends without having been seen running."""
        if started is None:
            return {}
        current = job.started_at if job_level else attempt.started_at
        return {} if current is not None else {"started_at": started}

    async def _handle_lost(
        self, job: Job, attempt: Attempt, why: str, *, status_lost: bool, quota: bool
    ) -> None:
        store = self.deps.store
        provider = attempt.provider
        if quota:
            self._react_provider(provider, QuotaExhausted(why, provider=provider))
        att = AttemptChange(attempt.id, AttemptPatch(state=AttemptState.LOST, lost_reason=why))
        if job.state is JobState.PROVISIONING:
            store.transition(
                job.id,
                from_state=JobState.PROVISIONING,
                to_state=JobState.QUEUED,
                reason=Reason.QUOTA_EXHAUSTED if quota else Reason.LOST_BEFORE_START,
                actor=ACTOR,
                attempt=att,
                patch=JobPatch(current_attempt_id=None),
                message=f"{provider} lost the run before it started ({why}); trying again",
            )
            return
        if job.spec.interactive:
            store.transition(
                job.id,
                from_state=job.state,
                to_state=JobState.FAILED,
                reason=Reason.INTERACTIVE_LOST,
                actor=ACTOR,
                attempt=att,
                message=(
                    f"interactive session on {provider} ended ({why}); interactive "
                    f"jobs are not migrated, start a new one"
                ),
                patch=JobPatch(failure_kind=FailureKind.PROVIDER_ERROR),
            )
            return
        latest = store.latest_checkpoint(job.id)
        nxt = (
            f"resuming from checkpoint {latest.seq} elsewhere"
            if latest
            else "restarting it from the beginning (no checkpoint yet)"
        )
        if quota:
            reason = Reason.QUOTA_EXHAUSTED
        elif status_lost:
            reason = Reason.STATUS_LOST
        else:
            reason = Reason.SESSION_LOST
        store.transition(
            job.id,
            from_state=job.state,
            to_state=JobState.MIGRATING,
            reason=reason,
            actor=ACTOR,
            attempt=att,
            message=f"{provider} session ended ({why}); {nxt}",
            detail={"previous_provider": provider, "checkpoint": latest.id if latest else None},
        )

    async def _status_trouble(self, job: Job, attempt: Attempt, exc: BaseException) -> None:
        store = self.deps.store
        cfg = self.deps.config.engine
        provider = attempt.provider
        if isinstance(exc, AuthRequired):
            self._react_provider(provider, exc)
        unreachable = self._unreachable_for(attempt)
        if unreachable >= cfg.unreachable_lost_after_s:
            with contextlib.suppress(*ADAPTER_FAILURES):
                await self.deps.caller.cancel(provider, remote_ref(attempt))
            self._stale_noted.discard(attempt.id)
            self._reachable(attempt)
            await self._handle_lost(
                job,
                attempt,
                f"unreachable for {fmt_duration(unreachable)}",
                status_lost=True,
                quota=False,
            )
            return
        if unreachable >= cfg.status_stale_after_s and attempt.id not in self._stale_noted:
            self._stale_noted.add(attempt.id)
            store.add_note(
                job.id,
                reason=Reason.STATUS_UNREACHABLE,
                actor=ACTOR,
                attempt_id=attempt.id,
                message=(
                    f"cannot reach {provider} for {fmt_duration(unreachable)} "
                    f"({_err_text(exc)}); still trying, the job is placed elsewhere "
                    f"after {fmt_duration(cfg.unreachable_lost_after_s)}"
                ),
                detail={"error": type(exc).__name__},
            )
        self._retry_k += 1
        await self.sleep(max(self._poll_interval(provider), self._backoff(self._retry_k)))

    # ------------------------------------------------------------------ cancel helpers

    def _cancel_timed_out(self, job: Job) -> bool:
        requested = job.cancel_requested_at or self._entered_state_at(job)
        return self._now - requested >= self.deps.config.engine.cancel_timeout_s

    def _finish_cancel(
        self,
        job: Job,
        attempt: Attempt | None,
        patch: AttemptPatch | None,
        message: str,
        *,
        reason: Reason = Reason.CANCELLED,
    ) -> None:
        change = None
        if attempt is not None and patch is not None:
            change = AttemptChange(attempt.id, patch)
        self.deps.store.transition(
            job.id,
            from_state=JobState.CANCELLING,
            to_state=JobState.CANCELLED,
            reason=reason,
            message=message,
            actor=ACTOR,
            attempt=change,
        )

    # ------------------------------------------------------------------ outputs

    async def fetch_outputs(self, job: Job, attempt: Attempt) -> bool:
        """crashpoint("before_fetch"); adapter.fetch into job.outputs_dir; note fetched or
        fetch_failed (the job still goes to done: outputs failure is explained, not fatal).
        Returns True when outputs were fetched."""
        crashpoints.crashpoint("before_fetch")
        store = self.deps.store
        if job.outputs_dir is None or attempt.remote_id is None:
            return False
        dest = Path(job.outputs_dir)
        try:
            res = await self.deps.caller.fetch(attempt.provider, remote_ref(attempt), dest)
        except ADAPTER_FAILURES as exc:
            store.add_note(
                job.id,
                reason=Reason.FETCH_FAILED,
                actor=ACTOR,
                attempt_id=attempt.id,
                message=(
                    f"could not fetch outputs from {attempt.provider} "
                    f"({_err_text(exc)}); retry with `gpu fetch {job.short_id}`"
                ),
                detail={"error": type(exc).__name__},
            )
            return False
        partial = f"; partial: {res.message}" if res.partial and res.message else ""
        store.add_note(
            job.id,
            reason=Reason.FETCHED,
            actor=ACTOR,
            attempt_id=attempt.id,
            message=f"outputs saved to {dest} ({res.files} files){partial}",
            detail={"files": res.files, "bytes": res.bytes, "partial": res.partial},
            patch=JobPatch(outputs_fetched=True),
        )
        return True

    # ------------------------------------------------------------------ checkpoint storage

    def _kind(self, provider: str) -> str:
        try:
            return self.deps.registry.get(provider).kind
        except ProviderNotFound:
            return ""

    def _check_gpu(self, job: Job, attempt: Attempt, seen: tuple[str, ...]) -> None:
        """Note once per attempt when the runner's nvidia-smi saw another GPU than the one
        the attempt was placed on (D56): status then says what really ran, never only
        the requested GPU."""
        got = gpu_mismatch(attempt.gpu, seen)
        if got is None:
            return
        self._note_once(
            job,
            f"gpu:{attempt.id}",
            reason=Reason.GPU_MISMATCH,
            attempt_id=attempt.id,
            message=(
                f"{attempt.provider} gave this run a {got}, not the {attempt.gpu} it was "
                f"placed on (nvidia-smi); gpu-router lets it run and says so when it ends"
            ),
            detail={"placed": attempt.gpu, "seen": list(seen)},
        )

    def _gpu_seen_text(self, job: Job, attempt: Attempt) -> str:
        """'; ran on a Tesla T4, not the L4 it was placed on' when this attempt has a
        gpu_mismatch note, else ''."""
        for ev in self.deps.store.events_for(job.id, limit=10_000):
            if (
                ev.kind == "note"
                and ev.reason == Reason.GPU_MISMATCH
                and ev.attempt_id == attempt.id
            ):
                got = gpu_mismatch(attempt.gpu, list(ev.detail.get("seen") or []))
                if got is not None:
                    return f"; ran on a {got}, not the {attempt.gpu} it was placed on"
        return ""

    def _note_once(self, job: Job, key: str, **note: Any) -> None:
        """add_note unless this driver (or, after a restart, the job's events) already has
        a note with the same reason for the same key."""
        if key in self._noted:
            return
        self._noted.add(key)
        reason = note["reason"]
        for ev in self.deps.store.events_for(job.id, limit=10_000):
            if ev.reason == reason and ev.kind == "note" and ev.detail.get("key") == key:
                return
        detail = dict(note.pop("detail", None) or {})
        detail["key"] = key
        self.deps.store.add_note(job.id, actor=ACTOR, detail=detail, **note)

    def _storage_waited(self, key: str) -> float:
        """How long the current run of storage trouble for `key` has lasted (0 = none)."""
        first = self._storage_trouble.get(key)
        return 0.0 if first is None else max(0.0, self._now - first)

    async def _storage_extras(self, job: Job, attempt: Attempt, *, degrade: bool) -> _Extras:
        """Checkpoint storage + dataset settings for this attempt's runner (phase 5):
        env (GPU_STORAGE, GPU_RESUME_URI, GPU_DATA, ...) and the storage token secret.
        Raises _DataProblem (reroute or fail) or StorageError (storage cannot be reached
        right now: the caller waits, D44). `degrade`: waited long enough, go ahead
        without what cannot be reached."""
        extras = _Extras()
        hub = self.deps.checkpoints
        if hub is None:
            return extras
        kind = self._kind(attempt.provider)
        resume: Checkpoint | None = None
        if attempt.resume_checkpoint_id is not None:
            for ckpt in self.deps.store.checkpoints_for(job.id):
                if ckpt.id == attempt.resume_checkpoint_id:
                    resume = ckpt
                    break
        try:
            got = await hub.call(
                hub.prepare_attempt,
                bulk=True,
                limit_s=BULK_TIMEOUT_S,
                job_id=job.id,
                attempt_n=attempt.n,
                kind=kind,
                resume=resume,
                secret_names=list(job.spec.secrets),
                degrade=degrade,
            )
        except StorageError as exc:
            if not degrade:
                raise
            got = None
            self.deps.store.add_note(
                job.id,
                reason=Reason.STORAGE_UNAVAILABLE,
                actor=ACTOR,
                attempt_id=attempt.id,
                message=(
                    f"checkpoint storage still cannot be reached ({exc.message}); attempt "
                    f"{attempt.n} goes ahead without it"
                    + (", starting over" if resume is not None else "")
                ),
            )
        if got is None:
            if not hub.is_local_kind(kind) and hub.remote_expected() and not job.spec.interactive:
                st = hub.status()
                hint = f"; {st.hint}" if st.hint else ""
                self._note_once(
                    job,
                    f"storage:{attempt.provider}",
                    reason=Reason.STORAGE_UNAVAILABLE,
                    attempt_id=attempt.id,
                    message=(
                        f"checkpoints of runs on {attempt.provider} stay on that machine "
                        f"({st.reason or 'no checkpoint storage'}): if the job has to move it "
                        f"starts over{hint}"
                    ),
                )
        else:
            extras.env.update(got.env)
            extras.secrets.update(got.secrets)
            for note in got.notes:
                self.deps.store.add_note(
                    job.id,
                    reason=Reason.CHECKPOINT_COPIED,
                    actor=ACTOR,
                    attempt_id=attempt.id,
                    message=note,
                    detail={"storage": got.kind},
                )
        if job.spec.data:
            remote_storage = got is not None and got.kind == "hf"
            extras.env["GPU_DATA"] = json.dumps(
                await self._prepare_data(
                    job, attempt, kind, degrade=degrade, remote_storage=remote_storage
                ),
                separators=(",", ":"),
            )
        return extras

    async def _prepare_data(
        self,
        job: Job,
        attempt: Attempt,
        kind: str,
        *,
        degrade: bool = False,
        remote_storage: bool = True,
    ) -> list[dict[str, str]]:
        """What GPU_DATA tells the runner per dataset: a local path (runs on this Mac), or
        the storage URI of its upload, uploaded once per content hash (data_cache)."""
        from gpu_router.checkpoint.data import resolve_data_path

        hub = self.deps.checkpoints
        assert hub is not None
        store = self.deps.store
        items: list[dict[str, str]] = []
        for ref in job.spec.data:
            if ref.uri is not None:
                items.append({"mount": ref.mount, "uri": ref.uri})
                continue
            assert ref.path is not None
            path = resolve_data_path(ref.path, job.spec.project_dir)
            if not path.exists():
                raise _DataProblem(f"dataset {ref.mount!r}: {path} does not exist", reroute=False)
            if hub.is_local_kind(kind):
                items.append({"mount": ref.mount, "local": str(path)})
                continue
            if await hub.call(hub.hf) is None or not remote_storage:
                if self.deps.registry.get(attempt.provider).capabilities.stage_data:
                    # the provider keeps datasets itself (Kaggle datasets): no storage needed
                    items.append(await self._stage_on_provider(job, attempt, ref, path, degrade))
                    continue
                # the runner downloads the upload with its own storage token: without
                # storage for this attempt (no HF_TOKEN_REMOTE, D44) it cannot
                st = hub.status()
                if hub.hf_down_for_now() and not degrade:
                    raise StorageError(
                        f"dataset {ref.mount!r} needs Hugging Face storage, which cannot be "
                        f"reached right now ({st.reason or 'no answer'})"
                    )
                raise _DataProblem(
                    f"dataset {ref.mount!r} cannot reach {attempt.provider}: it needs Hugging "
                    f"Face storage ({st.reason or 'not configured'})",
                    reroute=True,
                    hint=st.hint,
                )
            try:
                dig = await hub.call(hub.digest, path, bulk=True, limit_s=BULK_TIMEOUT_S)
            except OSError as exc:
                # a file vanished or cannot be read: a data problem naming the file (D44)
                where = exc.filename or path
                raise _DataProblem(
                    f"dataset {ref.mount!r}: cannot read {where} ({exc.strerror or exc})",
                    reroute=False,
                ) from None
            uri: str | None = None
            cached = store.get_data_cache(dig.sha256)
            try:
                if cached is not None:
                    uri = await hub.call(hub.dataset_uri, dig.sha256)
                    if uri is None:
                        store.delete_data_cache(dig.sha256)  # the upload is gone: again
                if uri is None:
                    uri = await hub.call(
                        hub.upload_dataset, path, dig, bulk=True, limit_s=BULK_TIMEOUT_S
                    )
                    uploaded = True
                else:
                    uploaded = False
            except StorageError as exc:
                if exc.retryable and not degrade:
                    raise
                # refused (no write access, the storage quota is full) or still failing
                # after the wait: this provider cannot get the data; try another (D44)
                raise _DataProblem(
                    f"dataset {ref.mount!r} could not be stored for {attempt.provider} "
                    f"({exc.message})",
                    reroute=True,
                    hint="check `gpu providers` and the bucket's storage quota",
                ) from None
            if not uploaded:
                store.touch_data_cache(dig.sha256, local_path=str(path))
                store.add_note(
                    job.id,
                    reason=Reason.DATA_REUSED,
                    actor=ACTOR,
                    attempt_id=attempt.id,
                    message=(
                        f"dataset {ref.mount} is already uploaded ({dig.count} files, "
                        f"{_mb(dig.size)}); reusing it"
                    ),
                    detail={"mount": ref.mount, "sha256": dig.sha256, "uri": uri},
                )
            else:
                store.put_data_cache(
                    content_hash=dig.sha256,
                    uri=uri,
                    local_path=str(path),
                    size_bytes=dig.size,
                    file_count=dig.count,
                )
                store.add_note(
                    job.id,
                    reason=Reason.DATA_UPLOADED,
                    actor=ACTOR,
                    attempt_id=attempt.id,
                    message=(
                        f"uploaded dataset {ref.mount} ({dig.count} files, {_mb(dig.size)}) "
                        f"to checkpoint storage; later runs reuse it"
                    ),
                    detail={"mount": ref.mount, "sha256": dig.sha256, "uri": uri},
                )
                await self._evict_datasets(keep=dig.sha256)
            items.append({"mount": ref.mount, "uri": uri, "sha256": dig.sha256})
        return items

    async def _stage_on_provider(
        self, job: Job, attempt: Attempt, ref: DataRef, path: Path, degrade: bool
    ) -> dict[str, str]:
        """A dataset put in the provider's own store through adapter.stage_data, once per
        content hash (the adapter reuses an earlier upload). Transient trouble waits like
        storage trouble (checkpoint.storage_wait_s), then the provider is excluded."""
        hub = self.deps.checkpoints
        assert hub is not None
        store = self.deps.store
        provider = attempt.provider
        try:
            dig = await hub.call(hub.digest, path, bulk=True, limit_s=BULK_TIMEOUT_S)
        except OSError as exc:
            where = exc.filename or path
            raise _DataProblem(
                f"dataset {ref.mount!r}: cannot read {where} ({exc.strerror or exc})",
                reroute=False,
            ) from None
        key = f"stage:{attempt.id}:{dig.sha256}"
        if dig.size >= STAGE_NOTE_BYTES and key not in self._noted:
            # only when the wait is noticeable: a tiny dataset read "putting ..." then
            # "reusing it" a second later (2026-10-04 field test)
            self._noted.add(key)
            store.add_note(
                job.id,
                reason=Reason.DATA_UPLOADING,
                actor=ACTOR,
                attempt_id=attempt.id,
                message=(
                    f"uploading dataset {ref.mount} ({dig.count} files, {_size(dig.size)}) to "
                    f"{provider} unless the same content is already there; the job starts "
                    "once it is"
                ),
                detail={"mount": ref.mount, "sha256": dig.sha256},
            )
        try:
            staged = await self.deps.caller.stage_data(provider, path, dig.sha256, dig.files)
        except (Unavailable, RateLimited, AdapterContractViolation) as exc:
            # invariant 7: a contract violation (a local error inside the adapter) is
            # treated like Unavailable: wait and retry, then exclude
            if not degrade:
                raise _StagePending(
                    f"dataset {ref.mount!r} is not on {provider} yet ({exc.message})"
                ) from None
            raise _DataProblem(
                f"dataset {ref.mount!r} could not be put on {provider} ({exc.message})",
                reroute=True,
                hint=getattr(exc, "hint", None),
            ) from None
        except AuthRequired as exc:
            # a login problem is the provider's, not the job's: mark it (health) and
            # requeue instead of excluding the provider for this job
            raise _DataProblem(exc.message, reroute=True, hint=exc.hint, error=exc) from None
        except AdapterError as exc:
            raise _DataProblem(
                f"dataset {ref.mount!r} cannot be put on {provider} ({exc.message})",
                reroute=True,
                hint=exc.hint,
            ) from None
        store.add_note(
            job.id,
            reason=Reason.DATA_UPLOADED if staged.uploaded else Reason.DATA_REUSED,
            actor=ACTOR,
            attempt_id=attempt.id,
            message=(
                f"uploaded dataset {ref.mount} ({dig.count} files, {_size(dig.size)}) as a "
                f"{staged.where}; later runs reuse it"
                if staged.uploaded
                else f"dataset {ref.mount} is already a {staged.where}; reusing it"
            ),
            detail={"mount": ref.mount, "sha256": dig.sha256, "uri": staged.uri},
        )
        return {"mount": ref.mount, "uri": staged.uri, "sha256": dig.sha256}

    def _record_stored(self, job: Job, found: StoredCheckpoint, why: str) -> Checkpoint | None:
        """Record a checkpoint storage knows about but the captured logs did not (a runner
        that died before its ckpt_end line reached us, a handoff ack). Only newer ones."""
        store = self.deps.store
        latest = store.latest_checkpoint(job.id)
        if latest is not None and found.seq <= latest.seq:
            return None
        attempts = store.attempts_for(job.id)
        if not attempts:
            return None
        by_n = {a.n: a for a in attempts}
        writer = by_n.get(found.attempt) if found.attempt is not None else None
        attempt = writer or attempts[-1]
        try:
            ckpt = store.record_checkpoint(
                job.id,
                attempt.id,
                seq=found.seq,
                uri=found.uri,
                step=found.step,
                size_bytes=found.size,
                sha256=found.sha256,
                created_at=found.created_at or self._now,
            )
        except ValueError:
            return None
        step = f" (step {found.step})" if found.step is not None else ""
        store.add_note(
            job.id,
            reason=Reason.CHECKPOINT_FOUND,
            actor=ACTOR,
            attempt_id=attempt.id,
            message=f"checkpoint {found.seq}{step} from attempt {attempt.n} is in storage ({why})",
            detail={"seq": found.seq, "uri": found.uri, "step": found.step},
        )
        return ckpt

    async def _evict_datasets(self, *, keep: str) -> None:
        """Delete uploaded datasets no run used for checkpoint.dataset_keep_days (LRU on
        data_cache.last_used_at), so the bucket's free storage does not fill up (D44).
        Best effort, after an upload."""
        hub = self.deps.checkpoints
        days = self.deps.config.checkpoint.dataset_keep_days
        if hub is None or days <= 0:
            return
        store = self.deps.store
        for entry in store.unused_data_cache(self._now - days * 86_400, limit=20):
            if entry.content_hash == keep:
                continue
            try:
                await hub.call(
                    hub.delete_dataset, entry.content_hash, bulk=True, limit_s=BULK_TIMEOUT_S
                )
            except StorageError:
                return
            store.delete_data_cache(entry.content_hash)
            emit(
                "storage.evict",
                f"deleted dataset {entry.content_hash[:12]} (unused for {days:g} days)",
                log=_logger,
                sha256=entry.content_hash,
            )

    async def _cleanup_storage(self, job: Job) -> None:
        """A finished job's checkpoints, owner and status files leave storage (D44):
        nothing resumes a finished job, and a bucket's free quota is small. Off with
        checkpoint.cleanup: false."""
        hub = self.deps.checkpoints
        if hub is None or not self.deps.config.checkpoint.cleanup or not job.attempt_count:
            return
        try:
            cleaned = await hub.call(hub.delete_job, job.id, bulk=True, limit_s=600.0)
        except (StorageError, OSError):
            return
        if cleaned:
            emit(
                "storage.cleanup",
                f"{job.short_id} {job.state}: removed its checkpoints from "
                + " and ".join(cleaned)
                + " storage",
                log=_logger,
                job_id=job.id,
            )

    async def _reconcile_checkpoints(self, job: Job) -> bool:
        """Before placing a migrating job: storage may hold a newer checkpoint than the
        logs reported (Kaggle only shows logs after the run; a session killed mid-run
        never delivers them). Once per migration. False = storage could not be asked and
        the step should run again later (bounded by checkpoint.storage_wait_s, D44)."""
        hub = self.deps.checkpoints
        if hub is None or job.spec.interactive:
            return True
        attempts = self.deps.store.attempts_for(job.id)
        marker = attempts[-1].id if attempts else None
        if marker is None or marker == self._reconciled:
            return True
        key = f"reconcile:{marker}"
        try:
            found = await hub.call(hub.latest, job.id)
        except StorageError as exc:
            limit = self.deps.config.checkpoint.storage_wait_s
            if exc.retryable and self._storage_waited(key) < limit:
                if key not in self._storage_trouble:
                    self._storage_trouble[key] = self._now
                    self.deps.store.add_note(
                        job.id,
                        reason=Reason.RETRY_SCHEDULED,
                        actor=ACTOR,
                        message=(
                            f"cannot ask checkpoint storage for a newer checkpoint "
                            f"({exc.message}); waiting before moving the job"
                        ),
                        detail={"error": "StorageError"},
                    )
                await self._transient_pause()
                return False
            self._reconciled = marker
            self._storage_trouble.pop(key, None)
            self.deps.store.add_note(
                job.id,
                reason=Reason.STORAGE_UNAVAILABLE,
                actor=ACTOR,
                message=(
                    f"could not ask checkpoint storage for a newer checkpoint "
                    f"({exc.message}); moving the job from the last one gpu-router knows"
                ),
            )
            return True
        self._reconciled = marker
        self._storage_trouble.pop(key, None)
        if found is not None:
            self._record_stored(job, found, "its log never reported it; resuming from it")
        return True

    # ---- planned handoff

    def _handoff_due(self, job: Job, attempt: Attempt) -> tuple[str, str] | None:
        """(code, why) when this attempt should hand off now: its session cap or the
        provider's free quota is within checkpoint.handoff_margin_min."""
        margin = self.deps.config.checkpoint.handoff_margin_min * 60
        if margin <= 0:
            return None
        now = self._now
        provider = attempt.provider
        deadline = attempt.session_deadline
        if deadline is not None and now >= deadline - margin:
            return (
                "session_cap",
                f"{provider} ends the session in {fmt_duration(max(0.0, deadline - now))} "
                f"(its session limit)",
            )
        left_h, _resets = quota_left_hours(self.deps, provider, attempt.gpu)
        if left_h is not None:
            left = left_h * 3600
            if left <= margin:
                return (
                    "quota",
                    f"{provider}'s free GPU quota runs out in about {fmt_duration(max(0.0, left))}",
                )
        return None

    def _restore_handoff(self, job: Job, attempt: Attempt) -> _Handoff | None:
        """After a daemon restart: a request this attempt already got (from job events)."""
        found: _Handoff | None = None
        for ev in self.deps.store.events_for(job.id, limit=10_000):
            if ev.attempt_id != attempt.id or ev.kind != "note":
                continue
            if ev.reason == Reason.HANDOFF_REQUESTED and ev.detail.get("request"):
                found = _Handoff(
                    request_id=str(ev.detail["request"]),
                    requested_at=ev.ts,
                    code=str(ev.detail.get("why") or "session_cap"),
                    why=ev.message,
                    wait_s=float(ev.detail.get("wait_s") or 0),
                )
            elif ev.reason == Reason.HANDOFF_SKIPPED:
                if found is None:
                    found = _Handoff("", ev.ts, "skipped", ev.message, 0.0)
                found.gave_up = True
        return found

    async def _handoff(self, job: Job, attempt: Attempt) -> bool:
        """Planned handoff (phase 5): shortly before the provider ends the session (cap or
        quota), ask the runner through storage to checkpoint, and once it answers, move
        the job (running -> migrating, reason handoff; the migrating step stops this
        attempt and resumes elsewhere from that checkpoint). Returns True when the job
        moved."""
        hub = self.deps.checkpoints
        if hub is None or job.spec.interactive or attempt.state is not AttemptState.RUNNING:
            return False
        state = self._handoff_state(job, attempt)
        kind = self._kind(attempt.provider)
        if state is None:
            due = self._handoff_due(job, attempt)
            if due is not None:
                await self._request_handoff(job, attempt, hub, kind, *due)
            return False
        if state.gave_up:
            return False
        return await self._check_handoff(job, attempt, hub, kind, state)

    def _handoff_state(self, job: Job, attempt: Attempt) -> _Handoff | None:
        """This attempt's checkpoint request, restored from job events after a restart."""
        state = self._handoffs.get(attempt.id)
        if state is None and attempt.id not in self._handoff_checked:
            self._handoff_checked.add(attempt.id)
            state = self._restore_handoff(job, attempt)
            if state is not None:
                self._handoffs[attempt.id] = state
        return state

    async def _request_handoff(
        self, job: Job, attempt: Attempt, hub: CheckpointHub, kind: str, code: str, why: str
    ) -> None:
        cfg = self.deps.config.checkpoint
        now = self._now
        wait_s = cfg.handoff_wait_min * 60
        rid = f"{attempt.id}.h{int(now)}"
        try:
            await hub.call(
                hub.request_checkpoint,
                job_id=job.id,
                attempt_n=attempt.n,
                kind=kind,
                request_id=rid,
                action="handoff",
                wait_s=wait_s,
                reason=why,
            )
        except StorageError as exc:
            state = _Handoff(rid, now, code, why, wait_s, gave_up=True)
            self._handoffs[attempt.id] = state
            then = (
                "so it is stopped without a fresh one"
                if code == OVERRUN
                else f"so it runs until {attempt.provider} stops it and then resumes from "
                "its last one"
            )
            self.deps.store.add_note(
                job.id,
                reason=Reason.HANDOFF_SKIPPED,
                actor=ACTOR,
                attempt_id=attempt.id,
                message=f"{why}; cannot ask the job for a checkpoint ({exc.message}), {then}",
                detail={"why": code},
            )
            return
        self._handoffs[attempt.id] = _Handoff(rid, now, code, why, wait_s)
        self.deps.store.add_note(
            job.id,
            reason=Reason.HANDOFF_REQUESTED,
            actor=ACTOR,
            attempt_id=attempt.id,
            message=f"{why}; asked the job to save a checkpoint so gpu-router can move it first",
            detail={"request": rid, "why": code, "wait_s": wait_s},
        )

    def _handoff_timeout(self, attempt: Attempt, state: _Handoff) -> float:
        cfg = self.deps.config.checkpoint
        poll = self._poll_interval(attempt.provider)
        slack = 2 * max(cfg.control_poll_s, cfg.status_push_s, poll)
        return state.wait_s + slack + 300

    async def _check_handoff(
        self, job: Job, attempt: Attempt, hub: CheckpointHub, kind: str, state: _Handoff
    ) -> bool:
        store = self.deps.store
        now = self._now
        try:
            ack = await hub.call(hub.read_ack, job_id=job.id, attempt_n=attempt.n, kind=kind)
        except StorageError:
            ack = None
        if ack is None or ack.request_id != state.request_id:
            if now - state.requested_at > self._handoff_timeout(attempt, state):
                state.gave_up = True
                then = (
                    "it is stopped anyway and resumes from its last checkpoint once you approve it"
                    if state.code == OVERRUN
                    else f"it keeps running on {attempt.provider} until the session ends, "
                    "then resumes from its last checkpoint"
                )
                store.add_note(
                    job.id,
                    reason=Reason.HANDOFF_SKIPPED,
                    actor=ACTOR,
                    attempt_id=attempt.id,
                    message=(
                        f"the job did not answer the checkpoint request within "
                        f"{fmt_duration(now - state.requested_at)}; {then}"
                    ),
                    detail={"request": state.request_id},
                )
            return False
        provider = attempt.provider
        try:
            st = await self.deps.caller.status(provider, remote_ref(attempt))
        except ADAPTER_FAILURES:
            st = None
        if st is not None and st.phase in (RemotePhase.SUCCEEDED, RemotePhase.FAILED):
            self._handoffs.pop(attempt.id, None)
            return False  # it finished meanwhile: the normal poll records how
        if ack.checkpoint is not None:
            self._record_stored(job, ack.checkpoint, "saved for the handoff")
        if state.code == OVERRUN:
            self._stop_overrun(job, attempt, "saved a checkpoint")
            return True
        if state.code == "quota":
            _left, resets_at = quota_left_hours(self.deps, provider)
            until = (
                resets_at
                if resets_at is not None and resets_at > now
                else now + self.deps.config.engine.unknown_quota_reset_s
            )
            store.upsert_provider_state(provider, exhausted_until=until)
        latest = store.latest_checkpoint(job.id)
        nxt = (
            f"resuming from checkpoint {latest.seq} elsewhere"
            if latest is not None
            else "no checkpoint yet, so it starts over elsewhere"
        )
        store.transition(
            job.id,
            from_state=job.state,
            to_state=JobState.MIGRATING,
            reason=Reason.HANDOFF,
            actor=ACTOR,
            message=f"handing off before {provider} stops it ({state.why}); {nxt}",
            detail={
                "previous_provider": provider,
                "checkpoint": latest.id if latest is not None else None,
                "why": state.code,
                "request": state.request_id,
            },
        )
        self._handoffs.pop(attempt.id, None)
        return True

    # ---- declared hours (D48)

    def _ran_s(self, job: Job) -> float:
        """Running time of the job so far, over all its attempts."""
        now = self._now
        total = 0.0
        for a in self.deps.store.attempts_for(job.id):
            if a.started_at is not None:
                end = a.ended_at if a.ended_at is not None else now
                total += max(0.0, end - a.started_at)
        return total

    def _overrun_event(self, job: Job) -> JobEvent | None:
        for ev in reversed(self.deps.store.events_for(job.id, limit=10_000)):
            if ev.reason == Reason.HOURS_EXCEEDED:
                return ev
        return None

    def _overrun_pending(self, job: Job) -> ApprovalDecision | None:
        """The approval a job stopped past its declared hours still needs, or None (never
        stopped, or the user approved it after the stop)."""
        ev = self._overrun_event(job)
        if ev is None or (job.approved_at is not None and job.approved_at >= ev.ts):
            return None
        hours = ev.detail.get("hours")
        ran = ev.detail.get("ran_s")
        declared = fmt_duration(float(hours) * 3600) if isinstance(hours, int | float) else "?"
        so_far = f" ({fmt_duration(float(ran))} so far)" if isinstance(ran, int | float) else ""
        return ApprovalDecision(
            required=True,
            reason=f"ran past its declared {declared}{so_far}; approve to let it finish",
            rule="hours_exceeded",
        )

    async def _enforce_hours(self, job: Job, attempt: Attempt) -> bool:
        """An agent's declared hours are what auto-approved its job, so a job that keeps
        running well past them (policy.hours_limit_s) is asked for a checkpoint and then
        moved (-> migrating, reason hours_exceeded), and placed again only after the user
        approves. Returns True when the job moved."""
        if attempt.state is not AttemptState.RUNNING:
            return False
        limit = hours_limit_s(self.deps.policy, job, attempt.provider)
        if limit is None:
            return False
        ran = self._ran_s(job)
        if ran < limit:
            return False
        ev = self._overrun_event(job)
        if ev is not None and job.approved_at is not None and job.approved_at >= ev.ts:
            return False  # the user let it run past its hours
        assert job.spec.hours is not None
        why = (
            f"ran past its declared {fmt_duration(job.spec.hours * 3600)} "
            f"({fmt_duration(ran)} of running; the limit is {fmt_duration(limit)})"
        )
        state = self._handoff_state(job, attempt)
        hub = self.deps.checkpoints
        if state is None and hub is not None:
            kind = self._kind(attempt.provider)
            await self._request_handoff(job, attempt, hub, kind, OVERRUN, why)
            state = self._handoffs.get(attempt.id)
            if state is not None and not state.gave_up:
                return False  # _check_handoff stops it once the job saved a checkpoint
        elif state is not None and not state.gave_up:
            return False  # a checkpoint request is out; _check_handoff finishes it
        self._stop_overrun(job, attempt, "no fresh checkpoint")
        return True

    def _stop_overrun(self, job: Job, attempt: Attempt, saved: str) -> None:
        store = self.deps.store
        hours = job.spec.hours or 0.0
        ran = self._ran_s(job)
        latest = store.latest_checkpoint(job.id)
        nxt = (
            f"approve it to resume from checkpoint {latest.seq}"
            if latest is not None
            else "approve it to run it again (no checkpoint yet, so it starts over)"
        )
        store.transition(
            job.id,
            from_state=job.state,
            to_state=JobState.MIGRATING,
            reason=Reason.HOURS_EXCEEDED,
            actor=ACTOR,
            message=(
                f"ran past its declared {fmt_duration(hours * 3600)} ({fmt_duration(ran)} of "
                f"running, {saved}); stopping it on {attempt.provider} and asking you "
                f"before it runs longer: {nxt}"
            ),
            detail={
                "previous_provider": attempt.provider,
                "hours": hours,
                "ran_s": round(ran, 1),
                "limit_s": hours_limit_s(self.deps.policy, job, attempt.provider),
                "checkpoint": latest.id if latest is not None else None,
            },
        )
        self._handoffs.pop(attempt.id, None)

    # ------------------------------------------------------------------ errors / sleep

    async def _internal_error(self, job: Job, exc: Exception) -> None:
        self._consecutive_errors += 1
        cfg = self.deps.config.engine
        k = self._consecutive_errors
        emit(
            "engine.bug",
            f"{job.short_id} {job.state}: {type(exc).__name__}: {exc}",
            level=logging.ERROR,
            exc_info=True,
            log=_logger,
            job_id=job.id,
            state=str(job.state),
            consecutive=k,
        )
        store = self.deps.store
        try:
            current = store.get_job(job.id)
            if is_terminal(current.state):
                return
            if k >= cfg.internal_error_limit and current.state is not JobState.CANCELLING:
                attempt = store.current_attempt(current)
                change = None
                if attempt is not None and attempt.state in LIVE_ATTEMPT_STATES:
                    change = AttemptChange(
                        attempt.id,
                        AttemptPatch(state=AttemptState.ABANDONED, lost_reason="internal error"),
                    )
                store.transition(
                    job.id,
                    from_state=current.state,
                    to_state=JobState.FAILED,
                    reason=Reason.INTERNAL_ERROR,
                    actor=ACTOR,
                    attempt=change,
                    message=(
                        f"gpu-router hit {k} internal errors in a row "
                        f"({type(exc).__name__}: {exc}); gave up. this is a bug, see "
                        f"the daemon log"
                    ),
                    patch=JobPatch(failure_kind=FailureKind.INTERNAL),
                    detail={"error": type(exc).__name__},
                )
                return
            delay = self._backoff(k)
            store.add_note(
                job.id,
                reason=Reason.INTERNAL_ERROR,
                actor=ACTOR,
                message=(
                    f"internal error ({type(exc).__name__}: {exc}); retrying in "
                    f"{fmt_duration(delay)}"
                ),
                detail={"error": type(exc).__name__, "consecutive": k},
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            emit(
                "engine.bug",
                f"{job.short_id}: recording an internal error failed",
                level=logging.ERROR,
                exc_info=True,
                log=_logger,
                job_id=job.id,
            )
            delay = self._backoff(k)
        await self.sleep(delay)

    async def sleep(self, seconds: float) -> bool:
        """clock.sleep that returns early when wake() is called; clears the wake flag.
        True when it was woken rather than timed out."""
        if self._wake.is_set():
            self._wake.clear()
            return True
        sleeper = asyncio.ensure_future(self.deps.clock.sleep(max(0.0, seconds)))
        waker = asyncio.ensure_future(self._wake.wait())
        try:
            await asyncio.wait({sleeper, waker}, return_when=asyncio.FIRST_COMPLETED)
            return waker.done() and not waker.cancelled()
        finally:
            for fut in (sleeper, waker):
                if not fut.done():
                    fut.cancel()
            self._wake.clear()


@dataclass
class _Extras:
    """Per-attempt additions from checkpoint storage (phase 5)."""

    env: dict[str, str] = field(default_factory=dict)
    secrets: dict[str, SecretStr] = field(default_factory=dict)


@dataclass
class _Handoff:
    """A planned-handoff request sent to one attempt's runner."""

    request_id: str
    requested_at: float
    code: str  # session_cap | quota
    why: str
    wait_s: float
    gave_up: bool = False


#: a dataset this big gets a note before stage_data runs (the upload can take minutes)
STAGE_NOTE_BYTES = 50 * 1024**2


def _size(n: int) -> str:
    from gpu_router.packaging.files import human_bytes

    return human_bytes(n)


class _StagePending(StorageError):
    """A dataset is still on its way to the provider's own store (stage_data was slow or
    the provider answered "try later"): wait and retry like storage trouble."""

    def __init__(self, message: str) -> None:
        super().__init__(message, retryable=True)


class _DataProblem(Exception):
    """A dataset cannot be made available to this attempt. `reroute`: another provider
    may manage (exclude this one); else the job cannot run anywhere as it is."""

    def __init__(
        self,
        message: str,
        *,
        reroute: bool,
        hint: str | None = None,
        error: AdapterError | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.reroute = reroute
        self.hint = hint
        self.error = error  # a definitive submit error to apply as-is (else InvalidJob)


def _mb(n: int) -> str:
    if n >= 1024**3:
        return f"{n / 1024**3:.1f} GB"
    return f"{n / 1024**2:.1f} MB"


class _MissingSecret(Exception):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.name = name


class _ReservedSecret(_MissingSecret):
    """A job's `secrets:` names one of gpu-router's own credentials (D48)."""


class _NoLookup(Exception):
    """The provider cannot look runs up by attempt key (or is no longer registered)."""


class _ProvisionTimeout(AdapterError):
    """Internal: a pending run exceeded provision_timeout_s (cooled down like Unavailable)."""
