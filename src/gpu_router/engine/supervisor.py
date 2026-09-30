"""Supervisor: owns every JobDriver and the provider health loop (phase 1; owner: group C).

Startup (`start()`), in order:
1. `ready = False` (API mutations answer 503 not_ready).
2. For every job from store.non_terminal_jobs() (oldest first): add note `recovered`
   (actor "recovery", detail {"action": RecoveryAction}) and start a JobDriver with
   recovery=statemachine.RECOVERY[state]. No adapter call happens before its driver runs.
3. Start the provider health loop: healthcheck every registered provider now, then
   re-check unhealthy ones on a backoff (config.engine.health_recheck_min_s after the first
   failure, doubling up to health_recheck_s; a healthy answer resets it); write
   store.upsert_provider_state(...) and log `provider.health`; on a transition back to OK,
   wake drivers of queued jobs. The loop ticks every health_tick_s; a tick that finds wall
   time far ahead of the loop's own clock means the Mac slept (asyncio's clock stops while
   macOS sleeps, D48): it waits wake_grace_s for the network, re-checks every provider and
   wakes queued drivers.
4. `ready = True`; log `daemon.recovery` with counts per action.

User actions are methods here (the API calls them on the event loop). Each resolves the
ref with store.resolve_ref, performs one store write, wakes the driver, and returns the
updated Job. They never call adapters directly (the driver does the remote work); the one
exception is `request_fetch`, which re-downloads outputs of a finished job in a background
task because no driver exists for terminal jobs.

Shutdown (`stop()`): cancel driver tasks, wait up to config.daemon.shutdown_grace_s, shut
the AdapterCaller down without waiting for stuck threads, close adapters. Remote runs are
never cancelled by a shutdown (invariant 11).
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import math
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

from gpu_router.adapters.base import Capabilities, Health
from gpu_router.api import ProviderView
from gpu_router.engine._obs import emit, fmt_duration
from gpu_router.engine.backoff import backoff_s
from gpu_router.engine.context import build_routing_context, quota_views
from gpu_router.engine.driver import ADAPTER_FAILURES, JobDriver, remote_ref
from gpu_router.errors import InvalidSpec, InvalidTransition, StaleState
from gpu_router.models import (
    Job,
    JobPatch,
    JobSpec,
    ProviderHealth,
    ProviderState,
    QuotaSnapshot,
    secret_env_problem,
    secret_names_problem,
)
from gpu_router.router.base import JobEstimate, RouteDecision, RoutingContext
from gpu_router.statemachine import (
    LIVE_ATTEMPT_STATES,
    RECOVERY,
    AttemptState,
    JobState,
    Reason,
    RecoveryAction,
    cancel_target,
    is_terminal,
)

if TYPE_CHECKING:
    from gpu_router.engine.deps import EngineDeps

_logger = logging.getLogger("gpu_router.engine.supervisor")

DRY_RUN_JOB_ID = "000000000000"

#: The health loop never sleeps less than this between turns (no busy loop).
MIN_HEALTH_SLEEP_S = 1.0
#: Wall time that ran ahead of the loop's own clock by more than this means the Mac slept.
WAKE_SLACK_S = 60.0
#: Health states whose view reason says when the health loop looks again.
_RECHECK_NOTE_HEALTH = frozenset(
    {ProviderHealth.UNAVAILABLE, ProviderHealth.AUTH_REQUIRED, ProviderHealth.DEGRADED}
)

_RECOVERY_TEXT: dict[RecoveryAction, str] = {
    RecoveryAction.RESUME: "resuming the queue wait",
    RecoveryAction.REROUTE: "choosing a provider again",
    RecoveryAction.WAIT: "still waiting for approval",
    RecoveryAction.RESOLVE_ATTEMPT: "checking whether the last submit reached the provider",
    RecoveryAction.REATTACH: "reattaching to the remote run",
    RecoveryAction.REMIGRATE: "continuing the migration",
    RecoveryAction.RECANCEL: "finishing the cancel",
    RecoveryAction.NONE: "nothing to do",
}


class Supervisor:
    def __init__(self, deps: EngineDeps) -> None:
        self.deps = deps
        self.ready = False
        self._drivers: dict[str, JobDriver] = {}
        self._health_task: asyncio.Task[None] | None = None
        self._background: set[asyncio.Task[None]] = set()
        self._stopping = False
        # health loop bookkeeping (in memory: a restarted daemon checks everything anyway)
        self._health_failures: dict[str, int] = {}  # consecutive failed healthchecks
        self._health_next: dict[str, float] = {}  # wall time of the next re-check

    # ------------------------------------------------------------------ lifecycle

    async def start(self) -> None:
        self.ready = False
        self._stopping = False
        store = self.deps.store
        counts: Counter[str] = Counter()
        for job in store.non_terminal_jobs():
            action = RECOVERY[job.state]
            counts[str(action)] += 1
            store.add_note(
                job.id,
                reason=Reason.RECOVERED,
                actor="recovery",
                message=f"daemon restarted; {_RECOVERY_TEXT[action]}",
                detail={"action": str(action)},
            )
            self._start_driver(job.id, recovery=action)
        self._health_task = asyncio.get_running_loop().create_task(
            self._health_loop(), name="provider-health"
        )
        self.ready = True
        emit(
            "daemon.recovery",
            f"recovered {sum(counts.values())} job(s)"
            + (": " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())) if counts else ""),
            log=_logger,
            counts=dict(counts),
        )

    async def stop(self) -> None:
        self._stopping = True
        self.ready = False
        tasks: list[asyncio.Task[None]] = []
        if self._health_task is not None:
            self._health_task.cancel()
            tasks.append(self._health_task)
        for task in self._background:
            task.cancel()
            tasks.append(task)
        for driver in self._drivers.values():
            if driver.task is not None and not driver.task.done():
                driver.task.cancel()
                tasks.append(driver.task)
        if tasks:
            await asyncio.wait(tasks, timeout=self.deps.config.daemon.shutdown_grace_s)
        self._drivers.clear()
        self._background.clear()
        self._health_task = None
        self.deps.caller.shutdown(wait=False)
        with contextlib.suppress(Exception):
            self.deps.registry.close()

    def driver(self, job_id: str) -> JobDriver | None:
        return self._drivers.get(job_id)

    def _start_driver(self, job_id: str, *, recovery: RecoveryAction | None = None) -> JobDriver:
        existing = self._drivers.get(job_id)
        if existing is not None and not existing.done:
            return existing
        driver = JobDriver(
            job_id, self.deps, recovery=recovery, on_attempt_ended=self._attempt_ended
        )
        task = driver.start()

        def _done(t: asyncio.Task[None], jid: str = job_id, d: JobDriver = driver) -> None:
            if self._drivers.get(jid) is d:
                del self._drivers[jid]
            if not t.cancelled() and t.exception() is not None:
                emit(
                    "engine.bug",
                    f"driver for {jid} crashed: {t.exception()!r}",
                    level=logging.ERROR,
                    log=_logger,
                    job_id=jid,
                )

        task.add_done_callback(_done)
        self._drivers[job_id] = driver
        return driver

    def _wake(self, job_id: str) -> None:
        driver = self._drivers.get(job_id)
        if driver is not None:
            driver.wake()
        elif not self._stopping:
            with contextlib.suppress(Exception):
                job = self.deps.store.get_job(job_id)
                if not is_terminal(job.state):
                    self._start_driver(job_id)

    def wake_all(self) -> None:
        for driver in list(self._drivers.values()):
            driver.wake()

    def _attempt_ended(self, job_id: str) -> None:
        """A live attempt ended (its provider has a free slot): wake the drivers of queued
        jobs so a job waiting for capacity routes now, not at its next backoff (D44).
        Drivers polling a remote run are left alone (no extra provider calls)."""
        self._wake_queued(skip=job_id)

    def _wake_queued(self, *, skip: str | None = None) -> None:
        for jid, driver in list(self._drivers.items()):
            if jid == skip:
                continue
            with contextlib.suppress(Exception):
                if self.deps.store.get_job(jid).state is JobState.QUEUED:
                    driver.wake()

    # ------------------------------------------------------------------ provider health

    async def _health_loop(self) -> None:
        """Healthcheck every provider now; then re-check unhealthy ones on their backoff,
        and every provider once the Mac is back from sleep."""
        cfg = self.deps.config.engine
        clock = self.deps.clock
        await self._check_many(self.deps.registry.names())
        offset = clock.now() - clock.monotonic()
        while True:
            try:
                delay = self._health_sleep_s()
            except Exception:
                self._health_bug("health loop bookkeeping failed")
                delay = cfg.health_tick_s
            before = clock.now()
            await clock.sleep(delay)
            now = clock.now()
            # Wall time that passed without the loop's clock: the Mac slept, either during
            # this sleep (it overran) or during the previous turn's checks (the monotonic
            # clock stopped while wall time went on).
            overrun = now - before - delay
            drift = (now - clock.monotonic()) - offset
            offset = now - clock.monotonic()
            try:
                if now - before > 2 * delay + WAKE_SLACK_S or drift > WAKE_SLACK_S:
                    await self._after_wake(max(overrun, drift))
                    offset = clock.now() - clock.monotonic()
                    continue
                await self._check_many(n for n in self.deps.registry.names() if self._due(n, now))
            except asyncio.CancelledError:
                raise
            except Exception:
                self._health_bug("health loop turn failed")

    async def _after_wake(self, asleep_s: float) -> None:
        """The Mac woke up: provider health from before the sleep says nothing about now,
        and a check made the moment the lid opens usually fails on a network that is not
        back yet. Give it wake_grace_s, then check every provider with a fresh backoff and
        let queued jobs route again (their sleeps ran on the stopped clock too)."""
        cfg = self.deps.config.engine
        grace = cfg.wake_grace_s
        emit(
            "daemon.wake",
            f"woke from sleep after about {fmt_duration(asleep_s)}; re-checking every "
            f"provider in {fmt_duration(grace)}, once the network is back",
            log=_logger,
            asleep_s=round(asleep_s),
            grace_s=grace,
        )
        self._health_failures.clear()
        if grace > 0:
            await self.deps.clock.sleep(grace)
        await self._check_many(self.deps.registry.names())
        self._wake_queued()

    def _health_bug(self, msg: str, **fields: object) -> None:
        emit("engine.bug", msg, level=logging.ERROR, exc_info=True, log=_logger, **fields)

    async def _check_many(self, names: Iterable[str]) -> None:
        """Healthcheck these providers concurrently (each call is bounded by the caller's
        timeout, so one slow CLI no longer delays the others)."""

        async def one(name: str) -> None:
            try:
                await self._check(name)
            except asyncio.CancelledError:
                raise
            except Exception:
                self._health_bug(f"healthcheck bookkeeping for {name} failed", provider=name)

        await asyncio.gather(*(one(name) for name in names))

    def _recheck_delay(self, failures: int) -> float:
        cfg = self.deps.config.engine
        return backoff_s(
            max(1, failures),
            base_s=min(cfg.health_recheck_min_s, cfg.health_recheck_s),
            cap_s=cfg.health_recheck_s,
        )

    def _recheck_at(self, name: str, state: ProviderState) -> float:
        """Wall time of the health loop's next look at an unhealthy provider. A provider a
        driver marked unhealthy (login needed) has no schedule yet: its first re-check is
        one short backoff after that; never checked at all = now."""
        at = self._health_next.get(name)
        if at is not None:
            return at
        if state.last_healthcheck_at is None:
            return -math.inf
        return state.last_healthcheck_at + self._recheck_delay(self._health_failures.get(name, 0))

    def _due(self, name: str, now: float) -> bool:
        state = self.deps.store.get_provider_state(name)
        return state.health is not ProviderHealth.OK and now >= self._recheck_at(name, state)

    def _health_sleep_s(self) -> float:
        """Until the next re-check is due, at most one tick (the tick is what notices a
        wake from sleep), at least MIN_HEALTH_SLEEP_S."""
        now = self.deps.clock.now()
        wait = self.deps.config.engine.health_tick_s
        for name in self.deps.registry.names():
            state = self.deps.store.get_provider_state(name)
            if state.health is not ProviderHealth.OK:
                wait = min(wait, self._recheck_at(name, state) - now)
        return max(MIN_HEALTH_SLEEP_S, wait)

    def _schedule(self, name: str, health: ProviderHealth, at: float) -> float | None:
        """Record a healthcheck outcome; return the delay until the next re-check (None =
        healthy, not re-checked until something fails)."""
        if health is ProviderHealth.OK:
            self._health_failures.pop(name, None)
            self._health_next.pop(name, None)
            return None
        failures = self._health_failures.get(name, 0) + 1
        self._health_failures[name] = failures
        delay = self._recheck_delay(failures)
        self._health_next[name] = at + delay
        return delay

    async def _check(self, name: str) -> Health:
        now = self.deps.clock.now()
        try:
            health = await self.deps.caller.healthcheck(name)
        except ADAPTER_FAILURES as exc:
            # incl. the caller's own timeout: Unavailable("<p> healthcheck timed out after Ns")
            health = Health(
                health=ProviderHealth.UNAVAILABLE,
                reason=getattr(exc, "message", str(exc)),
                checked_at=now,
            )
        before = self.deps.store.get_provider_state(name).health
        self.deps.store.upsert_provider_state(
            name,
            health=health.health,
            health_reason=health.reason,
            last_healthcheck_at=health.checked_at or now,
        )
        # a check can take up to its timeout: the backoff counts from its end
        next_in = self._schedule(name, health.health, self.deps.clock.now())
        what = f"{name}: {before} -> {health.health}" if before is not health.health else None
        if what is None and next_in is not None:
            what = f"{name} is still {health.health}"
        if what is not None:
            emit(
                "provider.health",
                what
                + (f" ({health.reason})" if health.reason else "")
                + (f"; re-checking in {fmt_duration(next_in)}" if next_in is not None else ""),
                level=logging.INFO if before is not health.health else logging.DEBUG,
                log=_logger,
                provider=name,
                health=str(health.health),
                previous=str(before),
                reason=health.reason,
                next_check_s=next_in,
            )
        if before is not health.health and health.health is ProviderHealth.OK:
            self.wake_all()
        return health

    def _health_note(
        self, name: str, state: ProviderState, now: float
    ) -> tuple[str | None, float | None]:
        """The view's reason for an unhealthy provider says when it is looked at again
        (computed now, so it never goes stale), plus that time for --json readers."""
        reason = state.health_reason
        if state.health not in _RECHECK_NOTE_HEALTH:
            return reason, None
        at = self._recheck_at(name, state)
        if not math.isfinite(at):
            return reason, None
        when = f"in {fmt_duration(at - now)}" if at - now >= 1 else "now"
        note = f"re-checking {when}"
        return (f"{reason}; {note}" if reason else note), at

    # ------------------------------------------------------------------ user actions

    async def submit(
        self, spec: JobSpec, *, actor: str, request_id: str | None = None
    ) -> tuple[Job, bool]:
        """Build the bundle (deps.bundler, phase 2), store.create_job (idempotent on
        request_id), record jobs.bundle_sha256 + materialize jobs/<id>/bundle, start the
        driver. Returns (job, created)."""
        _check_new_spec(spec)
        bundle = None
        bundler = self.deps.bundler
        if bundler is not None:
            # Blocking git + gzip work off the loop; raises BundleError (400 invalid_spec)
            # before any DB write, so a bad project never becomes a job.
            bundle = await asyncio.to_thread(bundler.prepare, spec)
        job, created = self.deps.store.create_job(spec, actor=actor, request_id=request_id)
        if created and bundle is not None and bundler is not None:
            job = self.deps.store.update_job(job.id, JobPatch(bundle_sha256=bundle.sha256))
            # Before the driver starts, so the first submit already sees jobs/<id>/bundle.
            # A crash right here is healed by the driver (it materializes from the cache).
            try:
                await asyncio.to_thread(bundler.materialize, bundle.sha256, job.id)
            except Exception as exc:
                # The job exists now: never leave it without a driver. The driver retries
                # materializing before the first submit (and fails the job after
                # internal_error_limit if it keeps failing, e.g. a corrupt cache entry).
                emit(
                    "job.bundle",
                    f"could not unpack the bundle yet ({type(exc).__name__}: {exc}); "
                    "the job driver retries before submitting",
                    level=logging.WARNING,
                    log=_logger,
                    job_id=job.id,
                    bundle_sha256=bundle.sha256,
                )
            emit(
                "job.bundle",
                f"bundled {bundle.file_count} file(s), {bundle.size_bytes} bytes"
                + (" (cached)" if bundle.cached else ""),
                log=_logger,
                job_id=job.id,
                bundle_sha256=bundle.sha256,
                deps=bundle.deps.kind,
                vram_gb_estimate=bundle.estimate.vram_gb,
                hours_estimate=bundle.estimate.hours,
                warnings=bundle.warnings,
            )
        if created or (not is_terminal(job.state) and job.id not in self._drivers):
            self._start_driver(job.id)
        return job, created

    async def cancel(self, ref: str, *, actor: str) -> Job:
        """statemachine.cancel_target -> transition (reason user_cancel, sets
        cancel_requested_at) or no-op if terminal/cancelling. Idempotent."""
        store = self.deps.store
        for _ in range(5):
            job = store.resolve_ref(ref)
            attempt = store.current_attempt(job)
            live = attempt is not None and attempt.state in LIVE_ATTEMPT_STATES
            target = cancel_target(job.state, has_live_attempt=live)
            if target is None:
                return job
            if target is JobState.CANCELLING:
                message = (
                    f"cancel requested by you; stopping the run on "
                    f"{attempt.provider if attempt else job.provider}"
                )
            else:
                message = "cancelled by you before anything ran"
            try:
                job = store.transition(
                    job.id,
                    from_state=job.state,
                    to_state=target,
                    reason=Reason.USER_CANCEL,
                    message=message,
                    actor=actor,
                    patch=JobPatch(cancel_requested_at=self.deps.clock.now()),
                )
            except StaleState:
                continue
            self._wake(job.id)
            return job
        return store.resolve_ref(ref)

    async def approve(self, ref: str, *, actor: str, reason: str | None = None) -> Job:
        """store.record_approval + wake. InvalidTransition unless awaiting_approval."""
        store = self.deps.store
        job = store.resolve_ref(ref)
        job = store.record_approval(job.id, actor=actor)
        if reason:
            store.add_note(
                job.id,
                reason=Reason.APPROVED,
                actor=actor,
                message=f"approval note: {reason}",
                detail={"reason": reason},
            )
            job = store.get_job(job.id)
        self._wake(job.id)
        return job

    async def deny(self, ref: str, *, actor: str, reason: str | None = None) -> Job:
        """awaiting_approval -> denied (reason denied)."""
        store = self.deps.store
        job = store.resolve_ref(ref)
        why = f": {reason}" if reason else ""
        job = store.transition(
            job.id,
            from_state=job.state,
            to_state=JobState.DENIED,
            reason=Reason.DENIED,
            message=f"denied by you{why}; nothing ran",
            actor=actor,
            detail={"reason": reason} if reason else None,
        )
        self._wake(job.id)
        return job

    async def request_fetch(self, ref: str, *, actor: str) -> Job:
        """Terminal job with a succeeded attempt: note fetch_requested and re-fetch outputs
        in a background task (note fetched / fetch_failed). Otherwise InvalidTransition."""
        store = self.deps.store
        job = store.resolve_ref(ref)
        attempt = next(
            (
                a
                for a in reversed(store.attempts_for(job.id))
                if a.state is AttemptState.SUCCEEDED and a.remote_id
            ),
            None,
        )
        if not is_terminal(job.state) or attempt is None:
            raise InvalidTransition(
                str(job.state),
                "fetch",
                hint="outputs can be fetched once the job has finished successfully",
            )
        store.add_note(
            job.id,
            reason=Reason.FETCH_REQUESTED,
            actor=actor,
            attempt_id=attempt.id,
            message=f"fetching outputs from {attempt.provider} again",
        )
        task = asyncio.get_running_loop().create_task(
            self._refetch(job.id, attempt.id), name=f"fetch-{job.id[:4]}"
        )
        self._background.add(task)
        task.add_done_callback(self._background.discard)
        return store.get_job(job.id)

    async def _refetch(self, job_id: str, attempt_id: str) -> None:
        store = self.deps.store
        job = store.get_job(job_id)
        attempt = store.get_attempt(attempt_id)
        if job.outputs_dir is None:
            return
        dest = Path(job.outputs_dir)
        try:
            res = await self.deps.caller.fetch(attempt.provider, remote_ref(attempt), dest)
        except ADAPTER_FAILURES as exc:
            store.add_note(
                job_id,
                reason=Reason.FETCH_FAILED,
                actor="engine",
                attempt_id=attempt_id,
                message=(
                    f"could not fetch outputs from {attempt.provider} "
                    f"({getattr(exc, 'message', exc)}); try again later"
                ),
                detail={"error": type(exc).__name__},
            )
            return
        store.add_note(
            job_id,
            reason=Reason.FETCHED,
            actor="engine",
            attempt_id=attempt_id,
            message=f"outputs saved to {dest} ({res.files} files)",
            detail={"files": res.files, "bytes": res.bytes, "partial": res.partial},
            patch=JobPatch(outputs_fetched=True),
        )

    # ------------------------------------------------------------------ queries

    def routing_context(self, job: Job, *, estimate: JobEstimate | None = None) -> RoutingContext:
        """Assemble the router input from registry, store (provider_state, live attempts,
        quota ledger, exclusions) and clock. Used by drivers and dry runs."""
        return build_routing_context(
            self.deps, job, persisted=job.id != DRY_RUN_JOB_ID, estimate=estimate
        )

    def dry_route(self, spec: JobSpec, *, estimate: JobEstimate | None = None) -> RouteDecision:
        """Route a spec without persisting anything (POST /v1/route). Builds a transient Job
        (id "000000000000", state queued) around the spec. `estimate` (phase 5) is the
        project's VRAM/runtime estimate, computed by the caller off the event loop."""
        _check_new_spec(spec)
        now = self.deps.clock.now()
        job = Job(
            id=DRY_RUN_JOB_ID,
            short_id=DRY_RUN_JOB_ID[:4],
            name=spec.display_name(),
            state=JobState.QUEUED,
            source=spec.source,
            project_dir=spec.project_dir,
            spec=spec,
            spec_hash=hashlib.sha256(spec.model_dump_json().encode()).hexdigest(),
            created_at=now,
            updated_at=now,
        )
        return self.deps.router.route(self.routing_context(job, estimate=estimate))

    def provider_views(self) -> list[ProviderView]:
        registry = self.deps.registry
        store = self.deps.store
        states = store.all_provider_states()
        quotas = quota_views(self.deps)  # phase 5: the quota ledger's view
        live = store.live_attempts_by_provider()
        now = self.deps.clock.now()
        views: list[ProviderView] = []
        for entry in registry.catalog.ordered():
            if entry.test_only and not self.deps.config.test_mode:
                continue
            if entry.lane != "gpu":
                # phase 7b: manual-only and verify-at-signup entries are not providers this
                # daemon runs; clients list them from the catalog (`gpu providers`)
                continue
            enabled = entry.name in registry
            state = states.get(entry.name)
            reason = state.health_reason if state else ("not enabled" if not enabled else None)
            next_check: float | None = None
            if enabled:
                caps = registry.get(entry.name).capabilities
                health = state.health if state else ProviderHealth.UNKNOWN
                if state is not None:
                    reason, next_check = self._health_note(entry.name, state, now)
            else:
                caps = Capabilities(
                    max_session_hours=entry.session_hours,
                    max_concurrency=entry.max_concurrency,
                    poll_interval_s=entry.poll_interval_s,
                )
                health = ProviderHealth.DISABLED
            views.append(
                ProviderView(
                    name=entry.name,
                    display_name=entry.display_name,
                    kind=entry.kind,
                    enabled=enabled,
                    health=health,
                    health_reason=reason,
                    state=state,
                    next_healthcheck_at=next_check,
                    capabilities=caps,
                    gpus=[g.label for g in entry.gpus],
                    session_hours=entry.session_hours,
                    live_attempts=live.get(entry.name, 0),
                    quota=quotas.get(entry.name),
                )
            )
        return views

    def provider_view(self, name: str) -> ProviderView:
        for view in self.provider_views():
            if view.name == name:
                return view
        self.deps.registry.catalog.get(name)  # ProviderNotFound for unknown names
        from gpu_router.errors import ProviderNotFound

        raise ProviderNotFound(
            f"provider {name!r} is test-only",
            hint="set GPU_ROUTER_TEST_MODE=1 to use the fake providers",
        )

    async def healthcheck(self, provider: str) -> ProviderView:
        """Run one healthcheck now, persist it, return the view."""
        self.deps.registry.get(provider)  # ProviderNotFound when not enabled
        await self._check(provider)
        return self.provider_view(provider)

    async def quotas(self) -> list[QuotaSnapshot]:
        """Live quota from every enabled provider (recorded as snapshots); on failure the
        latest stored snapshot is returned instead."""
        store = self.deps.store
        names = self.deps.registry.names()

        async def one(name: str) -> QuotaSnapshot | None:
            try:
                return await self.deps.caller.quota(name)
            except ADAPTER_FAILURES:
                return None

        results = await asyncio.gather(*(one(n) for n in names))
        stored = store.latest_quota_snapshots()
        out: list[QuotaSnapshot] = []
        for name, snap in zip(names, results, strict=True):
            if snap is not None:
                with contextlib.suppress(Exception):
                    store.record_quota_snapshot(snap)
                out.append(snap)
            elif name in stored:
                out.append(stored[name])
        return out


def _check_new_spec(spec: JobSpec) -> None:
    """Intake rules a stored spec is not re-checked against (D39): a secret-looking env
    name or value, or a `secrets:` name that is one of gpu-router's own credentials (D48),
    is a 400 invalid_spec before anything is bundled or stored."""
    problem = secret_env_problem(dict(spec.env))
    if problem is not None:
        raise InvalidSpec(problem, detail={"field": "env"})
    problem = secret_names_problem(list(spec.secrets))
    if problem is not None:
        raise InvalidSpec(problem, detail={"field": "secrets"})
