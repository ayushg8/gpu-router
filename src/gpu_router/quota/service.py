"""QuotaService (phase 5): the daemon's quota cache. Serves ledger views from the store and
keeps live readings fresh in the background, so neither routing nor `gpu quota` ever waits
on a slow provider for long.

  * `views()` / `snapshots()`: ledger views (sync, event-loop thread, no adapter calls).
  * `refresh(wait_s=...)`: for every provider whose adapter reports live quota and whose
    latest live reading is older than `ttl_s`, start ONE `quota()` call (joined if already
    in flight), record the result as a quota snapshot, then wait at most `wait_s` for them.
    Waiting never cancels a call; a slow provider's answer lands in the cache later.
    Providers that are disabled or need a login are skipped; a failed call is not retried
    for `retry_failed_s`.
  * `run()`: background loop, refresh every `refresh_s` (started by DaemonRuntime), on
    the injected clock. A reading counts as due `early_s` before it leaves the TTL, so the
    refresh lands before the ledger falls back to an estimate.

Unlimited providers (`reset: none`, the local Mac) are never polled. Every reading is one
quota_snapshots row (about 48 a day per live provider at the default 30 min); the Store
keeps the newest `QUOTA_SNAPSHOTS_KEEP` (500) rows per provider.

Invariant 9: the store is touched only from the event-loop thread; adapter calls run in
AdapterCaller's worker threads.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import TYPE_CHECKING

from gpu_router.engine._obs import emit
from gpu_router.models import ProviderHealth, QuotaSnapshot
from gpu_router.quota.ledger import ledger_views
from gpu_router.quota.settings import QuotaSettings

if TYPE_CHECKING:
    from collections.abc import Iterable

    from gpu_router.adapters.registry import AdapterRegistry
    from gpu_router.clock import Clock
    from gpu_router.engine.calls import AdapterCaller
    from gpu_router.store import Store

__all__ = ["QuotaService"]

_logger = logging.getLogger("gpu_router.quota")
_SKIP_HEALTH = frozenset({ProviderHealth.DISABLED, ProviderHealth.AUTH_REQUIRED})


class QuotaService:
    def __init__(
        self,
        *,
        store: Store,
        registry: AdapterRegistry,
        caller: AdapterCaller,
        clock: Clock,
        settings: QuotaSettings | None = None,
    ) -> None:
        self.store = store
        self.registry = registry
        self.caller = caller
        self.clock = clock
        self.settings = settings or QuotaSettings()
        self._inflight: dict[str, asyncio.Task[None]] = {}
        self._failed_at: dict[str, float] = {}
        self._loop_task: asyncio.Task[None] | None = None
        self._closed = False

    # ------------------------------------------------------------------ views

    def views(self) -> dict[str, QuotaSnapshot]:
        return ledger_views(self.store, self.registry, self.clock.now(), self.settings)

    def snapshots(self) -> list[QuotaSnapshot]:
        """Ledger views in registry order (the body of GET /v1/quota)."""
        views = self.views()
        return [views[n] for n in self.registry.names() if n in views]

    # ------------------------------------------------------------------ refresh

    @property
    def early_s(self) -> float:
        """Refresh this long before a reading leaves the TTL (D44): a reading is stamped
        after the provider's call returns, so a refresh exactly every ttl_s would find it
        a few seconds short of stale, skip it, and leave the ledger on an estimate for
        the next half hour (readings every 60 min instead of 30)."""
        return min(self.settings.ttl_s / 2, max(60.0, self.settings.refresh_s * 0.1))

    def stale(self, *, force: bool = False) -> list[str]:
        """Providers whose live reading should be refreshed now."""
        now = self.clock.now()
        latest = self.store.latest_quota_snapshots()
        states = self.store.all_provider_states()
        out: list[str] = []
        for name in self.registry.names():
            if not self.registry.get(name).capabilities.live_quota:
                continue
            if self.registry.entry(name).quota.reset == "none":
                continue  # unlimited (local): a reading teaches nothing, skip the row
            state = states.get(name)
            if state is not None and state.health in _SKIP_HEALTH:
                continue
            if force:
                out.append(name)
                continue
            failed = self._failed_at.get(name)
            if failed is not None and now - failed < self.settings.retry_failed_s:
                continue
            snap = latest.get(name)
            fresh = (
                snap is not None
                and snap.source == "live"
                and 0 <= now - snap.observed_at < self.settings.ttl_s - self.early_s
                and (snap.resets_at is None or snap.resets_at > now)
            )
            if not fresh:
                out.append(name)
        return out

    async def refresh(
        self,
        names: Iterable[str] | None = None,
        *,
        force: bool = False,
        wait_s: float | None = None,
    ) -> None:
        """Start live quota calls for `names` (default: the stale ones); wait up to
        `wait_s` seconds (None = do not wait, 0 = do not wait either)."""
        if self._closed:
            return
        targets = list(names) if names is not None else self.stale(force=force)
        tasks = [self._start(n) for n in targets if n in self.registry]
        if tasks and wait_s:
            await asyncio.wait(tasks, timeout=wait_s)

    def _start(self, name: str) -> asyncio.Task[None]:
        task = self._inflight.get(name)
        if task is not None and not task.done():
            return task
        task = asyncio.get_running_loop().create_task(self._one(name), name=f"quota-{name}")
        self._inflight[name] = task
        return task

    async def _one(self, name: str) -> None:
        try:
            snap = await self.caller.quota(name)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._failed_at[name] = self.clock.now()
            emit(
                "quota.refresh",
                f"{name}: live quota check failed ({type(exc).__name__}: {exc}); "
                f"using the estimate from job history",
                level=logging.WARNING,
                log=_logger,
                provider=name,
            )
            return
        if self._closed:
            return
        self._failed_at.pop(name, None)
        try:
            if snap.provider != name:
                snap = snap.model_copy(update={"provider": name})
            self.store.record_quota_snapshot(snap)
        except Exception as exc:
            emit(
                "quota.refresh",
                f"{name}: could not record the quota reading: {exc}",
                level=logging.WARNING,
                log=_logger,
                provider=name,
            )

    # ------------------------------------------------------------------ lifecycle

    async def run(self) -> None:
        while not self._closed:
            try:
                await self.refresh()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # the loop must survive anything
                emit(
                    "quota.refresh",
                    f"quota refresh loop error: {exc}",
                    level=logging.WARNING,
                    log=_logger,
                )
            await self.clock.sleep(self.settings.refresh_s)

    def start(self) -> None:
        """Start the background loop on the running event loop (idempotent)."""
        if self._loop_task is None and not self._closed:
            self._loop_task = asyncio.get_running_loop().create_task(
                self.run(), name="quota-refresh"
            )

    async def close(self) -> None:
        """Stop the loop and abandon in-flight calls (their worker threads finish alone)."""
        self._closed = True
        tasks = [t for t in (self._loop_task, *self._inflight.values()) if t is not None]
        for t in tasks:
            t.cancel()
        for t in tasks:
            with contextlib.suppress(BaseException):
                await t
        self._loop_task = None
        self._inflight.clear()
