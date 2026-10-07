"""AdapterCaller: the only way the engine invokes an adapter (phase 1; owner: group C).

Invariant 9: adapter methods block, so they run in a ThreadPoolExecutor
(config.engine.max_workers) and never on the event loop; the Store is never touched from
those threads. Per provider, an asyncio.Semaphore(config.engine.per_provider_concurrency)
bounds concurrent calls.

Every call:
1. awaits the provider semaphore, then `loop.run_in_executor(pool, fn)`;
2. waits at most config.engine.timeouts.<op>; on timeout raises
   errors.Unavailable("<provider> <op> timed out after Ns") and logs `adapter.timeout`
   (the worker thread cannot be killed; it is left to finish and its result discarded).
   Exception: a timed-out SUBMIT is an orphan that may still create a remote run, so its
   future is kept per attempt key (`orphan_submit`) and the provider semaphore stays held
   until the thread returns. The driver must not conclude "never submitted" from a lookup
   while the orphan is still running (invariant 6), and adopts or cancels its result;
3. passes AdapterError subclasses through unchanged;
4. wraps any other exception, and a wrong return type, in AdapterContractViolation and logs
   `adapter.bug` with the traceback (invariant 7);
5. logs `adapter.call` at DEBUG with provider, op, duration_ms, outcome.

`logs()` returns an iterator: the caller drains it inside the worker thread (bounded by the
logs timeout and `max_lines`) and returns a list of chunks, so iteration never happens on
the event loop.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Executor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from gpu_router.adapters.base import (
    AttemptContext,
    FetchResult,
    Health,
    LogChunk,
    RemoteRef,
    RemoteStatus,
    StagedData,
)
from gpu_router.engine._obs import emit
from gpu_router.errors import AdapterContractViolation, AdapterError, Unavailable
from gpu_router.models import Job, QuotaSnapshot

if TYPE_CHECKING:
    from gpu_router.adapters.registry import AdapterRegistry
    from gpu_router.config import EngineConfig

T = TypeVar("T")

_logger = logging.getLogger("gpu_router.engine.calls")


class _WrongType(Exception):
    def __init__(self, expected: str, got: object) -> None:
        super().__init__(f"returned {type(got).__name__}, expected {expected}")


def _expect(value: object, kind: type[Any] | tuple[type[Any], ...], label: str) -> None:
    if not isinstance(value, kind):
        raise _WrongType(label, value)


@dataclass(frozen=True, slots=True)
class OrphanSubmit:
    """A submit that outlived its timeout. `done` False: the worker is still running."""

    provider: str
    done: bool
    ref: RemoteRef | None = None
    error: Exception | None = None


def _retrieve(fut: asyncio.Future[Any]) -> None:
    """Done-callback: mark a discarded future's exception as retrieved (no loop warning)."""
    if not fut.cancelled():
        fut.exception()


class AdapterCaller:
    def __init__(
        self, registry: AdapterRegistry, config: EngineConfig, *, executor: Executor | None = None
    ) -> None:
        self.registry = registry
        self.config = config
        self._executor: Executor | None = executor  # created lazily if None
        self._owns_executor = executor is None
        self._semaphores: dict[str, asyncio.Semaphore] = {}
        self._closed = False
        # attempt key -> (provider, future of a submit that outlived its timeout)
        self._orphans: dict[str, tuple[str, asyncio.Future[Any]]] = {}

    # ------------------------------------------------------------------ plumbing

    def _pool(self) -> Executor:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=self.config.max_workers, thread_name_prefix="gpu-adapter"
            )
        return self._executor

    def _semaphore(self, provider: str) -> asyncio.Semaphore:
        sem = self._semaphores.get(provider)
        if sem is None:
            sem = asyncio.Semaphore(self.config.per_provider_concurrency)
            self._semaphores[provider] = sem
        return sem

    async def _call(
        self,
        provider: str,
        op: str,
        fn: Callable[[], T],
        *,
        timeout_s: float,
        orphan_key: str | None = None,
    ) -> T:
        if self._closed:
            raise Unavailable(
                f"{provider} {op} skipped: the daemon is shutting down", provider=provider
            )
        self.registry.get(provider)  # ProviderNotFound for unknown / disabled providers
        loop = asyncio.get_running_loop()
        sem = self._semaphore(provider)
        await sem.acquire()
        release = True
        started = time.monotonic()
        outcome = "ok"
        try:
            try:
                fut = loop.run_in_executor(self._pool(), fn)
            except RuntimeError:  # executor already shut down
                raise Unavailable(
                    f"{provider} {op} skipped: the daemon is shutting down", provider=provider
                ) from None
            try:
                done, _ = await asyncio.wait({fut}, timeout=timeout_s)
            except asyncio.CancelledError:
                outcome = "cancelled"
                fut.add_done_callback(_retrieve)
                raise
            if not done:
                outcome = "timeout"
                emit(
                    "adapter.timeout",
                    f"{provider} {op} timed out after {timeout_s:g}s",
                    level=logging.WARNING,
                    log=_logger,
                    provider=provider,
                    op=op,
                    timeout_s=timeout_s,
                )
                fut.add_done_callback(_retrieve)
                if orphan_key is not None:
                    # Keep the slot until the worker really returns (it may still submit).
                    release = False
                    self._orphans[orphan_key] = (provider, fut)
                    fut.add_done_callback(lambda _f: sem.release())
                raise Unavailable(
                    f"{provider} {op} timed out after {timeout_s:g}s", provider=provider
                )
            try:
                return fut.result()
            except AdapterError as exc:
                outcome = type(exc).__name__
                raise
            except Exception as exc:
                outcome = "contract_violation"
                raise self._violation(provider, op, exc) from exc
        finally:
            if release:
                sem.release()
            emit(
                "adapter.call",
                f"{provider} {op} {outcome}",
                level=logging.DEBUG,
                log=_logger,
                provider=provider,
                op=op,
                outcome=outcome,
                duration_ms=round((time.monotonic() - started) * 1000, 1),
            )

    def _violation(self, provider: str, op: str, exc: Exception) -> AdapterContractViolation:
        emit(
            "adapter.bug",
            f"{provider} {op} raised {type(exc).__name__}: {exc}",
            level=logging.ERROR,
            exc_info=exc,
            log=_logger,
            provider=provider,
            op=op,
        )
        return AdapterContractViolation(provider, op, exc)

    # ------------------------------------------------------------------ orphaned submits

    def orphan_submit(self, attempt_key: str) -> OrphanSubmit | None:
        """The submit for `attempt_key` that timed out in this process, if any (peek)."""
        entry = self._orphans.get(attempt_key)
        if entry is None:
            return None
        provider, fut = entry
        if not fut.done():
            return OrphanSubmit(provider, done=False)
        if fut.cancelled():
            return OrphanSubmit(provider, done=True, error=Unavailable("submit cancelled"))
        exc = fut.exception()
        if exc is None:
            return OrphanSubmit(provider, done=True, ref=fut.result())
        if isinstance(exc, AdapterError):
            return OrphanSubmit(provider, done=True, error=exc)
        if isinstance(exc, Exception):
            return OrphanSubmit(provider, done=True, error=self._violation(provider, "submit", exc))
        return OrphanSubmit(provider, done=True, error=Unavailable(str(exc)))

    def orphan_keys(self) -> list[str]:
        return list(self._orphans)

    def forget_orphan_submit(self, attempt_key: str) -> None:
        self._orphans.pop(attempt_key, None)

    async def wait_orphan_submit(self, attempt_key: str) -> OrphanSubmit | None:
        """Wait until the orphaned submit's worker returns, then report it like
        `orphan_submit`."""
        entry = self._orphans.get(attempt_key)
        if entry is not None and not entry[1].done():
            await asyncio.wait({entry[1]})
        return self.orphan_submit(attempt_key)

    # ------------------------------------------------------------------ calls

    async def submit(self, provider: str, job: Job, ctx: AttemptContext) -> RemoteRef:
        adapter = self.registry.get(provider)

        def fn() -> RemoteRef:
            ref = adapter.submit(job, ctx)
            _expect(ref, RemoteRef, "RemoteRef")
            return ref

        # A submit retried under the same key supersedes an old, finished orphan.
        entry = self._orphans.get(ctx.attempt_key)
        if entry is not None and entry[1].done():
            self._orphans.pop(ctx.attempt_key, None)
        return await self._call(
            provider,
            "submit",
            fn,
            timeout_s=self.config.timeouts.submit,
            orphan_key=ctx.attempt_key,
        )

    async def status(self, provider: str, ref: RemoteRef) -> RemoteStatus:
        adapter = self.registry.get(provider)

        def fn() -> RemoteStatus:
            st = adapter.status(ref)
            _expect(st, RemoteStatus, "RemoteStatus")
            return st

        return await self._call(provider, "status", fn, timeout_s=self.config.timeouts.status)

    async def logs(
        self, provider: str, ref: RemoteRef, *, since: str | None, max_lines: int = 10_000
    ) -> list[LogChunk]:
        """Non-follow logs, drained in the worker thread (stops after max_lines)."""
        adapter = self.registry.get(provider)

        def fn() -> list[LogChunk]:
            out: list[LogChunk] = []
            count = 0
            for chunk in adapter.logs(ref, follow=False, since=since):
                _expect(chunk, LogChunk, "LogChunk")
                out.append(chunk)
                count += len(chunk.lines)
                if count >= max_lines:
                    break
            return out

        return await self._call(provider, "logs", fn, timeout_s=self.config.timeouts.logs)

    async def fetch(self, provider: str, ref: RemoteRef, dest: Path) -> FetchResult:
        adapter = self.registry.get(provider)

        def fn() -> FetchResult:
            res = adapter.fetch(ref, dest)
            _expect(res, FetchResult, "FetchResult")
            return res

        return await self._call(provider, "fetch", fn, timeout_s=self.config.timeouts.fetch)

    async def cancel(self, provider: str, ref: RemoteRef) -> None:
        adapter = self.registry.get(provider)

        def fn() -> None:
            res = adapter.cancel(ref)
            if res is not None:
                raise _WrongType("None", res)

        await self._call(provider, "cancel", fn, timeout_s=self.config.timeouts.cancel)

    async def quota(self, provider: str) -> QuotaSnapshot:
        adapter = self.registry.get(provider)

        def fn() -> QuotaSnapshot:
            q = adapter.quota()
            _expect(q, QuotaSnapshot, "QuotaSnapshot")
            return q

        return await self._call(provider, "quota", fn, timeout_s=self.config.timeouts.quota)

    async def healthcheck(self, provider: str) -> Health:
        adapter = self.registry.get(provider)

        def fn() -> Health:
            h = adapter.healthcheck()
            _expect(h, Health, "Health")
            return h

        return await self._call(
            provider, "healthcheck", fn, timeout_s=self.config.timeouts.healthcheck
        )

    async def stage_data(
        self, provider: str, path: Path, sha256: str, files: Sequence[tuple[str, int]]
    ) -> StagedData:
        adapter = self.registry.get(provider)
        listed = tuple(files)

        def fn() -> StagedData:
            out = adapter.stage_data(path, sha256, listed)
            _expect(out, StagedData, "StagedData")
            return out

        return await self._call(
            provider, "stage_data", fn, timeout_s=self.config.timeouts.stage_data
        )

    async def lookup_by_key(self, provider: str, attempt_key: str) -> RemoteRef | None:
        """Raises Permanent if the adapter lacks capabilities.lookup_by_key (callers check
        the capability first)."""
        adapter = self.registry.get(provider)
        if not adapter.capabilities.lookup_by_key:
            from gpu_router.errors import Permanent

            raise Permanent(f"{provider} cannot look runs up by key", provider=provider)

        def fn() -> RemoteRef | None:
            ref = adapter.lookup_by_key(attempt_key)
            if ref is not None:
                _expect(ref, RemoteRef, "RemoteRef or None")
            return ref

        return await self._call(
            provider, "lookup_by_key", fn, timeout_s=self.config.timeouts.status
        )

    def shutdown(self, *, wait: bool = False) -> None:
        """Stop accepting calls; do not wait for stuck worker threads unless asked."""
        self._closed = True
        if self._executor is not None and self._owns_executor:
            self._executor.shutdown(wait=wait, cancel_futures=True)
