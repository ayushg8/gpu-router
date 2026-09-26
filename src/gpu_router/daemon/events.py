"""EventBus (phase 1; owner: group C).

Implements store.StoreListener. On every committed job change it (1) wakes long-poll
waiters of GET /v1/events and per-job log followers, (2) marks the state.json writer dirty,
(3) (phase 8) forwards terminal/approval events to notifications. Called on the event-loop
thread right after COMMIT; must not call back into the Store synchronously (invariant 9).
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable, Sequence

from gpu_router.engine._obs import emit
from gpu_router.models import JobEvent


class EventBus:
    def __init__(self) -> None:
        self._on_change: list[Callable[[str, Sequence[JobEvent]], None]] = []
        self._max_seq = 0
        self._global_waiters: set[asyncio.Event] = set()
        self._job_waiters: dict[str, set[asyncio.Event]] = {}

    @property
    def max_seq(self) -> int:
        """Highest event seq this bus has seen committed (0 before the first event)."""
        return self._max_seq

    def job_changed(self, job_id: str, events: Sequence[JobEvent]) -> None:
        for ev in events:
            if ev.seq > self._max_seq:
                self._max_seq = ev.seq
        if events:
            for waiter in self._global_waiters:
                waiter.set()
        for waiter in self._job_waiters.get(job_id, ()):
            waiter.set()
        for fn in list(self._on_change):
            try:
                fn(job_id, events)
            except Exception:  # a subscriber bug must not break the committing writer
                emit("api.bug", f"event subscriber {fn!r} failed", exc_info=True, job_id=job_id)

    def provider_changed(self, provider: str) -> None:
        """Optional store-listener hook: provider health/cooldown changed. Subscribers get
        job_id="" and no events (enough to mark the state.json writer dirty)."""
        for fn in list(self._on_change):
            try:
                fn("", ())
            except Exception:
                emit("api.bug", f"event subscriber {fn!r} failed", exc_info=True, provider=provider)

    def subscribe(self, fn: Callable[[str, Sequence[JobEvent]], None]) -> Callable[[], None]:
        """Register a synchronous callback; returns an unsubscribe function."""
        self._on_change.append(fn)

        def unsubscribe() -> None:
            with contextlib.suppress(ValueError):
                self._on_change.remove(fn)

        return unsubscribe

    async def wait_for_events(self, after_seq: int, timeout_s: float) -> None:
        """Return as soon as an event with seq > after_seq is committed, or after timeout_s
        (clock-independent: uses asyncio timeouts; long polls are wall-time by nature)."""
        if self._max_seq > after_seq or timeout_s <= 0:
            return
        waiter = asyncio.Event()
        self._global_waiters.add(waiter)
        try:
            loop = asyncio.get_running_loop()
            deadline = loop.time() + timeout_s
            while self._max_seq <= after_seq:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    return
                waiter.clear()
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(waiter.wait(), remaining)
        finally:
            self._global_waiters.discard(waiter)

    async def wait_for_job_change(self, job_id: str, timeout_s: float) -> bool:
        """True if the job changed (any write, including progress) within timeout_s."""
        waiter = asyncio.Event()
        self._job_waiters.setdefault(job_id, set()).add(waiter)
        try:
            await asyncio.wait_for(waiter.wait(), max(0.0, timeout_s))
            return True
        except TimeoutError:
            return False
        finally:
            waiters = self._job_waiters.get(job_id)
            if waiters is not None:
                waiters.discard(waiter)
                if not waiters:
                    self._job_waiters.pop(job_id, None)
