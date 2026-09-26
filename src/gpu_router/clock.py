"""Injectable time (phase 1; real code because every group's tests depend on it; owner: A).

Invariant 13: engine, store, providers, router and policy never read the wall clock
directly. They receive a `Clock`. Production uses `SystemClock`; in-process tests use
`FakeClock`, whose `sleep()` only returns when the test advances time past its deadline.

Crash-recovery tests run a real daemon subprocess and therefore use SystemClock with tiny
durations (fake directives in fractions of a second, poll intervals of ~0.05s).
"""

from __future__ import annotations

import asyncio
import heapq
import itertools
import time
from typing import Protocol, runtime_checkable

#: 2026-09-21T14:13:20Z. A fixed, recognisable start for FakeClock.
FAKE_EPOCH = 1_790_000_000.0


@runtime_checkable
class Clock(Protocol):
    def now(self) -> float:
        """Unix epoch seconds (UTC)."""
        ...

    def monotonic(self) -> float:
        """Seconds from an arbitrary origin; never goes backwards. Use for durations."""
        ...

    async def sleep(self, seconds: float) -> None:
        """Suspend the calling task for `seconds` of this clock's time."""
        ...


class SystemClock:
    """Real time."""

    def now(self) -> float:
        return time.time()

    def monotonic(self) -> float:
        return time.monotonic()

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(max(0.0, seconds))


class FakeClock:
    """Manually advanced clock for deterministic tests.

    - `now()` starts at `start` and only moves on `advance()` / `set()`.
    - `await sleep(s)` parks the task until time reaches now+s. `sleep(0)` just yields.
    - `advance(dt)` moves time and wakes every sleeper whose deadline passed, in deadline
      order. Woken tasks run at the next loop iteration: follow with `await settle()` to let
      them react before asserting.
    - Usable from sync code too (`now()`), e.g. to drive the fake provider's timeline.
    """

    def __init__(self, start: float = FAKE_EPOCH) -> None:
        self._now = float(start)
        self._mono = 0.0
        self._sleepers: list[tuple[float, int, asyncio.Future[None]]] = []
        self._counter = itertools.count()

    def now(self) -> float:
        return self._now

    def monotonic(self) -> float:
        return self._mono

    async def sleep(self, seconds: float) -> None:
        if seconds <= 0:
            await asyncio.sleep(0)
            return
        fut: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        heapq.heappush(self._sleepers, (self._now + seconds, next(self._counter), fut))
        await fut

    def advance(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("FakeClock cannot go backwards")
        self._now += seconds
        self._mono += seconds
        while self._sleepers and self._sleepers[0][0] <= self._now:
            _, _, fut = heapq.heappop(self._sleepers)
            if not fut.done():
                fut.set_result(None)

    def set(self, when: float) -> None:
        self.advance(when - self._now)

    @property
    def pending_sleepers(self) -> int:
        return sum(1 for _, _, f in self._sleepers if not f.done())

    def next_deadline(self) -> float | None:
        live = [d for d, _, f in self._sleepers if not f.done()]
        return min(live) if live else None


async def settle(rounds: int = 10) -> None:
    """Yield to the event loop `rounds` times so woken tasks run (tests only)."""
    for _ in range(rounds):
        await asyncio.sleep(0)
