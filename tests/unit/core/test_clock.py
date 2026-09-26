from __future__ import annotations

import asyncio

import pytest

from gpu_router.clock import FAKE_EPOCH, Clock, FakeClock, SystemClock, settle


def test_protocol() -> None:
    assert isinstance(SystemClock(), Clock)
    assert isinstance(FakeClock(), Clock)


def test_fake_clock_time() -> None:
    c = FakeClock()
    assert c.now() == FAKE_EPOCH
    assert c.monotonic() == 0
    c.advance(5)
    assert c.now() == FAKE_EPOCH + 5
    assert c.monotonic() == 5
    c.set(FAKE_EPOCH + 10)
    assert c.now() == FAKE_EPOCH + 10
    with pytest.raises(ValueError, match="backwards"):
        c.advance(-1)


async def test_fake_sleep_wakes_in_order() -> None:
    c = FakeClock()
    woke: list[str] = []

    async def sleeper(name: str, s: float) -> None:
        await c.sleep(s)
        woke.append(name)

    tasks = [asyncio.create_task(sleeper("b", 20)), asyncio.create_task(sleeper("a", 10))]
    await settle()
    assert c.pending_sleepers == 2
    assert c.next_deadline() == FAKE_EPOCH + 10
    c.advance(9)
    await settle()
    assert woke == []
    c.advance(100)
    await settle()
    assert woke == ["a", "b"]
    assert c.pending_sleepers == 0
    assert c.next_deadline() is None
    await asyncio.gather(*tasks)


async def test_sleep_zero_yields_and_system_clock() -> None:
    await FakeClock().sleep(0)
    s = SystemClock()
    t = s.monotonic()
    await s.sleep(-1)
    assert s.monotonic() >= t
    assert s.now() > FAKE_EPOCH - 10**9
