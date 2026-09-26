"""QuotaService: TTL cache of live readings, background refresh, slow/failed providers."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import pytest

from gpu_router.adapters.base import Capabilities
from gpu_router.clock import FakeClock
from gpu_router.errors import Unavailable
from gpu_router.models import ProviderHealth, QuotaSnapshot, QuotaUnit
from gpu_router.providers.catalog import ProviderEntry, load_catalog
from gpu_router.quota.service import QuotaService
from gpu_router.quota.settings import QuotaSettings
from gpu_router.store import Store

CAT = load_catalog()


@dataclass
class _Adapter:
    capabilities: Capabilities


@dataclass
class StubRegistry:
    entries: dict[str, ProviderEntry]
    caps: dict[str, Capabilities]

    def names(self) -> list[str]:
        return list(self.entries)

    def entry(self, name: str) -> ProviderEntry:
        return self.entries[name]

    def get(self, name: str) -> _Adapter:
        return _Adapter(self.caps[name])

    def __contains__(self, name: object) -> bool:
        return name in self.entries


@dataclass
class StubCaller:
    clock: FakeClock
    delay: dict[str, float] = field(default_factory=dict)
    fail: set[str] = field(default_factory=set)
    calls: list[str] = field(default_factory=list)
    gate: asyncio.Event | None = None

    async def quota(self, name: str) -> QuotaSnapshot:
        self.calls.append(name)
        if self.gate is not None:
            await self.gate.wait()
        if name in self.fail:
            raise Unavailable("kaggle api down", provider=name)
        return QuotaSnapshot(
            provider=name,
            used=12,
            limit=30,
            unit=QuotaUnit.GPU_HOURS,
            resets_at=self.clock.now() + 86_400,
            source="live",
            observed_at=self.clock.now(),
        )


Setup = tuple[QuotaService, StubCaller, Store, FakeClock]


@pytest.fixture
def setup() -> Setup:
    clock = FakeClock()
    store = Store.open_memory(clock)
    registry = StubRegistry(
        entries={n: CAT.get(n) for n in ("kaggle", "colab", "local")},
        caps={
            "kaggle": Capabilities(live_quota=True),
            "colab": Capabilities(live_quota=False),
            "local": Capabilities(live_quota=True),
        },
    )
    caller = StubCaller(clock)
    svc = QuotaService(
        store=store,
        registry=registry,  # type: ignore[arg-type]
        caller=caller,  # type: ignore[arg-type]
        clock=clock,
        settings=QuotaSettings(ttl_s=900, retry_failed_s=300),
    )
    return svc, caller, store, clock


async def test_refresh_only_stale_live_providers(setup: Setup) -> None:
    svc, caller, _store, clock = setup
    # colab has no live quota, local is unlimited: neither is ever called
    assert svc.stale() == ["kaggle"]
    await svc.refresh(wait_s=1)
    assert caller.calls == ["kaggle"]
    views = svc.views()
    assert views["kaggle"].source == "live"
    assert views["kaggle"].used == 12
    assert views["colab"].source == "estimate"
    assert svc.stale() == []  # fresh within the TTL
    clock.advance(901)
    assert svc.stale() == ["kaggle"]
    await svc.close()


async def test_slow_provider_never_blocks_views(setup: Setup) -> None:
    svc, caller, _store, _clock = setup
    caller.gate = asyncio.Event()  # every quota() call hangs until released
    await svc.refresh(["kaggle"], wait_s=0.05)  # returns after wait_s, call keeps going
    assert caller.calls == ["kaggle"]
    assert svc.views()["kaggle"].source == "estimate"  # estimate meanwhile
    await svc.refresh(["kaggle"], wait_s=0.01)  # joins the in-flight call, no second call
    assert caller.calls == ["kaggle"]
    caller.gate.set()
    await asyncio.sleep(0.01)
    assert svc.views()["kaggle"].source == "live"
    await svc.close()


async def test_failed_call_backs_off_and_skips_logged_out(setup: Setup) -> None:
    svc, caller, store, clock = setup
    caller.fail = {"kaggle"}
    await svc.refresh(wait_s=1)
    assert svc.views()["kaggle"].source == "estimate"
    assert "kaggle" not in svc.stale()  # failed recently
    clock.advance(301)
    assert "kaggle" in svc.stale()
    store.upsert_provider_state("kaggle", health=ProviderHealth.AUTH_REQUIRED)
    assert "kaggle" not in svc.stale()  # needs login: do not hammer it
    assert svc.stale(force=True) == []
    await svc.close()


async def test_snapshots_order_and_close_is_idempotent(setup: Setup) -> None:
    svc, caller, _store, _clock = setup
    assert [q.provider for q in svc.snapshots()] == ["kaggle", "colab", "local"]
    svc.start()
    await asyncio.sleep(0.01)  # the loop's first refresh ran
    assert caller.calls == ["kaggle"]
    await svc.close()
    await svc.close()
    await svc.refresh(force=True, wait_s=1)  # closed: no more calls
    assert caller.calls == ["kaggle"]


async def test_background_refresh_keeps_the_reading_live_every_period() -> None:
    """Review finding (D44): a reading stamped after a 3 s call was still 'fresh' at the
    next 1800 s tick, so it was skipped and the ledger fell back to an estimate for half
    of every hour. The loop also sleeps on the injected clock (invariant 13)."""
    clock = FakeClock()
    store = Store.open_memory(clock)
    registry = StubRegistry(
        entries={"kaggle": CAT.get("kaggle")}, caps={"kaggle": Capabilities(live_quota=True)}
    )
    caller = StubCaller(clock)
    svc = QuotaService(
        store=store,
        registry=registry,  # type: ignore[arg-type]
        caller=caller,  # type: ignore[arg-type]
        clock=clock,
    )
    t0 = clock.now()
    clock.advance(3)  # the provider's call took 3 s: the reading is stamped at t0 + 3
    await svc.refresh(wait_s=1)
    clock.set(t0 + svc.settings.refresh_s)
    assert svc.stale() == ["kaggle"]  # due at the next tick, not half an hour later
    caller.calls.clear()
    svc.start()
    for _ in range(3):
        await asyncio.sleep(0)
    assert caller.calls == ["kaggle"]
    clock.advance(svc.settings.refresh_s)  # fake time drives the loop
    for _ in range(5):
        await asyncio.sleep(0)
    assert caller.calls == ["kaggle", "kaggle"]
    await svc.close()
