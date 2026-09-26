"""Fixtures for Lightning adapter unit tests: a SimLightning on a FakeClock, an adapter
over it (logged in through the in-memory keyring), real bundles from throwaway projects."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from gpu_router.clock import FakeClock
from gpu_router.paths import Paths
from tests.contract.lightning.sim import SimLightning
from tests.contract.lightning.targets import SimLightningAdapter, store_sim_credentials
from tests.unit.packaging.helpers import isolate_git

#: 2026-09-24 12:00 UTC: mid-month, so month bounds are easy to state
NOW = 1_790_251_200.0


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    isolate_git(monkeypatch, tmp_path)
    for var in ("LIGHTNING_USER_ID", "LIGHTNING_API_KEY", "LIGHTNING_AUTH_TOKEN"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def fclock() -> FakeClock:
    return FakeClock(NOW)


@pytest.fixture
def sim(fclock: FakeClock) -> SimLightning:
    return SimLightning(fclock)


@pytest.fixture
def adapter(sim: SimLightning, paths: Paths) -> Iterator[SimLightningAdapter]:
    store_sim_credentials()
    a = SimLightningAdapter(sim, paths)
    yield a
    a.close()
