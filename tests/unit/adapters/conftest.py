"""Fixtures for adapter unit tests (owner: group B)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from gpu_router.adapters.base import AdapterDeps
from gpu_router.adapters.fake import FakeAdapter
from gpu_router.clock import FakeClock
from gpu_router.config import ProviderSettings
from gpu_router.models import Job
from gpu_router.paths import Paths
from gpu_router.providers.catalog import GpuOffer, ProviderEntry, QuotaSpec
from tests.contract.harness import ContractTarget, make_job

FAKE_ENTRY = ProviderEntry(
    name="fake",
    kind="fake",
    display_name="Fake A",
    priority=1,
    test_only=True,
    gpus=(GpuOffer(name="T4", vram_gb=16),),
    session_hours=12,
    max_concurrency=4,
    poll_interval_s=1,
    quota=QuotaSpec(limit=30, reset="weekly"),
)

MakeFake = Callable[..., FakeAdapter]
MakeJob = Callable[..., Job]


@pytest.fixture
def make_fake(paths: Paths, clock: FakeClock) -> MakeFake:
    """Build a FakeAdapter on the shared tmp home + FakeClock. Calling it twice gives two
    independent instances over the same disk state (a daemon restart)."""

    def build(name: str = "fake", entry: ProviderEntry = FAKE_ENTRY) -> FakeAdapter:
        return FakeAdapter(
            AdapterDeps(
                name=name,
                entry=entry.model_copy(update={"name": name}),
                settings=ProviderSettings(),
                paths=paths,
                clock=clock,
                test_mode=True,
            )
        )

    return build


@pytest.fixture
def fake(make_fake: MakeFake) -> FakeAdapter:
    return make_fake()


@pytest.fixture
def make_fake_job(clock: FakeClock) -> MakeJob:
    target = ContractTarget(
        name="fake",
        build=lambda: None,  # type: ignore[arg-type,return-value]
        clock=clock,
        supports_directives=True,
    )

    def build(**directives: Any) -> Job:
        return make_job(target, directives=directives)

    return build
