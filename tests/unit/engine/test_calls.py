from __future__ import annotations

import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from gpu_router.adapters.base import (
    Adapter,
    AdapterDeps,
    AttemptContext,
    FetchResult,
    Health,
    LogChunk,
    RemoteRef,
    RemoteStatus,
)
from gpu_router.adapters.registry import AdapterRegistry
from gpu_router.clock import FakeClock
from gpu_router.config import CallTimeouts, EngineConfig, ProviderSettings
from gpu_router.engine.calls import AdapterCaller
from gpu_router.errors import (
    AdapterContractViolation,
    Permanent,
    ProviderNotFound,
    RateLimited,
    Unavailable,
)
from gpu_router.models import Job, ProviderHealth, QuotaSnapshot
from gpu_router.paths import Paths
from gpu_router.providers.catalog import Catalog, GpuOffer, ProviderEntry

ENTRY = ProviderEntry(
    name="toy", kind="toy", display_name="Toy", gpus=(GpuOffer(name="T4", vram_gb=16),)
)


class Toy(Adapter):
    kind = "toy"

    def __init__(self, deps: AdapterDeps) -> None:
        super().__init__(deps)
        self.mode = "ok"
        self.release = threading.Event()

    def submit(self, job: Job, ctx: AttemptContext) -> RemoteRef:
        if self.mode == "rate":
            raise RateLimited("slow down", retry_after=3)
        if self.mode == "bug":
            raise KeyError("oops")
        if self.mode == "wrong":
            return "not a ref"  # type: ignore[return-value]
        if self.mode == "hang":
            self.release.wait(5)
        return RemoteRef(remote_id="r1")

    def status(self, ref: RemoteRef) -> RemoteStatus:
        raise NotImplementedError

    def logs(
        self, ref: RemoteRef, *, follow: bool = False, since: str | None = None
    ) -> Iterator[LogChunk]:
        for i in range(5):
            yield LogChunk(lines=[f"l{i}"] * 3, cursor=str(i))

    def fetch(self, ref: RemoteRef, dest: Path) -> FetchResult:
        return FetchResult(dest=dest, files=0, bytes=0)

    def cancel(self, ref: RemoteRef) -> None:
        return None

    def quota(self) -> QuotaSnapshot:
        raise NotImplementedError

    def healthcheck(self) -> Health:
        return Health(health=ProviderHealth.OK, checked_at=0)


@pytest.fixture
def toy(paths: Paths) -> Toy:
    return Toy(
        AdapterDeps(
            name="toy", entry=ENTRY, settings=ProviderSettings(), paths=paths, clock=FakeClock()
        )
    )


def _caller(toy: Toy, **timeouts: float) -> AdapterCaller:
    reg = AdapterRegistry.of({"toy": toy}, Catalog(providers={"toy": ENTRY}))
    return AdapterCaller(reg, EngineConfig(max_workers=2, timeouts=CallTimeouts(**timeouts)))


def _job() -> Any:
    return None  # the toy adapter never looks at the job


CTX = AttemptContext(attempt_id="a.1", attempt_key="gpu-aaaaaaaaaaaa-1", n=1)


async def test_success_and_taxonomy_passthrough(toy: Toy) -> None:
    caller = _caller(toy)
    try:
        assert (await caller.submit("toy", _job(), CTX)).remote_id == "r1"
        toy.mode = "rate"
        with pytest.raises(RateLimited) as info:
            await caller.submit("toy", _job(), CTX)
        assert info.value.retry_after == 3
    finally:
        caller.shutdown()


async def test_non_taxonomy_exception_and_wrong_type_are_contract_violations(toy: Toy) -> None:
    caller = _caller(toy)
    try:
        toy.mode = "bug"
        with pytest.raises(AdapterContractViolation, match="oops") as info:
            await caller.submit("toy", _job(), CTX)
        assert info.value.detail["cause"] == "KeyError"
        toy.mode = "wrong"
        with pytest.raises(AdapterContractViolation, match="RemoteRef"):
            await caller.submit("toy", _job(), CTX)
        with pytest.raises(AdapterContractViolation) as info2:
            await caller.quota("toy")
        assert info2.value.detail["cause"] == "NotImplementedError"
    finally:
        caller.shutdown()


async def test_timeout_becomes_unavailable(toy: Toy) -> None:
    caller = _caller(toy, submit=0.05)
    toy.mode = "hang"
    try:
        with pytest.raises(Unavailable, match="timed out"):
            await caller.submit("toy", _job(), CTX)
    finally:
        toy.release.set()
        caller.shutdown(wait=True)


async def test_logs_drained_in_worker_with_line_cap(toy: Toy) -> None:
    caller = _caller(toy)
    try:
        chunks = await caller.logs("toy", RemoteRef(remote_id="r1"), since=None, max_lines=7)
        assert [c.cursor for c in chunks] == ["0", "1", "2"]
        assert all(isinstance(c, LogChunk) for c in chunks)
    finally:
        caller.shutdown()


async def test_lookup_without_capability_and_unknown_provider(toy: Toy) -> None:
    caller = _caller(toy)
    try:
        with pytest.raises(Permanent):
            await caller.lookup_by_key("toy", "gpu-aaaaaaaaaaaa-1")
        with pytest.raises(ProviderNotFound):
            await caller.healthcheck("nope")
        assert (await caller.healthcheck("toy")).ok
        assert await caller.cancel("toy", RemoteRef(remote_id="r1")) is None
    finally:
        caller.shutdown()


async def test_calls_after_shutdown_are_unavailable(toy: Toy) -> None:
    caller = _caller(toy)
    caller.shutdown()
    with pytest.raises(Unavailable, match="shutting down"):
        await caller.healthcheck("toy")
