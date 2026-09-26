"""Engine test harness (owner: group C).

`Engine` wires a real Supervisor to the packaged catalog's fake providers ("fake" priority
1, T4 16GB; "fake-b" priority 2, A100 40GB), an in-memory Store and a FakeClock. Adapter
calls run inline (InlineExecutor) instead of in worker threads, so a test is fully
deterministic: `await eng.run_until(pred)` alternates `settle()` with `clock.advance(step)`.
"""

from __future__ import annotations

import concurrent.futures
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any

import pytest

from gpu_router.adapters.fake import FakeAdapter
from gpu_router.adapters.registry import AdapterRegistry
from gpu_router.clock import FakeClock, settle
from gpu_router.config import Config, DaemonConfig, EngineConfig
from gpu_router.engine.calls import AdapterCaller
from gpu_router.engine.deps import EngineDeps
from gpu_router.engine.supervisor import Supervisor
from gpu_router.models import Job, JobSpec, JobState, Source
from gpu_router.paths import Paths
from gpu_router.policy import default_policy
from gpu_router.providers.catalog import Catalog
from gpu_router.router.simple import SimpleRouter
from gpu_router.statemachine import is_terminal
from gpu_router.store import Store


class InlineExecutor(concurrent.futures.Executor):
    """Runs submitted callables immediately on the calling thread (tests only)."""

    def submit(
        self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any
    ) -> concurrent.futures.Future[Any]:
        fut: concurrent.futures.Future[Any] = concurrent.futures.Future()
        try:
            fut.set_result(fn(*args, **kwargs))
        except BaseException as exc:
            fut.set_exception(exc)
        return fut


def engine_config(**engine: Any) -> Config:
    base: dict[str, Any] = {
        "backoff_base_s": 1,
        "backoff_cap_s": 8,
        "default_poll_interval_s": 1,
        "max_workers": 2,
    }
    base.update(engine)
    return Config(
        test_mode=True, daemon=DaemonConfig(port=0, shutdown_grace_s=1), engine=EngineConfig(**base)
    )


@dataclass
class Engine:
    store: Store
    clock: FakeClock
    paths: Paths
    config: Config
    registry: AdapterRegistry
    supervisor: Supervisor
    project: str

    @property
    def deps(self) -> EngineDeps:
        return self.supervisor.deps

    def fake(self, name: str = "fake") -> FakeAdapter:
        adapter = self.registry.get(name)
        assert isinstance(adapter, FakeAdapter)
        return adapter

    def spec(
        self,
        *,
        fake: dict[str, Any] | None = None,
        options: dict[str, dict[str, Any]] | None = None,
        **fields: Any,
    ) -> JobSpec:
        opts: dict[str, dict[str, Any]] = {"fake": {"duration": 5, "steps": 5}}
        if fake is not None:
            opts = {"fake": {"duration": 5, "steps": 5, **fake}}
        if options is not None:
            opts = options
        fields.setdefault("script", "train.py")
        return JobSpec(project_dir=self.project, provider_options=opts, source=Source.API, **fields)

    async def submit(self, spec: JobSpec | None = None, **kw: Any) -> Job:
        job, created = await self.supervisor.submit(spec or self.spec(**kw), actor="api")
        assert created
        return job

    def job(self, job_id: str) -> Job:
        return self.store.get_job(job_id)

    async def run_until(
        self, pred: Callable[[], bool], *, max_s: float = 600, step: float = 0.5
    ) -> None:
        waited = 0.0
        while True:
            await settle(30)
            if pred():
                return
            if waited >= max_s:
                raise AssertionError(f"condition not met after {max_s}s of fake time")
            self.clock.advance(step)
            waited += step

    async def until_state(self, job_id: str, *states: JobState, max_s: float = 600) -> Job:
        await self.run_until(lambda: self.job(job_id).state in states, max_s=max_s)
        return self.job(job_id)

    async def until_terminal(self, job_id: str, max_s: float = 600) -> Job:
        await self.run_until(lambda: is_terminal(self.job(job_id).state), max_s=max_s)
        return self.job(job_id)

    def reasons(self, job_id: str) -> list[str]:
        return [e.reason for e in self.store.events_for(job_id, limit=10_000)]

    def transitions(self, job_id: str) -> list[tuple[str | None, str | None, str]]:
        return [
            (
                str(e.from_state) if e.from_state else None,
                str(e.to_state) if e.to_state else None,
                e.reason,
            )
            for e in self.store.events_for(job_id, limit=10_000)
            if e.kind == "transition"
        ]

    async def restart(self) -> Supervisor:
        """Simulate a daemon restart: stop the supervisor (remote runs untouched) and start
        a fresh one with fresh adapters over the same store and fake remote dir."""
        await self.supervisor.stop()
        self.supervisor = build_supervisor(
            self.store, self.clock, self.paths, self.config, self.registry.catalog
        )
        self.registry = self.supervisor.deps.registry
        await self.supervisor.start()
        return self.supervisor


def build_supervisor(
    store: Store, clock: FakeClock, paths: Paths, config: Config, catalog: Catalog
) -> Supervisor:
    registry = AdapterRegistry.build(config=config, catalog=catalog, paths=paths, clock=clock)
    caller = AdapterCaller(registry, config.engine, executor=InlineExecutor())
    deps = EngineDeps(
        store=store,
        registry=registry,
        router=SimpleRouter(),
        policy=default_policy(),
        caller=caller,
        clock=clock,
        config=config,
        paths=paths,
    )
    return Supervisor(deps)


@pytest.fixture
def engine_cfg() -> Config:
    """Override in a test module (or parametrize) to change engine timers."""
    return engine_config()


@pytest.fixture
async def eng(
    store: Store,
    clock: FakeClock,
    paths: Paths,
    engine_cfg: Config,
    catalog: Catalog,
    tmp_path: Any,
) -> AsyncIterator[Engine]:
    project = tmp_path / "project"
    project.mkdir()
    sup = build_supervisor(store, clock, paths, engine_cfg, catalog)
    e = Engine(
        store=store,
        clock=clock,
        paths=paths,
        config=engine_cfg,
        registry=sup.deps.registry,
        supervisor=sup,
        project=str(project),
    )
    await sup.start()
    await settle(30)
    try:
        yield e
    finally:
        await e.supervisor.stop()
