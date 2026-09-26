"""API test harness (owner: group C): a real DaemonRuntime on the tmp home (FakeClock, fake
providers, real worker threads) behind create_app(), driven through httpx.ASGITransport."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any

import httpx
import pytest

from gpu_router.clock import FakeClock
from gpu_router.config import Config
from gpu_router.daemon.app import create_app
from gpu_router.daemon.runtime import DaemonRuntime
from gpu_router.paths import Paths

BASE_URL = "http://127.0.0.1:47291"


@dataclass
class Api:
    runtime: DaemonRuntime
    client: httpx.AsyncClient
    clock: FakeClock
    project: str

    def spec(self, **fields: Any) -> dict[str, Any]:
        body: dict[str, Any] = {
            "project_dir": self.project,
            "script": "train.py",
            "provider_options": {"fake": {"duration": 3, "steps": 3}},
        }
        body.update(fields)
        return body

    async def drive(
        self, pred: Callable[[], Any], *, max_s: float = 300, step: float = 0.5
    ) -> None:
        """Advance fake time until pred() is truthy (adapter calls run in real threads)."""
        waited = 0.0
        while True:
            for _ in range(3):
                await asyncio.sleep(0.005)
            if pred():
                return
            if waited >= max_s:
                raise AssertionError(f"condition not met after {max_s}s of fake time")
            self.clock.advance(step)
            waited += step

    def state(self, job_id: str) -> str:
        return str(self.runtime.store.get_job(job_id).state)

    async def submit(self, **fields: Any) -> dict[str, Any]:
        resp = await self.client.post("/v1/jobs", json={"spec": self.spec(**fields)})
        assert resp.status_code == 201, resp.text
        data: dict[str, Any] = resp.json()
        return data


@pytest.fixture
async def api(paths: Paths, test_config: Config, tmp_path: Any) -> AsyncIterator[Api]:
    clock = FakeClock()
    runtime = DaemonRuntime.create(paths, test_config, clock, configure_logging=False)
    runtime.port = 47291
    app = create_app(runtime)
    await runtime.start()
    project = tmp_path / "proj"
    project.mkdir()
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(
        transport=transport,
        base_url=BASE_URL,
        headers={"Authorization": f"Bearer {runtime.token}", "X-Gpu-Router-Client": "cli"},
    ) as client:
        try:
            yield Api(runtime=runtime, client=client, clock=clock, project=str(project))
        finally:
            await runtime.stop()
