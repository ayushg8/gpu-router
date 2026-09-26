"""A real DaemonRuntime notifies through its EventBus (phase 8a): finished, failed, needs
approval, migrated, each once. The backend is swapped for a recorder: nothing reaches macOS
(and under pytest the daemon would pick the null backend anyway)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from gpu_router.clock import FakeClock
from gpu_router.config import Config
from gpu_router.daemon.app import create_app
from gpu_router.daemon.runtime import DaemonRuntime
from gpu_router.errors import ConfigError
from gpu_router.notify.backends import NullBackend
from gpu_router.paths import Paths
from tests.api.conftest import BASE_URL, Api


@pytest.fixture
async def api(paths: Paths, test_config: Config, tmp_path: Path) -> AsyncIterator[Api]:
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


def recorder(api: Api) -> NullBackend:
    notifier = api.runtime.notifier
    assert notifier is not None
    # test mode + pytest: the daemon chose the null backend on its own
    assert isinstance(notifier.backend, NullBackend)
    rec = NullBackend(why="recording")
    notifier.backend = rec
    return rec


def sent(api: Api, rec: NullBackend) -> list[Any]:
    notifier = api.runtime.notifier
    assert notifier is not None
    assert notifier.wait_idle()
    return list(rec.sent)


async def test_finished_job_notifies_once(api: Api) -> None:
    rec = recorder(api)
    job = await api.submit(name="train_yolo")
    await api.drive(lambda: api.state(job["id"]) == "done")
    got = sent(api, rec)
    assert [n.kind for n in got] == ["finished"]
    assert got[0].subtitle == "✓ train_yolo finished"
    assert "on fake" in got[0].body


async def test_failed_job_notifies_with_the_exit_code(api: Api) -> None:
    rec = recorder(api)
    job = await api.submit(
        name="bad", provider_options={"fake": {"duration": 5, "fail_at": 1, "exit_code": 3}}
    )
    await api.drive(lambda: api.state(job["id"]) == "failed")
    got = sent(api, rec)
    assert [n.kind for n in got] == ["failed"]
    assert got[0].body.startswith("exit 3")
    assert f"gpu logs {job['short_id']}" in got[0].body


async def test_approval_notifies_and_the_run_after_it_finishes(api: Api) -> None:
    rec = recorder(api)
    job = await api.submit(name="eval.py", requires_approval=True)
    await api.drive(lambda: api.state(job["id"]) == "awaiting_approval")
    got = sent(api, rec)
    assert [n.kind for n in got] == ["approval"]
    assert f"gpu approve {job['short_id']}" in got[0].body
    resp = await api.client.post(f"/v1/jobs/{job['id']}/approve", json={})
    assert resp.status_code == 200
    await api.drive(lambda: api.state(job["id"]) == "done")
    assert [n.kind for n in sent(api, rec)] == ["approval", "finished"]


async def test_a_migration_notifies(api: Api) -> None:
    rec = recorder(api)
    job = await api.submit(
        name="long",
        checkpoint_interval_min=1,
        provider_options={"fake": {"duration": 6, "attempts": {"1": {"die_after": 2}}}},
    )
    await api.drive(lambda: api.state(job["id"]) == "done")
    kinds = [n.kind for n in sent(api, rec)]
    assert kinds == ["migrated", "finished"]


async def test_cancelled_jobs_stay_quiet(api: Api) -> None:
    rec = recorder(api)
    job = await api.submit(name="x", provider_options={"fake": {"duration": 60}})
    await api.drive(lambda: api.state(job["id"]) == "running")
    resp = await api.client.post(f"/v1/jobs/{job['id']}/cancel")
    assert resp.status_code == 200
    await api.drive(lambda: api.state(job["id"]) == "cancelled")
    assert sent(api, rec) == []


def test_a_bad_notifications_section_stops_the_daemon_start(
    paths: Paths, test_config: Config
) -> None:
    config = test_config.model_copy(update={"notifications": {"backend": "growl"}})
    with pytest.raises(ConfigError, match="notifications"):
        DaemonRuntime.create(paths, config, FakeClock(), configure_logging=False)
