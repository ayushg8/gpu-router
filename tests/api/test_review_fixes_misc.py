"""Regression tests for review findings outside the engine: client path safety, launchd
respawn loop, home directory permissions."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import httpx
import pytest

from gpu_router.client import GpuClient
from gpu_router.daemon.__main__ import EXIT_OK, EXIT_STATE
from gpu_router.daemon.__main__ import main as daemon_main
from gpu_router.errors import GpuRouterError, InvalidRequest
from gpu_router.lock import InstanceLock
from gpu_router.paths import Paths
from tests.api.conftest import Api


def _client(seen: list[httpx.Request]) -> GpuClient:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(500, json={"error": {"code": "internal", "message": "x"}})

    return GpuClient("http://127.0.0.1:1", "tok", transport=httpx.MockTransport(handler))


@pytest.mark.parametrize(
    "ref", ["../daemon/shutdown#", "a7f2/../../daemon/shutdown", "a7f2?x=1", "%2e%2e", ""]
)
def test_client_rejects_path_escaping_refs(ref: str) -> None:
    seen: list[httpx.Request] = []
    with _client(seen) as client:
        for call in (client.cancel, client.job, client.fetch, client.approve, client.deny):
            with pytest.raises(InvalidRequest):
                call(ref)
        with pytest.raises(InvalidRequest):
            list(client.logs(ref))
    assert seen == []  # nothing reached the daemon


@pytest.mark.parametrize("name", ["../daemon/shutdown#", "Kaggle/x", "a b"])
def test_client_rejects_bad_provider_names(name: str) -> None:
    seen: list[httpx.Request] = []
    with _client(seen) as client:
        with pytest.raises(InvalidRequest):
            client.provider(name)
        with pytest.raises(InvalidRequest):
            client.healthcheck(name)
    assert seen == []


def test_client_accepts_normal_refs() -> None:
    seen: list[httpx.Request] = []
    with _client(seen) as client, pytest.raises(GpuRouterError):  # mock daemon answers 500
        client.cancel(" A7F2 ")
    assert [r.url.path for r in seen] == ["/v1/jobs/a7f2/cancel"]


def test_launchd_run_exits_zero_when_already_running(paths: Paths) -> None:
    paths.ensure()
    lock = InstanceLock.acquire(paths)
    try:
        assert daemon_main(["run", "--launchd"]) == EXIT_OK
        assert daemon_main(["run"]) == EXIT_STATE  # interactive runs still report it
    finally:
        lock.release()


def test_ensure_tightens_existing_home(tmp_path: Path) -> None:
    home = tmp_path / "gpu"
    home.mkdir(mode=0o755)
    (home / "logs").mkdir(mode=0o755)
    os.chmod(home, 0o755)
    os.chmod(home / "logs", 0o755)
    Paths(home=home).ensure()
    assert stat.S_IMODE(home.stat().st_mode) == 0o700
    assert stat.S_IMODE((home / "logs").stat().st_mode) == 0o700


async def test_new_specs_with_secret_looking_env_are_refused(api: Api) -> None:
    """D39: the daemon refuses `*_KEY` names and token-shaped values at submit and dry
    route (400 invalid_spec), before bundling; the JobSpec validator is unchanged."""
    for env in ({"KAGGLE_KEY": "abc"}, {"HF": "hf_" + "q" * 34}):
        for path in ("/v1/jobs", "/v1/route"):
            resp = await api.client.post(path, json={"spec": api.spec(env=env)})
            assert resp.status_code == 400, (path, env)
            err = resp.json()["error"]
            assert err["code"] == "invalid_spec"
            assert "gpu secrets set" in err["message"]
            assert "hf_" not in err["message"]
    assert api.runtime.store.list_jobs() == []
