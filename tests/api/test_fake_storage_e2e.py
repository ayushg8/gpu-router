"""D43 end to end through a real DaemonRuntime: with `checkpoint.fake_storage` on (test mode),
a fake job that dies mid-run resumes on a new attempt from its checkpoint in the LOCAL
storage backend (`<home>/storage/jobs/<id>/ckpt-NNNN`), handed over as GPU_RESUME_URI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from gpu_router.api import JobDetail
from gpu_router.config import CheckpointConfig, Config
from gpu_router.paths import Paths
from gpu_router.runner import storage as rs
from tests.api.conftest import Api


@pytest.fixture
def test_config(test_config: Config) -> Config:
    # cleanup off: the test inspects the finished job's storage (D44 deletes it by default)
    test_config.checkpoint = CheckpointConfig(fake_storage=True, cleanup=False)
    return test_config


async def _detail(api: Api, job_id: str) -> JobDetail:
    resp = await api.client.get(f"/v1/jobs/{job_id}")
    assert resp.status_code == 200, resp.text
    return JobDetail.model_validate(resp.json())


async def _logs(api: Api, job_id: str, attempt: int) -> list[str]:
    resp = await api.client.get(f"/v1/jobs/{job_id}/logs", params={"attempt": attempt})
    assert resp.status_code == 200, resp.text
    rows: list[dict[str, Any]] = [json.loads(x) for x in resp.text.splitlines() if x.strip()]
    return [r["line"] for r in rows if "line" in r]


async def test_fake_job_dies_and_resumes_from_local_storage(api: Api, paths: Paths) -> None:
    directives = {
        "duration": 6,
        "steps": 6,
        "checkpoint_every": 1,
        "attempts": {"1": {"die_after": 3.5}},
    }
    job = await api.submit(provider_options={"fake": directives})
    job_id = job["id"]
    await api.drive(lambda: api.state(job_id) in ("done", "failed"), max_s=600)

    detail = await _detail(api, job_id)
    assert detail.job.state == "done", [(e.reason, e.message) for e in detail.events]
    first, second = detail.attempts
    assert (first.state, second.state) == ("lost", "succeeded")

    store = rs.LocalStore(paths.home / "storage")
    prefix = store.uri(rs.job_prefix(job_id)) + "/ckpt-"
    assert detail.checkpoints
    assert all(c.uri.startswith(prefix) for c in detail.checkpoints)
    before = max((c for c in detail.checkpoints if c.attempt_id == first.id), key=lambda c: c.seq)
    assert second.resume_checkpoint_id == before.id
    # the simulated runner really wrote it, and attempt 2 was told to restore it from there
    assert store.path(rs.ckpt_key(job_id, before.seq) + "/state.json").is_file()
    lines = await _logs(api, job_id, 2)
    assert f"resuming from checkpoint {before.seq}" in lines
    assert f"fake: restored {before.uri}" in lines
    after = [c for c in detail.checkpoints if c.attempt_id == second.id]
    assert after
    assert all(c.seq > before.seq for c in after)
    latest = rs.read_latest(store, job_id)
    assert latest is not None
    assert latest["seq"] == max(c.seq for c in detail.checkpoints)
    assert latest["attempt"] == 2
    assert rs.read_owner(store, job_id) == 2
    assert (Path(api.project) / "runs" / job_id[:4] / "result.json").exists()
