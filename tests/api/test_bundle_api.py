"""Phase 2 wiring: POST /v1/jobs builds the bundle, records jobs.bundle_sha256 and
materializes jobs/<id>/bundle{,.tar.gz} before the driver submits."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from gpu_router.packaging.bundle import cached_archive
from tests.api.conftest import Api


def _write(api: Api, files: dict[str, str]) -> None:
    for rel, text in files.items():
        path = Path(api.project) / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


async def test_submit_builds_and_materializes_bundle(api: Api) -> None:
    _write(api, {"train.py": "import gpu\n", "requirements.txt": "numpy\n"})
    job: dict[str, Any] = await api.submit()
    sha = job["bundle_sha256"]
    assert isinstance(sha, str)
    assert len(sha) == 64
    paths = api.runtime.paths
    assert cached_archive(paths, sha).is_file()
    bundle_dir = paths.job_bundle_dir(job["id"])
    cached = cached_archive(paths, sha).read_bytes()
    assert paths.job_bundle_archive(job["id"]).read_bytes() == cached
    manifest = json.loads((bundle_dir / "manifest.json").read_text())
    assert manifest["deps"]["kind"] == "requirements"
    assert (bundle_dir / "code" / "train.py").is_file()
    assert (bundle_dir / "gpu_runner" / "gpu.py").is_file()
    await api.drive(lambda: api.state(job["id"]) == "done")


async def test_same_project_twice_reuses_cached_bundle(api: Api) -> None:
    _write(api, {"train.py": "print(1)\n"})
    first = await api.submit()
    second = await api.submit()
    assert first["id"] != second["id"]
    assert first["bundle_sha256"] == second["bundle_sha256"]


async def test_missing_project_dir_is_invalid_spec_and_creates_no_job(api: Api) -> None:
    resp = await api.client.post(
        "/v1/jobs", json={"spec": api.spec(project_dir=str(Path(api.project) / "nope"))}
    )
    assert resp.status_code == 400
    err = resp.json()["error"]
    assert err["code"] == "invalid_spec"
    assert "does not exist" in err["message"]
    assert (await api.client.get("/v1/jobs")).json()["jobs"] == []


async def test_oversized_bundle_is_rejected_with_hint(api: Api) -> None:
    _write(api, {"train.py": "", "big.txt": "x" * 4096})
    bundler = api.runtime.supervisor.deps.bundler
    assert bundler is not None
    bundler.max_mb = 0.001
    resp = await api.client.post("/v1/jobs", json={"spec": api.spec()})
    assert resp.status_code == 400
    err = resp.json()["error"]
    assert err["code"] == "invalid_spec"
    assert "big.txt" in err["message"]
    assert ".gitignore" in err["hint"]


async def test_driver_heals_missing_materialized_bundle(api: Api) -> None:
    """Crash between create_job and materialize: the driver rebuilds jobs/<id>/bundle from
    the cache when it builds the AttemptContext."""
    import shutil

    from gpu_router.engine.driver import JobDriver

    _write(api, {"train.py": ""})
    job = await api.submit()
    store = api.runtime.store
    await api.drive(lambda: bool(store.attempts_for(job["id"])))
    paths = api.runtime.paths
    shutil.rmtree(paths.job_bundle_dir(job["id"]))
    paths.job_bundle_archive(job["id"]).unlink()
    driver = JobDriver(job["id"], api.runtime.supervisor.deps)
    ctx = driver._attempt_context(store.get_job(job["id"]), store.attempts_for(job["id"])[0])
    assert ctx.bundle_dir == paths.job_bundle_dir(job["id"])
    assert ctx.bundle_archive == paths.job_bundle_archive(job["id"])
    assert (paths.job_bundle_dir(job["id"]) / "manifest.json").is_file()


async def test_materialize_failure_at_submit_still_starts_the_driver(
    api: Api, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review finding: materialize ran after create_job committed; an error there (full
    disk, corrupt cache entry) answered 500 and left the job queued with no driver."""
    _write(api, {"train.py": ""})
    bundler = api.runtime.supervisor.deps.bundler
    assert bundler is not None
    real = bundler.materialize
    calls = {"n": 0}

    def flaky(sha: str, job_id: str) -> tuple[Path, Path]:
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError(28, "No space left on device")
        return real(sha, job_id)

    monkeypatch.setattr(bundler, "materialize", flaky)
    job = await api.submit()
    assert job["id"] in api.runtime.supervisor._drivers
    await api.drive(lambda: api.state(job["id"]) == "done")
    assert calls["n"] >= 2  # the driver materialized it before submitting
    assert (api.runtime.paths.job_bundle_dir(job["id"]) / "manifest.json").is_file()
