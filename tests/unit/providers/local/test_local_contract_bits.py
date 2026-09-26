"""LocalAdapter: ids, idempotency, error taxonomy, quota, healthcheck, registry wiring."""

from __future__ import annotations

import platform
import sys
from pathlib import Path

import pytest

from gpu_router.adapters.base import RemotePhase, RemoteRef
from gpu_router.errors import InvalidJob, NotFound
from gpu_router.models import ProviderHealth
from gpu_router.paths import Paths
from tests.unit.providers.local.helpers import (
    build_project,
    make_adapter,
    make_bundle,
    make_ctx,
    make_job,
    wait_phase,
)

APPLE_SILICON = sys.platform == "darwin" and platform.machine() == "arm64"


def test_submit_is_idempotent_per_key_and_lookup_finds_it(paths: Paths, tmp_path: Path) -> None:
    adapter = make_adapter(paths)
    project = build_project(tmp_path, "print('once')\n")
    bundle = make_bundle(paths, project)
    job = make_job(project)
    ctx = make_ctx(job, bundle)
    assert adapter.lookup_by_key(ctx.attempt_key) is None
    first = adapter.submit(job, ctx)
    second = adapter.submit(job, ctx)
    assert first == second
    found = adapter.lookup_by_key(ctx.attempt_key)
    assert found == first
    other = adapter.submit(job, make_ctx(job, bundle, n=2))
    assert other.remote_id != first.remote_id
    wait_phase(adapter, first)
    wait_phase(adapter, other)
    log = (adapter.runs_root / first.remote_id / "console.log").read_text()
    assert log.count('::gpu:: {"t":"hello"') == 1  # one launch for two submits


@pytest.mark.parametrize(
    "remote_id", ["does-not-exist-000", "../../etc", "a/b", ".hidden", "", "x" * 200]
)
def test_unknown_or_unsafe_ids_are_not_found_everywhere(
    paths: Paths, tmp_path: Path, remote_id: str
) -> None:
    adapter = make_adapter(paths)
    ref = RemoteRef.model_construct(remote_id=remote_id, url=None, meta={})
    with pytest.raises(NotFound) as info:
        adapter.status(ref)
    assert info.value.provider == "local"
    assert info.value.message
    with pytest.raises(NotFound):
        list(adapter.logs(ref))
    with pytest.raises(NotFound):
        adapter.fetch(ref, tmp_path / "out")
    adapter.cancel(ref)
    assert adapter.lookup_by_key(remote_id) is None
    assert not (tmp_path / "out").exists()


def test_unsafe_attempt_key_is_invalid_job(paths: Paths, tmp_path: Path) -> None:
    adapter = make_adapter(paths)
    project = build_project(tmp_path, "print(1)\n")
    job = make_job(project)
    ctx = make_ctx(job, make_bundle(paths, project)).model_copy(update={"attempt_key": "../x"})
    with pytest.raises(InvalidJob):
        adapter.submit(job, ctx)


def test_interactive_jobs_are_refused_before_anything_starts(paths: Paths, tmp_path: Path) -> None:
    adapter = make_adapter(paths)
    project = build_project(tmp_path, "print(1)\n")
    job = make_job(project, interactive=True)
    ctx = make_ctx(job, make_bundle(paths, project))
    with pytest.raises(InvalidJob):
        adapter.submit(job, ctx)
    assert adapter.lookup_by_key(ctx.attempt_key) is None


def test_quota_is_live_and_unlimited(paths: Paths) -> None:
    adapter = make_adapter(paths)
    q = adapter.quota()
    assert q.provider == "local"
    assert q.source == "live"
    assert q.limit is None
    assert q.used == 0
    assert adapter.capabilities.live_quota


@pytest.mark.skipif(not APPLE_SILICON, reason="needs an Apple Silicon Mac")
def test_healthcheck_ok_on_apple_silicon_and_side_effect_free(paths: Paths) -> None:
    adapter = make_adapter(paths)
    health = adapter.healthcheck()
    assert health.health is ProviderHealth.OK
    assert health.detail["machine"] == "arm64"
    assert health.detail["env"] == "system"
    assert not (paths.home / "local").exists()
    assert not paths.provider_dir("local").exists()


def test_healthcheck_explains_a_non_mac(paths: Paths, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    health = make_adapter(paths).healthcheck()
    assert health.health is ProviderHealth.UNAVAILABLE
    assert health.reason
    assert health.hint


def test_healthcheck_explains_an_intel_python(
    paths: Paths, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    health = make_adapter(paths).healthcheck()
    assert health.health is ProviderHealth.UNAVAILABLE
    assert health.reason is not None
    assert "x86_64" in health.reason


@pytest.mark.skipif(not APPLE_SILICON, reason="needs an Apple Silicon Mac")
def test_healthcheck_degrades_on_a_full_disk(paths: Paths, monkeypatch: pytest.MonkeyPatch) -> None:
    import shutil

    from gpu_router.providers.local import adapter as local_adapter

    monkeypatch.setattr(
        local_adapter.shutil, "disk_usage", lambda _p: shutil._ntuple_diskusage(10, 9, 1)
    )
    health = make_adapter(paths).healthcheck()
    assert health.health is ProviderHealth.DEGRADED
    assert health.reason is not None
    assert "free" in health.reason


def test_registry_builds_the_local_adapter(paths: Paths) -> None:
    from gpu_router.adapters.registry import adapter_class
    from gpu_router.providers.local.adapter import LocalAdapter

    assert adapter_class("local") is LocalAdapter


def test_capabilities_are_honest() -> None:
    from gpu_router.providers.local.adapter import LocalAdapter

    caps = LocalAdapter.capabilities
    assert caps.lookup_by_key
    assert caps.resume
    assert caps.cancel_confirms
    assert caps.live_logs
    assert not caps.interactive
    assert caps.max_concurrency == 1


def test_a_succeeded_run_reports_times(paths: Paths, tmp_path: Path) -> None:
    adapter = make_adapter(paths)
    project = build_project(tmp_path, "print(1)\n")
    job = make_job(project)
    ref = adapter.submit(job, make_ctx(job, make_bundle(paths, project)))
    st = wait_phase(adapter, ref, {RemotePhase.SUCCEEDED})
    assert st.started_at is not None
    assert st.ended_at is not None
    assert st.ended_at >= st.started_at
