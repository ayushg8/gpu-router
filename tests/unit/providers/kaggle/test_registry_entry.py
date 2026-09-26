"""The registry builds KaggleAdapter for the packaged `kaggle` catalog entry."""

from __future__ import annotations

from dataclasses import replace

import pytest

from gpu_router.adapters.registry import AdapterRegistry, adapter_class
from gpu_router.clock import FakeClock
from gpu_router.config import Config
from gpu_router.errors import AuthRequired
from gpu_router.models import ProviderHealth
from gpu_router.paths import Paths
from gpu_router.providers.catalog import load_catalog
from gpu_router.providers.kaggle.adapter import KaggleAdapter
from tests.contract.kaggle.targets import kaggle_deps


def test_registry_builds_the_kaggle_adapter(paths: Paths) -> None:
    assert adapter_class("kaggle") is KaggleAdapter
    registry = AdapterRegistry.build(
        config=Config(), catalog=load_catalog(), paths=paths, clock=FakeClock()
    )
    adapter = registry.get("kaggle")
    assert isinstance(adapter, KaggleAdapter)
    entry = load_catalog().get("kaggle")
    assert {g.name for g in entry.gpus} == {"T4", "P100"}
    assert adapter.capabilities.max_session_hours == entry.session_hours
    assert adapter.capabilities.poll_interval_s == entry.poll_interval_s


def test_test_mode_daemon_never_calls_the_real_cli(
    paths: Paths, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Invariant 20: in test mode (GPU_ROUTER_TEST_MODE) the adapter stays offline (DISABLED
    health, refused calls) unless GPU_ROUTER_REAL_PROVIDERS lists kaggle, whether or not the
    registry registers it."""

    def boom(*_a: object, **_k: object) -> None:
        raise AssertionError("the real kaggle CLI was called")

    monkeypatch.setattr("gpu_router.providers.kaggle.cli.SubprocessRunner.__call__", boom)
    monkeypatch.delenv("GPU_ROUTER_REAL_PROVIDERS", raising=False)
    registry = AdapterRegistry.build(
        config=Config(test_mode=True), catalog=load_catalog(), paths=paths, clock=FakeClock()
    )
    if "kaggle" in registry:
        assert registry.get("kaggle").healthcheck().health is ProviderHealth.DISABLED
    deps = replace(kaggle_deps(paths, FakeClock()), test_mode=True)
    adapter = KaggleAdapter(deps)
    health = adapter.healthcheck()
    assert health.health is ProviderHealth.DISABLED
    assert health.hint is not None
    assert "GPU_ROUTER_REAL_PROVIDERS" in health.hint
    with pytest.raises(AuthRequired):
        adapter.quota()
    assert adapter.lookup_by_key("gpu-0123456789ab-1") is None
    monkeypatch.setenv("GPU_ROUTER_REAL_PROVIDERS", "kaggle")
    with pytest.raises(AssertionError, match="real kaggle CLI was called"):
        KaggleAdapter(deps).quota()
