"""Contract: quota() and healthcheck() (A1, A6). Owner: group B."""

from __future__ import annotations

from gpu_router.adapters.base import Adapter, Health
from gpu_router.models import ProviderHealth, QuotaSnapshot
from tests.contract.harness import ContractTarget


def test_quota_snapshot_shape(target: ContractTarget, adapter: Adapter) -> None:
    q = adapter.quota()
    assert isinstance(q, QuotaSnapshot)
    assert q.provider == target.name
    assert q.used >= 0
    assert q.source in ("live", "estimate")
    if adapter.capabilities.live_quota:
        assert q.source == "live"


def test_healthcheck_ok_or_explains(target: ContractTarget, adapter: Adapter) -> None:
    h = adapter.healthcheck()
    assert isinstance(h, Health)
    if h.health is not ProviderHealth.OK:
        assert h.reason


def test_capabilities_are_consistent(target: ContractTarget, adapter: Adapter) -> None:
    caps = adapter.capabilities
    assert caps.max_concurrency >= 1
    assert caps.poll_interval_s > 0
