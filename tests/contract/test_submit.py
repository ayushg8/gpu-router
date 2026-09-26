"""Contract: submit() and lookup_by_key() (rules A1, A3, A4). Owner: group B."""

from __future__ import annotations

import pytest

from gpu_router.adapters.base import Adapter, RemotePhase, RemoteRef
from gpu_router.errors import DEFINITIVE_SUBMIT_ERRORS, AdapterError
from tests.contract.harness import ContractTarget, make_ctx, make_job


def test_submit_returns_remote_ref(target: ContractTarget, adapter: Adapter) -> None:
    job = make_job(target)
    ref = adapter.submit(job, make_ctx(job))
    assert isinstance(ref, RemoteRef)
    assert ref.remote_id
    st = adapter.status(ref)
    assert st.phase in {RemotePhase.PENDING, RemotePhase.RUNNING, RemotePhase.SUCCEEDED}


def test_submit_is_idempotent_per_attempt_key(target: ContractTarget, adapter: Adapter) -> None:
    job = make_job(target)
    ctx = make_ctx(job)
    first = adapter.submit(job, ctx)
    second = adapter.submit(job, ctx)
    assert second.remote_id == first.remote_id


def test_new_attempt_key_gets_new_run(target: ContractTarget, adapter: Adapter) -> None:
    job = make_job(target)
    a = adapter.submit(job, make_ctx(job, 1))
    b = adapter.submit(job, make_ctx(job, 2))
    assert a.remote_id != b.remote_id


def test_lookup_by_key_finds_submitted_run(target: ContractTarget, adapter: Adapter) -> None:
    if not adapter.capabilities.lookup_by_key:
        pytest.skip("adapter does not support lookup_by_key")
    job = make_job(target)
    ctx = make_ctx(job)
    assert adapter.lookup_by_key(ctx.attempt_key) is None
    ref = adapter.submit(job, ctx)
    found = adapter.lookup_by_key(ctx.attempt_key)
    assert found is not None
    assert found.remote_id == ref.remote_id


@pytest.mark.parametrize(
    ("directives", "error_name"),
    [
        ({"rate_limit_n": 1}, "RateLimited"),
        ({"invalid": True}, "InvalidJob"),
        ({"permanent": True}, "Permanent"),
        ({"auth_required": True}, "AuthRequired"),
    ],
)
def test_definitive_submit_errors_create_nothing(
    target: ContractTarget, adapter: Adapter, directives: dict[str, object], error_name: str
) -> None:
    if not target.supports_directives:
        pytest.skip("needs fake directives")
    job = make_job(target, directives=directives)
    ctx = make_ctx(job)
    with pytest.raises(AdapterError) as info:
        adapter.submit(job, ctx)
    assert type(info.value).__name__ == error_name
    assert isinstance(info.value, DEFINITIVE_SUBMIT_ERRORS)
    if adapter.capabilities.lookup_by_key:
        assert adapter.lookup_by_key(ctx.attempt_key) is None
