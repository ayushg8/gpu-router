"""Contract: the error taxonomy (rule A3) on every call. Owner: group B.

Adapters raise only `AdapterError` subclasses, with the class that tells the engine what
to do: NotFound for runs the provider does not know, Unavailable for ambiguous submits,
QuotaExhausted (definitive, with resets_at when known) when free quota is gone.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gpu_router.adapters.base import Adapter, RemotePhase, RemoteRef
from gpu_router.errors import (
    DEFINITIVE_SUBMIT_ERRORS,
    AdapterError,
    NotFound,
    QuotaExhausted,
    Unavailable,
)
from tests.contract.harness import ContractTarget, make_ctx, make_job

TERMINAL = {p for p in RemotePhase if p.terminal}
UNKNOWN = RemoteRef(remote_id="does-not-exist-000")


def test_logs_of_unknown_run_raise_not_found(target: ContractTarget, adapter: Adapter) -> None:
    with pytest.raises(NotFound):
        list(adapter.logs(UNKNOWN))


def test_fetch_of_unknown_run_raises_not_found(
    target: ContractTarget, adapter: Adapter, tmp_path: Path
) -> None:
    with pytest.raises(NotFound):
        adapter.fetch(UNKNOWN, tmp_path / "out")


def test_lookup_of_unknown_key_is_none(target: ContractTarget, adapter: Adapter) -> None:
    if not adapter.capabilities.lookup_by_key:
        pytest.skip("adapter does not support lookup_by_key")
    assert adapter.lookup_by_key("gpu-000000000000-1") is None


def test_failed_run_has_no_outputs_or_raises_not_found(
    target: ContractTarget, adapter: Adapter, tmp_path: Path
) -> None:
    if not target.supports_directives:
        pytest.skip("needs fake directives")
    job = make_job(target, directives={"duration": 2, "exit_code": 2})
    ref = adapter.submit(job, make_ctx(job))
    target.wait_for_phase(adapter, ref, TERMINAL)
    with pytest.raises(NotFound):
        adapter.fetch(ref, tmp_path / "out")


def test_unavailable_submit_is_ambiguous_and_resolvable(
    target: ContractTarget, adapter: Adapter
) -> None:
    if not target.supports_directives:
        pytest.skip("needs fake directives")
    job = make_job(target, directives={"unavailable_n": 1, "duration": 1})
    ctx = make_ctx(job)
    with pytest.raises(Unavailable) as info:
        adapter.submit(job, ctx)
    assert not isinstance(info.value, DEFINITIVE_SUBMIT_ERRORS)
    if adapter.capabilities.lookup_by_key:
        assert adapter.lookup_by_key(ctx.attempt_key) is None
    retry = make_ctx(job, 2)
    ref = adapter.submit(job, retry)
    assert ref.remote_id


def test_quota_exhausted_is_definitive_and_says_when(
    target: ContractTarget, adapter: Adapter
) -> None:
    if not target.supports_directives:
        pytest.skip("needs fake directives")
    first = make_job(target, directives={"duration": 30, "quota_limit": 5, "steps": 5})
    ref = adapter.submit(first, make_ctx(first))
    st = target.wait_for_phase(adapter, ref, TERMINAL)
    assert st.phase is RemotePhase.LOST
    assert st.quota_exhausted
    second = make_job(target, directives={"duration": 1, "quota_limit": 5})
    ctx = make_ctx(second)
    with pytest.raises(QuotaExhausted) as info:
        adapter.submit(second, ctx)
    assert isinstance(info.value, DEFINITIVE_SUBMIT_ERRORS)
    assert info.value.resets_at is not None
    assert info.value.resets_at > target.clock.now()
    if adapter.capabilities.lookup_by_key:
        assert adapter.lookup_by_key(ctx.attempt_key) is None


def test_errors_carry_provider_and_message(target: ContractTarget, adapter: Adapter) -> None:
    with pytest.raises(AdapterError) as info:
        adapter.status(UNKNOWN)
    assert info.value.message
    assert info.value.provider in (None, target.name)
