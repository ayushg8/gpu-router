"""Contract: status() lifecycle and errors (rules A3, A6). Owner: group B."""

from __future__ import annotations

import pytest

from gpu_router.adapters.base import Adapter, RemotePhase, RemoteRef
from gpu_router.errors import NotFound
from tests.contract.harness import ContractTarget, make_ctx, make_job

TERMINAL = {p for p in RemotePhase if p.terminal}


def test_run_reaches_succeeded_with_exit_code_zero(
    target: ContractTarget, adapter: Adapter
) -> None:
    job = make_job(target)
    ref = adapter.submit(job, make_ctx(job))
    st = target.wait_for_phase(adapter, ref, TERMINAL)
    assert st.phase is RemotePhase.SUCCEEDED
    assert st.exit_code == 0


def test_status_is_side_effect_free(target: ContractTarget, adapter: Adapter) -> None:
    job = make_job(target)
    ref = adapter.submit(job, make_ctx(job))
    assert adapter.status(ref).phase == adapter.status(ref).phase


def test_unknown_remote_id_raises_not_found(target: ContractTarget, adapter: Adapter) -> None:
    with pytest.raises(NotFound):
        adapter.status(RemoteRef(remote_id="does-not-exist-000"))


def test_nonzero_exit_is_failed(target: ContractTarget, adapter: Adapter) -> None:
    if not target.supports_directives:
        pytest.skip("needs fake directives")
    job = make_job(target, directives={"duration": 3, "exit_code": 3})
    ref = adapter.submit(job, make_ctx(job))
    st = target.wait_for_phase(adapter, ref, TERMINAL)
    assert st.phase is RemotePhase.FAILED
    assert st.exit_code == 3


def test_session_death_is_lost(target: ContractTarget, adapter: Adapter) -> None:
    if not target.supports_directives:
        pytest.skip("needs fake directives")
    job = make_job(target, directives={"duration": 30, "die_after": 2})
    ref = adapter.submit(job, make_ctx(job))
    st = target.wait_for_phase(adapter, ref, TERMINAL)
    assert st.phase is RemotePhase.LOST
    assert st.lost_reason
