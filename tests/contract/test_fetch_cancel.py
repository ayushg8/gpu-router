"""Contract: fetch() (A9) and cancel() (A5). Owner: group B."""

from __future__ import annotations

from pathlib import Path

from gpu_router.adapters.base import Adapter, RemotePhase, RemoteRef
from tests.contract.harness import ContractTarget, make_ctx, make_job

TERMINAL = {p for p in RemotePhase if p.terminal}


def test_fetch_writes_outputs_and_is_rerunnable(
    target: ContractTarget, adapter: Adapter, tmp_path: Path
) -> None:
    if not adapter.capabilities.fetch:
        return
    job = make_job(target)
    ref = adapter.submit(job, make_ctx(job))
    target.wait_for_phase(adapter, ref, {RemotePhase.SUCCEEDED})
    dest = tmp_path / "out"
    first = adapter.fetch(ref, dest)
    assert first.files >= 1
    assert any(dest.iterdir())
    (dest / "user-note.txt").write_text("keep me")
    second = adapter.fetch(ref, dest)
    assert second.files == first.files
    assert (dest / "user-note.txt").read_text() == "keep me"


def test_cancel_stops_run(target: ContractTarget, adapter: Adapter) -> None:
    job = make_job(target, directives={"duration": 600} if target.supports_directives else None)
    ref = adapter.submit(job, make_ctx(job))
    adapter.cancel(ref)
    st = target.wait_for_phase(adapter, ref, TERMINAL)
    assert st.phase is RemotePhase.CANCELLED


def test_cancel_is_idempotent(target: ContractTarget, adapter: Adapter) -> None:
    job = make_job(target)
    ref = adapter.submit(job, make_ctx(job))
    target.wait_for_phase(adapter, ref, TERMINAL)
    adapter.cancel(ref)
    adapter.cancel(ref)
    adapter.cancel(RemoteRef(remote_id="does-not-exist-000"))
