"""Datasets staged in the provider's own store (adapter.stage_data; Kaggle datasets) when
there is no Hugging Face storage (2026-10-04 field test: `data=` could not reach Kaggle
with checkpoint.backend local). The engine digests the dataset, asks the adapter to stage
it, hands the runner the adapter's uri, and waits on transient trouble."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from gpu_router.adapters.base import StagedData
from gpu_router.adapters.fake import FakeAdapter
from gpu_router.clock import FakeClock
from gpu_router.errors import InvalidJob, Unavailable
from gpu_router.models import DataRef, JobState, Reason
from gpu_router.statemachine import is_terminal
from tests.unit.checkpoint.test_engine_storage import (  # fixtures + harness
    CEngine,
    _data_dir,
    ceng,
)

__all__ = ["ceng"]


@pytest.fixture
def hub_kw() -> dict[str, Any]:
    return {"local_kinds": frozenset({"local"})}  # the fakes are remote, no HF token


class Stager:
    """stage_data for a FakeAdapter: records calls; `fail` = errors raised first."""

    def __init__(self, fail: Sequence[Exception] = ()) -> None:
        self.calls: list[tuple[Path, str, tuple[tuple[str, int], ...]]] = []
        self.fail = list(fail)
        self.stored: set[str] = set()

    def __call__(self, path: Path, sha256: str, files: Sequence[tuple[str, int]]) -> StagedData:
        self.calls.append((path, sha256, tuple(files)))
        if self.fail:
            raise self.fail.pop(0)
        uploaded = sha256 not in self.stored
        self.stored.add(sha256)
        return StagedData(
            uri=f"kaggle://u/gpu-router-data-{sha256[:16]}/data-{sha256[:16]}.tar.bin",
            uploaded=uploaded,
            where=f"private kaggle dataset u/gpu-router-data-{sha256[:16]}",
        )


async def _with_stager(ceng: CEngine, stager: Stager, name: str = "fake") -> None:
    await ceng.restart()
    adapter = ceng.supervisor.deps.registry.get(name)
    assert isinstance(adapter, FakeAdapter)
    adapter.capabilities = adapter.capabilities.model_copy(update={"stage_data": True})
    adapter.stage_data = stager  # type: ignore[method-assign]


async def test_a_dataset_reaches_a_remote_run_without_hf_storage(ceng: CEngine) -> None:
    stager = Stager()
    await _with_stager(ceng, stager)
    data = _data_dir(ceng, "crops", "a,b\n1,2\n")
    spec = ceng.spec(data=[DataRef(mount="crops", path=str(data))], provider="fake")
    first = await ceng.submit(spec)
    await ceng.run_until(lambda: is_terminal(ceng.job(first.id).state), step=1)
    assert ceng.job(first.id).state is JobState.DONE
    ((path, sha, files),) = stager.calls
    assert path == data.resolve()
    assert files == (("rows.csv", 8),)
    assert ceng.notes(first.id, Reason.DATA_UPLOADING) == []  # tiny: no "uploading" note
    (uploaded,) = ceng.notes(first.id, Reason.DATA_UPLOADED)
    assert f"as a private kaggle dataset u/gpu-router-data-{sha[:16]}" in uploaded.message
    _name, ctx = ceng.contexts[-1]
    assert json.loads(ctx.env["GPU_DATA"]) == [
        {
            "mount": "crops",
            "uri": f"kaggle://u/gpu-router-data-{sha[:16]}/data-{sha[:16]}.tar.bin",
            "sha256": sha,
        }
    ]
    assert ceng.notes(first.id, Reason.PROVIDER_EXCLUDED) == []

    second = await ceng.submit(spec.model_copy(update={"name": "again"}))
    await ceng.run_until(lambda: is_terminal(ceng.job(second.id).state), step=1)
    assert ceng.job(second.id).state is JobState.DONE
    (reused,) = ceng.notes(second.id, Reason.DATA_REUSED)
    assert "already a private kaggle dataset" in reused.message


async def test_slow_staging_waits_and_retries(ceng: CEngine, clock: FakeClock) -> None:
    stager = Stager(fail=[Unavailable("still processing", provider="fake")] * 2)
    await _with_stager(ceng, stager)
    data = _data_dir(ceng, "crops", "x\n")
    job = await ceng.submit(
        ceng.spec(data=[DataRef(mount="crops", path=str(data))], provider="fake")
    )
    await ceng.run_until(lambda: is_terminal(ceng.job(job.id).state), step=5)
    assert ceng.job(job.id).state is JobState.DONE
    assert len(stager.calls) == 3
    waits = ceng.notes(job.id, Reason.RETRY_SCHEDULED)
    assert waits
    assert "dataset 'crops' is not on fake yet (still processing)" in waits[0].message
    assert ceng.notes(job.id, Reason.DATA_UPLOADING) == []  # small dataset


async def test_a_big_upload_is_announced_once(
    ceng: CEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gpu_router.engine import driver

    monkeypatch.setattr(driver, "STAGE_NOTE_BYTES", 1)
    stager = Stager(fail=[Unavailable("still processing", provider="fake")])
    await _with_stager(ceng, stager)
    data = _data_dir(ceng, "crops", "x\n")
    job = await ceng.submit(
        ceng.spec(data=[DataRef(mount="crops", path=str(data))], provider="fake")
    )
    await ceng.run_until(lambda: is_terminal(ceng.job(job.id).state), step=5)
    (note,) = ceng.notes(job.id, Reason.DATA_UPLOADING)  # once, though staged twice
    assert note.message.startswith("uploading dataset crops (1 files, 2 B) to fake unless")


async def test_a_refused_dataset_excludes_the_provider(ceng: CEngine) -> None:
    stager = Stager(fail=[InvalidJob("not enough free disk", provider="fake")])
    await _with_stager(ceng, stager)
    data = _data_dir(ceng, "crops", "x\n")
    job = await ceng.submit(
        ceng.spec(data=[DataRef(mount="crops", path=str(data))], provider="fake")
    )
    await ceng.run_until(lambda: is_terminal(ceng.job(job.id).state), step=1)
    assert ceng.job(job.id).state is JobState.FAILED  # pinned: nowhere else to go
    (excluded,) = ceng.notes(job.id, Reason.PROVIDER_EXCLUDED)
    assert "cannot be put on fake (not enough free disk)" in excluded.message


async def test_routing_skips_providers_the_data_cannot_reach(ceng: CEngine) -> None:
    """2026-10-04 field test: gpu_route with data= chose colab, which cannot receive a
    dataset without HF storage; the job would have burned an attempt there first."""
    stager = Stager()
    await _with_stager(ceng, stager, name="fake-b")  # only fake-b keeps datasets itself
    ceng.hub.hf()  # HF was tried and is not available (no token)
    data = _data_dir(ceng, "crops", "x\n")
    job = await ceng.submit(ceng.spec(data=[DataRef(mount="crops", path=str(data))]))
    await ceng.run_until(lambda: is_terminal(ceng.job(job.id).state), step=1)
    assert ceng.job(job.id).state is JobState.DONE
    assert [a.provider for a in ceng.store.attempts_for(job.id)] == ["fake-b"]
    placed = [e for e in ceng.store.events_for(job.id) if e.reason == Reason.PLACED]
    assert placed
    assert "fake-b" in str(placed[0].message)


async def test_an_adapter_bug_while_staging_waits_like_an_outage(ceng: CEngine) -> None:
    """Review of D61: an OSError inside stage_data reaches the driver as an
    AdapterContractViolation, which is no AdapterError; invariant 7 says treat it like
    Unavailable (it used to count toward internal_error_limit and fail the job)."""
    stager = Stager(fail=[RuntimeError("disk full while packing")])
    await _with_stager(ceng, stager)
    data = _data_dir(ceng, "crops", "x\n")
    job = await ceng.submit(
        ceng.spec(data=[DataRef(mount="crops", path=str(data))], provider="fake")
    )
    await ceng.run_until(lambda: is_terminal(ceng.job(job.id).state), step=5)
    assert ceng.job(job.id).state is JobState.DONE
    assert len(stager.calls) == 2
    assert ceng.notes(job.id, Reason.INTERNAL_ERROR) == []


async def test_a_login_problem_while_staging_marks_the_provider(ceng: CEngine) -> None:
    from gpu_router.errors import AuthRequired
    from gpu_router.models import ProviderHealth

    stager = Stager(fail=[AuthRequired("kaggle needs login", provider="fake")])
    await _with_stager(ceng, stager)
    data = _data_dir(ceng, "crops", "x\n")
    job = await ceng.submit(
        ceng.spec(data=[DataRef(mount="crops", path=str(data))], provider="fake")
    )
    await ceng.run_until(
        lambda: ceng.store.get_provider_state("fake").health is ProviderHealth.AUTH_REQUIRED
    )
    assert ceng.notes(job.id, Reason.PROVIDER_EXCLUDED) == []  # not the job's fault
    assert not is_terminal(ceng.job(job.id).state)  # waits for `gpu login`
