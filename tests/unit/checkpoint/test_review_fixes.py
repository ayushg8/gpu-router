"""Phase 4/5 review regressions for checkpoint storage, handoffs and datasets (D44).

Hub-level cases use the in-memory FakeHfApi; engine-level ones the FakeClock engine with a
real CheckpointHub (tests/unit/checkpoint/test_engine_storage.py's harness)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from gpu_router import secrets
from gpu_router.checkpoint import tokens
from gpu_router.checkpoint.data import digest, scan
from gpu_router.checkpoint.storage import StorageError
from gpu_router.clock import FakeClock
from gpu_router.config import CheckpointConfig
from gpu_router.errors import SecretsError
from gpu_router.models import (
    AttemptState,
    DataRef,
    FailureKind,
    JobState,
    Reason,
)
from gpu_router.paths import Paths
from gpu_router.runner import storage as rs
from gpu_router.statemachine import is_terminal
from tests.unit.checkpoint.fake_hfapi import FakeHfApi, HTTPError
from tests.unit.checkpoint.test_engine_storage import (  # fixtures + builders
    HOUR,
    TOKEN,
    CEngine,
    ceng,
    hub_kw,
)
from tests.unit.checkpoint.test_engine_storage import make_hub as make_engine_hub
from tests.unit.checkpoint.test_hub import REMOTE, ckpt, make_hub

__all__ = ["ceng", "hub_kw"]  # re-exported pytest fixtures

BUCKET = "tester/gpu-router"
HF_ROOT = f"hf://buckets/{BUCKET}"


def _tokens() -> None:
    secrets.set_secret("HF_TOKEN", TOKEN)
    secrets.set_secret("HF_TOKEN_REMOTE", REMOTE)


def _publish(store: Any, job: str, seq: int, tmp: Path, text: str, attempt: int = 1) -> str:
    """A real checkpoint (state.json) with a manifest carrying its size and sha256."""
    import hashlib

    src = tmp / f"stage-{job}-{seq}-{attempt}"
    src.mkdir(parents=True, exist_ok=True)
    data = text.encode()
    (src / "state.json").write_bytes(data)
    sha = hashlib.sha256(data).hexdigest()
    latest = rs.publish_checkpoint(
        store, job, seq, src, [("state.json", len(data), sha)], attempt=attempt, step=seq
    )
    return str(latest["uri"])


# --------------------------------------------------------------------------- placement waits


def test_hf_down_at_placement_waits_instead_of_starting_from_step_0(
    paths: Paths, clock: FakeClock
) -> None:
    """Finding: one failed create_bucket (Wi-Fi not up yet at login) made hf() None for
    300 s; a job placed then silently lost its hf:// resume and started from step 0."""
    _tokens()
    api = FakeHfApi()
    api.fail["create_bucket"] = [HTTPError(503)]
    hub = make_hub(paths, clock, api)
    resume = ckpt("j", 7, f"{HF_ROOT}/jobs/j/ckpt-0007")
    with pytest.raises(StorageError) as remote:
        hub.prepare_attempt(job_id="j", attempt_n=2, kind="kaggle", resume=resume)
    assert remote.value.retryable
    assert "cannot be reached right now" in remote.value.message
    with pytest.raises(StorageError) as local:
        hub.prepare_attempt(job_id="j", attempt_n=2, kind="local", resume=resume)
    assert local.value.retryable
    # after the engine's wait: go ahead without it, and say so
    assert (
        hub.prepare_attempt(job_id="j", attempt_n=2, kind="kaggle", resume=resume, degrade=True)
        is None
    )
    got = hub.prepare_attempt(job_id="j", attempt_n=2, kind="local", resume=resume, degrade=True)
    assert got is not None
    assert "GPU_RESUME_URI" not in got.env
    assert "cannot be reached" in got.notes[0]
    assert "starts over" in got.notes[0]
    # HF comes back: the resume URI is handed over as is
    clock.advance(301)
    got = hub.prepare_attempt(job_id="j", attempt_n=3, kind="kaggle", resume=resume)
    assert got is not None
    assert got.env["GPU_RESUME_URI"] == resume.uri


def test_no_token_is_not_a_reason_to_wait(paths: Paths, clock: FakeClock) -> None:
    api = FakeHfApi()
    hub = make_hub(paths, clock, api)
    resume = ckpt("j", 7, f"{HF_ROOT}/jobs/j/ckpt-0007")
    assert hub.prepare_attempt(job_id="j", attempt_n=2, kind="kaggle", resume=resume) is None
    got = hub.prepare_attempt(job_id="j", attempt_n=2, kind="local", resume=resume)
    assert got is not None
    assert "starts over" in got.notes[0]


def test_a_locked_keychain_at_the_recheck_keeps_the_bucket(
    paths: Paths, clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    _tokens()
    hub = make_hub(paths, clock, FakeHfApi())
    first = hub.hf()
    assert first is not None

    def locked() -> None:
        raise SecretsError("the login Keychain is locked")

    monkeypatch.setattr(tokens, "admin_token", locked)
    clock.advance(301)
    assert hub.hf() is first


def test_remote_runs_never_get_the_admin_token(paths: Paths, clock: FakeClock) -> None:
    """Finding: without HF_TOKEN_REMOTE the remote runtime got the admin HF_TOKEN, which
    the job (and pip) can read there."""
    secrets.set_secret("HF_TOKEN", TOKEN)
    hub = make_hub(paths, clock, FakeHfApi())
    assert hub.prepare_attempt(job_id="j", attempt_n=1, kind="kaggle", resume=None) is None
    st = hub.status()
    assert "HF_TOKEN_REMOTE" in (st.reason or "")
    assert "gpu login hf --remote" in (st.hint or "")
    secrets.set_secret("HF_TOKEN_REMOTE", REMOTE)
    got = hub.prepare_attempt(job_id="j", attempt_n=2, kind="kaggle", resume=None)
    assert got is not None
    assert got.secrets["GPU_STORAGE_TOKEN"].get_secret_value() == REMOTE


def test_a_refused_claim_degrades_instead_of_wedging(paths: Paths, clock: FakeClock) -> None:
    """Finding: every StorageError before submit was retried forever, including a 403 on
    owner.json (no write access, a full storage quota)."""
    _tokens()
    api = FakeHfApi()
    hub = make_hub(paths, clock, api)
    assert hub.hf() is not None
    api.fail["batch_bucket_files"] = [HTTPError(403)]
    assert hub.prepare_attempt(job_id="j", attempt_n=1, kind="kaggle", resume=None) is None
    assert "cannot write to the storage bucket" in (hub.status().reason or "")
    api.fail["batch_bucket_files"] = [HTTPError(503)]
    with pytest.raises(StorageError):  # transient: the engine waits for it
        hub.prepare_attempt(job_id="j", attempt_n=1, kind="kaggle", resume=None)


# --------------------------------------------------------------------------- copies


def test_a_missing_checkpoint_copy_falls_back_to_the_newest_intact_one(
    paths: Paths, clock: FakeClock, tmp_path: Path
) -> None:
    """Findings: a copy of a pruned checkpoint was a permanent error that wedged the job;
    copies were never checked against the manifest; the target's latest.json stayed
    stale after a copy."""
    _tokens()
    api = FakeHfApi()
    hub = make_hub(paths, clock, api)
    local = hub.local()
    assert local is not None
    for seq in (6, 7, 8):
        _publish(local.raw, "job1", seq, tmp_path, json.dumps({"step": seq}))
    gone = ckpt("job1", 5, local.uri("jobs/job1/ckpt-0005"))  # pruned already
    got = hub.prepare_attempt(job_id="job1", attempt_n=4, kind="kaggle", resume=gone)
    assert got is not None
    assert got.env["GPU_RESUME_URI"] == f"{HF_ROOT}/jobs/job1/ckpt-0008"
    assert "copied checkpoint 8 (the newest intact one)" in got.notes[0]
    files = api.files(BUCKET)
    assert json.loads(files["jobs/job1/ckpt-0008/state.json"]) == {"step": 8}
    latest = json.loads(files["jobs/job1/latest.json"])
    assert (latest["seq"], latest["uri"]) == (8, f"{HF_ROOT}/jobs/job1/ckpt-0008")

    # a torn newest one (bytes differ from the manifest) is skipped for the one before
    (Path(local.raw.root) / "jobs/job1/ckpt-0008/state.json").write_text('{"step": 88888}')
    torn = ckpt("job1", 8, local.uri("jobs/job1/ckpt-0008"))
    hub2 = make_hub(paths, clock, FakeHfApi())
    got2 = hub2.prepare_attempt(job_id="job1", attempt_n=5, kind="kaggle", resume=torn)
    assert got2 is not None
    assert got2.env["GPU_RESUME_URI"] == f"{HF_ROOT}/jobs/job1/ckpt-0007"

    # nothing intact at all: start over with a note, never an endless retry
    for seq in (6, 7):
        (Path(local.raw.root) / f"jobs/job1/ckpt-{seq:04d}/.gpu-ckpt.json").unlink()
    hub3 = make_hub(paths, clock, FakeHfApi())
    got3 = hub3.prepare_attempt(job_id="job1", attempt_n=6, kind="kaggle", resume=torn)
    assert got3 is not None
    assert "GPU_RESUME_URI" not in got3.env
    assert "starts over" in got3.notes[0]


def test_latest_raises_while_hf_is_down(paths: Paths, clock: FakeClock) -> None:
    _tokens()
    api = FakeHfApi()
    api.fail["create_bucket"] = [HTTPError(503)]
    hub = make_hub(paths, clock, api)
    with pytest.raises(StorageError):
        hub.latest("job1")
    clock.advance(301)
    assert hub.latest("job1") is None


# --------------------------------------------------------------------------- engine


@pytest.fixture
def remote_fake_hub_kw() -> dict[str, Any]:
    return {"local_kinds": frozenset({"local"})}


async def _remote_hf_engine(
    ceng: CEngine, paths: Paths, clock: FakeClock, api: FakeHfApi, **cfg: Any
) -> None:
    """fake is a remote kind with HF storage."""
    _tokens()
    ceng.config.checkpoint = CheckpointConfig(handoff_margin_min=30, handoff_wait_min=5, **cfg)
    ceng.hub = make_engine_hub(
        paths, clock, ceng.config.checkpoint, hf_api=api.factory, local_kinds=frozenset({"local"})
    )
    await ceng.restart()


async def test_a_placement_waits_for_hf_then_submits_with_storage(
    ceng: CEngine, paths: Paths, clock: FakeClock
) -> None:
    api = FakeHfApi()
    api.fail["create_bucket"] = [HTTPError(503)]
    await _remote_hf_engine(ceng, paths, clock, api)
    job = await ceng.submit(ceng.spec(provider="fake"))
    await ceng.run_until(lambda: bool(ceng.notes(job.id, Reason.RETRY_SCHEDULED)), step=1)
    assert ceng.contexts == []  # nothing submitted while storage is unreachable
    (waiting,) = ceng.notes(job.id, Reason.RETRY_SCHEDULED)
    assert "cannot be reached" in waiting.message
    await ceng.run_until(lambda: is_terminal(ceng.job(job.id).state), max_s=HOUR, step=5)
    assert ceng.job(job.id).state is JobState.DONE
    _name, ctx = ceng.contexts[-1]
    assert ctx.env["GPU_STORAGE"] == HF_ROOT
    assert len(ceng.store.attempts_for(job.id)) == 1


async def test_the_wait_for_storage_is_bounded(
    ceng: CEngine, paths: Paths, clock: FakeClock
) -> None:
    api = FakeHfApi()
    api.fail["create_bucket"] = [HTTPError(503)] * 1000
    await _remote_hf_engine(ceng, paths, clock, api, storage_wait_s=120)
    job = await ceng.submit(ceng.spec(provider="fake"))
    await ceng.run_until(lambda: is_terminal(ceng.job(job.id).state), max_s=HOUR, step=5)
    assert ceng.job(job.id).state is JobState.DONE
    (went,) = ceng.notes(job.id, Reason.STORAGE_UNAVAILABLE)
    assert "stay on that machine" in went.message
    assert "HTTP 503" in went.message
    waited = went.ts - ceng.store.events_for(job.id)[0].ts
    assert 120 <= waited < 600  # waited for storage, then went ahead
    _name, ctx = ceng.contexts[-1]
    assert "GPU_STORAGE" not in ctx.env
    assert len(ceng.notes(job.id, Reason.RETRY_SCHEDULED)) == 1  # one note, not one per try


async def test_a_refused_claim_runs_the_job_without_storage(
    ceng: CEngine, paths: Paths, clock: FakeClock
) -> None:
    api = FakeHfApi()
    await _remote_hf_engine(ceng, paths, clock, api)
    assert ceng.hub.hf() is not None
    api.fail["batch_bucket_files"] = [HTTPError(403)]
    job = await ceng.submit(ceng.spec(provider="fake"))
    await ceng.run_until(lambda: is_terminal(ceng.job(job.id).state), max_s=HOUR, step=2)
    assert ceng.job(job.id).state is JobState.DONE
    (note,) = ceng.notes(job.id, Reason.STORAGE_UNAVAILABLE)
    assert "cannot write to the storage bucket" in note.message


async def test_reconcile_retries_before_placing_a_migrating_job(ceng: CEngine) -> None:
    """Finding: one transient error while reading latest.json before a migration was
    swallowed and never retried, so the job resumed from an older checkpoint."""
    real = ceng.hub.latest
    calls = {"n": 0}

    def flaky(job_id: str) -> Any:
        calls["n"] += 1
        if calls["n"] <= 2:
            raise StorageError("hugging face returned HTTP 429")
        return real(job_id)

    ceng.hub.latest = flaky  # type: ignore[method-assign]
    job = await ceng.submit(
        ceng.spec(
            fake={"duration": 100, "die_after": 30, "attempts": {"2": {"duration": 10}}},
            provider="fake",
        )
    )
    await ceng.run_until(
        lambda: (
            (atts := ceng.store.attempts_for(job.id)) != []
            and atts[0].state is AttemptState.RUNNING
        ),
        step=1,
    )
    store = rs.LocalStore(ceng.local_root())
    src = ceng.local_root().parent / "stage"
    src.mkdir()
    (src / "w.pt").write_text("w")
    rs.publish_checkpoint(store, job.id, 3, src, [("w.pt", 1, "x")], attempt=1, step=30)
    await ceng.run_until(lambda: is_terminal(ceng.job(job.id).state), step=1)
    assert ceng.job(job.id).state is JobState.DONE
    assert calls["n"] >= 3
    second = ceng.store.attempts_for(job.id)[1]
    assert second.resume_checkpoint_id == f"{job.id}.c3"


async def test_a_finished_job_leaves_storage(ceng: CEngine) -> None:
    """Finding: nothing ever deleted jobs/<id>/ (checkpoints, owner, status files)."""
    job = await ceng.submit(ceng.spec(provider="fake"))
    await ceng.run_until(lambda: (ceng.local_root() / "jobs" / job.id).exists(), step=0.5)
    await ceng.run_until(lambda: is_terminal(ceng.job(job.id).state), step=1)
    await ceng.run_until(lambda: not (ceng.local_root() / "jobs" / job.id).exists(), step=1)


async def test_unused_datasets_are_evicted_after_an_upload(
    ceng: CEngine, paths: Paths, clock: FakeClock
) -> None:
    api = FakeHfApi()
    await _remote_hf_engine(ceng, paths, clock, api, dataset_keep_days=30)
    store = ceng.hub.hf()
    assert store is not None
    old = "0" * 64
    store.write_bytes(f"datasets/{old}/x.csv", b"1")
    ceng.store.put_data_cache(
        content_hash=old, uri=f"{HF_ROOT}/datasets/{old}", local_path="/x", size_bytes=1,
        file_count=1,
    )  # fmt: skip
    clock.advance(31 * 86_400)
    data = Path(ceng.project) / "ds"
    data.mkdir()
    (data / "rows.csv").write_text("a\n1\n")
    job = await ceng.submit(ceng.spec(data=[DataRef(mount="ds", path=str(data))]))
    await ceng.run_until(lambda: is_terminal(ceng.job(job.id).state), step=1)
    assert ceng.job(job.id).state is JobState.DONE
    assert ceng.store.get_data_cache(old) is None
    assert not any(k.startswith(f"datasets/{old}/") for k in api.files(BUCKET))


async def test_an_unreadable_dataset_file_is_a_data_problem(
    ceng: CEngine, paths: Paths, clock: FakeClock
) -> None:
    """Finding: an OSError while hashing escaped as an internal error ('this is a bug')
    with no word about which file."""
    await _remote_hf_engine(ceng, paths, clock, FakeHfApi())
    data = Path(ceng.project) / "ds"
    data.mkdir()
    secret = data / "locked.csv"
    secret.write_text("x")
    secret.chmod(0)
    try:
        job = await ceng.submit(ceng.spec(data=[DataRef(mount="ds", path=str(data))]))
        await ceng.run_until(lambda: is_terminal(ceng.job(job.id).state), step=1)
    finally:
        secret.chmod(0o600)
    done = ceng.job(job.id)
    assert done.state is JobState.FAILED
    assert done.failure_kind is FailureKind.USER_ERROR
    assert "locked.csv" in (done.message or "")
    assert not ceng.notes(job.id, Reason.INTERNAL_ERROR)


async def test_a_session_deadline_uses_the_submit_time_and_the_runs_own_limit(
    ceng: CEngine,
) -> None:
    """Finding: with no started_at from the provider (Kaggle) the deadline was anchored at
    the first poll that saw it running, and a shorter per-job timeout was ignored."""
    from gpu_router.adapters.fake import FakeAdapter

    adapter = ceng.supervisor.deps.registry.get("fake")
    assert isinstance(adapter, FakeAdapter)
    real_status = adapter.status
    real_submit = adapter.submit

    def status(ref: Any) -> Any:
        return real_status(ref).model_copy(update={"started_at": None})

    def submit(job: Any, ctx: Any) -> Any:
        ref = real_submit(job, ctx)
        return ref.model_copy(update={"meta": {**ref.meta, "session_s": "7200"}})

    adapter.status = status  # type: ignore[method-assign]
    adapter.submit = submit  # type: ignore[method-assign]
    job = await ceng.submit(ceng.spec(fake={"duration": 3 * HOUR}, provider="fake"))
    await ceng.run_until(
        lambda: (
            (atts := ceng.store.attempts_for(job.id)) != [] and atts[0].session_deadline is not None
        ),
        step=1,
    )
    att = ceng.store.attempts_for(job.id)[0]
    assert att.submitted_at is not None
    assert att.session_deadline == att.submitted_at + 7200


# --------------------------------------------------------------------------- datasets, fsync


def test_symlinked_dataset_dirs_are_part_of_the_dataset(tmp_path: Path) -> None:
    """Finding: scan() dropped symlinked subdirectories, so remote runs got a dataset
    without them (local runs saw them through their symlink)."""
    external = tmp_path / "external" / "images"
    external.mkdir(parents=True)
    (external / "a.png").write_bytes(b"png-a")
    data = tmp_path / "data"
    data.mkdir()
    (data / "labels.csv").write_text("a,1\n")
    (data / "images").symlink_to(external, target_is_directory=True)
    (data / "loop").symlink_to(data, target_is_directory=True)  # never followed forever
    rels = [rel for rel, _size, _m in scan(data)]
    assert rels == ["images/a.png", "labels.csv"]
    before = digest(data).sha256
    (external / "a.png").write_bytes(b"png-b")
    assert digest(data).sha256 != before  # the hash covers the linked content


def test_local_store_writes_are_fsynced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Finding: LocalStore wrote tmp + rename without fsync, so a power loss could leave
    latest.json naming files that never reached the disk."""
    synced: list[int] = []
    real_fsync = os.fsync

    def record(fd: int) -> None:
        synced.append(fd)
        real_fsync(fd)

    monkeypatch.setattr(rs.os, "fsync", record)
    monkeypatch.setattr(rs.sys, "platform", "linux")  # plain fsync, countable
    store = rs.LocalStore(tmp_path / "root")
    _publish(store, "job1", 1, tmp_path, '{"step": 1}')
    assert len(synced) >= 5  # the file, its dir, the manifest, latest.json, their dirs
