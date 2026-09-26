"""FakeAdapter with checkpoint storage (D43): with a local `file://` GPU_STORAGE the simulated
runner publishes its checkpoints into the runner/storage.py layout at their ckpt_end times,
stops once owner.json names a later attempt, and checks GPU_RESUME_URI at submit."""

from __future__ import annotations

import json
from pathlib import Path

from gpu_router.adapters.base import RemoteRef
from gpu_router.adapters.fake import FakeAdapter
from gpu_router.clock import FakeClock
from gpu_router.models import Checkpoint
from gpu_router.protocol import parse_line
from gpu_router.runner import storage as rs
from tests.contract.harness import make_ctx
from tests.unit.adapters.conftest import MakeJob


def _lines(fake: FakeAdapter, ref: RemoteRef) -> list[str]:
    return [line for chunk in fake.logs(ref) for line in chunk.lines]


def _store(tmp_path: Path) -> rs.LocalStore:
    root = tmp_path / "storage"
    root.mkdir()
    return rs.LocalStore(root)


def test_checkpoints_are_published_to_local_storage(
    fake: FakeAdapter, make_fake_job: MakeJob, clock: FakeClock, tmp_path: Path
) -> None:
    store = _store(tmp_path)
    job = make_fake_job(duration=10, steps=10, checkpoint_every=3)
    rs.claim(store, job.id, 1)
    ref = fake.submit(job, make_ctx(job, env={"GPU_STORAGE": store.root_uri}))
    clock.advance(4)  # ckpt 1 ends at 3.5
    fake.status(ref)
    latest = rs.read_latest(store, job.id)
    assert latest is not None
    assert latest["seq"] == 1
    assert latest["attempt"] == 1
    assert latest["uri"] == store.uri(rs.ckpt_key(job.id, 1))
    state = json.loads(store.path(rs.ckpt_key(job.id, 1) + "/state.json").read_text())
    assert state["seq"] == 1
    assert state["job"] == job.id
    assert store.path(rs.ckpt_key(job.id, 1) + "/" + rs.CKPT_MANIFEST).is_file()
    clock.advance(10)
    lines = _lines(fake, ref)
    ends = [p for p in (parse_line(x) for x in lines) if p is not None and p.t == "ckpt_end"]
    assert [e.seq for e in ends] == [1, 2, 3]
    assert all(e.uri == store.uri(rs.ckpt_key(job.id, e.seq)) for e in ends)
    assert rs.read_latest(store, job.id)["seq"] == 3


def test_a_later_owner_stops_the_old_runner(
    fake: FakeAdapter, make_fake_job: MakeJob, clock: FakeClock, tmp_path: Path
) -> None:
    store = _store(tmp_path)
    job = make_fake_job(duration=10, steps=10, checkpoint_every=2)
    rs.claim(store, job.id, 1)
    ref = fake.submit(job, make_ctx(job, env={"GPU_STORAGE": store.root_uri}))
    clock.advance(3)
    fake.status(ref)
    assert rs.read_latest(store, job.id)["seq"] == 1
    rs.claim(store, job.id, 2)  # the engine placed attempt 2 elsewhere
    clock.advance(10)
    fake.status(ref)
    assert rs.read_latest(store, job.id)["seq"] == 1


def test_resume_from_storage_and_missing_checkpoint(
    fake: FakeAdapter, make_fake_job: MakeJob, clock: FakeClock, tmp_path: Path
) -> None:
    store = _store(tmp_path)
    job = make_fake_job(duration=4, steps=4, checkpoint_every=10)
    src = tmp_path / "src"
    src.mkdir()
    (src / "state.json").write_text("{}")
    rs.publish_checkpoint(store, job.id, 4, src, [("state.json", 2, "x")], attempt=1)
    uri = store.uri(rs.ckpt_key(job.id, 4))
    ckpt = Checkpoint(
        id=f"{job.id}.c4",
        job_id=job.id,
        attempt_id=f"{job.id}.1",
        seq=4,
        uri=uri,
        created_at=clock.now(),
        recorded_at=clock.now(),
    )
    env = {"GPU_STORAGE": store.root_uri, "GPU_RESUME_URI": uri}
    ref = fake.submit(job, make_ctx(job, n=2, env=env, resume_from=ckpt))
    clock.advance(1)
    lines = _lines(fake, ref)
    assert "resuming from checkpoint 4" in lines
    assert f"fake: restored {uri}" in lines

    gone = store.uri(rs.ckpt_key(job.id, 9))
    ckpt9 = ckpt.model_copy(update={"seq": 9, "uri": gone, "id": f"{job.id}.c9"})
    env = {"GPU_STORAGE": store.root_uri, "GPU_RESUME_URI": gone}
    ref = fake.submit(job, make_ctx(job, n=3, env=env, resume_from=ckpt9))
    clock.advance(1)
    lines = _lines(fake, ref)
    assert not any(x.startswith("resuming from") for x in lines)
    assert any("checkpoint 9 is missing from storage" in x for x in lines)


def test_without_storage_nothing_changes(
    fake: FakeAdapter, make_fake_job: MakeJob, clock: FakeClock, tmp_path: Path
) -> None:
    job = make_fake_job(duration=4, steps=4, checkpoint_every=1)
    hf = {"GPU_STORAGE": "hf://buckets/me/gpu-router"}  # not reachable by the fake
    ref = fake.submit(job, make_ctx(job, env=hf))
    clock.advance(5)
    ends = [p for p in (parse_line(x) for x in _lines(fake, ref)) if p and p.t == "ckpt_end"]
    assert ends
    assert all(e.uri.startswith("fake://") for e in ends)
    assert fake.run_record(ref.remote_id).storage is None
