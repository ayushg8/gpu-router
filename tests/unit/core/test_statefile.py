from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import pytest

from gpu_router import statefile
from gpu_router.clock import FakeClock
from gpu_router.models import AttemptPatch, FailureKind, JobPatch, JobSpec
from gpu_router.statefile import (
    STATE_SCHEMA,
    ProviderSummary,
    StateFileWriter,
    StateSnapshot,
    build_snapshot,
    write_atomic,
)
from gpu_router.statemachine import AttemptState, JobState, Reason
from gpu_router.store import AttemptChange, Store

J = JobState


def _empty(now: float = 1.0) -> StateSnapshot:
    return StateSnapshot(
        schema=STATE_SCHEMA,
        written_at=now,
        daemon_pid=1,
        active=[],
        recent=[],
        counts={},
        providers=[],
    )


def _spec(**over: Any) -> JobSpec:
    return JobSpec.model_validate({"project_dir": "/proj", "script": "train.py", **over})


def _route(store: Store, job_id: str) -> None:
    store.transition(
        job_id,
        from_state=J.QUEUED,
        to_state=J.ROUTING,
        reason=Reason.ROUTING_STARTED,
        message="routing",
        actor="engine",
    )


# --------------------------------------------------------------------------- write_atomic


def test_write_atomic(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    write_atomic(path, _empty())
    assert json.loads(path.read_text())["schema"] == STATE_SCHEMA
    assert path.stat().st_mode & 0o777 == 0o644
    write_atomic(path, _empty(2.0))
    assert json.loads(path.read_text())["written_at"] == 2.0
    assert [p.name for p in tmp_path.iterdir()] == ["state.json"]


def test_write_atomic_never_leaves_partial_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "state.json"
    write_atomic(path, _empty())
    before = path.read_text()

    def broken_replace(src: str, dst: str) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", broken_replace)
    with pytest.raises(OSError, match="disk full"):
        write_atomic(path, _empty(99.0))
    assert path.read_text() == before  # old content intact
    assert [p.name for p in tmp_path.iterdir()] == ["state.json"]  # tmp cleaned up


# --------------------------------------------------------------------------- build_snapshot


def test_build_snapshot(store: Store, clock: FakeClock) -> None:
    t0 = clock.now()
    queued, _ = store.create_job(_spec(name="waiting"), actor="api")
    clock.advance(1)
    running, _ = store.create_job(_spec(name="bert", hours=0.5), actor="api")
    _route(store, running.id)
    _, att = store.place(
        running.id,
        from_state=J.ROUTING,
        provider="kaggle",
        gpu="2xT4",
        route_reason="r",
        message="placed",
    )
    store.record_submission(att.id, remote_id="r1", remote_url=None, remote_meta={})
    store.transition(
        running.id,
        from_state=J.PROVISIONING,
        to_state=J.RUNNING,
        reason=Reason.STARTED,
        message="running",
        actor="engine",
        attempt=AttemptChange(att.id, AttemptPatch(state=AttemptState.RUNNING)),
    )
    store.record_checkpoint(
        running.id,
        att.id,
        seq=1,
        uri="fake://c",
        step=5,
        size_bytes=None,
        sha256=None,
        created_at=clock.now(),
    )
    clock.advance(100)
    store.update_job(
        running.id,
        JobPatch(
            progress_step=25,
            progress_total=100,
            progress_source="helper",
            last_metrics={"acc": 0.2, "loss": 1.5},
        ),
    )
    approval, _ = store.create_job(_spec(name="big", hours=0.5), actor="agent")
    _route(store, approval.id)
    store.transition(
        approval.id,
        from_state=J.ROUTING,
        to_state=J.AWAITING_APPROVAL,
        reason=Reason.APPROVAL_REQUIRED,
        message="needs ok",
        actor="engine",
        patch=JobPatch(provider="colab", gpu="T4", approval_reason="agent job"),
    )
    old, _ = store.create_job(_spec(name="old"), actor="api")
    store.transition(
        old.id,
        from_state=J.QUEUED,
        to_state=J.FAILED,
        reason=Reason.GAVE_UP,
        message="gave up",
        actor="engine",
        patch=JobPatch(failure_kind=FailureKind.NO_PROVIDER),
    )
    clock.advance(1000)
    fresh, _ = store.create_job(_spec(name="fresh"), actor="api")
    store.transition(
        fresh.id,
        from_state=J.QUEUED,
        to_state=J.CANCELLED,
        reason=Reason.USER_CANCEL,
        message="cancelled by you",
        actor="user:cli",
    )

    providers = [
        ProviderSummary(
            name="kaggle",
            health="ok",
            used=3.0,
            limit=30.0,
            unit="gpu_hours",
            resets_at=None,
            source="live",
        )
    ]
    snap = build_snapshot(
        store=store,
        provider_summaries=providers,
        session_caps={"kaggle": 43200.0},
        now=clock.now(),
        daemon_pid=4242,
    )
    assert snap["schema"] == STATE_SCHEMA
    assert snap["daemon_pid"] == 4242
    assert [a["name"] for a in snap["active"]] == ["bert", "big", "waiting"]
    run = snap["active"][0]
    assert run["state"] == "running"
    assert run["gpu"] == "2xT4"
    assert run["session_cap_s"] == 43200.0
    assert run["started_at"] == t0 + 1
    assert run["step"] == 25
    assert run["total_steps"] == 100
    assert run["eta_s"] == pytest.approx((clock.now() - (t0 + 1)) / 25 * 75)
    assert run["metric"] == {"name": "loss", "value": 1.5, "trend": None}
    assert run["checkpoint_seq"] == 1
    appr = snap["active"][1]
    assert appr["route_summary"] == "colab T4 · ~30m"
    assert appr["approval_reason"] == "agent job"
    assert snap["active"][2]["started_at"] is None
    assert [r["name"] for r in snap["recent"]] == ["fresh"]  # "old" is outside the window
    rec = snap["recent"][0]
    assert rec["outputs_dir"] == f"./runs/{fresh.id[:4]}"
    assert rec["duration_s"] is None
    assert rec["message"] == "cancelled by you"
    assert snap["counts"] == {"queued": 1, "running": 1, "awaiting_approval": 1}
    assert snap["providers"] == providers
    json.dumps(snap)  # serialisable
    del queued


def test_snapshot_marks_migrated_jobs(store: Store, clock: FakeClock) -> None:
    job, _ = store.create_job(_spec(), actor="api")
    _route(store, job.id)
    _, att = store.place(
        job.id, from_state=J.ROUTING, provider="colab", gpu="T4", route_reason="r", message="m"
    )
    store.record_submission(att.id, remote_id="r1", remote_url=None, remote_meta={})
    store.transition(
        job.id,
        from_state=J.PROVISIONING,
        to_state=J.RUNNING,
        reason=Reason.STARTED,
        message="running",
        actor="engine",
        attempt=AttemptChange(att.id, AttemptPatch(state=AttemptState.RUNNING)),
    )
    t_mig = clock.now()
    store.transition(
        job.id,
        from_state=J.RUNNING,
        to_state=J.MIGRATING,
        reason=Reason.SESSION_LOST,
        message="session lost",
        actor="engine",
        attempt=AttemptChange(att.id, AttemptPatch(state=AttemptState.LOST)),
    )
    store.place(
        job.id,
        from_state=J.MIGRATING,
        provider="kaggle",
        gpu="P100",
        route_reason="r",
        message="resuming on kaggle",
    )
    snap = build_snapshot(
        store=store, provider_summaries=[], session_caps={}, now=clock.now(), daemon_pid=1
    )
    row = snap["active"][0]
    assert row["provider"] == "kaggle"
    assert row["migrated_from"] == "colab"
    assert row["migrated_at"] == t_mig
    clock.advance(statefile.RECENT_WINDOW_S + 1)
    later = build_snapshot(
        store=store, provider_summaries=[], session_caps={}, now=clock.now(), daemon_pid=1
    )
    assert "migrated_from" not in later["active"][0]


# --------------------------------------------------------------------------- writer


async def test_writer_coalesces(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    builds: list[int] = []

    def build() -> StateSnapshot:
        builds.append(1)
        return _empty(float(len(builds)))

    writer = StateFileWriter(path, build, min_interval_s=0.05)
    task = asyncio.create_task(writer.run())
    try:
        writer.mark_dirty()
        await asyncio.sleep(0.01)
        assert writer.writes == 1
        for _ in range(20):
            writer.mark_dirty()
        await asyncio.sleep(0.01)
        assert writer.writes == 1  # within the interval: coalesced, not yet written
        await asyncio.sleep(0.1)
        assert writer.writes == 2
        writer.flush()
        assert writer.writes == 3
        assert json.loads(path.read_text())["written_at"] == 3.0
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


async def test_writer_swallows_errors(tmp_path: Path) -> None:
    def build() -> StateSnapshot:
        raise RuntimeError("store closed")

    writer = StateFileWriter(tmp_path / "state.json", build)
    writer.flush()  # logged, not raised
    assert writer.writes == 0
    assert not (tmp_path / "state.json").exists()
