"""Daemon side of the status line: state.json carries everything `gpu status --line` draws
(phase-6b additive fields), the writer heartbeats while jobs are active, and a real
daemon's file renders through the fast path."""

from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any

import pytest

from gpu_router.clock import FakeClock
from gpu_router.models import AttemptPatch, FailureKind, JobPatch, JobSpec
from gpu_router.statefile import (
    HEARTBEAT_S,
    STATE_SCHEMA,
    StateFileWriter,
    StateSnapshot,
    build_snapshot,
)
from gpu_router.statemachine import AttemptState, JobState, Reason
from gpu_router.statusline import fast
from gpu_router.store import AttemptChange, Store
from tests.api.conftest import Api, api  # noqa: F401  (the fixture, re-exported)

J = JobState
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _spec(**over: Any) -> JobSpec:
    return JobSpec.model_validate({"project_dir": "/proj", "script": "src/train.py", **over})


def _to_running(store: Store, job_id: str, provider: str = "kaggle", **place: Any) -> str:
    store.transition(
        job_id,
        from_state=J.QUEUED,
        to_state=J.ROUTING,
        reason=Reason.ROUTING_STARTED,
        message="routing",
        actor="engine",
    )
    _, att = store.place(
        job_id,
        from_state=J.ROUTING,
        provider=provider,
        gpu="2xT4",
        route_reason="r",
        message="placed",
        **place,
    )
    store.record_submission(att.id, remote_id="r1", remote_url=None, remote_meta={})
    store.transition(
        job_id,
        from_state=J.PROVISIONING,
        to_state=J.RUNNING,
        reason=Reason.STARTED,
        message="running",
        actor="engine",
        attempt=AttemptChange(att.id, AttemptPatch(state=AttemptState.RUNNING)),
    )
    return att.id


def _snap(store: Store, clock: FakeClock, **kw: Any) -> StateSnapshot:
    return build_snapshot(
        store=store, provider_summaries=[], session_caps={}, now=clock.now(), daemon_pid=1, **kw
    )


def _metrics(path: Path, values: list[float], name: str = "loss") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for i, v in enumerate(values):
            fh.write(json.dumps({"ts": i, "attempt": 1, "step": i, "metrics": {name: v}}) + "\n")


@pytest.mark.parametrize(
    ("values", "trend"),
    [
        ([2.0 * 0.97**i for i in range(60)], "down"),
        ([0.1 + 0.01 * i for i in range(60)], "up"),
        ([0.5] * 60, "flat"),
        ([0.5], None),
    ],
)
def test_metric_trend_comes_from_metrics_jsonl(
    store: Store, clock: FakeClock, tmp_path: Path, values: list[float], trend: str | None
) -> None:
    job, _ = store.create_job(_spec(), actor="api")
    _to_running(store, job.id)
    store.update_job(job.id, JobPatch(last_metrics={"acc": 0.3, "loss": values[-1]}))
    _metrics(tmp_path / job.id / "metrics.jsonl", values)
    snap = _snap(store, clock, metrics_path=lambda jid: tmp_path / jid / "metrics.jsonl")
    assert snap["active"][0]["metric"] == {"name": "loss", "value": values[-1], "trend": trend}


def test_trend_reads_only_the_tail_of_a_big_file(
    store: Store, clock: FakeClock, tmp_path: Path
) -> None:
    job, _ = store.create_job(_spec(), actor="api")
    _to_running(store, job.id)
    store.update_job(job.id, JobPatch(last_metrics={"loss": 0.1}))
    # 50k rising points, then 100 falling ones: only the tail decides
    _metrics(
        tmp_path / "m.jsonl",
        [float(i) for i in range(50_000)] + [9.0 - i * 0.05 for i in range(100)],
    )
    snap = _snap(store, clock, metrics_path=lambda _jid: tmp_path / "m.jsonl")
    assert snap["active"][0]["metric"]["trend"] == "down"  # type: ignore[index]
    missing = _snap(store, clock, metrics_path=lambda _jid: tmp_path / "none.jsonl")
    assert missing["active"][0]["metric"]["trend"] is None  # type: ignore[index]


def test_script_project_attempt_and_resume_seq(store: Store, clock: FakeClock) -> None:
    job, _ = store.create_job(_spec(), actor="api")
    att = _to_running(store, job.id)
    store.record_checkpoint(
        job.id,
        att,
        seq=4,
        uri="fake://c4",
        step=40,
        size_bytes=None,
        sha256=None,
        created_at=clock.now(),
    )
    store.transition(
        job.id,
        from_state=J.RUNNING,
        to_state=J.MIGRATING,
        reason=Reason.SESSION_LOST,
        message="session lost",
        actor="engine",
        attempt=AttemptChange(att, AttemptPatch(state=AttemptState.LOST)),
    )
    moving = _snap(store, clock)["active"][0]
    assert moving["migrated_from"] == "kaggle"  # visible while it is still migrating
    assert moving["migrate_reason"] == "session_lost"
    assert moving["checkpoint_seq"] == 4
    store.place(
        job.id,
        from_state=J.MIGRATING,
        provider="colab",
        gpu="T4",
        route_reason="r",
        message="resuming",
        resume_checkpoint_id=f"{job.id}.c4",
    )
    row = _snap(store, clock)["active"][0]
    assert row["script"] == "train.py"
    assert row["project_dir"] == "/proj"
    assert row["attempt_n"] == 2
    assert row["resumed_from_seq"] == 4
    assert row["migrated_from"] == "kaggle"
    assert row["provider"] == "colab"


def test_approval_route_hours_from_spec_or_router(store: Store, clock: FakeClock) -> None:
    timed, _ = store.create_job(_spec(hours=0.5), actor="agent")
    guessed, _ = store.create_job(_spec(command=["python", "-m", "x"], script=None), actor="agent")
    for job in (timed, guessed):
        store.transition(
            job.id,
            from_state=J.QUEUED,
            to_state=J.ROUTING,
            reason=Reason.ROUTING_STARTED,
            message="routing",
            actor="engine",
        )
        store.transition(
            job.id,
            from_state=J.ROUTING,
            to_state=J.AWAITING_APPROVAL,
            reason=Reason.APPROVAL_REQUIRED,
            message="needs ok",
            actor="engine",
            detail={"hours": 2.5, "hours_source": "estimate", "rule": "agent_hours"},
            patch=JobPatch(provider="colab", gpu="T4", approval_reason="over 1h"),
        )
    rows = {r["id"]: r for r in _snap(store, clock)["active"]}
    assert rows[timed.id]["route_hours"] == 0.5
    assert rows[timed.id]["route_hours_source"] == "spec"
    assert rows[guessed.id]["route_hours"] == 2.5
    assert rows[guessed.id]["route_hours_source"] == "estimate"
    assert rows[guessed.id]["script"] == "python"


def test_recent_jobs_carry_outputs_and_failure(store: Store, clock: FakeClock) -> None:
    job, _ = store.create_job(_spec(), actor="api")
    att = _to_running(store, job.id)
    clock.advance(90)
    store.transition(
        job.id,
        from_state=J.RUNNING,
        to_state=J.FAILED,
        reason=Reason.SCRIPT_FAILED,
        message="exit 2",
        actor="engine",
        attempt=AttemptChange(att, AttemptPatch(state=AttemptState.FAILED, exit_code=2)),
        patch=JobPatch(failure_kind=FailureKind.USER_ERROR, exit_code=2),
    )
    (rec,) = _snap(store, clock)["recent"]
    assert rec["state"] == "failed"
    assert rec["failure_kind"] == "user_error"
    assert rec["exit_code"] == 2
    assert rec["project_dir"] == "/proj"
    assert rec["outputs_path"] == f"/proj/runs/{job.id[:4]}"
    assert rec["outputs_dir"] == f"./runs/{job.id[:4]}"
    assert rec["outputs_fetched"] is False
    assert rec["provider"] == "kaggle"
    assert rec["gpu"] == "2xT4"


def test_snapshot_carries_windows_and_heartbeat(store: Store, clock: FakeClock) -> None:
    snap = _snap(store, clock, recent_window_s=300, migrated_window_s=120)
    assert snap["schema"] == STATE_SCHEMA
    assert snap["finished_visible_s"] == 300
    assert snap["migrated_visible_s"] == 120
    assert snap["heartbeat_s"] == HEARTBEAT_S


async def test_writer_heartbeats_only_while_active(tmp_path: Path) -> None:
    active: list[Any] = [{"id": "x"}]

    def build() -> StateSnapshot:
        return StateSnapshot(
            schema=1,
            written_at=0.0,
            daemon_pid=1,
            active=list(active),
            recent=[],
            counts={},
            providers=[],
        )

    writer = StateFileWriter(tmp_path / "state.json", build, min_interval_s=0.0, heartbeat_s=0.05)
    task = asyncio.create_task(writer.run())
    try:
        writer.mark_dirty()
        await asyncio.sleep(0.4)
        busy = writer.writes
        assert busy >= 4  # the first write plus heartbeats, no mark_dirty needed
        active.clear()
        writer.mark_dirty()  # one more write that lists no active job
        await asyncio.sleep(0.05)
        idle = writer.writes
        await asyncio.sleep(0.3)
        assert writer.writes == idle  # idle: no heartbeat writes
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


# --------------------------------------------------------------------------- real daemon


def _render(h: Api) -> list[str]:
    h.runtime.statefile.flush()
    snap = json.loads(h.runtime.paths.state.read_text())
    assert snap["daemon_pid"] == os.getpid()
    return fast.rows_text(snap, now=h.clock.now(), cwd=h.project, color=False)


async def test_a_real_daemons_file_draws_every_row(api: Api) -> None:  # noqa: F811
    job = await api.submit(
        provider_options={"fake": {"duration": 120, "steps": 120, "checkpoint_every": 20}}
    )
    jid = job["id"]
    store = api.runtime.store
    await api.drive(
        lambda: (
            (store.get_job(jid).progress.step or 0) >= 45
            and store.get_job(jid).checkpoint_count >= 2
        )
    )
    bar, detail = _render(api)
    assert re.match(r"gpu █+░+ \d\d% \d:\d\d left  fake", bar), bar
    assert re.match(r"train · \S+ +loss \d\.\d+ ↓  ckpt (<1m|\d+m) ago$", detail), detail
    await api.drive(lambda: api.state(jid) == "done")
    (done,) = _render(api)
    assert re.match(rf"gpu ✓ train · 2m +→ \./runs/{jid[:4]}$", done), done
    api.clock.advance(601)
    assert _render(api) == []  # the finished row has expired

    wait = await api.submit(requires_approval=True, hours=0.25)
    await api.drive(lambda: api.state(wait["id"]) == "awaiting_approval")
    (row,) = _render(api)
    assert re.match(r"gpu ⏸ train\.py → fake( \S+)? +~15m  /gpu-approve$", row), row
