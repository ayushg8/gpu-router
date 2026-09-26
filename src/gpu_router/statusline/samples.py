"""Sample state.json snapshots for every status-line row state (phase 6b).

Used by `gpu statusline preview` and by the golden tests, so what the preview shows is
exactly what `gpu status --line` prints for that state. The clock is fixed at NOW
(Thu 2026-09-24 10:00 PDT); reset times render in the local timezone, so the golden tests
pin TZ=America/Los_Angeles.

`user_rows()` mirrors rows 1-2 of the reference status line
(tests/fixtures/statusline/statusline.sh) with the same tokens (a static sample, never a
real script), so the preview shows the gpu rows on the grid they share with the line above.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from gpu_router.statusline import fast
from gpu_router.statusline.fast import COL, DIM, FG, OFF, SP

NOW = 1790269200.0  # Thu 2026-09-24 17:00 UTC = 10:00 PDT
SAT_5PM = 1790467200.0  # Sat 2026-09-26 17:00 PDT: kaggle's weekly reset, local time
PROJECT = "/Users/you/code/yolo"
DEAD_PID = -1  # build_rows' pid check fails for it: the "daemon down" sample


@dataclass(frozen=True, slots=True)
class Sample:
    key: str
    title: str
    snapshot: dict[str, Any]
    cwd: str | None = PROJECT
    alive: bool = True

    def rows(self, *, color: bool = True) -> list[str]:
        return fast.rows_text(
            self.snapshot,
            now=NOW,
            cwd=self.cwd,
            color=color,
            pid_alive=lambda _pid: self.alive,
        )


PROVIDERS: list[dict[str, Any]] = [
    {
        "name": "kaggle",
        "health": "ok",
        "used": 22.0,
        "limit": 30.0,
        "unit": "gpu_hours",
        "resets_at": SAT_5PM,
        "source": "live",
        "unlimited": False,
        "remaining": 8.0,
    },
    {
        "name": "colab",
        "health": "ok",
        "used": 1.7,
        "limit": 12.0,
        "unit": "gpu_hours",
        "resets_at": NOW + 6 * 3600,
        "source": "estimate",
        "unlimited": False,
        "remaining": 10.3,
    },
    {
        "name": "local",
        "health": "ok",
        "used": 0.4,
        "limit": None,
        "unit": "gpu_hours",
        "resets_at": None,
        "source": "live",
        "unlimited": True,
        "remaining": None,
    },
]


def _job(**over: Any) -> dict[str, Any]:
    job: dict[str, Any] = {
        "id": "a7f2c0de9b41",
        "short_id": "a7f2",
        "name": "train_yolo",
        "state": "running",
        "provider": "kaggle",
        "gpu": "2xT4",
        "created_at": NOW - 4000,
        "started_at": NOW - 3720,
        "session_cap_s": 43200.0,
        "step": 380,
        "total_steps": 1000,
        "progress_source": "helper",
        "eta_s": 6620.0,  # at written_at (NOW - 20): ~1:50 left now
        "metric": {"name": "loss", "value": 0.412, "trend": "down"},
        "last_checkpoint_at": NOW - 180,
        "checkpoint_seq": 3,
        "route_summary": None,
        "approval_reason": None,
        "not_before": None,
        "script": "train_yolo.py",
        "project_dir": PROJECT,
        "attempt_n": 1,
        "resumed_from_seq": None,
        "route_hours": None,
        "route_hours_source": None,
    }
    job.update(over)
    return job


def _recent(**over: Any) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "id": "a7f2c0de9b41",
        "short_id": "a7f2",
        "name": "train_yolo",
        "state": "done",
        "finished_at": NOW - 120,
        "duration_s": 3 * 3600 + 12 * 60,
        "outputs_dir": "./runs/a7f2",
        "message": "done in 3h12m on kaggle 2xT4",
        "project_dir": PROJECT,
        "outputs_path": f"{PROJECT}/runs/a7f2",
        "outputs_fetched": True,
        "failure_kind": None,
        "exit_code": 0,
        "provider": "kaggle",
        "gpu": "2xT4",
    }
    rec.update(over)
    return rec


def snapshot(
    active: list[dict[str, Any]] | None = None,
    recent: list[dict[str, Any]] | None = None,
    **over: Any,
) -> dict[str, Any]:
    """A full state.json document in the phase-6b shape."""
    active = active or []
    counts: dict[str, int] = {}
    for job in active:
        counts[job["state"]] = counts.get(job["state"], 0) + 1
    snap: dict[str, Any] = {
        "schema": fast.STATE_SCHEMA,
        "written_at": NOW - 20,
        "daemon_pid": 4242,
        "active": active,
        "recent": recent or [],
        "counts": counts,
        "providers": copy.deepcopy(PROVIDERS),
        "finished_visible_s": 600.0,
        "migrated_visible_s": 600.0,
        "heartbeat_s": 60.0,
    }
    snap.update(over)
    return snap


APPROVAL = _job(
    id="c19e77aa0e12",
    short_id="c19e",
    name="eval",
    script="eval.py",
    state="awaiting_approval",
    provider="colab",
    gpu="T4",
    started_at=None,
    step=None,
    total_steps=None,
    progress_source=None,
    eta_s=None,
    metric=None,
    last_checkpoint_at=None,
    checkpoint_seq=None,
    route_summary="colab T4 · ~20m",
    approval_reason="agent job over the 1h auto-approve limit",
    route_hours=1 / 3,
    route_hours_source="spec",
    attempt_n=None,
)


def _queued(n: int) -> dict[str, Any]:
    return _job(
        id=f"q{n}00000000{n}",
        short_id=f"q{n}00",
        name=f"sweep_{n}",
        state="queued",
        provider=None,
        gpu=None,
        started_at=None,
        step=None,
        total_steps=None,
        eta_s=None,
        metric=None,
        last_checkpoint_at=None,
        checkpoint_seq=None,
    )


SAMPLES: list[Sample] = [
    Sample("running", "running (real steps from gpu.total_steps)", snapshot([_job()])),
    Sample(
        "running-elapsed",
        "running (no step total: elapsed vs the 12h session cap)",
        snapshot(
            [
                _job(
                    id="b81d00aa11cc",
                    short_id="b81d",
                    name="train_lm",
                    provider="colab",
                    gpu="T4",
                    started_at=NOW - (1 * 3600 + 42 * 60),
                    step=1200,
                    total_steps=None,
                    progress_source="stdout",
                    eta_s=None,
                    metric={"name": "loss", "value": 2.31, "trend": "flat"},
                    last_checkpoint_at=NOW - 12 * 60,
                )
            ]
        ),
    ),
    Sample("approval", "needs approval", snapshot([APPROVAL])),
    Sample("finished", "just finished (shown for 10 min)", snapshot(recent=[_recent()])),
    Sample(
        "migrated",
        "migrated (colab session ended; resumed on kaggle from checkpoint 4)",
        snapshot(
            [
                _job(
                    step=420,
                    eta_s=5400.0,
                    started_at=NOW - 240,
                    attempt_n=2,
                    resumed_from_seq=4,
                    checkpoint_seq=4,
                    migrated_from="colab",
                    migrated_at=NOW - 250,
                    migrate_reason="session_lost",
                )
            ]
        ),
    ),
    Sample(
        "migrating",
        "migrating (moving off colab, not placed yet)",
        snapshot(
            [
                _job(
                    state="migrating",
                    provider="colab",
                    gpu="T4",
                    checkpoint_seq=4,
                    migrated_from="colab",
                    migrated_at=NOW - 30,
                    migrate_reason="handoff",
                )
            ]
        ),
    ),
    Sample(
        "failed",
        "failed (shown for 10 min)",
        snapshot(
            recent=[
                _recent(
                    state="failed",
                    finished_at=NOW - 60,
                    duration_s=14 * 60,
                    failure_kind="user_error",
                    exit_code=1,
                    message="train_yolo.py exited with code 1",
                )
            ]
        ),
    ),
    Sample(
        "starting",
        "starting (provisioning a kaggle kernel)",
        snapshot(
            [
                _job(
                    state="provisioning",
                    started_at=None,
                    step=None,
                    total_steps=None,
                    eta_s=None,
                    metric=None,
                    last_checkpoint_at=None,
                    checkpoint_seq=None,
                )
            ]
        ),
    ),
    Sample(
        "several",
        "several jobs (approval first, then the running one, the rest counted)",
        snapshot([_job(), APPROVAL, _queued(1), _queued(2)]),
    ),
    Sample(
        "two-running",
        "two running + one queued (main job, then a count)",
        snapshot(
            [
                _job(),
                _job(id="d4e5f6a7b8c9", short_id="d4e5", name="bert_ft", provider="colab"),
                _queued(1),
            ]
        ),
    ),
    Sample(
        "local", "running on this Mac (no quota)", snapshot([_job(provider="local", gpu="MPS")])
    ),
    Sample("daemon-down", "daemon down while a job was running", snapshot([_job()]), alive=False),
    Sample("idle", "nothing active", snapshot()),
]


def user_rows(*, color: bool = True) -> list[str]:
    """Rows 1-2 of the user's status line, drawn with the script's own tokens."""
    i = 1 if color else 0
    blue = (fast.ACCENT + "Opus 5.5" + OFF) if color else "Opus 5.5"
    head = [("Opus 5.5", blue), SP, fast._dim("1M"), ("   ", "   "), fast._dim("medium")]
    w1 = fast._pw(head)
    row1 = "".join(p[i] for p in head) + " " * max(2, COL - w1)
    row1 += ("code/gpu-router" if not color else f"{FG}code/gpu-router{OFF}") + "  "
    row1 += "main*" if not color else f"{DIM}main*{OFF}"
    session = [
        fast._dim("session"),
        SP,
        fast._meter(20, FG),
        SP,
        fast._fg("20%"),
        SP,
        fast._dim("↻1pm"),
    ]
    week = [
        fast._dim("week"),
        SP,
        fast._meter(43, FG),
        SP,
        fast._fg("43%"),
        SP,
        fast._dim("↻Tue 4pm"),
    ]
    return [row1, fast.join_row(session, week, color=color)]


def preview(*, color: bool = True, keys: list[str] | None = None, with_line: bool = True) -> str:
    """Every sample (or those in `keys`) as a block: title, the user's row 2, gpu rows."""
    blocks: list[str] = []
    mine = user_rows(color=color)
    for sample in SAMPLES:
        if keys and sample.key not in keys:
            continue
        title = f"{sample.key}: {sample.title}"
        lines = [f"{DIM}{title}{OFF}" if color else f"-- {title}"]
        if with_line:
            lines.append(mine[1])
        rows = sample.rows(color=color)
        lines.extend(rows or ["(prints nothing)" if not color else f"{DIM}(prints nothing){OFF}"])
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)
