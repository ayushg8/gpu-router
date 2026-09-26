"""Which job events notify, and what the notifications say (phase 8a)."""

from __future__ import annotations

from collections import Counter
from typing import Any

import pytest

from gpu_router.models import Job, JobEvent, JobSpec, Source
from gpu_router.notify.format import (
    NAME_MAX,
    SOUND_ATTENTION,
    SOUND_FAILED,
    Kind,
    build,
    classify,
    harmless_notification,
    summary,
)
from gpu_router.statemachine import JobState, Reason

JOB_ID = "a7f2c19e0b3d"


def ev(
    to: JobState | None,
    reason: Reason | str,
    *,
    kind: str = "transition",
    message: str = "",
    detail: dict[str, Any] | None = None,
    frm: JobState | None = JobState.RUNNING,
) -> JobEvent:
    return JobEvent(
        seq=7,
        job_id=JOB_ID,
        kind=kind,  # type: ignore[arg-type]
        from_state=frm if kind == "transition" else None,
        to_state=to if kind == "transition" else None,
        reason=str(reason),
        message=message,
        detail=detail or {},
        actor="engine",
        ts=1_000.0,
    )


def job(**fields: Any) -> Job:
    base: dict[str, Any] = {
        "id": JOB_ID,
        "short_id": "a7f2",
        "name": "train_yolo",
        "state": JobState.DONE,
        "source": Source.CLI,
        "project_dir": "/Users/me/proj",
        "spec": JobSpec(project_dir="/Users/me/proj", script="train.py"),
        "spec_hash": "x",
        "provider": "kaggle",
        "gpu": "2xT4",
        "outputs_dir": "/Users/me/proj/runs/a7f2",
        "outputs_fetched": True,
        "created_at": 0.0,
        "updated_at": 0.0,
        "started_at": 100.0,
        "finished_at": 100.0 + 3 * 3600 + 12 * 60,
    }
    base.update(fields)
    return Job(**base)


@pytest.mark.parametrize(
    ("event", "kind"),
    [
        (ev(JobState.DONE, Reason.COMPLETED), Kind.FINISHED),
        (ev(JobState.FAILED, Reason.SCRIPT_FAILED), Kind.FAILED),
        (ev(JobState.FAILED, Reason.NO_PROVIDER_FITS, frm=JobState.ROUTING), Kind.FAILED),
        (
            ev(JobState.AWAITING_APPROVAL, Reason.APPROVAL_REQUIRED, frm=JobState.ROUTING),
            Kind.APPROVAL,
        ),
        (ev(JobState.MIGRATING, Reason.SESSION_LOST), Kind.MIGRATED),
        (ev(JobState.MIGRATING, Reason.HANDOFF), Kind.MIGRATED),
        (ev(JobState.MIGRATING, Reason.QUOTA_EXHAUSTED), Kind.MIGRATED),
        # D48: an overrun asks for approval right after; that notification covers it
        (ev(JobState.MIGRATING, Reason.HOURS_EXCEEDED), None),
        # D43 in-place re-ask
        (ev(None, Reason.APPROVAL_REQUIRED, kind="note", detail={"reask": True}), Kind.APPROVAL),
        (ev(None, Reason.APPROVAL_REQUIRED, kind="note"), None),
        # user-stopped and routine transitions never notify
        (ev(JobState.CANCELLED, Reason.USER_CANCEL), None),
        (ev(JobState.DENIED, Reason.DENIED, frm=JobState.AWAITING_APPROVAL), None),
        (ev(JobState.RUNNING, Reason.STARTED, frm=JobState.PROVISIONING), None),
        (ev(JobState.CHECKPOINTING, Reason.CHECKPOINT_BEGIN), None),
        (ev(None, Reason.FETCHED, kind="note"), None),
    ],
)
def test_classify(event: JobEvent, kind: Kind | None) -> None:
    assert classify(event) == kind


def test_finished_names_time_where_and_outputs() -> None:
    n = build(Kind.FINISHED, job(), ev(JobState.DONE, Reason.COMPLETED))
    assert n.title == "gpu-router"
    assert n.subtitle == "✓ train_yolo finished"
    assert n.body == "3h12m on kaggle 2xT4 · outputs in proj/runs/a7f2"
    assert n.sound is None
    assert n.group == f"gpu-router.{JOB_ID}"


def test_finished_without_fetched_outputs_says_how_to_get_them() -> None:
    n = build(Kind.FINISHED, job(outputs_fetched=False), ev(JobState.DONE, Reason.COMPLETED))
    assert "gpu fetch a7f2" in n.body


def test_failed_script_shows_exit_code_and_the_logs_command() -> None:
    j = job(state=JobState.FAILED, exit_code=1, finished_at=100.0 + 14 * 60)
    n = build(Kind.FAILED, j, ev(JobState.FAILED, Reason.SCRIPT_FAILED, message="script exited 1"))
    assert n.subtitle == "✗ train_yolo failed"
    assert n.body == "exit 1 after 14m on kaggle 2xT4 · gpu logs a7f2"
    assert n.sound == SOUND_FAILED


def test_failed_without_exit_code_uses_gpu_routers_own_message() -> None:
    j = job(state=JobState.FAILED, provider=None, gpu=None, started_at=None, exit_code=None)
    msg = "no provider can run this job: it needs 24GB of VRAM"
    n = build(Kind.FAILED, j, ev(JobState.FAILED, Reason.NO_PROVIDER_FITS, message=msg))
    assert n.body.startswith(msg)
    assert n.body.endswith("gpu logs a7f2")


def test_approval_names_route_reason_and_command() -> None:
    j = job(
        state=JobState.AWAITING_APPROVAL,
        provider="colab",
        gpu="T4",
        approval_reason="agent job over 1h",
        name="eval.py",
    )
    n = build(Kind.APPROVAL, j, ev(JobState.AWAITING_APPROVAL, Reason.APPROVAL_REQUIRED))
    assert n.subtitle == "⏸ eval.py needs your approval"
    assert n.body == "→ colab T4 · agent job over 1h · gpu approve a7f2"
    assert n.sound == SOUND_ATTENTION


def test_migrated_names_the_provider_it_left_not_the_next_one() -> None:
    # by the time the job is read, the driver may already have placed it on kaggle
    j = job(state=JobState.PROVISIONING, provider="kaggle")
    event = ev(
        JobState.MIGRATING,
        Reason.SESSION_LOST,
        message="colab session ended (session limit); resuming from checkpoint 4 elsewhere",
        detail={"previous_provider": "colab"},
    )
    n = build(Kind.MIGRATED, j, event)
    assert n.subtitle == "↪ train_yolo is moving off colab"
    assert "resuming from checkpoint 4" in n.body


def test_sound_off() -> None:
    j = job(state=JobState.FAILED, exit_code=2)
    assert (
        build(Kind.FAILED, j, ev(JobState.FAILED, Reason.SCRIPT_FAILED), sound=False).sound is None
    )


def test_job_controlled_names_are_cleaned_and_cut() -> None:
    evil = "x\x1b[31m" + "‮" + "n" * 200 + "\nline2"
    n = build(Kind.FINISHED, job(name=evil), ev(JobState.DONE, Reason.COMPLETED))
    assert "\x1b" not in n.subtitle
    assert "‮" not in n.subtitle
    assert "\n" not in n.subtitle
    assert len(n.subtitle) <= len("✓  finished") + NAME_MAX + 1


def test_missing_job_falls_back_to_the_event() -> None:
    n = build(
        Kind.FAILED, None, ev(JobState.FAILED, Reason.GAVE_UP, message="gave up after 6 tries")
    )
    assert n.subtitle == "✗ job a7f2 failed"
    assert "gave up after 6 tries" in n.body


def test_summary_counts_what_was_held_back() -> None:
    n = summary(Counter({"finished": 3, "failed": 1}))
    assert n.kind == Kind.SUMMARY
    assert n.subtitle == "4 more job updates"
    assert n.body.startswith("3 finished · 1 failed")


def test_the_test_notification_is_harmless() -> None:
    n = harmless_notification()
    assert n.subtitle == "gpu-router test notification"
    assert n.job_id is None
