"""CLI formatting helpers and exit-code mapping (render.py, exitcodes.py)."""

from __future__ import annotations

import pytest

from gpu_router.api import JobView
from gpu_router.cli import exitcodes, render
from gpu_router.errors import (
    AmbiguousJobRef,
    ApiError,
    Conflict,
    DaemonUnavailable,
    GpuRouterError,
    InvalidRequest,
    InvalidSpec,
    JobNotFound,
    NotReady,
    ProviderNotFound,
    SubmitUncertain,
)
from gpu_router.models import QuotaSnapshot, QuotaUnit
from gpu_router.statemachine import JobState


@pytest.mark.parametrize(
    ("secs", "text"),
    [(None, "-"), (0, "0s"), (42, "42s"), (192, "3m12s"), (3900, "1h05m"), (200000, "2d7h")],
)
def test_duration(secs: float | None, text: str) -> None:
    assert render.duration(secs) == text


def test_clock_time() -> None:
    assert render.clock_time(6130) == "01:42:10"
    assert render.clock_time(None) == "-"


@pytest.mark.parametrize(("v", "text"), [(16, "16"), (2.5, "2.5"), (0.0004, "0"), (29.96, "30")])
def test_fmt_num(v: float, text: str) -> None:
    assert render.fmt_num(v) == text


def test_reset_label() -> None:
    now = 1_790_000_000.0
    assert render.reset_label(None, now) == ""
    assert render.reset_label(now - 1, now) == "↻now"
    assert render.reset_label(now + 3 * 3600, now) == "↻3h00m"
    assert render.reset_label(now + 3 * 86400, now).startswith("↻")
    assert len(render.reset_label(now + 3 * 86400, now)) == 4  # ↻Sat
    assert " " in render.reset_label(now + 30 * 86400, now)  # ↻Oct 4


def test_quota_text() -> None:
    now = 1_790_000_000.0
    q = QuotaSnapshot(
        provider="kaggle",
        used=22,
        limit=30,
        unit=QuotaUnit.GPU_HOURS,
        source="estimate",
        observed_at=now,
    )
    assert render.quota_text(q, now) == "22/30h est"
    assert render.quota_text(None, now) == "quota unknown"


@pytest.mark.parametrize(
    ("state", "icon", "style"),
    [
        (JobState.RUNNING, "⚡", "green"),
        (JobState.CHECKPOINTING, "⚡", "green"),
        (JobState.QUEUED, "⏸", "yellow"),
        (JobState.AWAITING_APPROVAL, "⏸", "yellow"),
        (JobState.MIGRATING, "↪", "yellow"),
        (JobState.DONE, "✓", "green"),
        (JobState.FAILED, "✗", "red"),
        (JobState.CANCELLED, "✗", "dim"),
        (JobState.DENIED, "✗", "dim"),
    ],
)
def test_visual_language(state: JobState, icon: str, style: str) -> None:
    assert render.state_style(state) == (icon, style)


def test_every_state_has_a_style() -> None:
    for s in JobState:
        icon, _ = render.state_style(s)
        assert icon in "⚡⏸✓✗↪"


def test_bar() -> None:
    assert render.bar(0.5, width=4) == "██░░  50%"
    assert render.bar(None) == ""


@pytest.mark.parametrize(
    ("exc", "code"),
    [
        (InvalidSpec("x"), exitcodes.USAGE),
        (InvalidRequest("x"), exitcodes.USAGE),
        (DaemonUnavailable("x"), exitcodes.DAEMON),
        (NotReady("x"), exitcodes.DAEMON),
        (JobNotFound("x"), exitcodes.NOT_FOUND),
        (AmbiguousJobRef("x"), exitcodes.NOT_FOUND),
        (ProviderNotFound("x"), exitcodes.NOT_FOUND),
        (Conflict("x"), exitcodes.CONFLICT),
        (ApiError("invalid_transition", "x", status=409), exitcodes.CONFLICT),
        (ApiError("something_new", "x", status=500), exitcodes.ERROR),
        (GpuRouterError("x"), exitcodes.ERROR),
        (SubmitUncertain("x"), exitcodes.ERROR),  # not "daemon down" (exit 3)
    ],
)
def test_exit_code_for_error(exc: GpuRouterError, code: int) -> None:
    assert exitcodes.for_error(exc) == code


def test_exit_code_for_state() -> None:
    assert exitcodes.for_state(JobState.DONE) == 0
    assert exitcodes.for_state(JobState.FAILED) == exitcodes.JOB_FAILED
    assert exitcodes.for_state(JobState.CANCELLED) == exitcodes.JOB_STOPPED
    assert exitcodes.for_state(JobState.DENIED) == exitcodes.JOB_STOPPED


def _failed_view(**fields: object) -> JobView:
    body: dict[str, object] = {
        "id": "a7f2c19e0b3d",
        "short_id": "a7f2",
        "name": "train",
        "state": "failed",
        "source": "cli",
        "project_dir": "/p",
        "spec": {"project_dir": "/p", "script": "train.py"},
        "spec_hash": "x",
        "created_at": 0,
        "updated_at": 0,
    }
    body.update(fields)
    return JobView.model_validate(body)


def test_next_step_for_a_job_that_never_ran_points_at_route() -> None:
    """Review finding: `gpu logs <id>` for a job no provider ever ran says 'no output'."""
    never_ran = _failed_view(failure_kind="no_provider", accepted_attempts=0)
    assert render.next_step(never_ran) == "gpu route to see which providers fit and why"
    script_failed = _failed_view(failure_kind="user_error", accepted_attempts=1)
    assert render.next_step(script_failed) == "gpu logs a7f2 to see what happened"
