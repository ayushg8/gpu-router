"""Which job events become notifications, and their text (phase 8a). Pure: no I/O.

Spec UX principle 5: a macOS notification when a job finishes, fails, needs approval or
migrates, in the one visual language (✓ ✗ ⏸ ↪). Job names are job-controlled, so they are
stripped of control characters and cut; every other string is gpu-router's own text
(event messages, approval and route reasons say what happened and what happens next).
"""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum

from gpu_router.models import Job, JobEvent
from gpu_router.statefile import clean_text
from gpu_router.statemachine import JobState, Reason

__all__ = ["Kind", "Notification", "build", "classify", "harmless_notification", "summary"]

TITLE = "gpu-router"
NAME_MAX = 48
SUBTITLE_MAX = 80
BODY_MAX = 220
SOUND_ATTENTION = "Glass"  # approval: needs you
SOUND_FAILED = "Basso"  # failed


class Kind(StrEnum):
    FINISHED = "finished"
    FAILED = "failed"
    APPROVAL = "approval"
    MIGRATED = "migrated"
    SUMMARY = "summary"  # rate limit: "N more job updates"
    TEST = "test"  # `gpu notify test`


@dataclass(frozen=True, slots=True)
class Notification:
    kind: str
    title: str
    subtitle: str
    body: str
    job_id: str | None = None
    sound: str | None = None
    group: str | None = None  # terminal-notifier -group: a newer one replaces the older

    def to_json(self) -> dict[str, str | None]:
        return {
            "kind": self.kind,
            "title": self.title,
            "subtitle": self.subtitle,
            "body": self.body,
            "job_id": self.job_id,
            "sound": self.sound,
        }


def classify(event: JobEvent) -> Kind | None:
    """The notification an event deserves, or None.

    finished: -> done. failed: -> failed. approval: -> awaiting_approval, or the in-place
    re-ask note (approval_required with detail.reask, D43). migrated: -> migrating, except
    hours_exceeded (D48), which asks for approval right after and notifies as that.
    Cancelled and denied jobs were stopped by the user: no notification.
    """
    if event.kind == "note":
        if event.reason == Reason.APPROVAL_REQUIRED and event.detail.get("reask"):
            return Kind.APPROVAL
        return None
    to = event.to_state
    if to == JobState.DONE:
        return Kind.FINISHED
    if to == JobState.FAILED:
        return Kind.FAILED
    if to == JobState.AWAITING_APPROVAL:
        return Kind.APPROVAL
    if to == JobState.MIGRATING and event.reason != Reason.HOURS_EXCEEDED:
        return Kind.MIGRATED
    return None


def _cut(text: str, limit: int) -> str:
    text = " ".join(clean_text(text).split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _dur(seconds: float | None) -> str | None:
    if seconds is None or seconds < 0:
        return None
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    m = s // 60
    if m < 60:
        return f"{m}m"
    h, m = divmod(m, 60)
    if h < 48:
        return f"{h}h{m:02d}m" if m else f"{h}h"
    d, h = divmod(h, 24)
    return f"{d}d{h}h" if h else f"{d}d"


def _outputs(job: Job, user_home: str | None) -> str | None:
    """`myproj/runs/a7f2` (project folder + runs dir): readable without the full path."""
    if not job.outputs_dir:
        return None
    out = job.outputs_dir
    project = job.project_dir.rstrip("/")
    if project and out.startswith(project + "/"):
        return f"{os.path.basename(project)}/{out[len(project) + 1 :]}"
    home = (user_home or "").rstrip("/")
    if home and out.startswith(home + "/"):
        return "~/" + out[len(home) + 1 :]
    return out


def _where(job: Job) -> str | None:
    if not job.provider:
        return None
    return f"{job.provider} {job.gpu}" if job.gpu else job.provider


def build(
    kind: Kind,
    job: Job | None,
    event: JobEvent,
    *,
    sound: bool = True,
    user_home: str | None = None,
) -> Notification:
    """Text for one notification. `job` may be None (lookup failed): the event alone
    still says what happened, with the id prefix instead of the name."""
    short = job.short_id if job is not None else event.job_id[:4]
    name = _cut(job.name, NAME_MAX) if job is not None and job.name else f"job {short}"
    group = f"gpu-router.{event.job_id}"
    msg = _cut(event.message, BODY_MAX) if event.message else ""

    if kind is Kind.FINISHED:
        parts: list[str] = []
        took = (
            _dur(job.finished_at - job.started_at)
            if job and job.started_at and job.finished_at
            else None
        )
        where = _where(job) if job is not None else None
        if took and where:
            parts.append(f"{took} on {where}")
        elif where:
            parts.append(f"on {where}")
        elif took:
            parts.append(took)
        outputs = _outputs(job, user_home) if job is not None else None
        if outputs and job is not None and job.outputs_fetched:
            parts.append(f"outputs in {outputs}")
        elif job is not None and job.outputs_dir:
            parts.append(f"gpu fetch {short} downloads the outputs")
        body = " · ".join(parts) or msg or "done"
        return Notification(
            kind=kind,
            title=TITLE,
            subtitle=_cut(f"✓ {name} finished", SUBTITLE_MAX),
            body=_cut(body, BODY_MAX),
            job_id=event.job_id,
            group=group,
        )

    if kind is Kind.FAILED:
        head: list[str] = []
        if job is not None and job.exit_code is not None:
            head.append(f"exit {job.exit_code}")
        where = _where(job) if job is not None else None
        took = (
            _dur(job.finished_at - job.started_at)
            if job and job.started_at and job.finished_at
            else None
        )
        if where and took:
            head.append(f"after {took} on {where}")
        elif where:
            head.append(f"on {where}")
        first = " ".join(head)
        # a script failure is summed up by its exit code; anything else (no provider fits,
        # budgets spent) is explained by gpu-router's own message
        detail = first if (first and job is not None and job.exit_code is not None) else msg
        body = " · ".join(p for p in (detail, f"gpu logs {short}") if p)
        return Notification(
            kind=kind,
            title=TITLE,
            subtitle=_cut(f"✗ {name} failed", SUBTITLE_MAX),
            body=_cut(body, BODY_MAX),
            job_id=event.job_id,
            sound=SOUND_FAILED if sound else None,
            group=group,
        )

    if kind is Kind.APPROVAL:
        where = _where(job) if job is not None else None
        reason = ""
        if job is not None and job.approval_reason:
            reason = _cut(job.approval_reason, 120)
        elif msg:
            reason = msg
        parts = [f"→ {where}" if where else "", reason, f"gpu approve {short}"]
        return Notification(
            kind=kind,
            title=TITLE,
            subtitle=_cut(f"⏸ {name} needs your approval", SUBTITLE_MAX),
            body=_cut(" · ".join(p for p in parts if p), BODY_MAX),
            job_id=event.job_id,
            sound=SOUND_ATTENTION if sound else None,
            group=group,
        )

    if kind is Kind.MIGRATED:
        # the event names the provider it left; by the time the job is read it may already
        # be placed somewhere else
        prev = event.detail.get("previous_provider")
        origin = (prev if isinstance(prev, str) and prev else None) or (
            job.provider if job is not None else None
        )
        subtitle = f"↪ {name} is moving off {origin}" if origin else f"↪ {name} is moving"
        return Notification(
            kind=kind,
            title=TITLE,
            subtitle=_cut(subtitle, SUBTITLE_MAX),
            body=_cut(msg or "it resumes on another provider from its latest checkpoint", BODY_MAX),
            job_id=event.job_id,
            group=group,
        )

    raise ValueError(f"no job notification for {kind!r}")


def summary(counts: Counter[str]) -> Notification:
    """One notification for the updates the rate limit held back."""
    total = sum(counts.values())
    words = {
        "finished": "finished",
        "failed": "failed",
        "approval": "need approval",
        "migrated": "moved",
    }
    parts = [f"{n} {words.get(k, k)}" for k, n in counts.most_common()]
    return Notification(
        kind=Kind.SUMMARY,
        title=TITLE,
        subtitle=f"{total} more job update{'s' if total != 1 else ''}",
        body=_cut(" · ".join(parts) + " · gpu status shows them", BODY_MAX),
        group="gpu-router.summary",
    )


def harmless_notification() -> Notification:
    return Notification(
        kind=Kind.TEST,
        title=TITLE,
        subtitle="gpu-router test notification",
        body="notifications work; nothing ran and no job changed",
        group="gpu-router.test",
    )
