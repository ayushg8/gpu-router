"""Builders for API views used by the shell's pure-formatting tests."""

from __future__ import annotations

from typing import Any

from gpu_router.adapters.base import Capabilities
from gpu_router.api import JobView, ProviderView
from gpu_router.models import (
    JobSpec,
    JobState,
    Progress,
    ProviderHealth,
    QuotaSnapshot,
    QuotaUnit,
    Source,
)

NOW = 1_790_000_000.0  # a fixed "now" for formatting tests


def job(
    sid: str = "a7f2",
    name: str = "train_yolo.py",
    state: JobState = JobState.RUNNING,
    **fields: Any,
) -> JobView:
    job_id = (sid + "0" * 12)[:12]
    base: dict[str, Any] = {
        "id": job_id,
        "short_id": sid,
        "name": name,
        "state": state,
        "source": Source.CLI,
        "project_dir": "/tmp/proj",
        "spec": JobSpec(project_dir="/tmp/proj", script=name),
        "spec_hash": "x",
        "created_at": NOW - 120,
        "updated_at": NOW - 1,
    }
    if state in (JobState.RUNNING, JobState.CHECKPOINTING):
        base.update(
            provider="kaggle", gpu="2xT4", started_at=NOW - 6130, current_attempt_id=f"{job_id}.1"
        )
    base.update(fields)
    if "progress" in base and isinstance(base["progress"], dict):
        base["progress"] = Progress(**base["progress"])
    return JobView.model_validate(base)


def quota(
    provider: str,
    used: float,
    limit: float | None,
    *,
    unit: QuotaUnit = QuotaUnit.GPU_HOURS,
    resets_in: float | None = 3 * 86400,
    source: str = "live",
) -> QuotaSnapshot:
    return QuotaSnapshot(
        provider=provider,
        used=used,
        limit=limit,
        unit=unit,
        resets_at=NOW + resets_in if resets_in is not None else None,
        source=source,  # type: ignore[arg-type]
        observed_at=NOW - 5,
    )


def provider(
    name: str,
    *,
    health: ProviderHealth = ProviderHealth.OK,
    q: QuotaSnapshot | None = None,
    enabled: bool = True,
    reason: str | None = None,
) -> ProviderView:
    return ProviderView(
        name=name,
        display_name=name.title(),
        kind=name,
        enabled=enabled,
        health=health,
        health_reason=reason,
        capabilities=Capabilities(),
        gpus=["T4"],
        quota=q,
    )


def mockup_providers() -> list[ProviderView]:
    """The spec footer: kaggle 22/30h ↻Sat │ colab ● up │ lightning 14/20h (the mockup's
    `modal $24` is gone: modal was dropped in phase 7b because it needs a card)."""
    return [
        provider("kaggle", q=quota("kaggle", 22, 30)),
        provider("colab", q=quota("colab", 3, None, source="estimate", resets_in=None)),
        provider("lightning", q=quota("lightning", 14, 20, resets_in=20 * 86400)),
    ]
