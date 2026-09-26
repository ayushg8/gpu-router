"""`routing:` section of config.yaml, typed (phase 5). Until phase 5 it was a free-form,
unused mapping, so typing it needs no config migration: an empty mapping means defaults.

    routing:
      version: 1
      long_job_hours: 4             # "save Kaggle for long jobs (over 4 hr)"
      save_for_long_jobs: [kaggle]  # penalised for jobs at or under long_job_hours
      short_job_providers: [colab]  # preferred for short (<= long_job_hours) or interactive
      big_vram_providers: {}        # {name: GB}: kept for jobs needing MORE than GB VRAM
                                    #   (phase 7b: empty; modal, the only free big-GPU
                                    #   option, was dropped because it needs a card)
      smoke_max_minutes: 5          # --hours at or under this (no GPU/VRAM ask) = smoke test
      handoff_min_hours: 0.5        # quota a checkpointing job needs to be worth starting
      quota: {ttl_s: 1800, refresh_s: 1800, wait_s: 8, unknown_window_hours: 24}

Provider names are data here, not code: the router has no special case for "kaggle" or
"colab" beyond these lists. The local Mac is recognised by catalog `kind: local`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from gpu_router.quota.settings import QuotaSettings

__all__ = ["RoutingSettings", "routing_settings"]


class RoutingSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal[1] = 1
    long_job_hours: float = Field(default=4.0, gt=0)
    save_for_long_jobs: tuple[str, ...] = ("kaggle",)
    short_job_providers: tuple[str, ...] = ("colab",)
    big_vram_providers: dict[str, float] = Field(default_factory=dict)
    smoke_max_minutes: float = Field(default=5.0, ge=0)
    handoff_min_hours: float = Field(default=0.5, gt=0)
    quota: QuotaSettings = Field(default_factory=QuotaSettings)


def routing_settings(raw: Mapping[str, Any] | None) -> RoutingSettings:
    """Validate config.routing. Raises ConfigError naming the bad keys."""
    try:
        return RoutingSettings.model_validate(dict(raw or {}))
    except ValidationError as exc:
        from gpu_router.errors import ConfigError

        problems = "; ".join(
            f"routing.{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()[:5]
        )
        raise ConfigError(
            f"config.yaml is invalid: {problems}",
            hint="fix the listed keys under `routing:`, or remove the section to use defaults",
        ) from None
