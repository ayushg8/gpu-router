"""`routing.quota` settings in config.yaml (phase 5). Pydantic only; no store, no I/O."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class QuotaSettings(BaseModel):
    """How the quota ledger caches live readings and estimates the rest.

    routing:
      quota:
        ttl_s: 1800                # a live reading younger than this is used as is
        refresh_s: 1800            # background refresh of live readings (daemon)
        wait_s: 8                  # `gpu quota` waits this long for fresh live readings
        retry_failed_s: 300        # after a failed live call, wait this long to call again
        unknown_window_hours: 24   # rolling window for providers with an unknown reset
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    ttl_s: float = Field(default=1800, gt=0)
    refresh_s: float = Field(default=1800, gt=0)
    wait_s: float = Field(default=8, ge=0, le=120)
    retry_failed_s: float = Field(default=300, ge=0)
    unknown_window_hours: float = Field(default=24, gt=0, le=24 * 31)

    @property
    def unknown_window_s(self) -> float:
        return self.unknown_window_hours * 3600
