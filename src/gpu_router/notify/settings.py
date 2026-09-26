"""`notifications:` section of config.yaml, typed (phase 8a).

    notifications:
      enabled: true            # master switch
      backend: auto            # auto | terminal-notifier | osascript | off
      events:                  # one switch per event type
        finished: true         # a job reached done (outputs fetched)
        failed: true           # a job ended failed
        approval: true         # a job waits for /approve (also a re-ask, D43)
        migrated: true         # a job left its provider and is moving to another
      sound: true              # approval and failed play a sound; the others are silent
      dedupe_s: 600            # same job + same event within this window -> one notification
      max_per_minute: 6        # rate limit; the rest fold into one summary notification
      timeout_s: 10            # a notifier process slower than this is killed

`auto` picks terminal-notifier when it is on PATH, else osascript, and is OFF in test mode
and under pytest (a test suite must never pop real notifications); an explicit backend is
honoured in test mode but still never under pytest unless GPU_ROUTER_NOTIFY_REAL=1.
Config is free-form in `config.Config.notifications` (like `routing:` / `policy:`) and
validated here, so a bad section is a ConfigError at daemon start.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from gpu_router.errors import ConfigError

__all__ = ["EVENT_KINDS", "NotifyEvents", "NotifySettings", "notify_settings"]

#: Event types a user can switch on and off (the keys of `events:`).
EVENT_KINDS: tuple[str, ...] = ("finished", "failed", "approval", "migrated")

Backend = Literal["auto", "terminal-notifier", "osascript", "off"]


class _S(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NotifyEvents(_S):
    finished: bool = True
    failed: bool = True
    approval: bool = True
    migrated: bool = True


class NotifySettings(_S):
    version: int = 1
    enabled: bool = True
    backend: Backend = "auto"
    events: NotifyEvents = Field(default_factory=NotifyEvents)
    sound: bool = True
    dedupe_s: float = Field(default=600, ge=0)
    max_per_minute: int = Field(default=6, ge=1, le=60)
    timeout_s: float = Field(default=10, gt=0, le=60)

    def wants(self, kind: str) -> bool:
        """True when notifications are on and this event type is switched on."""
        if not self.enabled or self.backend == "off":
            return False
        return bool(getattr(self.events, kind, False))

    def enabled_kinds(self) -> list[str]:
        return [k for k in EVENT_KINDS if self.wants(k)]


def notify_settings(
    raw: Mapping[str, Any] | None, *, source: str = "config.yaml"
) -> NotifySettings:
    """Validate the raw `notifications:` mapping. Raises ConfigError naming the bad keys."""
    try:
        return NotifySettings.model_validate(dict(raw or {}))
    except ValidationError as exc:
        problems = "; ".join(
            f"notifications.{'.'.join(str(p) for p in err['loc'])}: {err['msg']}"
            for err in exc.errors()[:5]
        )
        raise ConfigError(
            f"{source}: {problems}",
            hint="fix the `notifications:` section (keys: enabled, backend, events.{"
            + ",".join(EVENT_KINDS)
            + "}, sound, dedupe_s, max_per_minute, timeout_s)",
            detail={"section": "notifications"},
        ) from None
