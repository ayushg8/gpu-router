"""Immutable data the feed hands to the UI thread (phase 4)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum

from gpu_router.api import JobView, ProviderView


class ConnStatus(StrEnum):
    CONNECTING = "connecting"  # first contact, nothing known yet
    STARTING = "starting"  # no daemon answered; auto-start is spawning one
    UP = "up"
    DOWN = "down"  # not running and not started (or start failed); retrying


@dataclass(frozen=True)
class Conn:
    status: ConnStatus
    message: str = ""
    hint: str = ""
    retry_s: float = 0.0
    last_ok: float | None = None  # clock time of the last successful status poll
    started_pid: int | None = None  # set once when this shell auto-started the daemon
    autostart: bool = True  # False: GPU_ROUTER_NO_AUTOSTART or GpuShell(autostart=False)


@dataclass(frozen=True)
class JobMetric:
    """What the panel needs from a job's metric history (a copy; the feed owns the rest)."""

    name: str | None = None
    values: tuple[float, ...] = ()  # newest last, at most PANEL_POINTS
    step: int | None = None
    total: int | None = None


@dataclass(frozen=True)
class Snapshot:
    conn: Conn
    active: tuple[JobView, ...] = ()
    recent: tuple[JobView, ...] = ()
    providers: tuple[ProviderView, ...] = ()
    metrics: Mapping[str, JobMetric] = field(default_factory=dict)  # job id -> metric
    ckpt_where: Mapping[str, str] = field(default_factory=dict)  # job id -> "HF Hub", ...
    taken_at: float = 0.0

    @property
    def running(self) -> int:
        from gpu_router.statemachine import JobState

        return sum(1 for j in self.active if j.state in (JobState.RUNNING, JobState.CHECKPOINTING))
