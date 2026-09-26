"""`gpu doctor` result shapes (phase 8a). Stable for `--json` consumers: additive only.

Every check has a status (ok | warn | fail | skip), a one-line summary of what it found,
and, when something needs doing, `fix`: the exact command to run (never a secret; paths are
shell-quoted). `skip` = the check could not apply here (daemon down for a live check, a
provider that is not enabled), with the reason as summary.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["GROUPS", "CheckResult", "DriftItem", "Report", "Status"]

#: Display order of the check groups.
GROUPS: tuple[str, ...] = (
    "daemon",
    "providers",
    "storage",
    "inference",
    "limits",
    "local",
    "integration",
)


class Status(StrEnum):
    OK = "ok"
    WARN = "warn"
    FAIL = "fail"
    SKIP = "skip"


_RANK = {Status.FAIL: 0, Status.WARN: 1, Status.SKIP: 2, Status.OK: 3}


def worst(*statuses: Status) -> Status:
    return min(statuses, key=lambda s: _RANK[s]) if statuses else Status.OK


class CheckResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str  # "provider.kaggle.live"
    group: str  # one of GROUPS
    title: str  # "kaggle live"
    status: Status
    summary: str
    fix: str | None = None
    detail: dict[str, Any] = Field(default_factory=dict)
    elapsed_ms: int = 0


class DriftItem(BaseModel):
    """A providers.yaml number that reality disagrees with (`gpu doctor --update-catalog`
    writes `live` into <home>/providers.yaml)."""

    model_config = ConfigDict(extra="forbid")

    provider: str
    key: str  # dotted path under providers.<name>: "quota.limit", "quota.reset_anchor"
    catalog: Any
    live: Any
    note: str


class Report(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str  # this CLI's gpu-router version
    home: str  # data dir checked
    checked_at: float
    elapsed_ms: int
    ok: bool  # no check failed
    counts: dict[str, int]
    checks: list[CheckResult]
    drift: list[DriftItem] = Field(default_factory=list)
    #: rows that verified nothing: the check crashed or did not finish by the deadline
    #: (their detail says "unknown": "crashed" | "timeout"). `gpu doctor` exits 1 when any
    #: exist and never calls the result "everything works" (phase 8 review fix).
    unknown: int = 0

    def by_group(self) -> list[tuple[str, list[CheckResult]]]:
        out: list[tuple[str, list[CheckResult]]] = []
        seen = [g for g in GROUPS] + sorted({c.group for c in self.checks} - set(GROUPS))
        for group in seen:
            rows = [c for c in self.checks if c.group == group]
            if rows:
                out.append((group, rows))
        return out
