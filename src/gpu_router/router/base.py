"""Router contract (phase 1; real code, frozen interface; owner: group C).

A router is PURE: it gets a `RoutingContext` snapshot assembled by the engine (catalog
entries, adapter capabilities, provider runtime state, live attempt counts, latest quota,
exclusions) and returns a `RouteDecision`. No I/O, no clock reads (ctx.now), no store
access. That makes `/v1/route` dry runs, the job detail view and tests trivial, and lets
phase 5 swap in the scoring router without touching the engine.

Spec "Router": 1 filter, 2 score, 3 pick and explain (one-line reason per decision),
4 fallback (the engine re-routes on provisioning failure; the ranked list is recorded in the
placement event's detail).
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from gpu_router.adapters.base import Capabilities
from gpu_router.models import Job, ProviderState, QuotaSnapshot
from gpu_router.providers.catalog import ProviderEntry

__all__ = [
    "TEMPORARY_REJECTIONS",
    "Candidate",
    "JobEstimate",
    "ProviderSnapshot",
    "RejectCode",
    "Rejection",
    "RouteDecision",
    "RouteOutcome",
    "Router",
    "RoutingContext",
    "earliest_retry",
]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ProviderSnapshot(_Frozen):
    """Everything the router may know about one registered provider at decision time."""

    name: str
    entry: ProviderEntry
    capabilities: Capabilities
    state: ProviderState
    live_attempts: int = 0
    quota: QuotaSnapshot | None = None


class JobEstimate(_Frozen):
    """Phase 5: the bundle's VRAM / runtime estimate (`manifest.json["estimate"]`, D17), or
    one computed for a dry run. Explicit spec values always win over it; a `heuristic`
    VRAM guess is a preference, never a hard filter."""

    vram_gb: float | None = None
    hours: float | None = None
    vram_source: Literal["spec", "heuristic"] | None = None
    hours_source: Literal["spec", "heuristic"] | None = None
    mode: str | None = None  # inference | lora | train | unknown
    reasons: tuple[str, ...] = ()


class RoutingContext(_Frozen):
    job: Job
    now: float
    providers: Sequence[ProviderSnapshot]  # registered (enabled) providers, catalog order
    excluded: frozenset[str] = frozenset()  # store.excluded_providers(job.id)
    previous_provider: str | None = None  # provider of the last attempt (migration)
    resuming: bool = False  # a checkpoint exists and will be resumed
    estimate: JobEstimate | None = None  # phase 5: bundle estimate (None = spec only)
    resume_step: int | None = None  # step of the checkpoint a resume starts from (D44)


class RejectCode(StrEnum):
    OVERRIDE = "override"  # spec.provider names another provider
    EXCLUDED = "excluded"  # InvalidJob/abandoned earlier for this job
    VRAM = "vram"  # no GPU with >= spec.vram_gb
    GPU_TYPE = "gpu_type"  # spec.gpu not offered
    SESSION = "session"  # spec.hours > session cap and resume impossible
    INTERACTIVE = "interactive"  # spec.interactive but provider cannot
    AUTH = "auth"  # health auth_required
    DISABLED = "disabled"  # health disabled
    UNHEALTHY = "unhealthy"  # health unavailable (temporary)
    COOLDOWN = "cooldown"  # cooldown_until > now (temporary)
    EXHAUSTED = "exhausted"  # exhausted_until > now (temporary)
    CAPACITY = "capacity"  # live_attempts >= max_concurrency (temporary)
    QUOTA = "quota"  # phase 5: remaining quota cannot fit spec.hours
    RESERVED = "reserved"  # phase 5: kept for other jobs (local: smoke tests, big_vram_providers)


#: Rejections that may clear by themselves: if every provider is rejected and at least one
#: rejection is temporary, the outcome is WAIT (retry_at = earliest `until`), not NO_FIT.
#: When one of them has no `until` (capacity, a login) the engine retries no later than its
#: own bounded backoff, so a far-off quota reset never hides a slot that frees up (D44).
#: AUTH counts as temporary: the user can `gpu login` while the job waits (up to
#: engine.max_queue_wait_s), and the job's message tells them to.
TEMPORARY_REJECTIONS: frozenset[RejectCode] = frozenset(
    {
        RejectCode.UNHEALTHY,
        RejectCode.COOLDOWN,
        RejectCode.EXHAUSTED,
        RejectCode.CAPACITY,
        RejectCode.AUTH,
    }
)


class Rejection(_Frozen):
    provider: str
    code: RejectCode
    reason: str  # "kaggle: in cooldown for 4m (rate limited)"
    until: float | None = None  # when a temporary rejection lifts, if known


def earliest_retry(temporary: Sequence[Rejection]) -> float | None:
    """WAIT's retry_at: the earliest known `until` (None when none is known). A rejection
    without one (a busy provider, a login) may clear any time: the engine then retries
    no later than its own backoff, whatever this says (D44)."""
    untils = [r.until for r in temporary if r.until is not None]
    return min(untils) if untils else None


class Candidate(_Frozen):
    provider: str
    gpu: str | None = None  # GpuOffer.label, e.g. "2xT4"
    vram_gb: float | None = None
    score: float = 0.0  # higher is better; phase 1 uses -priority
    reason: str  # one line: "colab: fits 16GB, kaggle saved for jobs over 4h"
    # phase 5 (scoring router; None when unknown). Read by the approval policy.
    quota_left: float | None = None  # remaining quota in `quota_unit` (ledger view)
    quota_unit: str | None = None  # QuotaUnit value
    quota_share: float | None = None  # job hours / quota_left (gpu_hours units only)
    resets_at: float | None = None  # when this provider's quota resets, if known
    quota_source: str | None = None  # "live" | "estimate": which the numbers are (D44)


class RouteOutcome(StrEnum):
    PLACE = "place"
    WAIT = "wait"
    NO_FIT = "no_fit"


class RouteDecision(_Frozen):
    outcome: RouteOutcome
    chosen: Candidate | None = None  # set iff outcome == PLACE (== candidates[0])
    candidates: list[Candidate] = Field(default_factory=list)  # ranked, best first
    rejected: list[Rejection] = Field(default_factory=list)
    reason: str  # one line shown in /route and job detail
    retry_at: float | None = None  # WAIT: earliest time something may free up
    router: Literal["simple", "scoring"] | str = "simple"
    # phase 5: what the scoring router assumed about the job (None = unknown)
    hours: float | None = None  # expected runtime used for filters, scores and policy
    hours_source: Literal["spec", "heuristic"] | None = None
    smoke: bool = False  # treated as a smoke test (local MPS preferred)

    def detail(self) -> dict[str, object]:
        """JSON-safe dict for job_events.detail (placement/no-capacity events)."""
        return self.model_dump(mode="json")


class Router(Protocol):
    name: str

    def route(self, ctx: RoutingContext) -> RouteDecision:
        """Filter, rank, pick, explain. Pure and deterministic for a given ctx."""
        ...
