"""Scoring router (phase 5): table-driven routing cases, reasons, fallback and waits.

Providers use the packaged catalog entries (real VRAM, session caps, quotas) with
capabilities set per test, and quota as the ledger would report it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from gpu_router.adapters.base import Capabilities
from gpu_router.models import (
    Job,
    JobSpec,
    JobState,
    ProviderHealth,
    ProviderState,
    QuotaSnapshot,
    QuotaUnit,
    Source,
)
from gpu_router.providers.catalog import GpuOffer, ProviderEntry, load_catalog
from gpu_router.router.base import (
    JobEstimate,
    ProviderSnapshot,
    RejectCode,
    RouteDecision,
    RouteOutcome,
    RoutingContext,
)
from gpu_router.router.scoring import ScoringRouter
from gpu_router.router.settings import RoutingSettings

NOW = 1_790_251_200.0  # Thu 2026-09-24 12:00 UTC
SAT = 1_790_380_800.0  # the next Saturday 00:00 UTC (Kaggle's reset)
H = 3600.0
CAT = load_catalog()
#: phase 7b: modal (the packaged big-VRAM provider) was dropped because it needs a card, so
#: the big-VRAM rules are exercised with a synthetic entry and explicit routing settings
BIG = ProviderEntry(
    name="bigvram",
    kind="fake",
    display_name="Big VRAM (test)",
    priority=40,
    gpus=(
        GpuOffer(name="T4", vram_gb=16),
        GpuOffer(name="L4", vram_gb=24),
        GpuOffer(name="A100-40GB", vram_gb=40),
    ),
    session_hours=24,
    max_concurrency=10,
)
EXTRA = {"bigvram": BIG}
WITH_BIG = RoutingSettings(big_vram_providers={"bigvram": 16})

CAPS: dict[str, dict[str, Any]] = {
    "local": {"resume": True, "live_quota": True, "max_concurrency": 1},
    "kaggle": {"resume": True, "live_quota": True, "max_concurrency": 1},
    "colab": {"resume": True, "interactive": True, "max_concurrency": 1},
    "bigvram": {"resume": True, "max_concurrency": 10},
    "lightning": {"resume": True, "max_concurrency": 1},
}


def quota(
    name: str,
    used: float | None,
    limit: float | None,
    *,
    resets_at: float | None = None,
    source: str = "estimate",
) -> QuotaSnapshot:
    return QuotaSnapshot(
        provider=name,
        used=used or 0.0,
        limit=limit,
        unit=QuotaUnit.GPU_HOURS,
        resets_at=resets_at,
        source=source,  # type: ignore[arg-type]
        observed_at=NOW,
    )


#: What the ledger would say on a quiet Thursday.
DEFAULT_QUOTA = {
    "kaggle": lambda: quota("kaggle", 8, 30, resets_at=SAT, source="live"),
    "colab": lambda: quota("colab", 1, None),
    "local": lambda: quota("local", 0, None, source="live"),
    "bigvram": lambda: quota("bigvram", 0, None),
    "lightning": lambda: quota("lightning", 0, None),
}


@dataclass
class P:
    """One provider in a routing case."""

    name: str
    state: dict[str, Any] = field(default_factory=dict)
    quota: QuotaSnapshot | str | None = "default"
    caps: dict[str, Any] = field(default_factory=dict)
    live: int = 0

    def snapshot(self) -> ProviderSnapshot:
        q = DEFAULT_QUOTA[self.name]() if self.quota == "default" else self.quota
        assert not isinstance(q, str)
        return ProviderSnapshot(
            name=self.name,
            entry=EXTRA.get(self.name) or CAT.get(self.name),
            capabilities=Capabilities(**{**CAPS[self.name], **self.caps}),
            state=ProviderState(provider=self.name, updated_at=NOW, **self.state),
            live_attempts=self.live,
            quota=q,
        )


STANDARD = ("local", "kaggle", "colab")


def job(source: Source = Source.CLI, **spec: Any) -> Job:
    s = JobSpec(project_dir="/p", script="train.py", source=source, **spec)
    return Job(
        id="b" * 12,
        short_id="bbbb",
        name="train",
        state=JobState.ROUTING,
        source=s.source,
        project_dir="/p",
        spec=s,
        spec_hash=hashlib.sha256(s.model_dump_json().encode()).hexdigest(),
        created_at=NOW,
        updated_at=NOW,
    )


def route(
    j: Job,
    providers: tuple[str | P, ...] = STANDARD,
    *,
    estimate: JobEstimate | None = None,
    excluded: frozenset[str] = frozenset(),
    settings: RoutingSettings | None = None,
) -> RouteDecision:
    snaps = [(p if isinstance(p, P) else P(p)).snapshot() for p in providers]
    ctx = RoutingContext(job=j, now=NOW, providers=snaps, excluded=excluded, estimate=estimate)
    d = ScoringRouter(settings).route(ctx)
    json.dumps(d.detail())  # always JSON-safe for job_events.detail
    return d


def rejected(d: RouteDecision) -> dict[str, RejectCode]:
    return {r.provider: r.code for r in d.rejected}


# --------------------------------------------------------------------------- the table


@dataclass
class Case:
    id: str
    spec: dict[str, Any]
    providers: tuple[str | P, ...] = STANDARD
    outcome: RouteOutcome = RouteOutcome.PLACE
    chosen: str | None = None
    order: list[str] | None = None
    reason_has: tuple[str, ...] = ()
    rejects: dict[str, RejectCode] = field(default_factory=dict)
    estimate: JobEstimate | None = None
    settings: RoutingSettings | None = None


CASES = [
    Case(
        "6h job goes to kaggle",
        {"hours": 6},
        chosen="kaggle",
        order=["kaggle", "colab"],
        reason_has=("kaggle: fits 16GB (2xT4)", "6h job suits its 12h sessions"),
        rejects={"local": RejectCode.RESERVED},
    ),
    Case(
        "20 min job goes to colab, kaggle saved",
        {"hours": 20 / 60},
        chosen="colab",
        order=["colab", "kaggle"],
        reason_has=("colab: fits 16GB, kaggle saved for jobs over 4h",),
    ),
    Case(
        "no hours at all counts as short",
        {},
        chosen="colab",
        reason_has=("kaggle saved for jobs over 4h",),
    ),
    Case(
        "24GB goes to a configured big-VRAM provider",
        {"vram_gb": 24, "hours": 1},
        providers=("local", "kaggle", "colab", "bigvram"),
        chosen="bigvram",
        reason_has=("bigvram: fits 24GB", "needs more than 16GB"),
        rejects={
            "local": RejectCode.VRAM,
            "kaggle": RejectCode.VRAM,
            "colab": RejectCode.VRAM,
        },
        settings=WITH_BIG,
    ),
    Case(
        "24GB on the free providers is no-fit and says no free provider is bigger",
        {"vram_gb": 24, "hours": 1},
        outcome=RouteOutcome.NO_FIT,
        reason_has=(
            "no provider fits: needs 24GB VRAM and no free provider has more than 16GB",
            "`gpu providers` lists the excluded ones",
        ),
    ),
    Case(
        "--smoke goes to the local Mac",
        {"smoke": True},
        chosen="local",
        order=["local", "colab", "kaggle"],
        reason_has=("local: MPS 16GB on this Mac, smoke test",),
    ),
    Case(
        "a tiny --hours job is a smoke test too",
        {"hours": 0.05},
        chosen="local",
    ),
    Case(
        "tiny hours with a GPU type is not a smoke test",
        {"hours": 0.05, "gpu": "T4"},
        chosen="colab",
        rejects={"local": RejectCode.GPU_TYPE},
    ),
    Case(
        "kaggle exhausted: 6h job goes to colab and says why",
        {"hours": 6},
        providers=("local", P("kaggle", state={"exhausted_until": NOW + 50 * H}), "colab"),
        chosen="colab",
        reason_has=("colab: fits 16GB, kaggle quota used up, resets in 2d",),
        rejects={"kaggle": RejectCode.EXHAUSTED, "local": RejectCode.RESERVED},
    ),
    Case(
        "kaggle ledger shows nothing left: colab, and kaggle waits for its reset",
        {"hours": 6},
        providers=(
            "local",
            P("kaggle", quota=quota("kaggle", 30, 30, resets_at=SAT, source="live")),
            "colab",
        ),
        chosen="colab",
        reason_has=("kaggle quota used up, resets Sat 00:00 UTC",),
        rejects={"kaggle": RejectCode.QUOTA},
    ),
    Case(
        "too little kaggle quota for a job that does not checkpoint",
        {"hours": 6, "checkpoint_interval_min": 0},
        providers=("local", P("kaggle", quota=quota("kaggle", 28, 30, resets_at=SAT)), "colab"),
        chosen="colab",
        reason_has=("kaggle 2h quota left (est), job needs 6h and does not checkpoint",),
    ),
    Case(
        "a checkpointing job may start on a small remainder and hand off",
        {"hours": 6},
        providers=("local", P("kaggle", quota=quota("kaggle", 28, 30, resets_at=SAT)), "colab"),
        chosen="kaggle",
    ),
    Case(
        "longer than every session and no checkpoints: no fit (the Mac is not a fallback)",
        {"hours": 14, "checkpoint_interval_min": 0},
        outcome=RouteOutcome.NO_FIT,
        reason_has=("14h exceeds its 12h session and the job does not checkpoint",),
        rejects={
            "kaggle": RejectCode.SESSION,
            "colab": RejectCode.SESSION,
            "local": RejectCode.RESERVED,
        },
    ),
    Case(
        "longer than a session but checkpointing: chained through handoff",
        {"hours": 30},
        chosen="kaggle",
    ),
    Case(
        "longer than 12h and no checkpoints: the big-VRAM provider as the last resort",
        {"hours": 14, "checkpoint_interval_min": 0},
        providers=("local", "kaggle", "colab", "bigvram"),
        chosen="bigvram",
        reason_has=("the only provider that can take this job",),
        settings=WITH_BIG,
    ),
    Case(
        "interactive goes to colab",
        {"interactive": True, "hours": 1},
        chosen="colab",
        reason_has=("interactive",),
        rejects={"kaggle": RejectCode.INTERACTIVE},
    ),
    Case(
        "everything busy: wait, the Mac stays reserved",
        {"hours": 1},
        providers=(
            "local",
            P("kaggle", state={"cooldown_until": NOW + 600}),
            P("colab", state={"health": ProviderHealth.UNAVAILABLE, "cooldown_until": NOW + 300}),
        ),
        outcome=RouteOutcome.WAIT,
        reason_has=("all providers busy",),
        rejects={
            "local": RejectCode.RESERVED,
            "kaggle": RejectCode.COOLDOWN,
            "colab": RejectCode.UNHEALTHY,
        },
    ),
    Case(
        "only the Mac is enabled: it runs everything",
        {"hours": 2},
        providers=("local",),
        chosen="local",
        reason_has=("no cloud provider can take this job",),
    ),
    Case(
        "kaggle quota about to expire unused: spend it even on a short job",
        {"hours": 0.5},
        providers=(
            "local",
            P("kaggle", quota=quota("kaggle", 10, 30, resets_at=NOW + 10 * H, source="live")),
            "colab",
        ),
        chosen="kaggle",
        reason_has=("20h left, resets in 10h: use it or lose it",),
    ),
    Case(
        "heuristic 6h estimate routes like an explicit 6h",
        {},
        estimate=JobEstimate(hours=6, hours_source="heuristic", vram_gb=8, vram_source="heuristic"),
        chosen="kaggle",
    ),
    Case(
        "heuristic VRAM over 16GB prefers a configured big-VRAM provider",
        {"hours": 1},
        providers=("local", "kaggle", "colab", "bigvram"),
        estimate=JobEstimate(hours=1, vram_gb=20, vram_source="heuristic"),
        chosen="bigvram",
        settings=WITH_BIG,
    ),
    Case(
        "heuristic VRAM over 16GB never blocks the free GPUs",
        {"hours": 1},
        estimate=JobEstimate(hours=1, vram_gb=20, vram_source="heuristic"),
        chosen="colab",
        reason_has=("colab: T4 16GB",),
    ),
]


@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
def test_routing_table(case: Case) -> None:
    d = route(job(**case.spec), case.providers, estimate=case.estimate, settings=case.settings)
    assert d.outcome is case.outcome, d.reason
    assert d.router == "scoring"
    if case.chosen is not None:
        assert d.chosen is not None
        assert d.chosen.provider == case.chosen, d.reason
        assert d.chosen == d.candidates[0]
    if case.order is not None:
        assert [c.provider for c in d.candidates] == case.order
    for text in case.reason_has:
        assert text in d.reason, d.reason
    for name, code in case.rejects.items():
        assert rejected(d).get(name) is code, d.rejected


# --------------------------------------------------------------------------- details


def test_every_candidate_has_a_one_line_reason_and_quota_facts() -> None:
    d = route(job(hours=2))
    assert [c.provider for c in d.candidates] == ["colab", "kaggle"]
    for c in d.candidates:
        assert c.reason.startswith(f"{c.provider}: ")
        assert "\n" not in c.reason
    kaggle = d.candidates[1]
    assert kaggle.reason == "kaggle: fits 16GB (2xT4), saved for jobs over 4h, resets Sat 00:00 UTC"
    assert kaggle.quota_left == 22
    assert kaggle.quota_unit == "gpu_hours"
    assert kaggle.quota_share == pytest.approx(2 / 22, abs=1e-3)
    assert kaggle.resets_at == SAT
    assert d.hours == 2
    assert d.hours_source == "spec"
    assert d.smoke is False


def test_wait_retries_at_the_earliest_known_time() -> None:
    d = route(
        job(hours=6),
        (
            P("kaggle", quota=quota("kaggle", 30, 30, resets_at=SAT, source="live")),
            P("colab", state={"cooldown_until": NOW + 900}),
        ),
    )
    assert d.outcome is RouteOutcome.WAIT
    assert d.retry_at == NOW + 900
    d2 = route(
        job(hours=6),
        (P("kaggle", quota=quota("kaggle", 30, 30, resets_at=SAT, source="live")),),
    )
    assert d2.outcome is RouteOutcome.WAIT  # quota comes back at the reset: wait, not fail
    assert d2.retry_at == SAT


def test_reset_soonest_breaks_ties_between_equal_providers() -> None:
    # two unreserved providers with no role: the one whose quota resets first wins
    settings = RoutingSettings(save_for_long_jobs=(), short_job_providers=())
    d = route(
        job(hours=1),
        (
            P("colab", quota=quota("colab", 0, 30, resets_at=NOW + 20 * 24 * H)),
            P("kaggle", quota=quota("kaggle", 0, 30, resets_at=NOW + 24 * H)),
        ),
        settings=settings,
    )
    assert d.chosen is not None
    assert d.chosen.provider == "kaggle"
    assert "resets in" in d.chosen.reason or "resets " in d.chosen.reason


def test_share_over_half_is_penalised_and_named() -> None:
    d = route(
        job(hours=6),
        ("local", P("kaggle", quota=quota("kaggle", 20, 30, resets_at=SAT)), "colab"),
    )
    kaggle = next(c for c in d.candidates if c.provider == "kaggle")
    assert kaggle.quota_share == pytest.approx(0.6)
    assert "uses 60% of the 10h left" in kaggle.reason


def test_a_busy_mac_never_makes_a_cloud_job_wait() -> None:
    d = route(
        job(hours=14, checkpoint_interval_min=0),
        (P("local", live=1), "kaggle", "colab"),
    )
    assert d.outcome is RouteOutcome.NO_FIT  # not WAIT for a Mac it may not use anyway
    assert rejected(d)["local"] is RejectCode.RESERVED
    assert d.reason.startswith("no provider fits: kaggle: 14h exceeds")
    smoke = route(job(smoke=True), (P("local", live=1), "colab"))
    assert smoke.chosen is not None
    assert smoke.chosen.provider == "colab"  # a smoke test does not wait for the Mac either
    assert rejected(smoke)["local"] is RejectCode.CAPACITY


def test_excluded_and_capacity() -> None:
    d = route(job(hours=6), ("kaggle", P("colab", live=1)), excluded=frozenset({"kaggle"}))
    assert rejected(d) == {"kaggle": RejectCode.EXCLUDED, "colab": RejectCode.CAPACITY}
    assert d.outcome is RouteOutcome.WAIT


def test_pinned_provider_uses_the_simple_router_with_quota_facts() -> None:
    d = route(job(hours=0.5, provider="kaggle"))
    assert d.router == "simple"
    assert d.chosen is not None
    assert d.chosen.provider == "kaggle"
    assert d.chosen.reason == "kaggle: pinned with --provider (2xT4 16GB)"
    assert d.chosen.quota_left == 22
    assert d.chosen.quota_share == pytest.approx(0.5 / 22, abs=1e-3)
    assert d.hours == 0.5
    local = route(job(provider="local"))
    assert local.chosen is not None
    assert local.chosen.provider == "local"  # a pin beats the smoke-test reservation
    unknown = route(job(provider="modal"))
    assert unknown.outcome is RouteOutcome.NO_FIT
    assert "not enabled" in unknown.reason


def test_settings_are_data() -> None:
    settings = RoutingSettings(long_job_hours=1, save_for_long_jobs=("colab",))
    d = route(job(hours=2), settings=settings)
    assert d.chosen is not None
    assert d.chosen.provider == "colab"  # colab is now the long-job provider


def test_no_providers() -> None:
    d = route(job(hours=1), ())
    assert d.outcome is RouteOutcome.NO_FIT
    assert "no providers are enabled" in d.reason
