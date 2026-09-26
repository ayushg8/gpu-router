"""Phase 4/5 review regressions for the scoring router and the approval policy (D44)."""

from __future__ import annotations

import pytest

from gpu_router.models import JobState, Progress, Source
from gpu_router.policy import RulesPolicy
from gpu_router.router.base import RejectCode, RouteOutcome, RoutingContext
from gpu_router.router.scoring import SHARE_NOTHING_LEFT, ScoringRouter
from tests.unit.router.test_scoring import NOW, SAT, P, job, quota, rejected, route


def test_a_reserved_mac_does_not_hide_that_every_cloud_provider_refused_the_job() -> None:
    """Finding: the local RESERVED rejection broke the all-excluded checks, so a job that
    Kaggle and Colab both refused as invalid failed as no_provider with a smoke-test hint."""
    d = route(job(hours=1), excluded=frozenset({"kaggle", "colab"}))
    assert d.outcome is RouteOutcome.NO_FIT
    assert rejected(d)["local"] is RejectCode.RESERVED
    assert d.reason == "no provider fits: every provider rejected this job (kaggle, colab)"
    # the VRAM summary ignores the reservation too
    big = route(job(hours=1, vram_gb=80))
    assert big.reason.startswith(
        "no provider fits: needs 80GB VRAM and no free provider has more than"
    )


def test_a_pinned_provider_with_nothing_left_still_trips_the_quota_rule() -> None:
    """Finding: 0h left gave quota_share None, so the >50% rule stayed silent while 0.1h
    left asked at 500%."""
    empty = P("kaggle", quota=quota("kaggle", 30, 30, resets_at=SAT, source="live"))
    d = route(job(Source.AGENT, hours=0.5, provider="kaggle"), (empty, "colab"))
    assert d.chosen is not None
    assert d.chosen.quota_left == 0
    assert d.chosen.quota_share == SHARE_NOTHING_LEFT
    got = RulesPolicy().evaluate(job(Source.AGENT, hours=0.5), d, d.chosen)
    assert got.required
    assert got.rule == "quota_share"
    assert "kaggle's quota is used up" in (got.reason or "")
    # a little left keeps the old wording
    low = P("kaggle", quota=quota("kaggle", 29.9, 30, resets_at=SAT, source="live"))
    d2 = route(job(Source.AGENT, hours=0.5, provider="kaggle"), (low, "colab"))
    assert d2.chosen is not None
    got2 = RulesPolicy().evaluate(job(Source.AGENT, hours=0.5), d2, d2.chosen)
    assert got2.required
    assert "would use 500% of kaggle's remaining 6m quota" in (got2.reason or "")


def test_quota_numbers_say_when_they_are_estimates() -> None:
    """Finding: 'Xh left' in reasons and approval requests never said the number was an
    estimate (the ledger's rule is 'always labels which')."""
    est = P("kaggle", quota=quota("kaggle", 22, 30, resets_at=SAT, source="estimate"))
    d = route(job(hours=6), ("local", est, "colab"))
    kaggle = next(c for c in d.candidates if c.provider == "kaggle")
    assert kaggle.quota_source == "estimate"
    assert "uses 75% of the 8h left (est)" in kaggle.reason
    got = RulesPolicy().evaluate(job(Source.CLI, hours=6), d, kaggle)
    assert got.rule == "quota_share"
    assert "remaining 8h quota (est)" in (got.reason or "")
    live = P("kaggle", quota=quota("kaggle", 22, 30, resets_at=SAT, source="live"))
    d2 = route(job(hours=6), ("local", live, "colab"))
    k2 = next(c for c in d2.candidates if c.provider == "kaggle")
    assert k2.quota_source == "live"
    assert "(est)" not in k2.reason
    # "use it or lose it" says est too
    expiring = P(
        "kaggle", quota=quota("kaggle", 0, 30, resets_at=NOW + 10 * 3600, source="estimate")
    )
    d3 = route(job(hours=1), ("local", expiring, "colab"))
    k3 = next(c for c in d3.candidates if c.provider == "kaggle")
    assert "left (est), resets in 10h: use it or lose it" in k3.reason


def test_a_resumed_job_needs_only_what_is_left_of_it() -> None:
    """Finding: a 10h job resumed at 90% was treated as 10h again: 83% of a 12h quota, so
    it went back to awaiting approval for about an hour of work."""
    base = job(hours=10)
    resumed = base.model_copy(update={"progress": Progress(step=900, total=1000)})
    twelve = P("kaggle", quota=quota("kaggle", 18, 30, resets_at=SAT, source="live"))
    snaps = [p.snapshot() for p in (P("local"), twelve, P("colab"))]
    fresh = ScoringRouter().route(RoutingContext(job=base, now=NOW, providers=snaps))
    assert fresh.hours == 10
    ctx = RoutingContext(job=resumed, now=NOW, providers=snaps, resuming=True, resume_step=900)
    d = ScoringRouter().route(ctx)
    assert d.outcome is RouteOutcome.PLACE
    assert d.hours == pytest.approx(1.0)
    assert d.chosen is not None
    assert "resuming: ~1h left of 10h" in d.chosen.reason
    for c in d.candidates:
        if c.provider == "kaggle":
            assert c.quota_share == pytest.approx(1 / 12, abs=1e-3)
    agent = resumed.model_copy(update={"source": Source.AGENT, "state": JobState.MIGRATING})
    assert not RulesPolicy().evaluate(agent, d, d.chosen).required
    # no checkpoint step or no total: the whole job
    unknown = ScoringRouter().route(
        RoutingContext(job=resumed, now=NOW, providers=snaps, resuming=True)
    )
    assert unknown.hours == 10


def test_a_default_runtime_guess_counts_as_unknown_for_agents() -> None:
    """Finding: the bundle estimate falls back to 1h, equal to the agent limit, so 'unknown
    runtime asks' never fired for a real submission."""
    from gpu_router.router.base import JobEstimate

    guess = JobEstimate(hours=1.0, hours_source="heuristic", vram_gb=8, vram_source="heuristic")
    d = route(job(Source.AGENT), estimate=guess)
    assert d.chosen is not None
    assert (d.hours, d.hours_source) == (1.0, "heuristic")
    got = RulesPolicy().evaluate(job(Source.AGENT), d, d.chosen)
    assert got.required
    assert got.rule == "unknown_hours"
    assert (got.reason or "").startswith("runtime not given (guess 1h)")
    # the user's own jobs still run (unknown_hours: auto)
    assert not RulesPolicy().evaluate(job(Source.CLI), d, d.chosen).required
    # an explicit --hours is known
    d2 = route(job(Source.AGENT, hours=1.0), estimate=guess)
    assert d2.chosen is not None
    assert not RulesPolicy().evaluate(job(Source.AGENT, hours=1.0), d2, d2.chosen).required
    # a guess over the limit keeps the hours rule's wording
    long_guess = JobEstimate(hours=6, hours_source="heuristic")
    d3 = route(job(Source.AGENT), estimate=long_guess)
    assert d3.chosen is not None
    assert RulesPolicy().evaluate(job(Source.AGENT), d3, d3.chosen).rule == "hours"
