"""D43 (phases 4-5 integration): after an approval the engine asks the policy again when it
re-routes, and only decisions marked `always` (the >50%-of-quota rule when the re-route
landed on another provider than the approval was for) make the job wait for a new answer.
awaiting_approval -> awaiting_approval is not a legal transition, so the re-ask updates the
approval fields in place with an approval_required note; the approval timeout restarts."""

from __future__ import annotations

import pytest

from gpu_router.clock import settle
from gpu_router.config import Config
from gpu_router.models import JobState, Reason
from gpu_router.policy import RulesPolicy
from gpu_router.router.base import Candidate, RouteDecision, RouteOutcome, RoutingContext
from tests.unit.engine.conftest import Engine, engine_config


class TargetRouter:
    """Places every job on `target` with the given quota share (hours / quota left)."""

    name = "target"

    def __init__(self, target: str, share: float) -> None:
        self.target = target
        self.share = share
        self.calls = 0

    def route(self, ctx: RoutingContext) -> RouteDecision:
        self.calls += 1
        cand = Candidate(
            provider=self.target,
            gpu="T4",
            vram_gb=16,
            reason=f"{self.target}: fits 16GB",
            quota_left=4.0,
            quota_unit="gpu_hours",
            quota_share=self.share,
        )
        return RouteDecision(
            outcome=RouteOutcome.PLACE,
            chosen=cand,
            candidates=[cand],
            reason=cand.reason,
            router="target",
            hours=self.share * 4.0,
            hours_source="spec",
        )


@pytest.fixture
def engine_cfg() -> Config:
    return engine_config(approval_timeout_s=600)


def _use(eng: Engine, router: TargetRouter) -> None:
    object.__setattr__(eng.supervisor.deps, "router", router)
    object.__setattr__(eng.supervisor.deps, "policy", RulesPolicy())


async def test_reroute_to_another_provider_over_half_its_quota_asks_again(eng: Engine) -> None:
    router = TargetRouter("fake", 0.75)
    _use(eng, router)
    job = await eng.submit(hours=3)
    waiting = await eng.until_state(job.id, JobState.AWAITING_APPROVAL)
    assert waiting.provider == "fake"
    assert "75% of fake's remaining 4h quota" in (waiting.approval_reason or "")

    router.target = "fake-b"  # quota moved on while the job waited
    await eng.supervisor.approve(job.id, actor="user:cli")
    await eng.run_until(
        lambda: any(
            e.reason == Reason.APPROVAL_REQUIRED and e.kind == "note"
            for e in eng.store.events_for(job.id, limit=10_000)
        )
    )
    again = eng.job(job.id)
    assert again.state is JobState.AWAITING_APPROVAL
    assert again.approved_at is None
    assert again.approved_by is None
    assert again.provider == "fake-b"
    assert "fake-b's remaining" in (again.approval_reason or "")
    assert "does not cover this placement" in again.message
    assert "gpu approve" in again.message
    note = [e for e in eng.store.events_for(job.id) if e.kind == "note"][-1]
    assert note.detail["reask"] is True
    assert note.detail["rule"] == "quota_share"
    assert eng.store.attempts_for(job.id) == []

    await eng.supervisor.approve(job.id, actor="user:cli")
    done = await eng.until_terminal(job.id)
    assert done.state is JobState.DONE
    attempts = eng.store.attempts_for(job.id)
    assert [a.provider for a in attempts] == ["fake-b"]
    # one transition into awaiting_approval: the re-ask was a note, not a transition
    assert [t[1] for t in eng.transitions(job.id)].count("awaiting_approval") == 1


async def test_reroute_to_the_same_provider_places_the_approved_job(eng: Engine) -> None:
    router = TargetRouter("fake", 0.75)
    _use(eng, router)
    job = await eng.submit(hours=3)
    await eng.until_state(job.id, JobState.AWAITING_APPROVAL)
    await eng.supervisor.approve(job.id, actor="user:cli")
    done = await eng.until_terminal(job.id)
    assert done.state is JobState.DONE
    assert [a.provider for a in eng.store.attempts_for(job.id)] == ["fake"]
    assert not [
        e for e in eng.store.events_for(job.id) if e.kind == "note" and e.detail.get("reask")
    ]


async def test_rules_that_do_not_always_ask_accept_the_approval_elsewhere(eng: Engine) -> None:
    # modal-style "provider asks" rule: approval covers the job wherever the re-route lands
    router = TargetRouter("fake", 0.1)
    _use(eng, router)
    object.__setattr__(
        eng.supervisor.deps,
        "policy",
        RulesPolicy(
            RulesPolicy().config.model_copy(
                update={
                    "user": RulesPolicy().config.user.model_copy(
                        update={"ask_providers": ("fake", "fake-b")}
                    )
                }
            )
        ),
    )
    job = await eng.submit(hours=0.4)
    waiting = await eng.until_state(job.id, JobState.AWAITING_APPROVAL)
    assert "always asks first" in (waiting.approval_reason or "")
    router.target = "fake-b"
    await eng.supervisor.approve(job.id, actor="user:cli")
    done = await eng.until_terminal(job.id)
    assert done.state is JobState.DONE
    assert [a.provider for a in eng.store.attempts_for(job.id)] == ["fake-b"]


async def test_approval_timeout_restarts_at_the_reask(eng: Engine) -> None:
    router = TargetRouter("fake", 0.75)
    _use(eng, router)
    job = await eng.submit(hours=3)
    await eng.until_state(job.id, JobState.AWAITING_APPROVAL)
    eng.clock.advance(500)  # most of the 600 s timeout used up by the first request
    await settle(30)
    router.target = "fake-b"
    await eng.supervisor.approve(job.id, actor="user:cli")
    await eng.run_until(lambda: eng.job(job.id).provider == "fake-b", max_s=5)
    eng.clock.advance(300)  # 800 s after the first request, 300 s after the re-ask
    await settle(30)
    assert eng.job(job.id).state is JobState.AWAITING_APPROVAL
    final = await eng.until_terminal(job.id, max_s=400)
    assert final.state is JobState.DENIED
    assert Reason.APPROVAL_EXPIRED in eng.reasons(job.id)
