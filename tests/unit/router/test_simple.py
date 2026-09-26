from __future__ import annotations

import hashlib
from typing import Any

from gpu_router.adapters.base import Capabilities
from gpu_router.models import Job, JobSpec, JobState, ProviderHealth, ProviderState, Source
from gpu_router.policy import RulesPolicy, SpecOnlyPolicy, default_policy
from gpu_router.providers.catalog import GpuOffer, ProviderEntry
from gpu_router.router.base import (
    ProviderSnapshot,
    RejectCode,
    RouteOutcome,
    RoutingContext,
)
from gpu_router.router.simple import SimpleRouter

NOW = 1_000_000.0


def entry(
    name: str, priority: int, *gpus: tuple[str, float, int], session: float = 12
) -> ProviderEntry:
    return ProviderEntry(
        name=name,
        kind="fake",
        display_name=name,
        priority=priority,
        gpus=tuple(GpuOffer(name=g, vram_gb=v, count=c) for g, v, c in gpus),
        session_hours=session,
        max_concurrency=2,
    )


def snap(
    e: ProviderEntry, *, live: int = 0, caps: dict[str, Any] | None = None, **state: Any
) -> ProviderSnapshot:
    return ProviderSnapshot(
        name=e.name,
        entry=e,
        capabilities=Capabilities(max_concurrency=2, **(caps or {})),
        state=ProviderState(provider=e.name, updated_at=NOW, **state),
        live_attempts=live,
    )


def job(**spec: Any) -> Job:
    s = JobSpec(project_dir="/p", script="t.py", source=Source.API, **spec)
    return Job(
        id="a" * 12,
        short_id="aaaa",
        name="t",
        state=JobState.ROUTING,
        source=s.source,
        project_dir="/p",
        spec=s,
        spec_hash=hashlib.sha256(s.model_dump_json().encode()).hexdigest(),
        created_at=NOW,
        updated_at=NOW,
    )


KAGGLE = entry("kaggle", 20, ("P100", 16, 1), ("T4", 16, 2))
COLAB = entry("colab", 10, ("T4", 16, 1))
MODAL = entry("modal", 40, ("T4", 16, 1), ("A100", 40, 1), ("H100", 80, 1), session=24)


def route(j: Job, *snaps: ProviderSnapshot, excluded: frozenset[str] = frozenset()) -> Any:
    return SimpleRouter().route(
        RoutingContext(job=j, now=NOW, providers=list(snaps), excluded=excluded)
    )


def test_first_fit_by_priority_with_smallest_gpu() -> None:
    d = route(job(), snap(KAGGLE), snap(COLAB), snap(MODAL))
    assert d.outcome is RouteOutcome.PLACE
    assert d.chosen.provider == "colab"
    assert [c.provider for c in d.candidates] == ["colab", "kaggle", "modal"]
    assert d.chosen.reason == "colab: first fit (T4 16GB)"
    assert d.chosen.score == -10
    assert d.reason == d.chosen.reason


def test_vram_picks_smallest_offer_that_fits() -> None:
    d = route(job(vram_gb=24), snap(COLAB), snap(MODAL))
    assert d.chosen.provider == "modal"
    assert d.chosen.gpu == "A100"
    assert d.rejected[0].code is RejectCode.VRAM


def test_no_fit_explains_largest_vram() -> None:
    d = route(job(vram_gb=100), snap(COLAB), snap(MODAL))
    assert d.outcome is RouteOutcome.NO_FIT
    assert d.reason == (
        "no provider fits: needs 100GB VRAM and no free provider has more than 80GB "
        "(largest: modal); `gpu providers` lists the excluded ones"
    )


def test_gpu_type_case_insensitive_and_labels() -> None:
    d = route(job(gpu="t4"), snap(KAGGLE))
    assert d.chosen.gpu == "2xT4"
    d2 = route(job(gpu="L4"), snap(KAGGLE))
    assert d2.outcome is RouteOutcome.NO_FIT
    assert d2.rejected[0].code is RejectCode.GPU_TYPE


def test_equal_vram_prefers_more_gpus() -> None:
    """D32 (phase-3 integration): kaggle's default placement is 2xT4, not the P100."""
    d = route(job(provider="kaggle"), snap(KAGGLE))
    assert d.chosen.gpu == "2xT4"
    d2 = route(job(gpu="P100"), snap(KAGGLE))
    assert d2.chosen.gpu == "P100"


def test_override_and_unknown_override() -> None:
    d = route(job(provider="kaggle"), snap(COLAB), snap(KAGGLE))
    assert d.chosen.provider == "kaggle"
    assert d.rejected[0].code is RejectCode.OVERRIDE
    d2 = route(job(provider="nope"), snap(COLAB))
    assert d2.outcome is RouteOutcome.NO_FIT
    assert "'nope' is not enabled" in d2.reason


def test_temporary_rejections_wait_with_earliest_until() -> None:
    d = route(
        job(),
        snap(COLAB, cooldown_until=NOW + 240),
        snap(KAGGLE, exhausted_until=NOW + 60),
        snap(MODAL, live=2),
    )
    assert d.outcome is RouteOutcome.WAIT
    assert d.retry_at == NOW + 60
    assert d.reason.startswith("all providers busy: colab: cooldown 4m")
    assert {r.code for r in d.rejected} == {
        RejectCode.COOLDOWN,
        RejectCode.EXHAUSTED,
        RejectCode.CAPACITY,
    }


def test_auth_is_temporary_without_until() -> None:
    d = route(job(), snap(COLAB, health=ProviderHealth.AUTH_REQUIRED))
    assert d.outcome is RouteOutcome.WAIT
    assert d.retry_at is None
    assert "gpu login colab" in d.reason


def test_disabled_excluded_interactive_session_are_permanent() -> None:
    d = route(
        job(interactive=True, hours=30),
        snap(COLAB, health=ProviderHealth.DISABLED),
        snap(KAGGLE),
        snap(MODAL, caps={"interactive": True, "resume": False}),
        excluded=frozenset({"kaggle"}),
    )
    assert d.outcome is RouteOutcome.NO_FIT
    assert [r.code for r in d.rejected] == [
        RejectCode.DISABLED,
        RejectCode.EXCLUDED,
        RejectCode.SESSION,
    ]


def test_unhealthy_uses_cooldown_as_until() -> None:
    d = route(
        job(),
        snap(
            COLAB, health=ProviderHealth.UNAVAILABLE, health_reason="down", cooldown_until=NOW + 10
        ),
    )
    assert d.outcome is RouteOutcome.WAIT
    assert d.retry_at == NOW + 10
    assert "unavailable (down)" in d.reason


def test_no_providers() -> None:
    d = route(job())
    assert d.outcome is RouteOutcome.NO_FIT
    assert "no providers are enabled" in d.reason


def test_decision_detail_is_json_safe() -> None:
    import json

    d = route(job(), snap(COLAB))
    json.dumps(d.detail())


def test_spec_only_policy() -> None:
    d = route(job(requires_approval=True), snap(COLAB))
    assert isinstance(default_policy(), RulesPolicy)  # phase 5 default; spec rule kept
    decision = SpecOnlyPolicy().evaluate(job(requires_approval=True), d, d.chosen)
    assert decision.required
    assert decision.rule == "spec"
    assert "colab T4" in (decision.reason or "")
    approved = job(requires_approval=True).model_copy(update={"approved_at": NOW})
    assert not SpecOnlyPolicy().evaluate(approved, d, d.chosen).required
    assert not SpecOnlyPolicy().evaluate(job(), d, d.chosen).required
