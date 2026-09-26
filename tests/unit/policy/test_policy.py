"""Approval policy rules (phase 5): thresholds per audience, rule order, editing."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import pytest

from gpu_router.config import Config
from gpu_router.errors import ConfigError, InvalidRequest
from gpu_router.models import Job, JobSpec, JobState, Source
from gpu_router.policy import (
    PolicyConfig,
    PolicyRules,
    RulesPolicy,
    apply_policy_setting,
    audience,
    default_policy,
    describe_rules,
    policy_config,
)
from gpu_router.router.base import Candidate, RouteDecision, RouteOutcome

NOW = 1_790_251_200.0


def job(
    source: Source = Source.AGENT,
    *,
    approved: bool = False,
    provider: str | None = None,
    **spec: Any,
) -> Job:
    s = JobSpec(project_dir="/p", script="train.py", source=source, **spec)
    return Job(
        id="c" * 12,
        short_id="cccc",
        name="train",
        state=JobState.ROUTING,
        source=s.source,
        project_dir="/p",
        spec=s,
        spec_hash=hashlib.sha256(s.model_dump_json().encode()).hexdigest(),
        provider=provider,
        approved_at=NOW if approved else None,
        created_at=NOW,
        updated_at=NOW,
    )


def placed(
    provider: str,
    hours: float | None,
    *,
    share: float | None = None,
    left: float | None = None,
    hours_source: str = "spec",
) -> tuple[RouteDecision, Candidate]:
    cand = Candidate(
        provider=provider,
        gpu="T4",
        vram_gb=16,
        reason=f"{provider}: fits 16GB",
        quota_share=share,
        quota_left=left,
    )
    decision = RouteDecision(
        outcome=RouteOutcome.PLACE,
        chosen=cand,
        candidates=[cand],
        reason=cand.reason,
        router="scoring",
        hours=hours,
        hours_source=hours_source if hours is not None else None,  # type: ignore[arg-type]
    )
    return decision, cand


@dataclass
class Case:
    id: str
    source: Source
    provider: str
    hours: float | None
    required: bool
    rule: str | None = None
    share: float | None = None
    approved: bool = False
    approved_for: str | None = None
    spec: dict[str, Any] | None = None


A, U = Source.AGENT, Source.CLI

CASES = [
    Case("agent 30m on colab runs", A, "colab", 0.5, False),
    Case("agent exactly 1h on kaggle runs", A, "kaggle", 1.0, False),
    Case("agent 1.5h on kaggle asks", A, "kaggle", 1.5, True, "hours"),
    Case("agent 20m on lightning runs", A, "lightning", 1 / 3, False),
    # phase 7b: modal was dropped (needs a card), so no provider asks by default
    Case("agent 6m on lightning runs: nothing asks by provider", A, "lightning", 0.1, False),
    Case("agent 10h on the Mac runs", A, "local", 10, False),
    Case("agent unknown runtime asks", A, "colab", None, True, "unknown_hours"),
    Case("agent share over half asks", A, "kaggle", 0.5, True, "quota_share", share=0.6),
    Case("agent share at half runs", A, "kaggle", 0.5, False, share=0.5),
    Case("agent approved job runs", A, "kaggle", 6, False, approved=True, approved_for="kaggle"),
    Case(
        "approval for kaggle does not cover half of colab's quota",
        A,
        "colab",
        6,
        True,
        "quota_share",
        share=0.9,
        approved=True,
        approved_for="kaggle",
    ),
    Case(
        "approval for this provider covers its quota share",
        A,
        "kaggle",
        6,
        False,
        share=0.9,
        approved=True,
        approved_for="kaggle",
    ),
    Case("user 6h on kaggle runs", U, "kaggle", 6, False),
    Case("user unknown runtime runs", U, "colab", None, False),
    Case("user 1h on lightning runs", U, "lightning", 1, False),
    Case("user share over half asks", U, "kaggle", 6, True, "quota_share", share=0.75),
    Case("shell jobs are user jobs", Source.SHELL, "kaggle", 6, False),
    Case("api jobs are user jobs", Source.API, "kaggle", 6, False),
    Case(
        "submitter asked for approval",
        U,
        "colab",
        0.2,
        True,
        "spec",
        spec={"requires_approval": True},
    ),
    Case("exempt beats the quota rule", A, "local", 1, False, share=0.99),
]


@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
def test_default_thresholds(case: Case) -> None:
    j = job(case.source, approved=case.approved, provider=case.approved_for, **(case.spec or {}))
    decision, cand = placed(case.provider, case.hours, share=case.share, left=10)
    got = RulesPolicy().evaluate(j, decision, cand)
    assert got.required is case.required, got
    assert got.rule == case.rule
    if got.required:
        assert got.reason
        assert case.provider in got.reason


def test_reasons_read_well() -> None:
    policy = RulesPolicy()
    d, c = placed("kaggle", 6, hours_source="heuristic")
    assert policy.evaluate(job(), d, c).reason == "over the 1h auto limit · kaggle T4 · ~6h"
    d, c = placed("kaggle", 0.5, share=0.6, left=0.8)
    assert policy.evaluate(job(), d, c).reason == (
        "would use 60% of kaggle's remaining 48m quota · kaggle T4 · 30m"
    )
    asks = RulesPolicy(PolicyConfig(agent=PolicyRules(ask_providers=("lightning",))))
    d, c = placed("lightning", 0.25)
    assert asks.evaluate(job(), d, c).reason == ("lightning always asks first · lightning T4 · 15m")


def test_spec_hours_used_when_the_router_gave_none() -> None:
    d, c = placed("kaggle", None)
    assert RulesPolicy().evaluate(job(hours=3), d, c).rule == "hours"


def test_audience() -> None:
    assert audience(Source.AGENT) == "agent"
    assert {audience(s) for s in (Source.CLI, Source.SHELL, Source.API)} == {"user"}


def test_rules_are_editable_data() -> None:
    policy = RulesPolicy()
    d, c = placed("kaggle", 3)
    assert policy.evaluate(job(), d, c).required
    policy.update(apply_policy_setting(policy.config, "agent.auto_max_hours", "4"))
    assert not policy.evaluate(job(), d, c).required
    policy.update(apply_policy_setting(policy.config, "ask_providers", "lightning,kaggle"))
    assert policy.evaluate(job(Source.CLI), d, c).rule == "provider"
    policy.update(apply_policy_setting(policy.config, "user.ask_providers", "none"))
    assert not policy.evaluate(job(Source.CLI), d, c).required


@pytest.mark.parametrize(
    ("key", "value", "agent", "user"),
    [
        ("agent.auto_max_hours", "2", {"auto_max_hours": 2.0}, {}),
        ("agent.auto_max_hours", "1.5h", {"auto_max_hours": 1.5}, {}),
        ("agent.auto_max_hours", "null", {"auto_max_hours": None}, {}),
        ("max_quota_share", "75%", {"max_quota_share": 0.75}, {"max_quota_share": 0.75}),
        ("user.max_quota_share", "off", {}, {"max_quota_share": None}),
        (
            "agent.ask-providers",
            "[kaggle, Lightning]",
            {"ask_providers": ("kaggle", "lightning")},
            {},
        ),
        ("user.exempt_providers", "none", {}, {"exempt_providers": ()}),
        ("agent.unknown_hours", "AUTO", {"unknown_hours": "auto"}, {}),
    ],
)
def test_apply_policy_setting(
    key: str, value: str, agent: dict[str, Any], user: dict[str, Any]
) -> None:
    base = PolicyConfig()
    new = apply_policy_setting(base, key, value)
    for field_name, want in agent.items():
        assert getattr(new.agent, field_name) == want
    for field_name, want in user.items():
        assert getattr(new.user, field_name) == want
    untouched = set(PolicyRules.model_fields) - set(agent)
    for field_name in untouched:
        assert getattr(new.agent, field_name) == getattr(base.agent, field_name)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("agent.nope", "1", "unknown policy rule"),
        ("robot.auto_max_hours", "1", "unknown policy rule"),
        ("agent.auto_max_hours", "lots", "cannot set"),
        ("agent.max_quota_share", "1.5", "cannot set"),
        ("agent.unknown_hours", "maybe", "cannot set"),
    ],
)
def test_apply_policy_setting_errors(key: str, value: str, message: str) -> None:
    with pytest.raises(InvalidRequest, match=message) as info:
        apply_policy_setting(PolicyConfig(), key, value)
    assert info.value.hint


def test_config_section_and_defaults() -> None:
    assert policy_config({}) == PolicyConfig()
    cfg = Config(policy={"version": 1, "agent": {"auto_max_hours": 3}})
    policy = default_policy(cfg)
    assert isinstance(policy, RulesPolicy)
    assert policy.config.agent.auto_max_hours == 3
    assert policy.config.user == PolicyConfig().user
    with pytest.raises(ConfigError, match=r"policy\.agent\.auto_max_hourz"):
        policy_config({"agent": {"auto_max_hourz": 3}})
    with pytest.raises(ConfigError, match=r"policy\.version"):
        policy_config({"version": 2})


def test_spec_defaults() -> None:
    d = PolicyConfig()
    assert (d.agent.auto_max_hours, d.agent.ask_providers, d.agent.max_quota_share) == (
        1.0,
        (),  # phase 7b: was ("modal",); modal was dropped because it needs a card
        0.5,
    )
    assert d.agent.unknown_hours == "ask"
    assert (d.user.auto_max_hours, d.user.unknown_hours) == (None, "auto")
    lines = dict(describe_rules(d.agent))
    assert lines["runs automatically"] == "up to 1h"
    assert "50%" in lines["quota guard"]
    assert dict(describe_rules(d.user))["runs automatically"] == "any length"
