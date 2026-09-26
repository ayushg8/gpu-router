"""Approval policy (phase 1 protocol; phase 5 rules).

The engine asks the policy after the router picked a placement and before `store.place`.
A policy is pure like a router: inputs in, `ApprovalDecision` out.

Phase 1 (`SpecOnlyPolicy`): approval is required only when the submitter asked for it
(`JobSpec.requires_approval`) and the job has not been approved yet.

Phase 5 (`RulesPolicy`, the daemon default; editable with `gpu policy` / the shell's
/policy, stored in config.yaml under `policy:`) implements the spec defaults:

  - under 1 GPU-hr on kaggle/colab/lightning       -> runs automatically
  - over 1 hr                                       -> asks first
    (the spec's "anything on Modal asks" is moot: Modal was dropped in phase 7b because
    it needs a card, so `ask_providers` is empty by default)
  - would use > 50% of a provider's remaining quota -> always asks

Rules are data (`PolicyConfig`), per audience:

  policy:
    version: 1
    agent:                        # jobs an agent submitted (source: agent; MCP, phase 6)
      auto_max_hours: 1           # at or under: runs automatically; null = no hours limit
      ask_providers: []           # always ask before running here, e.g. [lightning]
      exempt_providers: [local]   # never ask here (free, this Mac)
      max_quota_share: 0.5        # asks when the job would use more of the quota left;
                                  #   null = off
      unknown_hours: ask          # neither --hours nor an estimate: ask | auto
      ask_secrets: true           # asks when the job reads Keychain secrets (D48)
      enforce_hours: true         # a job that runs well past its declared hours is
                                  #   stopped for approval (D48; see hours_limit_s)
    user:                         # jobs you submitted (cli, shell, api)
      auto_max_hours: null        # you typed `gpu run`: that is the approval for its length
      ask_providers: []
      exempt_providers: [local]
      max_quota_share: 0.5
      unknown_hours: auto
      ask_secrets: false
      enforce_hours: false

Why user and agent differ: the spec's approval list answers "agent job requests" (/approve)
and `models.Source` says agents are subject to the policy. A human who typed `gpu run
train.py --hours 6` has already said yes to a 6-hour job, so asking again only adds a step.
The rules the spec calls out as protecting quota apply to both audiences by default: the
>50%-of-what-is-left rule, and `ask_providers` when set (the spec's credit-metered modal
was dropped in phase 7b).

Evaluation order (first match wins):
  1. spec           the submitter asked for approval (`requires_approval`) and none given yet
  2. exempt         candidate in exempt_providers -> auto
  3. quota_share    hours / quota left > max_quota_share -> ask ("always": even after an
                    approval, unless that approval was for this same provider)
  4. approved       the job was approved already -> auto (D10: approval covers the job)
  5. provider       candidate in ask_providers -> ask
  6. secrets        the job lists `secrets:` and ask_secrets is on -> ask (the names are
                    in the reason, so the user sees what the job would read; D48)
  7. unknown_hours  no --hours and no hours known, or only a bundle heuristic under the
                    auto limit (a default guess is not a known runtime) -> per
                    `unknown_hours`
  8. hours          hours > auto_max_hours -> ask
  9. auto

Declared hours are enforced for the agent audience (`enforce_hours`, D48): the hours an
agent gives are what auto-approves its job, so a job that keeps running past
max(hours x 1.5, hours + 15 min) of running time is asked to checkpoint and moved to
awaiting_approval ("ran past its declared 30m"); approving it lets it finish. The engine
asks `hours_limit_s(policy, job, provider)`; exempt providers (local) are never stopped.

Hours are the router's (`RouteDecision.hours`: spec, else bundle estimate) and are wall
hours on one GPU session, which is how the free tiers meter (a 2xT4 Kaggle hour is one
quota hour). The quota share comes from the router's candidate (`Candidate.quota_share`,
ledger view, GPU-hour quotas only); unknown quota (Colab) or unlimited (local) never trips
it. After an approval the engine asks again when it re-routes (D43) but honours only
decisions marked `always` (rule 3): if the re-route lands on another provider than the one
the approval was for, the job waits for a new answer; every other rule treats an approved
job as approved (D10).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from gpu_router.models import Job, Source
from gpu_router.router.base import Candidate, RouteDecision

if TYPE_CHECKING:
    from gpu_router.config import Config

__all__ = [
    "AUTO",
    "POLICY_VERSION",
    "ApprovalDecision",
    "ApprovalPolicy",
    "PolicyConfig",
    "PolicyRules",
    "RulesPolicy",
    "SpecOnlyPolicy",
    "apply_policy_setting",
    "audience",
    "default_policy",
    "describe_rules",
    "hours_limit_s",
    "overrun_limit_s",
    "policy_config",
]

POLICY_VERSION: Literal[1] = 1


class ApprovalDecision(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    required: bool
    reason: str | None = None  # one line, shown in the shell and status line
    rule: str | None = None  # machine name of the rule that fired ("spec", "provider", ...)
    # asks even for a job that was already approved (the engine re-asks the policy after an
    # approval and honours only these): the >50%-of-quota rule when the re-route landed on
    # another provider than the one the approval was for
    always: bool = False


AUTO = ApprovalDecision(required=False)


class ApprovalPolicy(Protocol):
    name: str

    def evaluate(self, job: Job, decision: RouteDecision, candidate: Candidate) -> ApprovalDecision:
        """Whether placing `job` on `candidate` needs a human yes/no."""
        ...


class SpecOnlyPolicy:
    """Phase 1: ask only when the job spec requests approval and none was given yet."""

    name = "spec_only"

    def evaluate(self, job: Job, decision: RouteDecision, candidate: Candidate) -> ApprovalDecision:
        if job.spec.requires_approval and job.approved_at is None:
            gpu = f" {candidate.gpu}" if candidate.gpu else ""
            return ApprovalDecision(
                required=True,
                reason=f"submitter asked for approval · {candidate.provider}{gpu}",
                rule="spec",
            )
        return AUTO


# --------------------------------------------------------------------------- phase 5 rules


class PolicyRules(BaseModel):
    """Approval rules for one audience (agent or user jobs)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    auto_max_hours: float | None = Field(default=1.0, ge=0)
    ask_providers: tuple[str, ...] = ()  # phase 7b: was ("modal",); modal was dropped
    exempt_providers: tuple[str, ...] = ("local",)
    max_quota_share: float | None = Field(default=0.5, gt=0, le=1)
    unknown_hours: Literal["ask", "auto"] = "ask"
    # D48 (additive; a config.yaml written before them gets the audience's default)
    ask_secrets: bool = False
    enforce_hours: bool = False


#: Rules added after phase 5 shipped, with the default each audience gets when config.yaml
#: (or a PUT /v1/policy body from an older client) does not mention them (D48).
_AUDIENCE_DEFAULTS: dict[str, dict[str, object]] = {
    "agent": {"ask_secrets": True, "enforce_hours": True},
    "user": {"ask_secrets": False, "enforce_hours": False},
}


def _agent_defaults() -> PolicyRules:
    return PolicyRules(**_AUDIENCE_DEFAULTS["agent"])


def _user_defaults() -> PolicyRules:
    return PolicyRules(
        auto_max_hours=None,
        unknown_hours="auto",
        **_AUDIENCE_DEFAULTS["user"],
    )


class PolicyConfig(BaseModel):
    """The `policy:` section of config.yaml (versioned on its own)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal[1] = POLICY_VERSION
    agent: PolicyRules = Field(default_factory=_agent_defaults)
    user: PolicyRules = Field(default_factory=_user_defaults)

    @model_validator(mode="before")
    @classmethod
    def _fill_new_rules(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        out = dict(data)
        for who, defaults in _AUDIENCE_DEFAULTS.items():
            rules = out.get(who)
            if isinstance(rules, dict):
                out[who] = {**defaults, **rules}
        return out

    def rules_for(self, source: Source) -> PolicyRules:
        return self.agent if audience(source) == "agent" else self.user


def audience(source: Source) -> Literal["agent", "user"]:
    """Which rule set a job's source uses: agents -> agent; cli, shell, api -> user."""
    return "agent" if source is Source.AGENT else "user"


def policy_config(raw: Mapping[str, Any] | None) -> PolicyConfig:
    """Validate config.policy (an empty mapping = defaults). Raises ConfigError."""
    try:
        return PolicyConfig.model_validate(dict(raw or {}))
    except ValidationError as exc:
        from gpu_router.errors import ConfigError

        problems = "; ".join(
            f"policy.{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()[:5]
        )
        raise ConfigError(
            f"config.yaml is invalid: {problems}",
            hint="fix the listed keys under `policy:` (or `gpu policy reset`)",
        ) from None


def _h(hours: float) -> str:
    if hours < 1:
        return f"{max(1, round(hours * 60))}m"
    r = round(hours, 1)
    return f"{int(r)}h" if r == int(r) else f"{r:.1f}h"


#: A job may run this much past its declared hours before it is stopped for approval
#: (D48): max(hours x OVERRUN_FACTOR, hours + OVERRUN_MIN_S).
OVERRUN_FACTOR = 1.5
OVERRUN_MIN_S = 15 * 60.0


def overrun_limit_s(hours: float) -> float:
    """Seconds of running a job that declared `hours` gets before it is stopped."""
    return max(hours * 3600 * OVERRUN_FACTOR, hours * 3600 + OVERRUN_MIN_S)


def hours_limit_s(policy: object, job: Job, provider: str | None) -> float | None:
    """Running seconds after which the engine stops `job` for approval (D48), or None when
    its declared hours are not enforced (policies without the rule, user jobs by default,
    no declared hours, interactive jobs, exempt providers)."""
    fn = getattr(policy, "hours_limit_s", None)
    if fn is None:
        return None
    limit = fn(job, provider)
    return float(limit) if isinstance(limit, int | float) else None


def _secret_list(names: list[str], limit: int = 4) -> str:
    shown = ", ".join(names[:limit])
    return shown + (f" +{len(names) - limit} more" if len(names) > limit else "")


class RulesPolicy:
    """Phase 5: the spec's approval defaults, as editable data (see module docstring)."""

    name = "rules"

    def __init__(self, config: PolicyConfig | None = None) -> None:
        self._config = config or PolicyConfig()

    @property
    def config(self) -> PolicyConfig:
        return self._config

    def update(self, config: PolicyConfig) -> None:
        """Swap the rules (event-loop thread; takes effect at the next evaluation)."""
        self._config = config

    def hours_limit_s(self, job: Job, provider: str | None) -> float | None:
        """See the module docstring (enforce_hours, D48)."""
        rules = self._config.rules_for(job.source)
        hours = job.spec.hours
        if not rules.enforce_hours or hours is None or job.spec.interactive:
            return None
        if provider is not None and provider in rules.exempt_providers:
            return None
        return overrun_limit_s(hours)

    def evaluate(self, job: Job, decision: RouteDecision, candidate: Candidate) -> ApprovalDecision:
        rules = self._config.rules_for(job.source)
        provider = candidate.provider
        gpu = f" {candidate.gpu}" if candidate.gpu else ""
        where = f"{provider}{gpu}"
        hours = decision.hours
        if hours is None:
            hours = job.spec.hours
        length = ""
        if hours is not None:
            est = "~" if decision.hours_source == "heuristic" else ""
            length = f" · {est}{_h(hours)}"

        if job.spec.requires_approval and job.approved_at is None:
            return ApprovalDecision(
                required=True, reason=f"submitter asked for approval · {where}", rule="spec"
            )
        if provider in rules.exempt_providers:
            return AUTO
        share = candidate.quota_share
        if (
            rules.max_quota_share is not None
            and share is not None
            and share > rules.max_quota_share
            and not (job.approved_at is not None and job.provider == provider)
        ):
            left = candidate.quota_left
            est = " (est)" if candidate.quota_source == "estimate" else ""
            if left is not None and left <= 0:
                what = f"{provider}'s quota is used up{est}; it may stop the job early"
            else:
                left_txt = ""
                if left is not None:
                    unit = candidate.quota_unit
                    left_txt = (
                        f" {_h(left)}" if unit in (None, "gpu_hours") else f" {left:g} {unit}"
                    )
                what = f"would use {min(share, 9.99):.0%} of {provider}'s remaining{left_txt}"
                what += f" quota{est}"
            return ApprovalDecision(
                required=True,
                reason=f"{what} · {where}{length}",
                rule="quota_share",
                always=True,
            )
        if job.approved_at is not None:
            return AUTO
        if provider in rules.ask_providers:
            return ApprovalDecision(
                required=True,
                reason=f"{provider} always asks first · {where}{length}",
                rule="provider",
            )
        if rules.ask_secrets and job.spec.secrets:
            names = _secret_list(list(job.spec.secrets))
            return ApprovalDecision(
                required=True,
                reason=f"reads Keychain secrets {names} · {where}{length}",
                rule="secrets",
            )
        over = hours is not None and (
            rules.auto_max_hours is not None and hours > rules.auto_max_hours
        )
        # a bundle heuristic is a guess, not a runtime the submitter gave: a guess under the
        # auto limit is "unknown" (the estimate falls back to a default 1h when it finds
        # nothing, which would otherwise always pass a 1h limit; D44)
        guessed = hours is not None and decision.hours_source == "heuristic"
        if job.spec.hours is None and (hours is None or (guessed and not over)):
            if rules.unknown_hours == "ask":
                why = "runtime unknown"
                if hours is not None:
                    why = f"runtime not given (guess {_h(hours)})"
                return ApprovalDecision(
                    required=True,
                    reason=f"{why}; set --hours · {where}",
                    rule="unknown_hours",
                )
            if hours is None:
                return AUTO
        if hours is None:
            return AUTO
        if over:
            assert rules.auto_max_hours is not None
            return ApprovalDecision(
                required=True,
                reason=f"over the {_h(rules.auto_max_hours)} auto limit · {where}{length}",
                rule="hours",
            )
        return AUTO


_NULLS = frozenset({"null", "none", "off", "no-limit", "unlimited", "~", ""})
_LIST_FIELDS = frozenset({"ask_providers", "exempt_providers"})


def _parse_value(field: str, text: str) -> object:
    raw = text.strip()
    if field in _LIST_FIELDS:
        body = raw.strip("[]")
        if body.strip().lower() in _NULLS:
            return []
        return [x.strip().lower() for x in body.split(",") if x.strip()]
    if raw.lower() in _NULLS and field in ("auto_max_hours", "max_quota_share"):
        return None
    if field in ("unknown_hours", "ask_secrets", "enforce_hours"):
        return raw.lower()  # pydantic reads true/false/yes/no/on/off for the booleans
    if field == "max_quota_share" and raw.endswith("%"):
        try:
            return float(raw[:-1]) / 100
        except ValueError:
            return raw
    if field == "auto_max_hours" and raw.lower().endswith("h"):
        raw = raw[:-1]
    try:
        return float(raw)
    except ValueError:
        return raw


def apply_policy_setting(config: PolicyConfig, key: str, value: str) -> PolicyConfig:
    """`gpu policy set KEY VALUE`: KEY is `agent.<rule>`, `user.<rule>` or a bare `<rule>`
    (both audiences). Values: numbers (`2`, `1.5h`, `50%`), `null`/`off` (no limit),
    comma lists for provider lists (`kaggle,lightning`; `none` = empty), `ask`/`auto`.
    Raises InvalidRequest naming the key and what would be valid."""
    from gpu_router.errors import InvalidRequest

    fields = list(PolicyRules.model_fields)
    parts = key.strip().lower().replace("-", "_").split(".")
    if len(parts) == 2 and parts[0] in ("agent", "user"):
        audiences, field = [parts[0]], parts[1]
    elif len(parts) == 1:
        audiences, field = ["agent", "user"], parts[0]
    else:
        audiences, field = [], ""
    if field not in fields:
        raise InvalidRequest(
            f"unknown policy rule {key!r}",
            hint="rules: " + ", ".join(fields) + " (prefix agent. or user. for one audience)",
            detail={"key": key},
        )
    doc = config.model_dump(mode="json")
    parsed = _parse_value(field, value)
    for who in audiences:
        doc[who][field] = parsed
    try:
        return PolicyConfig.model_validate(doc)
    except ValidationError as exc:
        msg = exc.errors()[0]["msg"]
        raise InvalidRequest(
            f"cannot set {key} to {value!r}: {msg}",
            hint=_VALUE_HINTS.get(field, ""),
            detail={"key": key, "value": value},
        ) from None


_VALUE_HINTS = {
    "auto_max_hours": "hours as a number (e.g. 2), or null for no limit",
    "max_quota_share": "a share from 0 to 1 or a percentage (e.g. 0.5 or 50%), or off",
    "unknown_hours": "ask or auto",
    "ask_secrets": "true or false",
    "enforce_hours": "true or false",
    "ask_providers": "comma-separated provider names, e.g. lightning (none = empty)",
    "exempt_providers": "comma-separated provider names, e.g. local (none = empty)",
}


def describe_rules(rules: PolicyRules) -> list[tuple[str, str]]:
    """Human lines for one audience: [(label, text)] (CLI and shell /policy)."""
    ask = ", ".join(rules.ask_providers) or "none"
    exempt = ", ".join(rules.exempt_providers) or "none"
    limit = rules.auto_max_hours
    auto = "any length" if limit is None else f"up to {_h(limit)}"
    except_ = f" (except {ask})" if rules.ask_providers else ""
    share = (
        "off"
        if rules.max_quota_share is None
        else f"asks when a job would use over {rules.max_quota_share:.0%} of a provider's "
        "quota left"
    )
    return [
        ("runs automatically", f"{auto}{except_}"),
        ("always asks on", ask),
        ("never asks on", exempt),
        ("quota guard", share),
        ("unknown runtime", "asks" if rules.unknown_hours == "ask" else "runs automatically"),
        ("keychain secrets", "asks" if rules.ask_secrets else "runs automatically"),
        (
            "declared hours",
            "enforced: stopped for approval well past them"
            if rules.enforce_hours
            else "not enforced",
        ),
    ]


def default_policy(config: Config | None = None) -> ApprovalPolicy:
    """The policy the daemon uses: phase 5 rules from config.yaml (defaults when absent)."""
    return RulesPolicy(policy_config(config.policy if config is not None else None))
