"""Phase-5 scoring router (spec "Router": filter, score, pick and explain, fallback).

Pure like every router (router/base.py): it reads the RoutingContext only. Quota numbers are
the ledger's view (`ProviderSnapshot.quota`, see quota/ledger.py), never a provider call.

Job facts
  hours     spec.hours, else the bundle estimate (`ctx.estimate.hours`, heuristic), else None
  VRAM      spec.vram_gb is a hard filter; a heuristic estimate only steers the score
  smoke     `spec.smoke` (gpu.yaml `smoke: true` / `--smoke`), or an explicit spec.hours at or
            under `smoke_max_minutes` (default 5) with no GPU type, no VRAM ask and not
            interactive
  handoff   the job checkpoints (checkpoint_interval_min > 0), is not interactive, and the
            provider can resume: such a job may exceed a session cap or a quota remainder
            and continue elsewhere from its latest checkpoint

1. Filter (one Rejection per dropped provider, first match wins): excluded, disabled,
   needs login, unavailable, cooldown, exhausted (provider refused for quota), GPU type,
   VRAM (explicit), interactive, SESSION (hours over the session cap and no handoff),
   QUOTA (ledger: nothing left, or less than the job needs: all of its hours without
   handoff, else min(hours, handoff_min_hours)), capacity. QUOTA is temporary (retry at the
   reset) like cooldowns. Then RESERVED: the local Mac (catalog kind `local`) is kept for
   smoke tests, and `big_vram_providers` (none by default since modal was dropped, 7b) for
   jobs needing more than their threshold
   (explicit VRAM, else the estimate). Reserved providers are a last resort: candidates
   only when nothing else survives and nothing else is merely temporarily blocked.
2. Score (higher is better; weights below):
   - smoke test on the local Mac                                      +100
   - `save_for_long_jobs` (kaggle): job over long_job_hours +30, else -30 "saved for jobs
     over 4h"; but +10 instead when more quota is left than hours until its reset (it
     would expire unused: "use it or lose it", spec "spend quota that resets soonest")
   - `short_job_providers` (colab): short (<= long_job_hours, unknown counts as short) or
     interactive +20; long -10 (sessions are not guaranteed)
   - `big_vram_providers` for a job needing more than the threshold             +25
   - heuristic VRAM estimate above every offer of the provider                   -40
   - quota that resets soonest: up to +15, linear over the 7 days before the reset
   - would use over half of the quota left                                       -10
   - tie-break: catalog priority (lower first), then name
3. Pick + explain: candidates ranked best first, each with a one-line reason, e.g.
   "colab: fits 16GB, kaggle saved for jobs over 4h". The chosen one names the runner-up's
   decisive handicap, or a better-suited provider that was ruled out ("kaggle quota used
   up, resets in 2d").
4. Fallback: the ranked candidates are recorded in the placement event; the engine re-routes
   after a failed placement (cooldown / exclusion), which moves to the next candidate.

A job pinned to a provider (`--provider`, spec.provider) is routed by `SimpleRouter`
(first fit on that provider; an unknown or disabled pin is NO_FIT, never a reroute), with
the ledger's quota facts added to its candidates so the approval policy still sees them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from gpu_router.models import ProviderHealth
from gpu_router.providers.catalog import GpuOffer
from gpu_router.quota.ledger import fmt_hours, remaining, to_gpu_hours
from gpu_router.router.base import (
    TEMPORARY_REJECTIONS,
    Candidate,
    ProviderSnapshot,
    RejectCode,
    Rejection,
    RouteDecision,
    RouteOutcome,
    RoutingContext,
    earliest_retry,
)
from gpu_router.router.settings import RoutingSettings
from gpu_router.router.simple import SimpleRouter, _dur, _fitting_offers, _gb, _no_fit_reason

__all__ = ["SCORING_TEMPORARY", "JobFacts", "ScoringRouter", "job_facts"]

#: QUOTA clears at the provider's reset, so it waits like a cooldown.
SCORING_TEMPORARY: frozenset[RejectCode] = TEMPORARY_REJECTIONS | {RejectCode.QUOTA}

W_SMOKE_LOCAL = 100.0
W_LONG_FIT = 30.0
W_SAVED = -30.0
W_EXPIRING = 10.0
W_SHORT = 20.0
W_SHORT_LONG = -10.0
W_BIG_VRAM = 25.0
W_EST_VRAM_MISS = -40.0
W_RESET_MAX = 15.0
W_SHARE = -10.0
RESET_HORIZON_S = 7 * 86_400.0
LAST_RESORT_RANK = 1_000.0
SHARE_WARN = 0.5
#: quota_share when a known remainder is zero: above every threshold (JSON has no inf)
SHARE_NOTHING_LEFT = 9.99
#: a resumed job is never assumed to be more than this far along (a bad total step count
#: must not make a long job look free)
RESUME_MIN_LEFT = 0.05


@dataclass(frozen=True, slots=True)
class JobFacts:
    hours: float | None  # what is left to run (a resumed job: its remaining share, D44)
    hours_source: str | None  # "spec" | "heuristic"
    vram_need: float | None  # explicit (hard)
    vram_est: float | None  # heuristic (soft)
    smoke: bool
    interactive: bool
    checkpoints: bool
    hours_total: float | None = None  # the whole job's hours when `hours` is a remainder

    @property
    def vram_wanted(self) -> float | None:
        return self.vram_need if self.vram_need is not None else self.vram_est

    def is_long(self, settings: RoutingSettings) -> bool:
        return self.hours is not None and self.hours > settings.long_job_hours

    def can_handoff(self, p: ProviderSnapshot) -> bool:
        return self.checkpoints and not self.interactive and p.capabilities.resume


def job_facts(ctx: RoutingContext, settings: RoutingSettings) -> JobFacts:
    spec = ctx.job.spec
    est = ctx.estimate
    hours, hours_source = spec.hours, "spec" if spec.hours is not None else None
    if hours is None and est is not None and est.hours is not None:
        hours, hours_source = est.hours, est.hours_source or "heuristic"
    vram_est = None
    if spec.vram_gb is None and est is not None and est.vram_gb is not None:
        vram_est = est.vram_gb
    hours_total: float | None = None
    total = ctx.job.progress.total
    if hours is not None and ctx.resuming and ctx.resume_step is not None and total:
        # a resumed job needs only what is left of it (quota need, share, approval): the
        # checkpoint's step against the job's total steps
        left = max(RESUME_MIN_LEFT, min(1.0, 1.0 - ctx.resume_step / total))
        if left < 1.0:
            hours_total, hours = hours, hours * left
    implicit_smoke = (
        spec.hours is not None
        and spec.hours * 60 <= settings.smoke_max_minutes
        and spec.gpu is None
        and spec.vram_gb is None
        and not spec.interactive
    )
    return JobFacts(
        hours=hours,
        hours_source=hours_source,
        vram_need=spec.vram_gb,
        vram_est=vram_est,
        smoke=bool(getattr(spec, "smoke", False)) or implicit_smoke,
        interactive=spec.interactive,
        checkpoints=spec.checkpoint_interval_min > 0,
        hours_total=hours_total,
    )


@dataclass
class _Scored:
    p: ProviderSnapshot
    offer: GpuOffer
    score: float = 0.0
    fit: str = ""
    good: list[tuple[float, str]] = field(default_factory=list)  # (weight, fragment)
    bad: list[tuple[float, str]] = field(default_factory=list)
    quota_left: float | None = None
    quota_share: float | None = None
    last_resort: bool = False

    def add(self, weight: float, fragment: str | None) -> None:
        self.score += weight
        if fragment:
            (self.good if weight > 0 else self.bad).append((weight, fragment))

    def top_bad(self) -> tuple[float, str] | None:
        return min(self.bad) if self.bad else None

    def top_good(self) -> tuple[float, str] | None:
        return max(self.good) if self.good else None


def _in(names: tuple[str, ...], p: ProviderSnapshot) -> bool:
    return p.name in names


def _is_local(p: ProviderSnapshot) -> bool:
    return p.entry.kind == "local"


def _span(seconds: float) -> str:
    """_dur, but whole days from 48 h on: '4m', '1h30m', '2d'."""
    return f"{round(seconds / 86_400)}d" if seconds >= 48 * 3600 else _dur(seconds)


def _reset_label(ts: float, now: float) -> str:
    left = ts - now
    if left < 86_400:
        return f"resets in {_dur(left)}"
    return "resets " + datetime.fromtimestamp(ts, tz=UTC).strftime("%a %H:%M UTC")


def _resume_fragment(f: JobFacts) -> str | None:
    """'~1h left of 10h' for a resumed job whose hours were scaled to what is left."""
    if f.hours_total is None or f.hours is None:
        return None
    return f"resuming: ~{fmt_hours(f.hours)} left of {fmt_hours(f.hours_total)}"


class ScoringRouter:
    name = "scoring"

    def __init__(self, settings: RoutingSettings | None = None) -> None:
        self.settings = settings or RoutingSettings()
        self._simple = SimpleRouter()

    # ------------------------------------------------------------------ entry

    def route(self, ctx: RoutingContext) -> RouteDecision:
        facts = job_facts(ctx, self.settings)
        if ctx.job.spec.provider is not None:
            return self._pinned(ctx, facts)
        rejected: list[Rejection] = []
        survivors: list[ProviderSnapshot] = []
        reserved: list[tuple[ProviderSnapshot, Rejection]] = []
        # Reserved providers are a last resort. The Mac only when it is the sole provider
        # registered (a cloud job that fits nowhere should fail loudly, not silently run
        # for hours on MPS); a big-VRAM provider whenever nothing else can take the job.
        only_local = all(_is_local(p) for p in ctx.providers)
        for p in ctx.providers:
            r = self._reject(p, ctx, facts)
            res = self._reserved(p, facts)
            if res is not None and _is_local(p) and not only_local:
                # never a fallback: a busy Mac must not make a cloud job wait for it
                keep = r if r is not None and r.code not in SCORING_TEMPORARY else res
                rejected.append(keep)
            elif r is not None:
                rejected.append(r)
            elif res is not None:
                reserved.append((p, res))
            else:
                survivors.append(p)

        temporary = [r for r in rejected if r.code in SCORING_TEMPORARY]
        last_resort = False
        if not survivors and not temporary and reserved:
            survivors = [p for p, _ in reserved]
            last_resort = True
        else:
            rejected.extend(r for _, r in reserved)

        scored = [self._score(p, ctx, facts, last_resort=last_resort) for p in survivors]
        scored.sort(key=lambda s: (-s.score, s.p.entry.priority, s.p.name))
        if scored:
            candidates = [
                self._candidate(s, i, scored, rejected, ctx, facts) for i, s in enumerate(scored)
            ]
            chosen = candidates[0]
            return self._decision(
                facts,
                outcome=RouteOutcome.PLACE,
                chosen=chosen,
                candidates=candidates,
                rejected=rejected,
                reason=chosen.reason,
            )
        if temporary:
            parts = [r.reason for r in temporary[:4]]
            more = f" (+{len(temporary) - 4} more)" if len(temporary) > 4 else ""
            return self._decision(
                facts,
                outcome=RouteOutcome.WAIT,
                rejected=rejected,
                reason="all providers busy: " + ", ".join(parts) + more,
                retry_at=earliest_retry(temporary),
            )
        # name the real obstacles first; "kept for smoke tests" only explains the Mac
        ordered = sorted(rejected, key=lambda r: r.code is RejectCode.RESERVED)
        return self._decision(
            facts,
            outcome=RouteOutcome.NO_FIT,
            rejected=rejected,
            reason=_no_fit_reason(ctx, ordered),
        )

    def _decision(self, facts: JobFacts, **kw: object) -> RouteDecision:
        return RouteDecision.model_validate(
            {
                **kw,
                "router": self.name,
                "hours": facts.hours,
                "hours_source": facts.hours_source,
                "smoke": facts.smoke,
            }
        )

    # ------------------------------------------------------------------ filter

    def _reject(self, p: ProviderSnapshot, ctx: RoutingContext, f: JobFacts) -> Rejection | None:
        spec = ctx.job.spec
        now = ctx.now
        name = p.name
        st = p.state
        if name in ctx.excluded:
            return Rejection(
                provider=name, code=RejectCode.EXCLUDED, reason=f"{name}: rejected this job earlier"
            )
        if name in ctx.data_unreachable:
            return Rejection(
                provider=name,
                code=RejectCode.EXCLUDED,
                reason=f"{name}: {ctx.data_unreachable[name]}",
            )
        if st.health is ProviderHealth.DISABLED:
            return Rejection(provider=name, code=RejectCode.DISABLED, reason=f"{name}: disabled")
        if st.health is ProviderHealth.AUTH_REQUIRED:
            return Rejection(
                provider=name,
                code=RejectCode.AUTH,
                reason=f"{name}: needs login (run `gpu login {name}`)",
            )
        if st.health is ProviderHealth.UNAVAILABLE:
            why = f" ({st.health_reason})" if st.health_reason else ""
            until = st.cooldown_until if (st.cooldown_until or 0) > now else None
            return Rejection(
                provider=name,
                code=RejectCode.UNHEALTHY,
                reason=f"{name}: unavailable{why}",
                until=until,
            )
        if st.cooldown_until is not None and st.cooldown_until > now:
            return Rejection(
                provider=name,
                code=RejectCode.COOLDOWN,
                reason=f"{name}: cooldown {_dur(st.cooldown_until - now)}",
                until=st.cooldown_until,
            )
        if st.exhausted_until is not None and st.exhausted_until > now:
            return Rejection(
                provider=name,
                code=RejectCode.EXHAUSTED,
                reason=f"{name}: quota used up, resets in {_span(st.exhausted_until - now)}",
                until=st.exhausted_until,
            )
        _, gpu_rejection = _fitting_offers(p, ctx)
        if gpu_rejection is not None:
            return gpu_rejection
        if spec.interactive and not p.capabilities.interactive:
            return Rejection(
                provider=name,
                code=RejectCode.INTERACTIVE,
                reason=f"{name}: cannot host interactive sessions",
            )
        cap = p.entry.session_hours
        if f.hours is not None and cap is not None and f.hours > cap and not f.can_handoff(p):
            why = (
                "and the job does not checkpoint"
                if p.capabilities.resume and not f.interactive
                else "and it cannot resume"
                if not p.capabilities.resume
                else "and interactive jobs cannot hand off"
            )
            return Rejection(
                provider=name,
                code=RejectCode.SESSION,
                reason=f"{name}: {fmt_hours(f.hours)} exceeds its {fmt_hours(cap)} session {why}",
            )
        quota = self._quota_rejection(p, ctx, f)
        if quota is not None:
            return quota
        if p.live_attempts >= p.capabilities.max_concurrency:
            return Rejection(
                provider=name,
                code=RejectCode.CAPACITY,
                reason=f"{name}: busy ({p.live_attempts}/{p.capabilities.max_concurrency} running)",
            )
        return None

    def _quota_rejection(
        self, p: ProviderSnapshot, ctx: RoutingContext, f: JobFacts
    ) -> Rejection | None:
        q = p.quota
        left = remaining(q)
        if q is None or left is None:
            return None
        name = p.name
        until = q.resets_at if q.resets_at is not None and q.resets_at > ctx.now else None
        when = f", {_reset_label(until, ctx.now)}" if until is not None else ""
        est = " (est)" if q.source == "estimate" else ""
        if left <= 0:
            return Rejection(
                provider=name,
                code=RejectCode.QUOTA,
                reason=f"{name}: quota used up{est}{when}",
                until=until,
            )
        left_h = to_gpu_hours(p.entry, left, self._offer(p, ctx, f).name)
        if left_h is None or f.hours is None:
            return None
        handoff = f.can_handoff(p)
        need = min(f.hours, self.settings.handoff_min_hours) if handoff else f.hours
        if left_h >= need:
            return None
        if handoff:
            why = f"a checkpointed run needs {fmt_hours(need)} to start"
        else:
            why = f"job needs {fmt_hours(f.hours)} and does not checkpoint"
        return Rejection(
            provider=name,
            code=RejectCode.QUOTA,
            reason=f"{name}: {fmt_hours(left_h)} quota left{est}, {why}{when}",
            until=until,
        )

    def _reserved(self, p: ProviderSnapshot, f: JobFacts) -> Rejection | None:
        if _is_local(p) and not f.smoke:
            return Rejection(
                provider=p.name,
                code=RejectCode.RESERVED,
                reason=f"{p.name}: kept for smoke tests (use --smoke or --provider {p.name})",
            )
        threshold = self.settings.big_vram_providers.get(p.name)
        if threshold is not None:
            want = f.vram_wanted
            if want is None or want <= threshold:
                return Rejection(
                    provider=p.name,
                    code=RejectCode.RESERVED,
                    reason=f"{p.name}: saved for jobs needing more than {_gb(threshold)} VRAM",
                )
        return None

    # ------------------------------------------------------------------ score

    def _offer(self, p: ProviderSnapshot, ctx: RoutingContext, f: JobFacts) -> GpuOffer:
        offers, _ = _fitting_offers(p, ctx)
        if f.vram_need is None and f.vram_est is not None:
            fits = [o for o in offers if o.vram_gb >= f.vram_est]
            if fits:
                offers = fits
            else:
                return max(offers, key=lambda o: (o.vram_gb, o.count, o.name))
        return min(offers, key=lambda o: (o.vram_gb, -o.count, o.name))  # D32

    def _score(
        self, p: ProviderSnapshot, ctx: RoutingContext, f: JobFacts, *, last_resort: bool
    ) -> _Scored:
        s = self.settings
        offer = self._offer(p, ctx, f)
        sc = _Scored(p=p, offer=offer, last_resort=last_resort)
        sc.score = -p.entry.priority / 1000.0
        label = f" ({offer.label})" if offer.count > 1 else ""
        if _is_local(p):
            sc.fit = f"{offer.label} {_gb(offer.vram_gb)} on this Mac"
        elif f.vram_need is not None or f.vram_est is None or offer.vram_gb >= f.vram_est:
            sc.fit = f"fits {_gb(offer.vram_gb)}{label}"
        else:
            sc.fit = f"{offer.label} {_gb(offer.vram_gb)}"
            sc.add(
                W_EST_VRAM_MISS,
                f"estimated {_gb(f.vram_est)} may not fit {_gb(offer.vram_gb)}",
            )
        if last_resort:  # display only (weight orders fragments; not added to the score)
            if _is_local(p):
                sc.good.append((LAST_RESORT_RANK, "no cloud provider can take this job"))
            else:
                sc.good.append((LAST_RESORT_RANK, "the only provider that can take this job"))

        if f.smoke and _is_local(p):
            sc.add(W_SMOKE_LOCAL, "smoke test")

        q = p.quota
        left = remaining(q)
        left_h = to_gpu_hours(p.entry, left, offer.name) if left is not None else None
        sc.quota_left = left
        est = " (est)" if q is not None and q.source == "estimate" else ""
        ttr = (q.resets_at - ctx.now) if q is not None and q.resets_at is not None else None
        if ttr is not None and ttr <= 0:
            ttr = None

        long_job = f.is_long(s)
        expiring = False
        if _in(s.save_for_long_jobs, p):
            if long_job:
                assert f.hours is not None
                cap = p.entry.session_hours
                sessions = f" suits its {fmt_hours(cap)} sessions" if cap else ""
                sc.add(W_LONG_FIT, f"{fmt_hours(f.hours)} job{sessions}")
            elif left_h is not None and ttr is not None and left_h >= ttr / 3600:
                # more quota left than hours until the reset: it expires unused anyway
                reset = _reset_label(ctx.now + ttr, ctx.now)
                sc.add(W_EXPIRING, f"{fmt_hours(left_h)} left{est}, {reset}: use it or lose it")
                expiring = True
            else:
                sc.add(W_SAVED, f"saved for jobs over {fmt_hours(s.long_job_hours)}")
        if _in(s.short_job_providers, p):
            if f.interactive:
                sc.add(W_SHORT, "interactive")
            elif not long_job:
                sc.add(W_SHORT, "short job" if f.hours is not None else "suits short jobs")
            else:
                sc.add(W_SHORT_LONG, "sessions not guaranteed for long jobs")
        threshold = s.big_vram_providers.get(p.name)
        want = f.vram_wanted
        if threshold is not None and want is not None and want > threshold:
            sc.add(W_BIG_VRAM, f"needs more than {_gb(threshold)}")

        if left is not None and left > 0 and ttr is not None:
            bonus = W_RESET_MAX * max(0.0, min(1.0, 1.0 - ttr / RESET_HORIZON_S))
            if bonus > 0:
                named = bonus >= 5 and not expiring  # "use it or lose it" already says it
                sc.add(bonus, _reset_label(ctx.now + ttr, ctx.now) if named else None)
        if left_h is not None and left_h > 0 and f.hours is not None:
            share = f.hours / left_h
            sc.quota_share = share
            if share > SHARE_WARN:
                pct = f"{min(share, SHARE_NOTHING_LEFT):.0%}"
                sc.add(W_SHARE, f"uses {pct} of the {fmt_hours(left_h)} left{est}")
        elif left is not None and left <= 0 and f.hours is not None:
            sc.quota_share = SHARE_NOTHING_LEFT  # nothing left: the policy's quota rule asks
        return sc

    # ------------------------------------------------------------------ explain

    def _role_weight(self, p: ProviderSnapshot, f: JobFacts) -> float:
        """How well `p` suits the job by role alone (for naming a ruled-out better fit)."""
        s = self.settings
        w = 0.0
        long_job = f.is_long(s)
        if _in(s.save_for_long_jobs, p):
            w += W_LONG_FIT if long_job else W_SAVED
        if _in(s.short_job_providers, p):
            w += W_SHORT if (f.interactive or not long_job) else W_SHORT_LONG
        if f.smoke and _is_local(p):
            w += W_SMOKE_LOCAL
        return w

    def _candidate(
        self,
        s: _Scored,
        rank: int,
        scored: list[_Scored],
        rejected: list[Rejection],
        ctx: RoutingContext,
        f: JobFacts,
    ) -> Candidate:
        name = s.p.name
        good = [frag for _, frag in sorted(s.good, key=lambda x: -x[0])]
        bad = [frag for _, frag in sorted(s.bad, key=lambda x: x[0])]
        parts = [s.fit]
        if rank == 0:
            headline = self._headline(s, scored, rejected, ctx, f)
            parts.extend([headline] if headline else [])
            parts.extend(bad[:1])
            resumed = _resume_fragment(f)
            parts.extend([resumed] if resumed else [])
        else:
            parts.extend(bad[:1] + good[:1])
        q = s.p.quota
        return Candidate(
            provider=name,
            gpu=s.offer.label,
            vram_gb=s.offer.vram_gb,
            score=round(s.score, 3),
            reason=f"{name}: " + ", ".join(p for p in parts if p),
            quota_left=s.quota_left,
            quota_unit=str(q.unit) if q is not None else None,
            quota_share=None if s.quota_share is None else round(s.quota_share, 4),
            resets_at=q.resets_at if q is not None else None,
            quota_source=q.source if q is not None else None,
        )

    def _headline(
        self,
        chosen: _Scored,
        scored: list[_Scored],
        rejected: list[Rejection],
        ctx: RoutingContext,
        f: JobFacts,
    ) -> str | None:
        """Why the chosen one, in one fragment: last resort; else a ruled-out provider
        that suits the job better by role ("kaggle quota used up, resets in 2d"); else the
        runner-up's handicap when it outweighs the chosen one's own best point ("kaggle
        saved for jobs over 4h"); else that point ("6h job suits its 12h sessions")."""
        own = chosen.top_good()
        if chosen.last_resort and own is not None:
            return own[1]
        blocked = self._blocked_better(chosen, rejected, ctx, f)
        if blocked is not None:
            return blocked
        if len(scored) > 1:
            handicap = scored[1].top_bad()
            if handicap is not None and (own is None or -handicap[0] > own[0]):
                return f"{scored[1].p.name} {handicap[1]}"
        return own[1] if own is not None else None

    def _blocked_better(
        self,
        chosen: _Scored,
        rejected: list[Rejection],
        ctx: RoutingContext,
        f: JobFacts,
    ) -> str | None:
        by_name = {p.name: p for p in ctx.providers}
        mine = self._role_weight(chosen.p, f)
        blocked = [
            r
            for r in rejected
            if r.code
            in (SCORING_TEMPORARY | {RejectCode.EXCLUDED, RejectCode.SESSION, RejectCode.QUOTA})
            and r.provider in by_name
            and self._role_weight(by_name[r.provider], f) > mine
        ]
        if not blocked:
            return None
        best = max(blocked, key=lambda r: self._role_weight(by_name[r.provider], f))
        return best.reason.replace(f"{best.provider}: ", f"{best.provider} ", 1)

    # ------------------------------------------------------------------ pinned

    def _pinned(self, ctx: RoutingContext, f: JobFacts) -> RouteDecision:
        base = self._simple.route(ctx)
        by_name = {p.name: p for p in ctx.providers}
        cands: list[Candidate] = []
        for c in base.candidates:
            p = by_name.get(c.provider)
            q = p.quota if p is not None else None
            left = remaining(q)
            left_h = (
                to_gpu_hours(p.entry, left, c.gpu) if (p is not None and left is not None) else None
            )
            share = None
            if left_h is not None and left_h > 0 and f.hours is not None:
                share = round(f.hours / left_h, 4)
            elif left is not None and left <= 0 and f.hours is not None:
                # the ledger says nothing is left: the policy's quota rule must still ask
                share = SHARE_NOTHING_LEFT
            vram = f" {_gb(c.vram_gb)}" if c.vram_gb is not None else ""
            cands.append(
                c.model_copy(
                    update={
                        "reason": f"{c.provider}: pinned with --provider ({c.gpu}{vram})",
                        "quota_left": left,
                        "quota_unit": str(q.unit) if q is not None else None,
                        "quota_share": share,
                        "resets_at": q.resets_at if q is not None else None,
                        "quota_source": q.source if q is not None else None,
                    }
                )
            )
        chosen = cands[0] if cands else None
        return base.model_copy(
            update={
                "candidates": cands,
                "chosen": chosen,
                "reason": chosen.reason if chosen is not None else base.reason,
                "hours": f.hours,
                "hours_source": f.hours_source,
                "smoke": f.smoke,
            }
        )
