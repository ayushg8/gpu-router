"""Phase-1 first-fit router (owner: group C). Phase 5 replaces it with router/scoring.py.

Filter each ProviderSnapshot in catalog order, recording one Rejection per dropped provider:
  OVERRIDE (spec.provider set and != name), EXCLUDED (ctx.excluded), DISABLED / AUTH /
  UNHEALTHY (state.health), COOLDOWN (state.cooldown_until > now, until=cooldown_until),
  EXHAUSTED (state.exhausted_until > now, until=exhausted_until), VRAM (no GpuOffer with
  vram_gb >= spec.vram_gb), GPU_TYPE (spec.gpu not among offer names, case-insensitive),
  INTERACTIVE (spec.interactive and not capabilities.interactive), SESSION (spec.hours >
  session_hours and not capabilities.resume), CAPACITY (live_attempts >= max_concurrency).
Rank survivors by (entry.priority, name); score = -priority; GPU = smallest offer that fits
(least per-GPU VRAM; at equal VRAM the offer with MORE GPUs, D32: same free-tier quota cost,
and Kaggle's 2xT4 is proven live while its P100 is not).
Outcome: PLACE the first; else WAIT if any rejection is in TEMPORARY_REJECTIONS
(retry_at = min known `until`, else None and the engine applies backoff; a rejection
without `until` caps the wait at the engine's backoff, D44); else NO_FIT.
Reasons are one line: "fake: first fit (T4 16GB)"; NO_FIT: "no provider fits: needs 80GB
VRAM, largest is 40GB (fake-b)"; WAIT: "all providers busy: kaggle cooldown 4m, ...".

A spec.provider override that names a provider which is not registered at all is NO_FIT
("provider 'x' is not enabled"), never a silent reroute.
"""

from __future__ import annotations

from gpu_router.models import ProviderHealth
from gpu_router.providers.catalog import GpuOffer
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


def _dur(seconds: float) -> str:
    s = max(0, round(seconds))
    if s < 60:
        return f"{s}s"
    m = s // 60
    if m < 60:
        return f"{m}m"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m" if m else f"{h}h"


def _gb(v: float) -> str:
    return f"{v:g}GB"


def _fitting_offers(
    p: ProviderSnapshot, ctx: RoutingContext
) -> tuple[list[GpuOffer], Rejection | None]:
    """Offers that satisfy the spec's GPU type and VRAM, or the rejection explaining why
    none does (GPU type is checked first: it is the more specific constraint)."""
    spec = ctx.job.spec
    offers = list(p.entry.gpus)
    if spec.gpu is not None:
        want = spec.gpu.lower()
        offers = [o for o in offers if o.name.lower() == want]
        if not offers:
            names = ", ".join(o.name for o in p.entry.gpus) or "none"
            return [], Rejection(
                provider=p.name,
                code=RejectCode.GPU_TYPE,
                reason=f"{p.name}: no {spec.gpu} (offers {names})",
            )
    if spec.vram_gb is not None:
        fits = [o for o in offers if o.vram_gb >= spec.vram_gb]
        if not fits:
            largest = max((o.vram_gb for o in offers), default=0.0)
            return [], Rejection(
                provider=p.name,
                code=RejectCode.VRAM,
                reason=f"{p.name}: needs {_gb(spec.vram_gb)} VRAM, largest is {_gb(largest)}",
            )
        offers = fits
    if not offers:
        return [], Rejection(
            provider=p.name, code=RejectCode.VRAM, reason=f"{p.name}: offers no GPU"
        )
    return offers, None


def _reject(p: ProviderSnapshot, ctx: RoutingContext) -> Rejection | None:
    """The first filter that drops `p`, in the documented order, or None if it survives."""
    spec = ctx.job.spec
    now = ctx.now
    name = p.name
    if spec.provider is not None and spec.provider != name:
        return Rejection(
            provider=name, code=RejectCode.OVERRIDE, reason=f"{name}: job asks for {spec.provider}"
        )
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
    health = p.state.health
    if health is ProviderHealth.DISABLED:
        return Rejection(provider=name, code=RejectCode.DISABLED, reason=f"{name}: disabled")
    if health is ProviderHealth.AUTH_REQUIRED:
        return Rejection(
            provider=name,
            code=RejectCode.AUTH,
            reason=f"{name}: needs login (run `gpu login {name}`)",
        )
    if health is ProviderHealth.UNAVAILABLE:
        why = f" ({p.state.health_reason})" if p.state.health_reason else ""
        until = p.state.cooldown_until if (p.state.cooldown_until or 0) > now else None
        return Rejection(
            provider=name,
            code=RejectCode.UNHEALTHY,
            reason=f"{name}: unavailable{why}",
            until=until,
        )
    if p.state.cooldown_until is not None and p.state.cooldown_until > now:
        return Rejection(
            provider=name,
            code=RejectCode.COOLDOWN,
            reason=f"{name}: cooldown {_dur(p.state.cooldown_until - now)}",
            until=p.state.cooldown_until,
        )
    if p.state.exhausted_until is not None and p.state.exhausted_until > now:
        return Rejection(
            provider=name,
            code=RejectCode.EXHAUSTED,
            reason=f"{name}: quota used up, resets in {_dur(p.state.exhausted_until - now)}",
            until=p.state.exhausted_until,
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
    session_h = p.entry.session_hours
    if (
        spec.hours is not None
        and session_h is not None
        and spec.hours > session_h
        and not p.capabilities.resume
    ):
        return Rejection(
            provider=name,
            code=RejectCode.SESSION,
            reason=f"{name}: {spec.hours:g}h exceeds its {session_h:g}h session "
            f"and it cannot resume",
        )
    if p.live_attempts >= p.capabilities.max_concurrency:
        return Rejection(
            provider=name,
            code=RejectCode.CAPACITY,
            reason=f"{name}: busy ({p.live_attempts}/{p.capabilities.max_concurrency} running)",
        )
    return None


def _no_fit_reason(ctx: RoutingContext, rejected: list[Rejection]) -> str:
    spec = ctx.job.spec
    if not ctx.providers:
        return "no provider fits: no providers are enabled (run `gpu setup`)"
    if spec.provider is not None and spec.provider not in {p.name for p in ctx.providers}:
        return f"no provider fits: provider {spec.provider!r} is not enabled"
    meaningful = [r for r in rejected if r.code is not RejectCode.OVERRIDE] or rejected
    # RESERVED (phase 5: the Mac kept for smoke tests, big_vram_providers for big jobs)
    # never explains why the job cannot run: judge "every provider rejected it" / "too
    # big" on the rest
    core = [r for r in meaningful if r.code is not RejectCode.RESERVED] or meaningful
    if spec.vram_gb is not None and all(r.code is RejectCode.VRAM for r in core):
        best = max(ctx.providers, key=lambda p: (p.entry.max_vram_gb, -p.entry.priority))
        # phase 7b: say it plainly, there is no bigger free GPU to wait for (modal was
        # dropped: it needs a card; lightning's free tier refuses L4, D56)
        return (
            f"no provider fits: needs {_gb(spec.vram_gb)} VRAM and no free provider has "
            f"more than {_gb(best.entry.max_vram_gb)} (largest: {best.name}); "
            "`gpu providers` lists the excluded ones"
        )
    if all(r.code is RejectCode.EXCLUDED for r in core):
        data = [r for r in core if r.provider in ctx.data_unreachable]
        if data:  # say why: the earlier-rejection wording hid it (2026-10-04 field test)
            return "no provider fits: " + "; ".join(r.reason for r in core[:3])
        names = ", ".join(r.provider for r in core)
        return f"no provider fits: every provider rejected this job ({names})"
    parts = [r.reason for r in meaningful[:3]]
    more = f" (+{len(meaningful) - 3} more)" if len(meaningful) > 3 else ""
    return "no provider fits: " + "; ".join(parts) + more


class SimpleRouter:
    name = "simple"

    def route(self, ctx: RoutingContext) -> RouteDecision:
        rejected: list[Rejection] = []
        survivors: list[ProviderSnapshot] = []
        for p in ctx.providers:
            r = _reject(p, ctx)
            if r is None:
                survivors.append(p)
            else:
                rejected.append(r)

        survivors.sort(key=lambda p: (p.entry.priority, p.name))
        candidates: list[Candidate] = []
        for p in survivors:
            offers, _ = _fitting_offers(p, ctx)
            best = min(offers, key=lambda o: (o.vram_gb, -o.count, o.name))  # D32
            candidates.append(
                Candidate(
                    provider=p.name,
                    gpu=best.label,
                    vram_gb=best.vram_gb,
                    score=float(-p.entry.priority),
                    reason=f"{p.name}: first fit ({best.label} {_gb(best.vram_gb)})",
                )
            )

        if candidates:
            chosen = candidates[0]
            return RouteDecision(
                outcome=RouteOutcome.PLACE,
                chosen=chosen,
                candidates=candidates,
                rejected=rejected,
                reason=chosen.reason,
                router=self.name,
            )

        temporary = [r for r in rejected if r.code in TEMPORARY_REJECTIONS]
        if temporary:
            retry_at = earliest_retry(temporary)
            parts = [r.reason for r in temporary[:4]]
            more = f" (+{len(temporary) - 4} more)" if len(temporary) > 4 else ""
            return RouteDecision(
                outcome=RouteOutcome.WAIT,
                rejected=rejected,
                reason="all providers busy: " + ", ".join(parts) + more,
                retry_at=retry_at,
                router=self.name,
            )

        return RouteDecision(
            outcome=RouteOutcome.NO_FIT,
            rejected=rejected,
            reason=_no_fit_reason(ctx, rejected),
            router=self.name,
        )
