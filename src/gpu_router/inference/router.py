"""Inference router (phase 7b): which free inference API answers this request, and why.

Pure given its inputs (catalog, ledger reads, key presence, now), like the GPU routers.

1. Candidates: routable providers that serve the model (an alias, a listed provider id,
   or a `passthrough` id), in catalog priority order. `--provider` pins one.
2. Filter (one rejection each, first match wins): disabled, no key (`gpu login <p>`),
   blocked (a used-up day until its reset, a per-minute cooldown, a rejected key), and
   QUOTA: for every unit the request needs (requests, tokens, neurons, usd) the ledger's
   remaining at the model's scope and the provider's scope must cover the estimate
   (live when a fresh reading exists, else the catalog limit minus our count).
3. Score (higher wins):  40 x share of the binding allowance left after this request
   (unknown limit: 40 x 0.5 - 5)  +20 when the allowance resets within a day ("use it or
   lose it", the spec's "spend quota that resets soonest first")  -0.25 x priority
   -15 for an unlisted passthrough id.
4. Explain: the chosen candidate's line names the binding counter ("870/1000 requests left
   today (live)"), plus a better-placed provider that was ruled out, else the runner-up.

Estimates: input tokens ~ characters / 4 (+8 per message), output = max_tokens or
DEFAULT_OUTPUT_TOKENS; neurons and usd from the catalog's per-million-token rates.
"""

from __future__ import annotations

import difflib
import math
from collections.abc import Callable
from dataclasses import dataclass

from gpu_router.inference.catalog import InferenceCatalog, InferenceEntry, ModelEntry, Unit
from gpu_router.inference.ledger import InferenceLedger, Remaining
from gpu_router.inference.models import (
    InferCandidate,
    InferRejection,
    InferRequest,
    InferRoute,
)

__all__ = [
    "DEFAULT_OUTPUT_TOKENS",
    "Need",
    "estimate_need",
    "fmt_amount",
    "route",
    "when",
]

DEFAULT_OUTPUT_TOKENS = 512
W_SHARE = 40.0
W_UNKNOWN = -5.0
W_DAILY = 20.0
W_PRIORITY = -0.25
W_UNLISTED = -15.0

KeyCheck = Callable[[InferenceEntry], tuple[bool, list[str], str | None]]


@dataclass(frozen=True)
class Need:
    input_tokens: int
    output_tokens: int
    amounts: dict[Unit, float]


def estimate_need(entry: InferenceEntry, model: ModelEntry, req: InferRequest) -> Need:
    messages = req.chat()
    in_tok = sum(math.ceil(len(m.content) / 4) + 8 for m in messages)
    out_tok = req.max_tokens or DEFAULT_OUTPUT_TOKENS
    amounts: dict[Unit, float] = {"requests": 1.0, "tokens": float(in_tok + out_tok)}
    neurons = entry.neurons(model, in_tok, out_tok)
    if neurons is not None:
        amounts["neurons"] = neurons
    usd = entry.usd(model, in_tok, out_tok)
    if usd is not None:
        amounts["usd"] = usd
    return Need(in_tok, out_tok, amounts)


def fmt_amount(value: float, unit: str) -> str:
    if unit == "usd":
        return f"${value:.2f}" if value >= 0.01 or value == 0 else f"${value:.3f}"
    if value >= 10_000:
        return f"{value / 1000:.0f}k"
    if value == int(value):
        return f"{value:.0f}"
    if value >= 1000:
        short = f"{value / 1000:.1f}"
        return (short[:-2] if short.endswith(".0") else short) + "k"
    return f"{value:.0f}" if value >= 10 else f"{value:.1f}"


def when(ts: float | None, now: float) -> str:
    """'in 45m' / 'in 6h' / 'in 3d' (never an absolute clock: the daemon has no locale)."""
    if ts is None:
        return "at an unknown time"
    left = max(0.0, ts - now)
    if left < 3600:
        return f"in {max(1, round(left / 60))}m"
    if left < 86_400 * 2:
        return f"in {round(left / 3600)}h"
    return f"in {round(left / 86_400)}d"


def _period(entry: InferenceEntry) -> str:
    return {"monthly": "this month", "rolling_24h": "in this 24h window"}.get(entry.reset, "today")


def _counter(r: Remaining, entry: InferenceEntry) -> str:
    unit = "requests" if r.unit == "requests" else r.unit
    tag = "live" if r.source == "live" else "est"
    what = f"{fmt_amount(r.remaining, r.unit)}"
    if r.limit is not None:
        what += f"/{fmt_amount(r.limit, r.unit)}"
    if r.unit == "usd":
        return f"{what} credits left {_period(entry)} ({tag})"
    return f"{what} {unit} left {_period(entry)} ({tag})"


@dataclass(frozen=True)
class _Eval:
    entry: InferenceEntry
    model: ModelEntry
    score: float
    binding: Remaining | None
    reason: str


def _check_quota(
    ledger: InferenceLedger, entry: InferenceEntry, model: ModelEntry, need: Need, now: float
) -> tuple[InferRejection | None, Remaining | None, float | None]:
    """(rejection, binding counter, share left after this request)."""
    binding: Remaining | None = None
    share: float | None = None
    for unit, amount in need.amounts.items():
        for scope in (model.id, ""):
            rem = ledger.remaining(entry.name, scope, unit)
            if rem is None:
                continue
            if rem.remaining < amount:
                where = f" for {model.id}" if scope else ""
                need_txt = (
                    f", needs ~{fmt_amount(amount, unit)}" if unit not in ("requests",) else ""
                )
                return (
                    InferRejection(
                        provider=entry.name,
                        code="quota",
                        reason=f"{entry.name}: {_counter(rem, entry)}{where}{need_txt}, "
                        f"resets {when(rem.resets_at, now)}",
                        retry_at=rem.resets_at,
                    ),
                    rem,
                    0.0,
                )
            if rem.limit:
                after = max(0.0, (rem.remaining - amount) / rem.limit)
                if share is None or after < share:
                    share, binding = after, rem
            elif binding is None:
                binding = rem
    return None, binding, share


def _evaluate(
    entry: InferenceEntry,
    model: ModelEntry,
    ledger: InferenceLedger,
    need: Need,
    now: float,
) -> _Eval | InferRejection:
    blocks = ledger.blocks(entry.name, model.id)
    if blocks:
        b = blocks[0]
        code = "exhausted" if b.kind == "exhausted" else "cooldown"
        if b.kind == "exhausted":
            what = f"{model.id} used up" if b.scope else "used up"
            text = f"{entry.name}: {what} {_period(entry)}, resets {when(b.until, now)}"
        elif b.kind == "auth":
            text = f"{entry.name}: key rejected (run `gpu login {entry.login_name}`)"
            code = "no_key"
        else:
            text = f"{entry.name}: cooling down {when(b.until, now)}"
        return InferRejection(provider=entry.name, code=code, reason=text, retry_at=b.until)
    rejection, binding, share = _check_quota(ledger, entry, model, need, now)
    if rejection is not None:
        return rejection
    score = W_PRIORITY * entry.priority
    _, window_end = ledger.window(entry.name)
    reset_at = binding.resets_at if binding is not None and binding.resets_at else window_end
    if share is not None:
        score += W_SHARE * share
        left = _counter(binding, entry) if binding is not None else ""
    else:
        score += W_SHARE * 0.5 + W_UNKNOWN
        left = "free limit unknown" + (f" ({_counter(binding, entry)})" if binding else "")
    if reset_at - now <= 86_400:
        score += W_DAILY
    if model.unlisted:
        score += W_UNLISTED
    shown = model.alias if model.alias == model.id else f"{model.alias} ({model.id})"
    reason = f"{entry.name}: {shown} · {left}, resets {when(reset_at, now)}"
    if model.unlisted:
        reason += " · not in the catalog, sent as-is"
    return _Eval(entry, model, score, binding, reason)


def route(
    req: InferRequest,
    catalog: InferenceCatalog,
    ledger: InferenceLedger,
    has_key: KeyCheck,
    now: float,
) -> InferRoute:
    """The routing decision for `req` (no call, no ledger writes)."""
    pin = req.provider
    entries = catalog.ordered()
    if pin is not None:
        found = catalog.get(pin)
        if found is None:
            names = [e.name for e in entries]
            close = difflib.get_close_matches(pin, names, n=1, cutoff=0.5)
            hint = f" (did you mean {close[0]}?)" if close else ""
            return InferRoute(
                outcome="no_fit",
                model=req.model,
                reason=f"no inference provider named {pin!r}{hint}; known: {', '.join(names)}",
            )
        entries = [found]
    rejected: list[InferRejection] = []
    evals: list[_Eval] = []
    serving = 0
    for entry in entries:
        if not entry.routable:
            if pin is not None:
                why = "disabled" if not entry.enabled else "has no chat endpoint (listed only)"
                rejected.append(
                    InferRejection(
                        provider=entry.name,
                        code="disabled" if not entry.enabled else "not_routable",
                        reason=f"{entry.name}: {why}",
                    )
                )
            continue
        model = entry.resolve(req.model)
        if model is None and pin is not None:
            model = ModelEntry(alias=req.model, id=req.model, unlisted=True)  # pinned: as-is
        if model is None:
            continue
        serving += 1
        ok, _missing, problem = has_key(entry)
        if not ok:
            why = problem or f"needs a key (run `gpu login {entry.login_name}`)"
            rejected.append(
                InferRejection(provider=entry.name, code="no_key", reason=f"{entry.name}: {why}")
            )
            continue
        need = estimate_need(entry, model, req)
        result = _evaluate(entry, model, ledger, need, now)
        if isinstance(result, InferRejection):
            rejected.append(result)
        else:
            evals.append(result)

    evals.sort(key=lambda e: (-e.score, e.entry.priority, e.entry.name))
    candidates = [
        InferCandidate(
            provider=e.entry.name,
            model_id=e.model.id,
            score=round(e.score, 2),
            reason=e.reason,
            left=_counter(e.binding, e.entry) if e.binding is not None else None,
            unlisted=e.model.unlisted,
        )
        for e in evals
    ]
    if candidates:
        chosen = candidates[0]
        extra = _why_not_better(evals[0], rejected, candidates, catalog)
        return InferRoute(
            outcome="place",
            model=req.model,
            chosen=chosen,
            candidates=candidates,
            rejected=rejected,
            reason=chosen.reason + (f"; {extra}" if extra else ""),
        )
    if serving == 0 and not rejected:
        known = catalog.aliases()
        close = difflib.get_close_matches(req.model, known, n=3, cutoff=0.4)
        hint = f"did you mean {', '.join(close)}? " if close else ""
        return InferRoute(
            outcome="no_fit",
            model=req.model,
            rejected=rejected,
            reason=f"no free inference provider serves {req.model!r}; {hint}"
            f"known models: {', '.join(known)} (or pin one with --provider to send it as-is)",
        )
    waiting = [r for r in rejected if r.retry_at is not None and r.code in ("cooldown", "quota")]
    waiting += [r for r in rejected if r.retry_at is not None and r.code == "exhausted"]
    if waiting:
        retry_at = min(r.retry_at for r in waiting if r.retry_at is not None)
        parts = "; ".join(r.reason for r in rejected[:4])
        return InferRoute(
            outcome="wait",
            model=req.model,
            rejected=rejected,
            reason=f"nothing free for {req.model} right now: {parts}",
            retry_at=retry_at,
        )
    parts = "; ".join(r.reason for r in rejected[:4])
    return InferRoute(
        outcome="no_fit",
        model=req.model,
        rejected=rejected,
        reason=f"no inference provider can take {req.model}: {parts}",
    )


def _why_not_better(
    best: _Eval,
    rejected: list[InferRejection],
    candidates: list[InferCandidate],
    catalog: InferenceCatalog,
) -> str | None:
    """A higher-priority provider that was ruled out, else the runner-up, in a few words."""
    for r in rejected:
        entry = catalog.get(r.provider)
        if entry is not None and entry.priority < best.entry.priority:
            return r.reason
    if len(candidates) > 1:
        nxt = candidates[1]
        return f"next: {nxt.provider}" + (f" ({nxt.left})" if nxt.left else "")
    return None
