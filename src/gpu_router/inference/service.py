"""`InferenceService` (phase 7b): the daemon's owner of the inference lane.

Holds the typed catalog, the ledger and one `ChatClient` per provider. Methods are
blocking (HTTP calls, Keychain reads) and run in worker threads (`asyncio.to_thread` in
daemon/lanes.py); the ledger has its own lock.

`infer(req)`: route -> call the chosen provider -> record usage + live readings -> result.
On a provider error the ledger learns from it (a per-minute cooldown, a used-up day until
its reset, a rejected key, an outage) and the next candidate is tried; a bad request
(too long, bad parameter) is not retried elsewhere. When every candidate is cooling down
for less than `req.wait_s`, it waits once and routes again.

Invariants: keys only through inference/keys.py (12); prompts and replies are never logged,
only provider, model, token counts and timings; a test-mode daemon never calls a real
provider unless a transport was injected or GPU_ROUTER_REAL_PROVIDERS names it (20); time
from the injected clock (13).
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

from gpu_router.errors import InvalidRequest
from gpu_router.inference.catalog import InferenceCatalog, InferenceEntry, ModelEntry
from gpu_router.inference.clients import ChatClient, ChatReply, read_timeout_for
from gpu_router.inference.errors import (
    InferAuthRequired,
    InferenceError,
    InferModelUnavailable,
    InferQuotaExhausted,
    InferRateLimited,
    InferUnavailable,
)
from gpu_router.inference.keys import key_status, load_keys
from gpu_router.inference.ledger import InferenceLedger, LiveReading, window_bounds
from gpu_router.inference.models import (
    InferAttempt,
    InferCounter,
    InferProviderView,
    InferQuotaView,
    InferRequest,
    InferResult,
    InferRoute,
    InferUsage,
)
from gpu_router.inference.router import KeyCheck, estimate_need, fmt_amount, route, when
from gpu_router.log import log_event

if TYPE_CHECKING:
    import httpx
    from pydantic import SecretStr

    from gpu_router.clock import Clock

__all__ = ["InferenceService", "ledger_path"]

_logger = logging.getLogger("gpu_router.inference")

MAX_ROUNDS = 8  # route/call rounds per request (fallbacks + one wait)
PER_PROVIDER_CONCURRENCY = 4
#: worker threads of the inference lane's own executor (4 providers x 4 slots): queued
#: requests wait there, never in the daemon's default executor (review fix)
INFER_WORKERS = 16
#: a request's overall budget when it names none (remote.py sends 400 s)
DEFAULT_DEADLINE_S = 390.0
#: a provider call gets at least this much of the budget, else the request gives up
MIN_CALL_S = 5.0
#: longest wait for one of a provider's PER_PROVIDER_CONCURRENCY slots before trying the
#: next provider
SLOT_WAIT_S = 30.0
UNAVAILABLE_COOLDOWN_S = 60.0
AUTH_BLOCK_S = 600.0
MODEL_MISSING_BLOCK_S = 3600.0

KeyLoader = Callable[[InferenceEntry], "dict[str, SecretStr]"]


def ledger_path(home: Path) -> Path:
    return home / "inference" / "ledger.json"


class InferenceService:
    def __init__(
        self,
        catalog: InferenceCatalog,
        clock: Clock,
        *,
        ledger_file: Path | None,
        test_mode: bool = False,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        key_check: KeyCheck = key_status,
        key_loader: KeyLoader = load_keys,
        timeout_s: float = 120.0,
    ) -> None:
        self.catalog = catalog
        self.clock = clock
        self.test_mode = test_mode
        self.ledger = InferenceLedger(ledger_file, catalog, clock)
        self._transport = transport
        self._sleep = sleep
        self._key_check = key_check
        self._key_loader = key_loader
        self._timeout_s = timeout_s
        self._clients: dict[str, ChatClient] = {}
        self._slots: dict[str, threading.BoundedSemaphore] = {}
        self._lock = threading.Lock()
        #: /v1/infer runs here (daemon/lanes.py), not on the default executor that
        #: submits, bundling and the other endpoints share
        self.executor = ThreadPoolExecutor(
            max_workers=INFER_WORKERS, thread_name_prefix="gpu-infer"
        )

    # ------------------------------------------------------------------ plumbing

    def close(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)
        with self._lock:
            clients, self._clients = self._clients, {}
        for client in clients.values():
            client.close()

    def _client(self, entry: InferenceEntry) -> ChatClient:
        with self._lock:
            client = self._clients.get(entry.name)
            if client is None:
                client = ChatClient(entry, transport=self._transport, timeout_s=self._timeout_s)
                self._clients[entry.name] = client
                self._slots[entry.name] = threading.BoundedSemaphore(PER_PROVIDER_CONCURRENCY)
            return client

    def _guard_test_mode(self, entry: InferenceEntry) -> None:
        if not self.test_mode or self._transport is not None:
            return
        from gpu_router.adapters.registry import real_providers_opted_in

        if entry.name not in real_providers_opted_in():
            raise InferUnavailable(
                f"test mode: real calls to {entry.name} are off",
                provider=entry.name,
                hint=f"GPU_ROUTER_REAL_PROVIDERS={entry.name} allows them (invariant 20)",
            )

    # ------------------------------------------------------------------ views

    def providers(self) -> list[InferProviderView]:
        views: list[InferProviderView] = []
        for entry in self.catalog.ordered():
            ok, missing, problem = self._key_check(entry) if entry.secrets else (True, [], None)
            views.append(
                InferProviderView(
                    name=entry.name,
                    display_name=entry.display_name,
                    routable=entry.routable,
                    logged_in=ok and bool(entry.secrets),
                    login=entry.login_name,
                    missing=missing,
                    reset=entry.reset,
                    limits={str(k): v for k, v in entry.limits.items()},
                    models={alias: m.id for alias, m in entry.models.items()},
                    note=entry.note,
                    link=entry.link,
                    docs=list(entry.docs),
                    verified_at=entry.verified_at.isoformat() if entry.verified_at else None,
                    problem=problem,
                )
            )
        for name, why in sorted(self.catalog.problems.items()):
            views.append(
                InferProviderView(
                    name=name,
                    display_name=name,
                    routable=False,
                    logged_in=False,
                    login=name,
                    reset="unknown",
                    problem=f"providers.yaml inference.{name} is invalid: {why}",
                )
            )
        return views

    def quota(self) -> list[InferQuotaView]:
        return [self.quota_view(e) for e in self.catalog.ordered()]

    def quota_view(self, entry: InferenceEntry) -> InferQuotaView:
        now = self.clock.now()
        snap = self.ledger.snapshot(entry.name)
        _, window_end = self.ledger.window(entry.name)
        counters: list[InferCounter] = []
        scopes: list[tuple[str, dict[str, float | None]]] = [
            ("", {str(u): v for u, v in entry.limits.items()})
        ]
        scopes += [
            (m.id, {str(u): v for u, v in m.limits.items()})
            for m in entry.models.values()
            if m.limits
        ]
        live_scopes = snap.get("live", {})
        for scope in live_scopes:
            if scope not in {s for s, _ in scopes}:
                scopes.append((scope, {}))
        for scope, limits in scopes:
            units = list(limits) + [u for u in live_scopes.get(scope, {}) if u not in limits]
            for unit in units:
                rem = self.ledger.remaining(entry.name, scope, unit)
                used = self.ledger.used(entry.name, scope, unit)
                counters.append(
                    InferCounter(
                        scope=scope,
                        unit=unit,
                        used=round(used, 4),
                        limit=rem.limit if rem is not None else limits.get(unit),
                        remaining=round(rem.remaining, 4) if rem is not None else None,
                        source="live" if rem is not None and rem.source == "live" else "estimate",
                        resets_at=rem.resets_at if rem is not None else window_end,
                    )
                )
        blocked: dict[str, str] = {}
        blocked_until: float | None = None
        for b in self.ledger.blocks(entry.name):
            label = {"exhausted": "used up", "cooldown": "cooling down", "auth": "key rejected"}
            blocked[b.scope or "*"] = f"{label.get(b.kind, b.kind)} until {when(b.until, now)}"
            if b.scope == "":
                blocked_until = max(blocked_until or 0.0, b.until)
        requests_today = int(self.ledger.used(entry.name, "", "requests"))
        return InferQuotaView(
            provider=entry.name,
            display_name=entry.display_name,
            window=entry.reset,
            window_resets_at=window_end,
            requests_today=requests_today,
            counters=counters,
            blocked=blocked,
            blocked_until=blocked_until,
            summary=self._summary(entry, counters, blocked, requests_today, window_end, now),
        )

    @staticmethod
    def _summary(
        entry: InferenceEntry,
        counters: list[InferCounter],
        blocked: dict[str, str],
        requests: int,
        window_end: float,
        now: float,
    ) -> str:
        if not entry.routable:
            return f"{entry.name}: listed only, never routed"
        period = {"monthly": "this month", "rolling_24h": "in 24h"}.get(entry.reset, "today")
        known = [c for c in counters if c.remaining is not None and c.limit]
        if known:
            tight = min(known, key=lambda c: (c.remaining or 0) / (c.limit or 1))
            tag = "live" if tight.source == "live" else "est"
            where = f" for {tight.scope}" if tight.scope else ""
            unit = "credits" if tight.unit == "usd" else tight.unit
            left = (
                f"{fmt_amount(tight.remaining or 0, tight.unit)}/"
                f"{fmt_amount(tight.limit or 0, tight.unit)} {unit} left {period}{where} ({tag})"
            )
        else:
            left = f"{requests} requests {period}, free limit unknown"
        text = f"{entry.name}: {left}, resets {when(window_end, now)}"
        if "*" in blocked:
            text += f" · {blocked['*']}"
        elif blocked:
            text += f" · {len(blocked)} model(s) blocked"
        return text

    # ------------------------------------------------------------------ routing

    def route(self, req: InferRequest, *, skip: frozenset[str] = frozenset()) -> InferRoute:
        catalog = self.catalog
        if skip:
            catalog = InferenceCatalog(
                entries={n: e for n, e in catalog.entries.items() if n not in skip},
                problems=catalog.problems,
            )
        return route(req, catalog, self.ledger, self._key_check, self.clock.now())

    # ------------------------------------------------------------------ calls

    def infer(self, req: InferRequest) -> InferResult:
        fallbacks: list[InferAttempt] = []
        skip: set[str] = set()
        waited = False
        last_error: InferenceError | None = None
        first: InferRoute | None = None
        started = self.clock.now()
        budget_s = req.deadline_s or DEFAULT_DEADLINE_S
        deadline = started + budget_s
        for _ in range(MAX_ROUNDS):
            left = deadline - self.clock.now()
            if left < MIN_CALL_S:
                raise self._deadline_error(req, budget_s, fallbacks)
            decision = self.route(req, skip=frozenset(skip))
            first = first or decision
            if decision.outcome == "place" and decision.chosen is not None:
                chosen = decision.chosen
                entry = self.catalog.entries[chosen.provider]
                model = entry.resolve(chosen.model_id) or ModelEntry(
                    alias=req.model, id=chosen.model_id, unlisted=True
                )
                try:
                    reply = self._call(entry, model, req, left)
                except InferenceError as exc:
                    if not isinstance(exc, _Busy):  # our own slot limit teaches nothing
                        self._absorb(entry, model, exc, req)
                    fallbacks.append(
                        InferAttempt(
                            provider=entry.name,
                            model_id=model.id,
                            code=str(exc.code),
                            message=exc.message,
                        )
                    )
                    last_error = exc
                    if not exc.reroute:
                        exc.detail["fallbacks"] = [f.model_dump() for f in fallbacks]
                        raise
                    if not isinstance(exc, InferRateLimited):
                        # a cooldown is in the ledger now: routing sees it and may wait
                        skip.add(entry.name)
                    continue
                return self._result(req, entry, model, reply, decision, fallbacks)
            now = self.clock.now()
            budget = min(req.wait_s - (now - started), deadline - now - MIN_CALL_S)
            if (
                decision.outcome == "wait"
                and not waited
                and decision.retry_at is not None
                and decision.retry_at - now <= budget
            ):
                waited = True
                self._sleep(max(0.0, decision.retry_at - now) + 0.05)
                continue
            raise self._no_route_error(req, decision, fallbacks, last_error, first)
        raise self._no_route_error(req, self.route(req), fallbacks, last_error, first)

    def _deadline_error(
        self, req: InferRequest, budget_s: float, fallbacks: list[InferAttempt]
    ) -> InferUnavailable:
        tried = "; ".join(f"{f.provider}: {f.message}" for f in fallbacks) or "no provider"
        return InferUnavailable(
            f"no answer for {req.model} within {budget_s:.0f}s ({tried})",
            hint="try again, or ask for fewer tokens (--max-tokens)",
            detail={"model": req.model, "fallbacks": [f.model_dump() for f in fallbacks]},
        )

    def _call(
        self, entry: InferenceEntry, model: ModelEntry, req: InferRequest, left_s: float
    ) -> ChatReply:
        self._guard_test_mode(entry)
        keys = self._key_loader(entry)
        client = self._client(entry)
        slot = self._slots[entry.name]
        started = self.clock.now()
        if not slot.acquire(timeout=max(0.0, min(SLOT_WAIT_S, left_s - MIN_CALL_S))):
            raise _Busy(
                f"{entry.name} already has {PER_PROVIDER_CONCURRENCY} requests running",
                provider=entry.name,
            )
        try:
            left = left_s - (self.clock.now() - started)
            return client.chat(
                keys,
                model.id,
                req.chat(),
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                now=self.clock.now(),
                timeout_s=max(
                    MIN_CALL_S, min(read_timeout_for(req.max_tokens, self._timeout_s), left)
                ),
            )
        finally:
            slot.release()

    def _absorb(
        self, entry: InferenceEntry, model: ModelEntry, exc: InferenceError, req: InferRequest
    ) -> None:
        """Teach the ledger what a failed call said, so routing avoids it next time."""
        now = self.clock.now()
        readings = [r for r in exc.live if isinstance(r, LiveReading)]
        self.ledger.observe(entry.name, readings)
        if isinstance(exc, InferUnavailable) and exc.sent:
            # the read timed out after the request went out: the provider most likely
            # finished (and billed) the generation, so count the estimate (review fix)
            need = estimate_need(entry, model, req)
            amounts = {str(unit): value for unit, value in need.amounts.items()}
            self.ledger.record(entry.name, model.id, amounts, [])
        per_model = model.id if entry.quota_scope == "model" else ""
        if isinstance(exc, InferQuotaExhausted):
            until = exc.resets_at or window_bounds(entry.reset, now)[1]
            self.ledger.block(entry.name, exc.scope, until=until, kind="exhausted", why=exc.message)
        elif isinstance(exc, InferRateLimited):
            wait = exc.retry_after if exc.retry_after is not None else 30.0
            self.ledger.block(entry.name, per_model, until=now + wait, kind="cooldown", why="429")
        elif isinstance(exc, InferModelUnavailable):
            self.ledger.block(
                entry.name,
                model.id,
                until=now + MODEL_MISSING_BLOCK_S,
                kind="cooldown",
                why=exc.message,
            )
        elif isinstance(exc, InferAuthRequired):
            ok, _, _ = self._key_check(entry)
            if ok:  # the key exists but was rejected: keep it out until someone logs in
                self.ledger.block(
                    entry.name, "", until=now + AUTH_BLOCK_S, kind="auth", why=exc.message
                )
        elif isinstance(exc, InferUnavailable) and "test mode" not in exc.message:
            # a 5xx that a per-model provider answered is about that model (Gemini's 503
            # "this model is currently experiencing high demand"); no answer at all
            # (network, timeout) cools the whole provider down
            scope = per_model if exc.status is not None else ""
            self.ledger.block(
                entry.name,
                scope,
                until=now + UNAVAILABLE_COOLDOWN_S,
                kind="cooldown",
                why=exc.message,
            )
        log_event(
            _logger,
            "infer.error",
            f"{entry.name} failed for {model.id}: {exc.code}",
            provider=entry.name,
            model=model.id,
            code=str(exc.code),
        )

    def _result(
        self,
        req: InferRequest,
        entry: InferenceEntry,
        model: ModelEntry,
        reply: ChatReply,
        decision: InferRoute,
        fallbacks: list[InferAttempt],
    ) -> InferResult:
        need = estimate_need(entry, model, req)
        in_tok = reply.input_tokens if reply.input_tokens is not None else need.input_tokens
        out_tok = reply.output_tokens
        if out_tok is None:
            out_tok = max(1, len(reply.text) // 4)
        neurons = entry.neurons(model, in_tok, out_tok)
        usd = entry.usd(model, in_tok, out_tok)
        amounts: dict[str, float] = {"requests": 1.0, "tokens": float(in_tok + out_tok)}
        if neurons is not None:
            amounts["neurons"] = neurons
        if usd is not None:
            amounts["usd"] = usd
        self.ledger.record(entry.name, model.id, amounts, reply.live)
        log_event(
            _logger,
            "infer.call",
            f"{entry.name} answered {model.id} ({in_tok}+{out_tok} tokens)",
            provider=entry.name,
            model=model.id,
            input_tokens=in_tok,
            output_tokens=out_tok,
            latency_s=round(reply.latency_s, 3),
            fallbacks=len(fallbacks),
        )
        return InferResult(
            provider=entry.name,
            model=req.model,
            model_id=model.id,
            text=reply.text,
            finish_reason=reply.finish_reason,
            usage=InferUsage(
                input_tokens=reply.input_tokens,
                output_tokens=reply.output_tokens,
                neurons=round(neurons, 3) if neurons is not None else None,
                usd=round(usd, 6) if usd is not None else None,
            ),
            latency_s=round(reply.latency_s, 3),
            route_reason=decision.reason,
            fallbacks=fallbacks,
            quota=self.quota_view(entry).summary,
        )

    def _no_route_error(
        self,
        req: InferRequest,
        decision: InferRoute,
        fallbacks: list[InferAttempt],
        last_error: InferenceError | None,
        first: InferRoute | None = None,
    ) -> Exception:
        shown = decision
        if fallbacks and first is not None and not decision.candidates and not decision.rejected:
            # the re-route skipped every provider that failed, so its "nobody serves it"
            # would mislead: show the decision that chose them
            shown = first
        detail: dict[str, object] = {
            "model": req.model,
            "route": shown.model_dump(mode="json"),
        }
        if fallbacks:
            detail["fallbacks"] = [f.model_dump() for f in fallbacks]
            tried = "; ".join(f"{f.provider}: {f.message}" for f in fallbacks)
            reason = f"every free provider for {req.model} failed: {tried}"
        else:
            reason = decision.reason
        codes = {r.code for r in decision.rejected}
        if not decision.rejected and not fallbacks:
            return InvalidRequest(reason, detail=detail)
        if decision.outcome == "no_fit" and codes <= {"no_key"} and not fallbacks:
            logins = sorted(
                {self.catalog.entries[r.provider].login_name for r in decision.rejected}
            )
            return InferAuthRequired(
                reason,
                hint="store a key with " + " or ".join(f"`gpu login {n}`" for n in logins),
                detail=detail,
            )
        if decision.retry_at is not None:
            now = self.clock.now()
            if codes <= {"exhausted", "quota", "no_key"}:
                return InferQuotaExhausted(
                    reason,
                    resets_at=decision.retry_at,
                    hint=f"the earliest reset is {when(decision.retry_at, now)}",
                    detail=detail,
                )
            return InferRateLimited(
                reason,
                retry_after=max(0.0, decision.retry_at - now),
                hint="try again shortly, or pass a longer wait",
                detail=detail,
            )
        if last_error is not None and not decision.candidates:
            last_error.detail.update(detail)
            if fallbacks:
                last_error.message = reason
                last_error.args = (reason,)
            return last_error
        return InferUnavailable(reason, detail=detail)


class _Busy(InferUnavailable):
    """All of a provider's slots stayed taken (our own limit, not the provider's): try the
    next provider, nothing goes into the ledger."""
