"""JSONL eval batches for the inference lane (phase 7b): `gpu infer --file evals.jsonl`
and the shell's `/infer --file`.

Input: one JSON object per line; blank lines and lines starting with `#` are skipped.

    {"id": "q1", "prompt": "2+2?", "expected": "4"}
    {"id": "q2", "messages": [{"role": "user", "content": "hi"}], "model": "gemini-3.5-flash"}

`prompt` or `messages` is required; `system`, `model`, `provider`, `max_tokens` and
`temperature` override the command's flags for that line; every other key (`expected`,
`tags`, ...) is copied into the result under `meta`, so a scorer can read one file.

Output: one result per input line, in order:

    {"id", "line", "ok", "provider", "model", "model_id", "output", "usage", "latency_s",
     "fallbacks", "error": {code, message, hint} | null, "meta": {...}}

The batch stops early after STOP_AFTER consecutive failures that no retry will fix soon
(every provider's day used up, no key): the rest are reported as not sent.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from gpu_router.errors import GpuRouterError
from gpu_router.inference.models import InferRequest, InferResult

__all__ = ["STOP_AFTER", "BatchItem", "BatchSummary", "parse_lines", "run_batch"]

STOP_AFTER = 3
_REQUEST_KEYS = frozenset(
    {"prompt", "messages", "system", "model", "provider", "max_tokens", "temperature"}
)
_FATAL_CODES = frozenset({"quota_exhausted", "auth_required"})


@dataclass(frozen=True)
class BatchItem:
    line: int
    id: str
    request: InferRequest | None
    error: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


def parse_lines(
    lines: Iterable[str],
    *,
    model: str | None,
    provider: str | None = None,
    system: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> Iterator[BatchItem]:
    """One BatchItem per non-blank, non-comment line (invalid lines carry `error`)."""
    for n, raw in enumerate(lines, start=1):
        text = raw.strip()
        if not text or text.startswith("#"):
            continue
        try:
            doc = json.loads(text)
        except json.JSONDecodeError as exc:
            yield BatchItem(line=n, id=str(n), request=None, error=f"not valid JSON: {exc.msg}")
            continue
        if not isinstance(doc, dict):
            yield BatchItem(line=n, id=str(n), request=None, error="not a JSON object")
            continue
        item_id = str(doc.get("id", n))
        meta = {k: v for k, v in doc.items() if k not in _REQUEST_KEYS and k != "id"}
        body: dict[str, Any] = {
            "model": doc.get("model", model),
            "provider": doc.get("provider", provider),
            "system": doc.get("system", system),
            "max_tokens": doc.get("max_tokens", max_tokens),
            "temperature": doc.get("temperature", temperature),
        }
        if "messages" in doc:
            body["messages"] = doc["messages"]
            body.pop("system")
        else:
            body["prompt"] = doc.get("prompt")
        if body["model"] is None:
            yield BatchItem(n, item_id, None, "no model (give --model or a `model` key)", meta)
            continue
        try:
            req = InferRequest.model_validate({k: v for k, v in body.items() if v is not None})
        except ValidationError as exc:
            first = exc.errors()[0]
            where = ".".join(str(p) for p in first["loc"])
            msg = f"{where}: {first['msg']}" if where else str(first["msg"])
            yield BatchItem(n, item_id, None, msg, meta)
            continue
        yield BatchItem(line=n, id=item_id, request=req, meta=meta)


def _record(item: BatchItem) -> dict[str, Any]:
    return {
        "id": item.id,
        "line": item.line,
        "ok": False,
        "provider": None,
        "model": item.request.model if item.request is not None else None,
        "model_id": None,
        "output": None,
        "usage": None,
        "latency_s": None,
        "fallbacks": [],
        "error": None,
        "meta": item.meta,
    }


@dataclass
class BatchSummary:
    total: int = 0
    answered: int = 0
    failed: int = 0
    not_sent: int = 0
    by_provider: Counter[str] = field(default_factory=Counter)
    input_tokens: int = 0
    output_tokens: int = 0
    stopped: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "answered": self.answered,
            "failed": self.failed,
            "not_sent": self.not_sent,
            "by_provider": dict(self.by_provider),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "stopped": self.stopped,
        }

    def line(self) -> str:
        where = ", ".join(f"{p} {n}" for p, n in self.by_provider.most_common())
        text = f"{self.total} prompts: {self.answered} answered"
        if where:
            text += f" ({where})"
        if self.failed:
            text += f", {self.failed} failed"
        if self.not_sent:
            text += f", {self.not_sent} not sent"
        return text


def run_batch(
    items: Iterable[BatchItem],
    call: Callable[[InferRequest], InferResult],
    emit: Callable[[dict[str, Any]], None],
    *,
    stop_after: int = STOP_AFTER,
) -> BatchSummary:
    """Send every item through `call`, in order, emitting one result record each."""
    summary = BatchSummary()
    fatal_streak = 0
    stop_reason: str | None = None
    for item in items:
        summary.total += 1
        rec = _record(item)
        if stop_reason is not None:
            summary.not_sent += 1
            rec["error"] = {"code": "not_sent", "message": f"not sent: {stop_reason}"}
            emit(rec)
            continue
        if item.request is None:
            summary.failed += 1
            rec["error"] = {"code": "invalid_request", "message": f"line {item.line}: {item.error}"}
            emit(rec)
            continue
        try:
            result = call(item.request)
        except GpuRouterError as exc:
            body = exc.to_body()
            code = str(getattr(exc, "raw_code", None) or body["code"])
            summary.failed += 1
            rec["error"] = {"code": code, "message": exc.message, "hint": exc.hint}
            emit(rec)
            fatal_streak = fatal_streak + 1 if code in _FATAL_CODES else 0
            if fatal_streak >= stop_after:
                stop_reason = exc.message
                summary.stopped = exc.message
            continue
        fatal_streak = 0
        summary.answered += 1
        summary.by_provider[result.provider] += 1
        summary.input_tokens += result.usage.input_tokens or 0
        summary.output_tokens += result.usage.output_tokens or 0
        rec.update(
            ok=True,
            provider=result.provider,
            model=result.model,
            model_id=result.model_id,
            output=result.text,
            usage=result.usage.model_dump(exclude_none=True),
            latency_s=result.latency_s,
            fallbacks=[f.model_dump() for f in result.fallbacks],
        )
        emit(rec)
    return summary
