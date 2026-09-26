"""JSONL eval batches (inference/batch.py)."""

from __future__ import annotations

from typing import Any

from gpu_router.errors import ApiError, InvalidRequest
from gpu_router.inference.batch import parse_lines, run_batch
from gpu_router.inference.models import InferRequest, InferResult, InferUsage


def result(req: InferRequest, provider: str = "groq") -> InferResult:
    return InferResult(
        provider=provider,
        model=req.model,
        model_id=f"id/{req.model}",
        text=f"answer to {req.chat()[-1].content}",
        usage=InferUsage(input_tokens=5, output_tokens=2),
        latency_s=0.1,
        route_reason="r",
    )


def test_parse_lines_overrides_and_meta() -> None:
    lines = [
        '{"id": 7, "prompt": "a", "expected": "b", "tags": ["x"]}',
        '{"messages": [{"role": "user", "content": "hi"}], "model": "other", "temperature": 0}',
        '{"prompt": "p", "system": "own system"}',
        "[1, 2]",
        '{"prompt": ""}',
    ]
    items = list(parse_lines(lines, model="m", system="flag system", max_tokens=9))
    first, second, third, arr, empty = items
    assert first.id == "7"
    assert first.meta == {"expected": "b", "tags": ["x"]}
    assert first.request is not None
    assert first.request.system == "flag system"
    assert first.request.max_tokens == 9
    assert second.request is not None
    assert (second.request.model, second.request.system, second.request.temperature) == (
        "other",
        None,  # messages carry their own system turn
        0,
    )
    assert third.request is not None
    assert third.request.system == "own system"
    assert arr.error == "not a JSON object"
    assert empty.request is None
    assert "prompt is empty" in (empty.error or "")
    [no_model] = parse_lines(['{"prompt": "x"}'], model=None)
    assert no_model.error
    assert "no model" in no_model.error


def test_run_batch_counts_and_stops_after_repeated_fatal_errors() -> None:
    items = list(parse_lines([f'{{"prompt": "q{i}"}}' for i in range(7)], model="m"))
    calls: list[str] = []

    def call(req: InferRequest) -> InferResult:
        calls.append(req.prompt or "")
        if req.prompt == "q0":
            return result(req)
        if req.prompt == "q1":
            raise InvalidRequest("too long")  # not fatal: the next line still goes
        raise ApiError("quota_exhausted", "every provider is used up", status=429)

    out: list[dict[str, Any]] = []
    summary = run_batch(items, call, out.append)
    assert [r["ok"] for r in out] == [True] + [False] * 6
    assert calls == ["q0", "q1", "q2", "q3", "q4"]  # 3 fatal in a row, then stop
    assert out[-1]["error"]["code"] == "not_sent"
    assert (summary.answered, summary.failed, summary.not_sent) == (1, 4, 2)
    assert summary.by_provider == {"groq": 1}
    assert summary.line() == "7 prompts: 1 answered (groq 1), 4 failed, 2 not sent"
    assert summary.as_dict()["stopped"] == "every provider is used up"
    assert out[0]["output"] == "answer to q0"
    assert out[0]["usage"] == {
        "input_tokens": 5,
        "output_tokens": 2,
    }
