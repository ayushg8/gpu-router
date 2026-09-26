"""Contract: logs() cursors (rule A7) and the ::gpu:: protocol lines. Owner: group B."""

from __future__ import annotations

from gpu_router.adapters.base import Adapter, LogChunk, RemotePhase
from gpu_router.protocol import parse_line
from tests.contract.harness import ContractTarget, make_ctx, make_job

TERMINAL = {p for p in RemotePhase if p.terminal}


def _drain(adapter: Adapter, ref: object, since: str | None) -> list[LogChunk]:
    return list(adapter.logs(ref, follow=False, since=since))  # type: ignore[arg-type]


def test_cursor_resume_never_repeats_lines(target: ContractTarget, adapter: Adapter) -> None:
    job = make_job(target)
    ref = adapter.submit(job, make_ctx(job))
    seen: list[str] = []
    cursor: str | None = None
    for _ in range(200):
        chunks = _drain(adapter, ref, cursor)
        for c in chunks:
            seen.extend(c.lines)
            cursor = c.cursor
        if adapter.status(ref).phase in TERMINAL and chunks and chunks[-1].eof:
            break
        target.advance(target.step_s)
    full: list[str] = []
    for c in _drain(adapter, ref, None):
        full.extend(c.lines)
    assert seen == full
    assert seen, "a finished run produced no log lines"


def test_logs_after_eof_are_empty(target: ContractTarget, adapter: Adapter) -> None:
    job = make_job(target)
    ref = adapter.submit(job, make_ctx(job))
    target.wait_for_phase(adapter, ref, TERMINAL)
    chunks = _drain(adapter, ref, None)
    last = chunks[-1]
    assert last.eof
    again = _drain(adapter, ref, last.cursor)
    assert all(not c.lines for c in again)


def test_fake_emits_metric_and_checkpoint_lines(target: ContractTarget, adapter: Adapter) -> None:
    if not target.supports_directives:
        return
    job = make_job(target, directives={"duration": 10, "steps": 10, "checkpoint_every": 3})
    ref = adapter.submit(job, make_ctx(job))
    target.wait_for_phase(adapter, ref, TERMINAL)
    events = [
        e
        for c in _drain(adapter, ref, None)
        for line in c.lines
        if (e := parse_line(line)) is not None
    ]
    kinds = {e.t for e in events}
    assert {"total", "metric", "ckpt_begin", "ckpt_end"} <= kinds
