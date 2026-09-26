"""The daemon's Notifier (phase 8a): classify on the bus, format off the commit, send on a
worker thread; dedupe, rate limit, never block. Backends are recorders or slow fakes."""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

from gpu_router.clock import FakeClock
from gpu_router.models import Job
from gpu_router.notify.backends import NotifyError, NullBackend
from gpu_router.notify.format import Kind, Notification
from gpu_router.notify.service import Notifier
from gpu_router.notify.settings import NotifySettings
from gpu_router.statemachine import JobState, Reason
from tests.unit.notify.test_format import JOB_ID, ev, job


class Slow:
    name = "slow"

    def __init__(self, delay: float) -> None:
        self.delay = delay
        self.sent: list[Notification] = []

    def send(self, n: Notification) -> None:
        time.sleep(self.delay)
        self.sent.append(n)


class Failing:
    name = "failing"

    def __init__(self) -> None:
        self.calls = 0

    def send(self, n: Notification) -> None:
        self.calls += 1
        raise NotifyError("osascript exited 1: Not authorized")


def make(
    settings: NotifySettings | None = None,
    backend: Any = None,
    jobs: dict[str, Job] | None = None,
    clock: FakeClock | None = None,
) -> tuple[Notifier, Any, list[str]]:
    lookups: list[str] = []
    table = jobs if jobs is not None else {JOB_ID: job()}

    def lookup(job_id: str) -> Job | None:
        lookups.append(job_id)
        return table.get(job_id)

    b = backend if backend is not None else NullBackend(why="test")
    n = Notifier(settings or NotifySettings(), b, job_lookup=lookup, clock=clock or FakeClock())
    return n, b, lookups


async def settle() -> None:
    for _ in range(3):
        await asyncio.sleep(0)


async def test_on_change_never_reads_the_store_and_formats_on_the_next_loop_turn() -> None:
    n, backend, lookups = make()
    n.on_change(JOB_ID, [ev(JobState.DONE, Reason.COMPLETED)])
    assert lookups == []  # the commit callback only classified
    await settle()
    assert lookups == [JOB_ID]
    assert n.wait_idle()
    assert [x.subtitle for x in backend.sent] == ["✓ train_yolo finished"]
    n.close()


async def test_irrelevant_and_switched_off_events_do_nothing() -> None:
    settings = NotifySettings.model_validate({"events": {"migrated": False}})
    n, backend, lookups = make(settings)
    n.on_change(JOB_ID, [ev(JobState.RUNNING, Reason.STARTED, frm=JobState.PROVISIONING)])
    n.on_change(JOB_ID, [ev(JobState.MIGRATING, Reason.SESSION_LOST)])
    n.on_change(JOB_ID, [])
    await settle()
    assert lookups == []
    assert backend.sent == []
    n.close()


async def test_terminal_notifications_once_per_job() -> None:
    n, backend, _ = make()
    for _ in range(3):
        n.on_change(JOB_ID, [ev(JobState.DONE, Reason.COMPLETED)])
        await settle()
    assert n.wait_idle()
    assert len(backend.sent) == 1
    assert n.stats["deduped"] == 2
    n.close()


async def test_migrations_collapse_within_the_window_then_notify_again() -> None:
    clock = FakeClock()
    n, backend, _ = make(NotifySettings(dedupe_s=600), clock=clock)
    moved = ev(JobState.MIGRATING, Reason.SESSION_LOST, detail={"previous_provider": "colab"})
    n.on_change(JOB_ID, [moved])
    await settle()
    clock.advance(120)
    n.on_change(JOB_ID, [moved])  # flapping: held back
    await settle()
    clock.advance(600)
    n.on_change(JOB_ID, [moved])
    await settle()
    assert n.wait_idle()
    assert len(backend.sent) == 2
    n.close()


async def test_a_reask_on_another_provider_is_a_new_question() -> None:
    table = {JOB_ID: job(state=JobState.AWAITING_APPROVAL, provider="colab", approval_reason="r1")}
    n, backend, _ = make(jobs=table)
    n.on_change(JOB_ID, [ev(JobState.AWAITING_APPROVAL, Reason.APPROVAL_REQUIRED)])
    await settle()
    n.on_change(JOB_ID, [ev(JobState.AWAITING_APPROVAL, Reason.APPROVAL_REQUIRED)])  # dup
    await settle()
    table[JOB_ID] = job(state=JobState.AWAITING_APPROVAL, provider="kaggle", approval_reason="r2")
    n.on_change(JOB_ID, [ev(None, Reason.APPROVAL_REQUIRED, kind="note", detail={"reask": True})])
    await settle()
    assert n.wait_idle()
    assert [x.body.split(" · ")[0] for x in backend.sent] == ["→ colab 2xT4", "→ kaggle 2xT4"]
    n.close()


async def test_rate_limit_folds_the_rest_into_one_summary() -> None:
    clock = FakeClock()
    jobs = {f"{i:012x}": job(id=f"{i:012x}", short_id=f"{i:04x}") for i in range(10)}
    n, backend, _ = make(NotifySettings(max_per_minute=3), jobs=jobs, clock=clock)
    for jid in jobs:
        e = ev(JobState.DONE, Reason.COMPLETED).model_copy(update={"job_id": jid})
        n.on_change(jid, [e])
    await settle()
    assert n.wait_idle()
    assert len(backend.sent) == 3
    assert n.stats["rate_limited"] == 7
    clock.advance(61)
    held = n.flush_summary()
    assert held is not None
    assert held.kind == Kind.SUMMARY
    assert held.subtitle == "7 more job updates"
    assert n.wait_idle()
    assert backend.sent[-1].kind == Kind.SUMMARY
    assert n.flush_summary() is None  # nothing left
    n.close()


async def test_a_slow_backend_never_blocks_the_event_loop() -> None:
    backend = Slow(delay=0.5)
    jobs = {f"{i:012x}": job(id=f"{i:012x}") for i in range(4)}
    n, _, _ = make(NotifySettings(max_per_minute=60), backend=backend, jobs=jobs)
    t0 = time.monotonic()
    for jid in jobs:
        e = ev(JobState.DONE, Reason.COMPLETED).model_copy(update={"job_id": jid})
        n.on_change(jid, [e])
    await settle()
    assert time.monotonic() - t0 < 0.2  # 4 x 0.5 s of sending happens elsewhere
    assert n.wait_idle(5)
    assert len(backend.sent) == 4
    n.close()


async def test_a_full_queue_drops_instead_of_waiting() -> None:
    gate = threading.Event()

    class Stuck:
        name = "stuck"
        sent: list[Notification] = []

        def send(self, n: Notification) -> None:
            gate.wait(5)

    jobs = {f"{i:012x}": job(id=f"{i:012x}") for i in range(8)}
    clock = FakeClock()
    n = Notifier(
        NotifySettings(max_per_minute=60),
        Stuck(),
        job_lookup=jobs.get,
        clock=clock,
        queue_size=2,
    )
    t0 = time.monotonic()
    for jid in jobs:
        e = ev(JobState.DONE, Reason.COMPLETED).model_copy(update={"job_id": jid})
        n.on_change(jid, [e])
    await settle()
    assert time.monotonic() - t0 < 0.5
    assert n.stats["dropped"] >= 1
    gate.set()
    n.close()


async def test_backend_failures_are_counted_and_never_raise() -> None:
    backend = Failing()
    jobs = {f"{i:012x}": job(id=f"{i:012x}") for i in range(3)}
    n, _, _ = make(backend=backend, jobs=jobs)
    for jid in jobs:
        e = ev(JobState.FAILED, Reason.SCRIPT_FAILED).model_copy(update={"job_id": jid})
        n.on_change(jid, [e])
    await settle()
    assert n.wait_idle()
    assert backend.calls == 3
    assert n.stats["failed"] == 3
    n.close()


def test_without_an_event_loop_the_event_alone_is_enough() -> None:
    n, backend, lookups = make()
    n.on_change(JOB_ID, [ev(JobState.FAILED, Reason.GAVE_UP, message="gave up")])
    assert lookups == []  # never reads the store synchronously
    assert n.wait_idle()
    assert backend.sent[0].subtitle == "✗ job a7f2 failed"
    n.close()


async def test_close_stops_taking_events() -> None:
    n, backend, _ = make()
    n.close()
    n.on_change(JOB_ID, [ev(JobState.DONE, Reason.COMPLETED)])
    await settle()
    assert backend.sent == []
    n.close()  # idempotent
