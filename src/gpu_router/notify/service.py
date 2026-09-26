"""The daemon's notifier (phase 8a): event bus -> macOS notifications, never blocking.

Flow (spec UX 5; CLAUDE.md "Notifications and doctor (phase 8a)"):

1. `on_change(job_id, events)` is an EventBus subscriber: it runs on the event-loop thread
   right after a COMMIT and may not read the Store (events.py), so it only classifies the
   events (`format.classify`, O(1) each) and schedules `_drain` with `loop.call_soon`.
2. `_drain` runs on the loop thread outside the commit: reads the job (one indexed SELECT),
   drops duplicates, applies the rate limit, formats, and hands the notification to a
   bounded queue with `put_nowait` (a full queue drops it: counted, never waited on).
3. One daemon worker thread takes notifications off the queue and runs the backend
   (osascript / terminal-notifier, each bounded by `timeout_s`). A failing backend is
   logged (`notify.failed`, first failure then every 20th) and never retried in a loop.

Dedupe: a finished/failed notification once per job, ever; approval once per (job,
provider, reason) within `dedupe_s` (a re-ask on another provider is a new question,
D43); migrated once per job within `dedupe_s` (a flapping job does not spam).
Rate limit: at most `max_per_minute` in any 60 s; the rest are counted and folded into one
"N more job updates" notification when the window frees.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import queue
import threading
from collections import Counter, OrderedDict, deque
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from gpu_router.log import get_logger, log_event
from gpu_router.notify.backends import Backend, NotifyError, choose_backend, describe
from gpu_router.notify.format import Kind, Notification, build, classify, summary
from gpu_router.notify.settings import NotifySettings, notify_settings

if TYPE_CHECKING:
    from gpu_router.clock import Clock
    from gpu_router.config import Config
    from gpu_router.daemon.events import EventBus
    from gpu_router.models import Job, JobEvent
    from gpu_router.store import Store

__all__ = ["Notifier", "attach_notifier"]

logger = get_logger("gpu_router.notify")

RATE_WINDOW_S = 60.0
QUEUE_SIZE = 64
MAX_KEYS = 2000  # dedupe memory (oldest forgotten first)
FAIL_LOG_EVERY = 20

_Key = tuple[str, ...]


def _emit(event: str, msg: str, *, level: int = logging.INFO, **fields: Any) -> None:
    # logging must never break the notifier (or the engine behind it)
    with contextlib.suppress(Exception):
        log_event(logger, event, msg, level=level, **fields)


class Notifier:
    def __init__(
        self,
        settings: NotifySettings,
        backend: Backend,
        *,
        job_lookup: Callable[[str], Job | None],
        clock: Clock,
        user_home: str | None = None,
        queue_size: int = QUEUE_SIZE,
    ) -> None:
        self.settings = settings
        self.backend = backend
        self._lookup = job_lookup
        self._clock = clock
        self._user_home = user_home
        self._pending: list[tuple[str, JobEvent, Kind]] = []
        self._scheduled = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._queue: queue.Queue[Notification | None] = queue.Queue(maxsize=queue_size)
        self._worker: threading.Thread | None = None
        self._worker_lock = threading.Lock()
        self._last: OrderedDict[_Key, float] = OrderedDict()
        self._times: deque[float] = deque()
        self._held: Counter[str] = Counter()
        self._summary_handle: asyncio.TimerHandle | None = None
        self._closed = False
        self.unsubscribe: Callable[[], None] | None = None
        self.stats: Counter[str] = Counter()

    # ------------------------------------------------------------------ bus side

    def on_change(self, job_id: str, events: Sequence[JobEvent]) -> None:
        """EventBus subscriber. Classifies only; never reads the Store, never blocks."""
        if self._closed or not events:
            return
        for ev in events:
            kind = classify(ev)
            if kind is not None and self.settings.wants(kind):
                self._pending.append((job_id, ev, kind))
        if not self._pending or self._scheduled:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is None:
            # no event loop (a synchronous caller): format from the event alone rather
            # than read the Store inside its commit callback
            self._drain(lookup=False)
            return
        self._loop = loop
        self._scheduled = True
        loop.call_soon(self._drain)

    def _drain(self, lookup: bool = True) -> None:
        self._scheduled = False
        pending, self._pending = self._pending, []
        if self._closed:
            return
        now = self._clock.now()
        for job_id, ev, kind in pending:
            job: Job | None = None
            if lookup:
                try:
                    job = self._lookup(job_id)
                except Exception:
                    job = None
            key, window = self._dedupe_key(kind, job_id, job)
            last = self._last.get(key)
            if last is not None and (window is None or now - last < window):
                self.stats["deduped"] += 1
                continue
            self._remember(key, now)
            try:
                note = build(kind, job, ev, sound=self.settings.sound, user_home=self._user_home)
            except Exception:
                self.stats["bugs"] += 1
                _emit(
                    "notify.bug",
                    f"could not format a {kind} notification",
                    level=logging.WARNING,
                    job_id=job_id,
                )
                continue
            self._offer(note, now)

    def _dedupe_key(self, kind: Kind, job_id: str, job: Job | None) -> tuple[_Key, float | None]:
        """(key, window in seconds; None = forever)."""
        if kind in (Kind.FINISHED, Kind.FAILED):
            return (job_id, "terminal"), None
        if kind is Kind.APPROVAL:
            provider = (job.provider or "") if job is not None else ""
            reason = (job.approval_reason or "") if job is not None else ""
            return (job_id, "approval", provider, reason), self.settings.dedupe_s
        return (job_id, str(kind)), self.settings.dedupe_s

    def _remember(self, key: _Key, now: float) -> None:
        self._last[key] = now
        self._last.move_to_end(key)
        while len(self._last) > MAX_KEYS:
            self._last.popitem(last=False)

    # ------------------------------------------------------------------ rate limit

    def _offer(self, note: Notification, now: float) -> None:
        while self._times and now - self._times[0] >= RATE_WINDOW_S:
            self._times.popleft()
        if len(self._times) >= self.settings.max_per_minute:
            self._held[note.kind] += 1
            self.stats["rate_limited"] += 1
            self._schedule_summary(now)
            return
        self._times.append(now)
        self._enqueue(note)

    def _schedule_summary(self, now: float) -> None:
        if self._summary_handle is not None or self._loop is None:
            return
        delay = max(0.5, RATE_WINDOW_S - (now - self._times[0])) if self._times else 0.5
        with contextlib.suppress(RuntimeError):  # loop closed
            self._summary_handle = self._loop.call_later(delay, self.flush_summary)

    def flush_summary(self) -> Notification | None:
        """Send one notification for everything the rate limit held back (if any)."""
        self._summary_handle = None
        if self._closed or not self._held:
            return None
        note = summary(self._held)
        self._held = Counter()
        self._times.append(self._clock.now())
        self._enqueue(note)
        return note

    # ------------------------------------------------------------------ worker

    def _enqueue(self, note: Notification) -> None:
        try:
            self._queue.put_nowait(note)
        except queue.Full:
            self.stats["dropped"] += 1
            return
        self._ensure_worker()

    def _ensure_worker(self) -> None:
        with self._worker_lock:
            if self._worker is not None and self._worker.is_alive():
                return
            self._worker = threading.Thread(target=self._work, name="gpu-notify", daemon=True)
            self._worker.start()

    def _work(self) -> None:
        while True:
            note = self._queue.get()
            if note is None:
                return
            try:
                self.backend.send(note)
            except NotifyError as exc:
                self.stats["failed"] += 1
                if self.stats["failed"] % FAIL_LOG_EVERY == 1:
                    _emit(
                        "notify.failed",
                        f"could not show a notification ({exc}); jobs are not affected",
                        level=logging.WARNING,
                        backend=self.backend.name,
                        kind=note.kind,
                        failures=self.stats["failed"],
                    )
            except Exception:
                self.stats["failed"] += 1
                _emit(
                    "notify.bug",
                    "notification backend raised",
                    level=logging.WARNING,
                    backend=self.backend.name,
                )
            else:
                self.stats["sent"] += 1
                _emit(
                    "notify.sent",
                    f"{note.kind}: {note.subtitle}",
                    job_id=note.job_id,
                    kind=note.kind,
                    backend=self.backend.name,
                )
            finally:
                self._queue.task_done()

    def wait_idle(self, timeout_s: float = 5.0) -> bool:
        """Tests: wait until the worker has sent everything queued so far."""
        done = threading.Event()

        def _join() -> None:
            self._queue.join()
            done.set()

        threading.Thread(target=_join, daemon=True).start()
        return done.wait(timeout_s)

    def close(self) -> None:
        """Stop taking events; the worker ends after what is queued (daemon thread: a stuck
        backend never holds up a daemon shutdown)."""
        if self._closed:
            return
        self._closed = True
        if self.unsubscribe is not None:
            self.unsubscribe()
        if self._summary_handle is not None:
            self._summary_handle.cancel()
            self._summary_handle = None
        with contextlib.suppress(queue.Full):
            self._queue.put_nowait(None)
        worker = self._worker
        if worker is not None:
            worker.join(timeout=0.5)


def attach_notifier(bus: EventBus, store: Store, config: Config, clock: Clock) -> Notifier:
    """Build the daemon's notifier from config.yaml `notifications:` and subscribe it to
    the bus. Raises ConfigError on a bad section (daemon start fails loudly)."""
    import os

    from gpu_router.errors import GpuRouterError

    settings = notify_settings(getattr(config, "notifications", None))
    backend = choose_backend(settings, test_mode=config.test_mode)

    def lookup(job_id: str) -> Job | None:
        try:
            return store.get_job(job_id)
        except GpuRouterError:
            return None

    notifier = Notifier(
        settings,
        backend,
        job_lookup=lookup,
        clock=clock,
        user_home=os.path.expanduser("~"),
    )
    notifier.unsubscribe = bus.subscribe(notifier.on_change)
    # which backend this process picked (a launchd daemon's PATH differs from a shell's)
    _emit("notify.backend", f"notifications: {describe(backend)}", backend=backend.name)
    return notifier
