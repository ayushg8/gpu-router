"""Background poller behind the job panel and the footer (phase 4).

`Feed.run(stop)` runs in a worker thread and publishes an immutable `Snapshot` through the
`publish` callback after every poll:

- connect: `daemon.spawn.connect` (the CLI's auto-start) on the first try, then plain
  reconnects every `retry_s` while the daemon is down (a failed start is not retried in a
  loop; any command the user types tries to start it again);
- status: `GET /v1/status` every `poll_s` (active jobs, recent, providers);
- quota: `GET /v1/quota` every `quota_s` on its own thread (providers may answer slowly;
  Kaggle's live quota runs its CLI), merged into the providers' quota when newer;
- metrics: for each running job whose row changed, the new lines of its current attempt's
  log with `protocol=true` (D42), parsed by `MetricHistory`;
- checkpoints: when a job's checkpoint count grows, `GET /v1/jobs/{id}` once for the
  latest checkpoint's URI ("HF Hub", "this Mac", the provider's disk).

Only the HTTP API is used (invariant 2); the feed never opens SQLite.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Callable
from dataclasses import dataclass, field

from gpu_router.api import JobView, ProviderView
from gpu_router.client import GpuClient
from gpu_router.clock import Clock, SystemClock
from gpu_router.errors import DaemonUnavailable, GpuRouterError
from gpu_router.models import QuotaSnapshot
from gpu_router.paths import Paths
from gpu_router.shell.metrics import MetricHistory
from gpu_router.shell.state import Conn, ConnStatus, JobMetric, Snapshot
from gpu_router.statemachine import JobState

PANEL_POINTS = 48  # sparkline history handed to the panel per job
MAX_TRACKED = 12  # running jobs whose logs are parsed for the panel
FIRST_READ_LINES = 5000  # a job seen for the first time: only its log's tail (D44)
STATUS_TIMEOUT_S = 5.0

_RUNNING = (JobState.RUNNING, JobState.CHECKPOINTING)


def attempt_number(job: JobView) -> int | None:
    """n of the job's current attempt ("<job>.<n>"), or None."""
    if not job.current_attempt_id:
        return None
    try:
        return int(job.current_attempt_id.rsplit(".", 1)[1])
    except (IndexError, ValueError):
        return None


#: runner/storage.py's checkpoint key, `jobs/<job id>/ckpt-NNNN` (phase 5 storage, D40)
_STORED = re.compile(r"/jobs/[0-9a-f]{12}/ckpt-\d+/?$")


def checkpoint_where(uri: str, provider: str | None) -> str:
    """Where a checkpoint lives, for "ckpt 3m ago → HF Hub": the HF bucket, the local
    storage backend on this Mac, else (no storage, phase 3) the machine that ran it."""
    if uri.startswith("hf://"):
        return "HF Hub"
    if uri.startswith("file://"):
        if _STORED.search(uri):
            return "local storage"
        return "this Mac" if provider in (None, "local") else f"{provider} disk"
    scheme = uri.split("://", 1)[0] if "://" in uri else ""
    return scheme or (provider or "")


def read_new_lines(client: GpuClient, job_id: str, n: int, hist: MetricHistory) -> bool:
    """Feed the attempt's log lines after `hist.positions[n]` into `hist`."""
    changed = False
    for rec in client.logs(job_id, attempt=n, offset=hist.positions.get(n, 0), protocol=True):
        if rec.line is None or rec.offset is None:
            continue
        changed = hist.feed(rec.line) or changed
        hist.positions[n] = rec.offset + 1
    return changed


def metric_view(hist: MetricHistory) -> JobMetric:
    name = hist.primary()
    values = tuple(hist.values(name)[-PANEL_POINTS:]) if name else ()
    return JobMetric(name=name, values=values, step=hist.step, total=hist.total)


def merge_quota(
    providers: tuple[ProviderView, ...], quotas: dict[str, QuotaSnapshot]
) -> tuple[ProviderView, ...]:
    """Providers with the newer of their own quota and the latest /v1/quota answer."""
    out: list[ProviderView] = []
    for p in providers:
        q = quotas.get(p.name)
        if q is not None and (p.quota is None or q.observed_at >= p.quota.observed_at):
            p = p.model_copy(update={"quota": q})
        out.append(p)
    return tuple(out)


@dataclass
class Feed:
    paths: Paths
    publish: Callable[[Snapshot], None]
    clock: Clock = field(default_factory=SystemClock)
    poll_s: float = 1.0
    quota_s: float = 60.0
    retry_s: float = 3.0
    autostart: bool = True
    _hist: dict[str, MetricHistory] = field(default_factory=dict)
    _seen: dict[str, tuple[float, int]] = field(default_factory=dict)  # id -> (updated, ver)
    _ckpt: dict[str, tuple[int, str]] = field(default_factory=dict)  # id -> (count, where)
    _quota: dict[str, QuotaSnapshot] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _wake: threading.Event = field(default_factory=threading.Event)
    _stop: threading.Event = field(default_factory=threading.Event)
    _last: Snapshot | None = None
    started_pid: int | None = None

    # ------------------------------------------------------------------ public

    def poke(self) -> None:
        """Poll now (after a command changed something) instead of at the next tick."""
        self._wake.set()

    def stop(self) -> None:
        """Ask `run` (and the quota thread) to return; they check at every wait."""
        self._stop.set()
        self._wake.set()

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    @property
    def last(self) -> Snapshot | None:
        return self._last

    def run(self) -> None:
        """Blocking loop; returns soon after `stop()`."""
        stop = self._stop
        client: GpuClient | None = None
        quota_thread: threading.Thread | None = None
        first = True
        last_ok: float | None = None
        while not stop.is_set():
            if client is None:
                try:
                    client = self._connect(start=first)
                except GpuRouterError as exc:
                    self._emit(
                        Snapshot(
                            conn=Conn(
                                ConnStatus.DOWN,
                                message=exc.message,
                                hint=exc.hint or "",
                                retry_s=self.retry_s,
                                last_ok=last_ok,
                                autostart=self._autostart_on(),
                            ),
                            taken_at=self.clock.now(),
                        )
                    )
                    first = False
                    self._sleep(stop, self.retry_s)
                    continue
                first = False
            if quota_thread is None or not quota_thread.is_alive():
                # every iteration, not only after a reconnect: the quota thread also ends
                # on its own (a daemon that moved port while the status client still had
                # a connection), and live quota would never refresh again (D44)
                quota_thread = threading.Thread(
                    target=self._quota_loop, name="gpu-shell-quota", daemon=True
                )
                quota_thread.start()
            try:
                snap = self._poll(client, last_ok)
            except DaemonUnavailable as exc:
                if isinstance(exc.detail, dict) and exc.detail.get("timeout"):
                    # it answered the connection but not in time: busy, not down; keep
                    # the last snapshot (the footer shows it going stale) instead of an
                    # empty "down" panel for one poll (D44)
                    self._sleep(stop, self.poll_s)
                    continue
                client.close()
                client = None
                self._emit(
                    Snapshot(
                        conn=Conn(
                            ConnStatus.DOWN,
                            message=exc.message,
                            hint=exc.hint or "",
                            retry_s=self.retry_s,
                            last_ok=last_ok,
                            autostart=self._autostart_on(),
                        ),
                        taken_at=self.clock.now(),
                    )
                )
                self._sleep(stop, self.retry_s)
                continue
            except (GpuRouterError, ValueError):
                # a 503 while the daemon recovers, a malformed answer: keep the last
                # snapshot; the footer shows it going stale
                self._sleep(stop, self.poll_s)
                continue
            last_ok = snap.conn.last_ok
            self._emit(snap)
            self._sleep(stop, self.poll_s)
        if client is not None:
            client.close()

    # ------------------------------------------------------------------ internals

    def _autostart_on(self) -> bool:
        from gpu_router.daemon.spawn import autostart_enabled

        return self.autostart and autostart_enabled()

    def _sleep(self, stop: threading.Event, seconds: float) -> None:
        if not stop.is_set():
            self._wake.wait(seconds)
        self._wake.clear()

    def _emit(self, snap: Snapshot) -> None:
        self._last = snap
        if not self._stop.is_set():
            self.publish(snap)

    def _connect(self, *, start: bool) -> GpuClient:
        from gpu_router.daemon.spawn import connect

        def starting() -> None:
            self._emit(Snapshot(conn=Conn(ConnStatus.STARTING), taken_at=self.clock.now()))

        conn = connect(self.paths, start=start and self.autostart, on_start=starting)
        if conn.started:
            self.started_pid = conn.pid
        conn.client.timeout_s = STATUS_TIMEOUT_S
        conn.client.client_name = "shell"
        return conn.client

    def _poll(self, client: GpuClient, last_ok: float | None) -> Snapshot:
        status = client.status()
        now = self.clock.now()
        active = tuple(status.active)
        live = {j.id for j in active}
        for gone in [k for k in self._hist if k not in live]:
            del self._hist[gone]
            self._seen.pop(gone, None)
        running = [j for j in active if j.state in _RUNNING][:MAX_TRACKED]
        for job in running:
            self._update_metrics(client, job)
        for job in active:
            self._update_ckpt(client, job)
        with self._lock:
            quotas = dict(self._quota)
        metrics = {jid: metric_view(h) for jid, h in self._hist.items()}
        return Snapshot(
            conn=Conn(ConnStatus.UP, last_ok=now, started_pid=self.started_pid),
            active=active,
            recent=tuple(status.recent),
            providers=merge_quota(tuple(status.providers), quotas),
            metrics=metrics,
            ckpt_where={k: v for k, (_, v) in self._ckpt.items() if k in live},
            taken_at=now,
        )

    def _update_metrics(self, client: GpuClient, job: JobView) -> None:
        n = attempt_number(job)
        if n is None:
            return
        mark = (job.updated_at, job.version)
        hist = self._hist.get(job.id)
        if hist is not None and self._seen.get(job.id) == mark:
            return  # nothing captured since the last read (captures update the job row)
        if hist is None:
            hist = self._hist[job.id] = MetricHistory()
            # first sight of a job that may have printed 100k lines already: the panel's
            # sparkline needs the recent points, not a full read that stalls the feed
            try:
                detail = client.job(job.id)
            except (GpuRouterError, ValueError):
                return
            lines = next((a.log_lines for a in detail.attempts if a.n == n), 0)
            if lines > FIRST_READ_LINES:
                hist.positions[n] = lines - FIRST_READ_LINES
        try:
            read_new_lines(client, job.id, n, hist)
        except (GpuRouterError, ValueError):
            return
        self._seen[job.id] = mark

    def _update_ckpt(self, client: GpuClient, job: JobView) -> None:
        count = job.checkpoint_count
        known = self._ckpt.get(job.id)
        if not count or (known is not None and known[0] == count):
            return
        try:
            detail = client.job(job.id)
        except (GpuRouterError, ValueError):
            return
        if detail.checkpoints:
            latest = detail.checkpoints[-1]
            provider = next(
                (a.provider for a in detail.attempts if a.id == latest.attempt_id), job.provider
            )
            self._ckpt[job.id] = (count, checkpoint_where(latest.uri, provider))

    def _quota_loop(self) -> None:
        stop = self._stop
        client: GpuClient | None = None
        while not stop.is_set():
            try:
                if client is None:
                    client = GpuClient.from_env(self.paths, client_name="shell", timeout_s=120.0)
                found = client.quota()
            except DaemonUnavailable as exc:
                if isinstance(exc.detail, dict) and exc.detail.get("timeout"):
                    found = []  # a slow provider behind /v1/quota: try again next period
                else:
                    if client is not None:
                        client.close()
                    return  # the main loop starts a new one (fresh daemon.json) next poll
            except (GpuRouterError, ValueError):
                found = []
            with self._lock:
                for q in found:
                    self._quota[q.provider] = q
            if found:
                self.poke()
            stop.wait(self.quota_s)
        if client is not None:
            client.close()
