"""state.json: the status-line cache (phase 1 writer; owner: group A).

The daemon rewrites `<home>/state.json` atomically (tmp file in the same dir, fsync,
os.replace, mode 0644) after every relevant change: any job event, progress/metric updates
(coalesced to at most one write per `MIN_WRITE_INTERVAL_S`), provider health or quota
changes, daemon start and clean stop. `gpu status --line` (phase 6, stdlib only) is the
only reader besides tests; it computes elapsed times from the absolute timestamps below so
the file need not be rewritten every second.

The shape is defined here as TypedDicts (stdlib `typing`, no pydantic) so the fast-path
reader can share it. Bump STATE_SCHEMA on any incompatible change; the reader prints
nothing when it sees an unknown schema. All timestamps are epoch seconds (floats).

Phase 6b (status line, additive; schema stays 1): every row state of docs/spec.md "GPU rows
by state" is drawable from this file alone. Added: the metric trend (from the tail of the
job's metrics.jsonl), the entry script, the current attempt and the checkpoint it resumed
from, the approval route's hours, the migration reason; recent jobs carry project dir,
absolute outputs path, fetched flag, failure kind, exit code, provider and GPU; providers
carry `unlimited` and `remaining`; the snapshot carries the visibility windows and the
writer's `heartbeat_s`. While a snapshot lists active jobs the writer also rewrites the
file every HEARTBEAT_S without a change, so a reader can tell a live daemon from a wedged
one (and ETAs and quota views stay fresh).
"""

from __future__ import annotations

import contextlib
import json
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, NotRequired, TypedDict

if TYPE_CHECKING:  # no runtime import: the fast-path reader may import this module
    from collections.abc import Callable

    from gpu_router.models import Attempt, Job
    from gpu_router.store import Store

STATE_SCHEMA = 1
MIN_WRITE_INTERVAL_S = 1.0
RECENT_WINDOW_S = 600.0  # finished/migrated rows stay visible this long (config.statusline)
HEARTBEAT_S = 60.0  # rewrite at least this often while jobs are active (phase 6b)
#: The heartbeat checks WALL time this often (D48): the event loop's clock is
#: mach_absolute_time on macOS, which stops while the Mac sleeps, so a timer of
#: HEARTBEAT_S alone would leave a stale file for up to a minute after wake.
HEARTBEAT_TICK_S = 5.0
METRICS_TAIL_BYTES = 32 * 1024  # metrics.jsonl tail read for the trend
TREND_POINTS = 100  # newest points the trend looks at


#: C0/C1 controls, DEL and the bidi overrides (models.CONTROL_CHARS): job names, metric
#: names, scripts and messages come from the job and its project, and the status line
#: prints them into a terminal, so an ESC must never get through (D48).
_CONTROL = re.compile(r"[\x00-\x1f\x7f-\x9f\u202a-\u202e\u2066-\u2069]")


def clean_text(value: str) -> str:
    """`value` without control characters (tabs and newlines become spaces)."""
    if value.isprintable():
        return value
    return _CONTROL.sub(lambda m: " " if m.group() in "\t\n\r" else "", value)


def _clean(value: str | None) -> str | None:
    return None if value is None else clean_text(value)


class MetricSummary(TypedDict):
    name: str  # "loss" preferred, else first metric reported
    value: float
    trend: Literal["down", "up", "flat"] | None


class ActiveJob(TypedDict):
    id: str
    short_id: str
    name: str
    state: str  # a JobState value
    provider: str | None
    gpu: str | None  # "2xT4"
    created_at: float
    started_at: float | None  # current attempt's start (for elapsed)
    session_cap_s: float | None  # provider session cap (progress fallback "1:42 of 12h")
    step: int | None
    total_steps: int | None
    progress_source: Literal["helper", "stdout"] | None
    eta_s: float | None  # from step rate when total known
    metric: MetricSummary | None
    last_checkpoint_at: float | None
    checkpoint_seq: int | None
    route_summary: str | None  # awaiting_approval: "colab T4 · ~20m"
    approval_reason: str | None
    not_before: float | None  # queued with backoff: retry time
    migrated_from: NotRequired[str]  # provider of the previous attempt, if migrated
    migrated_at: NotRequired[float]
    # phase 6b (additive)
    migrate_reason: NotRequired[str]  # reason of that migrating transition (handoff, ...)
    script: NotRequired[str | None]  # entry script file name ("eval.py"), or command word
    project_dir: NotRequired[str]
    attempt_n: NotRequired[int | None]  # current attempt number
    resumed_from_seq: NotRequired[int | None]  # checkpoint seq the current attempt resumed
    route_hours: NotRequired[float | None]  # awaiting_approval: spec hours, else the router's
    route_hours_source: NotRequired[str | None]  # "spec" | "estimate" | ...
    # D56 (additive): the Claude Code session the job came from; absent = no origin
    origin: NotRequired[dict[str, str]]  # {"claude_session", "claude_pid"}


class RecentJob(TypedDict):
    id: str
    short_id: str
    name: str
    state: str  # done | failed | cancelled | denied
    finished_at: float
    duration_s: float | None
    outputs_dir: str | None  # "./runs/a7f2" relative display form
    message: str
    # phase 6b (additive)
    project_dir: NotRequired[str]
    outputs_path: NotRequired[str | None]  # absolute outputs dir
    outputs_fetched: NotRequired[bool]
    failure_kind: NotRequired[str | None]
    exit_code: NotRequired[int | None]
    provider: NotRequired[str | None]
    gpu: NotRequired[str | None]
    origin: NotRequired[dict[str, str]]  # D56, as ActiveJob.origin


class ProviderSummary(TypedDict):
    name: str
    health: str  # a ProviderHealth value
    used: float | None
    limit: float | None
    unit: str | None
    resets_at: float | None
    source: Literal["live", "estimate"] | None
    # phase 6b (additive)
    unlimited: NotRequired[bool]  # no quota at all (the local Mac)
    remaining: NotRequired[float | None]  # left in `unit` per the ledger (exhausted -> 0)


class StateSnapshot(TypedDict):
    schema: int
    written_at: float
    daemon_pid: int
    # active: non-terminal jobs, most important first (running/checkpointing >
    # awaiting_approval > provisioning/migrating > cancelling > routing/queued), then oldest.
    active: list[ActiveJob]
    recent: list[RecentJob]  # terminal within RECENT_WINDOW_S, newest first
    counts: dict[str, int]  # state -> count, non-terminal states only
    providers: list[ProviderSummary]
    # phase 6b (additive)
    finished_visible_s: NotRequired[float]  # recent done/failed rows show this long
    migrated_visible_s: NotRequired[float]  # the migrated row shows this long
    heartbeat_s: NotRequired[float | None]  # rewritten at least this often while active


def build_snapshot(
    *,
    store: Store,
    provider_summaries: list[ProviderSummary],
    session_caps: dict[str, float | None],
    now: float,
    daemon_pid: int,
    recent_window_s: float = RECENT_WINDOW_S,
    migrated_window_s: float | None = None,
    metrics_path: Callable[[str], Path] | None = None,
    heartbeat_s: float | None = HEARTBEAT_S,
) -> StateSnapshot:
    """Assemble the snapshot from the Store. Pure apart from store reads (and, when
    `metrics_path` is given, a tail read of each running job's metrics.jsonl for the trend).
    `migrated_window_s` defaults to `recent_window_s`."""
    from gpu_router.statemachine import TERMINAL_STATES

    moved_window = recent_window_s if migrated_window_s is None else migrated_window_s
    jobs = sorted(
        store.non_terminal_jobs(),
        key=lambda j: (_ACTIVE_RANK.get(str(j.state), len(_ACTIVE_RANK)), j.created_at, j.id),
    )
    active = [_active_job(store, j, session_caps, now, moved_window, metrics_path) for j in jobs]
    recent = [_recent_job(j) for j in store.recent_finished(now - recent_window_s)]
    counts = {
        str(state): n
        for state, n in store.count_by_state().items()
        if state not in TERMINAL_STATES and n > 0
    }
    snap = StateSnapshot(
        schema=STATE_SCHEMA,
        written_at=now,
        daemon_pid=daemon_pid,
        active=active,
        recent=recent,
        counts=counts,
        providers=list(provider_summaries),
    )
    snap["finished_visible_s"] = recent_window_s
    snap["migrated_visible_s"] = moved_window
    snap["heartbeat_s"] = heartbeat_s
    return snap


#: Display priority of non-terminal states (lower first).
_ACTIVE_RANK: dict[str, int] = {
    "running": 0,
    "checkpointing": 0,
    "awaiting_approval": 1,
    "provisioning": 2,
    "migrating": 2,
    "cancelling": 3,
    "routing": 4,
    "queued": 4,
}


#: Which metric the status line shows (the shell panel's order, shell/metrics.PRIMARY).
_PRIMARY_METRICS = ("loss", "train_loss", "val_loss", "acc", "accuracy")
_TREND_WORDS: dict[str, Literal["down", "up", "flat"]] = {
    "\u2193": "down",
    "\u2191": "up",
    "\u2192": "flat",
}


def _metric_summary(
    metrics: dict[str, float], points: list[dict[str, Any]] | None = None
) -> MetricSummary | None:
    """The job's primary metric; its trend comes from `points` (the tail of the engine's
    metrics.jsonl), computed like the shell panel's arrow (shell/metrics.trend)."""
    if not metrics:
        return None
    name = next((m for m in _PRIMARY_METRICS if m in metrics), next(iter(metrics)))
    trend: Literal["down", "up", "flat"] | None = None
    if points:
        with contextlib.suppress(Exception):  # a status-line nicety must never fail a write
            from gpu_router.shell.metrics import trend as trend_glyph

            trend = _TREND_WORDS.get(trend_glyph(_metric_values(points, name)))
    return MetricSummary(name=clean_text(name), value=float(metrics[name]), trend=trend)


def _tail_points(path: Path) -> list[dict[str, Any]]:
    """The points in the last METRICS_TAIL_BYTES of metrics.jsonl (lines `{"ts",
    "attempt", "step", "metrics": {name: value}}`, all attempts in order)."""
    try:
        with path.open("rb") as fh:
            size = fh.seek(0, os.SEEK_END)
            fh.seek(max(0, size - METRICS_TAIL_BYTES))
            data = fh.read()
    except OSError:
        return []
    lines = data.split(b"\n")
    if len(data) >= METRICS_TAIL_BYTES:
        lines = lines[1:]  # the first line is probably cut
    points: list[dict[str, Any]] = []
    for raw in lines:
        if not raw.strip():
            continue
        try:
            point = json.loads(raw)
        except ValueError:
            continue
        if isinstance(point, dict):
            points.append(point)
    return points


def _metric_values(points: list[dict[str, Any]], name: str) -> list[float]:
    values: list[float] = []
    for point in points:
        metrics = point.get("metrics")
        value = metrics.get(name) if isinstance(metrics, dict) else None
        if isinstance(value, int | float) and not isinstance(value, bool):
            values.append(float(value))
    return values[-TREND_POINTS:]


def _metric_tail(path: Path, name: str) -> list[float]:
    """Newest values of metric `name` from the tail of metrics.jsonl."""
    return _metric_values(_tail_points(path), name)


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _eta(
    attempt: Attempt | None,
    step: int | None,
    total: int | None,
    now: float,
    points: list[dict[str, Any]],
    start_step: int | None,
) -> float | None:
    """Seconds left at `now`, from THIS attempt's step rate (D48).

    `job.progress` is job-level and a resumed script counts on from its checkpoint, so the
    old `elapsed / step` rate is only right for a fresh first attempt. Otherwise the rate
    comes from this attempt's own points in metrics.jsonl (two or more), or from one point
    and the step the attempt resumed at; with neither, None (no ETA beats a wrong one)."""
    if attempt is None or attempt.started_at is None or not step or not total or total <= step:
        return None
    mine: list[tuple[float, float]] = []
    for point in points:
        ts, st = _num(point.get("ts")), _num(point.get("step"))
        if point.get("attempt") == attempt.n and ts is not None and st is not None:
            mine.append((ts, st))
    rate: float | None = None
    if len(mine) >= 2 and mine[-1][1] > mine[0][1] and mine[-1][0] > mine[0][0]:
        (t0, s0), (t1, s1) = mine[0], mine[-1]
        rate = (s1 - s0) / (t1 - t0)
    elif attempt.n == 1 and attempt.resume_checkpoint_id is None:
        rate = step / max(1.0, now - attempt.started_at)
    elif mine and start_step is not None and mine[-1][1] > start_step:
        t1, s1 = mine[-1]
        rate = (s1 - start_step) / max(1.0, t1 - attempt.started_at)
    if not rate or rate <= 0:
        return None
    return (total - step) / rate


def _script_name(job: Job) -> str | None:
    if job.spec.script:
        return clean_text(os.path.basename(job.spec.script) or job.spec.script)
    if job.spec.command:
        return clean_text(os.path.basename(job.spec.command[0]) or job.spec.command[0])
    return None


def _origin(job: Job) -> dict[str, str] | None:
    """The job's Claude Code origin labels (D56, `gpu_router.origin`), or None."""
    labels = job.spec.labels
    out = {k: clean_text(labels[k]) for k in ("claude_session", "claude_pid") if labels.get(k)}
    return out or None


def _checkpoint_seq(checkpoint_id: str | None) -> int | None:
    """`<job>.c<seq>` -> seq."""
    if not checkpoint_id or ".c" not in checkpoint_id:
        return None
    try:
        return int(checkpoint_id.rsplit(".c", 1)[1])
    except ValueError:
        return None


def _hours_text(hours: float) -> str:
    return f"~{round(hours * 60)}m" if hours < 1 else f"~{hours:g}h"


def _active_job(
    store: Store,
    job: Job,
    session_caps: dict[str, float | None],
    now: float,
    recent_window_s: float,
    metrics_path: Callable[[str], Path] | None = None,
) -> ActiveJob:
    attempt = store.current_attempt(job)
    started_at = attempt.started_at if attempt is not None else None
    step, total = job.progress.step, job.progress.total
    running = str(job.state) in ("running", "checkpointing")
    points = _tail_points(metrics_path(job.id)) if metrics_path is not None and running else []
    start_step: int | None = None
    if attempt is not None and attempt.resume_checkpoint_id is not None and step and total:
        for ckpt in store.checkpoints_for(job.id):
            if ckpt.id == attempt.resume_checkpoint_id:
                start_step = ckpt.step
                break
    eta_s = _eta(attempt, step, total, now, points, start_step)
    checkpoint_seq: int | None = None
    if job.checkpoint_count:
        latest = store.latest_checkpoint(job.id)
        checkpoint_seq = latest.seq if latest is not None else None
    route_summary: str | None = None
    if str(job.state) == "awaiting_approval" and job.provider:
        parts = [job.provider]
        if job.gpu:
            parts.append(job.gpu)
        text = " ".join(parts)
        if job.spec.hours:
            text += f" \u00b7 {_hours_text(job.spec.hours)}"
        route_summary = text
    row = ActiveJob(
        id=job.id,
        short_id=job.short_id,
        name=clean_text(job.name),
        state=str(job.state),
        provider=_clean(job.provider),
        gpu=_clean(job.gpu),
        created_at=job.created_at,
        started_at=started_at,
        session_cap_s=session_caps.get(job.provider) if job.provider else None,
        step=step,
        total_steps=total,
        progress_source=job.progress.source,
        eta_s=eta_s,
        metric=_metric_summary(job.last_metrics, points),
        last_checkpoint_at=job.last_checkpoint_at,
        checkpoint_seq=checkpoint_seq,
        route_summary=_clean(route_summary),
        approval_reason=_clean(job.approval_reason),
        not_before=job.not_before,
    )
    row["script"] = _script_name(job)
    row["project_dir"] = clean_text(job.project_dir)
    row["attempt_n"] = attempt.n if attempt is not None else None
    row["resumed_from_seq"] = (
        _checkpoint_seq(attempt.resume_checkpoint_id) if attempt is not None else None
    )
    origin = _origin(job)
    if origin:
        row["origin"] = origin
    if str(job.state) == "awaiting_approval":
        hours: float | None = job.spec.hours
        source: str | None = "spec" if hours else None
        if not hours:  # the router's estimate, recorded on the approval_required event
            for ev in reversed(store.events_for(job.id)):
                if ev.reason == "approval_required":
                    raw = ev.detail.get("hours")
                    if isinstance(raw, int | float) and not isinstance(raw, bool) and raw > 0:
                        hours = float(raw)
                        src = ev.detail.get("hours_source")
                        source = str(src) if src else "estimate"
                    break
        row["route_hours"] = hours
        row["route_hours_source"] = source
    # a job that is moving now (migrating) or was moved recently (a later attempt)
    if job.attempt_count > 1 or str(job.state) == "migrating":
        migrated = [
            e
            for e in store.events_for(job.id)
            if e.kind == "transition" and str(e.to_state) == "migrating"
        ]
        if migrated and now - migrated[-1].ts <= recent_window_s and migrated[-1].attempt_id:
            with contextlib.suppress(Exception):
                row["migrated_from"] = clean_text(
                    store.get_attempt(migrated[-1].attempt_id).provider
                )
                row["migrated_at"] = migrated[-1].ts
                row["migrate_reason"] = migrated[-1].reason
    return row


def _recent_job(job: Job) -> RecentJob:
    assert job.finished_at is not None
    outputs = job.outputs_dir
    if outputs and outputs.startswith(job.project_dir.rstrip("/") + "/"):
        outputs = "./" + outputs[len(job.project_dir.rstrip("/")) + 1 :]
    rec = RecentJob(
        id=job.id,
        short_id=job.short_id,
        name=clean_text(job.name),
        state=str(job.state),
        finished_at=job.finished_at,
        duration_s=(job.finished_at - job.started_at) if job.started_at is not None else None,
        outputs_dir=_clean(outputs),
        message=clean_text(job.message),
        project_dir=clean_text(job.project_dir),
        outputs_path=_clean(job.outputs_dir),
        outputs_fetched=job.outputs_fetched,
        failure_kind=str(job.failure_kind) if job.failure_kind is not None else None,
        exit_code=job.exit_code,
        provider=_clean(job.provider),
        gpu=_clean(job.gpu),
    )
    origin = _origin(job)
    if origin:
        rec["origin"] = origin
    return rec


def write_atomic(path: Path, snapshot: StateSnapshot) -> None:
    """Serialise compactly and replace `path` atomically (never a partially written file).
    Writes a tmp file in the same directory, fsyncs it, then os.replace (mode 0644)."""
    data = json.dumps(snapshot, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        try:
            view = memoryview(data)
            while view:
                written = os.write(fd, view)
                view = view[written:]
            os.fsync(fd)
        finally:
            os.close(fd)
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


class StateFileWriter:
    """Coalescing writer used by the daemon.

    `mark_dirty()` is called from the store listener (event loop thread); a background task
    started by `run()` writes at most once per MIN_WRITE_INTERVAL_S, and immediately on
    `flush()` (daemon start/stop). Write errors are logged, never raised into the engine.
    """

    def __init__(
        self,
        path: Path,
        build: Callable[[], StateSnapshot],
        *,
        min_interval_s: float = MIN_WRITE_INTERVAL_S,
        heartbeat_s: float | None = HEARTBEAT_S,
        wall: Callable[[], float] | None = None,
    ) -> None:
        self.path = path
        self._build = build
        self._min_interval_s = min_interval_s
        self.heartbeat_s = heartbeat_s  # phase 6b: rewrite while jobs are active
        import asyncio  # lazy: the status-line fast path imports this module

        if wall is None:
            from gpu_router.clock import SystemClock

            wall = SystemClock().now
        self._wall = wall  # the clock the reader judges staleness by (written_at)
        self._active = False  # the last written snapshot listed active jobs
        self._dirty = False
        self._wake = asyncio.Event()
        self._last_write: float | None = None  # event-loop time of the last write
        self._last_wall: float | None = None  # wall time of the last write
        self.writes = 0

    def mark_dirty(self) -> None:
        self._dirty = True
        self._wake.set()

    async def run(self) -> None:
        """Loop until cancelled, writing when dirty."""
        import asyncio

        loop = asyncio.get_running_loop()
        while True:
            if self.heartbeat_s and self._active:
                tick = min(self.heartbeat_s, HEARTBEAT_TICK_S)
                try:
                    await asyncio.wait_for(self._wake.wait(), tick)
                except TimeoutError:  # nothing changed: heartbeat write once it is due
                    since = (
                        float("inf") if self._last_wall is None else self._wall() - self._last_wall
                    )
                    if since >= self.heartbeat_s - tick / 2:
                        self._dirty = True
            else:
                await self._wake.wait()
            self._wake.clear()
            if not self._dirty:
                continue
            if self._last_write is not None:
                remaining = self._min_interval_s - (loop.time() - self._last_write)
                if remaining > 0:
                    await asyncio.sleep(remaining)
            if self._dirty:
                self._write()
                self._last_write = loop.time()

    def flush(self) -> None:
        """Write now, synchronously."""
        import asyncio

        self._write()
        with contextlib.suppress(RuntimeError):
            self._last_write = asyncio.get_running_loop().time()

    def _write(self) -> None:
        self._dirty = False
        try:
            snapshot = self._build()
            write_atomic(self.path, snapshot)
            self._active = bool(snapshot.get("active"))
            self._last_wall = self._wall()
            self.writes += 1
        except Exception:
            import logging

            from gpu_router.log import log_event

            self._dirty = True  # try again on the next change
            log_event(
                logging.getLogger(__name__),
                "statefile.error",
                f"could not write {self.path}; will retry on the next change",
                level=logging.WARNING,
                exc_info=True,
            )
            return
        import logging

        from gpu_router.log import log_event

        log_event(
            logging.getLogger(__name__),
            "statefile.write",
            f"wrote {self.path.name}",
            level=logging.DEBUG,
        )
