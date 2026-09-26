"""The job store: every read and write of gpu.db (phase 1; owner: group A).

Contract (see CLAUDE.md "Core invariants"):

- Only the daemon constructs a Store, and `Store.open` demands a held InstanceLock.
- All methods are synchronous and called from the event-loop thread only (invariant 9).
  Each call is fast (indexed queries on a local file).
- Every job state change goes through `transition()` (or a composite built on it: `place`,
  `create_job`). One SQLite transaction does: compare-and-set on the current state, column
  updates, optional attempt update, one job_events row. After COMMIT: the listener is
  notified, then one structured log line `job.transition` is emitted.
- Validation order inside `transition()`: statemachine.check_transition (InvalidTransition,
  before touching the DB) -> CAS `UPDATE ... WHERE id=? AND state=?` (StaleState if 0 rows;
  JobNotFound if the id does not exist) -> writes.
- The store stamps `updated_at`, bumps `version` on every write to a job row, sets
  `finished_at` when entering a terminal state and `started_at` the first time a job enters
  `running`. Callers moving a job to `failed` must set `JobPatch.failure_kind` (checked
  before any write; the schema enforces it too).
- Attempts: entering `running` stamps `attempts.started_at` if unset; entering a terminal
  attempt state stamps `ended_at`.
- `Job.short_id` is computed on read (ids.shortest_unique_prefix against the neighbouring
  ids in sorted order).
- Timestamps come from the injected Clock.
- A listener that raises is logged (`store.listener`) and never undoes a committed write.
"""

# S608: every dynamic SQL fragment below is built from fixed column names and "?"
# placeholders owned by this module; values are always bound parameters.
# ruff: noqa: S608

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import sqlite3
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from gpu_router import ids
from gpu_router.clock import Clock
from gpu_router.db.connection import connect, transaction
from gpu_router.db.migrate import migrate
from gpu_router.errors import AmbiguousJobRef, InvalidTransition, JobNotFound, StaleState
from gpu_router.lock import InstanceLock
from gpu_router.log import get_logger, log_event
from gpu_router.models import (
    Attempt,
    AttemptPatch,
    Checkpoint,
    Job,
    JobEvent,
    JobPatch,
    JobSpec,
    ProviderState,
    QuotaSnapshot,
)
from gpu_router.statemachine import (
    LIVE_ATTEMPT_STATES,
    TERMINAL_ATTEMPT_STATES,
    TERMINAL_STATES,
    AttemptState,
    JobState,
    Reason,
    check_attempt_transition,
    check_transition,
)

_log = get_logger(__name__)

#: Every job read goes through this select so short_id can be computed from the ids
#: immediately before/after in sorted order (primary-key lookups, cheap).
_JOB_SELECT = (
    "SELECT j.*, "
    "(SELECT MAX(id) FROM jobs WHERE id < j.id) AS prev_id, "
    "(SELECT MIN(id) FROM jobs WHERE id > j.id) AS next_id "
    "FROM jobs j"
)

#: ProviderState fields writable through upsert_provider_state.
_PROVIDER_COLUMNS = frozenset(
    {
        "health",
        "health_reason",
        "last_healthcheck_at",
        "cooldown_until",
        "consecutive_failures",
        "exhausted_until",
    }
)

_TERMINAL_VALUES = tuple(str(s) for s in TERMINAL_STATES)
_LIVE_ATTEMPT_VALUES = tuple(str(s) for s in LIVE_ATTEMPT_STATES)


#: attempts.error_kind of an attempt abandoned because its provider stayed unreachable
#: (submit outcome unknown); excluded_providers ignores these.
UNREACHABLE_ERROR_KIND = "Unreachable"


@dataclass(frozen=True, slots=True)
class DataCacheEntry:
    """One row of `data_cache` (phase 5): a dataset uploaded once, reused by hash."""

    content_hash: str
    uri: str
    local_path: str
    size_bytes: int
    file_count: int
    uploaded_at: float
    last_used_at: float


class StoreListener(Protocol):
    """Notified after every committed write that changes a job.

    `events` is empty for field-only updates (progress, metrics). Implementations must be
    quick and must not call back into the Store synchronously (schedule work instead).
    The daemon's listener fans out to the event bus (/v1/events long-poll), the state.json
    writer, and (phase 8) notifications.
    """

    def job_changed(self, job_id: str, events: Sequence[JobEvent]) -> None: ...


# Optional listener hook (not part of the Protocol so older listeners stay valid):
# `provider_changed(provider: str) -> None` is called after upsert_provider_state commits,
# if the listener defines it. Same rules as job_changed (quick, no Store re-entry).


@dataclass(frozen=True, slots=True)
class AttemptChange:
    """An attempt update applied inside the same transaction as a job transition."""

    attempt_id: str
    patch: AttemptPatch


# --------------------------------------------------------------------------- row helpers


def _dumps(value: Mapping[str, Any] | None) -> str:
    return json.dumps(dict(value or {}), separators=(",", ":"), sort_keys=True)


def _placeholders(n: int) -> str:
    return ",".join("?" * n)


def _job_patch_columns(patch: JobPatch) -> dict[str, Any]:
    cols: dict[str, Any] = {}
    for name in patch.model_fields_set:
        value = getattr(patch, name)
        if name == "last_metrics":
            cols["last_metrics_json"] = _dumps(value)
        elif name == "outputs_fetched":
            cols["outputs_fetched"] = 1 if value else 0
        elif name == "failure_kind":
            cols["failure_kind"] = None if value is None else str(value)
        elif name == "message":
            cols["message"] = value or ""
        else:
            cols[name] = value
    return cols


def _attempt_patch_columns(patch: AttemptPatch) -> dict[str, Any]:
    cols: dict[str, Any] = {}
    for name in patch.model_fields_set:
        value = getattr(patch, name)
        if name == "state":
            continue  # handled by the caller (validation + timestamps)
        if name == "remote_meta":
            cols["remote_meta_json"] = _dumps(value)
        elif name == "log_lines":
            cols["log_lines"] = value or 0
        else:
            cols[name] = value
    return cols


def _job_from_row(row: sqlite3.Row) -> Job:
    job_id: str = row["id"]
    return Job(
        id=job_id,
        short_id=ids.shortest_unique_prefix(job_id, (row["prev_id"], row["next_id"])),
        name=row["name"],
        state=JobState(row["state"]),
        source=row["source"],
        request_id=row["request_id"],
        project_dir=row["project_dir"],
        spec=JobSpec.model_validate_json(row["spec_json"]),
        spec_hash=row["spec_hash"],
        bundle_sha256=row["bundle_sha256"],
        provider=row["provider"],
        gpu=row["gpu"],
        current_attempt_id=row["current_attempt_id"],
        attempt_count=row["attempt_count"],
        accepted_attempts=row["accepted_attempts"],
        route_reason=row["route_reason"],
        approval_reason=row["approval_reason"],
        approved_at=row["approved_at"],
        approved_by=row["approved_by"],
        not_before=row["not_before"],
        waiting_since=row["waiting_since"],
        cancel_requested_at=row["cancel_requested_at"],
        progress={
            "step": row["progress_step"],
            "total": row["progress_total"],
            "source": row["progress_source"],
        },
        last_metrics=json.loads(row["last_metrics_json"] or "{}"),
        checkpoint_count=row["checkpoint_count"],
        last_checkpoint_at=row["last_checkpoint_at"],
        outputs_dir=row["outputs_dir"],
        outputs_fetched=bool(row["outputs_fetched"]),
        exit_code=row["exit_code"],
        failure_kind=row["failure_kind"],
        message=row["message"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        version=row["version"],
    )


def _attempt_from_row(row: sqlite3.Row) -> Attempt:
    data = dict(row)
    data["remote_meta"] = json.loads(data.pop("remote_meta_json") or "{}")
    data["state"] = AttemptState(data["state"])
    return Attempt.model_validate(data)


def _event_from_row(row: sqlite3.Row) -> JobEvent:
    data = dict(row)
    data["detail"] = json.loads(data.pop("detail_json") or "{}")
    return JobEvent.model_validate(data)


def _checkpoint_from_row(row: sqlite3.Row) -> Checkpoint:
    return Checkpoint.model_validate(dict(row))


def _provider_from_row(row: sqlite3.Row) -> ProviderState:
    return ProviderState.model_validate(dict(row))


def _quota_from_row(row: sqlite3.Row) -> QuotaSnapshot:
    return QuotaSnapshot(
        provider=row["provider"],
        used=row["used"],
        limit=row["quota_limit"],
        unit=row["unit"],
        resets_at=row["resets_at"],
        source=row["source"],
        detail=json.loads(row["detail_json"] or "{}"),
        observed_at=row["observed_at"],
    )


#: quota_snapshots rows kept per provider (about 10 days of 30-min live readings).
QUOTA_SNAPSHOTS_KEEP = 500


# --------------------------------------------------------------------------- the store


class Store:
    def __init__(
        self,
        conn: sqlite3.Connection,
        clock: Clock,
        *,
        listener: StoreListener | None = None,
    ) -> None:
        self._conn = conn
        self._clock = clock
        self._listener = listener

    # ------------------------------------------------------------------ lifecycle

    @classmethod
    def open(
        cls,
        db_path: Path,
        *,
        lock: InstanceLock,
        clock: Clock,
        listener: StoreListener | None = None,
    ) -> Store:
        """Connect (db.connection.connect), migrate (db.migrate.migrate) and return a Store.
        Raises RuntimeError if `lock` is not held."""
        if not lock.held:
            raise RuntimeError("Store.open requires a held InstanceLock (invariant 1)")
        conn = connect(db_path)
        try:
            migrate(conn, db_path, clock)
        except BaseException:
            conn.close()
            raise
        return cls(conn, clock, listener=listener)

    @classmethod
    def open_memory(cls, clock: Clock, *, listener: StoreListener | None = None) -> Store:
        """In-memory, migrated store for unit tests (no lock needed)."""
        conn = connect(":memory:")
        migrate(conn, None, clock)
        return cls(conn, clock, listener=listener)

    def set_listener(self, listener: StoreListener | None) -> None:
        self._listener = listener

    def close(self) -> None:
        """Close the connection. Idempotent."""
        with contextlib.suppress(sqlite3.ProgrammingError):
            self._conn.close()

    # ------------------------------------------------------------------ internals

    def _notify(self, job_id: str, events: Sequence[JobEvent]) -> None:
        if self._listener is None:
            return
        try:
            self._listener.job_changed(job_id, events)
        except Exception:
            log_event(
                _log,
                "store.listener",
                f"store listener failed for job {job_id}",
                level=logging.ERROR,
                exc_info=True,
                job_id=job_id,
            )

    def _notify_provider(self, provider: str) -> None:
        hook = getattr(self._listener, "provider_changed", None)
        if hook is None:
            return
        try:
            hook(provider)
        except Exception:
            log_event(
                _log,
                "store.listener",
                f"store listener failed for provider {provider}",
                level=logging.ERROR,
                exc_info=True,
                provider=provider,
            )

    def _job_row(self, cur: sqlite3.Cursor | sqlite3.Connection, job_id: str) -> sqlite3.Row:
        row: sqlite3.Row | None = cur.execute(f"{_JOB_SELECT} WHERE j.id = ?", (job_id,)).fetchone()
        if row is None:
            raise JobNotFound(
                f"no job {job_id}", hint="list jobs with `gpu status`", detail={"ref": job_id}
            )
        return row

    def _attempt_row(
        self, cur: sqlite3.Cursor | sqlite3.Connection, attempt_id: str
    ) -> sqlite3.Row:
        row: sqlite3.Row | None = cur.execute(
            "SELECT * FROM attempts WHERE id = ?", (attempt_id,)
        ).fetchone()
        if row is None:
            raise JobNotFound(f"no attempt {attempt_id}", detail={"attempt_id": attempt_id})
        return row

    def _insert_event(
        self,
        cur: sqlite3.Cursor,
        *,
        job_id: str,
        attempt_id: str | None,
        kind: str,
        from_state: JobState | None,
        to_state: JobState | None,
        reason: str,
        message: str,
        detail: Mapping[str, Any] | None,
        actor: str,
        ts: float,
    ) -> JobEvent:
        detail_json = _dumps(detail)
        cur.execute(
            "INSERT INTO job_events (job_id, attempt_id, kind, from_state, to_state, reason, "
            "message, detail_json, actor, ts) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                job_id,
                attempt_id,
                kind,
                None if from_state is None else str(from_state),
                None if to_state is None else str(to_state),
                str(reason),
                message,
                detail_json,
                actor,
                ts,
            ),
        )
        seq = cur.lastrowid
        assert seq is not None
        return JobEvent(
            seq=seq,
            job_id=job_id,
            attempt_id=attempt_id,
            kind="transition" if kind == "transition" else "note",
            from_state=from_state,
            to_state=to_state,
            reason=str(reason),
            message=message,
            detail=json.loads(detail_json),
            actor=actor,
            ts=ts,
        )

    def _apply_job_columns(
        self, cur: sqlite3.Cursor, job_id: str, cols: Mapping[str, Any], now: float
    ) -> None:
        """Write non-state columns + updated_at + version bump. Raises JobNotFound."""
        sets = [f"{c} = ?" for c in cols] + ["updated_at = ?", "version = version + 1"]
        params = [*cols.values(), now, job_id]
        cur.execute(f"UPDATE jobs SET {', '.join(sets)} WHERE id = ?", params)
        if cur.rowcount == 0:
            raise JobNotFound(f"no job {job_id}", detail={"ref": job_id})

    def _apply_attempt_patch(
        self, cur: sqlite3.Cursor, attempt_id: str, patch: AttemptPatch, now: float
    ) -> None:
        row = self._attempt_row(cur, attempt_id)
        cols = _attempt_patch_columns(patch)
        if "state" in patch.model_fields_set and patch.state is not None:
            current = AttemptState(row["state"])
            new = patch.state
            if new is not current:
                check_attempt_transition(current, new)
                cols["state"] = str(new)
                if new in TERMINAL_ATTEMPT_STATES:
                    cols["ended_at"] = now
                if (
                    new is AttemptState.RUNNING
                    and row["started_at"] is None
                    and "started_at" not in cols
                ):
                    cols["started_at"] = now
        if not cols:
            return
        sets = ", ".join(f"{c} = ?" for c in cols)
        cur.execute(f"UPDATE attempts SET {sets} WHERE id = ?", [*cols.values(), attempt_id])

    def _transition_in_tx(
        self,
        cur: sqlite3.Cursor,
        job_id: str,
        *,
        from_state: JobState,
        to_state: JobState,
        reason: Reason,
        message: str,
        actor: str,
        detail: Mapping[str, Any] | None,
        cols: dict[str, Any],
        attempt: AttemptChange | None,
        event_attempt_id: str | None,
        now: float,
    ) -> JobEvent:
        """CAS the state and write columns/attempt/event. Caller validated the transition."""
        cols = dict(cols)
        cols.setdefault("message", message)
        cols["state"] = str(to_state)
        if to_state in TERMINAL_STATES:
            cols["finished_at"] = now
        sets = [f"{c} = ?" for c in cols]
        params: list[Any] = list(cols.values())
        if to_state is JobState.RUNNING and "started_at" not in cols:
            sets.append("started_at = COALESCE(started_at, ?)")
            params.append(now)
        sets += ["updated_at = ?", "version = version + 1"]
        params += [now, job_id, str(from_state)]
        cur.execute(f"UPDATE jobs SET {', '.join(sets)} WHERE id = ? AND state = ?", params)
        if cur.rowcount == 0:
            row = cur.execute("SELECT state FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if row is None:
                raise JobNotFound(
                    f"no job {job_id}", hint="list jobs with `gpu status`", detail={"ref": job_id}
                )
            raise StaleState(job_id, str(from_state), row["state"])
        if attempt is not None:
            self._apply_attempt_patch(cur, attempt.attempt_id, attempt.patch, now)
        if event_attempt_id is None:
            event_attempt_id = (
                attempt.attempt_id
                if attempt is not None
                else cur.execute(
                    "SELECT current_attempt_id FROM jobs WHERE id = ?", (job_id,)
                ).fetchone()[0]
            )
        return self._insert_event(
            cur,
            job_id=job_id,
            attempt_id=event_attempt_id,
            kind="transition",
            from_state=from_state,
            to_state=to_state,
            reason=reason,
            message=message,
            detail=detail,
            actor=actor,
            ts=now,
        )

    def _log_transition(self, job: Job, event: JobEvent) -> None:
        fields: dict[str, Any] = {
            "job_id": job.id,
            "attempt_id": event.attempt_id,
            "provider": job.provider,
            "from": None if event.from_state is None else str(event.from_state),
            "to": str(event.to_state),
            "reason": event.reason,
            "actor": event.actor,
        }
        log_event(
            _log,
            "job.transition",
            f"{job.short_id} {event.from_state or '(new)'} -> {event.to_state}: {event.message}",
            **fields,
        )

    def _log_note(self, job_id: str, event: JobEvent) -> None:
        log_event(
            _log,
            "job.note",
            f"{job_id[:4]} note {event.reason}: {event.message}",
            job_id=job_id,
            attempt_id=event.attempt_id,
            reason=event.reason,
            actor=event.actor,
        )

    # ------------------------------------------------------------------ jobs: create / read

    def create_job(
        self,
        spec: JobSpec,
        *,
        actor: str,
        request_id: str | None = None,
        message: str | None = None,
    ) -> tuple[Job, bool]:
        """Insert a job in `queued` with a fresh id (ids.new_job_id with a prefix-taken check)
        plus the creation event (kind transition, from NULL, to queued, reason submitted).
        `name` = spec.display_name(); `source`, `project_dir` copied from the spec;
        spec_hash = sha256(spec.model_dump_json()); outputs_dir =
        f"{spec.project_dir}/runs/{id[:4]}" (fixed forever, even if short_id grows later).
        `waiting_since` starts at the creation time (the placement wait begins now).

        Idempotency: if `request_id` is given and a job with that request_id exists, return
        (existing_job, False) without writing anything. Otherwise (new_job, True)."""
        if request_id is not None:
            row = self._conn.execute(
                f"{_JOB_SELECT} WHERE j.request_id = ?", (request_id,)
            ).fetchone()
            if row is not None:
                return _job_from_row(row), False

        spec_json = spec.model_dump_json()
        spec_hash = hashlib.sha256(spec_json.encode()).hexdigest()
        now = self._clock.now()
        text = message or "queued; looking for a provider next"

        def prefix_taken(prefix: str) -> bool:
            return (
                self._conn.execute(
                    "SELECT 1 FROM jobs WHERE substr(id, 1, ?) = ? LIMIT 1",
                    (len(prefix), prefix),
                ).fetchone()
                is not None
            )

        for _ in range(5):
            job_id = ids.new_job_id(prefix_taken)
            try:
                with transaction(self._conn) as cur:
                    cur.execute(
                        "INSERT INTO jobs (id, name, state, source, request_id, project_dir, "
                        "spec_json, spec_hash, outputs_dir, message, created_at, updated_at, "
                        "waiting_since) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            job_id,
                            spec.display_name(),
                            str(JobState.QUEUED),
                            str(spec.source),
                            request_id,
                            spec.project_dir,
                            spec_json,
                            spec_hash,
                            f"{spec.project_dir.rstrip('/')}/runs/{job_id[:4]}",
                            text,
                            now,
                            now,
                            now,
                        ),
                    )
                    event = self._insert_event(
                        cur,
                        job_id=job_id,
                        attempt_id=None,
                        kind="transition",
                        from_state=None,
                        to_state=JobState.QUEUED,
                        reason=Reason.SUBMITTED,
                        message=text,
                        detail={"request_id": request_id} if request_id else None,
                        actor=actor,
                        ts=now,
                    )
            except sqlite3.IntegrityError as exc:
                # A full-id collision (2^-48) retries; a request_id race returns the winner.
                if request_id is not None and "request_id" in str(exc):
                    existing = self._conn.execute(
                        f"{_JOB_SELECT} WHERE j.request_id = ?", (request_id,)
                    ).fetchone()
                    if existing is not None:
                        return _job_from_row(existing), False
                if "jobs.id" in str(exc):
                    continue
                raise
            job = self.get_job(job_id)
            self._notify(job_id, [event])
            self._log_transition(job, event)
            return job, True
        raise RuntimeError("could not allocate a unique job id")  # pragma: no cover

    def get_job(self, job_id: str) -> Job:
        """Exact id. Raises JobNotFound."""
        return _job_from_row(self._job_row(self._conn, job_id))

    def resolve_ref(self, ref: str) -> Job:
        """Full id or any unique prefix (ids.normalize_ref first). Raises JobNotFound, or
        AmbiguousJobRef with detail {"matches": [<up to 10 ids>]}."""
        prefix = ids.normalize_ref(ref)
        # 'g' sorts after every hex digit, so [prefix, prefix+'g') is exactly the prefix range.
        rows = self._conn.execute(
            "SELECT id FROM jobs WHERE id >= ? AND id < ? ORDER BY id LIMIT 11",
            (prefix, prefix + "g"),
        ).fetchall()
        if not rows:
            raise JobNotFound(
                f"no job matches {prefix!r}",
                hint="list jobs with `gpu status`",
                detail={"ref": prefix},
            )
        if len(rows) > 1:
            matches = [r["id"] for r in rows[:10]]
            raise AmbiguousJobRef(
                f"{prefix!r} matches more than one job",
                hint="type more characters of the job id",
                detail={"ref": prefix, "matches": matches},
            )
        return self.get_job(rows[0]["id"])

    def list_jobs(
        self,
        *,
        states: Collection[JobState] | None = None,
        project_dir: str | None = None,
        limit: int = 50,
        before: float | None = None,
    ) -> list[Job]:
        """Newest first (created_at DESC). `before` = created_at cursor for paging."""
        where: list[str] = []
        params: list[Any] = []
        if states is not None:
            state_list = [str(s) for s in states]
            if not state_list:
                return []
            where.append(f"j.state IN ({_placeholders(len(state_list))})")
            params += state_list
        if project_dir is not None:
            where.append("j.project_dir = ?")
            params.append(project_dir.rstrip("/") or "/")
        if before is not None:
            where.append("j.created_at < ?")
            params.append(before)
        clause = f" WHERE {' AND '.join(where)}" if where else ""
        params.append(max(0, limit))
        rows = self._conn.execute(
            f"{_JOB_SELECT}{clause} ORDER BY j.created_at DESC, j.id DESC LIMIT ?", params
        ).fetchall()
        return [_job_from_row(r) for r in rows]

    def non_terminal_jobs(self) -> list[Job]:
        """Every job not in TERMINAL_STATES, oldest first (recovery order)."""
        rows = self._conn.execute(
            f"{_JOB_SELECT} WHERE j.state NOT IN ({_placeholders(len(_TERMINAL_VALUES))}) "
            "ORDER BY j.created_at ASC, j.id ASC",
            _TERMINAL_VALUES,
        ).fetchall()
        return [_job_from_row(r) for r in rows]

    def recent_finished(self, since: float) -> list[Job]:
        """Terminal jobs with finished_at >= since, newest first (status line 'just finished')."""
        rows = self._conn.execute(
            f"{_JOB_SELECT} WHERE j.finished_at IS NOT NULL AND j.finished_at >= ? "
            "ORDER BY j.finished_at DESC, j.id DESC",
            (since,),
        ).fetchall()
        return [_job_from_row(r) for r in rows]

    def count_by_state(self) -> dict[JobState, int]:
        """Job count per state; states with no jobs are omitted."""
        rows = self._conn.execute("SELECT state, COUNT(*) AS n FROM jobs GROUP BY state")
        return {JobState(r["state"]): int(r["n"]) for r in rows}

    # ------------------------------------------------------------------ jobs: write

    def transition(
        self,
        job_id: str,
        *,
        from_state: JobState,
        to_state: JobState,
        reason: Reason,
        message: str,
        actor: str,
        detail: Mapping[str, Any] | None = None,
        patch: JobPatch | None = None,
        attempt: AttemptChange | None = None,
    ) -> Job:
        """The one write path for job state (see module docstring). `message` becomes both
        the event message and jobs.message unless `patch.message` is set. Returns the
        updated Job. Raises InvalidTransition, StaleState, JobNotFound."""
        check_transition(from_state, to_state)
        cols = _job_patch_columns(patch) if patch is not None else {}
        if to_state is JobState.FAILED and cols.get("failure_kind") is None:
            raise ValueError("a transition to failed must set JobPatch.failure_kind")
        if to_state is not JobState.FAILED and cols.get("failure_kind") is not None:
            raise ValueError("failure_kind may only be set when the job moves to failed")
        now = self._clock.now()
        with transaction(self._conn) as cur:
            event = self._transition_in_tx(
                cur,
                job_id,
                from_state=from_state,
                to_state=to_state,
                reason=reason,
                message=message,
                actor=actor,
                detail=detail,
                cols=cols,
                attempt=attempt,
                event_attempt_id=None,
                now=now,
            )
        job = self.get_job(job_id)
        self._notify(job_id, [event])
        self._log_transition(job, event)
        return job

    def update_job(self, job_id: str, patch: JobPatch) -> Job:
        """Non-state column changes (progress, metrics, not_before, message). No event.
        Notifies the listener with events=[]."""
        cols = _job_patch_columns(patch)
        if cols.get("failure_kind") is not None:
            raise ValueError("failure_kind is set only by a transition to failed")
        now = self._clock.now()
        with transaction(self._conn) as cur:
            self._apply_job_columns(cur, job_id, cols, now)
        job = self.get_job(job_id)
        self._notify(job_id, [])
        return job

    def add_note(
        self,
        job_id: str,
        *,
        reason: Reason,
        message: str,
        actor: str,
        detail: Mapping[str, Any] | None = None,
        attempt_id: str | None = None,
        patch: JobPatch | None = None,
    ) -> JobEvent:
        """Insert a kind='note' event (+ optional non-state patch) in one transaction.
        The job's `updated_at`/`version` move too, so readers see that something happened."""
        cols = _job_patch_columns(patch) if patch is not None else {}
        if cols.get("failure_kind") is not None:
            raise ValueError("failure_kind is set only by a transition to failed")
        now = self._clock.now()
        with transaction(self._conn) as cur:
            self._apply_job_columns(cur, job_id, cols, now)
            event = self._insert_event(
                cur,
                job_id=job_id,
                attempt_id=attempt_id,
                kind="note",
                from_state=None,
                to_state=None,
                reason=reason,
                message=message,
                detail=detail,
                actor=actor,
                ts=now,
            )
        self._notify(job_id, [event])
        self._log_note(job_id, event)
        return event

    def record_approval(self, job_id: str, *, actor: str) -> Job:
        """Set approved_at/approved_by and add note reason=approved. Raises
        InvalidTransition unless the job is awaiting_approval. Not a state change: the
        driver places the job right after (awaiting_approval -> provisioning)."""
        now = self._clock.now()
        with transaction(self._conn) as cur:
            row = cur.execute("SELECT state FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if row is None:
                raise JobNotFound(
                    f"no job {job_id}", hint="list jobs with `gpu status`", detail={"ref": job_id}
                )
            state = JobState(row["state"])
            if state is not JobState.AWAITING_APPROVAL:
                raise InvalidTransition(
                    str(state), "approved", hint="only jobs awaiting approval can be approved"
                )
            self._apply_job_columns(
                cur,
                job_id,
                {"approved_at": now, "approved_by": actor, "message": "approved; placing it now"},
                now,
            )
            event = self._insert_event(
                cur,
                job_id=job_id,
                attempt_id=None,
                kind="note",
                from_state=None,
                to_state=None,
                reason=Reason.APPROVED,
                message=f"approved by {actor}; placing it now",
                detail=None,
                actor=actor,
                ts=now,
            )
        job = self.get_job(job_id)
        self._notify(job_id, [event])
        self._log_note(job_id, event)
        return job

    # ------------------------------------------------------------------ attempts

    def place(
        self,
        job_id: str,
        *,
        from_state: JobState,
        provider: str,
        gpu: str | None,
        route_reason: str,
        message: str,
        detail: Mapping[str, Any] | None = None,
        resume_checkpoint_id: str | None = None,
        actor: str = "engine",
    ) -> tuple[Job, Attempt]:
        """One transaction: insert attempt n = attempt_count + 1 (state submitting,
        attempt_key = ids.attempt_key) and transition from_state -> provisioning with
        reason placed, setting provider, gpu, current_attempt_id, route_reason,
        attempt_count, waiting_since=NULL, not_before=NULL. `detail` should carry the
        RouteDecision (ranked candidates + rejections). IntegrityError on the live-attempt
        unique index means a bug (a live attempt already exists) and is re-raised."""
        check_transition(from_state, JobState.PROVISIONING)
        now = self._clock.now()
        with transaction(self._conn) as cur:
            row = cur.execute(
                "SELECT state, attempt_count FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
            if row is None:
                raise JobNotFound(
                    f"no job {job_id}", hint="list jobs with `gpu status`", detail={"ref": job_id}
                )
            if row["state"] != str(from_state):
                raise StaleState(job_id, str(from_state), row["state"])
            n = int(row["attempt_count"]) + 1
            att_id = ids.attempt_id(job_id, n)
            cur.execute(
                "INSERT INTO attempts (id, job_id, n, provider, attempt_key, state, gpu, "
                "route_reason, resume_checkpoint_id, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    att_id,
                    job_id,
                    n,
                    provider,
                    ids.attempt_key(job_id, n),
                    str(AttemptState.SUBMITTING),
                    gpu,
                    route_reason,
                    resume_checkpoint_id,
                    now,
                ),
            )
            event = self._transition_in_tx(
                cur,
                job_id,
                from_state=from_state,
                to_state=JobState.PROVISIONING,
                reason=Reason.PLACED,
                message=message,
                actor=actor,
                detail=detail,
                cols={
                    "provider": provider,
                    "gpu": gpu,
                    "current_attempt_id": att_id,
                    "route_reason": route_reason,
                    "attempt_count": n,
                    "waiting_since": None,
                    "not_before": None,
                },
                attempt=None,
                event_attempt_id=att_id,
                now=now,
            )
        job = self.get_job(job_id)
        attempt = self.get_attempt(att_id)
        self._notify(job_id, [event])
        self._log_transition(job, event)
        return job, attempt

    def record_submission(
        self,
        attempt_id: str,
        *,
        remote_id: str,
        remote_url: str | None,
        remote_meta: Mapping[str, str],
        actor: str = "engine",
    ) -> Attempt:
        """submitting -> submitted with remote ids + submitted_at; increments the job's
        accepted_attempts; adds note reason=submit_confirmed. One transaction. Idempotent:
        if the attempt already has the same remote_id, returns it unchanged."""
        now = self._clock.now()
        with transaction(self._conn) as cur:
            row = self._attempt_row(cur, attempt_id)
            if row["remote_id"] == remote_id:
                return _attempt_from_row(row)
            check_attempt_transition(AttemptState(row["state"]), AttemptState.SUBMITTED)
            cur.execute(
                "UPDATE attempts SET state = ?, remote_id = ?, remote_url = ?, "
                "remote_meta_json = ?, submitted_at = ?, last_seen_at = ? WHERE id = ?",
                (
                    str(AttemptState.SUBMITTED),
                    remote_id,
                    remote_url,
                    _dumps(remote_meta),
                    now,
                    now,
                    attempt_id,
                ),
            )
            job_id: str = row["job_id"]
            cur.execute(
                "UPDATE jobs SET accepted_attempts = accepted_attempts + 1, updated_at = ?, "
                "version = version + 1 WHERE id = ?",
                (now, job_id),
            )
            event = self._insert_event(
                cur,
                job_id=job_id,
                attempt_id=attempt_id,
                kind="note",
                from_state=None,
                to_state=None,
                reason=Reason.SUBMIT_CONFIRMED,
                message=f"{row['provider']} accepted the run ({remote_id}); waiting for it to "
                "start",
                detail={"remote_id": remote_id, "remote_url": remote_url},
                actor=actor,
                ts=now,
            )
        self._notify(job_id, [event])
        self._log_note(job_id, event)
        return self.get_attempt(attempt_id)

    def update_attempt(self, attempt_id: str, patch: AttemptPatch) -> Attempt:
        """Attempt-only change (validated with check_attempt_transition if state is set and
        differs from the current state; stamps ended_at on terminal states and started_at
        on the first entry to running). Notifies the listener with events=[]."""
        now = self._clock.now()
        with transaction(self._conn) as cur:
            self._apply_attempt_patch(cur, attempt_id, patch, now)
            job_id: str = self._attempt_row(cur, attempt_id)["job_id"]
        self._notify(job_id, [])
        return self.get_attempt(attempt_id)

    def get_attempt(self, attempt_id: str) -> Attempt:
        """Exact attempt id. Raises JobNotFound."""
        return _attempt_from_row(self._attempt_row(self._conn, attempt_id))

    def current_attempt(self, job: Job) -> Attempt | None:
        """The job's current (latest placed) attempt, or None before the first placement."""
        if job.current_attempt_id is None:
            return None
        return self.get_attempt(job.current_attempt_id)

    def attempts_for(self, job_id: str) -> list[Attempt]:
        """Oldest first."""
        rows = self._conn.execute(
            "SELECT * FROM attempts WHERE job_id = ? ORDER BY n", (job_id,)
        ).fetchall()
        return [_attempt_from_row(r) for r in rows]

    def live_attempts_by_provider(self) -> dict[str, int]:
        """provider -> number of attempts in LIVE_ATTEMPT_STATES (router capacity check)."""
        rows = self._conn.execute(
            "SELECT provider, COUNT(*) AS n FROM attempts "
            f"WHERE state IN ({_placeholders(len(_LIVE_ATTEMPT_VALUES))}) GROUP BY provider",
            _LIVE_ATTEMPT_VALUES,
        )
        return {r["provider"]: int(r["n"]) for r in rows}

    def attempts_in_state(self, states: Collection[AttemptState]) -> list[Attempt]:
        """Attempts in any of `states`, oldest first."""
        values = [str(s) for s in states]
        if not values:
            return []
        rows = self._conn.execute(
            f"SELECT * FROM attempts WHERE state IN ({_placeholders(len(values))}) "
            "ORDER BY created_at, id",
            values,
        ).fetchall()
        return [_attempt_from_row(r) for r in rows]

    def excluded_providers(self, job_id: str) -> set[str]:
        """Providers this job must not be placed on again, derived from its attempts (so it
        survives restarts): attempts with error_kind == "InvalidJob" or state == abandoned,
        except abandons caused by an outage (error_kind == UNREACHABLE_ERROR_KIND): an
        outage cools the provider down, it never bans it for the job (invariant 8)."""
        rows = self._conn.execute(
            "SELECT DISTINCT provider FROM attempts WHERE job_id = ? "
            "AND (error_kind = 'InvalidJob' OR (state = ? AND COALESCE(error_kind, '') != ?))",
            (job_id, str(AttemptState.ABANDONED), UNREACHABLE_ERROR_KIND),
        )
        return {r["provider"] for r in rows}

    def usage_seconds(self, provider: str, since: float, until: float) -> float:
        """GPU seconds on `provider` overlapping [since, until): sum over attempts of
        min(ended_at or until, until) - max(started_at, since). Phase 5 ledger input."""
        if until <= since:
            return 0.0
        rows = self._conn.execute(
            "SELECT started_at, ended_at FROM attempts WHERE provider = ? "
            "AND started_at IS NOT NULL AND started_at < ? "
            "AND (ended_at IS NULL OR ended_at > ?)",
            (provider, until, since),
        )
        total = 0.0
        for r in rows:
            end = until if r["ended_at"] is None else min(float(r["ended_at"]), until)
            start = max(float(r["started_at"]), since)
            total += max(0.0, end - start)
        return total

    # ------------------------------------------------------------------ events

    def events_for(self, job_id: str, *, after_seq: int = 0, limit: int = 500) -> list[JobEvent]:
        """One job's events with seq > after_seq, oldest first."""
        rows = self._conn.execute(
            "SELECT * FROM job_events WHERE job_id = ? AND seq > ? ORDER BY seq LIMIT ?",
            (job_id, after_seq, max(0, limit)),
        ).fetchall()
        return [_event_from_row(r) for r in rows]

    def events_after(self, after_seq: int, *, limit: int = 500) -> list[JobEvent]:
        """Global feed ordered by seq (for /v1/events)."""
        rows = self._conn.execute(
            "SELECT * FROM job_events WHERE seq > ? ORDER BY seq LIMIT ?",
            (after_seq, max(0, limit)),
        ).fetchall()
        return [_event_from_row(r) for r in rows]

    def max_event_seq(self) -> int:
        """Highest event seq, or 0 when there are no events."""
        value = self._conn.execute("SELECT MAX(seq) FROM job_events").fetchone()[0]
        return int(value) if value is not None else 0

    # ------------------------------------------------------------------ checkpoints

    def record_checkpoint(
        self,
        job_id: str,
        attempt_id: str,
        *,
        seq: int,
        uri: str,
        step: int | None,
        size_bytes: int | None,
        sha256: str | None,
        created_at: float,
    ) -> Checkpoint:
        """Insert (idempotent on (job_id, seq): an identical row returns the existing one)
        and update jobs.checkpoint_count / last_checkpoint_at. No event: the matching
        checkpointing -> running transition carries it. A different row with the same
        (job_id, seq) raises ValueError. Notifies the listener with events=[]."""
        now = self._clock.now()
        with transaction(self._conn) as cur:
            existing = cur.execute(
                "SELECT * FROM checkpoints WHERE job_id = ? AND seq = ?", (job_id, seq)
            ).fetchone()
            if existing is not None:
                same = (
                    existing["attempt_id"] == attempt_id
                    and existing["uri"] == uri
                    and existing["step"] == step
                    and existing["size_bytes"] == size_bytes
                    and existing["sha256"] == sha256
                )
                if not same:
                    raise ValueError(
                        f"checkpoint {ids.checkpoint_id(job_id, seq)} already recorded with "
                        "different contents"
                    )
                return _checkpoint_from_row(existing)
            ckpt_id = ids.checkpoint_id(job_id, seq)
            cur.execute(
                "INSERT INTO checkpoints (id, job_id, attempt_id, seq, step, uri, size_bytes, "
                "sha256, created_at, recorded_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (ckpt_id, job_id, attempt_id, seq, step, uri, size_bytes, sha256, created_at, now),
            )
            cur.execute(
                "UPDATE jobs SET checkpoint_count = "
                "(SELECT COUNT(*) FROM checkpoints WHERE job_id = ?), "
                "last_checkpoint_at = MAX(COALESCE(last_checkpoint_at, 0), ?), "
                "updated_at = ?, version = version + 1 WHERE id = ?",
                (job_id, created_at, now, job_id),
            )
            row = cur.execute("SELECT * FROM checkpoints WHERE id = ?", (ckpt_id,)).fetchone()
        self._notify(job_id, [])
        return _checkpoint_from_row(row)

    def latest_checkpoint(self, job_id: str) -> Checkpoint | None:
        """Highest-seq checkpoint of the job, across attempts."""
        row = self._conn.execute(
            "SELECT * FROM checkpoints WHERE job_id = ? ORDER BY seq DESC LIMIT 1", (job_id,)
        ).fetchone()
        return None if row is None else _checkpoint_from_row(row)

    def checkpoints_for(self, job_id: str) -> list[Checkpoint]:
        """All checkpoints of the job, by seq."""
        rows = self._conn.execute(
            "SELECT * FROM checkpoints WHERE job_id = ? ORDER BY seq", (job_id,)
        ).fetchall()
        return [_checkpoint_from_row(r) for r in rows]

    # ------------------------------------------------------------------ providers / quota

    def get_provider_state(self, provider: str) -> ProviderState:
        """Existing row, or a default (health unknown) that is NOT inserted."""
        row = self._conn.execute(
            "SELECT * FROM provider_state WHERE provider = ?", (provider,)
        ).fetchone()
        if row is None:
            return ProviderState(provider=provider, updated_at=self._clock.now())
        return _provider_from_row(row)

    def all_provider_states(self) -> dict[str, ProviderState]:
        """Every stored provider row, keyed by provider name."""
        rows = self._conn.execute("SELECT * FROM provider_state ORDER BY provider")
        return {r["provider"]: _provider_from_row(r) for r in rows}

    def upsert_provider_state(self, provider: str, **changes: Any) -> ProviderState:
        """Insert-or-update the listed columns of provider_state (names = ProviderState
        fields). Stamps updated_at. Raises ValueError for unknown column names. After COMMIT,
        calls the listener's optional `provider_changed(provider)` hook."""
        unknown = set(changes) - _PROVIDER_COLUMNS
        if unknown:
            raise ValueError(f"unknown provider_state fields: {sorted(unknown)}")
        cols = {k: (str(v) if k == "health" and v is not None else v) for k, v in changes.items()}
        now = self._clock.now()
        names = ["provider", *cols, "updated_at"]
        values = [provider, *cols.values(), now]
        updates = ", ".join(f"{c} = excluded.{c}" for c in [*cols, "updated_at"])
        with transaction(self._conn) as cur:
            cur.execute(
                f"INSERT INTO provider_state ({', '.join(names)}) "
                f"VALUES ({_placeholders(len(names))}) "
                f"ON CONFLICT (provider) DO UPDATE SET {updates}",
                values,
            )
        self._notify_provider(provider)
        return self.get_provider_state(provider)

    def record_quota_snapshot(
        self, snapshot: QuotaSnapshot, *, keep: int = QUOTA_SNAPSHOTS_KEEP
    ) -> None:
        """Append one quota observation and prune the provider's older rows beyond the
        newest `keep` (readers only use the latest per provider; ~48 live readings a day
        for Kaggle would otherwise grow the table forever)."""
        with transaction(self._conn) as cur:
            cur.execute(
                "INSERT INTO quota_snapshots (provider, used, quota_limit, unit, resets_at, "
                "source, detail_json, observed_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    snapshot.provider,
                    snapshot.used,
                    snapshot.limit,
                    str(snapshot.unit),
                    snapshot.resets_at,
                    snapshot.source,
                    _dumps(snapshot.detail),
                    snapshot.observed_at,
                ),
            )
            cur.execute(
                "DELETE FROM quota_snapshots WHERE provider = ? AND id NOT IN ("
                "SELECT id FROM quota_snapshots WHERE provider = ? "
                "ORDER BY observed_at DESC, id DESC LIMIT ?)",
                (snapshot.provider, snapshot.provider, max(1, keep)),
            )

    def latest_quota_snapshots(self) -> dict[str, QuotaSnapshot]:
        """provider -> its most recent observation."""
        rows = self._conn.execute(
            "SELECT q.* FROM quota_snapshots q WHERE q.id = ("
            "SELECT id FROM quota_snapshots WHERE provider = q.provider "
            "ORDER BY observed_at DESC, id DESC LIMIT 1) ORDER BY q.provider"
        )
        return {r["provider"]: _quota_from_row(r) for r in rows}

    # ------------------------------------------------------------------ data cache (phase 5)

    def get_data_cache(self, content_hash: str) -> DataCacheEntry | None:
        """The uploaded copy of a dataset with this content hash, or None."""
        row = self._conn.execute(
            "SELECT * FROM data_cache WHERE content_hash = ?", (content_hash,)
        ).fetchone()
        if row is None:
            return None
        return DataCacheEntry(
            content_hash=row["content_hash"],
            uri=row["uri"],
            local_path=row["local_path"],
            size_bytes=int(row["size_bytes"]),
            file_count=int(row["file_count"]),
            uploaded_at=float(row["uploaded_at"]),
            last_used_at=float(row["last_used_at"]),
        )

    def put_data_cache(
        self,
        *,
        content_hash: str,
        uri: str,
        local_path: str,
        size_bytes: int,
        file_count: int,
    ) -> DataCacheEntry:
        """Record (or replace) an uploaded dataset; uploaded_at = last_used_at = now."""
        now = self._clock.now()
        with transaction(self._conn) as cur:
            cur.execute(
                "INSERT INTO data_cache (content_hash, uri, local_path, size_bytes, "
                "file_count, uploaded_at, last_used_at) VALUES (?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT (content_hash) DO UPDATE SET uri = excluded.uri, "
                "local_path = excluded.local_path, size_bytes = excluded.size_bytes, "
                "file_count = excluded.file_count, uploaded_at = excluded.uploaded_at, "
                "last_used_at = excluded.last_used_at",
                (content_hash, uri, local_path, size_bytes, file_count, now, now),
            )
        entry = self.get_data_cache(content_hash)
        assert entry is not None
        return entry

    def touch_data_cache(self, content_hash: str, *, local_path: str | None = None) -> None:
        """A later run reused the dataset: bump last_used_at (and the last local path)."""
        now = self._clock.now()
        with transaction(self._conn) as cur:
            if local_path is None:
                cur.execute(
                    "UPDATE data_cache SET last_used_at = ? WHERE content_hash = ?",
                    (now, content_hash),
                )
            else:
                cur.execute(
                    "UPDATE data_cache SET last_used_at = ?, local_path = ? WHERE content_hash = ?",
                    (now, local_path, content_hash),
                )

    def delete_data_cache(self, content_hash: str) -> None:
        with transaction(self._conn) as cur:
            cur.execute("DELETE FROM data_cache WHERE content_hash = ?", (content_hash,))

    def unused_data_cache(self, before: float, *, limit: int = 100) -> list[DataCacheEntry]:
        """Uploaded datasets no run used since `before`, oldest first (eviction, D44)."""
        rows = self._conn.execute(
            "SELECT content_hash FROM data_cache WHERE last_used_at < ? "
            "ORDER BY last_used_at LIMIT ?",
            (before, limit),
        ).fetchall()
        out = [self.get_data_cache(str(r["content_hash"])) for r in rows]
        return [e for e in out if e is not None]

    # ------------------------------------------------------------------ meta

    def get_meta(self, key: str) -> str | None:
        row = self._conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return None if row is None else str(row["value"])

    def set_meta(self, key: str, value: str) -> None:
        with transaction(self._conn) as cur:
            cur.execute(
                "INSERT INTO meta (key, value) VALUES (?, ?) "
                "ON CONFLICT (key) DO UPDATE SET value = excluded.value",
                (key, value),
            )


__all__ = ["AttemptChange", "DataCacheEntry", "Store", "StoreListener"]
