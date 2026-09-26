"""The schema's CHECK constraints must mirror the Python enums (CLAUDE.md "Data model")."""

from __future__ import annotations

import re
import sqlite3

import pytest

from gpu_router.clock import FakeClock
from gpu_router.db.connection import connect
from gpu_router.db.migrate import discover, migrate
from gpu_router.models import FailureKind, ProviderHealth, QuotaUnit, Source
from gpu_router.statemachine import AttemptState, JobState

SQL = discover()[0].sql


def _check_values(column: str) -> set[str]:
    """The quoted values listed in `<column> ... CHECK (<column> IN (...))`."""
    match = re.search(rf"CHECK \({column} IN \((.*?)\)\)", SQL, re.S)
    assert match, column
    return set(re.findall(r"'([a-z_]+)'", match.group(1)))


def _column_checks(table: str, column: str) -> set[str]:
    body = re.search(rf"CREATE TABLE {table} \((.*?)\n\);", SQL, re.S)
    assert body, table
    match = re.search(
        rf"\n\s+{column}\s+TEXT.*?CHECK \({column} IN \((.*?)\)\)", body.group(1), re.S
    )
    assert match, (table, column)
    return set(re.findall(r"'([a-z_]+)'", match.group(1)))


def test_job_states_match() -> None:
    assert _column_checks("jobs", "state") == {s.value for s in JobState}


def test_attempt_states_match() -> None:
    assert _column_checks("attempts", "state") == {s.value for s in AttemptState}


def test_other_enums_match() -> None:
    assert _column_checks("jobs", "source") == {s.value for s in Source}
    assert _column_checks("jobs", "failure_kind") == {s.value for s in FailureKind}
    assert _column_checks("provider_state", "health") == {s.value for s in ProviderHealth}
    assert _column_checks("quota_snapshots", "unit") == {s.value for s in QuotaUnit}


def test_terminal_state_lists_match() -> None:
    from gpu_router.statemachine import TERMINAL_ATTEMPT_STATES, TERMINAL_STATES

    job_terminal = re.search(r"CHECK \(\(state IN \((.*?)\)\) = \(finished_at", SQL, re.S)
    assert job_terminal
    assert set(re.findall(r"'([a-z_]+)'", job_terminal.group(1))) == {
        s.value for s in TERMINAL_STATES
    }
    attempts = SQL[SQL.index("CREATE TABLE attempts") :]
    att_terminal = re.search(r"CHECK \(\(state IN \(([^)]*)\)\)\s*= \(ended_at", attempts)
    assert att_terminal
    assert set(re.findall(r"'([a-z_]+)'", att_terminal.group(1))) == {
        s.value for s in TERMINAL_ATTEMPT_STATES
    }
    live = re.search(r"attempts_one_live_per_job.*?WHERE state IN \((.*?)\)", SQL, re.S)
    assert live
    assert set(re.findall(r"'([a-z_]+)'", live.group(1))) == {
        "submitting",
        "submitted",
        "running",
    }


@pytest.fixture
def conn() -> sqlite3.Connection:
    c = connect(":memory:")
    migrate(c, None, FakeClock())
    return c


def _insert_job(conn: sqlite3.Connection, **over: object) -> None:
    row: dict[str, object] = {
        "id": "abcdef012345",
        "name": "t",
        "state": "queued",
        "source": "cli",
        "project_dir": "/p",
        "spec_json": "{}",
        "spec_hash": "h",
        "created_at": 1.0,
        "updated_at": 1.0,
    }
    row.update(over)
    cols = ", ".join(row)
    conn.execute(
        f"INSERT INTO jobs ({cols}) VALUES ({', '.join('?' * len(row))})", list(row.values())
    )


@pytest.mark.parametrize(
    "over",
    [
        {"state": "bogus"},
        {"id": "ABCDEF012345"},
        {"id": "abc"},
        {"source": "web"},
        {"state": "done"},  # terminal without finished_at
        {"finished_at": 2.0},  # finished_at on a live job
        {"state": "failed", "finished_at": 2.0},  # failed without failure_kind
        {"failure_kind": "user_error"},  # failure_kind on a non-failed job
    ],
)
def test_job_checks_reject(conn: sqlite3.Connection, over: dict[str, object]) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        _insert_job(conn, **over)


def test_job_checks_accept_valid_rows(conn: sqlite3.Connection) -> None:
    _insert_job(conn)
    _insert_job(
        conn,
        id="abcdef012346",
        state="failed",
        finished_at=2.0,
        failure_kind="user_error",
    )


def test_one_live_attempt_per_job(conn: sqlite3.Connection) -> None:
    _insert_job(conn)
    ins = (
        "INSERT INTO attempts (id, job_id, n, provider, attempt_key, state, created_at) "
        "VALUES (?, 'abcdef012345', ?, 'fake', ?, 'submitting', 1.0)"
    )
    conn.execute(ins, ("abcdef012345.1", 1, "gpu-abcdef012345-1"))
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(ins, ("abcdef012345.2", 2, "gpu-abcdef012345-2"))


def test_attempt_needs_remote_id_once_submitted(conn: sqlite3.Connection) -> None:
    _insert_job(conn)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO attempts (id, job_id, n, provider, attempt_key, state, created_at) "
            "VALUES ('abcdef012345.1', 'abcdef012345', 1, 'fake', 'k', 'submitted', 1.0)"
        )


def test_foreign_keys_enforced(conn: sqlite3.Connection) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO job_events (job_id, kind, to_state, reason, message, actor, ts) "
            "VALUES ('000000000000', 'transition', 'queued', 'submitted', 'm', 'a', 1.0)"
        )
