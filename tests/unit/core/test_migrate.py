"""Migrations: from empty, idempotent re-run, schema_too_new, backups, failures."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from gpu_router.clock import FakeClock
from gpu_router.db import migrate as migrate_mod
from gpu_router.db.connection import PRAGMAS, connect, transaction
from gpu_router.db.migrate import Migration, current_version, discover, latest_version, migrate
from gpu_router.errors import ConfigError, SchemaTooNew

EXPECTED_TABLES = {
    "schema_version",
    "meta",
    "jobs",
    "job_events",
    "checkpoints",
    "attempts",
    "provider_state",
    "quota_snapshots",
    "data_cache",
}


def _tables(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    return {r[0] for r in rows} - {"sqlite_sequence"}


def test_discover_is_contiguous_from_one() -> None:
    migrations = discover()
    assert [m.version for m in migrations] == list(range(1, len(migrations) + 1))
    assert migrations[0].name == "initial"
    assert migrations[0].filename == "0001_initial.sql"
    assert latest_version() == migrations[-1].version


def test_migrate_from_empty(clock: FakeClock) -> None:
    conn = connect(":memory:")
    assert current_version(conn) == 0
    assert migrate(conn, None, clock) == latest_version()
    assert current_version(conn) == latest_version()
    assert _tables(conn) == EXPECTED_TABLES
    row = conn.execute("SELECT version, name, applied_at FROM schema_version").fetchone()
    assert tuple(row) == (1, "initial", clock.now())
    assert not conn.in_transaction


def test_migrate_is_idempotent(clock: FakeClock) -> None:
    conn = connect(":memory:")
    migrate(conn, None, clock)
    clock.advance(10)
    assert migrate(conn, None, clock) == latest_version()
    assert conn.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0] == latest_version()


def test_schema_too_new(clock: FakeClock) -> None:
    conn = connect(":memory:")
    migrate(conn, None, clock)
    newer = latest_version() + 4
    conn.execute(
        "INSERT INTO schema_version (version, name, applied_at) VALUES (?, 'future', 0)", (newer,)
    )
    with pytest.raises(SchemaTooNew) as info:
        migrate(conn, None, clock)
    assert f"schema v{newer}" in info.value.message
    assert f"up to v{latest_version()}" in info.value.message
    assert info.value.hint
    assert "upgrade" in info.value.hint
    assert info.value.code == "schema_too_new"


def test_file_db_mode_wal_and_pragmas(tmp_path: Path, clock: FakeClock) -> None:
    db = tmp_path / "sub" / "gpu.db"
    conn = connect(db)
    migrate(conn, db, clock)
    assert db.stat().st_mode & 0o777 == 0o600
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
    assert conn.isolation_level is None
    assert len(PRAGMAS) == 5
    # No backup on a fresh database.
    assert not list(tmp_path.glob("sub/gpu.db.bak-*"))
    conn.close()


def test_backup_before_upgrading_existing_db(
    tmp_path: Path, clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "gpu.db"
    conn = connect(db)
    migrate(conn, db, clock)
    conn.execute("INSERT INTO meta (key, value) VALUES ('instance_id', 'x')")

    extra = Migration(2, "add_thing", "CREATE TABLE thing (id INTEGER PRIMARY KEY);")
    monkeypatch.setattr(migrate_mod, "discover", lambda: [*discover(), extra])
    assert migrate(conn, db, clock) == 2
    assert "thing" in _tables(conn)

    backup = tmp_path / "gpu.db.bak-v1"
    assert backup.exists()
    old = sqlite3.connect(backup)
    assert old.execute("SELECT MAX(version) FROM schema_version").fetchone()[0] == 1
    assert old.execute("SELECT value FROM meta").fetchone()[0] == "x"
    assert "thing" not in {r[0] for r in old.execute("SELECT name FROM sqlite_master")}
    old.close()
    conn.close()


def test_failed_migration_rolls_back(clock: FakeClock, monkeypatch: pytest.MonkeyPatch) -> None:
    conn = connect(":memory:")
    migrate(conn, None, clock)
    bad = Migration(2, "broken", "CREATE TABLE half (id INTEGER);\nTHIS IS NOT SQL;")
    monkeypatch.setattr(migrate_mod, "discover", lambda: [*discover(), bad])
    with pytest.raises(ConfigError) as info:
        migrate(conn, None, clock)
    assert "0002_broken.sql" in info.value.message
    assert current_version(conn) == 1
    assert "half" not in _tables(conn)
    assert not conn.in_transaction


def test_discover_rejects_gaps(monkeypatch: pytest.MonkeyPatch) -> None:
    class Fake:
        def __init__(self, name: str) -> None:
            self.name = name

        def read_text(self, encoding: str = "utf-8") -> str:
            return "SELECT 1;"

    class Dir:
        def iterdir(self) -> list[Fake]:
            return [Fake("0001_initial.sql"), Fake("0003_later.sql"), Fake("__init__.py")]

    monkeypatch.setattr(migrate_mod, "files", lambda _pkg: Dir())
    migrate_mod.discover.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="missing 0002"):
            migrate_mod.discover()
    finally:
        migrate_mod.discover.cache_clear()


def test_transaction_commits_and_rolls_back() -> None:
    conn = connect(":memory:")
    conn.execute("CREATE TABLE t (x INTEGER)")
    with transaction(conn) as cur:
        cur.execute("INSERT INTO t VALUES (1)")

    def failing_write() -> None:
        with transaction(conn) as cur:
            cur.execute("INSERT INTO t VALUES (2)")
            raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        failing_write()
    assert [r[0] for r in conn.execute("SELECT x FROM t")] == [1]
    assert not conn.in_transaction


def test_transaction_rejects_nesting() -> None:
    conn = connect(":memory:")
    with transaction(conn), pytest.raises(RuntimeError, match="nested"), transaction(conn):
        pass
