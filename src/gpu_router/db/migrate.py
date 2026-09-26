"""Schema migrations (phase 1; owner: group A).

Discovery: `importlib.resources.files("gpu_router.db.migrations")`, files named
NNNN_name.sql. Versions must be contiguous from 1; `LATEST` is the highest.

`migrate(conn, db_path, clock)`:
1. current = MAX(version) from schema_version, or 0 if the table does not exist.
2. current > LATEST  -> raise SchemaTooNew("gpu.db is schema v5 but this gpu-router knows
   up to v3", hint="upgrade gpu-router (uv tool upgrade gpu-router)").
3. current == LATEST -> return current (no-op).
4. current > 0 and a file-backed DB -> copy gpu.db to gpu.db.bak-v<current> first
   (sqlite3 backup API, so WAL contents are included).
5. For each pending migration, in order: executescript("BEGIN IMMEDIATE;" + sql +
   "INSERT INTO schema_version ...;COMMIT;"); on any error ROLLBACK and re-raise wrapped in
   ConfigError naming the migration file. One migration = one transaction.
6. Return the new version and log `db.migrate` with from/to.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from functools import cache
from importlib.resources import files
from pathlib import Path
from typing import Any

from gpu_router.clock import Clock
from gpu_router.errors import ConfigError, SchemaTooNew
from gpu_router.log import get_logger, log_event

MIGRATIONS_PACKAGE = "gpu_router.db.migrations"
_FILE_RE = re.compile(r"^(\d{4})_([a-z0-9_]+)\.sql$")

_log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class Migration:
    version: int
    name: str  # "initial" for 0001_initial.sql
    sql: str

    @property
    def filename(self) -> str:
        return f"{self.version:04d}_{self.name}.sql"


@cache
def discover() -> list[Migration]:
    """All packaged migrations sorted by version. Raises RuntimeError on gaps/duplicates."""
    found: dict[int, Migration] = {}
    for entry in files(MIGRATIONS_PACKAGE).iterdir():
        match = _FILE_RE.match(entry.name)
        if not match:
            continue
        version = int(match.group(1))
        if version in found:
            raise RuntimeError(f"duplicate migration version {version:04d}")
        found[version] = Migration(version, match.group(2), entry.read_text(encoding="utf-8"))
    ordered = [found[v] for v in sorted(found)]
    for expected, mig in enumerate(ordered, start=1):
        if mig.version != expected:
            raise RuntimeError(f"migrations are not contiguous: missing {expected:04d}")
    if not ordered:
        raise RuntimeError(f"no migrations found in {MIGRATIONS_PACKAGE}")
    return ordered


def latest_version() -> int:
    """Highest packaged migration version."""
    return discover()[-1].version


def current_version(conn: sqlite3.Connection) -> int:
    """MAX(schema_version.version), or 0 for an empty database."""
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'schema_version'"
    ).fetchone()
    if row is None:
        return 0
    value = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]
    return int(value) if value is not None else 0


def _backup(conn: sqlite3.Connection, db_path: Path, version: int) -> Path:
    dest_path = db_path.with_name(f"{db_path.name}.bak-v{version}")
    dest = sqlite3.connect(dest_path)
    try:
        conn.backup(dest)
    finally:
        dest.close()
    dest_path.chmod(0o600)
    return dest_path


def _apply(conn: sqlite3.Connection, mig: Migration, now: float) -> None:
    script = (
        "BEGIN IMMEDIATE;\n"
        f"{mig.sql}\n;\n"
        "INSERT INTO schema_version (version, name, applied_at) "
        f"VALUES ({mig.version}, '{mig.name}', {now!r});\n"
        "COMMIT;"
    )
    try:
        conn.executescript(script)
    except sqlite3.Error as exc:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise ConfigError(
            f"database migration {mig.filename} failed: {exc}",
            hint="gpu.db was left at the previous schema version; report this bug",
            detail={"migration": mig.filename},
        ) from exc


def migrate(conn: sqlite3.Connection, db_path: Path | None, clock: Clock) -> int:
    """Bring the database to latest_version() as described in the module docstring.
    `db_path=None` (in-memory) skips the backup step."""
    migrations = discover()
    latest = migrations[-1].version
    current = current_version(conn)
    if current > latest:
        raise SchemaTooNew(
            f"gpu.db is schema v{current} but this gpu-router knows up to v{latest}",
            hint="upgrade gpu-router (uv tool upgrade gpu-router)",
            detail={"db_version": current, "known_version": latest},
        )
    if current == latest:
        return current
    if current > 0 and db_path is not None:
        _backup(conn, db_path, current)
    for mig in migrations:
        if mig.version > current:
            _apply(conn, mig, clock.now())
    fields: dict[str, Any] = {"from": current, "to": latest}
    log_event(
        _log, "db.migrate", f"database schema migrated from v{current} to v{latest}", **fields
    )
    return latest
