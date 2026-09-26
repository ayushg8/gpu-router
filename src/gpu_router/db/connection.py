"""SQLite connection setup (phase 1; owner: group A).

`connect()` opens gpu.db in autocommit mode (`isolation_level=None`) so transactions are
explicit, and applies, in order:

    PRAGMA journal_mode = WAL;       -- readers never block the single writer
    PRAGMA synchronous = NORMAL;     -- durable across process crashes (WAL), fast
    PRAGMA foreign_keys = ON;
    PRAGMA busy_timeout = 5000;
    PRAGMA temp_store = MEMORY;

`check_same_thread=True`: the connection belongs to the daemon's event-loop thread
(invariant 9). Row factory is sqlite3.Row.

`transaction(conn)` is the only way to write: BEGIN IMMEDIATE ... COMMIT, ROLLBACK on any
exception (re-raised). Nesting is a programming error and raises RuntimeError.
"""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

PRAGMAS: tuple[str, ...] = (
    "PRAGMA journal_mode = WAL",
    "PRAGMA synchronous = NORMAL",
    "PRAGMA foreign_keys = ON",
    "PRAGMA busy_timeout = 5000",
    "PRAGMA temp_store = MEMORY",
)


def connect(db_path: Path | str) -> sqlite3.Connection:
    """Open (creating if needed, file mode 0600) and configure a connection.
    `":memory:"` is accepted for unit tests (WAL pragma is then a no-op)."""
    target = str(db_path)
    if target != ":memory:":
        path = Path(target)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            # Create with 0600 before sqlite opens it (sqlite would use the umask).
            os.close(os.open(path, os.O_RDWR | os.O_CREAT, 0o600))
    conn = sqlite3.connect(target, isolation_level=None, check_same_thread=True)
    conn.row_factory = sqlite3.Row
    try:
        for pragma in PRAGMAS:
            conn.execute(pragma)
    except BaseException:
        conn.close()
        raise
    return conn


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Cursor]:
    """BEGIN IMMEDIATE; yield a cursor; COMMIT. ROLLBACK and re-raise on exception."""
    if _in_transaction(conn):
        raise RuntimeError("nested transaction(): a transaction is already open")
    cur = conn.cursor()
    cur.execute("BEGIN IMMEDIATE")
    try:
        yield cur
    except BaseException:
        if _in_transaction(conn):
            conn.execute("ROLLBACK")
        raise
    else:
        try:
            conn.execute("COMMIT")
        except BaseException:
            if _in_transaction(conn):
                conn.execute("ROLLBACK")
            raise
    finally:
        cur.close()


def _in_transaction(conn: sqlite3.Connection) -> bool:
    # A function (not the attribute inline) so type checkers do not narrow it across the
    # statements that open/close the transaction.
    return bool(conn.in_transaction)
