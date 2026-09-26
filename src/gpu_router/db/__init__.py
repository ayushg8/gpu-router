"""SQLite access (phase 1; owner: group A).

- connection.py: open a connection with the required pragmas; the `transaction()` helper.
- migrate.py:    discover migrations/NNNN_*.sql and bring a database to the latest version.

Only gpu_router.store uses this package, and only inside the daemon (invariant 1).
"""
