"""SQL migration files, applied in order by gpu_router.db.migrate.

Naming: NNNN_snake_name.sql, contiguous from 0001. Each file is plain SQL executed inside
one transaction together with its schema_version row. Append-only once a phase ships
(invariant 21): change the schema with a new file, never by editing an applied one.
"""
