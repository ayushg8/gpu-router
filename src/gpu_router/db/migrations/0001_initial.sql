-- gpu-router schema v1.
--
-- Applied by gpu_router.db.migrate inside ONE transaction, followed by
--   INSERT INTO schema_version(version, name, applied_at) VALUES (1, 'initial', ?).
-- Conventions: timestamps are REAL unix epoch seconds (UTC); *_json columns hold compact
-- JSON objects; enums are TEXT guarded by CHECK constraints that mirror
-- gpu_router.statemachine / gpu_router.models (tests/unit/test_schema.py keeps them in sync).
-- Editable only until phase 1 is marked done; afterwards add 0002_*.sql instead.

CREATE TABLE schema_version (
    version     INTEGER PRIMARY KEY,
    name        TEXT    NOT NULL,
    applied_at  REAL    NOT NULL
);

-- Small daemon facts: instance_id, last_start_at, last_clean_shutdown_at, last_pid.
CREATE TABLE meta (
    key    TEXT PRIMARY KEY,
    value  TEXT NOT NULL
);

-- ----------------------------------------------------------------------------- jobs
CREATE TABLE jobs (
    id                   TEXT    PRIMARY KEY
                                 CHECK (length(id) = 12 AND id NOT GLOB '*[^0-9a-f]*'),
    name                 TEXT    NOT NULL,
    state                TEXT    NOT NULL CHECK (state IN (
                             'queued', 'routing', 'awaiting_approval', 'provisioning',
                             'running', 'checkpointing', 'migrating', 'cancelling',
                             'done', 'failed', 'cancelled', 'denied')),
    source               TEXT    NOT NULL CHECK (source IN ('cli', 'shell', 'agent', 'api')),
    request_id           TEXT    UNIQUE,            -- client idempotency key (Idempotency-Key)
    project_dir          TEXT    NOT NULL,
    spec_json            TEXT    NOT NULL,          -- JobSpec.model_dump_json()
    spec_hash            TEXT    NOT NULL,          -- sha256 of spec_json
    bundle_sha256        TEXT,                      -- set once the bundle is built
    provider             TEXT,                      -- provider of the current/last attempt
    gpu                  TEXT,                      -- GPU of the current/last attempt
    current_attempt_id   TEXT,                      -- attempts.id (no FK: circular)
    attempt_count        INTEGER NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
    accepted_attempts    INTEGER NOT NULL DEFAULT 0 CHECK (accepted_attempts >= 0),
    route_reason         TEXT,                      -- one line, e.g. "colab: fits 16GB, ..."
    approval_reason      TEXT,                      -- why approval is/was needed
    approved_at          REAL,
    approved_by          TEXT,
    not_before           REAL,                      -- engine does nothing before this
    waiting_since        REAL,                      -- start of the current placement wait
    cancel_requested_at  REAL,
    progress_step        INTEGER,
    progress_total       INTEGER,
    progress_source      TEXT    CHECK (progress_source IN ('helper', 'stdout')),
    last_metrics_json    TEXT    NOT NULL DEFAULT '{}',
    checkpoint_count     INTEGER NOT NULL DEFAULT 0,
    last_checkpoint_at   REAL,
    outputs_dir          TEXT,                      -- <project_dir>/runs/<short_id>, fixed at creation
    outputs_fetched      INTEGER NOT NULL DEFAULT 0 CHECK (outputs_fetched IN (0, 1)),
    exit_code            INTEGER,
    failure_kind         TEXT    CHECK (failure_kind IN (
                             'user_error', 'no_provider', 'provider_error',
                             'invalid_job', 'internal')),
    message              TEXT    NOT NULL DEFAULT '', -- latest human status line
    created_at           REAL    NOT NULL,
    updated_at           REAL    NOT NULL,
    started_at           REAL,                      -- first time any attempt ran
    finished_at          REAL,                      -- set iff terminal
    version              INTEGER NOT NULL DEFAULT 0, -- bumped on every write
    CHECK ((state IN ('done', 'failed', 'cancelled', 'denied')) = (finished_at IS NOT NULL)),
    CHECK ((state = 'failed') = (failure_kind IS NOT NULL))
);
CREATE INDEX jobs_state   ON jobs (state);
CREATE INDEX jobs_created ON jobs (created_at DESC);
CREATE INDEX jobs_project ON jobs (project_dir, created_at DESC);
CREATE INDEX jobs_finished ON jobs (finished_at DESC) WHERE finished_at IS NOT NULL;

-- ----------------------------------------------------------------------------- job_events
-- Every state change inserts one kind='transition' row in the same transaction as the
-- jobs.state update. kind='note' rows record non-state facts (retry scheduled, recovered).
CREATE TABLE job_events (
    seq          INTEGER PRIMARY KEY AUTOINCREMENT,  -- global, monotonic: feed cursor
    job_id       TEXT    NOT NULL REFERENCES jobs (id) ON DELETE CASCADE,
    attempt_id   TEXT,
    kind         TEXT    NOT NULL CHECK (kind IN ('transition', 'note')),
    from_state   TEXT,                               -- NULL only for the creation event
    to_state     TEXT,
    reason       TEXT    NOT NULL,                   -- statemachine.Reason value
    message      TEXT    NOT NULL,                   -- what happened + what the tool did next
    detail_json  TEXT    NOT NULL DEFAULT '{}',
    actor        TEXT    NOT NULL,                   -- engine | recovery | user:<front end> | agent | api
    ts           REAL    NOT NULL,
    CHECK ((kind = 'transition') = (to_state IS NOT NULL))
);
CREATE INDEX job_events_job ON job_events (job_id, seq);

-- ----------------------------------------------------------------------------- checkpoints
-- Written from the runner's ckpt_end protocol lines (phase 1 with the fake; HF Hub in phase 5).
CREATE TABLE checkpoints (
    id           TEXT    PRIMARY KEY,                -- "<job_id>.c<seq>"
    job_id       TEXT    NOT NULL REFERENCES jobs (id) ON DELETE CASCADE,
    attempt_id   TEXT    NOT NULL REFERENCES attempts (id),
    seq          INTEGER NOT NULL CHECK (seq >= 1), -- monotonic per job across attempts
    step         INTEGER,
    uri          TEXT    NOT NULL,                   -- hf://... or fake://...
    size_bytes   INTEGER,
    sha256       TEXT,
    created_at   REAL    NOT NULL,                   -- when the runner finished uploading
    recorded_at  REAL    NOT NULL,                   -- when the daemon saw it
    UNIQUE (job_id, seq)
);

-- ----------------------------------------------------------------------------- attempts
-- One row per placement. The row (state 'submitting', attempt_key) is committed BEFORE
-- adapter.submit() is called; that is what makes crash recovery double-submit-safe.
-- Also the usage ledger: GPU time per provider = sum(ended_at|now - started_at).
CREATE TABLE attempts (
    id                    TEXT    PRIMARY KEY,       -- "<job_id>.<n>"
    job_id                TEXT    NOT NULL REFERENCES jobs (id) ON DELETE CASCADE,
    n                     INTEGER NOT NULL CHECK (n >= 1),
    provider              TEXT    NOT NULL,
    attempt_key           TEXT    NOT NULL UNIQUE,   -- "gpu-<job_id>-<n>"
    state                 TEXT    NOT NULL CHECK (state IN (
                              'submitting', 'submitted', 'running', 'succeeded', 'failed',
                              'lost', 'cancelled', 'rejected', 'abandoned')),
    remote_id             TEXT,
    remote_url            TEXT,
    remote_meta_json      TEXT    NOT NULL DEFAULT '{}', -- RemoteRef.meta (adapter-private)
    remote_message        TEXT,                      -- last RemoteStatus.message ("waiting for GPU")
    gpu                   TEXT,
    route_reason          TEXT,
    resume_checkpoint_id  TEXT    REFERENCES checkpoints (id),
    error_kind            TEXT,                      -- AdapterError class name
    error_message         TEXT,
    lost_reason           TEXT,
    exit_code             INTEGER,
    log_lines             INTEGER NOT NULL DEFAULT 0, -- cache; the log file is the truth
    log_cursor            TEXT,                      -- LogChunk.cursor after the last persisted chunk
    session_deadline      REAL,                      -- started_at + provider session cap (phase 5 handoff)
    created_at            REAL    NOT NULL,
    submitted_at          REAL,
    started_at            REAL,
    ended_at              REAL,
    last_seen_at          REAL,
    UNIQUE (job_id, n),
    CHECK ((state IN ('succeeded', 'failed', 'lost', 'cancelled', 'rejected', 'abandoned'))
           = (ended_at IS NOT NULL)),
    CHECK (state = 'submitting' OR state IN ('rejected', 'abandoned', 'cancelled')
           OR remote_id IS NOT NULL)
);
-- Invariant 5: at most one live attempt per job, enforced by the database.
CREATE UNIQUE INDEX attempts_one_live_per_job ON attempts (job_id)
    WHERE state IN ('submitting', 'submitted', 'running');
CREATE INDEX attempts_job              ON attempts (job_id, n);
CREATE INDEX attempts_provider_started ON attempts (provider, started_at);

-- ----------------------------------------------------------------------------- provider_state
-- Engine-maintained runtime health per provider (survives restarts).
CREATE TABLE provider_state (
    provider              TEXT    PRIMARY KEY,
    health                TEXT    NOT NULL DEFAULT 'unknown' CHECK (health IN (
                              'unknown', 'ok', 'degraded', 'unavailable',
                              'auth_required', 'disabled')),
    health_reason         TEXT,
    last_healthcheck_at   REAL,
    cooldown_until        REAL,                      -- after RateLimited / Unavailable
    consecutive_failures  INTEGER NOT NULL DEFAULT 0,
    exhausted_until       REAL,                      -- after QuotaExhausted
    updated_at            REAL    NOT NULL
);

-- ----------------------------------------------------------------------------- quota (phase 5)
-- Observations from adapter.quota() (source 'live') or the ledger's own estimates.
CREATE TABLE quota_snapshots (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    provider     TEXT    NOT NULL,
    used         REAL    NOT NULL CHECK (used >= 0),
    quota_limit  REAL,
    unit         TEXT    NOT NULL CHECK (unit IN ('gpu_hours', 'credits', 'usd')),
    resets_at    REAL,
    source       TEXT    NOT NULL CHECK (source IN ('live', 'estimate')),
    detail_json  TEXT    NOT NULL DEFAULT '{}',
    observed_at  REAL    NOT NULL
);
CREATE INDEX quota_snapshots_provider ON quota_snapshots (provider, observed_at DESC);

-- ----------------------------------------------------------------------------- data cache (phase 5)
-- Datasets uploaded to HF Hub once, keyed by content hash, reused forever.
CREATE TABLE data_cache (
    content_hash  TEXT    PRIMARY KEY,               -- sha256 over sorted (relpath, file sha256)
    uri           TEXT    NOT NULL,                  -- hf://datasets/<owner>/<repo>@<rev>
    local_path    TEXT    NOT NULL,                  -- last local path seen with this hash
    size_bytes    INTEGER NOT NULL,
    file_count    INTEGER NOT NULL,
    uploaded_at   REAL    NOT NULL,
    last_used_at  REAL    NOT NULL
);
