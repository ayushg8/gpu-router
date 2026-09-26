"""Domain models (phase 1; real code, owner: group A).

These pydantic models are declarations, so they are complete now and every group can build
against them in parallel. Rules:

- Timestamps are `Timestamp`: a float of unix epoch seconds (UTC) in Python and SQLite,
  serialized as ISO-8601 UTC with a trailing `Z` in JSON (API, `--json`). Validation accepts
  either form. Never use naive datetimes.
- Models the API returns are frozen; mutate through the store (JobPatch / AttemptPatch).
- `JobSpec` is the job as submitted. The gpu.yaml + flags merge (phase 2) produces one.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PlainSerializer,
    field_validator,
    model_validator,
)

from gpu_router.statemachine import AttemptState, JobState, Reason

__all__ = [
    "Attempt",
    "AttemptPatch",
    "AttemptState",
    "Checkpoint",
    "DataRef",
    "DepsSpec",
    "FailureKind",
    "Job",
    "JobEvent",
    "JobPatch",
    "JobSpec",
    "JobState",
    "Progress",
    "ProviderHealth",
    "ProviderState",
    "QuotaSnapshot",
    "QuotaUnit",
    "Reason",
    "Source",
    "Timestamp",
    "to_iso",
]


# --------------------------------------------------------------------------- Timestamp


def to_iso(ts: float) -> str:
    """Epoch seconds -> '2026-09-23T12:00:00.000Z'."""
    return datetime.fromtimestamp(ts, UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _parse_ts(value: Any) -> Any:
    if isinstance(value, bool):
        raise ValueError("timestamp must be a number or ISO-8601 string")
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ValueError("naive datetimes are not allowed; use UTC")
        return value.timestamp()
    if isinstance(value, str):
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            raise ValueError("timestamp string must carry a timezone (use Z)")
        return dt.timestamp()
    return value


Timestamp = Annotated[
    float,
    BeforeValidator(_parse_ts),
    PlainSerializer(to_iso, return_type=str, when_used="json"),
]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", use_enum_values=False)


# --------------------------------------------------------------------------- enums


class Source(StrEnum):
    """Who submitted the job. Agents are subject to the approval policy (phase 5)."""

    CLI = "cli"
    SHELL = "shell"
    AGENT = "agent"
    API = "api"


class FailureKind(StrEnum):
    USER_ERROR = "user_error"  # the user's code exited non-zero
    NO_PROVIDER = "no_provider"  # nothing fits, or attempt/wait budget exhausted
    PROVIDER_ERROR = "provider_error"  # adapter raised Permanent
    INVALID_JOB = "invalid_job"  # every provider rejected the job as invalid
    INTERNAL = "internal"  # a gpu-router bug (internal error limit hit)


class ProviderHealth(StrEnum):
    UNKNOWN = "unknown"
    OK = "ok"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    AUTH_REQUIRED = "auth_required"
    DISABLED = "disabled"


class QuotaUnit(StrEnum):
    GPU_HOURS = "gpu_hours"
    CREDITS = "credits"
    USD = "usd"


# --------------------------------------------------------------------------- JobSpec

#: Env var names that look like credentials. JobSpec.env rejects them: secrets go through
#: JobSpec.secrets (keyring names) so they never land in the DB, bundle or logs. This is
#: also what stored specs are re-validated against on every load, so it only ever grows
#: together with a migration; new intake rules go in SECRET_ENV_NAME_STRICT.
SECRET_ENV_NAME = re.compile(
    r"(TOKEN|SECRET|PASSWORD|PASSWD|API_?KEY|ACCESS_?KEY|PRIVATE_?KEY|CREDENTIAL|AUTH)",
    re.IGNORECASE,
)
#: The intake rule (D39): SECRET_ENV_NAME plus `*_KEY` (KAGGLE_KEY, WANDB_KEY, OPENAI_KEY,
#: SSH_KEY), `*_PASS`, webhooks, DSNs and URLs that usually embed credentials. Checked by
#: `secret_env_problem` wherever a new spec enters (daemon submit/route, CLI), not by the
#: JobSpec validator, so specs stored before it keep loading.
SECRET_ENV_NAME_STRICT = re.compile(
    SECRET_ENV_NAME.pattern
    + r"|(^|_)KEY$|(^|_)PASS$|WEBHOOK|(^|_)DSN$|(^|_)COOKIE$"
    + r"|^(DATABASE|DB|REDIS|MONGO|MONGODB|POSTGRES|POSTGRESQL|PG|MYSQL|AMQP|BROKER"
    + r"|CELERY_BROKER|RABBITMQ)_UR[LI]$",
    re.IGNORECASE,
)


def secret_env_problem(env: dict[str, str]) -> str | None:
    """Why `env` must not be accepted for a NEW job, or None. Names matching
    SECRET_ENV_NAME_STRICT, and values that gpu_router.secrets.redact would change (a
    token-shaped value, `--env HF=hf_...`), are refused: env is published as-is into
    Kaggle kernel source and Colab exec code, and stored in spec_json (invariant 12). The
    message never contains the value."""
    from gpu_router import secrets

    for key, value in env.items():
        hint = f"store it with `gpu secrets set {key}` and list it under `secrets:` instead"
        if SECRET_ENV_NAME_STRICT.search(key):
            return f"env var {key!r} looks like a secret; {hint}"
        if secrets.redact(str(value)) != str(value):
            return f"env var {key!r} holds what looks like a credential; {hint}"
    return None


#: Keychain names gpu-router keeps its OWN credentials under (service "gpu-router", the same
#: place job secrets live): provider logins, the admin and remote HF tokens, the storage
#: token the engine injects. A job's `secrets:` may never name one (D48): the runner would
#: export it into the job's environment on a remote GPU. Compared case-insensitively.
RESERVED_SECRET_NAMES: frozenset[str] = frozenset(
    {
        "KAGGLE",  # the whole kaggle.json (providers/kaggle/credentials.py KEYCHAIN_JSON)
        "KAGGLE_API_TOKEN",
        "KAGGLE_KEY",
        "KAGGLE_USERNAME",
        "HF_TOKEN",  # the admin token (checkpoint/tokens.py ADMIN_SECRET)
        "HF_TOKEN_REMOTE",  # the remote runtimes' token (REMOTE_SECRET)
        "GPU_STORAGE_TOKEN",  # injected by the engine itself
        "MODAL_TOKEN_ID",
        "MODAL_TOKEN_SECRET",
        "LIGHTNING_USER_ID",
        "LIGHTNING_API_KEY",
        "LIGHTNING_AUTH_TOKEN",
        "COLAB",
        # phase 7b: inference-lane keys (inference/keys.py; `gpu login groq|gemini|cloudflare`)
        "INFER_GROQ_API_KEY",
        "INFER_GEMINI_API_KEY",
        "INFER_CLOUDFLARE_API_TOKEN",
        "INFER_CLOUDFLARE_ACCOUNT_ID",
    }
)


def is_reserved_secret(name: str) -> bool:
    return name.upper() in RESERVED_SECRET_NAMES


def secret_names_problem(names: list[str]) -> str | None:
    """Why a NEW job must not list these `secrets:` names, or None. Checked wherever a spec
    enters (daemon submit/route, CLI, MCP) and again right before a submit (a stored spec
    is not re-validated against it, D39)."""
    for name in names:
        if is_reserved_secret(name):
            return (
                f"secret {name!r} is one of gpu-router's own provider credentials; jobs never "
                "get it. store the job's own token under another name (`gpu secrets set "
                "MY_HF_TOKEN`) and list that under `secrets:`"
            )
    return None


_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
#: C0 and C1 control characters, DEL and the bidi overrides: never part of a name shown in
#: a terminal (an ESC in a job name would reach the status line as an escape sequence, D48).
CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f-\x9f‪-‮⁦-⁩]")


def strip_controls(text: str) -> str:
    """`text` without control characters (tabs and newlines become spaces)."""
    if text.isprintable():
        return text
    return CONTROL_CHARS.sub(lambda m: " " if m.group() in "\t\n\r" else "", text)


_PROVIDER_NAME = re.compile(r"^[a-z][a-z0-9-]{0,31}$")


class DepsSpec(_Frozen):
    """How the remote installs dependencies. `auto` is resolved at bundle time (phase 2):
    requirements.txt if present, else pyproject.toml, else none."""

    kind: Literal["auto", "requirements", "pyproject", "none"] = "auto"
    file: str | None = None  # relative to project_dir; required for requirements/pyproject


class DataRef(_Frozen):
    """A dataset the job reads. Local paths are uploaded to HF Hub once, cached by content
    hash (phase 5) and mounted read-only on the remote at /data/<mount> (via GPU_DATA_DIR)."""

    mount: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,63}$")
    path: str | None = None  # local file/dir (absolute or relative to project_dir)
    uri: str | None = None  # already-remote: hf://datasets/<owner>/<repo>[@rev][/sub/path]

    @model_validator(mode="after")
    def _one_source(self) -> DataRef:
        if (self.path is None) == (self.uri is None):
            raise ValueError("DataRef needs exactly one of path or uri")
        return self


class JobSpec(_Frozen):
    """A job as submitted. Frozen, validated, stored verbatim in jobs.spec_json.

    Exactly one of `script` / `command`. Everything except project_dir and the entrypoint
    has a default so `gpu run train.py` works with zero flags.
    """

    spec_version: Literal[1] = 1
    name: str | None = Field(default=None, max_length=80)
    project_dir: str
    script: str | None = None  # relative to project_dir, e.g. "train.py"
    command: list[str] | None = None  # argv run from project root, e.g. ["bash", "run.sh"]
    args: list[str] = Field(default_factory=list)

    vram_gb: float | None = Field(default=None, gt=0, le=640)  # None = estimate / any GPU
    hours: float | None = Field(default=None, gt=0, le=24 * 14)  # expected runtime
    provider: str | None = None  # hard override: route only here
    gpu: str | None = None  # GPU type constraint, e.g. "T4", "A100"

    env: dict[str, str] = Field(default_factory=dict)  # NON-secret env vars
    secrets: list[str] = Field(default_factory=list)  # keyring names exposed as env
    deps: DepsSpec = Field(default_factory=DepsSpec)
    data: list[DataRef] = Field(default_factory=list)

    checkpoint_interval_min: int = Field(default=20, ge=0, le=24 * 60)  # 0 disables sync
    interactive: bool = False
    requires_approval: bool = False  # submitter asks for approval regardless of policy
    smoke: bool = False  # phase 5: a quick smoke test; the router prefers the local Mac (MPS)
    source: Source = Source.CLI
    max_attempts: int | None = Field(default=None, ge=1, le=50)  # overrides engine default
    provider_options: dict[str, dict[str, Any]] = Field(default_factory=dict)
    """Opaque per-adapter knobs keyed by provider name (the fake reads its directives from
    provider_options[<its name>] or provider_options["fake"]). The router ignores them."""
    labels: dict[str, str] = Field(default_factory=dict)
    """Free-form, non-secret client metadata (e.g. {"agent": "claude-code", "session": "..."}).
    Stored with the spec, shown in job detail, never interpreted by the engine. Lets later
    phases (MCP, shell) attach context without a schema change."""

    @field_validator("name")
    @classmethod
    def _printable_name(cls, v: str | None) -> str | None:
        """Names are shown in terminals, the status line and agent context: control
        characters are dropped (not refused, so a stored spec always loads, D48)."""
        if v is None:
            return v
        clean = " ".join(strip_controls(v).split())
        return clean or None

    @field_validator("project_dir")
    @classmethod
    def _abs_project(cls, v: str) -> str:
        if not v.startswith("/"):
            raise ValueError("project_dir must be an absolute path")
        return v.rstrip("/") or "/"

    @field_validator("script")
    @classmethod
    def _relative_script(cls, v: str | None) -> str | None:
        if v is None:
            return v
        p = PurePosixPath(v)
        if p.is_absolute() or ".." in p.parts:
            raise ValueError("script must be a path inside project_dir (relative, no '..')")
        return str(p)

    @field_validator("provider")
    @classmethod
    def _provider_name(cls, v: str | None) -> str | None:
        if v is not None and not _PROVIDER_NAME.match(v):
            raise ValueError(f"not a provider name: {v!r}")
        return v

    @field_validator("env")
    @classmethod
    def _no_secret_env(cls, v: dict[str, str]) -> dict[str, str]:
        for key in v:
            if not _ENV_NAME.match(key):
                raise ValueError(f"invalid env var name: {key!r}")
            if SECRET_ENV_NAME.search(key):
                raise ValueError(
                    f"env var {key!r} looks like a secret; store it with `gpu secrets set "
                    f"{key}` and list it under `secrets:` instead"
                )
        return v

    @field_validator("secrets")
    @classmethod
    def _secret_names(cls, v: list[str]) -> list[str]:
        for key in v:
            if not _ENV_NAME.match(key):
                raise ValueError(f"invalid secret name: {key!r}")
        return v

    @model_validator(mode="after")
    def _one_entrypoint(self) -> JobSpec:
        if (self.script is None) == (self.command is None):
            raise ValueError("give exactly one of script or command")
        if self.command is not None and not self.command:
            raise ValueError("command must not be empty")
        return self

    def display_name(self) -> str:
        """`name`, else the script stem, else the command's first word."""
        if self.name:
            return self.name
        if self.script:
            return PurePosixPath(self.script).stem
        assert self.command
        return PurePosixPath(self.command[0]).name


# --------------------------------------------------------------------------- persisted records


class Progress(_Frozen):
    step: int | None = None
    total: int | None = None
    source: Literal["helper", "stdout"] | None = None

    @property
    def fraction(self) -> float | None:
        if self.step is None or not self.total:
            return None
        return max(0.0, min(1.0, self.step / self.total))


class Job(_Frozen):
    """One row of `jobs`, as the store returns it. `short_id` is the shortest unique id
    prefix (>= 4 chars) at read time."""

    id: str
    short_id: str
    name: str
    state: JobState
    source: Source
    request_id: str | None = None
    project_dir: str
    spec: JobSpec
    spec_hash: str
    bundle_sha256: str | None = None
    provider: str | None = None
    gpu: str | None = None
    current_attempt_id: str | None = None
    attempt_count: int = 0
    accepted_attempts: int = 0
    route_reason: str | None = None
    approval_reason: str | None = None
    approved_at: Timestamp | None = None
    approved_by: str | None = None
    not_before: Timestamp | None = None
    waiting_since: Timestamp | None = None
    cancel_requested_at: Timestamp | None = None
    progress: Progress = Field(default_factory=Progress)
    last_metrics: dict[str, float] = Field(default_factory=dict)
    checkpoint_count: int = 0
    last_checkpoint_at: Timestamp | None = None
    outputs_dir: str | None = None
    outputs_fetched: bool = False
    exit_code: int | None = None
    failure_kind: FailureKind | None = None
    message: str = ""
    created_at: Timestamp
    updated_at: Timestamp
    started_at: Timestamp | None = None
    finished_at: Timestamp | None = None
    version: int = 0


class Attempt(_Frozen):
    """One placement of a job on one provider (row of `attempts`)."""

    id: str  # "<job_id>.<n>"
    job_id: str
    n: int
    provider: str
    attempt_key: str  # "gpu-<job_id>-<n>": idempotency key handed to the adapter
    state: AttemptState
    remote_id: str | None = None
    remote_url: str | None = None
    remote_meta: dict[str, str] = Field(default_factory=dict)
    remote_message: str | None = None
    gpu: str | None = None
    route_reason: str | None = None
    resume_checkpoint_id: str | None = None
    error_kind: str | None = None
    error_message: str | None = None
    lost_reason: str | None = None
    exit_code: int | None = None
    log_lines: int = 0
    log_cursor: str | None = None
    session_deadline: Timestamp | None = None
    created_at: Timestamp
    submitted_at: Timestamp | None = None
    started_at: Timestamp | None = None
    ended_at: Timestamp | None = None
    last_seen_at: Timestamp | None = None


class JobEvent(_Frozen):
    """One row of `job_events`. kind='transition' has to_state; kind='note' does not."""

    seq: int
    job_id: str
    attempt_id: str | None = None
    kind: Literal["transition", "note"]
    from_state: JobState | None = None
    to_state: JobState | None = None
    reason: str  # a Reason value; typed str so older clients tolerate new codes
    message: str
    detail: dict[str, Any] = Field(default_factory=dict)
    actor: str  # "engine" | "recovery" | "user:cli" | "user:shell" | "agent" | "api"
    ts: Timestamp


class Checkpoint(_Frozen):
    id: str  # "<job_id>.c<seq>"
    job_id: str
    attempt_id: str
    seq: int
    step: int | None = None
    uri: str
    size_bytes: int | None = None
    sha256: str | None = None
    created_at: Timestamp
    recorded_at: Timestamp


class ProviderState(_Frozen):
    """Engine-maintained runtime state of one provider (row of `provider_state`)."""

    provider: str
    health: ProviderHealth = ProviderHealth.UNKNOWN
    health_reason: str | None = None
    last_healthcheck_at: Timestamp | None = None
    cooldown_until: Timestamp | None = None
    consecutive_failures: int = 0
    exhausted_until: Timestamp | None = None
    updated_at: Timestamp


class QuotaSnapshot(_Frozen):
    """One observation of a provider's quota (row of `quota_snapshots`; phase 5 writes)."""

    provider: str
    used: float
    limit: float | None
    unit: QuotaUnit
    resets_at: Timestamp | None = None
    source: Literal["live", "estimate"]
    detail: dict[str, Any] = Field(default_factory=dict)
    observed_at: Timestamp


# --------------------------------------------------------------------------- patches


class JobPatch(BaseModel):
    """Column changes applied by Store.transition / Store.update_job.

    Only fields explicitly set (``patch.model_fields_set``) are written, so
    ``JobPatch(provider=None)`` clears provider while ``JobPatch()`` touches nothing.
    `state`, `id`, `version`, timestamps managed by the store are deliberately absent.
    """

    model_config = ConfigDict(extra="forbid")

    provider: str | None = None
    gpu: str | None = None
    current_attempt_id: str | None = None
    route_reason: str | None = None
    approval_reason: str | None = None
    approved_at: float | None = None
    approved_by: str | None = None
    not_before: float | None = None
    waiting_since: float | None = None
    cancel_requested_at: float | None = None
    progress_step: int | None = None
    progress_total: int | None = None
    progress_source: Literal["helper", "stdout"] | None = None
    last_metrics: dict[str, float] | None = None
    outputs_dir: str | None = None
    outputs_fetched: bool | None = None
    exit_code: int | None = None
    failure_kind: FailureKind | None = None
    message: str | None = None
    started_at: float | None = None
    bundle_sha256: str | None = None


class AttemptPatch(BaseModel):
    """Attempt column changes; same set-fields-only semantics as JobPatch.
    If `state` is set the store validates it with statemachine.check_attempt_transition and
    stamps ended_at when it becomes terminal."""

    model_config = ConfigDict(extra="forbid")

    state: AttemptState | None = None
    remote_id: str | None = None
    remote_url: str | None = None
    remote_meta: dict[str, str] | None = None
    remote_message: str | None = None
    gpu: str | None = None
    error_kind: str | None = None
    error_message: str | None = None
    lost_reason: str | None = None
    exit_code: int | None = None
    log_lines: int | None = None
    log_cursor: str | None = None
    session_deadline: float | None = None
    submitted_at: float | None = None
    started_at: float | None = None
    last_seen_at: float | None = None
