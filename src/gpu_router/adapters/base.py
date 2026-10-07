"""The adapter contract (phase 1; real code, frozen interface; owner: group B).

Every provider implements `Adapter`. The engine is the only caller (invariant 2) and calls
every method from a worker thread through `engine.calls.AdapterCaller` (invariant 9), so
methods are plain blocking functions.

Rules A1-A10 (CLAUDE.md "Adapter contract") in short:

A1  implement all eight calls; declare `Capabilities` honestly.
A2  blocking; bound every subprocess / network call below config.engine.timeouts.<call>.
A3  raise only gpu_router.errors.AdapterError subclasses. From submit(), raise a
    DEFINITIVE_SUBMIT_ERRORS class only when certain nothing was created remotely; when in
    doubt raise Unavailable (the engine then resolves by attempt key, invariant 6).
A4  submit() is idempotent per ctx.attempt_key and tags the remote run with it.
A5  cancel() of a finished or unknown run returns normally.
A6  status() / logs() / quota() / healthcheck() / lookup_by_key() have no side effects.
A7  logs(since=c) never yields lines that came before cursor c; cursors are opaque strings
    that stay valid across daemon restarts (they are persisted in attempts.log_cursor).
A8  no DB access; no secret values in exceptions, logs, RemoteRef.meta or RemoteStatus.
A9  fetch() may be re-run; it writes into dest and never deletes files it did not write.
A10 read time only from the injected clock (invariant 13).

`remote_id` in the spec's contract table is `RemoteRef.remote_id` (Decisions log D7).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, SecretStr

from gpu_router.models import Checkpoint, Job, ProviderHealth, QuotaSnapshot, Timestamp

if TYPE_CHECKING:
    from gpu_router.clock import Clock
    from gpu_router.config import ProviderSettings
    from gpu_router.paths import Paths
    from gpu_router.providers.catalog import ProviderEntry

__all__ = [
    "Adapter",
    "AdapterDeps",
    "AttemptContext",
    "Capabilities",
    "FetchResult",
    "Health",
    "LogChunk",
    "RemotePhase",
    "RemoteRef",
    "RemoteStatus",
    "StagedData",
]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


# --------------------------------------------------------------------------- inputs


class AttemptContext(_Frozen):
    """Everything an adapter needs to start one attempt. Built by the engine per placement.

    `secrets` holds resolved values (never logged, never persisted): names come from
    JobSpec.secrets, values from gpu_router.secrets at submit time (invariant 12). Adapters
    pass them to the remote as environment variables through the provider's secret
    mechanism, never on a command line that ends up in logs.
    """

    attempt_id: str  # "<job_id>.<n>"
    attempt_key: str  # "gpu-<job_id>-<n>": idempotency key + remote tag (A4)
    n: int = Field(ge=1)
    bundle_dir: Path | None = None  # jobs/<id>/bundle (phase 2+); None in phase 1
    bundle_archive: Path | None = None  # jobs/<id>/bundle.tar.gz (phase 2+)
    resume_from: Checkpoint | None = None  # latest checkpoint when migrating
    env: Mapping[str, str] = Field(default_factory=dict)  # non-secret env incl. GPU_* vars
    secrets: Mapping[str, SecretStr] = Field(default_factory=dict)
    gpu: str | None = None  # GPU the router picked, e.g. "T4" (catalog name)
    checkpoint_interval_min: int = 20
    session_deadline: Timestamp | None = None  # when the provider will kill the session


# --------------------------------------------------------------------------- outputs


class RemoteRef(_Frozen):
    """Handle to one remote run. Persisted in attempts (remote_id, remote_url,
    remote_meta_json) and handed back verbatim to status/logs/fetch/cancel."""

    remote_id: str = Field(min_length=1)
    url: str | None = None  # human-facing link (console page), if any
    meta: dict[str, str] = Field(default_factory=dict)  # adapter-private, non-secret (A8)


class RemotePhase(StrEnum):
    PENDING = "pending"  # accepted, waiting for a GPU / starting
    RUNNING = "running"  # user code executing
    SUCCEEDED = "succeeded"  # exited 0
    FAILED = "failed"  # user code exited non-zero (exit_code set when known)
    CANCELLED = "cancelled"  # stopped (by us or by someone outside gpu-router)
    LOST = "lost"  # session died / preempted / timed out / quota cut it off

    @property
    def terminal(self) -> bool:
        return self not in (RemotePhase.PENDING, RemotePhase.RUNNING)


class RemoteStatus(_Frozen):
    """One observation of a remote run (A6: observing has no side effects)."""

    phase: RemotePhase
    message: str | None = None  # short provider text, e.g. "waiting for GPU"
    exit_code: int | None = None  # set for SUCCEEDED (0) / FAILED when known
    lost_reason: str | None = None  # set for LOST: "session limit", "preempted", ...
    quota_exhausted: bool = False  # LOST because free quota ran out (engine -> exhausted)
    gpu: str | None = None  # actual GPU, e.g. "2xT4"
    started_at: Timestamp | None = None
    ended_at: Timestamp | None = None
    url: str | None = None


class LogChunk(_Frozen):
    """A batch of log lines plus the cursor to resume after them (A7).

    `lines` have no trailing newlines. `cursor` resumes strictly after the last line of this
    chunk. `eof` is True when the run is terminal and no more lines will ever come.
    """

    lines: list[str] = Field(default_factory=list)
    cursor: str
    eof: bool = False


class FetchResult(_Frozen):
    dest: Path
    files: int = Field(ge=0)
    bytes: int = Field(ge=0)
    partial: bool = False  # some outputs could not be fetched (message says which)
    message: str | None = None


class StagedData(_Frozen):
    """stage_data() result: where a dataset now lives on the provider's side."""

    uri: str  # what the runner gets in GPU_DATA (the adapter's own scheme)
    uploaded: bool  # False: an earlier upload of the same content was reused
    where: str  # for the job's note: "private kaggle dataset me/gpu-router-data-..."


class Health(_Frozen):
    """healthcheck() result: ok, or the reason not."""

    health: ProviderHealth
    reason: str | None = None  # required when not OK: "kaggle CLI not logged in"
    hint: str | None = None  # what the user can do: "run `gpu login kaggle`"
    checked_at: Timestamp
    detail: dict[str, Any] = Field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.health is ProviderHealth.OK


class Capabilities(_Frozen):
    """Static facts the router and engine read before calling the adapter."""

    lookup_by_key: bool = False  # lookup_by_key() works (else ambiguous submits abandon)
    live_logs: bool = False  # logs(follow=True) streams in real time
    interactive: bool = False  # can host interactive sessions
    resume: bool = True  # honours ctx.resume_from (restores GPU_RESUME_DIR)
    fetch: bool = True  # fetch() returns outputs
    cancel_confirms: bool = True  # status() reflects a cancel (else engine trusts cancel())
    live_quota: bool = False  # quota() returns source="live"
    max_session_hours: float | None = None  # provider kills sessions after this
    max_concurrency: int = Field(default=1, ge=1)  # concurrent remote runs allowed
    max_bundle_mb: float | None = None
    poll_interval_s: float = Field(default=30, gt=0)  # recommended status() cadence
    # stage_data() works: datasets reach this provider without checkpoint storage (the
    # provider's own dataset store, e.g. Kaggle datasets; 2026-10-04)
    stage_data: bool = False


# --------------------------------------------------------------------------- the ABC


@dataclass(frozen=True, slots=True)
class AdapterDeps:
    """Constructor bundle for adapters (built by the registry)."""

    name: str  # provider name as registered ("kaggle", "fake-b")
    entry: ProviderEntry  # catalog facts for this provider
    settings: ProviderSettings  # user's non-secret knobs (config.providers.<name>)
    paths: Paths
    clock: Clock
    test_mode: bool = False


class Adapter(ABC):
    """Base class for provider adapters. Subclasses set `kind` and `capabilities`.

    Instances are created once per daemon by the registry and shared across threads, so
    methods must be thread-safe (the engine may call status() for two attempts concurrently,
    bounded by config.engine.per_provider_concurrency).
    """

    #: Catalog `kind` this class implements ("fake", "kaggle", ...).
    kind: str = ""
    capabilities: Capabilities = Capabilities()

    def __init__(self, deps: AdapterDeps) -> None:
        self.name: str = deps.name
        self.entry: ProviderEntry = deps.entry
        self.settings: ProviderSettings = deps.settings
        self.paths: Paths = deps.paths
        self.clock: Clock = deps.clock
        self.test_mode = deps.test_mode

    @property
    def scratch_dir(self) -> Path:
        """Adapter-private, non-secret scratch: <home>/providers/<name>/ (created lazily)."""
        d = self.paths.provider_dir(self.name)
        d.mkdir(mode=0o700, parents=True, exist_ok=True)
        return d

    @abstractmethod
    def submit(self, job: Job, ctx: AttemptContext) -> RemoteRef:
        """Start the job remotely, tagged with ctx.attempt_key. Idempotent per key (A4):
        a second call with the same key returns the same RemoteRef without starting another
        run. Raises RateLimited | QuotaExhausted | AuthRequired | InvalidJob | Permanent only
        when nothing was created; Unavailable when unsure."""

    @abstractmethod
    def status(self, ref: RemoteRef) -> RemoteStatus:
        """Observe the run. Raises NotFound if the provider has no such run (the engine then
        marks the attempt lost), Unavailable / RateLimited / AuthRequired on trouble."""

    @abstractmethod
    def logs(
        self, ref: RemoteRef, *, follow: bool = False, since: str | None = None
    ) -> Iterator[LogChunk]:
        """Yield log chunks after cursor `since` (None = from the start).

        follow=False: yield what exists now (possibly one empty chunk carrying the current
        cursor) and return. follow=True (only if capabilities.live_logs): keep yielding until
        the run is terminal, ending with a chunk whose eof=True. Lines are raw; the engine
        redacts and parses them."""

    @abstractmethod
    def fetch(self, ref: RemoteRef, dest: Path) -> FetchResult:
        """Download the run's outputs (GPU_OUTPUT_DIR contents) into dest (created if
        missing). Re-runnable (A9). Raises NotFound if outputs are gone."""

    @abstractmethod
    def cancel(self, ref: RemoteRef) -> None:
        """Ask the provider to stop the run. Idempotent (A5): finished/unknown runs return
        normally. Confirmation comes from a later status()."""

    @abstractmethod
    def quota(self) -> QuotaSnapshot:
        """Current usage vs limit. source="live" only if the provider reported it."""

    @abstractmethod
    def healthcheck(self) -> Health:
        """Cheap readiness probe: CLI installed, logged in, API reachable. Never starts a
        GPU. Returns a non-OK Health rather than raising for expected problems."""

    def lookup_by_key(self, attempt_key: str) -> RemoteRef | None:
        """Find a run previously submitted with this key (crash recovery, invariant 6).
        Returns None if no run exists. Only called when capabilities.lookup_by_key; the
        default raises Permanent so a mis-declared capability is loud."""
        from gpu_router.errors import Permanent

        raise Permanent(f"{self.name} cannot look runs up by key", provider=self.name)

    def stage_data(self, path: Path, sha256: str, files: Sequence[tuple[str, int]]) -> StagedData:
        """Put a dataset (`path`: a file or a directory; `files` = (relative path, size) as
        `checkpoint.data.digest` listed them, `sha256` its content hash) where this
        provider's runs can read it, once per content: a second call with the same sha256
        reuses the upload. Blocking, bounded below engine.timeouts.stage_data. Only called
        when capabilities.stage_data; the default raises Permanent."""
        from gpu_router.errors import Permanent

        raise Permanent(f"{self.name} cannot stage datasets", provider=self.name)

    def close(self) -> None:  # noqa: B027 - optional hook, deliberately not abstract
        """Release resources on daemon shutdown. Must not touch remote runs (invariant 11)."""

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name}>"
