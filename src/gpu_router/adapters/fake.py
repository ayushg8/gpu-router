"""FakeAdapter: a simulated provider for tests and test mode (phase 1; owner: group B).

Decisions log D9: the fake's "remote" lives on disk under `<home>/fake/<provider name>/`,
OUTSIDE the daemon, and every observable fact is derived from (submit record, injected
clock). So a SIGKILLed daemon that restarts finds its runs exactly where a real provider
would have them, and FakeClock tests are deterministic.

Disk layout (all JSON written atomically: tmp + os.replace; mutations hold an flock on
`.lock` so two adapter instances or threads never interleave):

    fake/<name>/.lock                    flock taken by submit / cancel / set_health
    fake/<name>/counters.json            {"submits": 7, "jobs": {"<job_id>": {"rate_limited": 1,
                                          "unavailable": 0}}}
    fake/<name>/health.json              optional health override (set_health)
    fake/<name>/keys/<attempt_key>       contains the remote_id (lookup_by_key, idempotency)
    fake/<name>/runs/<remote_id>/run.json   RunRecord (below); written first, it is the
                                          commit point of a submit
    fake/<name>/runs/<remote_id>/outputs/   created lazily by fetch() on success

Remote id: "fk-<attempt_key>" (deterministic, so submit is idempotent per key, A4).

Directives (FakeDirectives, read from job.spec.provider_options[<name>], else
provider_options["fake"]; an "attempts": {"<n>": {...}} sub-mapping overrides per attempt n):

    duration      seconds of RUNNING before exit (default 10)
    pending_s     seconds of PENDING before RUNNING (default 0)
    exit_code     exit code at the end (default 0 -> succeeded; non-zero -> failed)
    fail_at       seconds into RUNNING at which the run exits with exit_code or 1 (failed)
    die_after     seconds into RUNNING at which the session dies -> phase LOST
                  (lost_reason "session limit")
    rate_limit_n  the first N submits of this job to this provider raise
                  RateLimited(retry_after=1)
    unavailable_n the first N submits of this job raise Unavailable (nothing is created, so
                  lookup_by_key then returns None)
    quota_limit   GPU-seconds this provider allows in total (sum of RUNNING time over all
                  runs); submit raises QuotaExhausted(resets_at=now+3600) once used >= limit,
                  and a run becomes LOST with quota_exhausted=True when it has used the
                  budget that was left at its submit time
    invalid       submit raises InvalidJob
    auth_required submit raises AuthRequired
    permanent     submit raises Permanent
    steps         total steps reported (default 100): emits `total` then one `metric` line
                  and one human line ("step i/N loss=...") per step, evenly over duration
    checkpoint_every  seconds between checkpoints (default 0 = none): emits ckpt_begin then
                  ckpt_end(uri="fake://<name>/<remote_id>/ckpt-<seq>") min(1s, checkpoint_every/2)
                  later (so sub-second crash-test runs still finish checkpoints).
                  seq continues after ctx.resume_from.seq, so it is monotonic per job
    ignore_cancel cancel() is accepted but the run keeps going (tests cancel_unconfirmed)
    device        GPU names the simulated runner's nvidia-smi reports right after hello
                  (default none: no `device` line), e.g. ["Tesla T4, 15360 MiB"] (D56)

Checkpoint storage (phase 5, D43; only when the engine hands the attempt a local `file://`
GPU_STORAGE, i.e. test mode with `checkpoint.fake_storage: true`): the simulated runner
publishes each checkpoint into that storage (runner/storage.py layout: `jobs/<job>/ckpt-NNNN/`
with a small state.json, manifest, then latest.json) at its ckpt_end time, observed lazily
by status()/logs() like everything else here, and stops publishing once owner.json names a
later attempt; ckpt_end URIs are the storage URIs. GPU_RESUME_URI is checked at submit: a
checkpoint missing from storage means a fresh start ("... missing from storage") instead of
"resuming from checkpoint <seq>".

Counters for rate_limit_n / unavailable_n live in counters.json, keyed by job id, so they
survive restarts and one job's directives never consume another job's budget.

Health override (`set_health`): `health.json` makes healthcheck() report that health. While
it says auth_required, submit() and status() raise AuthRequired; while it says unavailable,
submit() and status() raise Unavailable (a simulated outage).

Timeline (t = clock.now() - submitted_at): [0, pending_s) PENDING; then RUNNING until the
first of fail_at / die_after / duration / quota crossing / cancel. The log for a run is a
pure function of its RunRecord and t, so logs() just regenerates lines and slices by cursor
(cursor = str(number of lines already returned)). Lines are timestamped and never depend
on how the run ends, so a later cancel never rewrites lines already returned (A7). Loss
follows 2.0 * 0.97**step.

Capabilities: lookup_by_key=True, live_logs=False (logs(follow=True) behaves like
follow=False), resume=True (records resume_from and logs "resuming from checkpoint <seq>"
as the first RUNNING line), live_quota=True, max_session_hours / max_concurrency /
poll_interval from the catalog entry.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from gpu_router import protocol
from gpu_router.adapters.base import (
    Adapter,
    AdapterDeps,
    AttemptContext,
    Capabilities,
    FetchResult,
    Health,
    LogChunk,
    RemotePhase,
    RemoteRef,
    RemoteStatus,
)
from gpu_router.errors import (
    AuthRequired,
    InvalidJob,
    NotFound,
    Permanent,
    QuotaExhausted,
    RateLimited,
    Unavailable,
)
from gpu_router.models import Job, ProviderHealth, QuotaSnapshot, QuotaUnit
from gpu_router.runner import storage as rstore

__all__ = ["FakeAdapter", "FakeDirectives", "RunRecord"]

#: Remote ids and attempt keys become path components; anything else is refused.
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
REMOTE_PREFIX = "fk-"
#: Max lines per LogChunk; logs() yields several chunks for long backlogs.
CHUNK_LINES = 1000
QUOTA_RESET_S = 3600.0
_PERIOD_S = {"daily": 86_400.0, "weekly": 7 * 86_400.0, "monthly": 30 * 86_400.0}


class FakeDirectives(BaseModel):
    """Behaviour knobs for one fake run. See the module docstring for each field."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    duration: float = Field(default=10, ge=0)
    pending_s: float = Field(default=0, ge=0)
    exit_code: int = 0
    fail_at: float | None = Field(default=None, ge=0)
    die_after: float | None = Field(default=None, ge=0)
    rate_limit_n: int = Field(default=0, ge=0)
    unavailable_n: int = Field(default=0, ge=0)
    quota_limit: float | None = Field(default=None, ge=0)
    invalid: bool = False
    auth_required: bool = False
    permanent: bool = False
    steps: int = Field(default=100, ge=0)
    checkpoint_every: float = Field(default=0, ge=0)
    ignore_cancel: bool = False
    device: list[str] | None = None

    @classmethod
    def for_attempt(
        cls, provider_options: Mapping[str, Mapping[str, Any]], provider: str, n: int
    ) -> FakeDirectives:
        """provider_options[provider] or provider_options["fake"], with
        ["attempts"][str(n)] merged on top. Raises InvalidJob on unknown keys."""
        base = provider_options.get(provider)
        if base is None:
            base = provider_options.get("fake", {})
        if not isinstance(base, Mapping):
            raise InvalidJob(f"provider_options.{provider} must be a mapping", provider=provider)
        merged: dict[str, Any] = {k: v for k, v in base.items() if k != "attempts"}
        attempts = base.get("attempts") or {}
        if not isinstance(attempts, Mapping):
            raise InvalidJob(
                f"provider_options.{provider}.attempts must map attempt numbers to directives",
                provider=provider,
            )
        override = attempts.get(str(n), attempts.get(n))
        if override is not None:
            if not isinstance(override, Mapping):
                raise InvalidJob(
                    f"provider_options.{provider}.attempts.{n} must be a mapping", provider=provider
                )
            merged.update(override)
        try:
            return cls.model_validate(merged)
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(str(p) for p in err['loc']) or 'directives'}: {err['msg']}"
                for err in exc.errors()
            )
            raise InvalidJob(
                f"bad fake directives for {provider}: {problems}",
                provider=provider,
                hint="see gpu_router/adapters/fake.py for valid directives",
            ) from None


class RunRecord(BaseModel):
    """Persisted state of one fake remote run (run.json)."""

    model_config = ConfigDict(extra="forbid")

    remote_id: str
    attempt_key: str
    job_id: str
    provider: str
    directives: FakeDirectives
    submitted_at: float
    resume_seq: int | None = None
    cancelled_at: float | None = None
    gpu: str = "T4"
    #: RUNNING seconds this run may use before quota runs out (None = unlimited).
    quota_budget_s: float | None = None
    #: set when cancel() was called with ignore_cancel (the run kept going).
    cancel_ignored_at: float | None = None
    #: phase 5 (D43): local storage root the simulated runner publishes to (GPU_STORAGE),
    #: the checkpoint it was told to restore (GPU_RESUME_URI) and whether that was missing.
    storage: str | None = None
    resume_uri: str | None = None
    resume_missing: bool = False
    attempt_n: int | None = None


@dataclass(frozen=True, slots=True)
class _Timeline:
    """Where a run is at one instant. Derived, never stored."""

    phase: RemotePhase
    run_start: float  # absolute time RUNNING begins
    end_at: float  # absolute time the run ends (or would end)
    end_kind: str  # exit | fail | die | quota | cancel
    exit_code: int | None
    lost_reason: str | None
    quota_exhausted: bool

    def running_seconds(self, at: float) -> float:
        return max(0.0, min(at, self.end_at) - self.run_start)


def _atomic_write(path: Path, data: str | bytes) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    raw = data.encode("utf-8") if isinstance(data, str) else data
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(raw)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


class FakeAdapter(Adapter):
    kind = "fake"
    capabilities = Capabilities(
        lookup_by_key=True,
        live_logs=False,
        interactive=False,
        resume=True,
        live_quota=True,
        max_concurrency=4,
        poll_interval_s=1,
    )

    def __init__(self, deps: AdapterDeps) -> None:
        super().__init__(deps)
        # Instance capabilities follow the catalog entry (session cap, concurrency, poll).
        self.capabilities = Capabilities(
            lookup_by_key=True,
            live_quota=True,
            max_session_hours=deps.entry.session_hours,
            max_concurrency=deps.entry.max_concurrency,
            poll_interval_s=deps.entry.poll_interval_s,
        )

    @property
    def root(self) -> Path:
        """<home>/fake/<name>/"""
        return self.paths.fake_dir / self.name

    # ------------------------------------------------------------------ contract

    def submit(self, job: Job, ctx: AttemptContext) -> RemoteRef:
        key = ctx.attempt_key
        if not _SAFE_NAME.match(key):
            raise InvalidJob(f"attempt key {key!r} is not a valid fake run tag", provider=self.name)
        with self._locked():
            existing = self._find_by_key(key)
            if existing is not None:
                return self._ref(existing)
            directives = FakeDirectives.for_attempt(job.spec.provider_options, self.name, ctx.n)
            self._raise_for_health(during="submit")
            if directives.auth_required:
                raise AuthRequired(
                    f"{self.name} is not logged in (fake directive)",
                    provider=self.name,
                    hint=f"run `gpu login {self.name}`",
                )
            if directives.invalid:
                raise InvalidJob(
                    f"{self.name} cannot run this job (fake directive)", provider=self.name
                )
            if directives.permanent:
                raise Permanent(
                    f"{self.name} refused the job permanently (fake directive)", provider=self.name
                )
            counters = self._read_counters()
            job_counts: dict[str, int] = counters.setdefault("jobs", {}).setdefault(job.id, {})
            if job_counts.get("rate_limited", 0) < directives.rate_limit_n:
                job_counts["rate_limited"] = job_counts.get("rate_limited", 0) + 1
                self._write_counters(counters)
                raise RateLimited(
                    f"{self.name} is rate limiting submits (fake directive)",
                    retry_after=1,
                    provider=self.name,
                )
            if job_counts.get("unavailable", 0) < directives.unavailable_n:
                job_counts["unavailable"] = job_counts.get("unavailable", 0) + 1
                self._write_counters(counters)
                raise Unavailable(
                    f"{self.name} is unavailable (fake directive)", provider=self.name
                )
            now = self.clock.now()
            budget: float | None = None
            if directives.quota_limit is not None:
                used = self._used_seconds(now)
                if used >= directives.quota_limit:
                    raise QuotaExhausted(
                        f"{self.name} free quota is used up ({used / 3600:.2f} gpu-h)",
                        resets_at=now + QUOTA_RESET_S,
                        provider=self.name,
                    )
                budget = directives.quota_limit - used
            storage = ctx.env.get("GPU_STORAGE")
            if storage is not None and not storage.startswith(rstore.FILE_PREFIX):
                storage = None  # the fake can only reach storage on this Mac
            resume_uri = ctx.env.get("GPU_RESUME_URI") if storage is not None else None
            record = RunRecord(
                remote_id=REMOTE_PREFIX + key,
                attempt_key=key,
                job_id=job.id,
                provider=self.name,
                directives=directives,
                submitted_at=now,
                resume_seq=ctx.resume_from.seq if ctx.resume_from is not None else None,
                gpu=ctx.gpu or (self.entry.gpus[0].label if self.entry.gpus else "T4"),
                quota_budget_s=budget,
                storage=storage,
                resume_uri=resume_uri,
                resume_missing=(
                    resume_uri is not None and _stored_manifest(storage, resume_uri) is None
                ),
                attempt_n=ctx.n,
            )
            self._write_record(record)  # commit point: the run now exists remotely
            _atomic_write(self.root / "keys" / key, record.remote_id)
            counters["submits"] = int(counters.get("submits", 0)) + 1
            self._write_counters(counters)
            return self._ref(record)

    def status(self, ref: RemoteRef) -> RemoteStatus:
        record = self.run_record(ref.remote_id)
        self._raise_for_health(during="status")
        now = self.clock.now()
        tl = self._timeline(record)
        self._publish_due(record, tl)
        # A run cancelled while still pending never started.
        started = tl.run_start if now >= tl.run_start and tl.end_at >= tl.run_start else None
        message = {
            RemotePhase.PENDING: "waiting for GPU",
            RemotePhase.RUNNING: f"running on {record.gpu}",
            RemotePhase.SUCCEEDED: "finished",
            RemotePhase.FAILED: f"script exited with code {tl.exit_code}",
            RemotePhase.CANCELLED: "cancelled",
            RemotePhase.LOST: f"session ended: {tl.lost_reason}",
        }[tl.phase]
        return RemoteStatus(
            phase=tl.phase,
            message=message,
            exit_code=tl.exit_code if tl.phase.terminal else None,
            lost_reason=tl.lost_reason if tl.phase is RemotePhase.LOST else None,
            quota_exhausted=tl.quota_exhausted and tl.phase is RemotePhase.LOST,
            gpu=record.gpu,
            started_at=started,
            ended_at=tl.end_at if tl.phase.terminal else None,
            url=self._url(record.remote_id),
        )

    def logs(
        self, ref: RemoteRef, *, follow: bool = False, since: str | None = None
    ) -> Iterator[LogChunk]:
        # live_logs=False: follow=True is served like follow=False (one snapshot).
        record = self.run_record(ref.remote_id)
        now = self.clock.now()
        tl = self._timeline(record)
        self._publish_due(record, tl)
        lines = self._log_lines(record, tl)
        visible = [line for ts, line in lines if ts <= now]
        complete = tl.phase.terminal and len(visible) == len(lines)
        start = min(self._parse_cursor(since), len(visible))
        if start >= len(visible):
            yield LogChunk(lines=[], cursor=str(start), eof=complete)
            return
        for lo in range(start, len(visible), CHUNK_LINES):
            hi = min(lo + CHUNK_LINES, len(visible))
            yield LogChunk(
                lines=visible[lo:hi], cursor=str(hi), eof=complete and hi == len(visible)
            )

    def fetch(self, ref: RemoteRef, dest: Path) -> FetchResult:
        """Writes dest/result.json ({"job_id", "remote_id", "steps", "final_loss"}) and
        dest/model.txt. NotFound unless the run succeeded."""
        record = self.run_record(ref.remote_id)
        tl = self._timeline(record)
        if tl.phase is not RemotePhase.SUCCEEDED:
            raise NotFound(
                f"{record.remote_id} has no outputs (run is {tl.phase})", provider=self.name
            )
        outputs = self._run_dir(record.remote_id) / "outputs"
        steps = record.directives.steps
        produced = {
            "result.json": json.dumps(
                {
                    "job_id": record.job_id,
                    "remote_id": record.remote_id,
                    "steps": steps,
                    "final_loss": round(_loss(steps), 6),
                },
                indent=2,
            )
            + "\n",
            "model.txt": f"fake model weights for job {record.job_id} after {steps} steps\n",
        }
        for name, body in produced.items():
            if not (outputs / name).is_file():
                _atomic_write(outputs / name, body)
        dest.mkdir(parents=True, exist_ok=True)
        files = total = 0
        for src in sorted(outputs.iterdir()):
            if not src.is_file():
                continue
            data = src.read_bytes()
            _atomic_write(dest / src.name, data)
            files += 1
            total += len(data)
        return FetchResult(dest=dest, files=files, bytes=total)

    def cancel(self, ref: RemoteRef) -> None:
        if not _SAFE_NAME.match(ref.remote_id):
            return
        with self._locked():
            try:
                record = self.run_record(ref.remote_id)
            except NotFound:
                return
            if record.cancelled_at is not None or record.cancel_ignored_at is not None:
                return
            if self._timeline(record).phase.terminal:
                return
            now = self.clock.now()
            if record.directives.ignore_cancel:
                record = record.model_copy(update={"cancel_ignored_at": now})
            else:
                record = record.model_copy(update={"cancelled_at": now})
            self._write_record(record)

    def quota(self) -> QuotaSnapshot:
        """used = GPU-hours of RUNNING time across all runs so far; limit = catalog quota
        limit, or quota_limit/3600 from the most recent run's directives if set."""
        now = self.clock.now()
        runs = self.all_runs()
        used_s = sum(self._timeline(r).running_seconds(now) for r in runs)
        spec = self.entry.quota
        limit = spec.limit
        unit = spec.unit
        resets_at: float | None = None
        period = _PERIOD_S.get(spec.reset)
        if period is not None:
            resets_at = now + period
            if spec.reset_anchor:  # a fixed calendar window, like kaggle's Saturday reset
                from gpu_router.quota.windows import current_window

                resets_at = current_window(spec.reset, spec.reset_anchor, now).resets_at or (
                    resets_at
                )
        latest = runs[-1] if runs else None
        if latest is not None and latest.directives.quota_limit is not None:
            limit = latest.directives.quota_limit / 3600
            unit = QuotaUnit.GPU_HOURS
            resets_at = now + QUOTA_RESET_S
        return QuotaSnapshot(
            provider=self.name,
            used=used_s / 3600,
            limit=limit,
            unit=unit,
            resets_at=resets_at,
            source="live",
            detail={"runs": len(runs)},
            observed_at=now,
        )

    def healthcheck(self) -> Health:
        """OK unless `<root>/health.json` exists ({"health": ..., "reason": ...}), which tests
        write through `set_health()` to simulate outages / logged-out CLIs."""
        now = self.clock.now()
        override = self._read_health()
        if override is None:
            return Health(
                health=ProviderHealth.OK,
                checked_at=now,
                detail={"runs_dir": str(self.root / "runs")},
            )
        health, reason = override
        if health is ProviderHealth.OK:
            return Health(health=health, checked_at=now)
        hint = f"run `gpu login {self.name}`" if health is ProviderHealth.AUTH_REQUIRED else None
        return Health(
            health=health,
            reason=reason or f"{self.name} is {health} (simulated)",
            hint=hint,
            checked_at=now,
        )

    def lookup_by_key(self, attempt_key: str) -> RemoteRef | None:
        if not _SAFE_NAME.match(attempt_key):
            return None
        record = self._find_by_key(attempt_key)
        return None if record is None else self._ref(record)

    # ------------------------------------------------------------------ test helpers

    def set_health(self, health: str | None, reason: str | None = None) -> None:
        """Write (or with None, remove) the health override file."""
        path = self.root / "health.json"
        with self._locked():
            if health is None:
                with contextlib.suppress(FileNotFoundError):
                    path.unlink()
                return
            value = ProviderHealth(health)  # ValueError on an unknown health name
            _atomic_write(path, json.dumps({"health": str(value), "reason": reason}))

    def run_record(self, remote_id: str) -> RunRecord:
        """Read run.json (NotFound if absent)."""
        if not _SAFE_NAME.match(remote_id):
            raise NotFound(f"{self.name} has no run {remote_id!r}", provider=self.name)
        path = self._run_dir(remote_id) / "run.json"
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            raise NotFound(f"{self.name} has no run {remote_id!r}", provider=self.name) from None
        try:
            return RunRecord.model_validate_json(text)
        except ValidationError as exc:
            raise Unavailable(
                f"{self.name} run {remote_id} record is unreadable: {exc.error_count()} errors",
                provider=self.name,
            ) from None

    def all_runs(self) -> list[RunRecord]:
        """Every run ever submitted to this fake, oldest first (tests assert no double submit)."""
        runs_dir = self.root / "runs"
        if not runs_dir.is_dir():
            return []
        records: list[RunRecord] = []
        for d in runs_dir.iterdir():
            if d.is_dir() and (d / "run.json").is_file():
                records.append(self.run_record(d.name))
        records.sort(key=lambda r: (r.submitted_at, r.remote_id))
        return records

    # ------------------------------------------------------------------ internals

    @contextlib.contextmanager
    def _locked(self) -> Iterator[None]:
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        with open(self.root / ".lock", "a+b") as fh:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)

    def _run_dir(self, remote_id: str) -> Path:
        return self.root / "runs" / remote_id

    def _url(self, remote_id: str) -> str:
        return f"fake://{self.name}/{remote_id}"

    def _ref(self, record: RunRecord) -> RemoteRef:
        return RemoteRef(
            remote_id=record.remote_id,
            url=self._url(record.remote_id),
            meta={"attempt_key": record.attempt_key},
        )

    def _write_record(self, record: RunRecord) -> None:
        _atomic_write(
            self._run_dir(record.remote_id) / "run.json", record.model_dump_json(indent=2)
        )

    def _find_by_key(self, attempt_key: str) -> RunRecord | None:
        key_file = self.root / "keys" / attempt_key
        try:
            remote_id = key_file.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            remote_id = REMOTE_PREFIX + attempt_key  # run.json may precede the key file
        try:
            return self.run_record(remote_id)
        except NotFound:
            return None

    def _read_counters(self) -> dict[str, Any]:
        try:
            data = json.loads((self.root / "counters.json").read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError):
            return {"submits": 0, "jobs": {}}
        return data if isinstance(data, dict) else {"submits": 0, "jobs": {}}

    def _write_counters(self, counters: dict[str, Any]) -> None:
        _atomic_write(self.root / "counters.json", json.dumps(counters, sort_keys=True))

    def _read_health(self) -> tuple[ProviderHealth, str | None] | None:
        try:
            data = json.loads((self.root / "health.json").read_text(encoding="utf-8"))
            return ProviderHealth(data["health"]), data.get("reason")
        except FileNotFoundError:
            return None
        except (ValueError, KeyError, TypeError):
            return ProviderHealth.DEGRADED, "health override file is unreadable"

    def _raise_for_health(self, *, during: str) -> None:
        override = self._read_health()
        if override is None:
            return
        health, reason = override
        if health is ProviderHealth.AUTH_REQUIRED:
            raise AuthRequired(
                reason or f"{self.name} is not logged in (simulated)",
                provider=self.name,
                hint=f"run `gpu login {self.name}`",
            )
        if health is ProviderHealth.UNAVAILABLE:
            raise Unavailable(
                reason or f"{self.name} is down (simulated, during {during})", provider=self.name
            )

    def _used_seconds(self, at: float) -> float:
        return sum(self._timeline(r).running_seconds(at) for r in self.all_runs())

    def _timeline(self, record: RunRecord) -> _Timeline:
        d = record.directives
        run_start = record.submitted_at + d.pending_s
        # (running seconds, priority, kind): on ties a failure beats a clean exit.
        ends: list[tuple[float, int, str]] = [(d.duration, 3, "exit")]
        if d.fail_at is not None:
            ends.append((d.fail_at, 0, "fail"))
        if d.die_after is not None:
            ends.append((d.die_after, 1, "die"))
        if record.quota_budget_s is not None:
            ends.append((max(0.0, record.quota_budget_s), 2, "quota"))
        end_r, _, kind = min(ends)
        end_at = run_start + end_r
        if record.cancelled_at is not None and record.cancelled_at < end_at:
            end_at, kind = record.cancelled_at, "cancel"
        now = self.clock.now()
        exit_code: int | None = None
        lost_reason: str | None = None
        quota_exhausted = False
        if kind == "exit":
            exit_code = d.exit_code
            phase = RemotePhase.SUCCEEDED if d.exit_code == 0 else RemotePhase.FAILED
        elif kind == "fail":
            exit_code = d.exit_code or 1
            phase = RemotePhase.FAILED
        elif kind == "die":
            lost_reason, phase = "session limit", RemotePhase.LOST
        elif kind == "quota":
            lost_reason, phase, quota_exhausted = "quota exhausted", RemotePhase.LOST, True
        else:
            phase = RemotePhase.CANCELLED
        if now < end_at:
            phase = RemotePhase.PENDING if now < run_start else RemotePhase.RUNNING
        return _Timeline(
            phase=phase,
            run_start=run_start,
            end_at=end_at,
            end_kind=kind,
            exit_code=exit_code,
            lost_reason=lost_reason,
            quota_exhausted=quota_exhausted,
        )

    def _log_lines(self, record: RunRecord, tl: _Timeline) -> list[tuple[float, str]]:
        """Every line the run will ever print up to its end, as (timestamp, line), in order.

        Lines strictly before the end never depend on how the run ends (A7 across cancel).
        """
        d = record.directives
        body: list[tuple[float, int, str]] = []

        def add(ts: float, line: str) -> None:
            if ts <= tl.end_at:
                body.append((ts, len(body), line))

        t0 = record.submitted_at
        add(t0, f"fake: accepted {record.remote_id} on {record.gpu}")
        if d.pending_s > 0:
            add(t0, "fake: waiting for a GPU")
        rs = tl.run_start
        if record.resume_seq is not None and record.resume_missing:
            add(
                rs,
                f"fake: checkpoint {record.resume_seq} is missing from storage "
                f"({record.resume_uri}); starting fresh",
            )
        elif record.resume_seq is not None:
            add(rs, f"resuming from checkpoint {record.resume_seq}")
            if record.resume_uri is not None:
                add(rs, f"fake: restored {record.resume_uri}")
        add(rs, protocol.hello("fake/1"))
        if d.device:
            add(rs, protocol.device(d.device))
        if d.steps > 0:
            add(rs, protocol.total(d.steps))
            for i in range(1, d.steps + 1):
                ts = rs + d.duration * i / d.steps
                loss = round(_loss(i), 6)
                add(ts, protocol.metric(i, {"loss": loss}, total_steps=d.steps))
                add(ts, f"step {i}/{d.steps} loss={loss:.4f}")
        for begin, end_ts, seq, step in self._checkpoints(record, tl):
            add(begin, protocol.ckpt_begin(seq))
            add(end_ts, protocol.ckpt_end(seq, self._ckpt_uri(record, seq), step=step))
        body.sort(key=lambda item: (item[0], item[1]))
        lines = [(ts, line) for ts, _, line in body]
        end = tl.end_at
        if tl.end_kind == "exit" and tl.exit_code == 0:
            lines += [(end, "fake: training finished"), (end, protocol.exit_line(0))]
        elif tl.end_kind in ("exit", "fail"):
            code = tl.exit_code if tl.exit_code is not None else 1
            lines += [
                (end, "Traceback (most recent call last):"),
                (end, "RuntimeError: simulated failure (fake directive)"),
                (end, protocol.exit_line(code)),
            ]
        elif tl.end_kind in ("die", "quota"):
            lines.append((end, f"fake: session ended ({tl.lost_reason})"))
        else:
            lines.append((end, "fake: cancelled"))
        return lines

    @staticmethod
    def _checkpoints(
        record: RunRecord, tl: _Timeline
    ) -> list[tuple[float, float, int, int | None]]:
        """(ckpt_begin ts, ckpt_end ts, seq, step) of every checkpoint the run starts
        before it ends; seq continues after the resumed one (monotonic per job)."""
        d = record.directives
        out: list[tuple[float, float, int, int | None]] = []
        if d.checkpoint_every <= 0:
            return out
        base = record.resume_seq or 0
        k = 1
        while (ts := tl.run_start + k * d.checkpoint_every) <= tl.end_at:
            end_ts = ts + min(1.0, d.checkpoint_every / 2)
            out.append((ts, end_ts, base + k, _step_at(d, k * d.checkpoint_every)))
            k += 1
        return out

    def _ckpt_uri(self, record: RunRecord, seq: int) -> str:
        if record.storage is not None:
            return str(rstore.open_store(record.storage).uri(rstore.ckpt_key(record.job_id, seq)))
        return f"fake://{record.provider}/{record.remote_id}/ckpt-{seq}"

    def _publish_due(self, record: RunRecord, tl: _Timeline) -> None:
        """The simulated runner's storage writes (D43): publish every checkpoint whose
        ckpt_end time has passed, once, unless a later attempt owns the job. Errors are
        swallowed like a runner that failed to publish (the log line still names it)."""
        if record.storage is None or record.directives.checkpoint_every <= 0:
            return
        now = self.clock.now()
        due = [c for c in self._checkpoints(record, tl) if c[1] <= min(now, tl.end_at)]
        if not due:
            return
        marker = self._run_dir(record.remote_id) / "published.json"
        with self._locked():
            try:
                done = int(json.loads(marker.read_text(encoding="utf-8")).get("seq", 0))
            except (FileNotFoundError, ValueError, AttributeError, TypeError):
                done = 0
            todo = [c for c in due if c[2] > done]
            if not todo:
                return
            try:
                store = rstore.open_store(record.storage)
                for _begin, end_ts, seq, step in todo:
                    owner = rstore.read_owner(store, record.job_id)
                    later = record.attempt_n is not None and (owner or 0) > record.attempt_n
                    if later:
                        return  # a later attempt took over: this runner stops publishing
                    src = self._run_dir(record.remote_id) / "ckpt-src"
                    body = json.dumps(
                        {
                            "job": record.job_id,
                            "seq": seq,
                            "step": step,
                            "loss": round(_loss(step or 0), 6),
                            "attempt": record.attempt_n,
                        },
                        sort_keys=True,
                    ).encode()
                    _atomic_write(src / "state.json", body)
                    files = [("state.json", len(body), hashlib.sha256(body).hexdigest())]
                    rstore.publish_checkpoint(
                        store,
                        record.job_id,
                        seq,
                        src,
                        files,
                        attempt=record.attempt_n,
                        step=step,
                        created_at=end_ts,
                    )
                    _atomic_write(marker, json.dumps({"seq": seq}))
            except (OSError, rstore.StorageError):
                return

    @staticmethod
    def _parse_cursor(since: str | None) -> int:
        if since is None:
            return 0
        try:
            return max(0, int(since))
        except ValueError:
            return 0


def _loss(step: int) -> float:
    return float(2.0 * 0.97**step)


def _stored_manifest(storage: str | None, uri: str) -> dict[str, Any] | None:
    """The checkpoint manifest at `uri` in the local storage `storage`, or None."""
    if storage is None:
        return None
    try:
        store = rstore.open_store(storage)
        key = store.key_of(uri)
        if key is None:
            return None
        found = rstore.read_json(store, f"{key}/{rstore.CKPT_MANIFEST}")
    except (OSError, rstore.StorageError):
        return None
    return found if isinstance(found, dict) else None


def _step_at(d: FakeDirectives, running_s: float) -> int | None:
    if d.steps <= 0:
        return None
    if d.duration <= 0:
        return d.steps
    return min(d.steps, int(d.steps * running_s / d.duration))
