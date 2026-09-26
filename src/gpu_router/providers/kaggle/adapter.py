"""KaggleAdapter: batch GPU jobs as private Kaggle script kernels (phase 3).

One attempt = one private kernel `<user>/gpu-router-<job_id>-<n>` (remote.py), pushed with
`kaggle kernels push`; the job bundle rides inside the kernel's run.py. Everything goes
through the kaggle CLI (D3, cli.py); facts and verified output shapes are in NOTES.md.

Contract notes (CLAUDE.md "Adapter contract"):

- A4 / lookup_by_key: the attempt key maps to exactly one kernel slug. submit() checks a
  local marker, then `kernels status`, before it pushes, so a retry never adds a version
  (which would start a second run). A `submits/<key>.pushing` intent file is written right
  before `kernels push` and removed when the call returns: if the daemon dies mid-push, the
  CLI child keeps uploading, so until T_PUSH + PUSH_ORPHAN_GRACE_S have passed lookup and
  submit answer Unavailable ("may still be uploading") instead of "nothing exists" (D35).
- status: QUEUED/NEW_SCRIPT -> pending, RUNNING -> running, CANCEL_* -> cancelled. For
  COMPLETE/ERROR the kernel log decides: the runner's `::gpu:: exit` line gives the exit
  code (0 succeeded, 90 = dependency install failed -> lost so the job reroutes, else
  failed). No exit line: time-limit / quota failures are lost (quota_exhausted when the
  live quota is gone), anything else lost with Kaggle's failure message. An empty log
  (COMPLETE or ERROR: run.py always prints first, so empty = not published yet) is retried
  as Unavailable for EMPTY_LOG_GRACE_S before it is judged without one. The final log
  (redacted with secrets.redact, line for line so cursors hold) and outcome are cached
  under <home>/providers/kaggle/final/ (immutable once terminal).
- logs: Kaggle's session log is read once the run is finished. While it runs, the runner's
  log-tail.json in checkpoint storage (phase 5, gpu_router/checkpoint/sidechannel.py) is
  served when the attempt has one (`status_uri` in RemoteRef.meta), with cursors
  `t<lines>:<hash>`; the switch to the final log aligns on the runner's hello line.
  Without a side channel, logs() returns an empty chunk that keeps the cursor. Final-log
  cursor = number of lines returned.
- fetch: `kernels output --file-pattern ^outputs/` into a private temp dir, then copied
  into dest (only GPU_OUTPUT_DIR contents; never deletes anything in dest, A9).
- cancel: the public API cancels only by a session id that no CLI response exposes, so
  cancel deletes the kernel (`kernels delete -y`) while it is queued/running (never after
  it finished: outputs survive) and leaves a local tombstone; status() reports cancelled
  once Kaggle no longer has the kernel (`cancel_confirms=True`). Verified live: deleting a
  running 2xT4 kernel stopped its GPU quota accrual at once (NOTES.md "Live runs"). Every
  push also carries a session timeout (-t) so a run can never outlive its budget.
- quota: live from `kaggle quota --format json` (GPU row).
- secrets (phase 5): CLI-pushed kernels have no secrets field and run.py is kept in version
  history, so values travel in a private dataset `<user>/gpu-router-secrets` (settings
  `providers.kaggle.secrets_dataset`, false = refuse jobs with secrets) attached through
  `dataset_sources`. It is versioned (old versions deleted) only when the values change: a
  keyed digest of the last upload is kept in <home>/providers/kaggle/secrets/.
"""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import json
import os
import re
import shutil
import tempfile
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

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
from gpu_router.checkpoint import sidechannel
from gpu_router.errors import (
    AdapterError,
    AuthRequired,
    InvalidJob,
    NotFound,
    Permanent,
    QuotaExhausted,
    RateLimited,
    Unavailable,
)
from gpu_router.models import Job, ProviderHealth, QuotaSnapshot, QuotaUnit
from gpu_router.providers.kaggle import credentials as creds_mod
from gpu_router.providers.kaggle import parse, remote
from gpu_router.providers.kaggle.cli import (
    LOGIN_HINT,
    PHONE_HINT,
    CliResult,
    KaggleCli,
    Runner,
    SubprocessRunner,
    classify,
    find_executable,
    snippet,
)
from gpu_router.secrets import redact

__all__ = ["GPU_SHAPES", "KaggleAdapter"]

#: catalog GPU name -> Kaggle machine_shape (verified names in NOTES.md).
GPU_SHAPES: dict[str, str] = {"T4": "NvidiaTeslaT4", "P100": "NvidiaTeslaP100"}
SHAPE_LABELS: dict[str, str] = {"NvidiaTeslaT4": "2xT4", "NvidiaTeslaP100": "P100"}
DEFAULT_GPU = "T4"
INSTALL_FAILED_EXIT = 90  # runner/bootstrap.py: dependency install failed, job never ran
DEFAULT_MAX_EMBED_MB = 10.0  # bundle size that rides inside run.py ([I] limit, NOTES.md)
SESSION_MARGIN_S = 300  # push -t = session cap minus this
MIN_TIMEOUT_S = 60
CHUNK_LINES = 1000
QUOTA_EPSILON_H = 0.0  # the CLI rounds to 2 decimals: "0.00h" left = exhausted
CRED_TTL_S = 300.0  # re-read the Keychain at most this often

# Per-call CLI timeouts (seconds). Each adapter method's worst case (sum of the calls it
# can make) stays below config.engine.timeouts.<call> (rule A2):
#   status   status 20 + final log 25 + final quota 12 = 57   (engine 60)
#   logs     = status when nothing is cached yet        = 57   (engine 60)
#   lookup   config 15 + status 20 + confirm quota 20   = 55   (status/submit budgets)
#   submit   config 15 + status 25 + quota 25 + push 180 + recheck 25 = 270 (engine 300)
#   fetch    status 57 + output 1500                    = 1557 (engine 1800)
#   cancel   status 20 (running, no finalize) + delete 15 = 35 (engine 60)
#   health   version 15 + config 15 + quota 15          = 45   (engine 60)
T_STATUS = 20.0
T_FINAL_LOG = 25.0
T_FINAL_QUOTA = 12.0
T_CONFIG = 15.0
T_QUOTA = 20.0
T_PRECHECK = 25.0
T_PUSH = 180.0
T_OUTPUT = 1500.0
T_DELETE = 15.0
T_HEALTH = 15.0  # engine healthcheck budget 60 = version + config + quota
PUSH_ORPHAN_GRACE_S = 60.0  # an interrupted push may upload for T_PUSH + this (D35)
# Secrets dataset (phase 5). When a submit had to (re)upload it, the push gets
# T_PUSH_AFTER_SECRETS instead of T_PUSH so the submit stays under the engine's 300 s:
#   config 15 + status 25 + secrets (status 15 + upload 45 + 6 x ready 5) 90 + quota 25
#   + push 90 + recheck 25 = 270
T_SECRETS_STATUS = 15.0
T_SECRETS_UPLOAD = 45.0
T_SECRETS_READY = 5.0
SECRETS_READY_POLLS = 6
SECRETS_READY_PAUSE_S = 5.0
T_PUSH_AFTER_SECRETS = 90.0
EMPTY_LOG_GRACE_S = 600.0  # a finished kernel's empty log is "not published yet" this long

_JOB_OPTIONS = {"timeout_s", "accelerator", "enable_internet"}
_TIME_LIMIT_MARKERS = (
    "time limit",
    "timeout",
    "timed out",
    "exceeded",
    "maximum run time",
    "session limit",
    "max runtime",
)
_LOCAL_PUSH_ERRORS = (
    "title must be",
    "a source file must be specified",
    "source file not found",
    "metadata file not found",
    "invalid folder",
    "a valid language must be",
    "a valid kernel type must be",
    "id or slug must be specified",
    "cannot contain a version",
    "docker_image_pinning_type",
    "invalid dataset",
    "invalid kernel",
    "invalid model",
)
_SAFE_REF = re.compile(
    r"^([A-Za-z0-9][A-Za-z0-9_.-]{0,63})/(gpu-router-[0-9a-f]{6,32}-[0-9]{1,4})$"
)


def _atomic_write(path: Path, data: str) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(data)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


OFFLINE_REASON = "real kaggle calls are off in test mode"
OFFLINE_HINT = "set GPU_ROUTER_REAL_PROVIDERS=kaggle to allow them"


def _real_opt_in(name: str) -> bool:
    raw = os.environ.get("GPU_ROUTER_REAL_PROVIDERS", "")
    return name in {p.strip() for p in raw.split(",") if p.strip()}


def _classify_push_error(provider: str, message: str) -> AdapterError:
    """A server-side `Kernel push error: <msg>`: the save was refused. Exact strings are
    not documented; NOTES.md records the ones seen live."""
    low = message.lower()
    short = snippet(message)
    if "quota" in low or "weekly" in low:
        return QuotaExhausted(f"kaggle GPU quota is used up: {short}", provider=provider)
    if "phone" in low or "verif" in low:
        return AuthRequired(f"kaggle refused the push: {short}", provider=provider, hint=PHONE_HINT)
    if any(m in low for m in ("concurrent", "at a time", "simultaneous", "maximum number")):
        return RateLimited(
            f"kaggle is already running its maximum number of sessions: {short}",
            provider=provider,
            retry_after=600,
        )
    if any(m in low for m in ("prohibit", "violat", "terms of service", "banned", "suspend")):
        return Permanent(f"kaggle refused this job: {short}", provider=provider)
    return InvalidJob(f"kaggle refused the push: {short}", provider=provider)


class KaggleAdapter(Adapter):
    kind = "kaggle"
    capabilities = Capabilities(
        lookup_by_key=True,
        live_logs=False,
        resume=True,
        cancel_confirms=True,
        live_quota=True,
        max_session_hours=12,
        max_bundle_mb=DEFAULT_MAX_EMBED_MB,
        poll_interval_s=60,
    )

    def __init__(self, deps: AdapterDeps, *, runner: Runner | None = None) -> None:
        super().__init__(deps)
        extra: dict[str, Any] = dict(deps.settings.model_extra or {})
        self._configured_user: str | None = extra.get("username") or None
        self._cli_path: str | None = extra.get("cli_path") or None
        self._cred_mode = str(extra.get("credentials") or "auto")
        self._max_embed_mb = float(extra.get("max_embed_mb") or DEFAULT_MAX_EMBED_MB)
        self._internet = bool(extra.get("enable_internet", True))
        raw_ds = extra.get("secrets_dataset", remote.SECRETS_DATASET_SLUG)
        self._secrets_slug: str | None = (
            None if raw_ds in (False, None, "", "false", "off") else str(raw_ds).split("/")[-1]
        )
        self._sleep: Callable[[float], None] = time.sleep  # tests replace it
        self.capabilities = Capabilities(
            lookup_by_key=True,
            live_logs=False,
            resume=True,
            cancel_confirms=True,
            live_quota=True,
            max_session_hours=deps.entry.session_hours,
            max_concurrency=deps.entry.max_concurrency,
            max_bundle_mb=self._max_embed_mb,
            poll_interval_s=deps.entry.poll_interval_s,
        )
        self._lock = threading.Lock()
        self._username: str | None = None
        self._creds: tuple[float, creds_mod.Credentials] | None = None
        self._cred_source = "unknown"
        # Invariant 20: a test-mode daemon (GPU_ROUTER_TEST_MODE) never calls the real
        # kaggle CLI unless GPU_ROUTER_REAL_PROVIDERS lists kaggle; tests inject a runner.
        self._offline = runner is None and deps.test_mode and not _real_opt_in(self.name)
        self.cli = KaggleCli(
            self.name,
            runner=runner or SubprocessRunner(),
            executable=lambda: find_executable(self._cli_path),
            env_factory=self._cred_env,
        )

    # ------------------------------------------------------------------ credentials

    def _credentials(self) -> creds_mod.Credentials:
        """Resolved at most every CRED_TTL_S (each resolve reads the Keychain)."""
        now = self.clock.now()
        with self._lock:
            cached = self._creds
            if cached is not None and now - cached[0] < CRED_TTL_S:
                return cached[1]
        creds = creds_mod.resolve(
            self._cred_mode, provider=self.name, config_dir=self.scratch_dir / "cli-config"
        )
        if creds.source != "cli":
            (self.scratch_dir / "cli-config").mkdir(mode=0o700, exist_ok=True)
        with self._lock:
            self._creds = (now, creds)
            self._cred_source = creds.source
        return creds

    def _cred_env(self) -> tuple[tuple[str, ...], Mapping[str, str]]:
        creds = self._credentials()
        if creds.source == "cli":
            return (), {}
        # With Keychain creds, inherited KAGGLE_* vars must not override them.
        return creds_mod.CLI_ENV_VARS, creds.plain_env()

    # ------------------------------------------------------------------ helpers

    def _run(self, args: list[str], *, timeout: float, cwd: Path | None = None) -> CliResult:
        if self._offline:
            raise AuthRequired(OFFLINE_REASON, provider=self.name, hint=OFFLINE_HINT)
        return self.cli.run(args, timeout=timeout, cwd=cwd)

    def _split(self, ref: RemoteRef) -> tuple[str, str]:
        m = _SAFE_REF.match(ref.remote_id)
        if m is None:
            raise NotFound(f"kaggle has no gpu-router kernel {ref.remote_id!r}", provider=self.name)
        return m.group(1), m.group(2)

    def _dir(self, *parts: str) -> Path:
        d = self.scratch_dir.joinpath(*parts)
        d.mkdir(mode=0o700, parents=True, exist_ok=True)
        return d

    def username(self) -> str:
        """Kaggle user that owns our kernels: settings, Keychain doc, else `config view`
        (which prints the username and never the key)."""
        with self._lock:
            if self._username is not None:
                return self._username
        name = self._configured_user or self._credentials().username
        if not name:
            res = self._run(["config", "view"], timeout=T_CONFIG)
            name = parse.parse_config_username(res.output)
            if not name:
                if not res.ok:
                    raise classify(self.name, "config view", res)
                raise AuthRequired(
                    "kaggle CLI has no username configured", provider=self.name, hint=LOGIN_HINT
                )
        with self._lock:
            self._username = name
        return name

    def _ref(self, owner: str, slug: str, **meta: str | None) -> RemoteRef:
        clean = {k: v for k, v in meta.items() if v is not None}
        return RemoteRef(
            remote_id=f"{owner}/{slug}",
            url=clean.pop("url", None) or f"https://www.kaggle.com/code/{owner}/{slug}",
            meta={"owner": owner, "slug": slug, **clean},
        )

    def _kernel_status(self, remote_id: str, *, timeout: float = T_STATUS) -> parse.StatusOutcome:
        res = self._run(["kernels", "status", remote_id], timeout=timeout)
        outcome = parse.parse_status(res.stdout)
        if outcome is not None:
            return outcome
        if res.ok:
            raise Unavailable(
                f"kaggle kernels status printed something unexpected: {snippet(res.output)}",
                provider=self.name,
            )
        raise classify(self.name, "kernels status", res)

    def _confirm_missing(self) -> None:
        """A kernel-scoped 401/403 looks exactly like a missing kernel; prove the account
        works (quota needs auth) before anyone acts on NotFound (invariant 6)."""
        self._quota_rows()

    def _quota_rows(self, timeout: float = T_QUOTA) -> dict[str, parse.QuotaRow]:
        res = self._run(["quota", "--format", "json"], timeout=timeout)
        if not res.ok:
            raise classify(self.name, "quota", res)
        try:
            return parse.parse_quota(res.stdout)
        except ValueError:
            raise Unavailable(
                f"kaggle quota printed something unexpected: {snippet(res.output)}",
                provider=self.name,
            ) from None

    # ------------------------------------------------------------------ submit

    def _job_options(self, job: Job) -> dict[str, Any]:
        raw = job.spec.provider_options.get(self.name) or {}
        unknown = set(raw) - _JOB_OPTIONS
        if unknown:
            raise InvalidJob(
                f"unknown provider_options.{self.name} keys: {', '.join(sorted(unknown))}",
                provider=self.name,
                hint=f"valid keys: {', '.join(sorted(_JOB_OPTIONS))}",
            )
        return dict(raw)

    def _machine_shape(self, gpu: str | None, opts: Mapping[str, Any]) -> str | None:
        choice = str(opts.get("accelerator") or gpu or DEFAULT_GPU)
        if choice.lower() in ("none", "cpu"):
            return None
        name = choice.upper().removeprefix("2X")
        if name in GPU_SHAPES:
            return GPU_SHAPES[name]
        if choice in SHAPE_LABELS:
            return choice
        raise InvalidJob(
            f"kaggle has no {choice} GPUs (it offers {', '.join(GPU_SHAPES)})",
            provider=self.name,
        )

    def _session_timeout(self, opts: Mapping[str, Any]) -> int:
        cap = self.entry.session_cap_s or 12 * 3600.0
        limit = max(MIN_TIMEOUT_S, int(cap) - SESSION_MARGIN_S)
        raw = opts.get("timeout_s")
        if raw is None:
            return limit
        try:
            value = int(raw)
        except (TypeError, ValueError):
            raise InvalidJob(
                f"provider_options.{self.name}.timeout_s must be seconds", provider=self.name
            ) from None
        return max(MIN_TIMEOUT_S, min(value, limit))

    def _resume_payload(self, ctx: AttemptContext) -> tuple[bytes | None, str | None, str | None]:
        """(archive bytes, sha256, note). Only checkpoints that are files on this Mac can
        travel today (phase 5 moves checkpoints through HF Hub)."""
        ckpt = ctx.resume_from
        if ckpt is None:
            return None, None, None
        if str(ctx.env.get("GPU_RESUME_URI") or "").startswith("hf://"):
            return None, None, None  # phase 5: the runner downloads it from the bucket
        parsed = urlparse(ckpt.uri)
        path = Path(unquote(parsed.path)) if parsed.scheme == "file" else None
        if path is not None and path.is_file():
            data = path.read_bytes()
            if len(data) <= self._max_embed_mb * 1024 * 1024:
                return data, hashlib.sha256(data).hexdigest(), None
        return (
            None,
            None,
            f"checkpoint {ckpt.seq} ({ckpt.uri}) cannot reach kaggle yet; starting fresh",
        )

    def _marker(self, key: str) -> Path:
        return self._dir("submits") / f"{key}.json"

    def _intent(self, key: str) -> Path:
        return self._dir("submits") / f"{key}.pushing"

    def _push_may_be_running(self, key: str) -> bool:
        """An intent file younger than T_PUSH + grace: a push that a dead daemon started
        may still be uploading (the CLI child outlives it). A stale intent is removed."""
        path = self._intent(key)
        try:
            started = float(json.loads(path.read_text(encoding="utf-8"))["started_at"])
        except FileNotFoundError:
            return False
        except (OSError, ValueError, KeyError, TypeError):
            started = None  # written atomically, so unreadable = damaged, not in progress
        if started is not None and self.clock.now() - started < T_PUSH + PUSH_ORPHAN_GRACE_S:
            return True
        with contextlib.suppress(OSError):
            path.unlink()
        return False

    def _push_in_flight(self, remote_id: str) -> Unavailable:
        return Unavailable(
            f"an interrupted push of {remote_id} may still be uploading; gpu-router checks "
            "again shortly",
            provider=self.name,
        )

    def submit(self, job: Job, ctx: AttemptContext) -> RemoteRef:
        key = ctx.attempt_key
        slug = remote.slug_for_key(key)
        if slug is None:
            raise InvalidJob(f"attempt key {key!r} is not a gpu-router key", provider=self.name)
        opts = self._job_options(job)
        shape = self._machine_shape(ctx.gpu or job.spec.gpu, opts)
        if ctx.secrets and self._secrets_slug is None:
            raise InvalidJob(
                "kaggle secrets are off (providers.kaggle.secrets_dataset is false)",
                provider=self.name,
                hint="enable the secrets dataset or run jobs that need secrets elsewhere",
            )
        archive = ctx.bundle_archive
        if archive is None or not archive.is_file():
            raise InvalidJob("the job has no bundle to send to kaggle", provider=self.name)
        size = archive.stat().st_size
        if size > self._max_embed_mb * 1024 * 1024:
            raise InvalidJob(
                f"bundle is {size / 1e6:.1f} MB; kaggle takes up to {self._max_embed_mb:g} MB",
                provider=self.name,
                hint="ship data through HF Hub or /data instead of the project folder",
            )

        # Idempotency (A4): a marker from an earlier push, else the kernel itself.
        marker = self._marker(key)
        if marker.is_file():
            with contextlib.suppress(OSError, ValueError):
                return RemoteRef.model_validate_json(marker.read_text(encoding="utf-8"))
        owner = self.username()
        remote_id = f"{owner}/{slug}"
        gpu_label = SHAPE_LABELS.get(shape, "cpu") if shape else "cpu"
        try:
            self._kernel_status(remote_id, timeout=T_PRECHECK)
        except NotFound:
            if self._push_may_be_running(key):
                raise self._push_in_flight(remote_id) from None  # never a second version
        else:
            ref = self._ref(owner, slug, attempt_key=key, gpu=gpu_label, machine_shape=shape)
            _atomic_write(marker, ref.model_dump_json())
            return ref

        secrets_ref: str | None = None
        uploaded = False
        if ctx.secrets:
            secrets_ref, uploaded = self._ensure_secrets_dataset(owner, ctx.secrets)

        if shape is not None:
            self._quota_precheck()

        bundle = archive.read_bytes()
        resume, resume_sha, resume_note = self._resume_payload(ctx)
        seq_start = ctx.resume_from.seq + 1 if ctx.resume_from is not None else 1
        runner_py = remote.render_runner(
            attempt_key=key,
            bundle=bundle,
            bundle_sha256=hashlib.sha256(bundle).hexdigest(),
            env=dict(ctx.env),
            ckpt_seq_start=seq_start,
            checkpoint_interval_min=ctx.checkpoint_interval_min,
            resume=resume,
            resume_sha256=resume_sha,
            resume_note=resume_note,
            secrets_dataset=secrets_ref,
        )
        internet = bool(opts.get("enable_internet", self._internet))
        meta = remote.kernel_metadata(
            owner=owner,
            slug=slug,
            title=remote.kernel_title(key),
            machine_shape=shape,
            enable_internet=internet,
            dataset_sources=[secrets_ref] if secrets_ref else None,
        )
        status_uri = self._status_uri(job, ctx)
        folder = self._dir("push") / slug
        shutil.rmtree(folder, ignore_errors=True)
        folder.mkdir(mode=0o700, parents=True)
        intent = self._intent(key)
        try:
            _atomic_write(folder / "kernel-metadata.json", json.dumps(meta, indent=2) + "\n")
            _atomic_write(folder / "run.py", runner_py)
            timeout_s = self._session_timeout(opts)
            # Written before the push, removed once the call returned (subprocess.run kills
            # the child at its timeout): only a daemon death leaves it behind (D35).
            _atomic_write(intent, json.dumps({"started_at": self.clock.now()}))
            res = self._run(
                ["kernels", "push", "-p", str(folder), "-t", str(timeout_s)],
                timeout=T_PUSH_AFTER_SECRETS if uploaded else T_PUSH,
            )
        finally:
            with contextlib.suppress(OSError):
                intent.unlink()
            shutil.rmtree(folder, ignore_errors=True)
        outcome = parse.parse_push(res.output)
        if outcome is not None and outcome.ok:
            ref = self._ref(
                owner,
                slug,
                url=outcome.url,
                attempt_key=key,
                version=str(outcome.version) if outcome.version is not None else None,
                gpu=gpu_label,
                machine_shape=shape,
                status_uri=status_uri,
                # the kernel's real limit (`-t`, maybe a job's shorter timeout_s): the
                # engine plans a handoff against it, not only the catalog cap (D44)
                session_s=str(timeout_s),
            )
            _atomic_write(marker, ref.model_dump_json())
            return ref
        if outcome is not None and outcome.error is not None:
            err = _classify_push_error(self.name, outcome.error)
            self._raise_if_created(remote_id, err)
        raise self._push_failure(res)

    @staticmethod
    def _status_uri(job: Job, ctx: AttemptContext) -> str | None:
        """Where this attempt's runner pushes heartbeat/log-tail (phase 5), if anywhere."""
        root = str(ctx.env.get("GPU_STORAGE") or "").rstrip("/")
        return f"{root}/jobs/{job.id}/attempts/{ctx.n}" if root else None

    def _secrets_salt(self) -> bytes:
        path = self._dir("secrets") / "salt"
        try:
            data = path.read_bytes()
            if len(data) >= 32:
                return data
        except OSError:
            pass
        data = os.urandom(32)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        return data

    def _dataset_exists(self, ref: str) -> bool:
        res = self._run(["datasets", "status", ref], timeout=T_SECRETS_STATUS)
        status = (res.stdout.strip().splitlines() or [""])[-1].strip().lower()
        if res.ok and status:
            return status not in ("deleted",)
        low = res.output.lower()
        if any(m in low for m in ("404", "not found", "403", "forbidden", "cannot access")):
            return False
        raise classify(self.name, "datasets status", res)

    def _ensure_secrets_dataset(self, owner: str, values: Mapping[str, Any]) -> tuple[str, bool]:
        """(`<owner>/<slug>`, uploaded now). The private dataset holds this attempt's
        secret values as JSON; a new version (old ones deleted) is made only when they
        changed since the last upload (keyed digest, never the values, in scratch)."""
        assert self._secrets_slug is not None
        ref = f"{owner}/{self._secrets_slug}"
        payload = json.dumps(
            {"v": 1, "values": {k: v.get_secret_value() for k, v in sorted(values.items())}},
            sort_keys=True,
        ).encode()
        digest = hmac.new(self._secrets_salt(), payload + ref.encode(), hashlib.sha256).hexdigest()
        marker = self._dir("secrets") / "dataset.json"
        with self._lock:
            try:
                seen = json.loads(marker.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                seen = {}
        if (
            isinstance(seen, dict)
            and seen.get("ref") == ref
            and seen.get("digest") == digest
            and seen.get("ready")
        ):
            return ref, False
        exists = self._dataset_exists(ref)
        folder = Path(tempfile.mkdtemp(prefix="push-", dir=self._dir("secrets")))
        try:
            meta = {"title": "gpu-router secrets", "id": ref, "licenses": [{"name": "unknown"}]}
            _atomic_write(folder / "dataset-metadata.json", json.dumps(meta) + "\n")
            fd = os.open(folder / remote.SECRETS_FILE, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as fh:
                fh.write(payload)
            if exists:
                args = ["datasets", "version", "-p", str(folder), "-m", "gpu-router", "-d", "-q"]
            else:
                args = ["datasets", "create", "-p", str(folder), "-q"]
            res = self._run(args, timeout=T_SECRETS_UPLOAD)
        finally:
            shutil.rmtree(folder, ignore_errors=True)  # plaintext values: gone at once
        low = res.output.lower()
        if "being created" not in low:
            err = classify(self.name, f"datasets {args[1]}", res)
            if isinstance(err, (AuthRequired, RateLimited)):
                raise err
            raise Unavailable(
                f"kaggle did not take the secrets dataset {ref}: {snippet(res.output)}",
                provider=self.name,
            )
        ready = False
        for i in range(SECRETS_READY_POLLS):
            st = self._run(["datasets", "status", ref], timeout=T_SECRETS_READY)
            status = (st.stdout.strip().splitlines() or [""])[-1].strip().lower()
            if st.ok and status == "ready":
                ready = True
                break
            if st.ok and status in ("failed", "deleted"):
                raise Unavailable(
                    f"kaggle could not process the secrets dataset {ref} ({status})",
                    provider=self.name,
                )
            if i + 1 < SECRETS_READY_POLLS:
                self._sleep(SECRETS_READY_PAUSE_S)
        with self._lock:
            _atomic_write(
                marker,
                json.dumps({"ref": ref, "digest": digest, "ready": ready, "at": self.clock.now()}),
            )
        return ref, True

    def _quota_precheck(self) -> None:
        """Definitive QuotaExhausted before pushing when the weekly GPU hours are gone.
        A quota endpoint outage does not block the push (the push reports quota itself)."""
        try:
            rows = self._quota_rows(timeout=T_PRECHECK)
        except (Unavailable, NotFound):
            return
        gpu = rows.get("GPU")
        if gpu is not None and gpu.total_h > 0 and gpu.remaining_h <= QUOTA_EPSILON_H:
            raise QuotaExhausted(
                f"kaggle weekly GPU quota is used up ({gpu.used_h:.2f}/{gpu.total_h:.2f} h)",
                provider=self.name,
                resets_at=gpu.refresh_at,
                hint="gpu-router uses other providers until it resets",
            )

    def _raise_if_created(self, remote_id: str, err: AdapterError) -> None:
        """Raise `err` only if the kernel really does not exist (A3); else Unavailable so
        the engine resolves the attempt through lookup_by_key."""
        try:
            self._kernel_status(remote_id, timeout=T_PRECHECK)
        except NotFound:
            raise err from None
        except AdapterError:
            raise Unavailable(
                f"kaggle push failed ({err.message}) and the kernel could not be checked",
                provider=self.name,
            ) from None
        raise Unavailable(
            f"kaggle push reported an error but {remote_id} exists ({err.message})",
            provider=self.name,
        )

    def _push_failure(self, res: CliResult) -> AdapterError:
        """No success line and no server push error: decide what is definitive."""
        low = res.output.lower()
        if not res.ok and any(m in low for m in _LOCAL_PUSH_ERRORS):
            # the CLI validated metadata locally and sent nothing
            return InvalidJob(
                f"kaggle push refused locally: {snippet(res.output)}", provider=self.name
            )
        if not res.ok and "403 client error" in low:
            return AuthRequired(
                "kaggle refused the push (403 forbidden)", provider=self.name, hint=PHONE_HINT
            )
        err = classify(self.name, "kernels push", res)
        if isinstance(err, (AuthRequired, RateLimited)):
            return err  # 401 / 429: rejected before anything was saved
        return Unavailable(
            f"kaggle push outcome unknown: {err.message}", provider=self.name, hint=err.hint
        )

    def lookup_by_key(self, attempt_key: str) -> RemoteRef | None:
        slug = remote.slug_for_key(attempt_key)
        if slug is None:
            return None
        marker = self._marker(attempt_key)
        if marker.is_file():
            with contextlib.suppress(OSError, ValueError):
                return RemoteRef.model_validate_json(marker.read_text(encoding="utf-8"))
        if self._offline:
            return None  # every push is refused while offline, so nothing can exist
        owner = self.username()
        try:
            self._kernel_status(f"{owner}/{slug}")
        except NotFound:
            if self._tombstone(slug).exists():
                return None
            if self._push_may_be_running(attempt_key):
                raise self._push_in_flight(f"{owner}/{slug}") from None
            self._confirm_missing()
            return None
        return self._ref(owner, slug, attempt_key=attempt_key)

    # ------------------------------------------------------------------ status

    def _final_path(self, slug: str) -> Path:
        return self._dir("final") / f"{slug}.json"

    def _tombstone(self, slug: str) -> Path:
        return self._dir("cancelled") / slug

    def _read_final(self, slug: str) -> dict[str, Any] | None:
        path = self._final_path(slug)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return data if isinstance(data, dict) else None

    def _fetch_log(self, remote_id: str) -> list[str]:
        res = self._run(["kernels", "logs", remote_id], timeout=T_FINAL_LOG)
        if not res.ok:
            raise classify(self.name, "kernels logs", res)
        return parse.trim_post_run(parse.parse_log(res.stdout))

    def _empty_log_path(self, slug: str) -> Path:
        return self._dir("final") / f"{slug}.empty-log"

    def _log_not_published(self, remote_id: str, slug: str) -> bool:
        """True while an empty final log still counts as "not published yet": run.py
        prints before anything else, so a finished kernel with no log lines has a log
        Kaggle has not made available. Bounded by EMPTY_LOG_GRACE_S from the first sight."""
        path = self._empty_log_path(slug)
        now = self.clock.now()
        try:
            first = float(path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            first = now
            _atomic_write(path, f"{now:.3f}\n")
        return now - first < EMPTY_LOG_GRACE_S

    def _finalize(self, remote_id: str, slug: str, outcome: parse.StatusOutcome) -> dict[str, Any]:
        """Work out the final outcome of a finished kernel once, and cache it."""
        lines: list[str]
        if outcome.status.cancelled:
            try:
                lines = self._fetch_log(remote_id)
            except AdapterError:
                lines = []
            record: dict[str, Any] = {"phase": str(RemotePhase.CANCELLED)}
        else:
            lines = self._fetch_log(remote_id)
            if not lines and self._log_not_published(remote_id, slug):
                # COMPLETE and ERROR alike: judging now would cache a verdict with no exit
                # line (a failed script would be "lost" and re-run elsewhere).
                raise Unavailable(
                    f"{remote_id} finished but kaggle has not published its log yet",
                    provider=self.name,
                )
            record = self._judge(outcome, lines)
        # Redacted before it hits disk, one output line per input line (cursors = indexes):
        # the engine redacts its own copy, this is the provider-side one.
        lines = [redact(line) for line in lines]
        record.update(
            kernel_status=str(outcome.status),
            failure_message=outcome.failure_message,
            lines=lines,
        )
        _atomic_write(self._final_path(slug), json.dumps(record))
        with contextlib.suppress(OSError):
            self._empty_log_path(slug).unlink()
        return record

    def _judge(self, outcome: parse.StatusOutcome, lines: list[str]) -> dict[str, Any]:
        code = parse.last_exit_code(lines)
        if code == 0:
            return {"phase": str(RemotePhase.SUCCEEDED), "exit_code": 0}
        if code == INSTALL_FAILED_EXIT:
            return {
                "phase": str(RemotePhase.LOST),
                "exit_code": code,
                "lost_reason": "dependency install failed on kaggle",
            }
        if code is not None:
            return {"phase": str(RemotePhase.FAILED), "exit_code": code}
        failure = (outcome.failure_message or "").strip()
        low = failure.lower()
        if "quota" in low:
            return {
                "phase": str(RemotePhase.LOST),
                "lost_reason": f"kaggle GPU quota ran out: {snippet(failure, 120)}",
                "quota_exhausted": True,
            }
        if any(m in low for m in _TIME_LIMIT_MARKERS):
            return {"phase": str(RemotePhase.LOST), "lost_reason": "kaggle session time limit"}
        exhausted = False
        with contextlib.suppress(AdapterError):
            gpu = self._quota_rows(timeout=T_FINAL_QUOTA).get("GPU")
            exhausted = bool(gpu and gpu.total_h > 0 and gpu.remaining_h <= QUOTA_EPSILON_H)
        if exhausted:
            return {
                "phase": str(RemotePhase.LOST),
                "lost_reason": "kaggle GPU quota ran out",
                "quota_exhausted": True,
            }
        if failure:
            reason = f"kaggle stopped the session: {snippet(failure, 160)}"
        elif outcome.status is parse.KernelStatus.COMPLETE:
            reason = "kaggle finished but the runner never reported an exit code"
        else:
            reason = "kaggle session ended before the runner finished"
        return {"phase": str(RemotePhase.LOST), "lost_reason": reason}

    def _status_from_final(self, ref: RemoteRef, record: Mapping[str, Any]) -> RemoteStatus:
        phase = RemotePhase(str(record.get("phase")))
        code = record.get("exit_code")
        message = {
            RemotePhase.SUCCEEDED: "finished",
            RemotePhase.FAILED: f"script exited with code {code}",
            RemotePhase.CANCELLED: "cancelled",
            RemotePhase.LOST: f"session ended: {record.get('lost_reason')}",
        }.get(phase, str(phase))
        return RemoteStatus(
            phase=phase,
            message=message,
            exit_code=int(code) if isinstance(code, int) else None,
            lost_reason=record.get("lost_reason") if phase is RemotePhase.LOST else None,
            quota_exhausted=bool(record.get("quota_exhausted")) and phase is RemotePhase.LOST,
            gpu=ref.meta.get("gpu"),
            url=ref.url,
        )

    def _cancelled_status(self, ref: RemoteRef) -> RemoteStatus:
        return RemoteStatus(
            phase=RemotePhase.CANCELLED,
            message="cancelled (kernel deleted by gpu-router)",
            gpu=ref.meta.get("gpu"),
            url=ref.url,
        )

    def status(self, ref: RemoteRef) -> RemoteStatus:
        _owner, slug = self._split(ref)
        final = self._read_final(slug)
        if final is not None:
            return self._status_from_final(ref, final)
        try:
            outcome = self._kernel_status(ref.remote_id)
        except NotFound:
            if self._tombstone(slug).exists():
                return self._cancelled_status(ref)
            self._confirm_missing()
            raise
        st = outcome.status
        if st in (parse.KernelStatus.QUEUED, parse.KernelStatus.NEW_SCRIPT):
            return RemoteStatus(
                phase=RemotePhase.PENDING,
                message="queued on kaggle, waiting for a GPU",
                gpu=ref.meta.get("gpu"),
                url=ref.url,
            )
        if st is parse.KernelStatus.RUNNING:
            return RemoteStatus(
                phase=RemotePhase.RUNNING,
                message=f"running on kaggle {ref.meta.get('gpu') or ''}".rstrip(),
                gpu=ref.meta.get("gpu"),
                url=ref.url,
            )
        return self._status_from_final(ref, self._finalize(ref.remote_id, slug, outcome))

    # ------------------------------------------------------------------ logs

    def logs(
        self, ref: RemoteRef, *, follow: bool = False, since: str | None = None
    ) -> Iterator[LogChunk]:
        _owner, slug = self._split(ref)
        tail = sidechannel.parse_cursor(since)
        try:
            start = max(0, int(since)) if since and tail is None else 0
        except ValueError:
            start = 0
        final = self._read_final(slug)
        if final is None:
            st = self.status(ref)
            if st.phase.terminal:
                final = self._read_final(slug) or {"lines": []}
        if final is None:
            status_uri = ref.meta.get("status_uri")
            if status_uri and (tail is not None or start == 0):
                consumed, last = tail if tail is not None else (0, None)
                read = sidechannel.read_tail(status_uri, consumed, last)
                if read is not None and read.lines:
                    yield LogChunk(
                        lines=read.lines,
                        cursor=sidechannel.format_cursor(read.consumed, read.last_hash),
                        eof=False,
                    )
                    return
            yield LogChunk(lines=[], cursor=since or str(start), eof=False)
            return
        lines = [str(x) for x in final.get("lines") or []]
        if tail is not None:
            # switching from the live tail to the final log: the lines run.py printed
            # before the runner started (notes about secrets/resume) were never in the
            # tail; they come first, then the log continues after the last line served
            start = sidechannel.align(lines, *tail)
            preamble = sidechannel.preamble(lines)
            if preamble and start > len(preamble):
                rest = lines[start : start + CHUNK_LINES]
                hi = start + len(rest)
                yield LogChunk(lines=preamble + rest, cursor=str(hi), eof=hi >= len(lines))
                start = hi
                if start >= len(lines):
                    return
        if start >= len(lines):
            yield LogChunk(lines=[], cursor=str(max(start, len(lines))), eof=True)
            return
        for lo in range(start, len(lines), CHUNK_LINES):
            hi = min(lo + CHUNK_LINES, len(lines))
            yield LogChunk(lines=lines[lo:hi], cursor=str(hi), eof=hi == len(lines))

    # ------------------------------------------------------------------ fetch

    def fetch(self, ref: RemoteRef, dest: Path) -> FetchResult:
        _owner, slug = self._split(ref)
        st = self.status(ref)
        if not st.phase.terminal:
            raise NotFound(
                f"{ref.remote_id} is still {st.phase}; outputs appear when it finishes",
                provider=self.name,
            )
        if st.phase is RemotePhase.CANCELLED and self._tombstone(slug).exists():
            raise NotFound(f"{ref.remote_id} was deleted by cancel", provider=self.name)
        tmp = Path(tempfile.mkdtemp(prefix=f"{slug}-", dir=self._dir("fetch")))
        try:
            res = self._run(
                [
                    "kernels",
                    "output",
                    ref.remote_id,
                    "-p",
                    str(tmp),
                    "--file-pattern",
                    remote.OUTPUT_PATTERN,
                    "-o",
                    "-q",
                ],
                timeout=T_OUTPUT,
            )
            if not res.ok:
                raise classify(self.name, "kernels output", res)
            files, total = self._copy_outputs(tmp / remote.OUTPUT_PREFIX, dest)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        if files == 0 and st.phase is not RemotePhase.SUCCEEDED:
            raise NotFound(f"{ref.remote_id} left no outputs (run {st.phase})", provider=self.name)
        return FetchResult(dest=dest, files=files, bytes=total)

    def _copy_outputs(self, src: Path, dest: Path) -> tuple[int, int]:
        dest.mkdir(parents=True, exist_ok=True)
        if not src.is_dir():
            return 0, 0
        root = dest.resolve()
        files = total = 0
        for path in sorted(src.rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            rel = path.relative_to(src)
            target = (dest / rel).resolve()
            if root not in target.parents:
                continue  # never write outside dest
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(f".{target.name}.gpu-fetch")
            shutil.copyfile(path, tmp)
            os.replace(tmp, target)
            files += 1
            total += target.stat().st_size
        return files, total

    # ------------------------------------------------------------------ cancel

    def cancel(self, ref: RemoteRef) -> None:
        try:
            _owner, slug = self._split(ref)
        except NotFound:
            return
        if self._tombstone(slug).exists():
            return
        try:
            st = self.status(ref)
        except NotFound:
            return
        if st.phase.terminal:
            return
        res = self._run(["kernels", "delete", ref.remote_id, "-y"], timeout=T_DELETE)
        low = res.output.lower()
        if res.ok or "deleted successfully" in low or "403 client error" in low:
            # 403 on delete = the kernel is already gone (verified live)
            _atomic_write(self._tombstone(slug), f"{self.clock.now():.3f}\n")
            return
        raise classify(self.name, "kernels delete", res)

    # ------------------------------------------------------------------ quota / health

    def quota(self) -> QuotaSnapshot:
        rows = self._quota_rows()
        gpu = rows.get("GPU")
        now = self.clock.now()
        if gpu is None:
            raise Unavailable("kaggle quota has no GPU row", provider=self.name)
        detail: dict[str, Any] = {"remaining_h": gpu.remaining_h}
        tpu = rows.get("TPU")
        if tpu is not None:
            detail["tpu_used_h"] = tpu.used_h
            detail["tpu_limit_h"] = tpu.total_h
        return QuotaSnapshot(
            provider=self.name,
            used=gpu.used_h,
            limit=gpu.total_h,
            unit=QuotaUnit.GPU_HOURS,
            resets_at=gpu.refresh_at,
            source="live",
            detail=detail,
            observed_at=now,
        )

    def healthcheck(self) -> Health:
        now = self.clock.now()
        if self._offline:
            return Health(
                health=ProviderHealth.DISABLED,
                reason=OFFLINE_REASON,
                hint=OFFLINE_HINT,
                checked_at=now,
            )
        try:
            ver = self._run(["--version"], timeout=T_HEALTH)
            version = parse.parse_version(ver.output)
            if version is None:
                return Health(
                    health=ProviderHealth.UNAVAILABLE,
                    reason=f"kaggle --version printed something unexpected: {snippet(ver.output)}",
                    hint="reinstall with `uv tool install --force kaggle`",
                    checked_at=now,
                )
            owner = self.username()
            rows = self._quota_rows(timeout=T_HEALTH)
        except AuthRequired as exc:
            return Health(
                health=ProviderHealth.AUTH_REQUIRED,
                reason=exc.message,
                hint=exc.hint or LOGIN_HINT,
                checked_at=now,
            )
        except (RateLimited, Unavailable, NotFound, Permanent, InvalidJob, QuotaExhausted) as exc:
            return Health(
                health=ProviderHealth.UNAVAILABLE,
                reason=exc.message,
                hint=exc.hint,
                checked_at=now,
            )
        gpu = rows.get("GPU")
        detail: dict[str, Any] = {
            "cli_version": version,
            "username": owner,
            "credentials": self._cred_source,
        }
        if gpu is not None:
            detail["gpu_used_h"] = gpu.used_h
            detail["gpu_limit_h"] = gpu.total_h
            detail["resets_at"] = gpu.refresh_at
        return Health(health=ProviderHealth.OK, checked_at=now, detail=detail)
