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
- large payloads (2026-10-04): SaveKernel refuses a code file over ~1 MB (probed live,
  remote.MAX_INLINE_SOURCE), so a bundle or resume archive that would push run.py past it
  travels as a private content-addressed dataset `<user>/gpu-router-<kind>-<sha16>`
  (`_ensure_blob`, reused by later attempts and jobs; records in
  <home>/providers/kaggle/blobs/), and `stage_data` puts `data:` datasets there too when
  no checkpoint storage exists (one tar per directory, the file itself for a file).
  Unused blobs are deleted by a background sweep (bundle/ckpt after 3 days, data after
  `data_keep_days`, default 30). A `400 Bad Request` from SaveKernel is definitive
  (InvalidJob): the server refused the save, nothing was created.
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
import tarfile
import tempfile
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
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
    StagedData,
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
# Bundles up to this size reach a kernel (inline in run.py below remote.MAX_INLINE_SOURCE,
# else as a blob dataset uploaded inside submit's budget). Was 10 MB inline, which Kaggle
# never accepted: SaveKernel refuses a source over ~1 MB (probed 2026-10-04).
DEFAULT_MAX_BUNDLE_MB = 100.0
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
# Blob datasets (2026-10-04). submit() spends at most SUBMIT_BUDGET_S (engine 300) across
# every call incl. blob uploads; a push gets what is left, at least MIN_PUSH_S.
# stage_data() spends at most T_STAGE_TOTAL (engine timeouts.stage_data 3600).
SUBMIT_BUDGET_S = 270.0
MIN_PUSH_S = 45.0
T_STAGE_TOTAL = 3300.0
T_BLOB_STATUS = 15.0
BLOB_READY_PAUSE_S = 5.0
BLOB_UPLOAD_GRACE_S = 900.0  # a fresh upload may read 403/404 this long before it shows
BLOB_MAX_TIMEOUTS = 2  # a resume archive that timed out this often: start fresh instead
BLOB_SWEEP_EVERY_S = 6 * 3600.0
BLOB_SWEEP_MAX = 10  # deletes per sweep
BLOB_KEEP_DAYS = {"bundle": 3.0, "ckpt": 3.0, "data": 30.0}
BLOB_SWEEP = True  # tests switch the background sweep off
BLOB_RETRY_S = 60  # retry_after when blob trouble stopped a submit before its push

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
_BLOB_REF = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}/gpu-router-(?:bundle|ckpt|data)-[0-9a-f]{16}$"
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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_size(source: Path, files: Sequence[tuple[str, int]] | None) -> int:
    if files is not None:
        return sum(size for _rel, size in files)
    try:
        return source.stat().st_size
    except OSError:
        return 0


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
        max_bundle_mb=DEFAULT_MAX_BUNDLE_MB,
        poll_interval_s=60,
        stage_data=True,
    )

    def __init__(self, deps: AdapterDeps, *, runner: Runner | None = None) -> None:
        super().__init__(deps)
        extra: dict[str, Any] = dict(deps.settings.model_extra or {})
        self._configured_user: str | None = extra.get("username") or None
        self._cli_path: str | None = extra.get("cli_path") or None
        self._cred_mode = str(extra.get("credentials") or "auto")
        self._max_bundle_mb = float(
            extra.get("max_bundle_mb") or extra.get("max_embed_mb") or DEFAULT_MAX_BUNDLE_MB
        )
        self._blobs = extra.get("blob_datasets", True) not in (False, "false", "off", 0)
        keep = dict(BLOB_KEEP_DAYS)
        with contextlib.suppress(TypeError, ValueError):
            keep["data"] = float(extra.get("data_keep_days") or keep["data"])
        self._blob_keep_s = {k: v * 86400.0 for k, v in keep.items()}
        self._blob_locks: dict[str, threading.Lock] = {}
        self._last_sweep = -1e18
        self._sweep_thread: threading.Thread | None = None
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
            max_bundle_mb=self._max_bundle_mb,
            poll_interval_s=deps.entry.poll_interval_s,
            stage_data=self._blobs,
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

    def _resume_source(self, ctx: AttemptContext) -> tuple[Path | None, str | None]:
        """(archive on this Mac, note). Only checkpoints that are files on this Mac can
        travel without checkpoint storage; they go inline or as a blob dataset."""
        ckpt = ctx.resume_from
        if ckpt is None:
            return None, None
        if str(ctx.env.get("GPU_RESUME_URI") or "").startswith("hf://"):
            return None, None  # phase 5: the runner downloads it from the bucket
        parsed = urlparse(ckpt.uri)
        path = Path(unquote(parsed.path)) if parsed.scheme == "file" else None
        if path is not None and path.is_file():
            return path, None
        return None, f"checkpoint {ckpt.seq} ({ckpt.uri}) cannot reach kaggle yet; starting fresh"

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
        deadline = self.clock.monotonic() + SUBMIT_BUDGET_S
        size = archive.stat().st_size
        cap_mb = self._max_bundle_mb if self._blobs else remote.MAX_INLINE_SOURCE * 3 / 4 / 1e6
        if size > cap_mb * 1e6:
            raise InvalidJob(
                f"bundle is {size / 1e6:.1f} MB; kaggle takes up to {cap_mb:g} MB",
                provider=self.name,
                hint="pass big files as data= (datasets) instead of shipping them as code",
            )
        data_refs = self._data_refs(ctx)

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

        try:
            runner_py, blob_refs, blob_uploaded = self._runner_source(
                owner, key, ctx, archive, secrets_ref, deadline
            )
        except Unavailable as exc:
            raise self._retry_soon(exc.message, exc.hint) from None
        uploaded = uploaded or blob_uploaded
        internet = bool(opts.get("enable_internet", self._internet))
        sources = [r for r in [secrets_ref, *blob_refs, *data_refs] if r]
        meta = remote.kernel_metadata(
            owner=owner,
            slug=slug,
            title=remote.kernel_title(key),
            machine_shape=shape,
            enable_internet=internet,
            dataset_sources=list(dict.fromkeys(sources)) or None,
        )
        push_s = min(T_PUSH_AFTER_SECRETS if uploaded else T_PUSH, self._left(deadline) - 25)
        if push_s < MIN_PUSH_S:
            raise self._retry_soon(
                "kaggle submit ran out of time uploading its datasets (they are kept and "
                "reused); gpu-router submits again shortly",
                None,
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
                ["kernels", "push", "-p", str(folder), "-t", str(timeout_s)], timeout=push_s
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

    # ------------------------------------------------------------------ blob datasets

    def _left(self, deadline: float) -> float:
        return deadline - self.clock.monotonic()

    def _data_refs(self, ctx: AttemptContext) -> list[str]:
        """Blob datasets the runner's GPU_DATA names (staged by stage_data), to attach."""
        raw = ctx.env.get("GPU_DATA")
        if not raw:
            return []
        try:
            items = json.loads(raw)
        except ValueError:
            return []
        refs: list[str] = []
        for item in items if isinstance(items, list) else []:
            uri = str(item.get("uri") or "") if isinstance(item, dict) else ""
            if not uri.startswith(remote.DATA_URI_PREFIX):
                continue
            ref = uri[len(remote.DATA_URI_PREFIX) :].rpartition("/")[0]
            if not _BLOB_REF.match(ref):
                raise InvalidJob(f"dataset uri {uri!r} is not a gpu-router kaggle dataset")
            refs.append(ref)
        return refs

    def _runner_source(
        self,
        owner: str,
        key: str,
        ctx: AttemptContext,
        archive: Path,
        secrets_ref: str | None,
        deadline: float,
    ) -> tuple[str, list[str], bool]:
        """(run.py text, blob datasets to attach, uploaded one now). Bundle and resume
        archive ride inline while run.py stays under remote.MAX_INLINE_SOURCE; else the
        bigger one moves to a blob dataset first."""
        bundle = archive.read_bytes()
        bundle_sha = hashlib.sha256(bundle).hexdigest()
        resume_path, resume_note = self._resume_source(ctx)
        resume_sha = _file_sha256(resume_path) if resume_path is not None else None
        seq_start = ctx.resume_from.seq + 1 if ctx.resume_from is not None else 1

        def fits_inline(n: int) -> bool:
            return n * 4 // 3 < remote.MAX_INLINE_SOURCE

        bundle_in: tuple[str, str] | None = None
        resume_in: tuple[str, str] | None = None
        refs: list[str] = []
        uploaded = False
        move_bundle = not fits_inline(len(bundle))
        move_resume = resume_path is not None and not fits_inline(resume_path.stat().st_size)
        for _ in range(3):
            if move_resume and resume_in is None and resume_path is not None:
                assert resume_sha is not None
                got = self._resume_blob(owner, resume_path, resume_sha, deadline)
                if got is None:
                    resume_path = resume_sha = None
                    seq = ctx.resume_from.seq if ctx.resume_from else "?"
                    resume_note = (
                        f"checkpoint {seq} could not be uploaded to kaggle in time; starting fresh"
                        if self._blobs
                        else f"checkpoint {seq} is too big to ride inside a kaggle kernel and "
                        "providers.kaggle.blob_datasets is off; starting fresh"
                    )
                else:
                    ref, name, up = got
                    resume_in, uploaded = (ref, name), uploaded or up
                    refs.append(ref)
            if move_bundle and bundle_in is None:
                self._need_blobs("the job's bundle", len(bundle))
                ref, name, up = self._ensure_blob(owner, "bundle", bundle_sha, archive, deadline)
                bundle_in, uploaded = (ref, name), uploaded or up
                refs.append(ref)
            text = remote.render_runner(
                attempt_key=key,
                bundle=None if bundle_in else bundle,
                bundle_input=bundle_in,
                bundle_sha256=bundle_sha,
                env=dict(ctx.env),
                ckpt_seq_start=seq_start,
                checkpoint_interval_min=ctx.checkpoint_interval_min,
                resume=resume_path.read_bytes() if resume_path and not resume_in else None,
                resume_sha256=resume_sha,
                resume_input=resume_in,
                resume_note=resume_note,
                secrets_dataset=secrets_ref,
            )
            if len(text.encode()) <= remote.MAX_INLINE_SOURCE:
                return text, refs, uploaded
            if resume_path is not None and resume_in is None:
                move_resume = True
            elif bundle_in is None:
                move_bundle = True
            else:
                break
        raise InvalidJob(
            f"kaggle's run.py would be over its {remote.MAX_INLINE_SOURCE // 1000} KB source "
            "limit even with the bundle in a dataset (is the job's env very large?)",
            provider=self.name,
        )

    def _retry_soon(self, message: str, hint: str | None) -> RateLimited:
        """Blob trouble before the push: nothing was created, so it is definitive (no
        ambiguous lookup) and only a short pause (BLOB_RETRY_S), not an outage cooldown;
        the uploads made so far are reused on the next try."""
        return RateLimited(message, provider=self.name, retry_after=BLOB_RETRY_S, hint=hint)

    def _need_blobs(self, what: str, size: int) -> None:
        if not self._blobs:
            raise InvalidJob(
                f"{what} ({size / 1e6:.1f} MB) is too big to ride inside the kernel (kaggle "
                f"takes a source up to ~{remote.MAX_INLINE_SOURCE // 1000} KB) and "
                "providers.kaggle.blob_datasets is off",
                provider=self.name,
                hint="turn providers.kaggle.blob_datasets back on",
            )

    def _resume_blob(
        self, owner: str, path: Path, sha: str, deadline: float
    ) -> tuple[str, str, bool] | None:
        """The resume archive as a blob dataset; None when it timed out too often (the
        attempt starts fresh rather than never starting)."""
        if not self._blobs:
            return None  # start fresh rather than exclude kaggle for the job
        rec = self._blob_record(remote.blob_slug("ckpt", sha))
        if int(rec.get("timeouts") or 0) >= BLOB_MAX_TIMEOUTS and not rec.get("ready"):
            return None
        return self._ensure_blob(owner, "ckpt", sha, path, deadline)

    def stage_data(self, path: Path, sha256: str, files: Sequence[tuple[str, int]]) -> StagedData:
        """A `data:` dataset as a private Kaggle dataset (no checkpoint storage needed),
        once per content hash: later jobs with the same data attach the same dataset."""
        self._need_blobs("a dataset", sum(size for _rel, size in files))
        deadline = self.clock.monotonic() + T_STAGE_TOTAL
        owner = self.username()
        listed = list(files) if path.is_dir() else None
        ref, name, uploaded = self._ensure_blob(owner, "data", sha256, path, deadline, listed)
        return StagedData(
            uri=remote.data_uri(ref, name), uploaded=uploaded, where=f"private kaggle dataset {ref}"
        )

    def _blob_lock(self, slug: str) -> threading.Lock:
        with self._lock:
            return self._blob_locks.setdefault(slug, threading.Lock())

    def _blob_record(self, slug: str) -> dict[str, Any]:
        try:
            data = json.loads((self._dir("blobs") / f"{slug}.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def _save_blob_record(self, slug: str, **fields: Any) -> None:
        rec = self._blob_record(slug)
        rec.update(fields)
        _atomic_write(self._dir("blobs") / f"{slug}.json", json.dumps(rec, sort_keys=True))

    def _blob_status(self, ref: str, deadline: float) -> str:
        """`ready`, `missing` (404, or 403 = not visible to us, also right after a create)
        or Kaggle's word for a dataset still being processed."""
        res = self._run(
            ["datasets", "status", ref], timeout=max(5.0, min(T_BLOB_STATUS, self._left(deadline)))
        )
        status = (res.stdout.strip().splitlines() or [""])[-1].strip().lower()
        if res.ok and status:
            return "missing" if status == "deleted" else status
        low = res.output.lower()
        if any(m in low for m in ("404", "not found", "403", "forbidden", "cannot access")):
            return "missing"
        raise classify(self.name, "datasets status", res)

    def _ensure_blob(
        self,
        owner: str,
        kind: str,
        sha256: str,
        source: Path,
        deadline: float,
        files: Sequence[tuple[str, int]] | None = None,
    ) -> tuple[str, str, bool]:
        """(`<owner>/<slug>`, file name, uploaded now) of the private dataset holding
        `source` (a tar of `files` when given). Reused when Kaggle still has it ready."""
        slug = remote.blob_slug(kind, sha256)
        ref = f"{owner}/{slug}"
        name = remote.blob_file(kind, sha256, tar=files is not None)
        with self._blob_lock(slug):
            rec = self._blob_record(slug)
            status = self._blob_status(ref, deadline)
            uploaded = False
            recent = rec.get("ref") == ref and (
                self.clock.now() - float(rec.get("uploaded_at") or 0) < BLOB_UPLOAD_GRACE_S
            )
            if status == "missing" and not recent:
                self._upload_blob(ref, kind, sha256, name, source, files, deadline)
                uploaded = True
            if status != "ready":
                self._wait_blob_ready(ref, deadline)
            self._save_blob_record(
                slug,
                ref=ref,
                kind=kind,
                file=name,
                ready=True,
                last_used=self.clock.now(),
                size=_source_size(source, files),
            )
        self._maybe_sweep_blobs()
        return ref, name, uploaded

    def _upload_blob(
        self,
        ref: str,
        kind: str,
        sha256: str,
        name: str,
        source: Path,
        files: Sequence[tuple[str, int]] | None,
        deadline: float,
    ) -> None:
        slug = ref.split("/")[-1]
        folder = Path(tempfile.mkdtemp(prefix=f"up-{slug}-", dir=self._dir("blobs")))
        try:
            size = _source_size(source, files)
            free = shutil.disk_usage(folder).free
            if files is not None and free < size + 512 * 1024 * 1024:
                raise InvalidJob(
                    f"not enough free disk on this Mac to pack {source.name} for kaggle "
                    f"({size / 1e9:.1f} GB needed, {free / 1e9:.1f} GB free)",
                    provider=self.name,
                )
            meta: dict[str, Any] = {
                "title": remote.blob_title(kind, sha256),
                "id": ref,
                "licenses": [{"name": "unknown"}],
            }
            _atomic_write(folder / "dataset-metadata.json", json.dumps(meta) + "\n")
            target = folder / name
            try:
                if files is not None:
                    with tarfile.open(
                        target, "w", format=tarfile.PAX_FORMAT, dereference=True
                    ) as tar:
                        for rel, _size in sorted(files):
                            tar.add(str(source / rel), arcname=rel, recursive=False)
                else:
                    try:
                        os.link(source, target)
                    except OSError:
                        shutil.copyfile(source, target)
            except OSError as exc:  # A3: a file vanished, the disk filled up
                raise Unavailable(
                    f"could not pack {source.name} for kaggle ({exc.strerror or exc})",
                    provider=self.name,
                ) from None
            budget = self._left(deadline) - 2 * T_BLOB_STATUS
            if budget < 10:
                raise Unavailable(
                    f"no time left to upload {ref} to kaggle; gpu-router tries again shortly",
                    provider=self.name,
                )
            self._save_blob_record(slug, ref=ref, kind=kind, uploading_at=self.clock.now())
            try:
                res = self._run(
                    ["datasets", "create", "-p", str(folder), "-q", "-r", "skip"], timeout=budget
                )
            except Unavailable:
                # timed out: the CLI was killed before its final create call; counted so a
                # resume archive too big for this link stops blocking the attempt
                rec = self._blob_record(slug)
                self._save_blob_record(slug, timeouts=int(rec.get("timeouts") or 0) + 1)
                raise
        finally:
            shutil.rmtree(folder, ignore_errors=True)
        low = res.output.lower()
        if "being created" in low or "already" in low:
            self._save_blob_record(slug, uploaded_at=self.clock.now())
            return
        err = classify(self.name, "datasets create", res)
        if isinstance(err, (AuthRequired, RateLimited)):
            raise err
        raise Unavailable(
            f"kaggle did not take the dataset {ref} ({size / 1e6:.1f} MB): "
            f"{snippet(res.output) or 'no answer'}",
            provider=self.name,
        )

    def _wait_blob_ready(self, ref: str, deadline: float) -> None:
        while True:
            status = self._blob_status(ref, deadline)
            if status == "ready":
                return
            if status in ("failed", "error"):
                raise Unavailable(
                    f"kaggle could not process the dataset {ref} ({status})", provider=self.name
                )
            if self._left(deadline) < BLOB_READY_PAUSE_S + T_BLOB_STATUS:
                raise Unavailable(
                    f"kaggle is still processing the dataset {ref} ({status}); gpu-router "
                    "tries again shortly and reuses the upload",
                    provider=self.name,
                )
            self._sleep(BLOB_READY_PAUSE_S)

    def _maybe_sweep_blobs(self) -> None:
        """At most every BLOB_SWEEP_EVERY_S, a background thread deletes blob datasets
        unused past their keep time (bundles/checkpoints 3 days, data data_keep_days)."""
        if not BLOB_SWEEP or self._offline:
            return
        now = self.clock.now()
        with self._lock:
            if self._sweep_thread is not None and self._sweep_thread.is_alive():
                return
            if now - self._last_sweep < BLOB_SWEEP_EVERY_S:
                return
            self._last_sweep = now
            self._sweep_thread = threading.Thread(
                target=self.sweep_blobs, name=f"{self.name}-blob-sweep", daemon=True
            )
            self._sweep_thread.start()

    def sweep_blobs(self) -> list[str]:
        """Delete stale blob datasets (bounded); returns the refs deleted."""
        deleted: list[str] = []
        now = self.clock.now()
        try:
            records = sorted(self._dir("blobs").glob("gpu-router-*.json"))
        except OSError:
            return deleted
        for path in records:
            if len(deleted) >= BLOB_SWEEP_MAX:
                break
            slug = path.stem
            # under the blob's lock, re-read: a submit that just ensured it (and is about
            # to push a kernel reading it) refreshed last_used, so it is not stale any more
            with self._blob_lock(slug):
                rec = self._blob_record(slug)
                ref, kind = str(rec.get("ref") or ""), str(rec.get("kind") or "")
                used = max(
                    float(rec.get(k) or 0) for k in ("last_used", "uploaded_at", "uploading_at")
                )
                keep = self._blob_keep_s.get(kind, 3 * 86400)
                if not _BLOB_REF.match(ref) or now - used < keep:
                    continue
                try:
                    res = self._run(["datasets", "delete", ref, "-y"], timeout=T_DELETE)
                except AdapterError:
                    break
                low = res.output.lower()
                if res.ok or "404" in low or "not found" in low:
                    with contextlib.suppress(OSError):
                        path.unlink()
                    deleted.append(ref)
        return deleted

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
        if not res.ok and "400 client error" in low:
            # the server validated the request and refused it: nothing was saved. Seen live
            # for a run.py over ~1 MB (2026-10-04), which submit no longer sends.
            return InvalidJob(
                f"kaggle refused the kernel (400 Bad Request): {snippet(res.output)}",
                provider=self.name,
                hint="the job runs elsewhere; `gpu doctor` checks the kaggle CLI",
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
