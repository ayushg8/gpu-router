"""LightningAdapter: batch GPU jobs as Lightning AI Studio jobs (phase 7a).

One attempt = one Studio job named `gr-<job_id>-<n>` (the attempt key with `gpu` -> `gr`)
in the account's teamspace, running in the environment of a Studio (settings `studio`,
default `gpu-router`, created once if missing; creating a Studio does not start it). The
lightning-sdk runs in its own environment through `sdk.SdkBridge` (D3); facts, verified
SDK shapes and open questions are in NOTES.md.

Contract notes (CLAUDE.md "Adapter contract"):

- A4 / lookup_by_key: the job name is the attempt key. The driver's submit returns an
  existing job of that name instead of creating one (Lightning would silently rename a
  second job; if that ever happens the duplicate is stopped and deleted). A
  `runs/<name>.submitting` intent lives while a submit call runs: a daemon that died
  mid-submit leaves its driver running (own session), so for T_SUBMIT + grace lookup and
  submit answer Unavailable rather than "nothing exists" (D35 pattern).
- Upload: launch.py + launch.json (non-secret parameters, env), the bundle, resume.tar.gz
  (a checkpoint file on this Mac) and secrets.json (job secrets + the storage token) go to
  the teamspace drive folder `uploads/gpu-router/<name>/`; the staging copy of secrets is
  deleted as soon as the call returns, the launcher deletes the drive copy when it starts
  and fetch/cancel/the next submit sweep it again.
- status: Pending/NotCreated -> pending, Running -> running, Stopping -> running
  ("stopping"). Terminal jobs are judged once from their log (the runner's `::gpu:: exit`
  line) and cached redacted under final/: exit 0 succeeded, 90 lost (dependency install),
  else failed; the launcher's WALL_MARK = lost (session limit, the job migrates); Stopped
  after our cancel = cancelled; Stopped for credits = lost + quota_exhausted; an
  interrupted (spot) job = lost (preempted); Stopped by someone else = cancelled.
- Wall clock: Jobs have no time limit and max_runtime is not one (NOTES.md), so the
  launcher stops the runner after `session_s` (catalog session_hours minus a margin,
  recorded in RemoteRef.meta so the engine plans its handoff against it, D44). Backstop:
  status() of a job still running BACKSTOP_GRACE_S past that stops it and reports lost once
  the stop took effect (else running + `stop_pending`, retried by status, healthcheck and
  quota). Like
  Colab's stop-inside-status (D31) this is a deliberate exception to A6: a job past its
  limit burns the month's credits and nothing else would stop it.
- logs: the job log snapshot while it runs (cursor = lines served), the cached final log
  after. No live follow (the engine polls, D8).
- fetch: outputs.tar.gz the launcher delivered (drive `out/`, else the job's artifacts),
  extracted safely into a temp dir and copied into dest (never deletes in dest, A9).
- cancel: job.stop() (idempotent, bounded) + a local cancel mark so a Stopped job reads
  as cancelled; Unavailable when the stop is not confirmed and the job still runs (the
  engine retries before it migrates or gives up).
- quota: credits. Live balance when the billing API answers, else what jobs in the
  teamspace cost this calendar month (both from Lightning); resets at the next month.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import os
import re
import shlex
import shutil
import tarfile
import tempfile
import threading
from collections.abc import Iterator, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

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
from gpu_router.providers.lightning import credentials as creds_mod
from gpu_router.providers.lightning import launch
from gpu_router.providers.lightning.sdk import (
    DEFAULT_SDK_VERSION,
    CallResult,
    Runner,
    SdkBridge,
    SubprocessRunner,
    resolve_interpreter,
    snippet,
    to_error,
)
from gpu_router.secrets import redact

__all__ = ["DRIVE_ROOT", "LightningAdapter", "job_command", "name_for_key"]

DRIVE_ROOT = "uploads/gpu-router"
MOUNT_ROOT = "/teamspace"
DEFAULT_GPU = "T4"
#: catalog GPU name -> lightning_sdk.Machine attribute (single GPU; NOTES.md "Summary").
#: L4 (Machine.L4, slug lit-l4-1) stays mapped for plans that run it; the packaged catalog
#: offers T4 only because the free tier answers an L4 create with 403 (D56)
MACHINES: dict[str, str] = {"T4": "T4", "L4": "L4"}
#: machines `provider_options.lightning.machine` may name -> the catalog GPU they are
#: priced as. Exact names only: L40S is not an L4 and T4_X_4 is four T4s, at rates the
#: router and the approval policy do not know (review fix).
EXPLICIT_MACHINES: dict[str, str] = {"T4": "T4", "T4_SMALL": "T4", "L4": "L4"}
_MACHINE_RE = re.compile(r"^[A-Z][A-Z0-9_]{1,31}$")
INSTALL_FAILED_EXIT = 90  # runner/bootstrap.py: dependency install failed, job never ran
DEFAULT_SESSION_H = 4.0  # used when the catalog has no session_hours
SESSION_MARGIN_S = 300  # launcher wall clock = session cap minus this
MIN_WALL_S = 120
BACKSTOP_GRACE_S = 900.0  # launcher kill grace (90 s) + outputs upload (<= 300 s) + slack
EMPTY_LOG_GRACE_S = 600.0  # a finished job's empty log is "not published yet" this long
MIN_BALANCE = 0.05  # credits; below this a submit is QuotaExhausted before anything starts
DEFAULT_MAX_BUNDLE_MB = 200.0
DEFAULT_MAX_RESUME_MB = 200.0
#: assumed uplink (megabits/s) when sizing what one submit call can upload in time;
#: settings `upload_mbps` overrides it (review fix: the fixed T_SUBMIT does not grow
#: with the upload)
DEFAULT_UPLOAD_MBPS = 8.0
#: seconds of T_SUBMIT kept for the SDK start, the Studio lookup and Job.run
SUBMIT_OVERHEAD_S = 60.0
CHUNK_LINES = 1000
CRED_TTL_S = 300.0

# Per-call driver timeouts (seconds); each method's worst case stays below
# config.engine.timeouts.<call> (rule A2):
#   status   status 40 + backstop stop 15            = 55  (engine 60)
#   logs     logs 45                                 = 45  (engine 60)
#   lookup   status 40                               = 40
#   submit   whoami 40 + sweep 30 + submit 200       = 270 (engine 300)
#   fetch    status 55 + fetch 1500 + sweep 30       = 1585 (engine 1800)
#   cancel   stop 45                                 = 45  (engine 60)
#   quota    pending stop 15 + quota 42              = 57  (engine 60)
#   health   pending stop 15 + whoami 42             = 57  (engine 60)
T_STATUS = 40.0
T_BACKSTOP = 15.0
T_LOGS = 45.0
T_WHOAMI = 40.0
T_HEALTH = 42.0
T_CLEANUP = 30.0
T_SUBMIT = 200.0
T_FETCH = 1500.0
T_STOP = 45.0
T_QUOTA = 42.0
SUBMIT_ORPHAN_GRACE_S = 60.0
LOG_WAIT_S = 20.0  # the driver's own bound on one log read
VERDICT_TAIL = 500  # lines of a finished job's log read when the whole log is slow
TAIL_WAIT_S = 8.0  # status: whole log 20 + tail 8 stays inside T_STATUS
#: a stop that answers with one of these took effect (or the job is already over)
_STOP_TOOK = ("Stopping", "Stopped", "Completed", "Failed")

_JOB_OPTIONS = {"machine", "interruptible", "timeout_s"}
_KEY_RE = re.compile(r"^gpu-([0-9a-f]{6,32})-([1-9][0-9]{0,3})$")
_NAME_RE = re.compile(r"^gr-[0-9a-f]{6,32}-[0-9]{1,4}$")
_TEAMSPACE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,99}/[A-Za-z0-9][A-Za-z0-9_.-]{0,99}$")
_QUOTA_WORDS = ("credit", "balance", "insufficient", "quota", "out of funds")
_TERMINAL = ("Completed", "Failed", "Stopped")

OFFLINE_REASON = "real lightning calls are off in test mode"
OFFLINE_HINT = "set GPU_ROUTER_REAL_PROVIDERS=lightning to allow them"

#: runs inside the job when the drive mount does not have our folder: downloads the files
#: with the Studio's own SDK (`<py> -c <this> <owner/ts> <drive dir> <dest> <files...>`).
_FETCH_SNIPPET = """
import os, sys, threading
os.environ["LIGHTNING_DISABLE_VERSION_CHECK"] = "1"
from lightning_sdk import Teamspace
ts = Teamspace(sys.argv[1])
src, dest = sys.argv[2], sys.argv[3]
def get(name):
    try:
        ts.download_file(src + "/" + name, os.path.join(dest, name))
    except Exception as exc:
        print("gpu-router: could not download %s: %s: %s" % (name, type(exc).__name__, exc))
for name in sys.argv[4:]:
    t = threading.Thread(target=get, args=(name,), daemon=True)
    t.start()
    t.join(600)
"""


def name_for_key(attempt_key: str) -> str | None:
    """`gpu-<job_id>-<n>` -> `gr-<job_id>-<n>`; None for anything else."""
    m = _KEY_RE.match(attempt_key)
    return None if m is None else f"gr-{m.group(1)}-{m.group(2)}"


def job_command(name: str, teamspace: str, files: list[str]) -> str:
    """The job's shell command: run launch.py from the drive mount, else from a copy
    downloaded with the Studio's SDK."""
    drive = f"{DRIVE_ROOT}/{name}"
    code = base64.b64encode(_FETCH_SNIPPET.encode()).decode("ascii")
    loader = f"import base64;exec(base64.b64decode('{code}'))"
    q = shlex.quote
    return (
        f"D={q(f'{MOUNT_ROOT}/{drive}')}; PY=$(command -v python3 || command -v python); "
        f'if [ ! -f "$D/launch.py" ]; then D="${{TMPDIR:-/tmp}}/gpu-router-dl/{name}"; '
        f'mkdir -p "$D"; "$PY" -c {q(loader)} {q(teamspace)} {q(drive)} "$D" '
        f"{' '.join(q(f) for f in files)}; fi; "
        f'exec "$PY" -u "$D/launch.py"'
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


def _real_opt_in(name: str) -> bool:
    raw = os.environ.get("GPU_ROUTER_REAL_PROVIDERS", "")
    return name in {p.strip() for p in raw.split(",") if p.strip()}


def month_bounds(now: float) -> tuple[float, float]:
    """(start of this calendar month, start of the next one), UTC epoch seconds."""
    dt = datetime.fromtimestamp(now, tz=UTC)
    start = dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    nxt = (
        start.replace(year=start.year + 1, month=1)
        if start.month == 12
        else start.replace(month=start.month + 1)
    )
    return start.timestamp(), nxt.timestamp()


def normalize(line: str) -> str:
    """A protocol line with a platform prefix in front gets the prefix cut off."""
    at = line.find(protocol.PREFIX)
    return line[at:] if at > 0 else line


def last_exit_code(lines: list[str]) -> int | None:
    for line in reversed(lines):
        if not line.startswith(protocol.PREFIX):
            continue
        event = protocol.parse_line(line)
        if event is not None and event.t == "exit" and event.code is not None:
            return event.code
    return None


def _float(value: Any) -> float | None:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


class LightningAdapter(Adapter):
    kind = "lightning"
    capabilities = Capabilities(
        lookup_by_key=True,
        live_logs=False,
        resume=True,
        cancel_confirms=True,
        live_quota=True,
        max_session_hours=DEFAULT_SESSION_H,
        max_bundle_mb=DEFAULT_MAX_BUNDLE_MB,
        poll_interval_s=60,
    )

    def __init__(
        self,
        deps: AdapterDeps,
        *,
        runner: Runner | None = None,
        interpreter: list[str] | None = None,
        credential_file: Path | None = None,
        extra_env: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(deps)
        extra: dict[str, Any] = dict(deps.settings.model_extra or {})
        self._teamspace_cfg: str | None = extra.get("teamspace") or None
        self._studio = str(extra.get("studio") or "gpu-router")
        self._create_studio = bool(extra.get("create_studio", True))
        self._interruptible = bool(extra.get("interruptible", False))
        # `login_source`: config.yaml refuses a providers.<name>.credentials key as "looks
        # like a secret" (config._reject_secret_keys), so that older name only works when
        # settings are built in code
        self._cred_mode = str(extra.get("login_source") or extra.get("credentials") or "auto")
        mbps = float(extra.get("upload_mbps") or DEFAULT_UPLOAD_MBPS)
        #: what one submit call can upload within T_SUBMIT at the assumed uplink
        self._max_upload_mb = max(1.0, (T_SUBMIT - SUBMIT_OVERHEAD_S) * mbps / 8.0)
        self._max_bundle_mb = min(
            float(extra.get("max_bundle_mb") or DEFAULT_MAX_BUNDLE_MB), self._max_upload_mb
        )
        self._max_resume_mb = float(extra.get("max_resume_mb") or DEFAULT_MAX_RESUME_MB)
        self._credential_file = credential_file
        cap_h = deps.entry.session_hours or DEFAULT_SESSION_H
        self.capabilities = Capabilities(
            lookup_by_key=True,
            live_logs=False,
            resume=True,
            cancel_confirms=True,
            live_quota=True,
            max_session_hours=cap_h,
            max_concurrency=deps.entry.max_concurrency,
            max_bundle_mb=self._max_bundle_mb,
            poll_interval_s=deps.entry.poll_interval_s,
        )
        self._session_cap_s = cap_h * 3600.0
        python = interpreter
        if python is None:
            python = resolve_interpreter(
                python=extra.get("python") or None,
                sdk_version=str(extra.get("sdk_version") or DEFAULT_SDK_VERSION),
                uv=extra.get("uv_path") or None,
            )
        self._interpreter = python
        self._lock = threading.Lock()
        self._creds: tuple[float, creds_mod.Credentials | None] | None = None
        # Invariant 20: a test-mode daemon never calls the real SDK unless
        # GPU_ROUTER_REAL_PROVIDERS lists lightning; tests inject a runner.
        self._offline = runner is None and deps.test_mode and not _real_opt_in(self.name)
        self.bridge = SdkBridge(
            self.name,
            runner=runner or SubprocessRunner(),
            interpreter=lambda: self._interpreter,
            env_factory=self._cred_env,
            home=lambda: self.scratch_dir / "sdk-home",
            extra_env=extra_env,
        )
        with contextlib.suppress(OSError):
            self._sweep_staging()

    # ------------------------------------------------------------------ credentials

    def credentials(self) -> creds_mod.Credentials | None:
        """Resolved at most every CRED_TTL_S (each resolve reads the Keychain)."""
        now = self.clock.now()
        with self._lock:
            cached = self._creds
            if cached is not None and now - cached[0] < CRED_TTL_S:
                return cached[1]
        found = creds_mod.resolve(self._cred_mode, provider=self.name, file=self._credential_file)
        with self._lock:
            self._creds = (now, found)
        return found

    def _cred_env(self) -> dict[str, str]:
        found = self.credentials()
        return found.plain_env() if found is not None else {}

    def _require_creds(self) -> creds_mod.Credentials:
        found = self.credentials()
        if found is None:
            raise AuthRequired(
                "no lightning credentials (Keychain, environment or ~/.lightning)",
                provider=self.name,
                hint=creds_mod.LOGIN_HINT,
            )
        return found

    # ------------------------------------------------------------------ plumbing

    def _dir(self, *parts: str) -> Path:
        d = self.scratch_dir.joinpath(*parts)
        d.mkdir(mode=0o700, parents=True, exist_ok=True)
        return d

    def _call(self, op: str, params: Mapping[str, Any], *, timeout: float) -> dict[str, Any]:
        """One driver op; a classified failure raises its taxonomy class."""
        if self._offline:
            raise AuthRequired(OFFLINE_REASON, provider=self.name, hint=OFFLINE_HINT)
        self._require_creds()
        res = self.bridge.call(op, params, timeout=timeout)
        if res.ok:
            return res.result
        raise self._error(op, res)

    def _error(self, op: str, res: CallResult) -> AdapterError:
        resets = month_bounds(self.clock.now())[1] if res.kind == "quota" else None
        return to_error(self.name, op, res.kind, res.error or "", resets_at=resets)

    def _name(self, ref: RemoteRef) -> str:
        if not _NAME_RE.match(ref.remote_id):
            raise NotFound(f"lightning has no gpu-router job {ref.remote_id!r}", provider=self.name)
        return ref.remote_id

    def _record_path(self, name: str) -> Path:
        return self._dir("runs") / f"{name}.json"

    def _intent_path(self, name: str) -> Path:
        return self._dir("runs") / f"{name}.submitting"

    def _final_path(self, name: str) -> Path:
        return self._dir("final") / f"{name}.json"

    def _read_json(self, path: Path) -> dict[str, Any] | None:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return data if isinstance(data, dict) else None

    def _record(self, name: str) -> dict[str, Any]:
        return self._read_json(self._record_path(name)) or {}

    def _update_record(self, name: str, **fields: Any) -> dict[str, Any]:
        with self._lock:
            rec = self._record(name)
            rec.update({k: v for k, v in fields.items()})
            _atomic_write(self._record_path(name), json.dumps(rec, sort_keys=True))
        return rec

    def _teamspace_of(self, ref: RemoteRef, rec: Mapping[str, Any] | None = None) -> str | None:
        return (
            ref.meta.get("teamspace")
            or (rec or {}).get("teamspace")
            or self._teamspace_cfg
            or self._cached_account().get("teamspace")
        )

    # ------------------------------------------------------------------ account

    def _account_path(self) -> Path:
        return self._dir("account") / "account.json"

    def _cred_fingerprint(self) -> str:
        found = self.credentials()
        if found is None:
            return ""
        return hashlib.sha256(found.user_id.get_secret_value().encode()).hexdigest()[:16]

    def _cached_account(self) -> dict[str, Any]:
        """The last whoami for these credentials, however old: the teamspace rarely
        changes, and a call path that had to ask first (quota, lookup) could exceed its
        budget. healthcheck() refreshes it (and so does anything when it is missing)."""
        data = self._read_json(self._account_path()) or {}
        if data.get("cred") != self._cred_fingerprint():
            return {}
        return data

    def whoami(self, *, timeout: float = T_WHOAMI) -> dict[str, Any]:
        params = {"teamspace": self._teamspace_cfg} if self._teamspace_cfg else {}
        res = self._call("whoami", params, timeout=timeout)
        record = {
            "user": res.get("user"),
            "teamspace": res.get("teamspace"),
            "teamspaces": res.get("teamspaces"),
            "sdk_version": res.get("sdk_version"),
            "cred": self._cred_fingerprint(),
            "at": self.clock.now(),
        }
        _atomic_write(self._account_path(), json.dumps(record, sort_keys=True))
        return record

    def teamspace(self) -> str:
        """`owner/name` the jobs run in: settings, else the account's only teamspace."""
        chosen: Any
        if self._teamspace_cfg:
            chosen = self._teamspace_cfg
        else:
            chosen = self._cached_account().get("teamspace") or self.whoami().get("teamspace")
        if not chosen:
            raise AuthRequired(
                "cannot pick a lightning teamspace automatically",
                provider=self.name,
                hint="set providers.lightning.teamspace to owner/name in config.yaml",
            )
        if not _TEAMSPACE_RE.match(str(chosen)):
            raise AuthRequired(
                f"lightning teamspace {chosen!r} is not owner/name",
                provider=self.name,
                hint="set providers.lightning.teamspace to owner/name in config.yaml",
            )
        return str(chosen)

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

    def _machine(self, gpu: str | None, opts: Mapping[str, Any]) -> tuple[str, str]:
        """(sdk Machine name, catalog GPU label). An explicit machine must be one the
        catalog prices (EXPLICIT_MACHINES) and the GPU the router placed the job on:
        otherwise the approval policy and the quota handoff would charge the wrong rate."""
        choice = str(gpu or DEFAULT_GPU).upper()
        explicit = opts.get("machine")
        if explicit:
            name = str(explicit).upper()
            if not _MACHINE_RE.match(name):
                raise InvalidJob(f"bad lightning machine name {explicit!r}", provider=self.name)
            label = EXPLICIT_MACHINES.get(name)
            if label is None:
                raise InvalidJob(
                    f"provider_options.{self.name}.machine {name} is not a machine gpu-router "
                    f"can price (it knows {', '.join(EXPLICIT_MACHINES)})",
                    provider=self.name,
                    hint="pick the GPU with `gpu:` in gpu.yaml or --gpu instead",
                )
            if label != choice:
                raise InvalidJob(
                    f"provider_options.{self.name}.machine {name} is a {label}, but the job "
                    f"was placed on a {choice}",
                    provider=self.name,
                    hint=f"set gpu: {label} in gpu.yaml so the router prices the {label}",
                )
            return name, label
        if choice in MACHINES:
            return MACHINES[choice], choice
        raise InvalidJob(
            f"lightning has no {choice} GPUs here (it offers {', '.join(MACHINES)})",
            provider=self.name,
        )

    def _wall_clock_s(self, opts: Mapping[str, Any]) -> int:
        limit = max(MIN_WALL_S, int(self._session_cap_s) - SESSION_MARGIN_S)
        raw = opts.get("timeout_s")
        if raw is None:
            return limit
        try:
            value = int(raw)
        except (TypeError, ValueError):
            raise InvalidJob(
                f"provider_options.{self.name}.timeout_s must be seconds", provider=self.name
            ) from None
        return max(MIN_WALL_S, min(value, limit))

    def _resume_file(self, ctx: AttemptContext) -> tuple[Path | None, str | None]:
        """(checkpoint archive on this Mac to upload, note). hf:// resumes are downloaded
        by the runner itself (phase 5 storage)."""
        ckpt = ctx.resume_from
        if ckpt is None or str(ctx.env.get("GPU_RESUME_URI") or "").startswith("hf://"):
            return None, None
        parsed = urlparse(ckpt.uri)
        path = Path(unquote(parsed.path)) if parsed.scheme == "file" else None
        limit = self._max_resume_mb * 1024 * 1024
        if path is not None and path.is_file() and path.stat().st_size <= limit:
            return path, None
        return None, f"checkpoint {ckpt.seq} ({ckpt.uri}) cannot reach lightning; starting fresh"

    def _intent_live(self, name: str) -> bool:
        data = self._read_json(self._intent_path(name))
        if data is None:
            return False
        started = _float(data.get("started_at"))
        if started is not None and self.clock.now() - started < T_SUBMIT + SUBMIT_ORPHAN_GRACE_S:
            return True
        with contextlib.suppress(OSError):
            self._intent_path(name).unlink()
        return False

    def _in_flight(self, name: str) -> Unavailable:
        return Unavailable(
            f"an interrupted submit of {name} may still be running; gpu-router checks again "
            "shortly",
            provider=self.name,
        )

    def _ref(self, name: str, rec: Mapping[str, Any]) -> RemoteRef:
        meta = {
            k: str(rec[k])
            for k in ("teamspace", "studio", "machine", "gpu", "attempt_key", "session_s")
            if rec.get(k) is not None
        }
        meta["drive_dir"] = f"{DRIVE_ROOT}/{name}"
        return RemoteRef(remote_id=name, url=rec.get("link") or None, meta=meta)

    def submit(self, job: Job, ctx: AttemptContext) -> RemoteRef:
        name = name_for_key(ctx.attempt_key)
        if name is None:
            raise InvalidJob(
                f"attempt key {ctx.attempt_key!r} is not a gpu-router key", provider=self.name
            )
        opts = self._job_options(job)
        machine, gpu_label = self._machine(ctx.gpu or job.spec.gpu, opts)
        archive = ctx.bundle_archive
        if archive is None or not archive.is_file():
            raise InvalidJob("the job has no bundle to send to lightning", provider=self.name)
        size = archive.stat().st_size
        if size > self._max_bundle_mb * 1024 * 1024:
            raise InvalidJob(
                f"bundle is {size / 1e6:.1f} MB; lightning takes up to {self._max_bundle_mb:g} MB",
                provider=self.name,
                hint="ship data through HF Hub instead of the project folder",
            )
        if self._offline:
            raise AuthRequired(OFFLINE_REASON, provider=self.name, hint=OFFLINE_HINT)
        self._require_creds()
        rec = self._record(name)
        if rec.get("submitted"):
            return self._ref(name, rec)  # A4: this key already has its job
        if self._intent_live(name):
            raise self._in_flight(name)
        wall_s = self._wall_clock_s(opts)
        interruptible = bool(opts.get("interruptible", self._interruptible))
        resume, resume_note = self._resume_file(ctx)
        self._check_upload_fits(size, resume)
        ts = self.teamspace()
        self._sweep_secrets(ts)
        with contextlib.suppress(OSError):
            self._sweep_staging()
        seq_start = ctx.resume_from.seq + 1 if ctx.resume_from is not None else 1
        # the intent exists before the staging dir, so a concurrent sweep never takes it
        _atomic_write(self._intent_path(name), json.dumps({"started_at": self.clock.now()}))
        stage = Path(tempfile.mkdtemp(prefix=f"{name}-", dir=self._dir("staging")))
        try:
            files = self._stage(
                stage,
                name=name,
                ctx=ctx,
                teamspace=ts,
                archive=archive,
                resume=resume,
                resume_note=resume_note,
                seq_start=seq_start,
                wall_s=wall_s,
            )
            params = {
                "teamspace": ts,
                "studio": self._studio,
                "create_studio": self._create_studio,
                "name": name,
                "machine": machine,
                "interruptible": interruptible,
                "command": job_command(name, ts, [Path(f["remote"]).name for f in files]),
                "env": {"GPU_ROUTER_ATTEMPT_KEY": ctx.attempt_key},
                "files": files,
                "min_balance": MIN_BALANCE,
            }
            self._update_record(
                name,
                attempt_key=ctx.attempt_key,
                teamspace=ts,
                studio=self._studio,
                machine=machine,
                gpu=gpu_label,
                session_s=wall_s,
                secrets_remote=bool(ctx.secrets),
                created_at=self.clock.now(),
            )
            _atomic_write(self._intent_path(name), json.dumps({"started_at": self.clock.now()}))
            res = self.bridge.call("submit", params, timeout=T_SUBMIT)
        finally:
            shutil.rmtree(stage, ignore_errors=True)  # plaintext secrets: gone at once
            with contextlib.suppress(OSError):
                self._intent_path(name).unlink()
        if res.ok:
            out = res.result
            rec = self._update_record(
                name,
                submitted=True,
                link=out.get("link"),
                job_id=out.get("id"),
                teamspace=out.get("teamspace") or ts,
                submitted_at=self.clock.now(),
            )
            return self._ref(name, rec)
        refused = self._machine_refused(res, gpu_label)
        if refused is not None:
            raise refused
        raise self._submit_error(res)

    def _machine_refused(self, res: CallResult, gpu_label: str) -> InvalidJob | None:
        """A 403 to the create call itself for a machine other than the free T4 (and a
        lookup found no job) is the plan refusing that GPU, not a login problem: live
        2026-09-25 the free account got 403 for L4 while T4 creates worked (D56). An
        InvalidJob keeps Lightning usable for other jobs and says what happened; the
        engine then places the job elsewhere (or explains why nothing else fits)."""
        if (
            gpu_label == DEFAULT_GPU
            or res.stage != "run"
            or res.status != 403
            or res.kind not in ("auth", "verify")
        ):
            return None
        return InvalidJob(
            f"lightning refused to create a {gpu_label} job (403 Forbidden): this account's "
            f"plan does not include {gpu_label} machines (the free tier runs T4 only)",
            provider=self.name,
            hint="ask for a T4 (--gpu T4) or leave --gpu out; lightning stays on for T4 jobs",
        )

    def _check_upload_fits(self, bundle_bytes: int, resume: Path | None) -> None:
        """One submit call uploads everything within T_SUBMIT: refuse (InvalidJob, so the
        engine places the job elsewhere) what cannot be uploaded in time at the assumed
        uplink, instead of timing out on every placement."""
        total = bundle_bytes + (resume.stat().st_size if resume is not None else 0)
        if total <= self._max_upload_mb * 1024 * 1024:
            return
        what = "bundle + checkpoint" if resume is not None else "bundle"
        raise InvalidJob(
            f"{what} is {total / 1e6:.0f} MB; one lightning submit can upload about "
            f"{self._max_upload_mb:.0f} MB in its {T_SUBMIT:.0f}s budget",
            provider=self.name,
            hint="set providers.lightning.upload_mbps to your real uplink if it is faster, "
            "or ship data through HF Hub",
        )

    def _sweep_staging(self) -> None:
        """Remove staging dirs (plaintext secrets.json) left by a daemon that died during a
        submit. A dir whose job name has a live `.submitting` intent is kept: an orphaned
        driver may still be uploading from it."""
        root = self.scratch_dir / "staging"
        if not root.is_dir():
            return
        for entry in root.iterdir():
            m = re.match(r"^(gr-[0-9a-f]{6,32}-[0-9]{1,4})-", entry.name)
            if m is not None and self._intent_live(m.group(1)):
                continue
            if entry.is_dir() and not entry.is_symlink():
                shutil.rmtree(entry, ignore_errors=True)
            else:
                with contextlib.suppress(OSError):
                    entry.unlink()

    def _submit_error(self, res: CallResult) -> AdapterError:
        """Definitive only when certain nothing was created (A3): failures before the
        create call ("pre"), or a 4xx answer to it that a lookup confirmed ("run").
        "post" and a missing stage may have created the job (driver.op_submit)."""
        err = self._error("submit", res)
        definitive = (RateLimited, QuotaExhausted, AuthRequired, InvalidJob, Permanent)
        created_maybe = res.stage != "pre" and not (
            res.stage == "run" and res.status is not None and 400 <= res.status < 500
        )
        if isinstance(err, definitive) and not created_maybe:
            return err
        if isinstance(err, NotFound) and res.stage == "pre":
            return InvalidJob(err.message, provider=self.name)
        return Unavailable(
            f"lightning submit outcome unknown: {err.message}", provider=self.name, hint=err.hint
        )

    def _stage(
        self,
        stage: Path,
        *,
        name: str,
        ctx: AttemptContext,
        teamspace: str,
        archive: Path,
        resume: Path | None,
        resume_note: str | None,
        seq_start: int,
        wall_s: int,
    ) -> list[dict[str, str]]:
        """Write launch.py/launch.json (+ secrets.json, 0600) into `stage`; return the
        upload list (local path -> drive path)."""
        drive = f"{DRIVE_ROOT}/{name}"
        cfg: dict[str, Any] = {
            "name": name,
            "attempt_key": ctx.attempt_key,
            "teamspace": teamspace,
            "drive_dir": drive,
            "env": dict(ctx.env),
            "ckpt_seq_start": seq_start,
            "checkpoint_interval_min": ctx.checkpoint_interval_min,
            "wall_clock_s": wall_s,
            "bundle_sha256": _sha256(archive),
            "resume_sha256": _sha256(resume) if resume is not None else None,
            "resume_note": resume_note,
            "secrets": bool(ctx.secrets),
        }
        _atomic_write(stage / "launch.json", json.dumps(cfg, indent=1, sort_keys=True))
        shutil.copyfile(Path(launch.__file__), stage / "launch.py")
        files = [
            {"local": str(stage / "launch.json"), "remote": f"{drive}/launch.json"},
            {"local": str(stage / "launch.py"), "remote": f"{drive}/launch.py"},
            {"local": str(archive), "remote": f"{drive}/bundle.tar.gz"},
        ]
        if resume is not None:
            files.append({"local": str(resume), "remote": f"{drive}/resume.tar.gz"})
        if ctx.secrets:
            payload = json.dumps(
                {"v": 1, "values": {k: v.get_secret_value() for k, v in ctx.secrets.items()}}
            )
            path = stage / "secrets.json"
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(payload)
            files.append({"local": str(path), "remote": f"{drive}/secrets.json"})
        return files

    def _sweep_secrets(self, teamspace: str) -> None:
        """Forget uploaded secrets files of attempts that are over (best effort)."""
        paths: list[str] = []
        names: list[str] = []
        for path in sorted(self._dir("runs").glob("gr-*.json")):
            rec = self._read_json(path) or {}
            name = path.stem
            if not rec.get("secrets_remote") or rec.get("secrets_cleaned"):
                continue
            if rec.get("teamspace") not in (None, teamspace):
                continue
            over = self._final_path(name).exists() or rec.get("cancel_requested_at")
            old = self.clock.now() - float(rec.get("created_at") or 0) > 13 * 3600
            if over or old:
                paths.append(f"{DRIVE_ROOT}/{name}/secrets.json")
                names.append(name)
        if not paths:
            return
        try:
            res = self._call("cleanup", {"teamspace": teamspace, "paths": paths}, timeout=T_CLEANUP)
        except AdapterError:
            return
        removed = set(res.get("removed") or [])
        for name in names:
            if f"{DRIVE_ROOT}/{name}/secrets.json" in removed:
                self._update_record(name, secrets_cleaned=True)

    def lookup_by_key(self, attempt_key: str) -> RemoteRef | None:
        name = name_for_key(attempt_key)
        if name is None:
            return None
        rec = self._record(name)
        if rec.get("submitted"):
            return self._ref(name, rec)
        if self._offline:
            return None  # every submit is refused while offline, so nothing can exist
        if self._intent_live(name):
            raise self._in_flight(name)
        ts = rec.get("teamspace") or self.teamspace()
        try:
            found = self._call("status", {"teamspace": ts, "name": name}, timeout=T_STATUS)
        except NotFound:
            return None
        rec = self._update_record(name, submitted=True, teamspace=ts, job_id=found.get("id"))
        return self._ref(name, rec)

    # ------------------------------------------------------------------ status

    def _status_from_final(self, ref: RemoteRef, record: Mapping[str, Any]) -> RemoteStatus:
        phase = RemotePhase(str(record.get("phase")))
        code = record.get("exit_code")
        message = {
            RemotePhase.SUCCEEDED: "finished",
            RemotePhase.FAILED: f"script exited with code {code}",
            RemotePhase.CANCELLED: str(record.get("message") or "cancelled"),
            RemotePhase.LOST: f"session ended: {record.get('lost_reason')}",
        }.get(phase, str(phase))
        return RemoteStatus(
            phase=phase,
            message=message,
            exit_code=int(code) if isinstance(code, int) else None,
            lost_reason=record.get("lost_reason") if phase is RemotePhase.LOST else None,
            quota_exhausted=bool(record.get("quota_exhausted")) and phase is RemotePhase.LOST,
            gpu=ref.meta.get("gpu"),
            started_at=_float(record.get("started_at")),
            ended_at=_float(record.get("ended_at")),
            url=ref.url,
        )

    def _empty_log_waits(self, name: str) -> bool:
        path = self._dir("final") / f"{name}.empty-log"
        now = self.clock.now()
        try:
            first = float(path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            first = now
            _atomic_write(path, f"{now:.3f}\n")
        return now - first < EMPTY_LOG_GRACE_S

    def _judge(
        self, ref: RemoteRef, rec: Mapping[str, Any], res: Mapping[str, Any], lines: list[str]
    ) -> dict[str, Any]:
        status = str(res.get("status"))
        code = last_exit_code(lines)
        text = " ".join(str(res.get(k) or "") for k in ("message", "server_error")).strip()
        low = text.lower()
        hours = float(ref.meta.get("session_s") or rec.get("session_s") or 0) / 3600
        if any(launch.WALL_MARK in line for line in lines) or rec.get("wall_stop_at"):
            return {
                "phase": str(RemotePhase.LOST),
                "exit_code": code,
                "lost_reason": f"lightning wall-clock limit ({hours:.1f}h) reached",
            }
        if status == "Stopped":
            if rec.get("cancel_requested_at"):
                return {"phase": str(RemotePhase.CANCELLED), "message": "cancelled"}
            if any(w in low for w in _QUOTA_WORDS):
                return {
                    "phase": str(RemotePhase.LOST),
                    "lost_reason": f"lightning credits ran out: {snippet(text, 120)}",
                    "quota_exhausted": True,
                }
            if res.get("interrupted"):
                return {"phase": str(RemotePhase.LOST), "lost_reason": "preempted (interruptible)"}
            if code is None:
                return {
                    "phase": str(RemotePhase.CANCELLED),
                    "message": "stopped outside gpu-router",
                }
        if code == 0 or (code is None and status == "Completed"):
            return {"phase": str(RemotePhase.SUCCEEDED), "exit_code": 0}
        if code == INSTALL_FAILED_EXIT:
            return {
                "phase": str(RemotePhase.LOST),
                "exit_code": code,
                "lost_reason": "dependency install failed on lightning",
            }
        if code is not None:
            return {"phase": str(RemotePhase.FAILED), "exit_code": code}
        if any(w in low for w in _QUOTA_WORDS):
            return {
                "phase": str(RemotePhase.LOST),
                "lost_reason": f"lightning credits ran out: {snippet(text, 120)}",
                "quota_exhausted": True,
            }
        reason = (
            f"lightning job failed before the runner finished: {snippet(text, 160)}"
            if text
            else "lightning job ended before the runner reported an exit code"
        )
        return {"phase": str(RemotePhase.LOST), "lost_reason": reason}

    def _finalize(
        self, ref: RemoteRef, name: str, rec: Mapping[str, Any], res: Mapping[str, Any]
    ) -> dict[str, Any]:
        log = res.get("log") or {}
        lines = [normalize(str(x)) for x in (log.get("lines") or [])]
        status = str(res.get("status"))
        partial = bool(log.get("timeout"))
        if partial and not lines:
            # a slow log read is not an empty log: no verdict, and the empty-log clock
            # does not start (review fix)
            raise Unavailable(
                f"{name} finished but reading its log from lightning timed out; "
                "gpu-router checks again",
                provider=self.name,
            )
        waits_for_log = status in ("Completed", "Failed") and not rec.get("cancel_requested_at")
        if not lines and waits_for_log and self._empty_log_waits(name):
            # judging now would cache a verdict with no exit line (a failed script would
            # read as lost and run again elsewhere)
            raise Unavailable(
                f"{name} finished but lightning has not published its log yet",
                provider=self.name,
            )
        record = self._judge(ref, rec, res, lines)
        record.update(
            lightning_status=status,
            started_at=_float(res.get("started_at")),
            ended_at=_float(res.get("stopped_at")),
            total_cost=_float(res.get("total_cost")),
            lines=[redact(line) for line in lines],
        )
        if partial:
            # judged from the log's tail; logs() serves nothing from it (its line numbers
            # are not the whole log's) and upgrades the cache once a whole read works
            record["lines_partial"] = True
        _atomic_write(self._final_path(name), json.dumps(record))
        with contextlib.suppress(OSError):
            (self._dir("final") / f"{name}.empty-log").unlink()
        return record

    def _observe(self, ref: RemoteRef, op: str) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """(driver result, final record if the job is over). Raises NotFound when
        Lightning has no such job and we never cancelled it."""
        name = self._name(ref)
        rec = self._record(name)
        ts = self._teamspace_of(ref, rec)
        if not ts:
            ts = self.teamspace()
        params: dict[str, Any] = {"teamspace": ts, "name": name, "log_wait_s": LOG_WAIT_S}
        if op == "status":
            params.update(with_log=True, log_tail=VERDICT_TAIL, tail_wait_s=TAIL_WAIT_S)
        try:
            res = self._call(op, params, timeout=T_STATUS if op == "status" else T_LOGS)
        except NotFound:
            if rec.get("cancel_requested_at"):
                record = {"phase": str(RemotePhase.CANCELLED), "message": "cancelled", "lines": []}
                _atomic_write(self._final_path(name), json.dumps(record))
                return {"status": "Stopped"}, record
            raise
        if str(res.get("status")) in _TERMINAL:
            return res, self._finalize(ref, name, rec, res)
        return res, None

    def status(self, ref: RemoteRef) -> RemoteStatus:
        name = self._name(ref)
        final = self._read_json(self._final_path(name))
        if final is not None:
            return self._status_from_final(ref, final)
        res, final = self._observe(ref, "status")
        if final is not None:
            return self._status_from_final(ref, final)
        status = str(res.get("status"))
        gpu = ref.meta.get("gpu")
        started = _float(res.get("started_at"))
        if status == "Running":
            over = self._backstop(ref, name, started)
            if over is not None:
                return over
            return RemoteStatus(
                phase=RemotePhase.RUNNING,
                message=f"running on lightning {gpu or ''}".rstrip(),
                gpu=gpu,
                started_at=started,
                url=ref.url,
            )
        if status == "Stopping":
            return RemoteStatus(
                phase=RemotePhase.RUNNING,
                message="lightning is stopping the job",
                gpu=gpu,
                started_at=started,
                url=ref.url,
            )
        return RemoteStatus(
            phase=RemotePhase.PENDING,
            message=f"queued on lightning, waiting for a {gpu or 'GPU'}",
            gpu=gpu,
            url=ref.url,
        )

    def _backstop(self, ref: RemoteRef, name: str, started: float | None) -> RemoteStatus | None:
        """Stop a job running BACKSTOP_GRACE_S past its wall clock (the launcher should
        have ended it; see the module docstring for why status() does this). LOST only
        once the stop took effect: until then the job reads as running (so the engine
        keeps polling and this retries) and it stays on the pending-stop list that
        healthcheck() and quota() retry too (review fix)."""
        session_s = _to_float(ref.meta.get("session_s"))
        if started is None or session_s is None:
            return None
        over = self.clock.now() - started - session_s
        if over < BACKSTOP_GRACE_S:
            return None
        rec = self._record(name)
        if not rec.get("wall_stop_at"):
            rec = self._update_record(name, wall_stop_at=self.clock.now())
        ts = self._teamspace_of(ref, rec) or self.teamspace()
        if not self._stop_now(name, ts, wait_s=5, timeout=T_BACKSTOP):
            return RemoteStatus(
                phase=RemotePhase.RUNNING,
                message="ran past its wall-clock limit; gpu-router is stopping it",
                gpu=ref.meta.get("gpu"),
                started_at=started,
                url=ref.url,
            )
        return RemoteStatus(
            phase=RemotePhase.LOST,
            message="session ended: ran past its wall-clock limit; gpu-router stopped it",
            lost_reason="ran past its lightning wall-clock limit; stopped by gpu-router",
            gpu=ref.meta.get("gpu"),
            started_at=started,
            url=ref.url,
        )

    @staticmethod
    def _stop_took(res: Mapping[str, Any]) -> bool:
        return bool(
            res.get("missing")
            or res.get("already")
            or res.get("confirmed")
            or str(res.get("status")) in _STOP_TOOK
        )

    def _stop_now(self, name: str, teamspace: str, *, wait_s: float, timeout: float) -> bool:
        """One backstop stop; True when it took effect. Otherwise the job is recorded as
        `stop_pending` so a later call retries it."""
        try:
            res = self._call(
                "stop", {"teamspace": teamspace, "name": name, "wait_s": wait_s}, timeout=timeout
            )
        except NotFound:
            res = {"missing": True}
        except AdapterError:
            res = {}
        took = self._stop_took(res)
        self._update_record(name, stop_pending=not took, teamspace=teamspace)
        return took

    def _retry_pending_stops(self) -> None:
        """At most one backstop stop that did not take effect yet (bounded, best effort):
        the engine may have stopped polling that job, and nothing else would end it."""
        try:
            records = sorted(self._dir("runs").glob("gr-*.json"))
        except OSError:
            return
        for path in records:
            rec = self._read_json(path) or {}
            if not rec.get("stop_pending") or self._final_path(path.stem).exists():
                continue
            ts = rec.get("teamspace") or self._teamspace_cfg
            if ts:
                self._stop_now(path.stem, str(ts), wait_s=5, timeout=T_BACKSTOP)
            return

    # ------------------------------------------------------------------ logs

    def logs(
        self, ref: RemoteRef, *, follow: bool = False, since: str | None = None
    ) -> Iterator[LogChunk]:
        name = self._name(ref)
        try:
            start = max(0, int(since)) if since else 0
        except ValueError:
            start = 0
        final = self._read_json(self._final_path(name))
        if final is None:
            res, final = self._observe(ref, "logs")
            if final is None:
                log = res.get("log") or {}
                lines = [normalize(str(x)) for x in (log.get("lines") or [])]
                if start >= len(lines):
                    yield LogChunk(lines=[], cursor=str(start), eof=False)
                    return
                for lo in range(start, len(lines), CHUNK_LINES):
                    hi = min(lo + CHUNK_LINES, len(lines))
                    yield LogChunk(lines=lines[lo:hi], cursor=str(hi), eof=False)
                return
        if final.get("lines_partial"):
            final = self._upgrade_final_log(ref, name, final)
            if final.get("lines_partial"):
                # only the tail is known: its line numbers are not the whole log's
                yield LogChunk(lines=[], cursor=str(start), eof=False)
                return
        lines = [str(x) for x in final.get("lines") or []]
        if start >= len(lines):
            yield LogChunk(lines=[], cursor=str(max(start, len(lines))), eof=True)
            return
        for lo in range(start, len(lines), CHUNK_LINES):
            hi = min(lo + CHUNK_LINES, len(lines))
            yield LogChunk(lines=lines[lo:hi], cursor=str(hi), eof=hi == len(lines))

    def _upgrade_final_log(
        self, ref: RemoteRef, name: str, final: dict[str, Any]
    ) -> dict[str, Any]:
        """A verdict judged from the log's tail: try one whole read and cache it."""
        rec = self._record(name)
        ts = self._teamspace_of(ref, rec) or self.teamspace()
        try:
            res = self._call(
                "logs", {"teamspace": ts, "name": name, "log_wait_s": LOG_WAIT_S}, timeout=T_LOGS
            )
        except AdapterError:
            return final
        log = res.get("log") or {}
        if log.get("timeout"):
            return final
        lines = [redact(normalize(str(x))) for x in (log.get("lines") or [])]
        upgraded = {**final, "lines": lines}
        upgraded.pop("lines_partial", None)
        _atomic_write(self._final_path(name), json.dumps(upgraded))
        return upgraded

    # ------------------------------------------------------------------ fetch

    def fetch(self, ref: RemoteRef, dest: Path) -> FetchResult:
        name = self._name(ref)
        st = self.status(ref)
        if not st.phase.terminal:
            raise NotFound(
                f"{name} is still {st.phase}; outputs appear when it finishes", provider=self.name
            )
        rec = self._record(name)
        ts = self._teamspace_of(ref, rec) or self.teamspace()
        tmp = Path(tempfile.mkdtemp(prefix=f"{name}-", dir=self._dir("fetch")))
        try:
            res = self._call(
                "fetch",
                {
                    "teamspace": ts,
                    "name": name,
                    "drive_dir": f"{DRIVE_ROOT}/{name}",
                    "dest": str(tmp / "dl"),
                },
                timeout=T_FETCH,
            )
            archive = Path(str(res.get("archive"))) if res.get("archive") else None
            if archive is None or not archive.is_file():
                notes = "; ".join(str(n) for n in res.get("notes") or []) or "no archive"
                if st.phase is RemotePhase.SUCCEEDED:
                    return FetchResult(
                        dest=dest,
                        files=0,
                        bytes=0,
                        partial=True,
                        message=f"lightning kept no outputs archive for {name} ({notes})",
                    )
                raise NotFound(f"{name} left no outputs (run {st.phase})", provider=self.name)
            files, total = self._extract(archive, tmp / "x", dest)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        self._sweep_secrets(ts)
        if files == 0 and st.phase is not RemotePhase.SUCCEEDED:
            raise NotFound(f"{name} left no outputs (run {st.phase})", provider=self.name)
        return FetchResult(dest=dest, files=files, bytes=total)

    def _extract(self, archive: Path, work: Path, dest: Path) -> tuple[int, int]:
        """Safe extract (regular files only, no absolute or parent paths), then copy into
        dest without deleting anything there (A9)."""
        work.mkdir(parents=True, exist_ok=True)
        dest.mkdir(parents=True, exist_ok=True)
        root = dest.resolve()
        files = total = 0
        try:
            with tarfile.open(archive, "r:gz") as tar:
                for member in tar.getmembers():
                    rel = Path(member.name)
                    if not member.isfile() or rel.is_absolute() or ".." in rel.parts:
                        continue
                    src = tar.extractfile(member)
                    if src is None:
                        continue
                    target = (dest / rel).resolve()
                    if root not in target.parents:
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    tmp = target.with_name(f".{target.name}.gpu-fetch")
                    with src, open(tmp, "wb") as out:
                        shutil.copyfileobj(src, out)
                    os.replace(tmp, target)
                    files += 1
                    total += target.stat().st_size
        except (tarfile.TarError, EOFError, OSError) as exc:
            raise Unavailable(
                f"the outputs archive from lightning is damaged: {exc}", provider=self.name
            ) from None
        return files, total

    # ------------------------------------------------------------------ cancel

    def cancel(self, ref: RemoteRef) -> None:
        try:
            name = self._name(ref)
        except NotFound:
            return
        if self._final_path(name).exists():
            return
        if self._offline:
            return
        rec = self._update_record(name, cancel_requested_at=self.clock.now())
        ts = self._teamspace_of(ref, rec) or self.teamspace()
        try:
            res = self._call("stop", {"teamspace": ts, "name": name, "wait_s": 20}, timeout=T_STOP)
        except NotFound:
            return
        if not self._stop_took(res):
            # the stop did not finish and the job still runs (the driver's stop thread may
            # never have reached Lightning): the engine retries before it migrates
            raise Unavailable(
                f"lightning has not confirmed the stop of {name} yet (it is "
                f"{res.get('status') or 'still running'}); gpu-router tries again",
                provider=self.name,
            )

    # ------------------------------------------------------------------ quota / health

    def quota(self) -> QuotaSnapshot:
        now = self.clock.now()
        start, resets = month_bounds(now)
        ts = self.teamspace()
        self._retry_pending_stops()
        res = self._call(
            "quota",
            {"teamspace": ts, "since": start, "machines": sorted(set(MACHINES.values()))},
            timeout=T_QUOTA,
        )
        catalog_limit = self.entry.quota.limit
        balance = _float(res.get("balance"))
        jobs_cost = _float(res.get("jobs_cost"))
        detail: dict[str, Any] = {
            "rates": res.get("rates") or {},
            "jobs_cost_month": jobs_cost,
            "jobs_counted": res.get("jobs_counted"),
            "teamspace": ts,
            "reset_basis": "calendar month (UTC) [unverified]",
        }
        if balance is not None:
            # Lightning reports what is left, not the month's grant (a new account had 5.0,
            # not the 15 the catalog quotes [3P], 2026-09-25), so the limit is derived:
            # left + this month's job costs. Never floored at the catalog number, which
            # would show a 5-credit account as "10 of 15 used".
            used = max(0.0, jobs_cost or 0.0)
            limit = balance + used
            detail.update(
                basis="balance",
                balance=balance,
                remaining=balance,
                total_spent=_float(res.get("total_spent")),
                catalog_limit=catalog_limit,
                note="limit = credits left + this month's job costs (Studio time not included)",
            )
            return QuotaSnapshot(
                provider=self.name,
                used=used,
                limit=limit,
                unit=QuotaUnit.CREDITS,
                resets_at=resets,
                source="live",
                detail=detail,
                observed_at=now,
            )
        if jobs_cost is None:
            raise Unavailable(
                f"lightning reported neither a balance nor job costs "
                f"({snippet(str(res.get('balance_error') or ''))})",
                provider=self.name,
            )
        detail.update(
            basis="job costs",
            balance_error=snippet(str(res.get("balance_error") or "")),
            note="credits spent on jobs this month per lightning (Studio time not included)",
        )
        return QuotaSnapshot(
            provider=self.name,
            used=jobs_cost,
            limit=float(catalog_limit) if catalog_limit is not None else None,
            unit=QuotaUnit.CREDITS,
            resets_at=resets,
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
            found = self.credentials()
        except AuthRequired as exc:
            return Health(
                health=ProviderHealth.AUTH_REQUIRED,
                reason=exc.message,
                hint=exc.hint,
                checked_at=now,
            )
        if found is None:
            return Health(
                health=ProviderHealth.AUTH_REQUIRED,
                reason="lightning is not logged in (run `gpu login lightning`)",
                hint=creds_mod.LOGIN_HINT,
                checked_at=now,
            )
        self._retry_pending_stops()
        try:
            who = self.whoami(timeout=T_HEALTH)
        except AuthRequired as exc:
            return Health(
                health=ProviderHealth.AUTH_REQUIRED,
                reason=exc.message,
                hint=exc.hint or creds_mod.LOGIN_HINT,
                checked_at=now,
            )
        except AdapterError as exc:
            return Health(
                health=ProviderHealth.UNAVAILABLE,
                reason=exc.message,
                hint=exc.hint,
                checked_at=now,
            )
        detail: dict[str, Any] = {
            "user": who.get("user"),
            "teamspace": who.get("teamspace"),
            "teamspaces": who.get("teamspaces"),
            "sdk_version": who.get("sdk_version"),
            "credentials": found.source,
            "studio": self._studio,
        }
        if not who.get("teamspace"):
            return Health(
                health=ProviderHealth.AUTH_REQUIRED,
                reason="cannot pick a lightning teamspace automatically",
                hint="set providers.lightning.teamspace to one of: "
                + (", ".join(who.get("teamspaces") or []) or "none found"),
                checked_at=now,
                detail=detail,
            )
        return Health(health=ProviderHealth.OK, checked_at=now, detail=detail)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _to_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
