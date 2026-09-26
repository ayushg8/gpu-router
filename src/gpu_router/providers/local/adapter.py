"""LocalAdapter: runs job bundles on this Mac with PyTorch MPS (phase 3; kind "local").

The "remote" is a directory per attempt, `<home>/local/<remote_id>/` (remote_id = the
attempt key, so submit is idempotent per key and lookup_by_key is a directory check), plus
one detached process tree started through `launcher.py`. Everything observable lives on
disk, so a daemon restart reattaches to a run that is still going or sees that it died
(invariants 10, 11):

    <home>/local/<remote_id>/
      alive.lock    flock held by the run for its whole life (liveness, pid-reuse proof)
      run.json      submit record, written before the launcher starts (commit point)
      launch.json   the launcher's plan: env choice, bundle, bootstrap args (no secrets)
      pid.json      pid of the detached run (its own session + process group)
      phase         prepare | env | install | run            (PENDING until "run")
      env.json      interpreter / venv the run uses
      cancel.json   cancel() was asked while the run was alive
      console.log   launcher + bootstrap stdout/stderr = the run's log (cursor = byte offset)
      ckpt-sync/    checkpoint archives (bootstrap --checkpoint-sync-dir, file:// URIs)
      work/         bootstrap --workdir: EXIT, job.log, bundle/, checkpoints/, outputs/
    <home>/providers/local/venvs/<deps key>/   shared python envs (installed once per key)
    <home>/providers/local/data/               GPU_DATA_DIR, shared by local runs
    <home>/providers/local/served/<remote_id>  when logs() first served a dead run's log to
                                               eof (injected clock; ".purged" once deleted)

Python env (CLAUDE.md decision D28): `providers.local.env` in config.yaml, or
`provider_options.local.env` per job: "venv" (default) = a uv-managed venv per deps key
created from `python` (default: the interpreter the daemon runs on), so deps install once
and later runs start in well under a second; "system" = run with `python` as it is (a
conda env, a hand-made venv) and install nothing. `base_packages` (settings) are
installed into every venv, e.g. [torch, numpy] to mirror the Kaggle/Colab images.
`PYTORCH_ENABLE_MPS_FALLBACK=1` is set unless the job sets it.

Status mapping (process dead = alive.lock free):
    alive, phase != run            -> PENDING ("unpacking", "setting up the python env", ...)
    alive, phase == run            -> RUNNING
    dead, EXIT 0                   -> SUCCEEDED (even if a cancel raced the finish)
    dead, cancel.json              -> CANCELLED
    dead, EXIT 90                  -> LOST: env/dependency install failed (D22/D26: reroute)
    dead, EXIT 128+HUP/INT/KILL/TERM, no cancel.json -> LOST (Mac shutdown, jetsam, kill)
    dead, other EXIT               -> FAILED with that exit code
    dead, no EXIT                  -> LOST (runner killed, Mac restarted, or never started)

Raw logs (D37): console.log and work/job.log are the job's unredacted output (the engine
keeps its own redacted copy). They are deleted RAW_LOG_GRACE_S after logs() served a dead
run's log to eof, and a dead run's whole dir RETENTION_S after that (or after its submit);
the sweep runs at submit() and healthcheck().

Job environment (D38): an allowlist of the daemon's env (PATH, HOME, locale, TMPDIR, proxy
and CA settings, uv/pip/torch/HF cache knobs), never the rest: a daemon auto-started from
a shell would otherwise hand every job that shell's KAGGLE_KEY / AWS_* / OPENAI_API_KEY.
Allowlisted names that look like secrets and values with credentials are dropped too. Jobs
get secrets only through JobSpec.secrets.

Resume: a checkpoint this Mac cannot read (a file:// path on a Colab or Kaggle VM) starts
the job fresh with a note in the log, like the cloud adapters (D34); Colab's local mirror
of one (`<home>/providers/<colab>/runs/<session>/ckpt/`) is used when it exists.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import logging
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import threading
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
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
from gpu_router.errors import InvalidJob, NotFound, Unavailable
from gpu_router.models import Job, ProviderHealth, QuotaSnapshot
from gpu_router.providers.local import launcher as L

__all__ = ["LocalAdapter"]

_log = logging.getLogger(__name__)

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
LAUNCHER_PATH = Path(L.__file__).resolve()
MPS_GPU = "MPS"
MPS_FALLBACK_ENV = "PYTORCH_ENABLE_MPS_FALLBACK"

SPAWN_TIMEOUT_S = 20.0  # the launcher's first process only forks and writes pid.json
PID_WAIT_S = 2.0
CANCEL_GRACE_S = 15.0  # SIGTERM -> bootstrap stops the entrypoint, drains, final ckpt sync
KILL_WAIT_S = 5.0
PS_TIMEOUT_S = 5.0
POLL_STEP_S = 0.1
FOLLOW_POLL_S = 0.5
LOG_CHUNK_BYTES = 1 << 20
MAX_LINE_BYTES = 4 << 20  # a "line" longer than this without a newline is cut
LOW_DISK_BYTES = 2 << 30
RAW_LOG_GRACE_S = 3600.0  # raw console.log kept this long after logs() served it to eof
RETENTION_S = 7 * 24 * 3600.0  # a dead run's dir (the engine keeps logs and outputs)

#: Daemon-side variables that must never reach a job (the daemon's own venv, test hooks).
_STRIP_ENV = frozenset(
    {
        "VIRTUAL_ENV",
        "PYTHONHOME",
        "PYTHONPATH",
        "PYTHONSTARTUP",
        "PYTHONEXECUTABLE",
        "PYTHONSAFEPATH",
        "__PYVENV_LAUNCHER__",
        "UV_PROJECT_ENVIRONMENT",
        "UV_RUN_RECURSION_DEPTH",
    }
)
#: bootstrap reads these; set by anyone but this adapter they would move EXIT, the log or
#: the outputs out of the run dir, so they are dropped from both the daemon and job env.
_RUNNER_CONTROL_ENV = frozenset(
    {
        "GPU_BUNDLE",
        "GPU_WORKDIR",
        "GPU_RESUME_SRC",
        "GPU_RESUME_DIR",
        "GPU_LOG_FILE",
        "GPU_EXIT_FILE",
        "GPU_HEARTBEAT_S",
        "GPU_CHECKPOINT_SYNC_DIR",
        "GPU_CKPT_SEQ_START",
        "GPU_SKIP_INSTALL",
        "GPU_CHECKPOINT_DIR",
        "GPU_OUTPUT_DIR",
    }
)
#: Daemon env vars a job may inherit (D38). Everything else is dropped.
_ENV_ALLOW = frozenset(
    {
        "PATH",
        "HOME",
        "USER",
        "LOGNAME",
        "SHELL",
        "TERM",
        "TMPDIR",
        "TZ",
        "LANG",
        "__CF_USER_TEXT_ENCODING",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "no_proxy",
        "all_proxy",
        "HF_HOME",
        "HF_HUB_CACHE",
        "HF_DATASETS_CACHE",
        "HF_HUB_OFFLINE",
        "HF_HUB_ENABLE_HF_TRANSFER",
        "TRANSFORMERS_CACHE",
        "TORCH_HOME",
        "XDG_CACHE_HOME",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
        "OMP_NUM_THREADS",
    }
)
_ENV_ALLOW_PREFIXES = ("LC_", "UV_", "PIP_", "PYTORCH_")
_URL_USERINFO = re.compile(r"://[^/@\s]*@")
_LOST_SIGNALS = frozenset({signal.SIGHUP, signal.SIGINT, signal.SIGKILL, signal.SIGTERM})
_UV_CANDIDATES = ("~/.local/bin/uv", "~/.cargo/bin/uv", "/opt/homebrew/bin/uv", "/usr/local/bin/uv")
_PHASE_MESSAGES = {
    L.PHASE_PREPARE: "unpacking the bundle",
    L.PHASE_ENV: "setting up the python env",
    L.PHASE_INSTALL: "installing dependencies",
}


@dataclass(frozen=True, slots=True)
class EnvChoice:
    """How a run gets its interpreter (settings merged with per-job provider_options)."""

    mode: str  # L.ENV_VENV | L.ENV_SYSTEM
    python: str  # absolute path: venv base interpreter, or the interpreter itself
    uv: str | None  # uv binary for venv mode (None = stdlib venv + pip)
    base_packages: tuple[str, ...] = ()


class _EnvProblem(Exception):
    def __init__(self, message: str, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint


class LocalAdapter(Adapter):
    kind = "local"
    capabilities = Capabilities(
        lookup_by_key=True,
        live_logs=True,
        interactive=False,
        resume=True,
        fetch=True,
        cancel_confirms=True,
        live_quota=True,
        max_session_hours=None,
        max_concurrency=1,
        max_bundle_mb=None,
        poll_interval_s=5,
    )

    def __init__(self, deps: AdapterDeps) -> None:
        super().__init__(deps)
        self._submit_lock = threading.Lock()

    # ------------------------------------------------------------------ layout

    @property
    def runs_root(self) -> Path:
        """`<home>/local/`: one directory per remote run (the fake's `<home>/fake/`, D9)."""
        return self.paths.home / "local"

    @property
    def venvs_root(self) -> Path:
        return self.paths.provider_dir(self.name) / "venvs"

    @property
    def data_root(self) -> Path:
        return self.paths.provider_dir(self.name) / "data"

    # ------------------------------------------------------------------ contract

    def submit(self, job: Job, ctx: AttemptContext) -> RemoteRef:
        remote_id = ctx.attempt_key
        if not _SAFE_ID.match(remote_id):
            raise InvalidJob(
                f"attempt key {remote_id!r} is not a usable run id", provider=self.name
            )
        with self._submit_lock:
            run_dir = self.runs_root / remote_id
            record = self._read_record(run_dir)
            if record is not None:
                return self._ref(run_dir, record)  # A4: same key, same run
            self._sweep_runs(exclude=remote_id)
            if job.spec.interactive:
                raise InvalidJob("the local provider runs batch jobs only", provider=self.name)
            archive, bundle_dir = self._bundle_for(job, ctx)
            try:
                choice = self._env_choice(job)
            except _EnvProblem as exc:
                raise InvalidJob(exc.message, provider=self.name, hint=exc.hint) from None
            resume, resume_note = self._resume_path(ctx)
            env = self._child_env(job, ctx, run_dir)
            plan = {
                "v": L.LAUNCH_VERSION,
                "remote_id": remote_id,
                "env": choice.mode,
                "python": choice.python,
                "uv": choice.uv,
                "base_packages": list(choice.base_packages),
                "venvs_dir": str(self.venvs_root),
                "bundle_archive": str(archive) if archive is not None else None,
                "bundle_dir": str(bundle_dir) if bundle_dir is not None else None,
                "bootstrap_fallback": str(_packaged_bootstrap()),
                "bootstrap_args": self._bootstrap_args(ctx, run_dir, resume),
            }
            record = {
                "remote_id": remote_id,
                "provider": self.name,
                "attempt_key": ctx.attempt_key,
                "attempt_id": ctx.attempt_id,
                "job_id": job.id,
                "n": ctx.n,
                "env": choice.mode,
                "python": choice.python,
                "submitted_at": self.clock.now(),
                "resume": None
                if ctx.resume_from is None
                else ("restored" if resume is not None else "unavailable"),
            }
            notes = [resume_note] if resume_note else []
            return self._start(run_dir, plan, record, env, notes=notes)

    def status(self, ref: RemoteRef) -> RemoteStatus:
        run_dir, _record = self._require_run(ref.remote_id)
        url = _dir_url(run_dir)
        cancelling = (run_dir / L.CANCEL_JSON).exists()
        phase = _read_text(run_dir / L.PHASE_FILE)
        started_at = _mtime(run_dir / L.PHASE_FILE) if phase == L.PHASE_RUN else None
        if _alive(run_dir):
            if phase == L.PHASE_RUN:
                return RemoteStatus(
                    phase=RemotePhase.RUNNING,
                    message="cancelling" if cancelling else None,
                    gpu=MPS_GPU,
                    started_at=started_at,
                    url=url,
                )
            message = _PHASE_MESSAGES.get(phase or "", "starting the local runner")
            return RemoteStatus(
                phase=RemotePhase.PENDING,
                message=f"{message}; cancelling" if cancelling else message,
                gpu=MPS_GPU,
                url=url,
            )
        exit_file = run_dir / L.WORK_DIR / L.EXIT_NAME
        code = _read_exit(exit_file)
        ended_at = _mtime(exit_file) or _mtime(run_dir / L.CONSOLE_LOG)
        common: dict[str, Any] = {
            "gpu": MPS_GPU,
            "started_at": started_at,
            "ended_at": ended_at,
            "url": url,
        }
        if code == 0:
            return RemoteStatus(phase=RemotePhase.SUCCEEDED, exit_code=0, **common)
        if cancelling:
            return RemoteStatus(
                phase=RemotePhase.CANCELLED, message="cancelled", exit_code=code, **common
            )
        if code is None:
            if not (run_dir / L.PID_JSON).exists():
                why = "the local runner never started (gpu-router stopped during submit)"
            else:
                why = "the local runner died without an exit code (killed, or the Mac restarted)"
            return RemoteStatus(phase=RemotePhase.LOST, lost_reason=why, **common)
        if code == L.INSTALL_FAILED_EXIT:
            return RemoteStatus(
                phase=RemotePhase.LOST,
                lost_reason=(
                    f"python env or dependency install failed (exit {code}); the job never started"
                ),
                exit_code=code,
                **common,
            )
        sig = code - 128
        if sig in _LOST_SIGNALS:
            return RemoteStatus(
                phase=RemotePhase.LOST,
                lost_reason=(
                    f"stopped by {signal.Signals(sig).name} outside gpu-router "
                    "(Mac shutdown or logout, memory pressure, or a manual kill)"
                ),
                exit_code=code,
                **common,
            )
        return RemoteStatus(phase=RemotePhase.FAILED, exit_code=code, **common)

    def logs(
        self, ref: RemoteRef, *, follow: bool = False, since: str | None = None
    ) -> Iterator[LogChunk]:
        run_dir, _record = self._require_run(ref.remote_id)  # eager: NotFound before iterating
        return self._iter_logs(run_dir, _parse_cursor(since), follow)

    def fetch(self, ref: RemoteRef, dest: Path) -> FetchResult:
        run_dir, _record = self._require_run(ref.remote_id)
        src = run_dir / L.WORK_DIR / L.OUTPUTS_NAME
        if not src.is_dir():
            raise NotFound(
                f"local run {ref.remote_id} has no outputs",
                provider=self.name,
                hint="the run stopped before its output dir was created; see `gpu logs`",
            )
        return _copy_outputs(src, dest)

    def cancel(self, ref: RemoteRef) -> None:
        if not _SAFE_ID.match(ref.remote_id):
            return
        run_dir = self.runs_root / ref.remote_id
        if self._read_record(run_dir) is None or not _alive(run_dir):
            return  # A5: unknown or already finished
        _write_json(run_dir / L.CANCEL_JSON, {"requested_at": self.clock.now()})
        pid = _read_pid(run_dir)
        for _ in range(int(PID_WAIT_S / POLL_STEP_S)):
            if pid is not None or not _alive(run_dir):
                break
            _pause(POLL_STEP_S)
            pid = _read_pid(run_dir)
        if pid is None:
            return  # died meanwhile, or the launcher never reported: status() tells
        # Lock held => pid is our run (exec keeps the pid); the whole group gets it:
        # the launcher and uv while preparing, bootstrap (which forwards to the entrypoint's
        # own group, drains and writes EXIT) once running.
        _signal_group(pid, signal.SIGTERM)
        if _wait_dead(run_dir, CANCEL_GRACE_S):
            return
        for pgid in _child_groups(pid):  # the entrypoint ignores SIGTERM: stop it for good
            _signal_group(pgid, signal.SIGKILL)
        _signal_group(pid, signal.SIGKILL)
        _wait_dead(run_dir, KILL_WAIT_S)

    def quota(self) -> QuotaSnapshot:
        return QuotaSnapshot(
            provider=self.name,
            used=0.0,
            limit=None,
            unit=self.entry.quota.unit,
            resets_at=None,
            source="live",
            detail={"unlimited": True, "note": "local runs are free and unmetered"},
            observed_at=self.clock.now(),
        )

    def healthcheck(self) -> Health:
        now = self.clock.now()
        if self._submit_lock.acquire(blocking=False):  # local file hygiene only (D37)
            try:
                self._sweep_runs()
            finally:
                self._submit_lock.release()
        machine = platform.machine()
        detail: dict[str, Any] = {"machine": machine, "platform": sys.platform}
        if sys.platform != "darwin":
            return Health(
                health=ProviderHealth.UNAVAILABLE,
                reason="not a Mac: the local provider runs jobs on Apple Silicon (MPS)",
                hint="set `providers.local.enabled: false` in config.yaml",
                checked_at=now,
                detail=detail,
            )
        macos = platform.mac_ver()[0]
        detail["macos"] = macos
        if machine != "arm64":
            return Health(
                health=ProviderHealth.UNAVAILABLE,
                reason=(
                    f"no Apple Silicon GPU for this python ({machine}); MPS needs an arm64 "
                    "python on an M-series Mac"
                ),
                hint="run gpu-router with an arm64 python, or disable the local provider",
                checked_at=now,
                detail=detail,
            )
        try:
            choice = self._env_choice(None)
        except _EnvProblem as exc:
            return Health(
                health=ProviderHealth.UNAVAILABLE,
                reason=exc.message,
                hint=exc.hint,
                checked_at=now,
                detail=detail,
            )
        detail.update(
            {
                "env": choice.mode,
                "python": choice.python,
                "uv": choice.uv,
                "base_packages": list(choice.base_packages),
            }
        )
        memory = _memory_bytes()
        if memory is not None:
            detail["memory_gb"] = round(memory / (1 << 30), 1)
        probe = self.paths.home if self.paths.home.exists() else self.paths.home.parent
        try:
            free = shutil.disk_usage(probe).free
        except OSError:
            free = None
        if free is not None:
            detail["free_disk_gb"] = round(free / (1 << 30), 1)
            if free < LOW_DISK_BYTES:
                return Health(
                    health=ProviderHealth.DEGRADED,
                    reason=f"only {free / (1 << 30):.1f} GB free on the disk holding {probe}",
                    hint="free some space; python envs and outputs of local runs live there",
                    checked_at=now,
                    detail=detail,
                )
        return Health(health=ProviderHealth.OK, checked_at=now, detail=detail)

    def lookup_by_key(self, attempt_key: str) -> RemoteRef | None:
        if not _SAFE_ID.match(attempt_key):
            return None
        run_dir = self.runs_root / attempt_key
        record = self._read_record(run_dir)
        return None if record is None else self._ref(run_dir, record)

    # ------------------------------------------------------------------ submit helpers

    def _start(
        self,
        run_dir: Path,
        plan: dict[str, Any],
        record: dict[str, Any],
        env: dict[str, str],
        *,
        notes: list[str] | None = None,
    ) -> RemoteRef:
        """Claim the run dir, start the launcher holding alive.lock, return the ref."""
        if run_dir.exists():
            # Left by a submit that died before writing run.json: nothing was started
            # (the launcher only starts after run.json), so the dir is ours to reuse.
            shutil.rmtree(run_dir)
        self.runs_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        run_dir.mkdir(mode=0o700)
        (run_dir / L.WORK_DIR).mkdir(mode=0o700)
        self.data_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        lock_fd = os.open(run_dir / L.ALIVE_LOCK, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            # Held from before run.json exists until the run ends (the child inherits it),
            # so no observer can ever see a committed run without a live holder.
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            _write_json(run_dir / L.LAUNCH_JSON, plan)
            _write_json(run_dir / L.RUN_JSON, record)
            console_fd = os.open(
                run_dir / L.CONSOLE_LOG, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600
            )
            for note in notes or ():
                os.write(console_fd, f"gpu-router: {note}\n".encode())
            try:
                proc = subprocess.Popen(
                    [
                        sys.executable,
                        "-I",
                        str(LAUNCHER_PATH),
                        str(run_dir),
                        "--lock-fd",
                        str(lock_fd),
                    ],
                    stdin=subprocess.DEVNULL,
                    stdout=console_fd,
                    stderr=subprocess.STDOUT,
                    cwd=run_dir,
                    env=env,
                    start_new_session=True,
                    pass_fds=(lock_fd,),
                )
            except OSError as exc:
                shutil.rmtree(run_dir, ignore_errors=True)  # certain: nothing started
                raise Unavailable(
                    f"could not start the local launcher: {exc.strerror or exc}",
                    provider=self.name,
                ) from None
            finally:
                os.close(console_fd)
            try:
                rc = proc.wait(timeout=SPAWN_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                raise Unavailable(
                    "the local launcher did not report in time", provider=self.name
                ) from None
        except OSError as exc:
            raise Unavailable(
                f"could not prepare local run {run_dir.name}: {exc.strerror or exc}",
                provider=self.name,
            ) from None
        finally:
            os.close(lock_fd)  # never LOCK_UN: that would release the child's lock too
        if rc != 0:
            raise Unavailable(
                f"the local launcher exited with {rc} before detaching", provider=self.name
            )
        return self._ref(run_dir, record)

    def _bundle_for(self, job: Job, ctx: AttemptContext) -> tuple[Path | None, Path | None]:
        """(archive, dir) to run; the archive is preferred (a clean copy per attempt)."""
        archive = ctx.bundle_archive
        bundle_dir = ctx.bundle_dir
        if archive is None and bundle_dir is None:
            # A context built without bundle paths: use the job's materialized bundle.
            archive = self.paths.job_bundle_archive(job.id)
            bundle_dir = self.paths.job_bundle_dir(job.id)
        ok_archive = archive if archive is not None and archive.is_file() else None
        ok_dir = (
            bundle_dir
            if bundle_dir is not None and (bundle_dir / "manifest.json").is_file()
            else None
        )
        if ok_archive is None and ok_dir is None:
            raise InvalidJob(
                f"job {job.short_id} has no bundle to run",
                provider=self.name,
                hint="resubmit the job so the daemon packages the project again",
            )
        return ok_archive, (ok_dir if ok_archive is None else None)

    def _resume_path(self, ctx: AttemptContext) -> tuple[Path | None, str | None]:
        """(checkpoint to resume from, note). The engine hands every placement the job's
        latest checkpoint, whichever provider wrote it: one this Mac cannot read starts the
        run fresh with a note (D34), never InvalidJob, which would exclude local for the
        rest of the job."""
        ckpt = ctx.resume_from
        if ckpt is None:
            return None, None
        staged = ctx.env.get("GPU_RESUME_URI")
        if staged and staged.startswith("file:"):
            # phase 5: the engine put this checkpoint into local checkpoint storage (copied
            # from the HF bucket when a remote attempt wrote it)
            local = Path(unquote(urlparse(staged).path))
            if local.exists():
                return local, None
        parsed = urlparse(ckpt.uri)
        path: Path | None = None
        if parsed.scheme == "file" and parsed.netloc in ("", "localhost"):
            path = Path(unquote(parsed.path))
        elif parsed.scheme == "" and ckpt.uri.startswith("/"):
            path = Path(ckpt.uri)
        if path is not None:
            if path.exists():
                return path, None
            mirror = self._mirrored_checkpoint(path)
            if mirror is not None:
                return mirror, None
        note = f"checkpoint {ckpt.seq} ({ckpt.uri}) is not reachable from this Mac; starting fresh"
        _log.warning("local: %s (attempt %s)", note, ctx.attempt_id)
        return None, note

    def _mirrored_checkpoint(self, path: Path) -> Path | None:
        """A cloud adapter's Mac-side copy of a VM checkpoint: Colab mirrors
        `<remote_root>/<session>/ckpt-sync/<name>` to
        `<home>/providers/<name>/runs/<session>/ckpt/<name>` (colab adapter,
        _mirror_checkpoint)."""
        if path.parent.name != "ckpt-sync" or not _SAFE_ID.match(path.parent.parent.name):
            return None
        if not re.match(r"^ckpt-\d+\.tar\.gz$", path.name):
            return None
        providers = self.paths.provider_dir(self.name).parent
        try:
            dirs = sorted(d for d in providers.iterdir() if d.is_dir())
        except OSError:
            return None
        for d in dirs:
            candidate = d / "runs" / path.parent.parent.name / "ckpt" / path.name
            if candidate.is_file():
                return candidate
        return None

    def _bootstrap_args(self, ctx: AttemptContext, run_dir: Path, resume: Path | None) -> list[str]:
        seq_start = ctx.resume_from.seq + 1 if ctx.resume_from is not None else 1
        args = [
            "--checkpoint-sync-dir",
            str(run_dir / L.CKPT_SYNC_DIR),
            "--ckpt-seq-start",
            str(seq_start),
            "--checkpoint-interval-min",
            str(ctx.checkpoint_interval_min),
        ]
        if resume is not None:
            args += ["--resume", str(resume)]
        return args

    def _child_env(self, job: Job, ctx: AttemptContext, run_dir: Path) -> dict[str, str]:
        """The job's environment: an allowlist of the daemon's env (D38), then the job's
        env, the run layout and the resolved secrets (in memory only, invariant 12)."""
        env = {k: v for k, v in os.environ.items() if _inheritable(k, v)}
        if "PATH" in env and sys.prefix != sys.base_prefix:
            # `uv run gpu daemon` puts the daemon's own venv first on PATH; a job's `python`
            # must never resolve to it.
            own_bin = str(Path(sys.prefix) / "bin")
            env["PATH"] = os.pathsep.join(
                p for p in env["PATH"].split(os.pathsep) if p and p != own_bin
            )
        for key, value in ctx.env.items():
            if key not in _RUNNER_CONTROL_ENV:
                env[key] = value
        work = run_dir / L.WORK_DIR
        env.setdefault("GPU_ROUTER_JOB_ID", job.id)
        env.setdefault("GPU_ROUTER_ATTEMPT", str(ctx.n))
        env["GPU_ROUTER_PROTOCOL"] = "1"
        env["GPU_CHECKPOINT_DIR"] = str(work / L.CHECKPOINTS_NAME)
        env["GPU_OUTPUT_DIR"] = str(work / L.OUTPUTS_NAME)
        env.setdefault("GPU_DATA_DIR", str(self.data_root))
        env.setdefault(MPS_FALLBACK_ENV, "1")
        env.setdefault("PYTHONUNBUFFERED", "1")
        for name, secret in ctx.secrets.items():
            env[name] = secret.get_secret_value()
        return env

    def _uv(self, opts: Mapping[str, Any]) -> str | None:
        """settings `uv`: a path, or false/"" to never use uv (stdlib venv + pip); unset =
        find it."""
        if "uv" not in opts:
            return _find_uv()
        raw = opts["uv"]
        if not raw:
            return None
        found = _resolve_executable(str(raw))
        if found is None:
            raise _EnvProblem(
                f"uv for local runs not found: {raw}",
                hint=f"fix providers.{self.name}.uv, or set it to false to use pip",
            )
        return found

    def _env_choice(self, job: Job | None) -> EnvChoice:
        """Resolve settings (config.yaml providers.<name>) + per-job provider_options."""
        opts: dict[str, Any] = dict(self.settings.model_extra or {})
        if job is not None:
            per_job = job.spec.provider_options.get(self.name) or {}
            opts.update({k: v for k, v in per_job.items() if k in ("env", "python")})
        mode = str(opts.get("env") or L.ENV_VENV)
        if mode not in (L.ENV_VENV, L.ENV_SYSTEM):
            raise _EnvProblem(
                f"providers.{self.name}.env is {mode!r}; use venv or system",
                hint="fix config.yaml",
            )
        raw_python = opts.get("python")
        if raw_python:
            python = _resolve_executable(str(raw_python))
            if python is None:
                raise _EnvProblem(
                    f"python for local runs not found: {raw_python}",
                    hint=f"set providers.{self.name}.python to an existing interpreter",
                )
        elif mode == L.ENV_SYSTEM:
            raise _EnvProblem(
                f"providers.{self.name}.env is system but no python is set",
                hint=f"set providers.{self.name}.python to the interpreter to use",
            )
        else:
            python = _default_base_python()
            if _resolve_executable(python) is None:
                raise _EnvProblem(
                    f"the daemon's base python {python} is missing",
                    hint=f"set providers.{self.name}.python to an existing interpreter",
                )
        uv = self._uv(opts) if mode == L.ENV_VENV else None
        raw_base = opts.get("base_packages") or ()
        if isinstance(raw_base, str):
            raw_base = raw_base.split()
        if not isinstance(raw_base, list | tuple):
            raise _EnvProblem(
                f"providers.{self.name}.base_packages must be a list of requirement strings"
            )
        return EnvChoice(
            mode=mode, python=python, uv=uv, base_packages=tuple(str(p) for p in raw_base)
        )

    # ------------------------------------------------------------------ raw log hygiene

    @property
    def served_root(self) -> Path:
        """Adapter scratch, not the run dir: logs() never changes a run (A6)."""
        return self.paths.provider_dir(self.name) / "served"

    def _mark_served(self, run_dir: Path) -> None:
        marker = self.served_root / run_dir.name
        if marker.exists():
            return
        with contextlib.suppress(OSError):
            marker.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            marker.write_text(f"{self.clock.now():.3f}\n", encoding="utf-8")

    def _sweep_runs(self, *, exclude: str | None = None) -> None:
        """Delete raw logs RAW_LOG_GRACE_S after they were served, and dead runs' dirs
        RETENTION_S after that (or after their submit). Live runs are never touched."""
        root = self.runs_root
        if not root.is_dir():
            return
        now = self.clock.now()
        for run_dir in root.iterdir():
            if run_dir.name == exclude or not _SAFE_ID.match(run_dir.name):
                continue
            if not run_dir.is_dir() or run_dir.is_symlink():
                continue
            record = self._read_record(run_dir)
            if record is None or _alive(run_dir):
                continue
            marker = self.served_root / run_dir.name
            purged = marker.with_name(f"{run_dir.name}.purged")
            served = _read_float(marker)
            submitted = record.get("submitted_at")
            since = max(
                float(submitted) if isinstance(submitted, int | float) else 0.0, served or 0.0
            )
            if now - since > RETENTION_S:
                shutil.rmtree(run_dir, ignore_errors=True)
                for f in (marker, purged):
                    with contextlib.suppress(OSError):
                        f.unlink()
                continue
            if served is None or now - served < RAW_LOG_GRACE_S or purged.exists():
                continue
            work = run_dir / L.WORK_DIR
            for raw in (run_dir / L.CONSOLE_LOG, work / "job.log", work / "job.log.prev"):
                with contextlib.suppress(OSError):
                    raw.unlink()
            with contextlib.suppress(OSError):
                purged.write_text(f"{now:.3f}\n", encoding="utf-8")

    # ------------------------------------------------------------------ records

    def _read_record(self, run_dir: Path) -> dict[str, Any] | None:
        try:
            data = json.loads((run_dir / L.RUN_JSON).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return data if isinstance(data, dict) else None

    def _require_run(self, remote_id: str) -> tuple[Path, dict[str, Any]]:
        if _SAFE_ID.match(remote_id):
            run_dir = self.runs_root / remote_id
            record = self._read_record(run_dir)
            if record is not None:
                return run_dir, record
        raise NotFound(f"{self.name} has no run {remote_id!r}", provider=self.name)

    def _ref(self, run_dir: Path, record: Mapping[str, Any]) -> RemoteRef:
        return RemoteRef(
            remote_id=run_dir.name,
            url=_dir_url(run_dir),
            meta={
                "run_dir": str(run_dir),
                "env": str(record.get("env") or ""),
                "python": str(record.get("python") or ""),
            },
        )

    def _iter_logs(self, run_dir: Path, offset: int, follow: bool) -> Iterator[LogChunk]:
        log = run_dir / L.CONSOLE_LOG
        while True:
            terminal = not _alive(run_dir)  # decided BEFORE reading: a dead run's log is final
            last: LogChunk | None = None
            for chunk in _read_chunks(log, offset, terminal):
                offset = int(chunk.cursor)
                last = chunk
                if chunk.eof:
                    self._mark_served(run_dir)  # before the yield: consumers may stop here
                yield chunk
            if not follow or (last is not None and last.eof):
                return
            _pause(FOLLOW_POLL_S)


# --------------------------------------------------------------------------- helpers


def _daemon_only(key: str) -> bool:
    return key in _STRIP_ENV or key in _RUNNER_CONTROL_ENV or key.startswith("GPU_ROUTER_")


def _inheritable(key: str, value: str) -> bool:
    """May a job inherit this daemon env var (D38)? Allowlisted names only; never one
    that looks like a secret, a value carrying URL credentials, or one redaction would
    change (a token-shaped value)."""
    if _daemon_only(key):
        return False
    if key not in _ENV_ALLOW and not key.startswith(_ENV_ALLOW_PREFIXES):
        return False
    from gpu_router import secrets
    from gpu_router.models import SECRET_ENV_NAME_STRICT

    if SECRET_ENV_NAME_STRICT.search(key):
        return False
    if _URL_USERINFO.search(value):
        return False
    return secrets.redact(value) == value


def _read_float(path: Path) -> float | None:
    text = _read_text(path)
    try:
        return float(text) if text is not None else None
    except ValueError:
        return None


def _alive(run_dir: Path) -> bool:
    """True while some process holds alive.lock (the run). Probe = non-blocking SHARED lock,
    released at once: concurrent probes never block each other or look like a holder."""
    try:
        fd = os.open(run_dir / L.ALIVE_LOCK, os.O_RDONLY)
    except OSError:
        return False
    try:
        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    except OSError:
        return False
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)


def _pause(seconds: float) -> None:
    """Blocking wait that never reads the wall clock (A10)."""
    threading.Event().wait(seconds)


def _wait_dead(run_dir: Path, budget_s: float) -> bool:
    for _ in range(max(1, int(budget_s / POLL_STEP_S))):
        if not _alive(run_dir):
            return True
        _pause(POLL_STEP_S)
    return not _alive(run_dir)


def _signal_group(pgid: int, signum: int) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pgid, signum)


def _child_groups(pid: int) -> list[int]:
    """Process groups of `pid`'s direct children other than its own (bootstrap runs the
    entrypoint and pip in their own sessions)."""
    try:
        out = subprocess.run(
            ["/bin/ps", "-A", "-o", "pid=,ppid=,pgid="],
            capture_output=True,
            text=True,
            timeout=PS_TIMEOUT_S,
            check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    groups: set[int] = set()
    for row in out.splitlines():
        parts = row.split()
        if len(parts) != 3 or not all(p.isdigit() for p in parts):
            continue
        _child, ppid, pgid = (int(p) for p in parts)
        if ppid == pid and pgid != pid:
            groups.add(pgid)
    return sorted(groups)


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _read_exit(path: Path) -> int | None:
    text = _read_text(path)
    if text is None:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _read_pid(run_dir: Path) -> int | None:
    try:
        data = json.loads((run_dir / L.PID_JSON).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    pid = data.get("pid") if isinstance(data, dict) else None
    return pid if isinstance(pid, int) and not isinstance(pid, bool) and pid > 1 else None


def _mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _dir_url(run_dir: Path) -> str:
    return run_dir.as_uri()


def _parse_cursor(since: str | None) -> int:
    if since is None:
        return 0
    try:
        return max(0, int(since))
    except ValueError:
        return 0


def _decode(raw: bytes) -> str:
    return raw.decode("utf-8", "replace").rstrip("\r")


def _read_chunks(log: Path, offset: int, terminal: bool) -> Iterator[LogChunk]:
    """Complete lines after byte `offset`; a trailing partial line only once the run is
    terminal. Always yields at least one chunk; the last one has eof=terminal."""
    try:
        fh = open(log, "rb")  # noqa: SIM115 - closed in finally; generator scope
    except FileNotFoundError:
        yield LogChunk(lines=[], cursor=str(offset), eof=terminal)
        return
    try:
        size = os.fstat(fh.fileno()).st_size
        offset = min(offset, size)  # a log never shrinks; a cursor past its end clamps
        fh.seek(offset)
        pending = b""
        while True:
            block = fh.read(LOG_CHUNK_BYTES)
            if not block:
                break
            data = pending + block
            cut = data.rfind(b"\n")
            if cut < 0:
                if len(data) < MAX_LINE_BYTES:
                    pending = data
                    continue
                cut = len(data) - 1  # an endless line: emit what we have as one line
            complete, pending = data[: cut + 1], data[cut + 1 :]
            text = complete[:-1] if complete.endswith(b"\n") else complete
            offset += len(complete)
            yield LogChunk(
                lines=[_decode(x) for x in text.split(b"\n")], cursor=str(offset), eof=False
            )
        if terminal and pending:
            offset += len(pending)
            yield LogChunk(lines=[_decode(pending)], cursor=str(offset), eof=True)
        else:
            yield LogChunk(lines=[], cursor=str(offset), eof=terminal)
    finally:
        fh.close()


def _copy_outputs(src: Path, dest: Path) -> FetchResult:
    """Copy src/** into dest (A9: overwrite what we write, never delete anything). Files
    already identical (size + mtime, copy2 keeps mtime) are counted but not copied again."""
    dest.mkdir(parents=True, exist_ok=True)
    files = 0
    total = 0
    skipped: list[str] = []
    for root, dirs, names in os.walk(src):
        here = Path(root)
        rel_root = here.relative_to(src)
        for d in list(dirs):
            if d.startswith(".gpu-") or (here / d).is_symlink():
                dirs.remove(d)
                if not d.startswith(".gpu-"):
                    skipped.append(f"{rel_root / d} (directory symlink)")
        for name in names:
            if name.startswith(".gpu-"):
                continue
            s = here / name
            rel = rel_root / name
            d_path = dest / rel
            try:
                if not s.is_file():
                    skipped.append(f"{rel} (not a regular file)")
                    continue
                st = s.stat()
                if d_path.is_dir() and not d_path.is_symlink():
                    skipped.append(f"{rel} (a directory with that name exists in {dest})")
                    continue
                d_path.parent.mkdir(parents=True, exist_ok=True)
                if not _same_file(st, d_path):
                    tmp = d_path.with_name(f".{d_path.name}.gpu-fetch")
                    shutil.copy2(s, tmp)
                    os.replace(tmp, d_path)
                files += 1
                total += st.st_size
            except OSError as exc:
                skipped.append(f"{rel} ({exc.strerror or exc})")
    message = None
    if skipped:
        shown = "; ".join(skipped[:5]) + (
            f"; and {len(skipped) - 5} more" if len(skipped) > 5 else ""
        )
        message = f"skipped {len(skipped)}: {shown}"
    return FetchResult(dest=dest, files=files, bytes=total, partial=bool(skipped), message=message)


def _same_file(st: os.stat_result, dest: Path) -> bool:
    try:
        if dest.is_symlink():
            return False
        dst = dest.stat()
    except OSError:
        return False
    return dst.st_size == st.st_size and dst.st_mtime_ns == st.st_mtime_ns


def _packaged_bootstrap() -> Path:
    from importlib import resources

    return Path(str(resources.files("gpu_router.runner").joinpath("bootstrap.py")))


def _default_base_python() -> str:
    """The interpreter the daemon runs on, outside its venv (venvs are made from it)."""
    base = getattr(sys, "_base_executable", None) or sys.executable
    return str(base)


def _resolve_executable(raw: str) -> str | None:
    candidate = os.path.expanduser(raw)
    if os.sep not in candidate:
        found = shutil.which(candidate)
        if found is None:
            return None
        candidate = found
    path = Path(candidate)
    if not path.is_absolute():
        path = path.absolute()
    return str(path) if path.is_file() and os.access(path, os.X_OK) else None


def _find_uv() -> str | None:
    """$UV (set by `uv run`) > PATH > the usual install locations (launchd's PATH is
    minimal)."""
    env_uv = os.environ.get("UV")
    candidates = [env_uv] if env_uv else []
    which = shutil.which("uv")
    if which:
        candidates.append(which)
    candidates += [os.path.expanduser(c) for c in _UV_CANDIDATES]
    for c in candidates:
        p = Path(c)
        if p.is_file() and os.access(p, os.X_OK):
            return str(p)
    return None


def _memory_bytes() -> int | None:
    try:
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        return None
