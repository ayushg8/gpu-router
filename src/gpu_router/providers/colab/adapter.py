"""ColabAdapter: batch jobs on a free Colab T4 through the official `colab` CLI (phase 3).

Pattern (proven by an existing Colab batch tool, NOTES.md "Submit"): one fresh session per
attempt, upload the bundle, launch the runner DETACHED on the VM (a long blocking `colab exec`
loses its websocket), then observe with short `colab exec` polls, download what is needed, and stop
the session. Free-tier ToS: batch only; never `colab ssh`/`console`, servers or web UIs.

Remote id = session name = "gr" + attempt key minus its "gpu" prefix (`gr-<job>-<n>`), so
the name alone finds the run again after a daemon crash (A4, invariant 6). Everything the
adapter knows about a run is in `state.RunRecord` under `<home>/providers/colab/runs/`.

Teardown is the one place this adapter acts during observation. A Colab session keeps its
GPU (and the account's free quota) until someone stops it, and the engine has no "release"
call, so when `status()` first sees the runner's exit it harvests what later calls need
(the log, and the outputs of a successful run when they are small) and stops the session.
That never changes what `status()` reports for the run (A6 is about observing the run, and
the run is over), but it does mean the first terminal `status()` takes a few extra CLI
calls. Whatever did not fit (a failed `colab stop`, a short budget) is finished by the next
`logs()` / `fetch()` / `cancel()` of that run, and otherwise by a janitor thread that retries
every JANITOR_INTERVAL_S without waiting for another Colab job (D36). A successful run with
large outputs keeps its session until `fetch()` pulls them (at most
PENDING_OUTPUTS_GIVE_UP_S); anything left by a crash is also stopped by the reaper at the
next `submit()`.

A `colab new` runs in its own session, so it survives a daemon crash. Its pid is recorded
in the run record, and a restarted daemon treats the attempt as pending until that CLI call
is gone (or, without a pid, until NEW_TIMEOUT_S + ORPHAN_GRACE_S have passed), then stops
the name idempotently: an orphaned `new` that registers the session late can never leave a
VM running with no record that owns it (D35).

Sessions are only ever stopped by the names this adapter created, from its own `--config`
session file: sessions made by other tools or by hand are invisible to it and never touched.
"""

from __future__ import annotations

import base64
import contextlib
import gzip
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import tarfile
import tempfile
import threading
import uuid
from collections.abc import Iterator
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
from gpu_router.errors import (
    DEFINITIVE_SUBMIT_ERRORS,
    AdapterError,
    AuthRequired,
    InvalidJob,
    NotFound,
    QuotaExhausted,
    Unavailable,
)
from gpu_router.models import Job, ProviderHealth, QuotaSnapshot
from gpu_router.providers.colab import cli as colab_cli
from gpu_router.providers.colab import remote
from gpu_router.providers.colab.cli import (
    ADC_HINT,
    INSTALL_HINT,
    CliResult,
    ColabCli,
    SessionGone,
    classify,
    resolve_cli,
    session_gone,
)
from gpu_router.providers.colab.state import (
    NO_RUN_STATES,
    STARTING_STATES,
    RunRecord,
    RunState,
    RunStore,
)

_log = logging.getLogger(__name__)

#: `colab new --gpu` values the CLI knows. Anything else silently becomes an A100 request
#: (NOTES.md "Quirks"), so requests are checked against this list AND the catalog entry.
SUPPORTED_GPUS: tuple[str, ...] = ("T4", "L4", "G4", "H100", "A100")
DEFAULT_REMOTE_ROOT = "/content/gr"
ENV_REAL_PROVIDERS = "GPU_ROUTER_REAL_PROVIDERS"
ADC_FILE = Path.home() / ".config" / "gcloud" / "application_default_credentials.json"

# Time budgets (seconds). Each stays below the engine's per-call timeout
# (config.engine.timeouts: submit 300, status/logs/cancel/quota/healthcheck 60, fetch 1800).
SUBMIT_BUDGET_S = 270.0
NEW_TIMEOUT_S = 180.0
EXEC_TIMEOUT_S = 30.0  # `colab exec --timeout` for the short scripts
EXEC_SLACK_S = 20.0  # websocket connect + kernel attach on top of the exec timeout
STATUS_BUDGET_S = 52.0
LOGS_BUDGET_S = 52.0
CANCEL_BUDGET_S = 52.0
HEALTH_TIMEOUT_S = 45.0
FETCH_BUDGET_S = 1700.0
STOP_TIMEOUT_S = 40.0
MIN_STEP_S = 12.0  # don't start a CLI call with less budget than this
STOP_RESERVE_S = 12.0  # budget kept for `colab stop` while harvesting inside one call
CANCEL_POLL_S = 25.0  # cancel's look at whether the runner already exited
JANITOR_INTERVAL_S = 120.0  # retry of teardowns nobody else finished (D36)
JANITOR_GIVE_UP_S = 13 * 3600.0  # after the 12 h session cap the VM is gone anyway
ORPHAN_GRACE_S = 30.0  # slack after NEW_TIMEOUT_S for a `colab new` with no recorded pid
PS_TIMEOUT_S = 5.0
RAW_LOG_GRACE_S = 3600.0  # raw harvested job.log kept this long after logs() served eof
CLI_LOG_MAX_BYTES = 5 * 1024 * 1024  # private colab.log is truncated beyond this
JANITOR_ENABLED = True  # tests switch it off unless they test the janitor itself

EAGER_OUTPUT_MAX_BYTES = 64 * 1024 * 1024  # outputs pulled inside status() up to this size
MIRROR_CKPT_MAX_BYTES = 64 * 1024 * 1024  # checkpoint archives mirrored to the Mac
LOG_READ_BYTES = 256 * 1024  # per remote log read
CHUNK_LINES = 2000  # lines per LogChunk
MAX_SETTLE_TRIES = 3  # harvest attempts before a session is stopped without its log
STALE_START_S = 600.0  # a creating/setup record older than this with no submit is dead
PENDING_OUTPUTS_GIVE_UP_S = 3600.0  # reaper stops a session whose outputs nobody fetched
RETENTION_S = 7 * 24 * 3600.0  # local run copies (record, log, outputs, ckpt mirror)
QUOTA_WINDOW_S = 24 * 3600.0
QUOTA_RESET_S = 24 * 3600.0  # batch-tool heuristic: a refused T4 comes back within ~a day
DEFAULT_MAX_BUNDLE_MB = 200.0

INSTALL_FAILED_EXIT = 90  # runner/bootstrap.py: dependency install failed (D22, D26)
OOM_EXITS = frozenset({137, -9})

RECLAIMED = (
    "the colab session ended (VM reclaimed: usage limit, idle timeout, Mac sleep or the 12 h cap)"
)

_SESSION_OK = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_SESSIONS_LINE = re.compile(r"^\[(?P<name>[^\]]+)\]\s+\S+\s+\|\s+Hardware:\s+(?P<hw>[^|]+)")


def session_name(attempt_key: str) -> str:
    """`gpu-<job>-<n>` -> `gr-<job>-<n>`; anything else -> `gr-<hash>` (still stable)."""
    if attempt_key.startswith("gpu-") and _SESSION_OK.match("gr" + attempt_key[3:]):
        return "gr" + attempt_key[3:]
    import hashlib

    return "gr-" + hashlib.sha256(attempt_key.encode()).hexdigest()[:20]


def short_gpu(name: str | None) -> str | None:
    """'Tesla T4, 15360 MiB' -> 'T4'. Unknown names pass through (trimmed)."""
    if not name:
        return None
    head = name.split(",")[0].strip()
    upper = head.upper()
    for gpu in ("H100", "A100", "L4", "T4", "G4"):
        if re.search(rf"(^|[^A-Z0-9]){gpu}([^A-Z0-9]|$)", upper):
            return gpu
    return head or None


def _real_opt_in() -> bool:
    raw = os.environ.get(ENV_REAL_PROVIDERS, "")
    return "colab" in {p.strip() for p in raw.split(",")}


class _Budget:
    """Wall-clock budget for one adapter call, measured with the injected clock (A10)."""

    def __init__(self, adapter: ColabAdapter, seconds: float) -> None:
        self._adapter = adapter
        self._clock = adapter.clock
        self._deadline = self._clock.monotonic() + seconds

    def remaining(self) -> float:
        return self._deadline - self._clock.monotonic()

    def clip(self, want: float, *, reserve: float = 0.0) -> float:
        return max(1.0, min(want, self.remaining() - reserve))

    def has(self, need: float = MIN_STEP_S) -> bool:
        return self.remaining() >= need

    def sub(self, seconds: float) -> _Budget:
        """A budget of at most `seconds`, never past this one's deadline."""
        return _Budget(self._adapter, max(0.0, min(seconds, self.remaining())))


class _SetupFailed(Exception):
    pass


def _parse_cursor(since: str | None) -> tuple[int, int]:
    """Cursor = "<lines>:<bytes>" (bytes = offset into job.log). Unknown -> start."""
    if not since:
        return 0, 0
    lines_s, _, bytes_s = since.partition(":")
    try:
        lines, off = int(lines_s), int(bytes_s)
    except ValueError:
        return 0, 0
    return max(0, lines), max(0, off)


def _cursor(lines: int, off: int) -> str:
    return f"{lines}:{off}"


class ColabAdapter(Adapter):
    kind = "colab"
    capabilities = Capabilities(
        lookup_by_key=True,
        live_logs=False,
        interactive=False,
        resume=True,
        fetch=True,
        cancel_confirms=True,
        live_quota=False,
        max_session_hours=12,
        max_concurrency=1,
        max_bundle_mb=DEFAULT_MAX_BUNDLE_MB,
        poll_interval_s=30,
    )

    def __init__(self, deps: AdapterDeps) -> None:
        super().__init__(deps)
        extra: dict[str, Any] = dict(deps.settings.model_extra or {})
        self._cli_setting: object | None = extra.get("cli")
        self._remote_root = str(extra.get("remote_root") or DEFAULT_REMOTE_ROOT).rstrip("/")
        self._max_bundle_mb = float(extra.get("max_bundle_mb") or DEFAULT_MAX_BUNDLE_MB)
        self._heartbeat_s = float(extra.get("heartbeat_s") or 60)
        # Invariant 20: in test mode the real CLI is never run unless a test points `cli`
        # at a simulator or GPU_ROUTER_REAL_PROVIDERS opts colab in.
        self._inert = deps.test_mode and self._cli_setting is None and not _real_opt_in()
        self.capabilities = Capabilities(
            lookup_by_key=True,
            resume=True,
            fetch=True,
            cancel_confirms=True,
            live_quota=False,
            max_session_hours=deps.entry.session_hours,
            max_concurrency=deps.entry.max_concurrency,
            max_bundle_mb=self._max_bundle_mb,
            poll_interval_s=deps.entry.poll_interval_s,
        )
        self._token = uuid.uuid4().hex
        self._guard = threading.Lock()
        self._locks: dict[str, threading.Lock] = {}
        self._inflight: set[str] = set()
        self._tmp_live: set[str] = set()  # temp files an in-progress call still needs
        self._janitor: threading.Thread | None = None
        self._janitor_wanted = False
        self._closed = threading.Event()
        # A daemon killed mid-submit leaves plaintext secrets in tmp/ (invariant 12).
        self._sweep_tmp()

    # ------------------------------------------------------------------ plumbing

    @property
    def store(self) -> RunStore:
        return RunStore(self.scratch_dir / "runs")

    @property
    def config_file(self) -> Path:
        """The dedicated `colab --config` session file (never ~/.config/colab-cli)."""
        return self.scratch_dir / "colab-cli" / "sessions.json"

    @property
    def cli_home(self) -> Path:
        """HOME for every CLI call: its colab.log (proxy tokens) and history (exec code and
        output) stay inside the 0700 data dir (D37)."""
        return self.scratch_dir / "cli-home"

    def run_dir_remote(self, session: str) -> str:
        return f"{self._remote_root}/{session}"

    def _lock(self, session: str) -> threading.Lock:
        with self._guard:
            lock = self._locks.get(session)
            if lock is None:
                lock = self._locks[session] = threading.Lock()
            return lock

    def _cli(self) -> ColabCli:
        prefix = resolve_cli(self._cli_setting)
        if prefix is None:
            raise AuthRequired(
                "the colab CLI is not installed", provider=self.name, hint=INSTALL_HINT
            )
        return ColabCli(prefix, self.config_file, home=self.cli_home)

    def _ref(self, rec: RunRecord) -> RemoteRef:
        meta = {
            "session": rec.session,
            "gpu": rec.gpu_requested,
            "remote_dir": self.run_dir_remote(rec.session),
        }
        return RemoteRef(remote_id=rec.session, meta=meta)

    def _load(self, ref: RemoteRef) -> RunRecord:
        rid = ref.remote_id
        rec = self.store.load(rid) if _SESSION_OK.match(rid) else None
        if rec is None:
            raise NotFound(f"colab has no run {rid!r}", provider=self.name)
        return rec

    def _tmp_dir(self) -> Path:
        d = self.scratch_dir / "tmp"
        d.mkdir(mode=0o700, parents=True, exist_ok=True)
        return d

    def _tmp_file(self, prefix: str, suffix: str) -> tuple[int, str]:
        """mkstemp in tmp/, registered as live until `_tmp_done` (so a sweep skips it)."""
        with self._guard:
            fd, path = tempfile.mkstemp(prefix=prefix, suffix=suffix, dir=self._tmp_dir())
            self._tmp_live.add(path)
        return fd, path

    def _tmp_done(self, path: str) -> None:
        with contextlib.suppress(OSError):
            os.unlink(path)
        with self._guard:
            self._tmp_live.discard(path)

    def _sweep_tmp(self) -> None:
        """Delete temp files no call of this process still uses: exec scripts and, above
        all, `.secrets-*.json` left by a daemon that died between writing and unlinking
        one (it holds plaintext secret values)."""
        d = self.paths.provider_dir(self.name) / "tmp"  # no mkdir: nothing to sweep then
        if not d.is_dir():
            return
        with self._guard:  # held throughout: _tmp_file creates + registers under it too
            for child in d.iterdir():
                if str(child) in self._tmp_live:
                    continue
                with contextlib.suppress(OSError):
                    if child.is_dir() and not child.is_symlink():
                        shutil.rmtree(child)
                    else:
                        child.unlink()

    def _save(self, rec: RunRecord) -> RunRecord:
        return self.store.save(rec)

    def _exec(
        self,
        cli: ColabCli,
        session: str,
        script: str,
        params: dict[str, Any],
        *,
        budget: _Budget,
        exec_timeout: float = EXEC_TIMEOUT_S,
    ) -> Any:
        """Run one remote script; returns its result. Raises SessionGone, an AdapterError
        (via classify), or Unavailable for unreadable output / a failing script."""
        source = remote.build(script, params)
        fd, path = self._tmp_file(f"{script}-", ".py")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(source)
            wall = budget.clip(exec_timeout + EXEC_SLACK_S)
            kernel_timeout = max(5.0, min(exec_timeout, wall - 5.0))
            res = cli.run(
                ["exec", "-s", session, "-f", path, "--timeout", f"{kernel_timeout:.0f}"],
                timeout=wall,
            )
        finally:
            self._tmp_done(path)
        if not res.ok:
            if session_gone(res):
                raise SessionGone(res.tail())
            raise self._classify(res, op=f"exec {script}")
        try:
            return remote.parse_result(res.stdout)
        except remote.RemoteScriptError as exc:
            raise Unavailable(
                f"colab {script} script failed on the VM: {exc.error}", provider=self.name
            ) from None
        except ValueError as exc:
            if session_gone(res):
                raise SessionGone(res.tail()) from None
            raise Unavailable(
                f"colab {script} gave no result ({exc}): {res.tail(300)}", provider=self.name
            ) from None

    def _transfer(self, cli: ColabCli, args: list[str], *, timeout: float, op: str) -> None:
        res = cli.run(args, timeout=timeout)
        if res.ok:
            return
        if session_gone(res):
            raise SessionGone(res.tail())
        raise self._classify(res, op=op)

    def _download(
        self, cli: ColabCli, session: str, remote_path: str, local: Path, *, timeout: float
    ) -> None:
        local.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        tmp = local.with_name(f".{local.name}.part")
        with contextlib.suppress(OSError):
            tmp.unlink()
        try:
            self._transfer(
                cli,
                ["download", "-s", session, remote_path, str(tmp)],
                timeout=timeout,
                op="download",
            )
            if not tmp.is_file():
                raise Unavailable(
                    f"colab download of {remote_path} wrote nothing", provider=self.name
                )
            os.replace(tmp, local)
        finally:
            with contextlib.suppress(OSError):
                tmp.unlink()

    def _upload(
        self, cli: ColabCli, session: str, local: Path, remote_path: str, *, timeout: float
    ) -> None:
        self._transfer(
            cli, ["upload", "-s", session, str(local), remote_path], timeout=timeout, op="upload"
        )

    def _stop(self, cli: ColabCli, session: str, *, timeout: float = STOP_TIMEOUT_S) -> None:
        """Stop one of OUR sessions by name. A session that is already gone counts as
        stopped. Raises an AdapterError when colab could not be asked."""
        res = cli.run(["stop", "-s", session], timeout=timeout)
        if res.ok:
            return
        text = res.text
        if session_gone(res) or "Not Found" in text or " 404" in text:
            return
        raise self._classify(res, op="stop")

    def _stop_quietly(self, cli: ColabCli | None, rec: RunRecord, *, timeout: float) -> bool:
        if rec.stopped:
            return True
        if cli is None:
            return False
        try:
            self._stop(cli, rec.session, timeout=timeout)
        except AdapterError as exc:
            rec.stop_error = exc.message[:300]
            _log.warning("colab: could not stop session %s: %s", rec.session, exc.message)
            return False
        rec.stopped = True
        rec.stop_error = None
        self._forget_history(rec.session)
        return True

    def _forget_history(self, session: str) -> None:
        """The CLI's history of one of OUR stopped sessions: exec code plus output (live
        log reads come back base64, so no redaction can see into them). gpu-router owns
        the gr-* names in its private CLI home; nothing else reads them (D37)."""
        path = self.cli_home / ".config" / "colab-cli" / "history" / f"{session}.jsonl"
        with contextlib.suppress(OSError):
            path.unlink()

    # ------------------------------------------------------------------ orphaned CLI calls

    def _note_cli_pid(self, rec: RunRecord, pid: int) -> None:
        rec.cli_pid = pid
        self._save(rec)

    @staticmethod
    def _pid_is_our_cli(pid: int, session: str) -> bool | None:
        """True: `pid` is alive and is a colab CLI call for `session`; False: it is gone
        or something else; None: alive but its command line could not be read."""
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return False  # another user's process: never ours
        except OSError:
            return None
        try:
            out = subprocess.run(
                ["ps", "-o", "command=", "-p", str(pid)],
                capture_output=True,
                text=True,
                timeout=PS_TIMEOUT_S,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if out.returncode != 0:
            return False if not out.stdout.strip() else None
        return session in out.stdout

    def _orphan_new(self, rec: RunRecord) -> bool | None:
        """Could a `colab new` for this record still be running outside this process?
        False = certainly not (it is done: whatever it created is registered by now);
        True = it is; None = maybe (no pid recorded, still inside the time window)."""
        if rec.state is not RunState.CREATING:
            return False  # `new` returned before the record left `creating`
        with self._guard:
            if rec.session in self._inflight:
                return True
        if rec.cli_pid is not None:
            alive = self._pid_is_our_cli(rec.cli_pid, rec.session)
            return True if alive is None else alive
        if self.clock.now() - rec.created_at < NEW_TIMEOUT_S + ORPHAN_GRACE_S:
            return None
        return False

    def _kill_orphan(self, rec: RunRecord) -> None:
        """Kill an orphaned `colab new` of ours (its own process group: start_new_session),
        only after its command line proved it is the call for this session."""
        pid = rec.cli_pid
        if pid is None or self._pid_is_our_cli(pid, rec.session) is not True:
            return
        with contextlib.suppress(OSError):
            if os.getpgid(pid) == pid:
                os.killpg(pid, signal.SIGKILL)
            else:
                os.kill(pid, signal.SIGKILL)
        _log.warning("colab: killed an interrupted `colab new` for %s (pid %s)", rec.session, pid)
        rec.cli_pid = None
        self._save(rec)

    # ------------------------------------------------------------------ submit

    def _pick_gpu(self, job: Job, ctx: AttemptContext) -> str:
        offered = {g.name.upper(): g.name for g in self.entry.gpus}
        allowed = [g for g in SUPPORTED_GPUS if g in offered]
        if not allowed:
            raise InvalidJob(
                "the colab catalog entry offers no GPU the colab CLI supports",
                provider=self.name,
                detail={"offered": sorted(offered)},
            )
        want = (ctx.gpu or job.spec.gpu or allowed[0]).strip().upper()
        if want not in allowed:
            raise InvalidJob(
                f"colab cannot provide a {ctx.gpu or job.spec.gpu} here "
                f"(this account gets: {', '.join(allowed)})",
                provider=self.name,
                detail={"requested": want, "allowed": allowed},
            )
        return want

    def _bundle(self, ctx: AttemptContext) -> Path:
        archive = ctx.bundle_archive
        if archive is None or not Path(archive).is_file():
            raise InvalidJob(
                "colab needs the job bundle archive and the attempt has none",
                provider=self.name,
                hint="resubmit the job so the daemon builds its bundle",
            )
        size_mb = Path(archive).stat().st_size / 1e6
        if size_mb > self._max_bundle_mb:
            raise InvalidJob(
                f"the job bundle is {size_mb:.0f} MB; colab uploads are capped at "
                f"{self._max_bundle_mb:.0f} MB",
                provider=self.name,
                hint="move large data to the Hugging Face Hub and read it from the job",
            )
        return Path(archive)

    def _resume_source(self, ctx: AttemptContext) -> Path | None:
        """A local file for ctx.resume_from: a Mac-side checkpoint (file:// that exists),
        or this adapter's mirror of an archive that lived on an earlier Colab VM."""
        ck = ctx.resume_from
        if ck is None:
            return None
        parsed = urlparse(ck.uri)
        if parsed.scheme not in ("file", ""):
            return None
        path = Path(unquote(parsed.path))
        if path.is_file():
            return path
        root = self._remote_root + "/"
        text = str(path)
        if text.startswith(root):
            parts = text[len(root) :].split("/")
            if len(parts) >= 2 and _SESSION_OK.match(parts[0]):
                mirror = self.store.run_dir(parts[0]) / "ckpt" / parts[-1]
                if mirror.is_file():
                    return mirror
        return None

    def submit(self, job: Job, ctx: AttemptContext) -> RemoteRef:
        if self._inert:
            raise InvalidJob(
                "colab is off in test mode",
                provider=self.name,
                hint=f"set {ENV_REAL_PROVIDERS}=colab to use the real colab",
            )
        session = session_name(ctx.attempt_key)
        gpu = self._pick_gpu(job, ctx)
        archive = self._bundle(ctx)
        with self._lock(session):
            rec = self.store.load(session)
            if rec is not None and rec.state not in STARTING_STATES | NO_RUN_STATES:
                return self._ref(rec)  # A4: same key, same run
            cli = self._cli()
            budget = _Budget(self, SUBMIT_BUDGET_S)
            self._sweep_tmp()
            if rec is not None and rec.state in STARTING_STATES:
                # A submit that died with the daemon: its half-built session is ours, but
                # its `colab new` may still be running (it survives the daemon, D35).
                if self._orphan_new(rec) is not False:
                    if self.clock.now() - rec.created_at < STALE_START_S:
                        raise Unavailable(
                            f"an interrupted submit of {session} may still be creating its "
                            "colab session; gpu-router checks again shortly",
                            provider=self.name,
                        )
                    self._kill_orphan(rec)
                self._stop_quietly(cli, rec, timeout=budget.clip(STOP_TIMEOUT_S))
            self._reap(cli, budget, exclude=session)
            rec = self._save(
                RunRecord(
                    session=session,
                    attempt_key=ctx.attempt_key,
                    job_id=job.id,
                    attempt_n=ctx.n,
                    state=RunState.CREATING,
                    owner=self._token,
                    gpu_requested=gpu,
                    created_at=self.clock.now(),
                )
            )
            with self._guard:
                self._inflight.add(session)
            try:
                return self._create_and_launch(cli, rec, ctx, gpu, archive, budget)
            finally:
                with self._guard:
                    self._inflight.discard(session)

    def _create_and_launch(
        self,
        cli: ColabCli,
        rec: RunRecord,
        ctx: AttemptContext,
        gpu: str,
        archive: Path,
        budget: _Budget,
    ) -> RemoteRef:
        session = rec.session
        res = cli.run(
            ["new", "--gpu", gpu, "-s", session],
            timeout=budget.clip(NEW_TIMEOUT_S),
            on_spawn=lambda pid: self._note_cli_pid(rec, pid),
        )
        rec.cli_pid = None  # returned (or killed at the timeout): no longer in flight
        if not res.ok:
            err = self._classify(res, op="new", gpu=gpu, now=self.clock.now())
            if isinstance(err, DEFINITIVE_SUBMIT_ERRORS) and not res.timed_out:
                rec.state = RunState.REJECTED
                rec.stopped = True  # nothing was created
                rec.message = err.message
                self._save(rec)
                if isinstance(err, QuotaExhausted):
                    self._note_refusal(gpu)
                raise err
            # Maybe created (timeout, unknown failure): stop our name, then say "unsure".
            self._stop_quietly(cli, rec, timeout=budget.clip(STOP_TIMEOUT_S))
            rec.state = RunState.ABANDONED
            rec.message = err.message
            self._save(rec)
            raise Unavailable(err.message, provider=self.name, hint=err.hint) from None

        rec.state = RunState.SETUP
        self._save(rec)
        with contextlib.suppress(OSError):
            os.chmod(self.config_file, 0o600)  # it holds runtime proxy tokens
        run_dir = self.run_dir_remote(session)
        try:
            prep = self._exec(cli, session, "prepare", {"run_dir": run_dir}, budget=budget)
            gpu_seen = prep.get("gpu") if isinstance(prep, dict) else None
            if not gpu_seen:
                raise _SetupFailed("colab gave a VM without a visible GPU (nvidia-smi: none)")
            rec.gpu_seen = str(gpu_seen)
            self._save(rec)
            if not budget.has(MIN_STEP_S * 2):
                raise _SetupFailed("ran out of time before the upload")
            size_mb = archive.stat().st_size / 1e6
            self._upload(
                cli,
                session,
                archive,
                f"{run_dir}/bundle.tar.gz",
                timeout=budget.clip(60 + size_mb * 4),
            )
            resume = False
            staged = str(ctx.env.get("GPU_RESUME_URI") or "")
            if ctx.resume_from is not None and staged.startswith("hf://"):
                # phase 5: the runner downloads it from the checkpoint bucket itself
                rec.resume = "storage"
            elif ctx.resume_from is not None:
                src = self._resume_source(ctx)
                if src is not None:
                    mb = src.stat().st_size / 1e6
                    self._upload(
                        cli,
                        session,
                        src,
                        f"{run_dir}/resume.tar.gz",
                        timeout=budget.clip(60 + mb * 4),
                    )
                    resume = True
                    rec.resume = "uploaded"
                else:
                    rec.resume = "unavailable"
                    _log.warning(
                        "colab: checkpoint %s is not reachable from colab; attempt %s starts fresh",
                        ctx.resume_from.id,
                        ctx.attempt_id,
                    )
            if ctx.secrets:
                self._upload_secrets(cli, session, ctx, run_dir, budget)
            params: dict[str, Any] = {
                "run_dir": run_dir,
                "env": dict(ctx.env),
                "ckpt_seq_start": (ctx.resume_from.seq + 1) if ctx.resume_from else 1,
                "heartbeat_s": self._heartbeat_s,
                "checkpoint_interval_min": ctx.checkpoint_interval_min,
                "resume": resume,
                "gpu": rec.gpu_seen,
            }
            self._exec(cli, session, "launch", params, budget=budget)
        except (AdapterError, SessionGone, _SetupFailed) as exc:
            why = exc.message if isinstance(exc, AdapterError) else str(exc)
            self._stop_quietly(cli, rec, timeout=max(5.0, min(STOP_TIMEOUT_S, budget.remaining())))
            rec.state = RunState.ABANDONED
            rec.message = why[:500]
            self._save(rec)
            raise Unavailable(
                f"colab session {session} could not be set up ({why}); it was stopped",
                provider=self.name,
            ) from None
        rec.state = RunState.LAUNCHED
        rec.launched_at = self.clock.now()
        self._save(rec)
        return self._ref(rec)

    def _upload_secrets(
        self, cli: ColabCli, session: str, ctx: AttemptContext, run_dir: str, budget: _Budget
    ) -> None:
        """Secrets go up as a 0600 file that launch reads and deletes: never argv, never
        the exec source (which the CLI records in its history)."""
        fd, path = self._tmp_file(".secrets-", ".json")
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump({k: v.get_secret_value() for k, v in ctx.secrets.items()}, fh)
            self._upload(
                cli, session, Path(path), f"{run_dir}/.secrets.json", timeout=budget.clip(60)
            )
        finally:
            self._tmp_done(path)  # a daemon killed right here: _sweep_tmp removes it

    def _reap(self, cli: ColabCli, budget: _Budget, *, exclude: str) -> None:
        """Stop sessions of ours that no run needs any more: finished runs whose teardown
        did not complete, and half-built sessions left by a crashed daemon. Also drops the
        local copies of runs that ended more than RETENTION_S ago (the engine keeps its own
        log and outputs copies)."""
        now = self.clock.now()
        for rec in self.store.all():
            if rec.session == exclude:
                continue
            if rec.stopped:
                if rec.final and now - (rec.ended_at or rec.created_at) > RETENTION_S:
                    shutil.rmtree(self.store.run_dir(rec.session), ignore_errors=True)
                continue
            stale_start = (
                rec.state in STARTING_STATES
                and rec.owner != self._token
                and now - rec.created_at > STALE_START_S
            )
            finished = rec.final and (
                not rec.outputs_pending
                or now - (rec.ended_at or rec.created_at) > PENDING_OUTPUTS_GIVE_UP_S
            )
            if not (stale_start or finished):
                continue
            if not budget.has(STOP_TIMEOUT_S + 60):
                return
            lock = self._lock(rec.session)
            if not lock.acquire(blocking=False):
                continue
            try:
                fresh = self.store.load(rec.session) or rec
                if fresh.stopped:
                    continue
                if stale_start:
                    self._kill_orphan(fresh)
                if self._stop_quietly(cli, fresh, timeout=STOP_TIMEOUT_S) and stale_start:
                    fresh.state = RunState.ABANDONED
                    fresh.message = "setup was interrupted by a daemon restart"
                self._save(fresh)
            finally:
                lock.release()
        self._sweep_local(now)

    # ------------------------------------------------------------------ status

    def status(self, ref: RemoteRef) -> RemoteStatus:
        rec = self._load(ref)
        if rec.state in STARTING_STATES:
            with self._guard:
                if rec.session in self._inflight:
                    return self._status_of(rec)
        if rec.state in NO_RUN_STATES or (rec.final and rec.stopped):
            return self._status_of(rec)
        budget = _Budget(self, STATUS_BUDGET_S)
        with self._lock(rec.session):
            rec = self.store.load(rec.session) or rec
            if rec.state in STARTING_STATES:
                st = self._resolve_stale_start(rec, budget)
            elif rec.state is RunState.LAUNCHED:
                st = self._poll(rec, budget)
            else:
                if rec.final and not rec.stopped:
                    self._settle(rec, budget, packed=None)
                st = self._status_of(rec)
        if rec.final and not rec.stopped:
            self._kick_janitor()  # D36: the engine never calls status() again after this
        return st

    def _status_of(self, rec: RunRecord) -> RemoteStatus:
        common: dict[str, Any] = {
            "gpu": short_gpu(rec.gpu_seen),
            "started_at": rec.launched_at,
            "ended_at": rec.ended_at,
        }
        st = rec.state
        if st in STARTING_STATES:
            return RemoteStatus(
                phase=RemotePhase.PENDING,
                message=f"starting a colab {rec.gpu_requested} session",
                gpu=None,
            )
        if st is RunState.LAUNCHED:
            return RemoteStatus(phase=RemotePhase.RUNNING, message="running on colab", **common)
        if st is RunState.EXITED:
            code = rec.exit_code
            if code == 0:
                return RemoteStatus(phase=RemotePhase.SUCCEEDED, exit_code=0, **common)
            if code == INSTALL_FAILED_EXIT:
                return RemoteStatus(
                    phase=RemotePhase.LOST,
                    lost_reason="dependency install failed on colab (exit 90)",
                    message="pip install failed on the colab VM",
                    **common,
                )
            if code in OOM_EXITS:
                return RemoteStatus(
                    phase=RemotePhase.LOST,
                    lost_reason="killed by the OS, likely out of RAM (colab T4 VMs have ~12 GB)",
                    **common,
                )
            return RemoteStatus(phase=RemotePhase.FAILED, exit_code=code, **common)
        if st is RunState.CANCELLED:
            return RemoteStatus(
                phase=RemotePhase.CANCELLED, message="colab session stopped", **common
            )
        reason = rec.lost_reason or rec.message or "the colab run did not start"
        return RemoteStatus(
            phase=RemotePhase.LOST,
            lost_reason=reason,
            quota_exhausted=rec.quota_exhausted,
            **common,
        )

    def _mark_lost(self, rec: RunRecord, reason: str, *, stopped: bool) -> None:
        rec.state = RunState.LOST
        rec.lost_reason = reason
        rec.ended_at = rec.ended_at or self.clock.now()
        rec.stopped = rec.stopped or stopped
        self._save(rec)

    def _resolve_stale_start(self, rec: RunRecord, budget: _Budget) -> RemoteStatus:
        """A creating/setup record with no live submit (daemon restarted mid-submit).

        A `colab new` that was running when the daemon died is still running (its own
        session) and registers the session only when the VM is assigned: until it is gone
        the attempt stays pending, and past STALE_START_S it is killed (D35)."""
        cli = self._cli()
        orphan = self._orphan_new(rec)
        if orphan is not False:
            if self.clock.now() - rec.created_at < STALE_START_S:
                return self._status_of(rec)  # pending: the interrupted `new` may still finish
            self._kill_orphan(rec)
        try:
            poll = self._exec(
                cli,
                rec.session,
                "poll",
                {"run_dir": self.run_dir_remote(rec.session), "pack": True},
                budget=budget,
            )
        except SessionGone:
            # Nothing answers to the name. Still ask colab to stop it (idempotent): "not
            # found" from the session store is final only once no `new` can add it.
            self._stop_quietly(cli, rec, timeout=budget.clip(STOP_TIMEOUT_S))
            self._mark_lost(
                rec, "setup was interrupted before the job started", stopped=rec.stopped
            )
            return self._status_of(rec)
        if isinstance(poll, dict) and poll.get("launched"):
            rec.state = RunState.LAUNCHED
            rec.launched_at = rec.launched_at or self.clock.now()
            self._save(rec)
            return self._apply_poll(cli, rec, poll, budget)
        self._stop_quietly(cli, rec, timeout=budget.clip(STOP_TIMEOUT_S))
        self._mark_lost(rec, "setup was interrupted before the job started", stopped=rec.stopped)
        return self._status_of(rec)

    def _poll(self, rec: RunRecord, budget: _Budget) -> RemoteStatus:
        cli = self._cli()
        try:
            poll = self._exec(
                cli,
                rec.session,
                "poll",
                {"run_dir": self.run_dir_remote(rec.session), "pack": True},
                budget=budget,
            )
        except SessionGone:
            self._mark_lost(rec, RECLAIMED, stopped=True)
            return self._status_of(rec)
        return self._apply_poll(cli, rec, poll, budget)

    def _apply_poll(
        self, cli: ColabCli, rec: RunRecord, poll: Any, budget: _Budget
    ) -> RemoteStatus:
        if not isinstance(poll, dict) or not poll.get("run_dir"):
            self._stop_quietly(cli, rec, timeout=budget.clip(STOP_TIMEOUT_S))
            self._mark_lost(
                rec, "the colab VM was reset (the run directory is gone)", stopped=rec.stopped
            )
            return self._status_of(rec)
        if not poll.get("launched"):
            self._stop_quietly(cli, rec, timeout=budget.clip(STOP_TIMEOUT_S))
            self._mark_lost(rec, "the runner never started on the colab VM", stopped=rec.stopped)
            return self._status_of(rec)
        exit_code = poll.get("exit")
        rc = poll.get("rc")
        if exit_code is None and rc is None and poll.get("alive"):
            self._mirror_checkpoint(cli, rec, poll.get("ckpt"), budget)
            return self._status_of(rec)
        code = exit_code if exit_code is not None else rc
        if code != 0:  # a success is never resumed; the log and stop come first (D36)
            self._mirror_checkpoint(
                cli, rec, poll.get("ckpt"), budget, reserve=MIN_STEP_S + STOP_RESERVE_S
            )
        rec.ended_at = self.clock.now()
        if code is None:
            rec.state = RunState.LOST
            rec.lost_reason = "the runner process on the colab VM vanished without an exit code"
        else:
            rec.state = RunState.EXITED
            rec.exit_code = int(code)
        self._save(rec)
        packed = poll.get("packed") if isinstance(poll.get("packed"), dict) else None
        self._settle(rec, budget, packed=packed, cli=cli)
        return self._status_of(rec)

    def _mirror_checkpoint(
        self, cli: ColabCli, rec: RunRecord, ckpt: Any, budget: _Budget, *, reserve: float = 0.0
    ) -> None:
        """Copy the newest small checkpoint archive to the Mac so a later attempt can
        resume after this VM is gone (phase 5 replaces this with the HF Hub). `reserve`:
        seconds of `budget` this step must leave for what follows (log pull + stop)."""
        if not isinstance(ckpt, dict):
            return
        name = str(ckpt.get("name") or "")
        size = int(ckpt.get("size") or 0)
        if not name or name == rec.ckpt_mirrored or size > MIRROR_CKPT_MAX_BYTES:
            return
        if not re.match(r"^ckpt-\d+\.tar\.gz$", name):
            return
        if not budget.has(MIN_STEP_S + 8 + reserve + size / 1e6):
            return
        dest_dir = self.store.run_dir(rec.session) / "ckpt"
        try:
            self._download(
                cli,
                rec.session,
                f"{self.run_dir_remote(rec.session)}/ckpt-sync/{name}",
                dest_dir / name,
                timeout=budget.clip(30 + size / 1e6 * 2, reserve=reserve),
            )
        except (AdapterError, SessionGone) as exc:
            _log.info("colab: checkpoint mirror of %s skipped: %s", name, exc)
            return
        rec.ckpt_mirrored = name
        self._save(rec)
        keep = sorted(p.name for p in dest_dir.glob("ckpt-*.tar.gz"))[-2:]
        for p in dest_dir.glob("ckpt-*.tar.gz"):
            if p.name not in keep:
                with contextlib.suppress(OSError):
                    p.unlink()

    # ------------------------------------------------------------------ teardown

    def _settle(
        self,
        rec: RunRecord,
        budget: _Budget,
        *,
        packed: dict[str, Any] | None,
        cli: ColabCli | None = None,
    ) -> None:
        """Finish a final run's teardown within `budget`: cache the log, cache a successful
        run's outputs when small, then stop the session. Every harvest step leaves
        STOP_RESERVE_S for the stop. Never raises: what does not fit is finished by the
        next status/logs/fetch/cancel call, the janitor or the reaper."""
        if rec.stopped:
            return
        if cli is None:
            try:
                cli = self._cli()
            except AdapterError:
                return
        run_dir = self.run_dir_remote(rec.session)
        wants_outputs = rec.state is RunState.EXITED and rec.exit_code == 0
        step = MIN_STEP_S + STOP_RESERVE_S
        try:
            need_out = wants_outputs and not rec.outputs_cached and not rec.outputs_pending
            if (not rec.log_cached or need_out) and rec.settle_tries < MAX_SETTLE_TRIES:
                if packed is None:
                    if not budget.has(step):
                        return
                    rec.settle_tries += 1  # counted only when a harvest really starts
                    self._save(rec)
                    got = self._exec(
                        cli,
                        rec.session,
                        "pack",
                        {"run_dir": run_dir, "outputs": wants_outputs and not rec.outputs_cached},
                        budget=budget.sub(max(1.0, budget.remaining() - STOP_RESERVE_S)),
                    )
                    packed = got if isinstance(got, dict) else {}
                if not rec.log_cached:
                    if "log_gz_bytes" in packed:
                        if not budget.has(step):
                            return
                        self._pull_log(cli, rec, packed, budget)
                    else:
                        self._write_local_log(rec, b"")  # the runner never wrote a log
                        rec.log_cached = True
                    self._save(rec)
                if need_out:
                    size = packed.get("outputs_archive_bytes")
                    if (
                        isinstance(size, int)
                        and size <= EAGER_OUTPUT_MAX_BYTES
                        and budget.has(step + size / 1e6)
                    ):
                        self._pull_outputs(cli, rec, packed, budget)
                    else:
                        rec.outputs_pending = True
                        self._save(rec)
            if (
                wants_outputs
                and not rec.outputs_cached
                and (rec.outputs_pending or rec.settle_tries < MAX_SETTLE_TRIES)
            ):
                return  # fetch() pulls them and then stops the session
            if not budget.has(8):
                return
            self._stop_quietly(cli, rec, timeout=budget.clip(STOP_TIMEOUT_S))
            self._save(rec)
        except SessionGone:
            rec.stopped = True
            self._save(rec)
            self._forget_history(rec.session)
        except AdapterError as exc:
            _log.info("colab: teardown of %s postponed: %s", rec.session, exc.message)

    def _write_local_log(self, rec: RunRecord, data: bytes) -> None:
        path = self.store.run_dir(rec.session) / "job.log"
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        tmp = path.with_name(".job.log.part")
        tmp.write_bytes(data)
        os.replace(tmp, path)

    def _pull_log(
        self, cli: ColabCli, rec: RunRecord, packed: dict[str, Any], budget: _Budget
    ) -> None:
        gz_bytes = int(packed.get("log_gz_bytes") or 0)
        local_gz = self.store.run_dir(rec.session) / "job.log.gz"
        self._download(
            cli,
            rec.session,
            f"{self.run_dir_remote(rec.session)}/job.log.gz",
            local_gz,
            timeout=budget.clip(30 + gz_bytes / 1e6 * 2, reserve=STOP_RESERVE_S),
        )
        try:
            data = gzip.decompress(local_gz.read_bytes())
        except (OSError, EOFError) as exc:
            raise Unavailable(f"downloaded colab log is corrupt: {exc}") from None
        finally:
            with contextlib.suppress(OSError):
                local_gz.unlink()
        self._write_local_log(rec, data)
        rec.log_cached = True

    def _pull_outputs(
        self, cli: ColabCli, rec: RunRecord, packed: dict[str, Any], budget: _Budget
    ) -> None:
        size = int(packed.get("outputs_archive_bytes") or 0)
        self._download(
            cli,
            rec.session,
            f"{self.run_dir_remote(rec.session)}/outputs.tar.gz",
            self.store.run_dir(rec.session) / "outputs.tar.gz",
            timeout=budget.clip(30 + size / 1e6 * 2, reserve=STOP_RESERVE_S),
        )
        rec.outputs_cached = True
        rec.outputs_pending = False
        rec.outputs_files = packed.get("outputs_files")
        rec.outputs_bytes = packed.get("outputs_bytes")
        self._save(rec)

    # ------------------------------------------------------------------ logs

    def logs(
        self, ref: RemoteRef, *, follow: bool = False, since: str | None = None
    ) -> Iterator[LogChunk]:
        # live_logs=False: follow=True is served like follow=False (one snapshot).
        # Chunks are computed first and any pending teardown runs BEFORE they are yielded:
        # the engine stops iterating after max_lines, and code after a yield would not run.
        rec = self._load(ref)
        lines0, off0 = _parse_cursor(since)
        budget = _Budget(self, LOGS_BUDGET_S)
        if rec.log_cached:
            chunks = self._local_chunks(rec, lines0, off0)
        elif rec.state in STARTING_STATES or rec.state in NO_RUN_STATES or rec.stopped:
            chunks = [LogChunk(lines=[], cursor=_cursor(lines0, off0), eof=rec.final)]
        else:
            chunks = self._remote_chunks(rec, lines0, off0, budget)
        if rec.final and (not rec.stopped or (chunks[-1].eof and rec.log_served_at is None)):
            with self._lock(rec.session):
                fresh = self.store.load(rec.session) or rec
                if not fresh.stopped:
                    self._settle(fresh, budget, packed=None)  # D36: logs() finishes teardown
                if chunks[-1].eof and fresh.log_cached and fresh.log_served_at is None:
                    fresh.log_served_at = self.clock.now()  # starts the raw copy's purge clock
                    self._save(fresh)
                    self._kick_janitor()  # purges it RAW_LOG_GRACE_S from now
                rec = fresh
        if rec.final and not rec.stopped:
            self._kick_janitor()
        yield from chunks

    def _local_chunks(self, rec: RunRecord, lines0: int, off0: int) -> list[LogChunk]:
        path = self.store.run_dir(rec.session) / "job.log"
        try:
            with path.open("rb") as fh:
                fh.seek(off0)
                data = fh.read()
        except OSError:
            data = b""
        return self._chunk(data, lines0, off0, final=rec.final, eof_when_done=rec.final)

    @staticmethod
    def _chunk(
        data: bytes, lines0: int, off0: int, *, final: bool, eof_when_done: bool
    ) -> list[LogChunk]:
        """Split `data` (the bytes after offset off0) into LogChunks of CHUNK_LINES lines."""
        pairs = remote.split_log_bytes(data, final=final)
        out: list[LogChunk] = []
        for i in range(0, len(pairs), CHUNK_LINES):
            group = pairs[i : i + CHUNK_LINES]
            out.append(
                LogChunk(
                    lines=[line for line, _ in group],
                    cursor=_cursor(lines0 + i + len(group), off0 + group[-1][1]),
                )
            )
        if not out:
            return [LogChunk(lines=[], cursor=_cursor(lines0, off0), eof=eof_when_done)]
        done = eof_when_done and pairs[-1][1] >= len(data)
        last = out[-1]
        out[-1] = LogChunk(lines=last.lines, cursor=last.cursor, eof=done)
        return out

    def _remote_chunks(
        self, rec: RunRecord, lines0: int, off0: int, budget: _Budget
    ) -> list[LogChunk]:
        """Read job.log straight from the VM while the session is up (running, or ended
        but not yet harvested)."""
        try:
            cli = self._cli()
        except AdapterError:
            return [LogChunk(lines=[], cursor=_cursor(lines0, off0), eof=False)]
        out: list[LogChunk] = []
        n_lines, off = lines0, off0
        reached_end = False
        path = f"{self.run_dir_remote(rec.session)}/job.log"
        while budget.has():
            try:
                got = self._exec(
                    cli,
                    rec.session,
                    "logread",
                    {"path": path, "byte_offset": off, "max_bytes": LOG_READ_BYTES},
                    budget=budget,
                )
            except (AdapterError, SessionGone) as exc:
                _log.debug("colab: log read for %s failed: %s", rec.session, exc)
                break
            if not isinstance(got, dict) or not got.get("exists"):
                reached_end = True  # no log file (yet): nothing more to read now
                break
            data = base64.b64decode(got.get("data") or "")
            size = int(got.get("size") or 0)
            at_end = off + len(data) >= size
            chunks = self._chunk(
                data, n_lines, off, final=rec.final and at_end, eof_when_done=False
            )
            for c in chunks:
                if c.lines:
                    out.append(c)
            last_lines, last_off = _parse_cursor(chunks[-1].cursor)
            advanced = last_off > off
            n_lines, off = last_lines, last_off
            if at_end:
                reached_end = True
                break
            if not advanced:
                break
        eof = rec.final and reached_end
        if not out:
            return [LogChunk(lines=[], cursor=_cursor(n_lines, off), eof=eof)]
        last = out[-1]
        out[-1] = LogChunk(lines=last.lines, cursor=last.cursor, eof=eof)
        return out

    # ------------------------------------------------------------------ fetch

    def fetch(self, ref: RemoteRef, dest: Path) -> FetchResult:
        rec = self._load(ref)
        if rec.state is not RunState.EXITED or rec.exit_code != 0:
            raise NotFound(
                f"colab run {rec.session} has no outputs (it did not succeed)",
                provider=self.name,
            )
        with self._lock(rec.session):
            rec = self.store.load(rec.session) or rec
            budget = _Budget(self, FETCH_BUDGET_S)
            cli: ColabCli | None = None
            if not rec.outputs_cached:
                if rec.stopped:
                    raise NotFound(
                        f"the outputs of {rec.session} are gone: its colab session ended "
                        "before they were downloaded",
                        provider=self.name,
                    )
                cli = self._cli()
                try:
                    packed = self._exec(
                        cli,
                        rec.session,
                        "pack",
                        {"run_dir": self.run_dir_remote(rec.session), "outputs": True},
                        budget=budget,
                        exec_timeout=300,
                    )
                    self._pull_outputs(cli, rec, packed if isinstance(packed, dict) else {}, budget)
                except SessionGone:
                    rec.stopped = True
                    self._save(rec)
                    self._forget_history(rec.session)
                    raise NotFound(
                        f"the outputs of {rec.session} are gone: its colab session ended",
                        provider=self.name,
                    ) from None
            if not rec.stopped:  # D36: whatever the cached flags say, finish the teardown
                self._settle(rec, budget, packed=None, cli=cli)
            if not rec.stopped:
                self._kick_janitor()
        return self._extract_outputs(rec, dest)

    def _extract_outputs(self, rec: RunRecord, dest: Path) -> FetchResult:
        archive = self.store.run_dir(rec.session) / "outputs.tar.gz"
        dest.mkdir(parents=True, exist_ok=True)
        files = 0
        total = 0
        try:
            with tarfile.open(archive, "r:gz") as tf:
                members = [m for m in tf.getmembers() if m.isfile()]
                for m in members:
                    files += 1
                    total += m.size
                tf.extractall(dest, members=members, filter="data")
        except (OSError, tarfile.TarError) as exc:
            raise Unavailable(
                f"could not unpack the outputs of {rec.session}: {exc}", provider=self.name
            ) from None
        return FetchResult(dest=dest, files=files, bytes=total)

    # ------------------------------------------------------------------ cancel

    def cancel(self, ref: RemoteRef) -> None:
        try:
            rec = self._load(ref)
        except NotFound:
            return  # A5
        with self._lock(rec.session):
            rec = self.store.load(rec.session) or rec
            if not rec.stopped:
                budget = _Budget(self, CANCEL_BUDGET_S)
                cli = self._cli()
                if rec.state is RunState.LAUNCHED:
                    # The runner may have exited since the last poll (polls are 30 s apart):
                    # harvest it instead of destroying its log and outputs with the VM.
                    self._harvest_before_cancel(cli, rec, budget)
                if rec.state is RunState.EXITED and not rec.stopped:
                    # finished before the cancel landed: keep what it produced if we can
                    self._settle(rec, budget.sub(budget.remaining() - 15), packed=None, cli=cli)
                if rec.state in STARTING_STATES:
                    self._kill_orphan(rec)  # an interrupted `colab new` must not finish later
                keep_for_fetch = (
                    rec.state is RunState.EXITED and rec.exit_code == 0 and not rec.outputs_cached
                )
                if not rec.stopped and not keep_for_fetch:
                    # (a finished success keeps its session for fetch(), which then stops it;
                    # the janitor gives up on it after PENDING_OUTPUTS_GIVE_UP_S)
                    self._stop(cli, rec.session, timeout=budget.clip(STOP_TIMEOUT_S))
                    rec.stopped = True
                    self._forget_history(rec.session)
            if rec.state in STARTING_STATES or rec.state is RunState.LAUNCHED:
                rec.state = RunState.CANCELLED
                rec.ended_at = self.clock.now()
            self._save(rec)
            if rec.final and not rec.stopped:
                self._kick_janitor()

    def _harvest_before_cancel(self, cli: ColabCli, rec: RunRecord, budget: _Budget) -> None:
        """One `poll` of a launched run; if the runner already exited, record that and
        settle it (log, outputs, stop) like a status() would have."""
        look = budget.sub(min(CANCEL_POLL_S, budget.remaining() - STOP_RESERVE_S))
        try:
            poll = self._exec(
                cli,
                rec.session,
                "poll",
                {"run_dir": self.run_dir_remote(rec.session), "pack": True},
                budget=look,
            )
        except SessionGone:
            rec.stopped = True  # the VM is gone already: nothing left to stop or keep
            self._forget_history(rec.session)
            return
        except AdapterError as exc:
            _log.info("colab: pre-cancel look at %s failed (%s); stopping it", rec.session, exc)
            return
        if not isinstance(poll, dict) or not poll.get("run_dir") or not poll.get("launched"):
            return
        ended = poll.get("exit") is not None or poll.get("rc") is not None or not poll.get("alive")
        if ended:
            self._apply_poll(cli, rec, poll, budget.sub(budget.remaining() - STOP_RESERVE_S))

    # ------------------------------------------------------------------ lookup

    def lookup_by_key(self, attempt_key: str) -> RemoteRef | None:
        rec = self.store.load(session_name(attempt_key))
        if rec is None or rec.state in NO_RUN_STATES:
            return None
        return self._ref(rec)

    # ------------------------------------------------------------------ quota

    def _refusal_file(self) -> Path:
        return self.scratch_dir / "quota.json"

    def _note_refusal(self, gpu: str) -> None:
        data = {"refused_at": self.clock.now(), "gpu": gpu}
        path = self._refusal_file()
        tmp = path.with_name(".quota.json.part")
        tmp.write_text(json.dumps(data), encoding="utf-8")
        os.replace(tmp, path)

    def _last_refusal(self) -> float | None:
        try:
            data = json.loads(self._refusal_file().read_text(encoding="utf-8"))
            return float(data["refused_at"])
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def quota(self) -> QuotaSnapshot:
        """Colab does not publish or expose free GPU limits (NOTES.md "Quota"): an estimate
        from this adapter's own runs (GPU time in the last 24 h) plus the last refusal."""
        now = self.clock.now()
        start = now - QUOTA_WINDOW_S
        used_s = 0.0
        runs = 0
        for rec in self.store.all():
            if rec.launched_at is None:
                continue
            end = rec.ended_at if rec.ended_at is not None else now
            span = min(end, now) - max(rec.launched_at, start)
            if span > 0:
                used_s += span
                runs += 1
        refused = self._last_refusal()
        resets_at = None
        detail: dict[str, Any] = {"window_hours": QUOTA_WINDOW_S / 3600, "runs": runs}
        if refused is not None and refused + QUOTA_RESET_S > now:
            resets_at = refused + QUOTA_RESET_S
            detail["refused_at"] = refused
        return QuotaSnapshot(
            provider=self.name,
            used=used_s / 3600,
            limit=self.entry.quota.limit,
            unit=self.entry.quota.unit,
            resets_at=resets_at,
            source="estimate",
            detail=detail,
            observed_at=now,
        )

    # ------------------------------------------------------------------ health

    def healthcheck(self) -> Health:
        now = self.clock.now()
        if self._inert:
            return Health(
                health=ProviderHealth.DISABLED,
                reason="colab is off in test mode",
                hint=f"set {ENV_REAL_PROVIDERS}=colab to use the real colab",
                checked_at=now,
            )
        # Local hygiene only (no remote call): runs once at daemon start, so leftovers of a
        # crash (tmp secrets, unfinished teardowns) are handled without a new Colab job.
        self._sweep_local(now)
        if self._janitor_has_work():
            self._kick_janitor()
        prefix = resolve_cli(self._cli_setting)
        if prefix is None:
            return Health(
                health=ProviderHealth.UNAVAILABLE,
                reason="the colab CLI is not installed",
                hint=INSTALL_HINT,
                checked_at=now,
            )
        if self._cli_setting is None and not self._adc_present():
            return Health(
                health=ProviderHealth.AUTH_REQUIRED,
                reason="no Google application-default credentials on this Mac",
                hint=ADC_HINT,
                checked_at=now,
            )
        res = ColabCli(prefix, self.config_file, home=self.cli_home).run(
            ["sessions"], timeout=HEALTH_TIMEOUT_S
        )
        if res.returncode == 127:
            return Health(
                health=ProviderHealth.UNAVAILABLE,
                reason="the colab CLI is not installed",
                hint=INSTALL_HINT,
                checked_at=now,
            )
        if not res.ok or "No valid default credentials" in res.text:
            err = self._classify(res, op="sessions")
            if isinstance(err, AuthRequired) and "No valid default credentials" in res.text:
                err = AuthRequired("no valid Google application-default credentials", hint=ADC_HINT)
            health = (
                ProviderHealth.AUTH_REQUIRED
                if isinstance(err, AuthRequired)
                else ProviderHealth.UNAVAILABLE
            )
            return Health(health=health, reason=err.message, hint=err.hint, checked_at=now)
        ours, others = self._parse_sessions(res)
        detail = {"sessions": len(ours), "other_sessions": len(others)}
        other_gpus = [hw for _, hw in others if hw.upper() not in ("CPU", "NONE", "")]
        if other_gpus and not ours:
            return Health(
                health=ProviderHealth.DEGRADED,
                reason=(
                    f"{len(other_gpus)} other colab GPU session(s) are running on this account "
                    "(the free tier usually allows one)"
                ),
                hint="`colab --auth=adc sessions` lists them; gpu-router never stops them",
                checked_at=now,
                detail=detail,
            )
        return Health(health=ProviderHealth.OK, checked_at=now, detail=detail)

    def _classify(self, res: CliResult, *, op: str, **kw: Any) -> AdapterError:
        """`classify` with the reachability probe: an auth-looking failure while Google's
        token endpoint is unreachable is an outage, not a login problem."""
        return classify(res, provider=self.name, op=op, reachable=self._google_reachable, **kw)

    @staticmethod
    def _google_reachable() -> bool:
        return colab_cli.google_reachable()  # looked up per call: tests replace it

    @staticmethod
    def _adc_present() -> bool:
        explicit = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if explicit:
            return Path(explicit).is_file()
        return ADC_FILE.is_file()

    def _parse_sessions(self, res: CliResult) -> tuple[list[str], list[tuple[str, str]]]:
        ours: list[str] = []
        others: list[tuple[str, str]] = []
        for line in res.stdout.splitlines():
            m = _SESSIONS_LINE.match(line.strip())
            if not m:
                continue
            name, hw = m.group("name"), m.group("hw").strip()
            if name != "?" and self.store.load(name) is not None:
                ours.append(name)
            else:
                others.append((name, hw))
        return ours, others

    def close(self) -> None:
        """Stops the janitor thread; remote runs are never touched on shutdown (invariant
        11: a janitor only ever stops sessions whose runs are over)."""
        self._closed.set()

    # ------------------------------------------------------------------ janitor (D36)

    def _needs_teardown(self, rec: RunRecord) -> bool:
        if rec.stopped:
            return False
        return rec.final or (rec.state in STARTING_STATES and rec.owner != self._token)

    def _janitor_has_work(self) -> bool:
        now = self.clock.now()
        for rec in self.store.all():
            if self._needs_teardown(rec) or self._raw_log_pending(rec, now) is not None:
                return True
        return False

    def _kick_janitor(self) -> None:
        """Make sure the janitor thread runs: it retries teardowns no call finished (a
        failed `colab stop`, a short budget) and purges raw log copies, every
        JANITOR_INTERVAL_S, until nothing is left. It never waits for another Colab job."""
        if self._inert or self._closed.is_set() or not JANITOR_ENABLED:
            return
        with self._guard:
            self._janitor_wanted = True
            if self._janitor is not None and self._janitor.is_alive():
                return
            thread = threading.Thread(
                target=self._janitor_loop, name=f"gpu-router-{self.name}-janitor", daemon=True
            )
            self._janitor = thread
        thread.start()

    def _janitor_loop(self) -> None:
        while not self._closed.wait(JANITOR_INTERVAL_S):
            with self._guard:
                self._janitor_wanted = False
            try:
                more = self._janitor_pass()
            except Exception:  # never let the thread die with work left
                _log.warning("colab: janitor pass failed; retrying later", exc_info=True)
                more = self.store.root.is_dir()
            if more:
                continue
            with self._guard:
                if not self._janitor_wanted:
                    self._janitor = None
                    return

    def _janitor_pass(self) -> bool:
        """One round over our records. Returns True while work remains."""
        now = self.clock.now()
        busy = False
        cli: ColabCli | None = None
        for rec in self.store.all():
            if not self._needs_teardown(rec):
                continue
            age = now - (rec.ended_at or rec.created_at)
            if rec.state in STARTING_STATES and now - rec.created_at <= STALE_START_S:
                busy = True  # the engine (or an interrupted `colab new`) still owns it
                continue
            if (
                rec.final
                and age <= PENDING_OUTPUTS_GIVE_UP_S
                and rec.outputs_pending
                and not rec.outputs_cached
            ):
                busy = True  # waiting for fetch(), which pulls them and then stops
                continue
            lock = self._lock(rec.session)
            if not lock.acquire(blocking=False):
                busy = True
                continue
            try:
                fresh = self.store.load(rec.session) or rec
                if not self._needs_teardown(fresh):
                    continue
                if age > JANITOR_GIVE_UP_S:
                    fresh.stopped = True  # colab's 12 h cap has ended that VM by now
                    fresh.stop_error = "gave up stopping it; the colab session cap ended it"
                    self._save(fresh)
                    continue
                if cli is None:
                    cli = self._cli()
                budget = _Budget(self, STATUS_BUDGET_S)
                if fresh.state in STARTING_STATES:
                    self._kill_orphan(fresh)
                    if self._stop_quietly(cli, fresh, timeout=budget.clip(STOP_TIMEOUT_S)):
                        fresh.state = RunState.ABANDONED
                        fresh.message = "setup was interrupted by a daemon restart"
                    self._save(fresh)
                elif fresh.outputs_pending and not fresh.outputs_cached:
                    self._stop_quietly(cli, fresh, timeout=budget.clip(STOP_TIMEOUT_S))
                    self._save(fresh)  # nobody fetched them in PENDING_OUTPUTS_GIVE_UP_S
                else:
                    self._settle(fresh, budget, packed=None, cli=cli)
                busy = busy or not fresh.stopped
            finally:
                lock.release()
        return self._sweep_local(now) or busy

    # ------------------------------------------------------------------ local hygiene

    def _raw_log_pending(self, rec: RunRecord, now: float) -> bool | None:
        """None: nothing to purge; False: purge now; True: purge later."""
        if not (rec.stopped and rec.log_cached and not rec.log_purged):
            return None
        if rec.log_served_at is None:
            return None  # never fully served: kept until RETENTION_S drops the run dir
        return now - rec.log_served_at < RAW_LOG_GRACE_S

    def _sweep_local(self, now: float) -> bool:
        """Local copies that must not outlive their use (D37): the raw harvested job.log
        once logs() served it to eof + RAW_LOG_GRACE_S (the engine keeps a redacted copy),
        the CLI's history files of stopped sessions, an oversized private colab.log and
        leftover temp files. Returns True while a purge is still due later."""
        later = False
        known: dict[str, RunRecord] = {}
        for rec in self.store.all():
            known[rec.session] = rec
            if rec.stopped:
                self._forget_history(rec.session)
            due = self._raw_log_pending(rec, now)
            if due is None:
                continue
            if due:
                later = True
                continue
            lock = self._lock(rec.session)
            if not lock.acquire(blocking=False):
                later = True
                continue
            try:
                fresh = self.store.load(rec.session) or rec
                with contextlib.suppress(OSError):
                    (self.store.run_dir(fresh.session) / "job.log").unlink()
                fresh.log_purged = True
                self._save(fresh)
            finally:
                lock.release()
        cli_dir = self.cli_home / ".config" / "colab-cli"
        hist = cli_dir / "history"
        if hist.is_dir():
            for f in hist.glob("gr-*.jsonl"):
                owner = known.get(f.stem)
                if owner is None or owner.stopped:
                    with contextlib.suppress(OSError):
                        f.unlink()
        cli_log = cli_dir / "colab.log"
        with contextlib.suppress(OSError):
            if cli_log.stat().st_size > CLI_LOG_MAX_BYTES:
                with cli_log.open("w"):
                    pass
        self._sweep_tmp()
        return later


__all__ = ["ColabAdapter", "session_name", "short_gpu"]
