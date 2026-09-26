"""Remote entrypoint for one job attempt. STDLIB ONLY, Python 3.8+ (invariant 14, D2).

Adapters ship the bundle (see gpu_router/packaging) and run:

    python gpu_runner/bootstrap.py                         # from an extracted bundle
    python bootstrap.py --bundle bundle.tar.gz --workdir /kaggle/working/job

Steps:
 1. remove a stale EXIT file, rotate job.log, print `::gpu:: hello`; unpack the bundle (safe
    extraction; re-extracted into a clean dir when the archive's sha256 differs from the
    one stamped on the extracted copy) unless it is a directory
 2. read manifest.json; claim the checkpoint/output dirs for this job (a dir stamped with
    another GPU_ROUTER_JOB_ID is emptied first)
 3. restore the resume checkpoint (--resume, tar.gz or dir) into a clean GPU_RESUME_DIR and
    seed an empty GPU_CHECKPOINT_DIR with COPIES of it (never hard links: an in-place
    torch.save would truncate every linked copy, including the source checkpoint)
 4. install deps (pip) unless --skip-install / GPU_SKIP_INSTALL=1; on failure print
    `::gpu:: {"t":"install_failed","code":n}` and exit INSTALL_FAILED_EXIT (the job never
    started; adapters treat it as an environment failure, not a script failure)
 5. run the entrypoint from code/ in its own process group with the `gpu` helper on
    PYTHONPATH; stdout+stderr are merged and teed to our stdout and the log file. A
    protocol line glued behind a tqdm `\\r` segment is split back onto its own line. When the
    entrypoint exits, leftover processes that still hold the pipe (tensorboard, workers)
    get a short grace period, then the whole group is killed
 6. while it runs: a heartbeat line every --heartbeat-s; every checkpoint_interval_min, if
    GPU_CHECKPOINT_DIR changed, publish it (on its own thread, so heartbeats keep ticking)
    and print ckpt_begin / ckpt_end. With --storage (GPU_STORAGE, phase 5) the checkpoint
    goes to checkpoint storage (storage.py: an HF Storage Bucket, or a directory for the
    local provider) as `jobs/<job>/ckpt-NNNN/`; without it, as a tar.gz archive in
    --checkpoint-sync-dir (phase 2 behaviour)
 7. print `::gpu:: exit`, write the exit code to the EXIT file, exit with it. EXIT is
    written for every outcome, including bad arguments and SIGTERM/SIGINT at any step
    (code 128+n)

Checkpoint sync never publishes a torn checkpoint: files must be unchanged for STABLE_S
before they are archived (a sync is postponed while any file is still being written, up to
MAX_POSTPONE_S), the dir is fingerprinted again after archiving (or staging, for storage)
and the copy is dropped if anything changed, and after a non-zero exit the final sync is
skipped when files changed within UNSAFE_WINDOW_S of the end (a kill mid-save). Symlinks
are archived as the files they point to. Only the newest KEEP_ARCHIVES archives (storage:
checkpoints) are kept.

Checkpoint storage (phase 5, all optional, set by the engine through the adapter):
  GPU_STORAGE           storage root: hf://buckets/<ns>/<name> or file:///dir
  GPU_STORAGE_TOKEN     token for an hf:// root (a secret: removed from the environment
                        before anything else runs, never passed to the job or pip)
  GPU_RESUME_URI        checkpoint to restore when --resume / GPU_RESUME_SRC is not given
                        (hf://.../ckpt-NNNN or file://...); a missing one = fresh start
  GPU_STATUS_PUSH_S     push heartbeat.json + log-tail.json to storage this often (60)
  GPU_CONTROL_POLL_S    check control.json (the daemon's checkpoint request) this often (60)
  GPU_CKPT_KEEP         checkpoints kept in storage (3)
  GPU_DATA              JSON list of datasets to put under GPU_DATA_DIR/<mount> first
  GPU_SECRET_NAMES      comma list of env names whose values are masked in log-tail.json
  GPU_CHECKPOINT_INTERVAL_S  seconds between syncs (tests/dev; overrides the minutes)
  GPU_RUNNER_CACHE      where huggingface_hub is pip-installed (--target) when the image
                        lacks a new enough one (default <workdir>/.gpu-cache)
A checkpoint request (control.json {"id", "action": "checkpoint"|"handoff", "wait_s"})
creates `<checkpoint dir>/.gpu-checkpoint-request` (gpu.checkpoint_requested() is then
True), waits up to wait_s for a new checkpoint to settle, syncs, and answers in
control-ack.json. After a "handoff" answer the runner publishes nothing more: the daemon
stops this attempt and resumes the job elsewhere from the acknowledged checkpoint. Before
each publish the runner checks owner.json: an attempt the daemon started later owns the
job, and an older runner that is somehow still alive stops publishing.

Heartbeat and install_failed lines use types that protocol.parse_line ignores by design
(unknown types are skipped), so they show up in raw logs without changing the protocol.
Right after hello, `::gpu:: {"t":"device","gpus":["Tesla T4, 15360 MiB"]}` reports what
nvidia-smi sees (nothing when there is no nvidia-smi); the daemon notes a GPU other than
the one the job was placed on (D56).
"""

import argparse
import collections
import contextlib
import hashlib
import importlib
import importlib.util
import json
import os
import re
import select
import shutil
import signal
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path

RUNNER_VERSION = "bootstrap/0.3"
PREFIX = "::gpu:: "  # must equal gpu_router.protocol.PREFIX
PROTOCOL_VERSION = 1
MANIFEST = "manifest.json"
PROGRESS_FLUSH_S = 5.0  # a '\r'-only progress update (tqdm) is logged at most this often
INSTALL_FAILED_EXIT = 90  # reserved: dependency install failed, the entrypoint never ran
STABLE_S = 2.0  # a checkpoint file must be unchanged this long before it is archived
MAX_POSTPONE_S = 120.0  # then archive anyway; the post-archive fingerprint check still guards
UNSAFE_WINDOW_S = 10.0  # after a non-zero exit, files this fresh may be half-written
KEEP_ARCHIVES = 3  # checkpoint archives kept in the sync dir
DRAIN_GRACE_S = 2.0  # output drained after the entrypoint exits, before its group is killed
SMI_TIMEOUT_S = 15.0  # nvidia-smi for the device line; a hung driver never delays the job more
MAX_DEVICES = 16
ARCHIVE_RE = re.compile(r"^ckpt-(\d+)\.tar\.gz$")
PRIVATE_PREFIX = ".gpu-"  # runner/helper bookkeeping inside user dirs; never archived
JOB_STAMP = ".gpu-router-job"  # sibling stamp: .<dir name>.gpu-router-job
BUNDLE_STAMP = ".gpu-bundle-sha256"
REQUEST_FILE = ".gpu-checkpoint-request"  # in the checkpoint dir; gpu.checkpoint_requested()
STATUS_PUSH_S = 60.0  # heartbeat.json + log-tail.json cadence when storage is set
CONTROL_POLL_S = 60.0  # control.json cadence when storage is set
UPLOAD_RETRY_S = 60.0  # a failed checkpoint upload is retried this soon
REQUEST_WAIT_S = 600.0  # default wait for the job to save after a checkpoint request
FINAL_PUSH_S = 30.0  # the status push after the job ended waits at most this long
RESUME_TRIES = 3  # a transient failure downloading the resume checkpoint is retried
STORAGE_TOKEN_ENV = "GPU_STORAGE_TOKEN"  # noqa: S105 - an env var name
# the token in a 0600 file this process deletes after reading (Kaggle and Colab launchers):
# a token in the environment stays readable in /proc/<pid>/environ for the whole run
STORAGE_TOKEN_FILE_ENV = "GPU_STORAGE_TOKEN_FILE"  # noqa: S105 - an env var name
TOKEN_RE = re.compile(r"hf_[A-Za-z0-9]{30,}")
HF_MIN_PYTHON = (3, 10)  # huggingface_hub >= 1.32 (the bucket API) needs Python 3.10+
ACK_TRIES = 5  # a failed control-ack write is retried (backoff 1, 2, 4, 8 s) before giving up
HANDOFF_UPLOAD_GRACE_S = 120.0  # a failed handoff upload is retried this long past wait_s
STAGE_FREE_MARGIN = 64 * 1024 * 1024  # bytes left free on the disk after staging a copy
RESUME_DL = ".gpu-resume-dl"  # <workdir>/<this>: a downloaded resume checkpoint


class ResumeUnavailable(Exception):
    """The resume checkpoint exists but could not be downloaded (network, auth). The run
    ends with INSTALL_FAILED_EXIT so adapters reroute it (an environment failure) instead
    of silently restarting a long job from scratch."""


class Terminated(BaseException):
    """SIGTERM/SIGINT outside the entrypoint. BaseException so `except Exception` blocks
    (checkpoint sync, extraction) cannot swallow it; run() turns it into EXIT 128+n."""

    def __init__(self, signum):
        BaseException.__init__(self, signum)
        self.signum = signum


# --------------------------------------------------------------------------- output


class Tee(object):
    """Thread-safe line writer to our stdout and the attempt log file."""

    def __init__(self, log_path, tail_lines=1000):
        self._lock = threading.Lock()
        self._log = None
        self._tail = collections.deque(maxlen=tail_lines)  # for log-tail.json
        self.count = 0  # lines written so far (absolute line numbers for the tail)
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log = open(log_path, "a", encoding="utf-8")  # noqa: SIM115

    def snapshot(self):
        """(last lines, total line count): what log-tail.json publishes."""
        with self._lock:
            return list(self._tail), self.count

    def line(self, text):
        with self._lock:
            for part in text.split("\n"):
                self._tail.append(part)
                self.count += 1
            for stream in (sys.stdout, self._log):
                if stream is None:
                    continue
                try:
                    stream.write(text + "\n")
                    stream.flush()
                except Exception:  # noqa: S110 - a broken stdout must not stop the job
                    pass

    def event(self, t, **fields):
        body = {"t": t}
        body.update({k: v for k, v in fields.items() if v is not None})
        self.line(PREFIX + json.dumps(body, separators=(",", ":")))

    def say(self, text):
        self.line("gpu-router: " + text)

    def close(self):
        with self._lock:
            if self._log is not None:
                with contextlib.suppress(Exception):
                    self._log.close()
                self._log = None


# --------------------------------------------------------------------------- files


def _safe_members(tar, dest):
    dest = os.path.realpath(str(dest))
    for member in tar.getmembers():
        name = member.name
        if name.startswith("/") or ".." in Path(name).parts:
            raise ValueError("unsafe path in archive: %r" % name)
        target = os.path.realpath(os.path.join(dest, name))
        if target != dest and not target.startswith(dest + os.sep):
            raise ValueError("unsafe path in archive: %r" % name)
        if member.issym() or member.islnk():
            link = member.linkname
            if link.startswith("/") or ".." in Path(link).parts:
                raise ValueError("unsafe link in archive: %r -> %r" % (name, link))
        elif not (member.isfile() or member.isdir()):
            continue  # devices, fifos: never extracted
        yield member


def extract(archive, dest):
    """Extract a tar(.gz) into dest, refusing absolute paths, '..' and escaping links."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(str(archive), "r:*") as tar:
        members = list(_safe_members(tar, dest))
        if hasattr(tarfile, "data_filter"):
            tar.extractall(str(dest), members=members, filter="data")
        else:  # Python < 3.12 without the backported filter: members already checked
            tar.extractall(str(dest), members=members)  # noqa: S202


def _copy_tree(src, dst):
    """Copy src into dst (real copies, symlinks followed). Existing files are kept."""
    for root, dirs, files in os.walk(str(src)):
        dirs[:] = [d for d in dirs if not d.startswith(PRIVATE_PREFIX)]
        rel = os.path.relpath(root, str(src))
        out = Path(dst) / rel if rel != "." else Path(dst)
        out.mkdir(parents=True, exist_ok=True)
        for name in files:
            if name.startswith(PRIVATE_PREFIX):
                continue
            d = out / name
            if d.exists():
                continue
            shutil.copy2(os.path.join(root, name), str(d))


def _clear_dir(path):
    """Empty a directory (keep the directory itself)."""
    if not path.is_dir():
        return
    for child in path.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(str(child))
        else:
            child.unlink()


def _dir_has_content(path):
    try:
        return path.is_dir() and any(not p.name.startswith(PRIVATE_PREFIX) for p in path.iterdir())
    except OSError:
        return False


def _claim_dir(path, job_id, tee, what):
    """Make `path` belong to this job: a dir stamped with another job id is emptied (a
    reused workdir must never hand job 1's checkpoints or outputs to job 2). The stamp is a
    hidden sibling (`.<name>.gpu-router-job`), so the dir itself holds only user files."""
    path.mkdir(parents=True, exist_ok=True)
    if not job_id:
        return
    stamp = path.parent / (".%s%s" % (path.name, JOB_STAMP))
    try:
        owner = stamp.read_text(encoding="utf-8").strip()
    except OSError:
        owner = None
    if owner and owner != job_id:
        tee.say("%s %s belonged to job %s; emptied it for this job" % (what, path, owner))
        _clear_dir(path)
    if owner != job_id:
        _write_atomic(stamp, job_id + "\n")


def _fingerprint(path):
    """Change detector for the checkpoint dir: (relpath, size, mtime_ns) of every file.
    Symlinks count as the file they point to; broken links and runner/helper bookkeeping
    (`.gpu-*`) are skipped."""
    items = []
    if not path.is_dir():
        return ()
    for root, dirs, files in os.walk(str(path)):
        dirs[:] = [d for d in dirs if not d.startswith(PRIVATE_PREFIX)]
        for name in files:
            if name.startswith(PRIVATE_PREFIX):
                continue
            full = os.path.join(root, name)
            try:
                st = os.stat(full)
            except OSError:
                continue
            items.append((os.path.relpath(full, str(path)), st.st_size, st.st_mtime_ns))
    return tuple(sorted(items))


def _newest_mtime(fp):
    return max((m for _rel, _size, m in fp), default=0) / 1e9


def _sha256_file(path):
    h = hashlib.sha256()
    with open(str(path), "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _write_atomic(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(str(tmp), "w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(str(tmp), str(path))


def _abs(raw):
    """A path from the command line or environment, absolute against OUR cwd (the child
    runs in code/, so a relative value would mean something else there)."""
    return Path(os.path.expanduser(str(raw))).resolve()


# --------------------------------------------------------------------------- checkpoints


class CheckpointSyncer(object):
    """Publishes GPU_CHECKPOINT_DIR when it changed; emits ckpt lines.

    Two targets: checkpoint storage (`store`, phase 5: `jobs/<job>/ckpt-NNNN/` + latest.json
    via storage.py) or a local sync dir of tar.gz archives (phase 2). Either way the copy is
    built and verified BEFORE ckpt_begin, so a seq number is only used for a checkpoint that
    will be published; a failed upload keeps its seq for the retry.
    """

    def __init__(
        self,
        tee,
        ckpt_dir,
        sync_dir,
        seq_start,
        interval_s,
        store=None,
        smod=None,
        job_id=None,
        attempt=None,
        keep=KEEP_ARCHIVES,
        stage_root=None,
    ):
        self.tee = tee
        self.ckpt_dir = ckpt_dir
        self.sync_dir = sync_dir
        self.seq = seq_start - 1
        self.interval_s = interval_s
        self.store = store
        self.smod = smod
        self.job_id = job_id
        self.attempt = attempt
        self.keep = max(1, int(keep))
        self.stage_root = stage_root
        self.last_fp = _fingerprint(ckpt_dir)  # restored content does not count as new
        self.last_sync = time.monotonic()
        self.last_step = None
        self.last_published = None  # latest.json dict of the newest checkpoint we know
        self.frozen = None  # why this runner publishes nothing more (handoff, superseded)
        self.postponed_since = None
        self._lock = threading.Lock()
        self._thread = None

    @property
    def active(self):
        return (self.sync_dir is not None or self.store is not None) and self.frozen is None

    def freeze(self, why):
        """Publish nothing more (the daemon moves the job; a later sync would race the
        next attempt's checkpoints)."""
        with self._lock:
            if self.frozen is None:
                self.frozen = why

    def unfreeze(self, why):
        """Undo freeze(why) (a handoff answer that could not be delivered: keep syncing
        until the request is answered)."""
        with self._lock:
            if self.frozen == why:
                self.frozen = None

    def due(self):
        if not self.active or self.interval_s <= 0:
            return False
        if self.postponed_since is not None:
            return True  # retry every tick until the files settle
        return time.monotonic() - self.last_sync >= self.interval_s

    def start_background(self):
        """Run one periodic sync on its own thread (the tick thread keeps heartbeating).
        No-op while a sync is already running."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self.sync, name="gpu-ckpt-sync", daemon=True)
        self._thread.start()

    def wait(self, timeout=None):
        t = self._thread
        if t is not None:
            t.join(timeout)

    def sync(self, final=False, min_age_s=None):
        """Archive the checkpoint dir if it changed since the last sync. Never raises
        (except Terminated). `min_age_s`: files must be at least this old (default
        STABLE_S); for the final sync, a younger file means skip, not postpone."""
        if not self.active:
            return None
        min_age = STABLE_S if min_age_s is None else min_age_s
        with self._lock:
            if self.frozen is not None:
                return None
            fp = _fingerprint(self.ckpt_dir)
            if not fp or fp == self.last_fp:
                self.last_sync = time.monotonic()
                self.postponed_since = None
                return None
            changed = set(fp) - set(self.last_fp)
            age = time.time() - _newest_mtime(changed) if changed else float("inf")
            if min_age > 0 and age < min_age:
                if final:
                    self.tee.say(
                        "not syncing the last checkpoint: it changed %.1fs before the job "
                        "ended and may be half-written; the previous checkpoint stands" % age
                    )
                    return None
                now = time.monotonic()
                if self.postponed_since is None:
                    self.postponed_since = now
                if now - self.postponed_since < MAX_POSTPONE_S:
                    return None  # still being written; try again next tick
            return self._archive(fp)

    def _archive(self, fp):
        if self.store is not None:
            return self._publish(fp)
        seq = self.seq + 1
        self.sync_dir.mkdir(parents=True, exist_ok=True)
        out = self.sync_dir / ("ckpt-%04d.tar.gz" % seq)
        tmp = out.with_name(out.name + ".tmp")
        published = False
        try:
            with tarfile.open(str(tmp), "w:gz", compresslevel=1, dereference=True) as tar:
                for rel, _size, _mtime in fp:
                    tar.add(str(self.ckpt_dir / rel), arcname=rel, recursive=False)
            after = _fingerprint(self.ckpt_dir)
            if after != fp:
                # a file changed while it was being archived: never publish a torn copy
                if self.postponed_since is None:
                    self.postponed_since = time.monotonic()
                return None
            self.seq = seq
            self.tee.event("ckpt_begin", seq=seq)
            os.replace(str(tmp), str(out))
            published = True
            self.last_fp = fp
            self.last_sync = time.monotonic()
            self.postponed_since = None
            self.tee.event(
                "ckpt_end",
                seq=seq,
                uri=out.resolve().as_uri(),
                step=self.last_step,
                size=out.stat().st_size,
                sha256=_sha256_file(out),
            )
            self._prune()
            return seq
        except Exception as exc:
            self.last_sync = time.monotonic()
            self.tee.say("checkpoint %d sync failed: %s; will retry next interval" % (seq, exc))
            return None
        finally:
            if not published:
                with contextlib.suppress(OSError):
                    tmp.unlink()

    # ---- storage target (phase 5)

    def _still_owner(self):
        """False (and frozen) once the daemon has handed the job to a later attempt."""
        if self.attempt is None:
            return True
        try:
            owner = self.smod.read_owner(self.store, self.job_id)
        except Exception:
            return True  # cannot tell: publish (the pointer names this attempt)
        if owner is not None and owner > self.attempt:
            self.frozen = "attempt %d owns this job now" % owner
            self.tee.say(
                "not syncing checkpoints any more: gpu-router moved this job to attempt %d" % owner
            )
            return False
        return True

    def _stage(self, fp, stage):
        """Copy the fingerprinted files into a private stage dir, hashing on the way.
        Returns [(relpath, size, sha256)]."""
        if stage.exists():
            shutil.rmtree(str(stage))
        stage.mkdir(parents=True)
        files = []
        for rel, _size, _mtime in fp:
            src = self.ckpt_dir / rel
            out = stage / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            h = hashlib.sha256()
            size = 0
            with open(str(src), "rb") as fin, open(str(out), "wb") as fout:
                for block in iter(lambda: fin.read(1 << 20), b""):
                    h.update(block)
                    fout.write(block)
                    size += len(block)
            files.append((rel.replace(os.sep, "/"), size, h.hexdigest()))
        return files

    def _stage_fits(self, fp):
        """True when a staged copy of fp fits on the stage disk with a margin (a full disk
        would break the job's own saves)."""
        need = sum(size for _rel, size, _mtime in fp)
        probe = self.stage_root
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        try:
            free = shutil.disk_usage(str(probe)).free
        except OSError:
            return True
        return free >= need + STAGE_FREE_MARGIN

    def _hash_in_place(self, fp):
        """[(relpath, size, sha256)] of the checkpoint dir's files, read where they are."""
        files = []
        for rel, _size, _mtime in fp:
            h = hashlib.sha256()
            size = 0
            with open(str(self.ckpt_dir / rel), "rb") as fin:
                for block in iter(lambda: fin.read(1 << 20), b""):
                    h.update(block)
                    size += len(block)
            files.append((rel.replace(os.sep, "/"), size, h.hexdigest()))
        return files

    def _landed(self, seq):
        """After a failed publish: latest.json may already name seq (an upload that
        succeeded server-side but timed out here). Returns latest.json when this attempt's
        seq landed; otherwise makes sure a seq that latest.json names is never reused."""
        try:
            latest = self.smod.read_latest(self.store, self.job_id)
        except Exception:
            return None
        if latest is None or latest["seq"] < seq:
            return None
        if latest["seq"] == seq and latest.get("attempt") == self.attempt:
            return latest
        self.seq = max(self.seq, int(latest["seq"]))
        return None

    def _publish(self, fp):
        seq = self.seq + 1
        if not self._still_owner():
            return None
        stage = self.stage_root / ("ckpt-%04d" % seq)
        direct = not self._stage_fits(fp)
        try:
            try:
                # no room for a staged copy: upload from the checkpoint dir itself and
                # check afterwards that nothing changed before naming it (D44)
                files = self._hash_in_place(fp) if direct else self._stage(fp, stage)
            except OSError as exc:
                self.last_sync = time.monotonic()
                self.tee.say("checkpoint %d could not be staged: %s; will retry" % (seq, exc))
                return None
            if _fingerprint(self.ckpt_dir) != fp:
                # a file changed while it was being copied: never publish a torn copy
                if self.postponed_since is None:
                    self.postponed_since = time.monotonic()
                return None
            self.tee.event("ckpt_begin", seq=seq)
            try:
                if direct:
                    rels = [rel for rel, _size, _sha in files]
                    self.store.upload_dir(self.ckpt_dir, self.smod.ckpt_key(self.job_id, seq), rels)
                    if _fingerprint(self.ckpt_dir) != fp:
                        self.tee.say(
                            "checkpoint %d changed while it was uploading; will retry" % seq
                        )
                        if self.postponed_since is None:
                            self.postponed_since = time.monotonic()
                        return None
                latest = self.smod.publish_checkpoint(
                    self.store,
                    self.job_id,
                    seq,
                    stage,
                    files,
                    attempt=self.attempt,
                    step=self.last_step,
                    move=True,
                    skip_upload=direct,
                )
            except Exception as exc:
                latest = self._landed(seq)
                if latest is None:
                    retry = min(self.interval_s, UPLOAD_RETRY_S) if self.interval_s > 0 else 0
                    self.last_sync = time.monotonic() - max(0.0, self.interval_s - retry)
                    self.tee.say(
                        "checkpoint %d upload failed: %s; retrying in %ds"
                        % (seq, getattr(exc, "message", exc), retry)
                    )
                    return None
        finally:
            if stage.exists():
                shutil.rmtree(str(stage), ignore_errors=True)
        self.seq = seq
        self.last_fp = fp
        self.last_sync = time.monotonic()
        self.postponed_since = None
        self.last_published = latest
        self.tee.event(
            "ckpt_end",
            seq=seq,
            uri=latest["uri"],
            step=self.last_step,
            size=latest.get("size"),
            sha256=latest.get("sha256"),
        )
        if seq - self.keep >= 1:
            # everything below the kept window, not just seq - keep: seq jumps (a move
            # between backends, a resumed attempt) would otherwise leak old checkpoints
            with contextlib.suppress(Exception):
                for old in self.smod.list_checkpoints(self.store, self.job_id):
                    if old <= seq - self.keep:
                        self.store.delete_prefix(self.smod.ckpt_key(self.job_id, old))
        return seq

    def _prune(self):
        """Keep only the newest KEEP_ARCHIVES archives (a full disk kills the job)."""
        found = []
        for p in self.sync_dir.iterdir():
            m = ARCHIVE_RE.match(p.name)
            if m:
                found.append((int(m.group(1)), p))
        for _seq, p in sorted(found)[:-KEEP_ARCHIVES]:
            with contextlib.suppress(OSError):
                p.unlink()


# --------------------------------------------------------------------------- status channel


class Redactor(object):
    """Masks secret values (the storage token, GPU_SECRET_NAMES values) and HF-token-shaped
    strings in what the runner publishes to storage. The job's log file is not changed
    (the daemon redacts its own copy)."""

    def __init__(self, values):
        self.values = sorted({v for v in values if v and len(v) >= 6}, key=len, reverse=True)

    def __call__(self, text):
        for value in self.values:
            if value in text:
                text = text.replace(value, "***")
        return TOKEN_RE.sub("***", text)


class StatusChannel(object):
    """Near-live status through checkpoint storage, for providers without live logs
    (Kaggle): every push_s it writes heartbeat.json + log-tail.json for this attempt; every
    control_s it checks control.json for a checkpoint request from the daemon (planned
    handoff before a session cap or quota end) and answers it in control-ack.json."""

    def __init__(
        self, tee, store, smod, job_id, attempt, syncer, ckpt_dir, redact, push_s, control_s
    ):
        self.tee = tee
        self.store = store
        self.smod = smod
        self.job_id = job_id
        self.attempt = attempt or 0
        self.syncer = syncer
        self.ckpt_dir = ckpt_dir
        self.redact = redact
        self.push_s = push_s
        self.control_s = control_s
        self.started = time.monotonic()
        self.step = None
        self.total_steps = None
        self.metrics = {}
        self.handled = None  # id of the last control request taken
        self._ctl_token = None
        self._push_failing = False
        self._stop = threading.Event()
        self._thread = None
        self._handler = None

    def note_line(self, body):
        """Protocol lines seen by the runner (metric/total) feed the heartbeat."""
        t = body.get("t")
        if t == "total" and isinstance(body.get("steps"), int):
            self.total_steps = body["steps"]
        elif t == "metric":
            if isinstance(body.get("step"), int) and not isinstance(body.get("step"), bool):
                self.step = body["step"]
            if isinstance(body.get("total"), int):
                self.total_steps = body["total"]
            if isinstance(body.get("metrics"), dict):
                self.metrics.update(body["metrics"])

    def start(self):
        if self.push_s <= 0 and self.control_s <= 0:
            return
        self._thread = threading.Thread(target=self._loop, name="gpu-status", daemon=True)
        self._thread.start()

    def stop(self, code=None):
        """Stop the loop and push once more (the tail then ends with the exit line). The
        last push is bounded: a hung upload must not keep a finished session alive."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
        if self._handler is not None:
            self._handler.join(timeout=5)
        if self.push_s > 0:
            last = threading.Thread(
                target=self.push, kwargs={"phase": "exited", "code": code}, daemon=True
            )
            last.start()
            last.join(timeout=FINAL_PUSH_S)

    def _loop(self):
        next_push = 0.0
        next_ctl = 0.0
        while not self._stop.wait(1.0):
            now = time.monotonic()
            if self.push_s > 0 and now >= next_push:
                next_push = now + self.push_s
                self.push()
            if self.control_s > 0 and now >= next_ctl:
                next_ctl = now + self.control_s
                self.poll_control()

    def push(self, phase="running", code=None):
        lines, total = self.tee.snapshot()
        max_len = self.smod.LOG_TAIL_LINE_MAX
        tail = {
            "v": 1,
            "attempt": self.attempt,
            "first": total - len(lines),
            "total": total,
            "lines": [self.redact(line)[:max_len] for line in lines],
            "ts": round(time.time(), 3),
        }
        latest = self.syncer.last_published or {}
        beat = {
            "v": 1,
            "attempt": self.attempt,
            "ts": round(time.time(), 3),
            "elapsed_s": round(time.monotonic() - self.started, 1),
            "phase": phase,
            "exit_code": code,
            "step": self.step,
            "total_steps": self.total_steps,
            "metrics": dict(self.metrics),
            "ckpt_seq": latest.get("seq"),
            "lines": total,
            "runner": RUNNER_VERSION,
        }
        try:
            self.store.write_many(
                {
                    self.smod.heartbeat_key(self.job_id, self.attempt): self.smod.dumps(beat),
                    self.smod.log_tail_key(self.job_id, self.attempt): self.smod.dumps(tail),
                }
            )
        except Exception as exc:
            if not self._push_failing:
                self._push_failing = True
                self.tee.say("could not push status to storage: %s" % getattr(exc, "message", exc))
            return
        if self._push_failing:
            self._push_failing = False
            self.tee.say("status pushes to storage work again")

    def poll_control(self):
        key = self.smod.control_key(self.job_id, self.attempt)
        try:
            st = self.store.stat(key)
            if st is None or st[1] == self._ctl_token:
                return
            req = self.smod.read_json(self.store, key)
        except Exception:
            return
        if not req or not req.get("id") or req.get("id") == self.handled:
            self._ctl_token = st[1]
            return
        if self._handler is not None and self._handler.is_alive():
            return  # still answering the previous request: look again next poll
        self._ctl_token = st[1]
        self.handled = req.get("id")
        self._handler = threading.Thread(
            target=self.handle_request, args=(req,), name="gpu-ckpt-request", daemon=True
        )
        self._handler.start()

    def handle_request(self, req):
        """Ask the job for a checkpoint, wait for it to settle, sync, acknowledge."""
        rid = str(req.get("id"))
        action = req.get("action") or "checkpoint"
        try:
            wait_s = max(0.0, float(req.get("wait_s", REQUEST_WAIT_S)))
        except (TypeError, ValueError):
            wait_s = REQUEST_WAIT_S
        flag = self.ckpt_dir / REQUEST_FILE
        with contextlib.suppress(OSError):
            _write_atomic(flag, rid + "\n")
        why = req.get("reason") or "checkpoint requested"
        self.tee.say(
            "gpu-router asked for a checkpoint (%s); waiting up to %ds for the job to save one"
            % (why, wait_s)
        )
        before = self.syncer.last_fp
        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline and not self._stop.is_set():
            fp = _fingerprint(self.ckpt_dir)
            changed = set(fp) - set(before)
            if fp and changed and time.time() - _newest_mtime(changed) >= STABLE_S:
                break
            time.sleep(0.5)
        seq = self.syncer.sync()
        # a save that did not make it into storage (upload failed, still settling) is
        # retried until a little past the wait instead of handing over the older one
        retry_until = max(deadline, time.monotonic()) + HANDOFF_UPLOAD_GRACE_S
        while (
            seq is None
            and self.syncer.active
            and _fingerprint(self.ckpt_dir) not in ([], self.syncer.last_fp)
            and time.monotonic() < retry_until
            and not self._stop.is_set()
        ):
            time.sleep(min(5.0, max(0.1, retry_until - time.monotonic())))
            seq = self.syncer.sync()
        with contextlib.suppress(OSError):
            flag.unlink()
        latest = self.syncer.last_published or {}
        if action == "handoff":
            # freeze before answering: a periodic sync after the daemon read the answer
            # would race the next attempt's checkpoints; undone if the answer fails
            self.syncer.freeze("handed off")
        ack = {
            "v": 1,
            "id": rid,
            "attempt": self.attempt,
            "new": seq is not None,
            "seq": latest.get("seq"),
            "uri": latest.get("uri"),
            "step": latest.get("step"),
            "size": latest.get("size"),
            "sha256": latest.get("sha256"),
            "created_at": latest.get("created_at"),
            "ts": round(time.time(), 3),
        }
        problem = None
        for i in range(ACK_TRIES):
            try:
                self.store.write_bytes(
                    self.smod.ack_key(self.job_id, self.attempt), self.smod.dumps(ack)
                )
                problem = None
                break
            except Exception as exc:
                problem = getattr(exc, "message", exc)
                if i + 1 < ACK_TRIES and not self._stop.is_set():
                    time.sleep(2**i)
        if problem is not None:
            self.tee.say("could not answer the checkpoint request: %s" % problem)
            if action == "handoff":
                self.syncer.unfreeze("handed off")  # the daemon did not hear it: keep going
            self.handled = None  # answer the next poll again (the same control.json)
            self._ctl_token = None
            return
        if latest.get("seq"):
            self.tee.say(
                "checkpoint %s is ready for the handoff%s"
                % (latest["seq"], "" if seq is not None else " (no newer save appeared)")
            )
        else:
            self.tee.say("no checkpoint to hand over yet; the next attempt starts fresh")


# --------------------------------------------------------------------------- storage setup


def _load_storage(root):
    """The storage module shipped in the bundle (gpu_runner/storage.py), else the copy next
    to this file. Loaded by path so a user module named `storage` never shadows it."""
    for candidate in (
        root / "gpu_runner" / "storage.py",
        Path(__file__).resolve().parent / "storage.py",
    ):
        if candidate.is_file():
            spec = importlib.util.spec_from_file_location("gpu_runner_storage", str(candidate))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    raise RuntimeError("the bundle has no gpu_runner/storage.py")


def _hf_usable():
    try:
        from huggingface_hub import HfApi
    except Exception:
        return False
    return hasattr(HfApi, "batch_bucket_files") and hasattr(HfApi, "download_bucket_files")


def _ensure_hf(smod, cache_root, tee):
    """huggingface_hub with the bucket API, importable by THIS process. An image with an
    older one (transformers 4.x pins huggingface_hub < 1) keeps it: the new version goes
    into a private --target dir that only the runner puts on sys.path."""
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")  # runner process only
    if sys.version_info < HF_MIN_PYTHON:
        tee.say(
            "checkpoint storage on Hugging Face needs Python %d.%d+ (%s here is %d.%d)"
            % (*HF_MIN_PYTHON, sys.executable, *sys.version_info[:2])
        )
        return False
    if _hf_usable():
        return True
    tag = hashlib.sha256(smod.HF_REQUIREMENT.encode()).hexdigest()[:8]
    target = Path(cache_root) / ("hf-" + tag)
    ok = target / ".gpu-ok"
    if not ok.is_file():
        tee.say("installing %s for checkpoint storage (runner only)" % smod.HF_REQUIREMENT)
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        rc = _run_streaming(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--quiet",
                "--disable-pip-version-check",
                "--no-input",
                "--target",
                str(target),
                smod.HF_REQUIREMENT,
            ],
            str(Path(cache_root)),
            env,
            tee,
            None,
        )
        if rc != 0:
            tee.say("could not install huggingface_hub (pip exit %d)" % rc)
            return False
        with contextlib.suppress(OSError):
            ok.write_text("ok\n")
    if str(target) not in sys.path:
        sys.path.insert(0, str(target))
    for name in list(sys.modules):
        if name == "huggingface_hub" or name.startswith("huggingface_hub."):
            del sys.modules[name]
    importlib.invalidate_caches()
    return _hf_usable()


def _open_storage(uri, token, smod, cache_root, tee):
    """(store, why not). Never raises."""
    try:
        if str(uri).startswith(smod.HF_PREFIX) and not _ensure_hf(smod, cache_root, tee):
            if sys.version_info < HF_MIN_PYTHON:
                return None, "Hugging Face storage needs Python %d.%d+" % HF_MIN_PYTHON
            return None, "huggingface_hub could not be installed"
        store = smod.open_store(uri, token=token)
        if store.kind == "local":
            store.ensure()
        return store, None
    except Exception as exc:
        return None, str(getattr(exc, "message", exc))


def _fetch_resume(uri, token, smod, store, dest, cache_root, tee):
    """Download a storage checkpoint (hf://...) into dest and check it against its
    manifest. Returns True when restored, False when it does not exist or is incomplete
    (the caller tries the next candidate). Raises ResumeUnavailable on repeated transient
    failures."""
    split = smod.split_hf_uri(uri)
    if split is None:
        return False
    root, key = split
    if store is None or store.root_uri != root:
        if not _ensure_hf(smod, cache_root, tee):
            raise ResumeUnavailable("huggingface_hub is not available to download %s" % uri)
        store = smod.open_store(root, token=token)
    last = None
    for i in range(RESUME_TRIES):
        if dest.exists():
            shutil.rmtree(str(dest))
        dest.mkdir(parents=True)
        try:
            n = smod.restore_checkpoint(store, key, dest)
            problem = smod.verify_checkpoint(store, key, dest)
            name = key.rsplit("/", 1)[-1]
            if problem is not None:
                tee.say("checkpoint %s is incomplete (%s); not restoring it" % (name, problem))
                return False
            tee.say("downloaded checkpoint %s (%d files)" % (name, n))
            return True
        except Exception as exc:
            if getattr(exc, "missing", False):
                return False
            if not getattr(exc, "retryable", True):
                raise ResumeUnavailable(str(getattr(exc, "message", exc))) from None
            last = exc
            if i + 1 < RESUME_TRIES:
                time.sleep(5 * (i + 1))
    raise ResumeUnavailable(str(getattr(last, "message", last)))


# --------------------------------------------------------------------------- datasets


_HF_DATASET_RE = re.compile(
    r"^hf://datasets/(?P<repo>[^/@]+/[^/@]+)(?:@(?P<rev>[^/]+))?(?:/(?P<sub>.+))?$"
)


def _materialize_data(items, data_dir, token, smod, cache_root, tee):
    """Put each dataset at <data_dir>/<mount>: a symlink for a path on this machine, a
    download (cached on this machine by content hash) for a storage or HF dataset URI.
    Raises RuntimeError with a message when one cannot be made available."""
    data_dir.mkdir(parents=True, exist_ok=True)
    for item in items:
        mount = str(item.get("mount") or "")
        if not re.match(r"^[a-z0-9][a-z0-9._-]{0,63}$", mount):
            raise RuntimeError("bad dataset mount name %r" % mount)
        target = data_dir / mount
        stamp = data_dir / (".%s.gpu-data" % mount)
        if item.get("local"):
            src = Path(str(item["local"]))
            if not src.exists():
                raise RuntimeError("dataset %s: %s does not exist on this machine" % (mount, src))
            if target.is_symlink() or target.is_file():
                target.unlink()
            elif target.is_dir():
                shutil.rmtree(str(target))
            os.symlink(str(src), str(target))
            with contextlib.suppress(OSError):
                stamp.unlink()
            tee.say("dataset %s -> %s" % (mount, src))
            continue
        uri = str(item.get("uri") or "")
        want = str(item.get("sha256") or uri)
        try:
            have = stamp.read_text(encoding="utf-8").strip()
        except OSError:
            have = None
        if have == want and target.is_dir():
            tee.say("dataset %s already on this machine (%s)" % (mount, uri))
            continue
        tmp = data_dir / (".%s.tmp" % mount)
        if tmp.exists():
            shutil.rmtree(str(tmp))
        tmp.mkdir(parents=True)
        tee.say("downloading dataset %s (%s)" % (mount, uri))
        if not _ensure_hf(smod, cache_root, tee):
            raise RuntimeError("huggingface_hub is not available to download %s" % uri)
        split = smod.split_hf_uri(uri)
        m = _HF_DATASET_RE.match(uri)
        if split is not None:
            root, key = split
            store = smod.open_store(root, token=token)
            try:
                n = store.download_dir(key, tmp, skip=(smod.DATA_MANIFEST,))
            except Exception as exc:
                raise RuntimeError(
                    "dataset %s: %s" % (mount, getattr(exc, "message", exc))
                ) from None
        elif m is not None:
            from huggingface_hub import snapshot_download

            sub = m.group("sub")
            try:
                snapshot_download(
                    repo_id=m.group("repo"),
                    repo_type="dataset",
                    revision=m.group("rev"),
                    allow_patterns=[sub.rstrip("/") + "/**", sub] if sub else None,
                    local_dir=str(tmp),
                    token=os.environ.get("HF_TOKEN") or token or None,
                )
            except Exception as exc:
                raise RuntimeError(
                    "dataset %s: %s" % (mount, str(exc).splitlines()[0][:200])
                ) from None
            n = sum(len(f) for _r, _d, f in os.walk(str(tmp)))
        else:
            raise RuntimeError("dataset %s: unsupported uri %r" % (mount, uri))
        if target.is_symlink() or target.is_file():
            target.unlink()
        elif target.exists():
            shutil.rmtree(str(target))
        os.rename(str(tmp), str(target))
        _write_atomic(stamp, want + "\n")
        tee.say("dataset %s ready at %s (%d files)" % (mount, target, n))


# --------------------------------------------------------------------------- deps


def install_deps(manifest, code_dir, python, tee):
    """pip-install what the manifest recorded. Returns pip's exit code (0 = nothing to do)."""
    deps = manifest.get("deps") or {}
    kind = deps.get("kind", "none")
    cmd = [python, "-m", "pip", "install", "--disable-pip-version-check", "--no-input"]
    if kind == "requirements" and deps.get("file"):
        req = code_dir / deps["file"]
        if not req.is_file():
            tee.say("requirements file %s is missing from the bundle; skipping" % deps["file"])
            return 0
        cmd += ["-r", str(req)]
        what = deps["file"]
    elif kind == "pyproject" and deps.get("packages"):
        cmd += list(deps["packages"])
        what = "%d packages from pyproject.toml" % len(deps["packages"])
    else:
        return 0
    tee.say("installing dependencies (%s)" % what)
    return _run_streaming(cmd, str(code_dir), dict(os.environ), tee, None)


# --------------------------------------------------------------------------- process

_current_child = {"proc": None}  # the process group run_streaming is relaying, if any


def _signal_group(proc, signum):
    """Send signum to the child's whole process group (it is a session leader)."""
    try:
        os.killpg(proc.pid, signum)
    except (OSError, AttributeError):
        with contextlib.suppress(Exception):
            proc.send_signal(signum)


def _on_signal(signum, _frame):
    """Process-wide SIGTERM/SIGINT handler: forward to the running entrypoint's group, or
    abort whatever the runner is doing (extraction, restore, final sync)."""
    proc = _current_child["proc"]
    if proc is not None and proc.poll() is None:
        _signal_group(proc, signum)
        return
    raise Terminated(signum)


def install_signal_handlers():
    """Returns the previous handlers (restore with restore_signal_handlers)."""
    previous = {}
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGTERM, signal.SIGINT):
            with contextlib.suppress(ValueError, OSError):
                previous[sig] = signal.signal(sig, _on_signal)
    return previous


def restore_signal_handlers(previous):
    for sig, handler in previous.items():
        with contextlib.suppress(ValueError, OSError, TypeError):
            signal.signal(sig, handler)


def _run_streaming(argv, cwd, env, tee, on_line, on_tick=None):
    """Run argv in its own process group, relaying merged stdout/stderr line by line.
    Returns a POSIX-ish exit code (signals -> 128+n)."""
    try:
        proc = subprocess.Popen(
            argv,
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
            start_new_session=True,
        )
    except OSError as exc:
        tee.say("could not start %s: %s" % (argv[0], exc))
        return 127

    previous = {}
    installed_here = False
    if threading.current_thread() is threading.main_thread():
        current = signal.getsignal(signal.SIGTERM)
        if current is not _on_signal:  # run() normally installed them already
            previous = install_signal_handlers()
            installed_here = True
    _current_child["proc"] = proc

    stop = threading.Event()
    ticker = None
    if on_tick is not None:

        def tick_loop():
            while not stop.wait(1.0):
                with contextlib.suppress(Exception):
                    on_tick()

        ticker = threading.Thread(target=tick_loop, name="gpu-tick", daemon=True)
        ticker.start()

    state = {"buf": b"", "pending": None, "last_progress": 0.0}

    def emit(raw):
        text = raw.decode("utf-8", "replace")
        tee.line(text)
        if on_line is not None:
            on_line(text)

    def progress(seg):
        if seg.strip():
            state["pending"] = seg
        now = time.monotonic()
        if state["pending"] is not None and now - state["last_progress"] >= PROGRESS_FLUSH_S:
            emit(state["pending"])
            state["pending"] = None
            state["last_progress"] = now

    def emit_line(line):
        # a protocol line glued behind a '\r' progress segment (tqdm on stderr, gpu.log on
        # stdout, one pipe): the segment is progress, the protocol part its own line
        idx = line.find(PREFIX.encode())
        if idx > 0:
            progress(line[:idx])
            line = line[idx:]
        emit(line)
        state["pending"] = None

    def feed(chunk):
        buf = state["buf"] + chunk
        while True:
            nl = buf.find(b"\n")
            cr = buf.find(b"\r")
            if nl == -1 and cr == -1:
                break
            if nl != -1 and (cr == -1 or nl < cr):
                line, buf = buf[:nl], buf[nl + 1 :]
                emit_line(line)
            elif cr + 1 < len(buf) and buf[cr + 1 : cr + 2] == b"\n":
                line, buf = buf[:cr], buf[cr + 2 :]  # CRLF
                emit_line(line)
            elif cr + 1 >= len(buf):
                break  # wait for the next byte to tell CR from CRLF
            else:
                seg, buf = buf[:cr], buf[cr + 1 :]
                progress(seg)
        state["buf"] = buf

    def drain(fd, budget_s):
        """Read whatever is left (bounded: a writer outside the group may never stop)."""
        deadline = time.monotonic() + budget_s
        with contextlib.suppress(Exception):
            while time.monotonic() < deadline and select.select([fd], [], [], 0.1)[0]:
                chunk = os.read(fd, 65536)
                if not chunk:
                    return
                feed(chunk)

    rc = None
    try:
        fd = proc.stdout.fileno()
        exited_at = None
        while True:
            now = time.monotonic()
            if exited_at is None and proc.poll() is not None:
                exited_at = now  # the entrypoint is gone; drain, then stop
            if exited_at is not None and now - exited_at >= DRAIN_GRACE_S:
                # something the entrypoint started still holds the pipe: end the session
                _signal_group(proc, signal.SIGKILL)
                drain(fd, 1.0)
                tee.say("stopped processes the job left running (they kept its output open)")
                break
            timeout = 0.5 if exited_at is None else DRAIN_GRACE_S - (now - exited_at)
            ready, _, _ = select.select([fd], [], [], max(0.0, timeout))
            if not ready:
                continue
            chunk = os.read(fd, 65536)
            if not chunk:
                break  # EOF: nobody holds the pipe any more
            feed(chunk)
        buf = state["buf"]
        state["buf"] = b""
        if buf.strip():
            emit_line(buf.rstrip(b"\r"))
        elif state["pending"] is not None:
            emit(state["pending"])
        rc = proc.wait()
    finally:
        _current_child["proc"] = None
        # leftovers after a normal EOF too (a detached grandchild with its own output)
        with contextlib.suppress(Exception):
            if proc.poll() is None:
                _signal_group(proc, signal.SIGKILL)
                proc.wait()
        with contextlib.suppress(Exception):
            proc.stdout.close()
        stop.set()
        if ticker is not None:
            ticker.join(timeout=5)
        if installed_here:
            restore_signal_handlers(previous)
    if rc < 0:
        return 128 + (-rc)
    return rc


# --------------------------------------------------------------------------- main


def _env_number(name, default, cast, notes):
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return cast(raw)
    except ValueError:
        notes.append("ignored %s=%r (not a number); using %s" % (name, raw, default))
        return default


def _parse_args(argv, notes):
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(prog="bootstrap.py", description="gpu-router remote runner")
    p.add_argument("--bundle", default=os.environ.get("GPU_BUNDLE"))
    p.add_argument("--workdir", default=os.environ.get("GPU_WORKDIR"))
    p.add_argument("--resume", default=os.environ.get("GPU_RESUME_SRC") or None)
    p.add_argument("--log-file", default=os.environ.get("GPU_LOG_FILE"))
    p.add_argument("--exit-file", default=os.environ.get("GPU_EXIT_FILE"))
    p.add_argument(
        "--heartbeat-s", type=float, default=_env_number("GPU_HEARTBEAT_S", 60.0, float, notes)
    )
    p.add_argument(
        "--checkpoint-sync-dir", default=os.environ.get("GPU_CHECKPOINT_SYNC_DIR") or None
    )
    p.add_argument(
        "--ckpt-seq-start", type=int, default=_env_number("GPU_CKPT_SEQ_START", 1, int, notes)
    )
    p.add_argument("--checkpoint-interval-min", type=float, default=None)
    p.add_argument(
        "--skip-install",
        action="store_true",
        default=os.environ.get("GPU_SKIP_INSTALL") == "1",
    )
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--storage", default=os.environ.get("GPU_STORAGE") or None)
    args = p.parse_args(argv)
    args.resume_uri = os.environ.get("GPU_RESUME_URI") or None
    args.interval_s = _env_number("GPU_CHECKPOINT_INTERVAL_S", None, float, notes)
    args.status_push_s = _env_number("GPU_STATUS_PUSH_S", STATUS_PUSH_S, float, notes)
    args.control_poll_s = _env_number("GPU_CONTROL_POLL_S", CONTROL_POLL_S, float, notes)
    args.keep = _env_number("GPU_CKPT_KEEP", KEEP_ARCHIVES, int, notes)
    if args.bundle is None and (here.parent / MANIFEST).is_file():
        args.bundle = str(here.parent)  # running from an extracted bundle
    return args


def _resolve_workdir(args):
    """Where EXIT, job.log, checkpoints/ and outputs/ live (decided before any output)."""
    if args.workdir:
        return _abs(args.workdir)
    if args.bundle is not None and Path(args.bundle).is_dir():
        return _abs(args.bundle)
    return Path.cwd() / "gpu-job"


def _default_exit_file(argv):
    """Where EXIT goes when the arguments could not even be parsed."""
    raw = os.environ.get("GPU_EXIT_FILE")
    if raw:
        return _abs(raw)
    workdir = os.environ.get("GPU_WORKDIR")
    argv = list(argv if argv is not None else sys.argv[1:])
    for i, a in enumerate(argv):
        if a == "--exit-file" and i + 1 < len(argv):
            return _abs(argv[i + 1])
        if a.startswith("--exit-file="):
            return _abs(a.split("=", 1)[1])
        if a == "--workdir" and i + 1 < len(argv):
            workdir = argv[i + 1]
        elif a.startswith("--workdir="):
            workdir = a.split("=", 1)[1]
    return (_abs(workdir) if workdir else Path.cwd() / "gpu-job") / "EXIT"


def _prepare_bundle(args, workdir, tee):
    """Returns the bundle root (the directory holding manifest.json). An archive is
    extracted into <workdir>/bundle, stamped with its sha256; a different archive gets a
    clean re-extraction (a reused workdir must never run the previous job's code)."""
    if args.bundle is None:
        raise ValueError("no bundle: pass --bundle or run from an extracted bundle")
    src = _abs(args.bundle)
    if src.is_dir():
        root = src
    else:
        if not src.is_file():
            raise ValueError("bundle %s not found" % src)
        root = workdir / "bundle"
        sha = _sha256_file(src)
        stamp = root / BUNDLE_STAMP
        try:
            current = stamp.read_text(encoding="utf-8").strip()
        except OSError:
            current = None
        if current != sha or not (root / MANIFEST).is_file():
            tee.say("unpacking %s" % src.name)
            if root.exists():
                shutil.rmtree(str(root))
            extract(src, root)
            _write_atomic(stamp, sha + "\n")
    if not (root / MANIFEST).is_file():
        raise ValueError("%s has no %s" % (root, MANIFEST))
    return root


def _entry_argv(manifest, python):
    ep = manifest.get("entrypoint") or {}
    args = [str(a) for a in ep.get("args") or []]
    if ep.get("script"):
        return [python, "-u", ep["script"], *args]
    if ep.get("command"):
        return [str(a) for a in ep["command"]] + args
    raise ValueError("manifest has no entrypoint")


def _resume_candidates(raw, latest, store, smod):
    """URIs to try, best first: the one the engine named, storage's latest.json when it
    names another (the engine's DB can lag storage: log lines lost in a tail gap), then
    the older checkpoints storage still holds (a torn newest one)."""
    out = [raw]
    if latest and latest.get("uri") and latest["uri"] != raw:
        out.append(str(latest["uri"]))
    if store is not None and smod is not None and store.key_of(raw) is not None:
        want = smod.seq_of(raw)
        job_key = store.key_of(raw).rsplit("/", 1)[0]
        job_id = job_key.split("/")[-1]
        if want is not None:
            try:
                older = [q for q in smod.list_checkpoints(store, job_id) if q < want]
            except Exception:
                older = []
            for q in sorted(older, reverse=True):
                out.append(store.uri(smod.ckpt_key(job_id, q)))
    seen = set()
    return [u for u in out if not (u in seen or seen.add(u))]


def _restore_candidate(uri, workdir, storage, tee):
    """A local path holding checkpoint `uri` (downloaded for hf://), or None when it is
    gone or incomplete. Raises ResumeUnavailable when it cannot be downloaded."""
    smod, store, token, cache_root = storage
    if uri.startswith("hf://"):
        if smod is None:
            raise ResumeUnavailable("no storage module to download %s" % uri)
        dest = workdir / RESUME_DL
        return dest if _fetch_resume(uri, token, smod, store, dest, cache_root, tee) else None
    path = _abs(_uri_path(uri)) if uri.startswith("file://") else _abs(uri)
    if not (path.is_dir() or path.is_file()):
        return None
    key = store.key_of(uri) if store is not None and uri.startswith("file://") else None
    if key is not None and path.is_dir():
        problem = smod.verify_checkpoint(store, key, path)
        if problem is not None:
            tee.say("checkpoint %s is incomplete (%s); not restoring it" % (path.name, problem))
            return None
    return path


def _resume_source(args, workdir, storage, tee, latest=None, latest_error=None):
    """Where the checkpoint to restore is on this machine, or None for a fresh start.
    --resume / GPU_RESUME_SRC (a local path or file:// URI, set by adapters) wins; else
    GPU_RESUME_URI (set by the engine: hf://... is downloaded first). A checkpoint that
    is gone or incomplete falls back to storage's latest.json, then to older intact ones
    (D44); only when none exists does the job start fresh (D34). A missing checkpoint
    while storage's pointer cannot be read, or a download that keeps failing, raises
    ResumeUnavailable (exit 90: the engine retries) instead of discarding progress."""
    raw = args.resume or args.resume_uri
    if not raw:
        return None
    raw = str(raw)
    smod, store, _token, _cache_root = storage
    if args.resume and not raw.startswith("hf://"):
        path = _abs(_uri_path(raw)) if raw.startswith("file://") else _abs(raw)
        if path.is_dir() or path.is_file():
            return path
        raise ValueError("resume checkpoint %s not found" % path)
    candidates = _resume_candidates(raw, latest, store, smod)
    for i, uri in enumerate(candidates):
        got = _restore_candidate(uri, workdir, storage, tee)
        if got is not None:
            if i:
                tee.say("resuming from %s instead of %s" % (uri, raw))
            return got
        if i == 0 and len(candidates) > 1:
            tee.say("checkpoint %s is not usable from here; trying the others in storage" % raw)
    if latest_error is not None:
        raise ResumeUnavailable(
            "checkpoint %s is not usable and the latest checkpoint pointer could not be "
            "read (%s)" % (raw, latest_error)
        )
    if len(candidates) > 1:
        tee.say("no intact checkpoint for this job in storage; starting fresh")
    elif raw.startswith("hf://"):
        tee.say("checkpoint %s no longer exists; starting fresh" % raw)
    else:
        tee.say("checkpoint %s is not reachable from here; starting fresh" % raw)
    return None


def _uri_path(uri):
    from urllib.parse import unquote, urlparse

    return unquote(urlparse(uri).path)


def _restore(
    args,
    env,
    workdir,
    ckpt_dir,
    tee,
    storage=(None, None, None, None),
    latest=None,
    latest_error=None,
):
    """--resume / GPU_RESUME_URI: a clean GPU_RESUME_DIR with the checkpoint, and real
    copies of it in an empty GPU_CHECKPOINT_DIR (resume-from-checkpoint-dir code keeps
    working). A downloaded checkpoint is moved into the resume dir, not copied (disk)."""
    src = _resume_source(args, workdir, storage, tee, latest, latest_error)
    if src is None:
        env.pop("GPU_RESUME_DIR", None)  # fresh start: gpu.resume_dir() is None
        return
    resume_dir = _abs(os.environ.get("GPU_RESUME_DIR") or str(workdir / "resume"))
    if not (src.is_dir() or src.is_file()):
        raise ValueError("resume checkpoint %s not found" % src)
    if src == resume_dir or resume_dir in src.parents or src in resume_dir.parents:
        raise ValueError("--resume %s overlaps the resume dir %s" % (src, resume_dir))
    resume_dir.mkdir(parents=True, exist_ok=True)
    _clear_dir(resume_dir)  # exactly this checkpoint, nothing left from an earlier run
    moved = False
    if src.is_dir() and src == workdir / RESUME_DL:
        try:
            os.rmdir(str(resume_dir))
            os.rename(str(src), str(resume_dir))  # one copy on disk, not two (D44)
            moved = True
        except OSError:
            resume_dir.mkdir(parents=True, exist_ok=True)
    if moved:
        pass
    elif src.is_dir():
        _copy_tree(src, resume_dir)
    else:
        extract(src, resume_dir)
    env["GPU_RESUME_DIR"] = str(resume_dir)
    if not _dir_has_content(ckpt_dir):
        _copy_tree(resume_dir, ckpt_dir)
    tee.say("restored checkpoint into %s" % resume_dir)


def _final_sync_min_age(code):
    """How old checkpoint files must be for the final sync: after a clean exit every
    write is finished; after a crash or kill a fresh file may be half-written."""
    return 0.0 if code == 0 else UNSAFE_WINDOW_S


def gpu_devices():
    """One "name, memory" string per GPU nvidia-smi sees; [] without nvidia-smi or on any
    failure (the device line is informational and must never stop a job)."""
    smi = shutil.which("nvidia-smi")
    if not smi:
        return []
    try:
        proc = subprocess.run(
            [smi, "--query-gpu=name,memory.total", "--format=csv,noheader"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=SMI_TIMEOUT_S,
            check=False,
        )
    except Exception:
        return []
    if proc.returncode != 0:
        return []
    text = proc.stdout.decode("utf-8", "replace")
    lines = [" ".join(line.split())[:120] for line in text.splitlines()]
    return [line for line in lines if line][:MAX_DEVICES]


def run(argv=None):
    notes = []
    previous_handlers = install_signal_handlers()
    # A storage token is a secret: out of the environment before anything can inherit it
    # (pip, the job, its subprocesses). Only the runner's own storage calls use it.
    token = os.environ.pop(STORAGE_TOKEN_ENV, None) or None
    token_file = os.environ.pop(STORAGE_TOKEN_FILE_ENV, None)
    if token_file:
        # preferred channel: the file is read once and deleted, so neither this process's
        # environment nor the disk keeps the token (D44)
        try:
            with open(token_file, encoding="utf-8") as fh:
                token = fh.read().strip() or token
        except OSError:
            pass
        with contextlib.suppress(OSError):
            os.unlink(token_file)
    tee = None
    exit_file = None
    channel = None
    code = 1
    try:
        try:
            args = _parse_args(argv, notes)
        except SystemExit as exc:  # argparse already printed why
            code = exc.code if isinstance(exc.code, int) else 2
            exit_file = _default_exit_file(argv)
            return code
        started = time.monotonic()
        workdir = _resolve_workdir(args)
        exit_file = _abs(args.exit_file) if args.exit_file else workdir / "EXIT"
        with contextlib.suppress(OSError):
            exit_file.unlink()  # a stale EXIT from an earlier run must not end this one early
        workdir.mkdir(parents=True, exist_ok=True)
        log_path = _abs(args.log_file) if args.log_file else workdir / "job.log"
        if log_path.exists():
            with contextlib.suppress(OSError):
                os.replace(str(log_path), str(log_path) + ".prev")
        tee = Tee(log_path)
        tee.event("hello", v=PROTOCOL_VERSION, runner=RUNNER_VERSION)
        devices = gpu_devices()
        if devices:
            tee.event("device", gpus=devices)
        for note in notes:
            tee.say(note)
        root = _prepare_bundle(args, workdir, tee)
        with open(str(root / MANIFEST), encoding="utf-8") as fh:
            manifest = json.load(fh)
        code_dir = root / "code"
        runner_dir = root / "gpu_runner"

        env = dict(os.environ)
        env["GPU_ROUTER_PROTOCOL"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        job_id = env.get("GPU_ROUTER_JOB_ID") or ""
        attempt = _int_or_none(env.get("GPU_ROUTER_ATTEMPT"))
        ckpt_dir = _abs(env.get("GPU_CHECKPOINT_DIR") or str(workdir / "checkpoints"))
        out_dir = _abs(env.get("GPU_OUTPUT_DIR") or str(workdir / "outputs"))
        env["GPU_CHECKPOINT_DIR"] = str(ckpt_dir)
        env["GPU_OUTPUT_DIR"] = str(out_dir)
        env["GPU_DATA_DIR"] = str(_abs(env.get("GPU_DATA_DIR") or "/data"))
        _claim_dir(ckpt_dir, job_id, tee, "checkpoint dir")
        _claim_dir(out_dir, job_id, tee, "output dir")
        with contextlib.suppress(OSError):
            (ckpt_dir / REQUEST_FILE).unlink()  # a request from an earlier run is not ours
        py_path = [str(runner_dir), str(code_dir)]
        if env.get("PYTHONPATH"):
            py_path.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(py_path)

        # ---- checkpoint storage (phase 5)
        cache_root = _abs(os.environ.get("GPU_RUNNER_CACHE") or str(workdir / ".gpu-cache"))
        data_items = _data_items(os.environ.get("GPU_DATA"), tee)
        smod = None
        store = None
        wants_hf = any(str(r or "").startswith("hf://") for r in (args.resume, args.resume_uri))
        if args.storage or data_items or wants_hf:
            try:
                smod = _load_storage(root)
            except Exception as exc:
                tee.say("checkpoint storage code is missing from the bundle (%s)" % exc)
        if args.storage and smod is not None:
            store, why = _open_storage(args.storage, token, smod, cache_root, tee)
            if store is None:
                tee.say(
                    "checkpoint storage %s is not usable (%s); checkpoints stay on this "
                    "machine" % (args.storage, why)
                )
        latest = None
        latest_error = None
        seq_start = args.ckpt_seq_start
        if store is not None and job_id:
            try:
                latest = smod.read_latest(store, job_id)
            except Exception as exc:
                latest_error = getattr(exc, "message", exc)
                tee.say("could not read the latest checkpoint pointer: %s" % latest_error)
            if latest is not None and latest["seq"] >= seq_start:
                seq_start = latest["seq"] + 1

        try:
            _restore(
                args,
                env,
                workdir,
                ckpt_dir,
                tee,
                (smod, store, token, cache_root),
                latest,
                latest_error,
            )
        except ResumeUnavailable as exc:
            tee.say(
                "could not download the checkpoint to resume from (%s); the job did not "
                "start (exit %d), gpu-router tries again elsewhere" % (exc, INSTALL_FAILED_EXIT)
            )
            tee.event("resume_failed")
            code = INSTALL_FAILED_EXIT
            return code
        except OSError as exc:
            # a full disk or unreadable files while restoring is this machine's problem,
            # not the script's: exit 90 so the engine tries elsewhere (D44)
            tee.say(
                "could not restore the checkpoint on this machine (%s); the job did not "
                "start (exit %d)" % (exc, INSTALL_FAILED_EXIT)
            )
            tee.event("resume_failed")
            code = INSTALL_FAILED_EXIT
            return code
        finally:
            if (workdir / RESUME_DL).exists():
                shutil.rmtree(str(workdir / RESUME_DL), ignore_errors=True)

        if not args.skip_install:
            rc = install_deps(manifest, code_dir, args.python, tee)
            if rc != 0:
                tee.say(
                    "dependency install failed (pip exit %d); the job did not start "
                    "(exit %d = install failed)" % (rc, INSTALL_FAILED_EXIT)
                )
                tee.event("install_failed", code=rc)
                code = INSTALL_FAILED_EXIT
                return code

        if data_items:
            try:
                _materialize_data(
                    data_items, Path(env["GPU_DATA_DIR"]), token, smod, cache_root, tee
                )
            except Exception as exc:
                tee.say(
                    "datasets are not available (%s); the job did not start (exit %d)"
                    % (exc, INSTALL_FAILED_EXIT)
                )
                tee.event("data_failed")
                code = INSTALL_FAILED_EXIT
                return code

        interval_min = args.checkpoint_interval_min
        if interval_min is None:
            interval_min = float(manifest.get("checkpoint_interval_min") or 0)
        interval_s = interval_min * 60 if args.interval_s is None else args.interval_s
        sync_dir = _abs(args.checkpoint_sync_dir) if args.checkpoint_sync_dir else None
        syncer = CheckpointSyncer(
            tee,
            ckpt_dir,
            None if store is not None else sync_dir,
            seq_start,
            interval_s,
            store=store,
            smod=smod,
            job_id=job_id or "unknown",
            attempt=attempt,
            keep=args.keep,
            stage_root=workdir / ".gpu-stage",
        )
        syncer.last_published = latest
        if store is not None and job_id:
            secret_names = [
                n.strip() for n in os.environ.get("GPU_SECRET_NAMES", "").split(",") if n.strip()
            ]
            redact = Redactor([token or ""] + [os.environ.get(n, "") for n in secret_names])
            channel = StatusChannel(
                tee,
                store,
                smod,
                job_id,
                attempt,
                syncer,
                ckpt_dir,
                redact,
                args.status_push_s,
                args.control_poll_s,
            )
        state = {"last_beat": time.monotonic()}

        def on_line(text):
            if text.startswith(PREFIX):
                try:
                    body = json.loads(text[len(PREFIX) :])
                except ValueError:
                    return
                if not isinstance(body, dict):
                    return
                if body.get("t") == "metric":
                    step = body.get("step")
                    if isinstance(step, int) and not isinstance(step, bool):
                        syncer.last_step = step
                if channel is not None:
                    channel.note_line(body)

        def on_tick():
            now = time.monotonic()
            if args.heartbeat_s > 0 and now - state["last_beat"] >= args.heartbeat_s:
                state["last_beat"] = now
                tee.event("heartbeat", ts=round(time.time(), 3), elapsed=round(now - started, 1))
            if syncer.due():
                syncer.start_background()

        argv_child = _entry_argv(manifest, args.python)
        shown = argv_child[2:] if manifest["entrypoint"].get("script") else argv_child
        if channel is not None:
            channel.start()
        tee.say("running %s" % " ".join(shown))
        code = _run_streaming(argv_child, str(code_dir), env, tee, on_line, on_tick)
        syncer.wait()  # a periodic sync still running
        syncer.sync(final=True, min_age_s=_final_sync_min_age(code))
        return code
    except Terminated as exc:
        code = 128 + exc.signum
        if tee is not None:
            tee.say("stopped by signal %d" % exc.signum)
        return code
    except Exception as exc:
        if tee is not None:
            tee.say("runner error: %s" % exc)
        else:
            sys.stderr.write("gpu-router: runner error: %s\n" % exc)
        code = 1 if code == 0 else code
        return code
    finally:
        if threading.current_thread() is threading.main_thread():
            for sig in (signal.SIGTERM, signal.SIGINT):  # nothing may skip the EXIT file
                with contextlib.suppress(ValueError, OSError):
                    signal.signal(sig, signal.SIG_IGN)
        if tee is not None:
            tee.event("exit", code=code)
        if exit_file is None:
            exit_file = _default_exit_file(argv)
        try:
            _write_atomic(exit_file, "%d\n" % code)
        except Exception as exc:
            if tee is not None:
                tee.say("could not write %s: %s" % (exit_file, exc))
        if channel is not None:
            with contextlib.suppress(Exception):
                channel.stop(code)  # last push: the tail ends with the exit line
        if tee is not None:
            tee.close()
        restore_signal_handlers(previous_handlers)


def _int_or_none(raw):
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        return None


def _data_items(raw, tee):
    """GPU_DATA (JSON list of {mount, local|uri, sha256?}) -> list; bad JSON is a note."""
    if not raw:
        return []
    try:
        items = json.loads(raw)
    except ValueError:
        tee.say("ignored GPU_DATA (not JSON)")
        return []
    if not isinstance(items, list):
        tee.say("ignored GPU_DATA (not a list)")
        return []
    return [i for i in items if isinstance(i, dict)]


def main(argv=None):
    return run(argv)


if __name__ == "__main__":
    sys.exit(main())
