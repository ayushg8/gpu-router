"""Checkpoint, data and status storage shared by the runner and the daemon.

STDLIB ONLY, Python 3.8+ (invariant 14, D2): this file ships in every bundle as
`gpu_runner/storage.py`. `huggingface_hub` is imported lazily, only for `hf://` roots; the
runner installs it into a private directory when the image lacks it (bootstrap.py). The
daemon imports this module too (`gpu_router.checkpoint.storage` wraps it with types), so
both sides read and write exactly the same layout.

Roots (one per gpu-router install):

    file:///abs/dir              LocalStore: a directory on this machine (the local provider,
                                 tests). Files are written tmp + rename.
    hf://buckets/<ns>/<name>     HfBucketStore: a private Hugging Face Storage Bucket (mutable,
                                 no version history, Xet dedup; docs/notes/hf-hub.md).

Layout under a root (keys are "/"-separated, no leading slash):

    jobs/<job_id>/owner.json                   {"attempt": n}: the only attempt allowed to publish
    jobs/<job_id>/latest.json                  pointer to the newest complete checkpoint
    jobs/<job_id>/ckpt-NNNN/<files>            one checkpoint = the checkpoint dir's files
    jobs/<job_id>/ckpt-NNNN/.gpu-ckpt.json     its manifest, written after the files
    jobs/<job_id>/attempts/<n>/heartbeat.json  runner status (ts, step, metrics, last ckpt)
    jobs/<job_id>/attempts/<n>/log-tail.json   last LOG_TAIL_LINES log lines + absolute count
    jobs/<job_id>/attempts/<n>/control.json    written by the daemon (checkpoint request)
    jobs/<job_id>/attempts/<n>/control-ack.json  written by the runner (request answered)
    datasets/<sha256>/<files>                  uploaded datasets, content-addressed
    datasets/<sha256>/.gpu-data.json           manifest, written last (= upload complete)

A checkpoint is published files first, then its manifest, then latest.json, so a reader
that follows latest.json never sees a torn checkpoint even though buckets have no
transactions (a runtime that dies mid-upload leaves an unreferenced ckpt dir behind).

Errors: every failure is a StorageError; `missing` marks "no such file/prefix" and
`retryable` separates transient trouble (network, 429, 5xx) from auth/permission problems.
"""

import contextlib
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

LAYOUT_VERSION = 1
LOG_TAIL_LINES = 1000  # lines kept in log-tail.json
LOG_TAIL_LINE_MAX = 2000  # characters per line in log-tail.json (the full log comes after)
CKPT_MANIFEST = ".gpu-ckpt.json"
DATA_MANIFEST = ".gpu-data.json"
HF_PREFIX = "hf://buckets/"
FILE_PREFIX = "file://"
HF_REQUIREMENT = "huggingface_hub>=1.32,<2"


class StorageError(Exception):
    """A storage call failed. `missing`: the key does not exist. `retryable`: try again
    later (network, rate limit, server error); False for auth/permission problems."""

    def __init__(self, message, retryable=True, missing=False):
        Exception.__init__(self, message)
        self.message = message
        self.retryable = retryable
        self.missing = missing


# --------------------------------------------------------------------------- layout


def job_prefix(job_id):
    return "jobs/%s" % job_id


def ckpt_name(seq):
    return "ckpt-%04d" % int(seq)


def ckpt_key(job_id, seq):
    return "%s/%s" % (job_prefix(job_id), ckpt_name(seq))


def latest_key(job_id):
    return job_prefix(job_id) + "/latest.json"


def owner_key(job_id):
    return job_prefix(job_id) + "/owner.json"


def attempt_prefix(job_id, n):
    return "%s/attempts/%d" % (job_prefix(job_id), int(n))


def heartbeat_key(job_id, n):
    return attempt_prefix(job_id, n) + "/heartbeat.json"


def log_tail_key(job_id, n):
    return attempt_prefix(job_id, n) + "/log-tail.json"


def control_key(job_id, n):
    return attempt_prefix(job_id, n) + "/control.json"


def ack_key(job_id, n):
    return attempt_prefix(job_id, n) + "/control-ack.json"


def dataset_key(sha256):
    return "datasets/%s" % sha256


def _clean_key(key):
    key = str(key).strip("/")
    parts = key.split("/")
    if not key or any(p in ("", ".", "..") for p in parts):
        raise StorageError("bad storage key %r" % key, retryable=False)
    return key


def dumps(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _fsync_fd(fd, full=False):
    """Flush a file's data to stable storage. `full` on macOS asks the drive to empty its
    own cache too (F_FULLFSYNC; plain fsync stops at the drive), for the pointer files a
    crash must never leave empty."""
    if full and sys.platform == "darwin":
        try:
            import fcntl

            fcntl.fcntl(fd, getattr(fcntl, "F_FULLFSYNC", 51))
            return
        except (ImportError, OSError, ValueError):
            pass
    with contextlib.suppress(OSError):
        os.fsync(fd)


def _fsync_path(path, full=False):
    """fsync a file or a directory by path (a directory: its entries, i.e. renames)."""
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        _fsync_fd(fd, full)
    finally:
        os.close(fd)


# --------------------------------------------------------------------------- local


class LocalStore(object):
    """A directory on this machine. Keys map to paths under `root`."""

    kind = "local"

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root_uri = self.root.as_uri()

    def __repr__(self):
        return "LocalStore(%s)" % self.root

    def path(self, key):
        return self.root / _clean_key(key)

    def uri(self, key):
        return self.path(key).as_uri()

    def key_of(self, uri):
        """The key a file:// URI under this root names, else None."""
        if not str(uri).startswith(FILE_PREFIX):
            return None
        path = Path(_file_path(uri)).resolve()
        try:
            rel = path.relative_to(self.root)
        except ValueError:
            return None
        return str(rel).replace(os.sep, "/") or None

    def read_bytes(self, key):
        try:
            return self.path(key).read_bytes()
        except FileNotFoundError:
            return None
        except IsADirectoryError:
            return None
        except OSError as exc:
            raise StorageError("could not read %s: %s" % (key, exc)) from None

    def write_bytes(self, key, data):
        self.write_many({key: data})

    def write_many(self, items):
        """Each file: tmp, fsync, rename, fsync of the directory, so a power loss never
        leaves latest.json (or a manifest) pointing at data that never reached the disk."""
        for key, data in items.items():
            target = self.path(key)
            tmp = None
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                fd, tmp = tempfile.mkstemp(dir=str(target.parent), prefix=".tmp-")
                with os.fdopen(fd, "wb") as fh:
                    fh.write(data)
                    fh.flush()
                    _fsync_fd(fh.fileno(), full=True)
                os.replace(tmp, str(target))
                _fsync_path(target.parent)
            except OSError as exc:
                if tmp is not None:
                    with contextlib.suppress(Exception):
                        os.unlink(tmp)
                raise StorageError("could not write %s: %s" % (key, exc)) from None

    def stat(self, key):
        """(size, change token) or None when absent."""
        try:
            st = self.path(key).stat()
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise StorageError("could not stat %s: %s" % (key, exc)) from None
        return st.st_size, "%d:%d" % (st.st_size, st.st_mtime_ns)

    def upload_dir(self, src, key, files, move=False):
        """Put `files` (paths relative to src) under `key`/, replacing what was there.
        `move`: src is a private staging dir that may be renamed into place."""
        src = Path(src)
        dest = self.path(key)
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = Path(tempfile.mkdtemp(dir=str(dest.parent), prefix=".up-%s-" % dest.name))
            try:
                if move and _same_device(src, dest.parent):
                    os.rmdir(str(tmp))
                    os.rename(str(src), str(tmp))
                else:
                    for rel in files:
                        out = tmp / rel
                        out.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(str(src / rel), str(out))
                # the data is on disk before anything (manifest, latest.json) names it
                for rel in files:
                    _fsync_path(tmp / rel)
                for d in sorted({(tmp / rel).parent for rel in files}, reverse=True):
                    _fsync_path(d)
                if dest.exists():
                    shutil.rmtree(str(dest))
                os.rename(str(tmp), str(dest))
                _fsync_path(dest.parent)
            finally:
                if tmp.exists():
                    shutil.rmtree(str(tmp), ignore_errors=True)
        except OSError as exc:
            raise StorageError("could not store %s: %s" % (key, exc)) from None

    def list_files(self, prefix):
        """[(key, size)] of every file under prefix/ (recursive, sorted)."""
        base = self.path(prefix)
        out = []
        if not base.is_dir():
            return out
        for root, _dirs, names in os.walk(str(base)):
            for name in names:
                if name.startswith(".tmp-"):
                    continue
                full = Path(root) / name
                rel = str(full.relative_to(self.root)).replace(os.sep, "/")
                with contextlib.suppress(OSError):
                    out.append((rel, full.stat().st_size))
        return sorted(out)

    def download_dir(self, key, dest, skip=()):
        """Copy every file under key/ into dest (relative paths kept). Returns the count."""
        base = self.path(key)
        if not base.is_dir():
            raise StorageError("%s not found" % key, retryable=False, missing=True)
        count = 0
        for fkey, _size in self.list_files(key):
            rel = fkey[len(_clean_key(key)) + 1 :]
            if rel.split("/")[-1] in skip:
                continue
            out = Path(dest) / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copyfile(str(self.root / fkey), str(out))
            except OSError as exc:
                raise StorageError("could not copy %s: %s" % (fkey, exc)) from None
            count += 1
        return count

    def delete_prefix(self, prefix):
        target = self.path(prefix)
        try:
            if target.is_dir():
                shutil.rmtree(str(target))
            elif target.exists():
                target.unlink()
        except OSError as exc:
            raise StorageError("could not delete %s: %s" % (prefix, exc)) from None

    def ensure(self):
        try:
            self.root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise StorageError(
                "could not create %s: %s" % (self.root, exc), retryable=False
            ) from None


def _same_device(a, b):
    try:
        return os.stat(str(a)).st_dev == os.stat(str(b)).st_dev
    except OSError:
        return False


def _file_path(uri):
    from urllib.parse import unquote, urlparse

    parsed = urlparse(str(uri))
    return unquote(parsed.path)


# --------------------------------------------------------------------------- hugging face


def _status_of(exc):
    response = getattr(exc, "response", None)
    code = getattr(response, "status_code", None)
    return code if isinstance(code, int) else None


def _hf_error(exc, what):
    """Map a huggingface_hub / network exception onto StorageError (never the token)."""
    code = _status_of(exc)
    name = type(exc).__name__
    if name in ("EntryNotFoundError", "RemoteEntryNotFoundError") or code == 404:
        return StorageError("%s: not found" % what, retryable=False, missing=True)
    if name in ("BucketNotFoundError", "RepositoryNotFoundError"):
        return StorageError(
            "%s: bucket not found (or the token cannot see it)" % what, retryable=False
        )
    if code in (401, 403):
        return StorageError(
            "%s: hugging face refused the token (HTTP %d)" % (what, code), retryable=False
        )
    if code is not None:
        return StorageError("%s: hugging face returned HTTP %d" % (what, code))
    text = str(exc).splitlines()[0][:200] if str(exc) else name
    return StorageError("%s: %s (%s)" % (what, text, name))


class HfBucketStore(object):
    """A Hugging Face Storage Bucket `<ns>/<name>` (huggingface_hub >= 1.32 bucket API)."""

    kind = "hf"

    def __init__(self, bucket_id, token=None, api=None):
        self.bucket_id = bucket_id.strip("/")
        if self.bucket_id.count("/") != 1:
            raise StorageError("bucket id must be <namespace>/<name>", retryable=False)
        self.root_uri = HF_PREFIX + self.bucket_id
        self._token = token
        self._api = api

    def __repr__(self):
        return "HfBucketStore(%s)" % self.bucket_id

    @property
    def api(self):
        if self._api is None:
            # no tqdm bars in job logs or the daemon log (set before the first import)
            os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
            try:
                from huggingface_hub import HfApi
            except ImportError:
                raise StorageError(
                    "huggingface_hub is not installed (need %s)" % HF_REQUIREMENT,
                    retryable=False,
                ) from None
            self._api = HfApi(token=self._token) if self._token else HfApi()
        return self._api

    def uri(self, key):
        return "%s/%s" % (self.root_uri, _clean_key(key))

    def key_of(self, uri):
        prefix = self.root_uri + "/"
        text = str(uri)
        return text[len(prefix) :].strip("/") or None if text.startswith(prefix) else None

    def _call(self, what, fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except StorageError:
            raise
        except Exception as exc:
            raise _hf_error(exc, what) from None

    def _info(self, key):
        infos = self._call(
            "stat %s" % key,
            lambda: list(self.api.get_bucket_paths_info(self.bucket_id, [_clean_key(key)])),
        )
        return infos[0] if infos else None

    def stat(self, key):
        info = self._info(key)
        if info is None:
            return None
        size = int(getattr(info, "size", 0) or 0)
        token = getattr(info, "xet_hash", None) or getattr(info, "mtime", None) or size
        return size, str(token)

    def read_bytes(self, key):
        info = self._info(key)
        if info is None:
            return None
        tmp = tempfile.mkdtemp(prefix="gpu-hf-")
        try:
            out = os.path.join(tmp, "f")
            self._call(
                "download %s" % key,
                self.api.download_bucket_files,
                self.bucket_id,
                [(info, out)],
            )
            try:
                with open(out, "rb") as fh:
                    return fh.read()
            except FileNotFoundError:
                return b"" if int(getattr(info, "size", 0) or 0) == 0 else None
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def write_bytes(self, key, data):
        self.write_many({key: data})

    def write_many(self, items):
        add = [(bytes(data), _clean_key(key)) for key, data in items.items()]
        if add:
            self._call(
                "upload %s" % ", ".join(k for _d, k in add),
                self.api.batch_bucket_files,
                self.bucket_id,
                add=add,
            )

    def upload_dir(self, src, key, files, move=False):
        """Upload files (relative to src) under key/, then delete files under key/ that are
        not part of this upload (a re-published checkpoint must not keep stale files)."""
        base = _clean_key(key)
        src = Path(src)
        add = [(str(src / rel), "%s/%s" % (base, rel.replace(os.sep, "/"))) for rel in files]
        wanted = {dest for _p, dest in add}
        stale = [k for k, _s in self.list_files(base) if k not in wanted]
        if add:
            self._call("upload %s/" % base, self.api.batch_bucket_files, self.bucket_id, add=add)
        if stale:
            self._call(
                "clean %s/" % base, self.api.batch_bucket_files, self.bucket_id, delete=stale
            )

    def _entries(self, prefix):
        base = _clean_key(prefix)

        def listing():
            out = []
            for entry in self.api.list_bucket_tree(self.bucket_id, prefix=base, recursive=True):
                if getattr(entry, "type", "file") != "file":
                    continue
                path = str(getattr(entry, "path", ""))
                if path == base or path.startswith(base + "/"):
                    out.append(entry)
            return out

        try:
            return self._call("list %s/" % base, listing)
        except StorageError as exc:
            if exc.missing:
                return []
            raise

    def list_files(self, prefix):
        return sorted((str(e.path), int(getattr(e, "size", 0) or 0)) for e in self._entries(prefix))

    def download_dir(self, key, dest, skip=()):
        base = _clean_key(key)
        entries = [e for e in self._entries(base) if str(e.path) != base]
        if not entries:
            raise StorageError("%s not found" % base, retryable=False, missing=True)
        files = []
        for e in entries:
            rel = str(e.path)[len(base) + 1 :]
            if rel.startswith("/") or ".." in rel.split("/") or "\\" in rel:
                raise StorageError("unsafe path %r under %s" % (rel, base), retryable=False)
            if rel.split("/")[-1] in skip:
                continue
            out = Path(dest) / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            files.append((e, str(out)))
        if files:
            self._call("download %s/" % base, self.api.download_bucket_files, self.bucket_id, files)
        return len(files)

    def delete_prefix(self, prefix):
        keys = [k for k, _s in self.list_files(prefix)]
        if keys:
            self._call(
                "delete %s/" % prefix, self.api.batch_bucket_files, self.bucket_id, delete=keys
            )

    def ensure(self):
        """Create the bucket (private) if it does not exist."""
        self._call(
            "create bucket %s" % self.bucket_id,
            self.api.create_bucket,
            self.bucket_id,
            private=True,
            exist_ok=True,
        )


# --------------------------------------------------------------------------- open


def open_store(root_uri, token=None, api=None):
    """A store for `file:///dir` or `hf://buckets/<ns>/<name>`."""
    text = str(root_uri).rstrip("/")
    if text.startswith(HF_PREFIX):
        return HfBucketStore(text[len(HF_PREFIX) :], token=token, api=api)
    if text.startswith(FILE_PREFIX):
        return LocalStore(_file_path(text))
    if text.startswith("/"):
        return LocalStore(text)
    raise StorageError("unknown storage root %r" % root_uri, retryable=False)


def split_hf_uri(uri):
    """hf://buckets/<ns>/<name>/<key> -> ("hf://buckets/<ns>/<name>", key) or None."""
    text = str(uri)
    if not text.startswith(HF_PREFIX):
        return None
    parts = text[len(HF_PREFIX) :].strip("/").split("/", 2)
    if len(parts) < 3 or not all(parts):
        return None
    return HF_PREFIX + parts[0] + "/" + parts[1], parts[2]


def read_json(store, key):
    """Parsed JSON object at key, or None when absent or not a JSON object."""
    raw = store.read_bytes(key)
    if raw is None:
        return None
    try:
        data = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None
    return data if isinstance(data, dict) else None


# --------------------------------------------------------------------------- checkpoints


def publish_checkpoint(
    store,
    job_id,
    seq,
    src_dir,
    files,
    attempt=None,
    step=None,
    move=False,
    created_at=None,
    skip_upload=False,
):
    """Upload one checkpoint (files relative to src_dir) as ckpt-NNNN, then its manifest,
    then latest.json. `files` = [(relpath, size, sha256)]. Returns latest.json's dict.
    `created_at` defaults to now (the fake provider passes its simulated time).
    `skip_upload`: the files are already under ckpt-NNNN (a direct upload), write only
    the manifest and latest.json."""
    key = ckpt_key(job_id, seq)
    rels = [rel for rel, _size, _sha in files]
    now = time.time() if created_at is None else float(created_at)
    manifest = {
        "v": LAYOUT_VERSION,
        "job": job_id,
        "seq": int(seq),
        "attempt": attempt,
        "step": step,
        "created_at": round(now, 3),
        "files": [{"path": rel, "size": size, "sha256": sha} for rel, size, sha in files],
        "size": sum(size for _rel, size, _sha in files),
    }
    blob = dumps(manifest)
    if not skip_upload:
        store.upload_dir(src_dir, key, rels, move=move)
    store.write_bytes(key + "/" + CKPT_MANIFEST, blob)
    latest = {
        "v": LAYOUT_VERSION,
        "job": job_id,
        "seq": int(seq),
        "uri": store.uri(key),
        "attempt": attempt,
        "step": step,
        "size": manifest["size"],
        "files": len(files),
        "sha256": hashlib.sha256(blob).hexdigest(),
        "created_at": manifest["created_at"],
    }
    store.write_bytes(latest_key(job_id), dumps(latest))
    return latest


def read_latest(store, job_id):
    """latest.json as a dict with int seq and str uri, or None."""
    data = read_json(store, latest_key(job_id))
    if not data:
        return None
    try:
        seq = int(data.get("seq"))
    except (TypeError, ValueError):
        return None
    if seq < 1 or not isinstance(data.get("uri"), str):
        return None
    data["seq"] = seq
    return data


def read_owner(store, job_id):
    data = read_json(store, owner_key(job_id))
    if not data:
        return None
    try:
        return int(data.get("attempt"))
    except (TypeError, ValueError):
        return None


def claim(store, job_id, attempt):
    """Mark `attempt` as the only one allowed to publish checkpoints for the job (the
    daemon writes it before each submit; an older runner that is still alive stops)."""
    store.write_bytes(owner_key(job_id), dumps({"attempt": int(attempt), "ts": time.time()}))


def restore_checkpoint(store, key, dest):
    """Download checkpoint `key` (ckpt-NNNN) into dest, without its manifest."""
    return store.download_dir(key, dest, skip=(CKPT_MANIFEST,))


_CKPT_DIR_RE = re.compile(r"/ckpt-(\d+)/")
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")


def list_checkpoints(store, job_id):
    """Sorted seqs of every ckpt-NNNN directory storage holds for the job (complete or
    not; `verify_checkpoint` tells them apart)."""
    seqs = set()
    for fkey, _size in store.list_files(job_prefix(job_id)):
        m = _CKPT_DIR_RE.search("/" + fkey)
        if m:
            seqs.add(int(m.group(1)))
    return sorted(seqs)


def seq_of(uri):
    """The NNNN of a checkpoint URI ending in ckpt-NNNN, else None."""
    m = re.search(r"ckpt-(\d+)/?$", str(uri))
    return int(m.group(1)) if m else None


def verify_checkpoint(store, key, local_dir):
    """Compare the files in local_dir (a restored copy of checkpoint `key`, or the local
    checkpoint dir itself) with the checkpoint's manifest: every file present with its size
    and sha256, nothing extra. Returns why they differ, or None when they match. A
    checkpoint without a manifest never finished publishing."""
    manifest = read_json(store, key + "/" + CKPT_MANIFEST)
    if manifest is None:
        return "it has no manifest (its upload never finished)"
    if not isinstance(manifest.get("files"), list):
        return None  # a manifest without a file list (older layout): nothing to compare
    local_dir = Path(local_dir)
    want = {}
    for item in manifest["files"]:
        if isinstance(item, dict) and isinstance(item.get("path"), str):
            want[item["path"]] = item
    have = set()
    for root, _dirs, names in os.walk(str(local_dir)):
        for name in names:
            if name == CKPT_MANIFEST:
                continue
            have.add(str((Path(root) / name).relative_to(local_dir)).replace(os.sep, "/"))
    extra = sorted(have - set(want))
    if extra:
        return "%d file(s) the manifest does not list (%s)" % (len(extra), extra[0])
    for rel, item in sorted(want.items()):
        path = local_dir / rel
        if rel not in have or not path.is_file():
            return "%s is missing" % rel
        size = item.get("size")
        if isinstance(size, int) and path.stat().st_size != size:
            return "%s has %d bytes, the manifest says %d" % (rel, path.stat().st_size, size)
        sha = item.get("sha256")
        if isinstance(sha, str) and _SHA_RE.match(sha):
            h = hashlib.sha256()
            with open(str(path), "rb") as fh:
                for block in iter(lambda: fh.read(1 << 20), b""):
                    h.update(block)
            if h.hexdigest() != sha:
                return "%s does not match its sha256" % rel
    return None


# --------------------------------------------------------------------------- datasets


def dataset_complete(store, sha256):
    return read_json(store, dataset_key(sha256) + "/" + DATA_MANIFEST) is not None


def upload_dataset(store, sha256, src, files):
    """Upload a dataset (files = [(relpath, size)]) under datasets/<sha>, manifest last."""
    key = dataset_key(sha256)
    rels = [rel for rel, _size in files]
    if Path(src).is_file():
        # a single-file dataset: the file keeps its name inside the mount dir
        store.upload_dir(Path(src).parent, key, [Path(src).name])
    else:
        store.upload_dir(src, key, rels)
    manifest = {
        "v": LAYOUT_VERSION,
        "sha256": sha256,
        "files": len(files),
        "size": sum(size for _rel, size in files),
        "created_at": round(time.time(), 3),
    }
    store.write_bytes(key + "/" + DATA_MANIFEST, dumps(manifest))
    return store.uri(key)
