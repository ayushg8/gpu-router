"""CheckpointHub: the daemon's side of checkpoint handoff and data movement (phase 5).

Storage backends (config.yaml `checkpoint:`; gpu_router/runner/storage.py has the layout):

- **hf**: a private Hugging Face Storage Bucket, `<namespace>/<checkpoint.bucket>`
  (default bucket `gpu-router`, namespace from whoami, cached in
  `<data dir>/storage/hf.json`). Needs the Keychain secret HF_TOKEN (`gpu login hf`).
  Remote runtimes (Kaggle, Colab) write checkpoints, heartbeats and log tails here and
  read checkpoint requests; they get HF_TOKEN_REMOTE (else HF_TOKEN) as the job secret
  GPU_STORAGE_TOKEN through their adapter's secret channel.
- **local**: `<data dir>/storage/` (or `checkpoint.local_dir`), a plain directory. Runs on
  this Mac (adapter kinds in `local_kinds`) always use it, so the local runner never needs
  huggingface_hub; the daemon copies a checkpoint between backends when a job moves
  between this Mac and a remote provider (`checkpoint_for`).

`backend: auto` (default) = hf for remote runs when a token exists, local for local runs.
Without a token, remote runs get no storage: their checkpoints stay on the provider's
machine as in phase 3 (the job restarts from scratch if it has to move) and the engine
says so once per job. `backend: off` turns everything off.

The hub never touches the Store (invariant 9 spirit): the engine reads/writes the DB on
the event loop and calls the blocking methods here through `call()` (small executor,
bounded) or `call(bulk=True)` (dataset uploads, checkpoint copies: unbounded).
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import hashlib
import json
import os
import shutil
import tempfile
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import Executor, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import SecretStr

from gpu_router.checkpoint import tokens
from gpu_router.checkpoint.data import DataDigest, DataIndex
from gpu_router.checkpoint.data import digest as data_digest
from gpu_router.checkpoint.storage import (
    CKPT_MANIFEST,
    DATA_MANIFEST,
    HF_PREFIX,
    Storage,
    StorageError,
    ack_key,
    control_key,
    dataset_key,
    heartbeat_key,
    job_prefix,
    log_tail_key,
    open_storage,
    split_hf_uri,
)

if TYPE_CHECKING:
    from gpu_router.clock import Clock
    from gpu_router.config import CheckpointConfig, Config
    from gpu_router.models import Checkpoint
    from gpu_router.paths import Paths

__all__ = [
    "LOCAL_KINDS",
    "STORAGE_TOKEN_ENV",
    "AttemptStorage",
    "CheckpointAck",
    "CheckpointHub",
    "StorageStatus",
    "StoredCheckpoint",
]

T = TypeVar("T")

#: Adapter kinds whose runner runs on this Mac (can use the local backend directly).
LOCAL_KINDS: frozenset[str] = frozenset({"local"})
#: The fake providers' kind; a local kind in test mode when `checkpoint.fake_storage` is on.
FAKE_KIND = "fake"
#: Job secret carrying the storage token to a remote runner (bootstrap pops it at start).
STORAGE_TOKEN_ENV = "GPU_STORAGE_TOKEN"  # noqa: S105 - a name
SMALL_TIMEOUT_S = 60.0
BULK_TIMEOUT_S = 3 * 3600.0  # a dataset upload or checkpoint copy
RETRY_NO_TOKEN_S = 60.0
RETRY_TRANSIENT_S = 300.0
RETRY_PERMANENT_S = 900.0
ENV_REAL_PROVIDERS = "GPU_ROUTER_REAL_PROVIDERS"
LOGIN_HINT = "run `gpu login hf` (a token with write access to Storage Buckets)"
REMOTE_LOGIN_HINT = (
    "run `gpu login hf --remote` with a fine-grained token that can only write your "
    "buckets: remote runtimes can read the token they get"
)
TOKEN_RECHECK_S = 300.0  # re-read the Keychain this often to notice a new token


def _fingerprint(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()[:16]


def storage_dir(paths: Paths) -> Path:
    return paths.home / "storage"


@dataclass(frozen=True, slots=True)
class StorageStatus:
    """What `gpu providers` / `/doctor` / engine notes can say about storage."""

    backend: str  # configured: auto | hf | local | off
    local_uri: str | None
    hf_uri: str | None  # set once the bucket was reached
    reason: str | None = None  # why remote runs have no storage (None = they do, or unknown)
    hint: str | None = None


@dataclass(frozen=True, slots=True)
class StoredCheckpoint:
    """A checkpoint as storage describes it (latest.json or a control ack)."""

    seq: int
    uri: str
    attempt: int | None = None
    step: int | None = None
    size: int | None = None
    sha256: str | None = None
    created_at: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> StoredCheckpoint | None:
        if not data:
            return None
        try:
            seq = int(data["seq"])
        except (KeyError, TypeError, ValueError):
            return None
        uri = data.get("uri")
        if seq < 1 or not isinstance(uri, str) or not uri:
            return None
        return cls(
            seq=seq,
            uri=uri,
            attempt=_opt_int(data.get("attempt")),
            step=_opt_int(data.get("step")),
            size=_opt_int(data.get("size")),
            sha256=data.get("sha256") if isinstance(data.get("sha256"), str) else None,
            created_at=_opt_float(data.get("created_at")),
        )


@dataclass(frozen=True, slots=True)
class CheckpointAck:
    """The runner's answer to a checkpoint request (control-ack.json)."""

    request_id: str
    new: bool
    checkpoint: StoredCheckpoint | None


@dataclass(frozen=True, slots=True)
class AttemptStorage:
    """What one attempt's runner gets: env vars, secrets, notes for the job log."""

    kind: str  # hf | local
    root_uri: str
    env: dict[str, str] = field(default_factory=dict)
    secrets: dict[str, SecretStr] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def _opt_int(v: Any) -> int | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float) and v.is_integer():
        return int(v)
    return None


def _opt_float(v: Any) -> float | None:
    if isinstance(v, bool):
        return None
    return float(v) if isinstance(v, int | float) else None


def _real_opt_in(name: str) -> bool:
    raw = os.environ.get(ENV_REAL_PROVIDERS, "")
    return name in {p.strip() for p in raw.split(",") if p.strip()}


class CheckpointHub:
    def __init__(
        self,
        config: CheckpointConfig,
        paths: Paths,
        clock: Clock,
        *,
        test_mode: bool = False,
        local_kinds: frozenset[str] = LOCAL_KINDS,
        hf_api: Callable[[str | None], Any] | None = None,
        executor: Executor | None = None,
        bulk_executor: Executor | None = None,
    ) -> None:
        """`hf_api(token)` builds the HfApi-like client (tests inject a fake; None = the
        real huggingface_hub.HfApi). In test mode the real HF backend stays off unless
        GPU_ROUTER_REAL_PROVIDERS lists `hf` or a fake client is injected (invariant 20),
        and the Keychain is not read for it."""
        self.config = config
        self.paths = paths
        self.clock = clock
        self.test_mode = test_mode
        self.local_kinds = local_kinds
        self._hf_api = hf_api
        self._own_small = executor is None
        self._own_bulk = bulk_executor is None
        self._small: Executor = executor or ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="gpu-storage"
        )
        self._bulk: Executor = bulk_executor or ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="gpu-storage-bulk"
        )
        self._lock = threading.Lock()
        self._local: Storage | None = None
        self._hf: Storage | None = None
        self._hf_reason: str | None = None
        self._hf_hint: str | None = None
        self._hf_retry_at = 0.0
        self._hf_transient = False  # the last failure should pass (network, 5xx, a locked Keychain)
        self._hf_fp: str | None = None
        self._hf_checked_at = 0.0
        self._extra_roots: dict[str, Storage] = {}
        self.index = DataIndex(storage_dir(paths) / "data-index.json")

    @classmethod
    def from_config(cls, config: Config, paths: Paths, clock: Clock) -> CheckpointHub:
        kinds = LOCAL_KINDS
        if config.test_mode and config.checkpoint.fake_storage:
            kinds = kinds | {FAKE_KIND}  # the fake's simulated runner lives on this Mac (D43)
        return cls(config.checkpoint, paths, clock, test_mode=config.test_mode, local_kinds=kinds)

    def close(self) -> None:
        if self._own_small:
            self._small.shutdown(wait=False, cancel_futures=True)
        if self._own_bulk:
            self._bulk.shutdown(wait=False, cancel_futures=True)

    async def call(
        self,
        fn: Callable[..., T],
        *args: Any,
        bulk: bool = False,
        limit_s: float | None = SMALL_TIMEOUT_S,
        **kwargs: Any,
    ) -> T:
        """Run a blocking hub method off the event loop, bounded by `limit_s`
        (StorageError on expiry; the thread is left to finish). `bulk` (uploads, copies)
        uses its own pool so slow transfers never delay small reads; callers give bulk
        calls a long limit (BULK_TIMEOUT_S)."""
        loop = asyncio.get_running_loop()
        ex = self._bulk if bulk else self._small
        fut = loop.run_in_executor(ex, functools.partial(fn, *args, **kwargs))
        if limit_s is None:
            return await fut
        try:
            return await asyncio.wait_for(fut, limit_s)
        except TimeoutError:
            raise StorageError(f"storage call timed out after {limit_s:.0f}s") from None

    # ------------------------------------------------------------------ backends

    @property
    def enabled(self) -> bool:
        return self.config.backend != "off"

    def local(self) -> Storage | None:
        """The local backend (runs on this Mac), or None when storage is off."""
        if not self.enabled:
            return None
        with self._lock:
            if self._local is None:
                root = (
                    Path(self.config.local_dir).expanduser()
                    if self.config.local_dir
                    else storage_dir(self.paths)
                )
                root.mkdir(mode=0o700, parents=True, exist_ok=True)
                self._local = open_storage(root.resolve().as_uri())
            return self._local

    def _api(self, token: str | None) -> Any:
        return None if self._hf_api is None else self._hf_api(token)

    def _hf_allowed(self) -> str | None:
        """Why the HF backend is not considered at all, or None."""
        if self.config.backend in ("off", "local"):
            return f"checkpoint.backend is {self.config.backend}"
        if self.test_mode and self._hf_api is None and not _real_opt_in("hf"):
            return "hf storage is off in test mode"
        return None

    def remote_expected(self) -> bool:
        """True when remote runs are meant to get HF storage (so its absence is worth
        telling the user); False when config or test mode turned it off on purpose."""
        return self._hf_allowed() is None

    def hf(self) -> Storage | None:
        """The HF bucket (blocking: whoami + create once), or None with `status().reason`
        saying why. Failures are retried after a pause, never on every call."""
        blocked = self._hf_allowed()
        if blocked is not None:
            with self._lock:
                self._hf_reason, self._hf_hint = blocked, None
            return None
        now = self.clock.now()
        with self._lock:
            cached = self._hf
            if cached is not None and now - self._hf_checked_at < TOKEN_RECHECK_S:
                return cached
            if cached is None and now < self._hf_retry_at:
                return None
        token, problem = tokens.safe_admin_token()
        if cached is not None:
            if token is None and problem is not None:
                # the Keychain cannot be read right now (locked after login, D44): keep
                # the client we have instead of dropping storage for every placement
                with self._lock:
                    self._hf_checked_at = now
                return cached
            # a cached bucket client: keep it unless `gpu login hf` changed the token
            fp = _fingerprint(token.get_secret_value()) if token is not None else None
            with self._lock:
                self._hf_checked_at = now
                if fp is not None and fp == self._hf_fp:
                    return cached
                self._hf = None
        if token is None:
            self._hf_failed(
                problem or "no Hugging Face token in the Keychain",
                LOGIN_HINT if problem is None else "unlock the login Keychain",
                RETRY_NO_TOKEN_S,
                transient=problem is not None,
            )
            return None
        value = token.get_secret_value()
        try:
            bucket_id = self._bucket_id(value)
            store = open_storage(HF_PREFIX + bucket_id, token=value, api=self._api(value))
            store.ensure()
        except StorageError as exc:
            self._hf_failed(
                exc.message,
                None if exc.retryable else "check the token's permissions (`gpu login hf`)",
                RETRY_TRANSIENT_S if exc.retryable else RETRY_PERMANENT_S,
                transient=exc.retryable,
            )
            return None
        with self._lock:
            self._hf = store
            self._hf_fp = _fingerprint(value)
            self._hf_checked_at = now
            self._hf_reason = self._hf_hint = None
            self._hf_transient = False
        return store

    def _hf_failed(
        self, reason: str, hint: str | None, retry_s: float, *, transient: bool = False
    ) -> None:
        with self._lock:
            self._hf_reason = reason
            self._hf_hint = hint
            self._hf_retry_at = self.clock.now() + retry_s
            self._hf_transient = transient

    def hf_down_for_now(self) -> bool:
        """HF storage is meant to be used but cannot be reached for a reason that should
        pass (network, a 5xx, a locked Keychain). Placements wait for it (up to
        checkpoint.storage_wait_s) instead of silently running without it (D44). Reflects
        the last `hf()` call."""
        if self._hf_allowed() is not None:
            return False
        with self._lock:
            return self._hf is None and self._hf_transient

    def _unreachable(self, what: str) -> StorageError:
        why = self.status().reason or "no answer"
        return StorageError(f"{what}, which cannot be reached right now ({why})", retryable=True)

    def _bucket_id(self, token: str) -> str:
        """`<ns>/<name>`: checkpoint.bucket as given, or the token owner's namespace from
        whoami (cached per token fingerprint: /whoami-v2 is heavily rate limited)."""
        name = self.config.bucket.strip("/")
        if "/" in name:
            return name
        fp = _fingerprint(token)
        cache = storage_dir(self.paths) / "hf.json"
        try:
            data = json.loads(cache.read_text(encoding="utf-8"))
            if isinstance(data, dict) and data.get("token_fp") == fp and data.get("namespace"):
                return f"{data['namespace']}/{name}"
        except (OSError, ValueError):
            pass
        api = self._api(token)
        if api is None:
            os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
            try:
                from huggingface_hub import HfApi
            except ImportError as exc:  # a project dependency; only a broken install
                raise StorageError(f"huggingface_hub is missing ({exc})", retryable=False) from None
            api = HfApi(token=token)
        try:
            who = api.whoami()
        except Exception as exc:
            code = getattr(getattr(exc, "response", None), "status_code", None)
            if code in (401, 403):
                raise StorageError(
                    "hugging face rejected the token (whoami)", retryable=False
                ) from None
            raise StorageError(
                f"could not ask hugging face who the token belongs to: {type(exc).__name__}"
            ) from None
        namespace = str(who.get("name") or "") if isinstance(who, dict) else ""
        if not namespace:
            raise StorageError("hugging face whoami returned no user name", retryable=False)
        cache.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        tmp = cache.with_name(f".hf.json.{os.getpid()}")
        tmp.write_text(json.dumps({"namespace": namespace, "token_fp": fp}), encoding="utf-8")
        os.replace(tmp, cache)
        return f"{namespace}/{name}"

    def status(self) -> StorageStatus:
        """Cached view (no network)."""
        local = self._local.root_uri if self._local is not None else None
        if self.enabled and local is None:
            local = storage_dir(self.paths).resolve().as_uri()
        with self._lock:
            return StorageStatus(
                backend=self.config.backend,
                local_uri=local if self.enabled else None,
                hf_uri=self._hf.root_uri if self._hf is not None else None,
                reason=self._hf_reason,
                hint=self._hf_hint,
            )

    def is_local_kind(self, kind: str) -> bool:
        return kind in self.local_kinds

    def backend_for(self, kind: str) -> Storage | None:
        """Where runners of this adapter kind keep checkpoints (blocking for hf)."""
        if not self.enabled:
            return None
        if self.is_local_kind(kind):
            return self.local()
        return self.hf()

    def _root(self, uri: str) -> Storage | None:
        """A storage we can read that owns `uri` (local, the configured bucket, or another
        bucket of ours after a config change)."""
        local = self.local()
        if local is not None and local.owns(uri):
            return local
        split = split_hf_uri(uri)
        if split is None:
            return None
        hf = self.hf()
        if hf is not None and hf.owns(uri):
            return hf
        if hf is None:
            return None
        root = split[0]
        with self._lock:
            known = self._extra_roots.get(root)
        if known is not None:
            return known
        token, _problem = tokens.safe_admin_token()
        if token is None:
            return None
        value = token.get_secret_value()
        store = open_storage(root, token=value, api=self._api(value))
        with self._lock:
            self._extra_roots[root] = store
        return store

    # ------------------------------------------------------------------ per attempt

    def prepare_attempt(
        self,
        *,
        job_id: str,
        attempt_n: int,
        kind: str,
        resume: Checkpoint | None,
        secret_names: Sequence[str] = (),
        degrade: bool = False,
    ) -> AttemptStorage | None:
        """Storage settings for one attempt, or None when its runner gets no storage
        (`status().reason` says why). Claims the job for this attempt (owner.json) and,
        when the resume checkpoint lives in the other backend, copies it over first.

        Raises StorageError(retryable) when the storage it needs cannot be reached right
        now (HF down, the Keychain locked): the engine waits and asks again instead of
        submitting an attempt that silently restarts from step 0 (D44). `degrade` (the
        engine waited long enough): go ahead without it, with a note. Refusals that will
        not pass (no token, a token without bucket write, a full storage quota) return
        None: the attempt runs without storage."""
        store = self.backend_for(kind)
        if store is None:
            if not degrade and not self.is_local_kind(kind) and self.hf_down_for_now():
                raise self._unreachable("checkpoint storage (Hugging Face)")
            return None
        out = AttemptStorage(kind=store.kind, root_uri=store.root_uri)
        if store.kind == "hf":
            token, problem = tokens.safe_remote_token()
            if token is None:
                if problem is not None and not degrade:
                    raise StorageError(f"cannot read the storage token ({problem})")
                self._hf_failed(
                    problem or "no HF_TOKEN_REMOTE in the Keychain for remote runs",
                    REMOTE_LOGIN_HINT if problem is None else "unlock the login Keychain",
                    0,
                )
                return None
            out.secrets[STORAGE_TOKEN_ENV] = token
        try:
            store.claim(job_id, attempt_n)
        except StorageError as exc:
            if exc.retryable and not degrade:
                raise
            # refused for good (no write access, the bucket's storage quota is full) or
            # still failing after the wait: this attempt runs without storage
            if store.kind == "hf":
                self._hf_failed(
                    f"cannot write to the storage bucket ({exc.message})",
                    "check the token's write access and the bucket's storage quota",
                    RETRY_TRANSIENT_S if exc.retryable else RETRY_PERMANENT_S,
                    transient=exc.retryable,
                )
            return None
        cfg = self.config
        out.env.update(
            {
                "GPU_STORAGE": store.root_uri,
                "GPU_CKPT_KEEP": str(cfg.keep),
                "GPU_STATUS_PUSH_S": f"{cfg.status_push_s:g}",
                "GPU_CONTROL_POLL_S": f"{cfg.control_poll_s:g}",
            }
        )
        if secret_names:
            out.env["GPU_SECRET_NAMES"] = ",".join(sorted(set(secret_names)))
        if resume is not None:
            uri, note = self.checkpoint_for(resume, store, degrade=degrade)
            if uri is not None:
                out.env["GPU_RESUME_URI"] = uri
            if note:
                out.notes.append(note)
        return out

    def checkpoint_for(
        self, ckpt: Checkpoint, target: Storage, *, degrade: bool = False
    ) -> tuple[str | None, str | None]:
        """(URI the target's runner can restore, note). A checkpoint in the other backend
        is copied under the same key first (checked against its manifest; a torn or
        missing one falls back to the newest intact checkpoint there, D44). (None, None)
        for URIs no backend owns (fake://, a path on a VM): the adapter's own resume logic
        handles those. Raises StorageError(retryable) while the source cannot be reached,
        unless `degrade`; (None, note) when the attempt has to start over."""
        if target.owns(ckpt.uri):
            return ckpt.uri, None
        source = self._root(ckpt.uri)
        if source is None:
            if split_hf_uri(ckpt.uri) is None:
                return None, None
            if not degrade and self.hf_down_for_now():
                raise self._unreachable(f"checkpoint {ckpt.seq} is in Hugging Face storage")
            why = self.status().reason or "Hugging Face storage is not set up"
            return None, (
                f"checkpoint {ckpt.seq} is in Hugging Face storage, which cannot be reached "
                f"({why}); this attempt starts over"
            )
        key = source.key_of(ckpt.uri)
        if key is None:
            return None, None
        try:
            tried: list[int] = []
            for seq, cand in self._copy_candidates(source, ckpt, key):
                tried.append(seq)
                manifest = self._copy_checkpoint(source, target, cand)
                if manifest is None:
                    continue  # torn or gone: try the next older one
                self._merge_latest(target, ckpt, seq, cand, manifest)
                if seq == ckpt.seq:
                    note = (
                        f"copied checkpoint {seq} from {source.kind} storage to {target.kind} "
                        f"storage so this attempt can resume from it"
                    )
                else:
                    note = (
                        f"checkpoint {ckpt.seq} is missing or incomplete in {source.kind} "
                        f"storage; copied checkpoint {seq} (the newest intact one) to "
                        f"{target.kind} storage and resuming from it"
                    )
                return target.uri(cand), note
        except StorageError as exc:
            if exc.retryable and not degrade:
                raise
            return None, (
                f"could not copy checkpoint {ckpt.seq} from {source.kind} storage "
                f"({exc.message}); this attempt starts over"
            )
        return None, (
            f"checkpoint {ckpt.seq} is no longer intact in {source.kind} storage and no older "
            f"one is; this attempt starts over"
        )

    def _copy_candidates(
        self, source: Storage, ckpt: Checkpoint, key: str
    ) -> list[tuple[int, str]]:
        """(seq, key) to try, best first: the checkpoint itself, then storage's latest if
        newer, then the older ones storage still holds."""
        prefix = key.rsplit("/", 1)[0] if "/" in key else job_prefix(ckpt.job_id)
        out: list[tuple[int, str]] = [(ckpt.seq, key)]
        found = StoredCheckpoint.from_dict(source.read_latest(ckpt.job_id))
        if found is not None and found.seq > ckpt.seq:
            other = source.key_of(found.uri)
            if other is not None:
                out.insert(0, (found.seq, other))
        for seq in sorted(source.list_checkpoints(ckpt.job_id), reverse=True):
            if seq < ckpt.seq:
                out.append((seq, f"{prefix}/ckpt-{seq:04d}"))
        return out

    def _copy_checkpoint(self, source: Storage, target: Storage, key: str) -> bytes | None:
        """Copy checkpoint `key` from source to target, checked against its manifest.
        Returns the manifest, or None when the checkpoint is gone, unfinished or torn."""
        local = self.local()
        base = Path(local.raw.root) if local is not None else storage_dir(self.paths)
        tmp_root = base / ".tmp"
        tmp_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix="ckpt-copy-", dir=tmp_root))
        try:
            try:
                source.restore_checkpoint(key, tmp)
            except StorageError as exc:
                if exc.missing:
                    return None
                raise
            if source.verify_checkpoint(key, tmp) is not None:
                return None
            manifest = source.read_bytes(f"{key}/{CKPT_MANIFEST}")
            if manifest is None:
                return None
            files = sorted(
                str(p.relative_to(tmp)).replace(os.sep, "/") for p in tmp.rglob("*") if p.is_file()
            )
            target.upload_dir(tmp, key, files, move=target.kind == "local")
            target.write_bytes(f"{key}/{CKPT_MANIFEST}", manifest)
            return manifest
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def _merge_latest(
        self, target: Storage, ckpt: Checkpoint, seq: int, key: str, manifest: bytes
    ) -> None:
        """Point the target's latest.json at a copied checkpoint when it is newer than what
        the target knew (a stale pointer would make the next runner restart its seq below
        what it resumed from, and prune the wrong checkpoints)."""
        current = StoredCheckpoint.from_dict(target.read_latest(ckpt.job_id))
        if current is not None and current.seq >= seq:
            return
        try:
            meta = json.loads(manifest.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            meta = {}
        meta = meta if isinstance(meta, dict) else {}
        target.write_latest(
            ckpt.job_id,
            {
                "v": 1,
                "job": ckpt.job_id,
                "seq": seq,
                "uri": target.uri(key),
                "attempt": meta.get("attempt"),
                "step": meta.get("step", ckpt.step if seq == ckpt.seq else None),
                "size": meta.get("size"),
                "files": len(meta.get("files") or []),
                "sha256": hashlib.sha256(manifest).hexdigest(),
                "created_at": meta.get("created_at"),
            },
        )

    def latest(self, job_id: str) -> StoredCheckpoint | None:
        """The newest checkpoint any of our backends has for the job (latest.json).
        Raises StorageError when a backend that should answer cannot (HF down, a read
        error): "nothing newer" must not be concluded from an outage (D44)."""
        best: StoredCheckpoint | None = None
        stores: list[Storage] = []
        local = self.local()
        if local is not None:
            stores.append(local)
        if self._hf_allowed() is None:
            hf = self.hf()
            if hf is not None:
                stores.append(hf)
            elif self.hf_down_for_now():
                raise self._unreachable("Hugging Face storage")
        for store in stores:
            found = StoredCheckpoint.from_dict(store.read_latest(job_id))
            if found is not None and (best is None or found.seq > best.seq):
                best = found
        return best

    def delete_job(self, job_id: str) -> list[str]:
        """Remove everything storage holds for a finished job (checkpoints, owner, status
        files) from every backend we can reach; returns the backends cleaned. Best effort:
        a failure is left for the next finished job's sweep to not matter (D44)."""
        cleaned: list[str] = []
        stores: list[Storage] = []
        local = self.local()
        if local is not None:
            stores.append(local)
        if self._hf_allowed() is None:
            hf = self.hf()
            if hf is not None:
                stores.append(hf)
        for store in stores:
            try:
                if store.list_files(job_prefix(job_id)):
                    store.delete_prefix(job_prefix(job_id))
                    cleaned.append(store.kind)
            except StorageError:
                continue
        return cleaned

    def delete_dataset(self, sha256: str) -> None:
        """Remove an uploaded dataset from the bucket (LRU eviction, D44). Raises
        StorageError."""
        hf = self.hf()
        if hf is not None:
            hf.delete_prefix(dataset_key(sha256))

    # ------------------------------------------------------------------ handoff control

    def request_checkpoint(
        self,
        *,
        job_id: str,
        attempt_n: int,
        kind: str,
        request_id: str,
        action: str,
        wait_s: float,
        reason: str,
    ) -> None:
        store = self.backend_for(kind)
        if store is None:
            raise StorageError(self.status().reason or "no checkpoint storage", retryable=False)
        store.write_json(
            control_key(job_id, attempt_n),
            {
                "v": 1,
                "id": request_id,
                "action": action,
                "wait_s": wait_s,
                "reason": reason,
                "ts": self.clock.now(),
            },
        )

    def read_ack(self, *, job_id: str, attempt_n: int, kind: str) -> CheckpointAck | None:
        store = self.backend_for(kind)
        if store is None:
            return None
        data = store.read_json(ack_key(job_id, attempt_n))
        if not data or not isinstance(data.get("id"), str):
            return None
        return CheckpointAck(
            request_id=data["id"],
            new=bool(data.get("new")),
            checkpoint=StoredCheckpoint.from_dict(data),
        )

    def read_heartbeat(self, *, job_id: str, attempt_n: int, kind: str) -> dict[str, Any] | None:
        store = self.backend_for(kind)
        return None if store is None else store.read_json(heartbeat_key(job_id, attempt_n))

    def read_log_tail(self, *, job_id: str, attempt_n: int, kind: str) -> dict[str, Any] | None:
        store = self.backend_for(kind)
        return None if store is None else store.read_json(log_tail_key(job_id, attempt_n))

    # ------------------------------------------------------------------ datasets

    def digest(self, path: Path) -> DataDigest:
        return data_digest(path, self.index)

    def dataset_uri(self, sha256: str) -> str | None:
        """URI of datasets/<sha> in the bucket if its upload completed there, else None."""
        hf = self.hf()
        if hf is None:
            return None
        return hf.uri(dataset_key(sha256)) if hf.dataset_complete(sha256) else None

    def upload_dataset(self, path: Path, dig: DataDigest) -> str:
        """Upload a dataset to the bucket (manifest last). Raises StorageError."""
        hf = self.hf()
        if hf is None:
            raise StorageError(self.status().reason or "no hf storage", retryable=False)
        return hf.upload_dataset(dig.sha256, path, list(dig.files))

    def owns_dataset(self, uri: str) -> bool:
        hf = self._hf
        return hf is not None and hf.owns(uri) and uri.rstrip("/").split("/")[-1] != DATA_MANIFEST

    # ------------------------------------------------------------------ side channel

    def open_status_uri(self, uri: str) -> tuple[Storage, str] | None:
        """(storage, attempt prefix key) for an adapter's `status_uri` (Kaggle keeps it
        in RemoteRef.meta), or None when no backend of ours can read it."""
        store = self._root(uri)
        if store is None:
            return None
        key = store.key_of(uri)
        return None if key is None else (store, key)


def read_json_file(path: Path) -> dict[str, Any] | None:
    with contextlib.suppress(OSError, ValueError):
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    return None
