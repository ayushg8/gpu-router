"""Typed daemon-side facade over the runner's storage module (phase 5).

`gpu_router/runner/storage.py` is the one implementation of the storage layout (it ships
in every bundle and runs on Python 3.8 remotes); this module gives the daemon typed calls
and turns its errors into `StorageError` (a GpuRouterError). See that module's docstring
for the layout: `jobs/<job>/ckpt-NNNN/`, `latest.json`, `owner.json`,
`attempts/<n>/{heartbeat,log-tail,control,control-ack}.json`, `datasets/<sha>/`.

Every method blocks (disk or network): the engine calls them from CheckpointHub's
executors, never on the event loop (invariant 9 spirit: the Store stays on the loop).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from gpu_router.errors import GpuRouterError
from gpu_router.runner import storage as rs

__all__ = [
    "CKPT_MANIFEST",
    "DATA_MANIFEST",
    "HF_PREFIX",
    "Storage",
    "StorageError",
    "ack_key",
    "ckpt_key",
    "control_key",
    "dataset_key",
    "heartbeat_key",
    "job_prefix",
    "latest_key",
    "log_tail_key",
    "open_storage",
    "owner_key",
    "seq_of",
    "split_hf_uri",
]

CKPT_MANIFEST: str = rs.CKPT_MANIFEST
DATA_MANIFEST: str = rs.DATA_MANIFEST
HF_PREFIX: str = rs.HF_PREFIX


class StorageError(GpuRouterError):
    """A checkpoint-storage call failed. `retryable`: transient (network, 429, 5xx);
    `missing`: the key does not exist. Messages never contain token values."""

    http_status = 502

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = True,
        missing: bool = False,
        hint: str | None = None,
    ) -> None:
        super().__init__(message, hint=hint)
        self.retryable = retryable
        self.missing = missing


def _wrap[R](fn: Callable[[], R]) -> R:
    try:
        return fn()
    except rs.StorageError as exc:
        raise StorageError(
            str(exc.message), retryable=bool(exc.retryable), missing=bool(exc.missing)
        ) from None
    except OSError as exc:
        raise StorageError(f"storage i/o failed: {exc}") from None


def job_prefix(job_id: str) -> str:
    return str(rs.job_prefix(job_id))


def ckpt_key(job_id: str, seq: int) -> str:
    return str(rs.ckpt_key(job_id, seq))


def latest_key(job_id: str) -> str:
    return str(rs.latest_key(job_id))


def owner_key(job_id: str) -> str:
    return str(rs.owner_key(job_id))


def heartbeat_key(job_id: str, n: int) -> str:
    return str(rs.heartbeat_key(job_id, n))


def log_tail_key(job_id: str, n: int) -> str:
    return str(rs.log_tail_key(job_id, n))


def control_key(job_id: str, n: int) -> str:
    return str(rs.control_key(job_id, n))


def ack_key(job_id: str, n: int) -> str:
    return str(rs.ack_key(job_id, n))


def dataset_key(sha256: str) -> str:
    return str(rs.dataset_key(sha256))


def seq_of(uri: str) -> int | None:
    """The NNNN of a checkpoint URI ending in ckpt-NNNN, else None."""
    got = rs.seq_of(uri)
    return None if got is None else int(got)


def split_hf_uri(uri: str) -> tuple[str, str] | None:
    """hf://buckets/<ns>/<name>/<key> -> (root uri, key), else None."""
    got = rs.split_hf_uri(uri)
    return None if got is None else (str(got[0]), str(got[1]))


class Storage:
    """One storage root (`kind` "hf" or "local") with typed, error-mapped calls."""

    def __init__(self, raw: Any) -> None:
        self._raw = raw
        self.kind: str = str(raw.kind)
        self.root_uri: str = str(raw.root_uri)

    def __repr__(self) -> str:
        return f"Storage({self.root_uri})"

    @property
    def raw(self) -> Any:
        """The runner-side store object (for runner.storage helpers)."""
        return self._raw

    def uri(self, key: str) -> str:
        return _wrap(lambda: str(self._raw.uri(key)))

    def key_of(self, uri: str) -> str | None:
        """The key a URI under this root names, else None."""
        got = self._raw.key_of(uri)
        return None if got is None else str(got)

    def owns(self, uri: str) -> bool:
        return self.key_of(uri) is not None

    def read_bytes(self, key: str) -> bytes | None:
        got = _wrap(lambda: self._raw.read_bytes(key))
        return None if got is None else bytes(got)

    def read_json(self, key: str) -> dict[str, Any] | None:
        raw = self.read_bytes(key)
        if raw is None:
            return None
        try:
            data = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return None
        return data if isinstance(data, dict) else None

    def write_bytes(self, key: str, data: bytes) -> None:
        _wrap(lambda: self._raw.write_bytes(key, data))

    def write_json(self, key: str, obj: Mapping[str, Any]) -> None:
        self.write_bytes(key, bytes(rs.dumps(dict(obj))))

    def stat(self, key: str) -> tuple[int, str] | None:
        got = _wrap(lambda: self._raw.stat(key))
        return None if got is None else (int(got[0]), str(got[1]))

    def upload_dir(self, src: Path, key: str, files: Sequence[str], *, move: bool = False) -> None:
        _wrap(lambda: self._raw.upload_dir(src, key, list(files), move=move))

    def download_dir(self, key: str, dest: Path, *, skip: Sequence[str] = ()) -> int:
        return int(_wrap(lambda: self._raw.download_dir(key, dest, skip=tuple(skip))))

    def list_files(self, prefix: str) -> list[tuple[str, int]]:
        rows = _wrap(lambda: self._raw.list_files(prefix))
        return [(str(k), int(s)) for k, s in rows]

    def delete_prefix(self, prefix: str) -> None:
        _wrap(lambda: self._raw.delete_prefix(prefix))

    def ensure(self) -> None:
        _wrap(lambda: self._raw.ensure())

    # ---- layout-level helpers (shared with the runner)

    def claim(self, job_id: str, attempt: int) -> None:
        _wrap(lambda: rs.claim(self._raw, job_id, attempt))

    def read_latest(self, job_id: str) -> dict[str, Any] | None:
        got = _wrap(lambda: rs.read_latest(self._raw, job_id))
        return dict(got) if got else None

    def restore_checkpoint(self, key: str, dest: Path) -> int:
        return int(_wrap(lambda: rs.restore_checkpoint(self._raw, key, dest)))

    def list_checkpoints(self, job_id: str) -> list[int]:
        """Seqs of the ckpt-NNNN dirs storage holds for the job (complete or not)."""
        return [int(x) for x in _wrap(lambda: rs.list_checkpoints(self._raw, job_id))]

    def verify_checkpoint(self, key: str, local_dir: Path) -> str | None:
        """Why the restored files in local_dir differ from checkpoint `key`'s manifest."""
        got = _wrap(lambda: rs.verify_checkpoint(self._raw, key, local_dir))
        return None if got is None else str(got)

    def write_latest(self, job_id: str, latest: Mapping[str, Any]) -> None:
        self.write_json(rs.latest_key(job_id), latest)

    def dataset_complete(self, sha256: str) -> bool:
        return bool(_wrap(lambda: rs.dataset_complete(self._raw, sha256)))

    def upload_dataset(self, sha256: str, src: Path, files: Sequence[tuple[str, int]]) -> str:
        return str(_wrap(lambda: rs.upload_dataset(self._raw, sha256, src, list(files))))


def open_storage(root_uri: str, *, token: str | None = None, api: Any = None) -> Storage:
    """A Storage for `file:///dir` (or an absolute path) or `hf://buckets/<ns>/<name>`.
    `api` replaces huggingface_hub.HfApi (tests pass a fake)."""
    return Storage(_wrap(lambda: rs.open_store(root_uri, token=token, api=api)))
