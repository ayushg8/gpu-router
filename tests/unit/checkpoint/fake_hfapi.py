"""An in-memory stand-in for huggingface_hub.HfApi's bucket API (tests only).

Implements the subset gpu-router uses, with the same call shapes as huggingface_hub
1.32: whoami, create_bucket, bucket_info, batch_bucket_files, get_bucket_paths_info,
download_bucket_files, list_bucket_tree. `fail[op]` is a list of exceptions raised by
the next calls of that op (HTTPError(status) carries `.response.status_code` like
huggingface_hub's HfHubHTTPError). Every call is recorded in `calls`.
"""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any


class HTTPError(Exception):
    def __init__(self, status: int, message: str = "") -> None:
        super().__init__(message or f"HTTP {status}")
        self.response = SimpleNamespace(status_code=status)


@dataclass(frozen=True)
class FakeBucketFile:
    path: str
    size: int
    xet_hash: str
    type: str = "file"


@dataclass
class FakeHfApi:
    user: str = "tester"
    buckets: dict[str, dict[str, bytes]] = field(default_factory=dict)
    private: dict[str, bool] = field(default_factory=dict)
    calls: list[tuple[str, Any]] = field(default_factory=list)
    fail: dict[str, list[Exception]] = field(default_factory=dict)
    tokens: list[str | None] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    # ---- test helpers

    def factory(self, token: str | None) -> FakeHfApi:
        """Use as CheckpointHub(hf_api=api.factory): records the token it was built with."""
        self.tokens.append(token)
        return self

    def files(self, bucket: str) -> dict[str, bytes]:
        return self.buckets.get(bucket, {})

    def ops(self, name: str) -> list[Any]:
        return [arg for op, arg in self.calls if op == name]

    def _hit(self, op: str, arg: Any = None) -> None:
        with self._lock:
            self.calls.append((op, arg))
            pending = self.fail.get(op)
            if pending:
                raise pending.pop(0)

    def _bucket(self, bucket_id: str) -> dict[str, bytes]:
        if bucket_id not in self.buckets:
            raise HTTPError(404, f"bucket {bucket_id} not found")
        return self.buckets[bucket_id]

    @staticmethod
    def _info(path: str, data: bytes) -> FakeBucketFile:
        return FakeBucketFile(path=path, size=len(data), xet_hash=hashlib.sha256(data).hexdigest())

    # ---- HfApi surface

    def whoami(self) -> dict[str, Any]:
        self._hit("whoami")
        return {"name": self.user, "type": "user"}

    def create_bucket(
        self, bucket_id: str, *, private: bool | None = None, exist_ok: bool = False
    ) -> str:
        self._hit("create_bucket", bucket_id)
        if bucket_id in self.buckets:
            if not exist_ok:
                raise HTTPError(409, "exists")
        else:
            self.buckets[bucket_id] = {}
            self.private[bucket_id] = bool(private)
        return f"https://huggingface.co/buckets/{bucket_id}"

    def bucket_info(self, bucket_id: str) -> SimpleNamespace:
        self._hit("bucket_info", bucket_id)
        files = self._bucket(bucket_id)
        return SimpleNamespace(
            id=bucket_id,
            private=self.private.get(bucket_id, False),
            size=sum(len(v) for v in files.values()),
            total_files=len(files),
        )

    def batch_bucket_files(
        self,
        bucket_id: str,
        *,
        add: list[tuple[str | Path | bytes, str]] | None = None,
        copy: list[Any] | None = None,
        delete: list[str] | None = None,
    ) -> None:
        self._hit("batch_bucket_files", {"add": [d for _s, d in add or []], "delete": delete})
        files = self._bucket(bucket_id)
        for src, dest in add or []:
            data = bytes(src) if isinstance(src, bytes | bytearray) else Path(src).read_bytes()
            files[dest] = data
        for key in delete or []:
            files.pop(key, None)

    def get_bucket_paths_info(
        self, bucket_id: str, paths: Iterable[str]
    ) -> Iterator[FakeBucketFile]:
        wanted = list(paths)
        self._hit("get_bucket_paths_info", wanted)
        files = self._bucket(bucket_id)
        for p in wanted:
            if p in files:
                yield self._info(p, files[p])

    def download_bucket_files(
        self,
        bucket_id: str,
        files: list[tuple[str | FakeBucketFile, str | Path]],
        *,
        raise_on_missing_files: bool = False,
    ) -> None:
        self._hit("download_bucket_files", [getattr(r, "path", r) for r, _l in files])
        store = self._bucket(bucket_id)
        for remote, local in files:
            path = remote.path if isinstance(remote, FakeBucketFile) else str(remote)
            if path not in store:
                if raise_on_missing_files:
                    raise HTTPError(404, path)
                continue
            out = Path(local)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(store[path])

    def list_bucket_tree(
        self, bucket_id: str, prefix: str | None = None, *, recursive: bool | None = None
    ) -> Iterator[FakeBucketFile]:
        self._hit("list_bucket_tree", prefix)
        store = self._bucket(bucket_id)
        base = (prefix or "").strip("/")
        for path in sorted(store):
            if not base or path == base or path.startswith(base + "/"):
                yield self._info(path, store[path])
