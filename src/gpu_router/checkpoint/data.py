"""Datasets for `gpu run --data PATH` / gpu.yaml `data:` (phase 5).

A local path is identified by its content hash: sha256 over the sorted
`(relative path, file sha256)` pairs (the `data_cache.content_hash` definition in schema
0001). Hashing a large dataset takes a while, so the result is remembered per
(absolute path, file list with sizes and mtimes) in `<data dir>/storage/data-index.json`;
an unchanged tree is not read again.

What the remote sees: `$GPU_DATA_DIR/<mount>/` (`gpu.data_dir() / mount`). The runner
makes it from GPU_DATA: a symlink to the path for runs on this Mac, else a download of
`datasets/<sha>/` from the storage bucket, cached on that machine by hash.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

__all__ = ["DataDigest", "DataIndex", "digest", "resolve_data_path", "scan"]

_INDEX_MAX = 500  # remembered trees


@dataclass(frozen=True, slots=True)
class DataDigest:
    sha256: str
    files: tuple[tuple[str, int], ...]  # (relative path, size); a file dataset: its name

    @property
    def size(self) -> int:
        return sum(size for _rel, size in self.files)

    @property
    def count(self) -> int:
        return len(self.files)


def resolve_data_path(raw: str, project_dir: str) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = Path(project_dir) / path
    return path.resolve()


def scan(path: Path) -> list[tuple[str, int, int]]:
    """(relpath, size, mtime_ns) of every regular file, sorted. Symlinked files and
    symlinked directories are followed (a `data/images -> /Volumes/External/images` layout
    is part of the dataset: local runs see it through their symlink, so remote runs must
    get it too, and the content hash covers it, D44); a link back to one of its own
    ancestors (a loop) is skipped. A file dataset is one entry named after the file.

    Credentials never become dataset files (invariant 12, D48): a directory that resolves
    into a credential store (`packaging.files.credential_path_problem`) is not entered,
    and files named like credentials or linking into a store are left out."""
    from gpu_router.packaging.files import SECRET_DIRS, credential_path_problem, secret_name

    if path.is_file():
        st = path.stat()
        return [(path.name, st.st_size, st.st_mtime_ns)]
    if not path.is_dir():
        raise FileNotFoundError(f"{path} does not exist")
    out: list[tuple[str, int, int]] = []
    for root, dirs, names in os.walk(path, followlinks=True):
        real_root = os.path.realpath(root)
        keep = []
        for d in sorted(dirs):
            full_dir = os.path.join(root, d)
            real = os.path.realpath(full_dir)
            if real_root == real or real_root.startswith(real.rstrip(os.sep) + os.sep):
                continue  # links to an ancestor: following it would never end
            if d.lower() in SECRET_DIRS:
                continue
            if os.path.islink(full_dir) and credential_path_problem(real) is not None:
                continue
            keep.append(d)
        dirs[:] = keep
        for name in names:
            if secret_name(name):
                continue
            full = Path(root) / name
            try:
                st = full.stat()
            except OSError:
                continue
            if not full.is_file():
                continue
            if full.is_symlink() and credential_path_problem(os.path.realpath(full)):
                continue
            rel = str(full.relative_to(path)).replace(os.sep, "/")
            out.append((rel, st.st_size, st.st_mtime_ns))
    return sorted(out)


def _file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class DataIndex:
    """(path + file signature) -> content hash, persisted as a small JSON file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def _load(self) -> dict[str, str]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}

    def get(self, signature: str) -> str | None:
        with self._lock:
            return self._load().get(signature)

    def put(self, signature: str, sha: str) -> None:
        with self._lock:
            data = self._load()
            data[signature] = sha
            if len(data) > _INDEX_MAX:
                data = dict(list(data.items())[-_INDEX_MAX:])
            self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".data-index-")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump(data, fh)
                os.replace(tmp, self.path)
            except BaseException:
                with contextlib.suppress(OSError):
                    os.unlink(tmp)
                raise


def digest(path: Path, index: DataIndex | None = None) -> DataDigest:
    """Content hash of a file or directory (blocking; reads every byte unless the index
    knows this exact tree)."""
    entries = scan(path)
    signature = hashlib.sha256(
        json.dumps([str(path), entries], separators=(",", ":")).encode()
    ).hexdigest()
    files = tuple((rel, size) for rel, size, _m in entries)
    known = index.get(signature) if index is not None else None
    if known:
        return DataDigest(sha256=known, files=files)
    h = hashlib.sha256()
    for rel, _size, _m in entries:
        full = path if path.is_file() else path / rel
        h.update(rel.encode("utf-8") + b"\0" + _file_sha(full).encode() + b"\n")
    sha = h.hexdigest()
    if index is not None:
        index.put(signature, sha)
    return DataDigest(sha256=sha, files=files)
