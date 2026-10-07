"""Job bundles: project code + the `gpu` runner -> deterministic, content-addressed tar.gz.

Archive layout (what bootstrap.py expects once extracted):

    manifest.json        entrypoint, deps, estimate, file stats, warnings (see build_manifest)
    code/<rel path>      git-tracked + untracked-not-ignored project files, plus what the
                         spec's `include:` matches (files.py, D60)
    gpu_runner/gpu.py    the `import gpu` helper (on PYTHONPATH remotely)
    gpu_runner/bootstrap.py
    gpu_runner/storage.py  checkpoint/data/status storage (phase 5; used by bootstrap.py)

Determinism: files sorted by path, fixed mtime/uid/gid/owner names, normalised modes
(0755 if executable else 0644), gzip header without name or time, manifest JSON with sorted
keys and no timestamps. Same project content + same spec -> same bytes -> same sha256.

Cache: `<data dir>/bundles/<sha256>.tar.gz`. A job gets `jobs/<id>/bundle.tar.gz` (hard link
to the cache entry, copy as a fallback) and `jobs/<id>/bundle/` (extracted), the two paths
the engine hands adapters in `AttemptContext`. The archive is fsynced before it is renamed
into the cache, and a rebuild always replaces an existing entry with the fresh (identical)
bytes, so a cache entry truncated by a crash heals on the next build instead of being
trusted forever.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import shutil
import tarfile
import tempfile
from dataclasses import dataclass, field, replace
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any

from gpu_router.errors import InvalidSpec
from gpu_router.packaging.deps import DepsError, DepsInfo, detect_deps
from gpu_router.packaging.estimate import Estimate, estimate
from gpu_router.packaging.files import (
    FileSelection,
    GitError,
    IncludeError,
    LeftOut,
    ProjectFile,
    human_bytes,
    left_out,
    select_files,
)

if TYPE_CHECKING:
    from gpu_router.models import JobSpec
    from gpu_router.paths import Paths

MANIFEST_VERSION = 1
DEFAULT_MAX_BUNDLE_MB = 200.0
RUNNER_FILES: tuple[str, ...] = ("gpu.py", "bootstrap.py", "storage.py")  # storage: phase 5
FIXED_MTIME = 315532800  # 1980-01-01T00:00:00Z: the earliest time zip tools accept
_MB = 1024 * 1024


class BundleError(InvalidSpec):
    """The project cannot be packaged as asked (missing dir, bad deps file, ...)."""


class BundleTooLarge(BundleError):
    """The code to ship is over the size limit."""


@dataclass(frozen=True, slots=True)
class Bundle:
    sha256: str
    archive: Path  # cache entry: <home>/bundles/<sha256>.tar.gz
    manifest: dict[str, Any]
    size_bytes: int  # archive size
    code_bytes: int  # uncompressed project files
    file_count: int
    deps: DepsInfo
    estimate: Estimate
    warnings: list[str] = field(default_factory=list)
    cached: bool = False  # True when an identical bundle was already in the cache
    selection: FileSelection | None = field(default=None, repr=False, compare=False)


def bundles_dir(paths: Paths) -> Path:
    return paths.home / "bundles"


def cached_archive(paths: Paths, sha256: str) -> Path:
    return bundles_dir(paths) / f"{sha256}.tar.gz"


def runner_sources() -> dict[str, bytes]:
    """The runner files shipped in every bundle, read from the installed package."""
    pkg = resources.files("gpu_router.runner")
    return {name: (pkg / name).read_bytes() for name in RUNNER_FILES}


def _mb(n: int) -> str:
    return f"{n / _MB:.1f} MB"


def _size_guard(sel: FileSelection, max_mb: float) -> None:
    total = sel.total_bytes
    if total <= max_mb * _MB:
        return
    biggest = sorted(sel.files, key=lambda f: (-f.size, f.rel))[:5]
    listing = ", ".join(f"{f.rel} ({_mb(f.size)})" for f in biggest)
    hint = (
        "add large files (datasets, weights, outputs) to .gitignore and pass datasets "
        "with `data:` so they upload once to HF Hub instead of every run"
    )
    if sel.included:
        hint = f"`include:` adds {_mb(sel.included_bytes)}: narrow it, or {hint}"
    raise BundleTooLarge(
        f"code bundle would be {_mb(total)}, over the {max_mb:g} MB limit; biggest: {listing}",
        hint=hint,
        detail={
            "bytes": total,
            "limit_mb": max_mb,
            "biggest": [{"path": f.rel, "bytes": f.size} for f in biggest],
        },
    )


def _entry_warnings(spec: JobSpec, sel: FileSelection, project: Path) -> list[str]:
    if spec.script is None:
        return []
    shipped = {f.rel for f in sel.files}
    if spec.script in shipped:
        return []
    if (project / spec.script).is_file():
        return [
            f"{spec.script} exists but is ignored by git, so it will not ship; "
            "commit it, un-ignore it or add it to gpu.yaml `include:`"
        ]
    return [f"{spec.script} is not in the project; the job will fail to start"]


def build_manifest(
    spec: JobSpec,
    sel: FileSelection,
    deps: DepsInfo,
    est: Estimate,
    tree_sha256: str,
    warnings: list[str],
) -> dict[str, Any]:
    return {
        "manifest_version": MANIFEST_VERSION,
        "name": spec.display_name(),
        "entrypoint": {"script": spec.script, "command": spec.command, "args": list(spec.args)},
        "deps": deps.to_manifest(),
        "estimate": est.to_manifest(),
        "files": {
            "count": len(sel.files),
            "bytes": sel.total_bytes,
            "source": sel.source,
            "tree_sha256": tree_sha256,
            "untracked": len(sel.untracked),
            "included": {"count": len(sel.included), "bytes": sel.included_bytes},  # D60
        },
        "checkpoint_interval_min": spec.checkpoint_interval_min,
        "runner": {
            "dir": "gpu_runner",
            "bootstrap": "gpu_runner/bootstrap.py",
            "helper": "gpu_runner/gpu.py",
        },
        "warnings": warnings,
    }


def _tarinfo(name: str, size: int, *, executable: bool = False) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = size
    info.mtime = FIXED_MTIME
    info.mode = 0o755 if executable else 0o644
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.type = tarfile.REGTYPE
    return info


def _read_file(f: ProjectFile) -> bytes:
    try:
        return f.path.read_bytes()
    except OSError as exc:
        raise BundleError(
            f"could not read {f.rel}: {exc.strerror or exc}",
            hint="check the file's permissions, or ignore it in .gitignore",
        ) from exc


def _write_archive(
    out: Path,
    spec: JobSpec,
    sel: FileSelection,
    deps: DepsInfo,
    est: Estimate,
    warnings: list[str],
) -> dict[str, Any]:
    """Write the archive to `out`; returns the manifest written into it (last member)."""
    tree = hashlib.sha256()
    with (
        out.open("wb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=6) as gz,
        tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar,
    ):
        for f in sel.files:
            data = _read_file(f)
            digest = hashlib.sha256(data).hexdigest()
            tree.update(f"{f.rel}\0{digest}\0{int(f.executable)}\n".encode())
            tar.addfile(
                _tarinfo(f"code/{f.rel}", len(data), executable=f.executable), io.BytesIO(data)
            )
        for name, data in runner_sources().items():
            tar.addfile(_tarinfo(f"gpu_runner/{name}", len(data)), io.BytesIO(data))
        manifest = build_manifest(spec, sel, deps, est, tree.hexdigest(), warnings)
        body = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
        tar.addfile(_tarinfo("manifest.json", len(body)), io.BytesIO(body))
    _fsync(out)
    return manifest


def _fsync(path: Path) -> None:
    """Flush a file (or directory) to disk; best effort for directories."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    except OSError:
        if not path.is_dir():
            raise
    finally:
        os.close(fd)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(_MB), b""):
            h.update(block)
    return h.hexdigest()


def build_bundle(
    project_dir: str | Path,
    spec: JobSpec,
    *,
    paths: Paths | None = None,
    max_mb: float = DEFAULT_MAX_BUNDLE_MB,
) -> Bundle:
    """Package `project_dir` for `spec` into the content-addressed cache. Blocking (git,
    file reads, gzip): call it from a worker thread in async code. Raises BundleError
    (an InvalidSpec, so the API answers 400 invalid_spec) with a hint on every failure."""
    if paths is None:
        from gpu_router.paths import Paths as _Paths

        paths = _Paths.from_env()
    project = Path(project_dir)
    sel = _select(project, spec)
    project = project.resolve()
    _size_guard(sel, max_mb)
    shipped = {f.rel for f in sel.files}
    try:
        deps = detect_deps(project, spec.deps, shipped)
    except DepsError as exc:
        raise BundleError(str(exc), hint="fix `deps:` in gpu.yaml or the file itself") from exc
    est = estimate(spec, sel.files)
    warnings = [*sel.warnings, *_entry_warnings(spec, sel, project), *deps.warnings]

    cache = bundles_dir(paths)
    cache.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".build-", suffix=".tar.gz", dir=cache)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        manifest = _write_archive(tmp, spec, sel, deps, est, warnings)
        sha = _sha256_file(tmp)
        final = cached_archive(paths, sha)
        cached = final.exists()
        # Replace even an existing entry: the bytes are identical by construction, and a
        # truncated entry left by a crash must never be reused. Links held by earlier jobs
        # (jobs/<id>/bundle.tar.gz) keep their own inode.
        tmp.chmod(0o600)
        os.replace(tmp, final)
        _fsync(cache)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return Bundle(
        sha256=sha,
        archive=final,
        manifest=manifest,
        size_bytes=final.stat().st_size,
        code_bytes=sel.total_bytes,
        file_count=len(sel.files),
        deps=deps,
        estimate=est,
        warnings=warnings,
        cached=cached,
        selection=sel,
    )


def _select(project: Path, spec: JobSpec) -> FileSelection:
    """files.select_files with its failures as BundleError (400 invalid_spec + hint)."""
    try:
        return select_files(project, spec.include)
    except FileNotFoundError as exc:
        raise BundleError(
            f"project dir {project} does not exist", hint="run from your project folder"
        ) from exc
    except NotADirectoryError as exc:
        raise BundleError(f"project dir {project} is not a directory") from exc
    except (GitError, IncludeError) as exc:
        raise BundleError(str(exc), hint=exc.hint) from exc


#: The one-line advice that comes with a non-empty `left_out` (D60).
LEFT_OUT_HINT = (
    "ignored paths do not ship: add them to gpu.yaml include: (code, small files) or pass "
    "them as data= (datasets)"
)
#: Bundle warnings carried in a summary (each cut to SUMMARY_WARNING_CHARS).
SUMMARY_WARNINGS = 6
SUMMARY_WARNING_CHARS = 300


def left_out_view(
    project: str | Path,
    sel: FileSelection,
    include: list[str] | tuple[str, ...] = (),
    data_paths: list[str] | tuple[str, ...] = (),
) -> dict[str, Any]:
    """{"left_out": ["data/ (2.1 GB, ignored)", ...], "left_out_not_shown": n, "hint"},
    or {} when nothing worth naming is left out. Paths the job already passes as data=
    (`data_paths`, as in the spec) are not named: they reach the GPU that way (seen in the
    2026-10-04 field test: `data/` was "left out" while `data/rows` was its dataset).
    Best effort: never raises."""
    try:
        items, more = left_out(project, sel, include)
        root = Path(project).resolve()
        rels = [r for r in (_rel_to(root, p) for p in data_paths) if r]
        items = [k for k in (_data_note(i, rels) for i in items) if k is not None]
    except Exception:  # a summary must never fail the submit it describes
        return {}
    if not items:
        return {}
    out: dict[str, Any] = {"left_out": [item.text() for item in items]}
    if more:
        out["left_out_not_shown"] = more
    out["hint"] = LEFT_OUT_HINT
    return out


def _rel_to(root: Path, raw: str) -> str | None:
    """A data path as a project-relative posix path, or None when it is outside."""
    path = Path(raw).expanduser()
    path = (path if path.is_absolute() else root / path).resolve()
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return None


def _data_note(item: LeftOut, data_rels: list[str]) -> LeftOut | None:
    """None when `item` is a data path or inside one; a dir holding one gets a note."""
    entry = item.path.rstrip("/")
    if any(entry == r or entry.startswith(r + "/") for r in data_rels):
        return None
    inside = [r for r in data_rels if r.startswith(entry + "/")]
    if not inside:
        return item
    return replace(item, why=f"{item.why}; {', '.join(r + '/' for r in inside)} is passed as data=")


def bundle_summary(spec: JobSpec, *, max_mb: float = DEFAULT_MAX_BUNDLE_MB) -> dict[str, Any]:
    """What a submit of `spec` ships and what it leaves out, for agents (gpu_submit,
    gpu_route): the same file selection the daemon makes, without building the archive.
    Cheap: one `git ls-files` per question, ignored dirs never walked (sizes within a small
    budget). Shape: {files, bytes, size, included?: {files, bytes}, left_out?: [str],
    left_out_not_shown?, hint?, warnings?: [str]} or {error, hint} when the project
    cannot be packaged (the submit fails with the same error)."""
    project = Path(spec.project_dir)
    try:
        sel = _select(project, spec)
    except BundleError as exc:
        return {"error": exc.message, "hint": exc.hint}
    out: dict[str, Any] = {
        "files": len(sel.files),
        "bytes": sel.total_bytes,
        "size": human_bytes(sel.total_bytes),
    }
    if spec.include:
        out["included"] = {"files": len(sel.included), "bytes": sel.included_bytes}
    data_paths = [d.path for d in spec.data if d.path]
    out.update(left_out_view(project, sel, spec.include, data_paths))
    warnings = [*sel.warnings, *_entry_warnings(spec, sel, project.resolve())]
    if sel.total_bytes > max_mb * _MB:
        why = (
            f"`include:` adds {human_bytes(sel.included_bytes)}: narrow it, and pass datasets "
            "as data="
            if sel.included
            else "pass datasets as data="
        )
        warnings.insert(0, f"over the {max_mb:g} MB bundle limit, so the submit is refused; {why}")
    if warnings:
        out["warnings"] = [
            w if len(w) <= SUMMARY_WARNING_CHARS else w[: SUMMARY_WARNING_CHARS - 1] + "…"
            for w in warnings[:SUMMARY_WARNINGS]
        ]
    return out


def extract_bundle(archive: Path, dest: Path) -> None:
    """Extract a bundle archive into `dest` (created). Uses tarfile's `data` filter, which
    refuses absolute paths, '..' and links escaping `dest`."""
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(dest, filter="data")


def materialize(paths: Paths, sha256: str, job_id: str) -> tuple[Path, Path]:
    """Give job `job_id` its bundle: `jobs/<id>/bundle.tar.gz` + `jobs/<id>/bundle/`.
    Idempotent (existing paths are kept). Returns (bundle_dir, bundle_archive)."""
    src = cached_archive(paths, sha256)
    if not src.is_file():
        raise BundleError(
            f"bundle {sha256[:12]} is missing from the cache",
            hint="resubmit the job so the bundle is rebuilt",
        )
    archive = paths.job_bundle_archive(job_id)
    bundle_dir = paths.job_bundle_dir(job_id)
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        tmp = archive.with_name(archive.name + ".tmp")
        tmp.unlink(missing_ok=True)
        try:
            os.link(src, tmp)
        except OSError:
            shutil.copy2(src, tmp)
        os.replace(tmp, archive)
    if not (bundle_dir / "manifest.json").is_file():
        staging = Path(tempfile.mkdtemp(prefix=".bundle-", dir=archive.parent))
        try:
            extract_bundle(archive, staging)
            if bundle_dir.exists():
                shutil.rmtree(bundle_dir)
            os.replace(staging, bundle_dir)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
    return bundle_dir, archive


class BundleBuilder:
    """What the engine uses (EngineDeps.bundler): build at submit, materialize per job."""

    def __init__(self, paths: Paths, *, max_mb: float = DEFAULT_MAX_BUNDLE_MB) -> None:
        self.paths = paths
        self.max_mb = max_mb

    def prepare(self, spec: JobSpec) -> Bundle:
        return build_bundle(spec.project_dir, spec, paths=self.paths, max_mb=self.max_mb)

    def materialize(self, sha256: str, job_id: str) -> tuple[Path, Path]:
        return materialize(self.paths, sha256, job_id)
