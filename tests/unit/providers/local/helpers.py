"""Shared helpers for LocalAdapter tests and the local contract target.

These tests start real processes (the launcher, bootstrap, the job's python) on this Mac,
under the per-test tmp GPU_ROUTER_HOME. Nothing leaves the tmp dir except `uv venv`
reading the user's uv cache in the few tests that use the real uv.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import sys
import time
from pathlib import Path
from typing import Any

from gpu_router.adapters.base import AdapterDeps, AttemptContext, LogChunk, RemotePhase, RemoteRef
from gpu_router.clock import SystemClock
from gpu_router.config import ProviderSettings
from gpu_router.models import Job, JobSpec, JobState, Source
from gpu_router.packaging.bundle import Bundle, build_bundle
from gpu_router.paths import Paths
from gpu_router.providers.catalog import GpuOffer, ProviderEntry, QuotaSpec
from gpu_router.providers.local.adapter import LocalAdapter

LOCAL_ENTRY = ProviderEntry(
    name="local",
    kind="local",
    display_name="Local Mac (MPS)",
    priority=90,
    gpus=(GpuOffer(name="MPS", vram_gb=16),),
    session_hours=None,
    max_concurrency=1,
    poll_interval_s=5,
    quota=QuotaSpec(limit=None, reset="none"),
)

TERMINAL = frozenset(p for p in RemotePhase if p.terminal)

_counter = 0


def system_settings(**extra: Any) -> ProviderSettings:
    """env: system with the test interpreter: no venv, no installs, fastest."""
    return ProviderSettings(env="system", python=sys.executable, **extra)


def make_adapter(paths: Paths, settings: ProviderSettings | None = None) -> LocalAdapter:
    return LocalAdapter(
        AdapterDeps(
            name="local",
            entry=LOCAL_ENTRY,
            settings=settings if settings is not None else system_settings(),
            paths=paths,
            clock=SystemClock(),
        )
    )


def build_project(
    root: Path, script: str, *, requirements: str | None = None, name: str = "proj"
) -> Path:
    project = root / name
    project.mkdir(parents=True, exist_ok=True)
    (project / "train.py").write_text(script)
    if requirements is not None:
        (project / "requirements.txt").write_text(requirements)
    return project


def make_bundle(paths: Paths, project: Path) -> Bundle:
    spec = JobSpec(project_dir=str(project), script="train.py", source=Source.API)
    return build_bundle(project, spec, paths=paths)


def make_job(project: Path, **spec_fields: Any) -> Job:
    global _counter
    _counter += 1
    job_id = hashlib.sha256(f"local-{_counter}-{time.monotonic_ns()}".encode()).hexdigest()[:12]
    spec = JobSpec(project_dir=str(project), script="train.py", source=Source.API, **spec_fields)
    now = SystemClock().now()
    return Job(
        id=job_id,
        short_id=job_id[:4],
        name=spec.display_name(),
        state=JobState.PROVISIONING,
        source=spec.source,
        project_dir=spec.project_dir,
        spec=spec,
        spec_hash=hashlib.sha256(spec.model_dump_json().encode()).hexdigest(),
        provider="local",
        created_at=now,
        updated_at=now,
    )


def make_ctx(job: Job, bundle: Bundle | None, n: int = 1, **fields: Any) -> AttemptContext:
    if bundle is not None and "bundle_archive" not in fields and "bundle_dir" not in fields:
        fields["bundle_archive"] = bundle.archive
    return AttemptContext(
        attempt_id=f"{job.id}.{n}", attempt_key=f"gpu-{job.id}-{n}", n=n, **fields
    )


def wait_phase(
    adapter: LocalAdapter,
    ref: RemoteRef,
    phases: frozenset[RemotePhase] | set[RemotePhase] = TERMINAL,
    timeout_s: float = 60.0,
) -> Any:
    deadline = time.monotonic() + timeout_s
    while True:
        st = adapter.status(ref)
        if st.phase in phases:
            return st
        if time.monotonic() > deadline:
            raise AssertionError(f"still {st.phase} ({st.message}) after {timeout_s}s")
        time.sleep(0.05)


def all_lines(adapter: LocalAdapter, ref: RemoteRef, since: str | None = None) -> list[str]:
    return [line for c in adapter.logs(ref, since=since) for line in c.lines]


def wait_for_line(
    adapter: LocalAdapter, ref: RemoteRef, needle: str, timeout_s: float = 30.0
) -> None:
    deadline = time.monotonic() + timeout_s
    while not any(needle in line for line in all_lines(adapter, ref)):
        if time.monotonic() > deadline:
            raise AssertionError(f"{needle!r} never showed up; log: {all_lines(adapter, ref)}")
        time.sleep(0.05)


def drain(chunks: list[LogChunk]) -> tuple[list[str], str | None, bool]:
    lines = [line for c in chunks for line in c.lines]
    return lines, (chunks[-1].cursor if chunks else None), bool(chunks and chunks[-1].eof)


def run_dir(adapter: LocalAdapter, ref: RemoteRef) -> Path:
    return adapter.runs_root / ref.remote_id


def read_pid(adapter: LocalAdapter, ref: RemoteRef) -> int:
    data = json.loads((run_dir(adapter, ref) / "pid.json").read_text())
    return int(data["pid"])


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def write_fake_uv(path: Path, *, pip_exit: int = 0) -> Path:
    """A stand-in for uv: `venv` makes a real (pip-less) stdlib venv, `pip install` logs its
    arguments and exits with `pip_exit`. Every call is appended to <path>.calls."""
    calls = path.with_suffix(".calls")
    path.write_text(
        f"""#!{sys.executable}
import subprocess, sys
args = sys.argv[1:]
with open({str(calls)!r}, "a") as fh:
    fh.write(" ".join(args) + "\\n")
if args and args[0] == "venv":
    base = args[args.index("--python") + 1]
    sys.exit(subprocess.call([base, "-m", "venv", "--without-pip", args[-1]]))
if args[:2] == ["pip", "install"]:
    print("fake uv: pip install", " ".join(args[2:]), flush=True)
    sys.exit({pip_exit})
sys.exit(2)
"""
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def fake_uv_calls(path: Path) -> list[str]:
    calls = path.with_suffix(".calls")
    return calls.read_text().splitlines() if calls.exists() else []


def real_uv() -> str | None:
    found = os.environ.get("UV") or shutil.which("uv")
    return found if found and Path(found).is_file() else None
