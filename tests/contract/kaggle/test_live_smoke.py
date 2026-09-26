"""LIVE Kaggle smoke test: one private GPU kernel end to end through KaggleAdapter.

Opt-in only:
    GPU_ROUTER_REAL_PROVIDERS=kaggle uv run pytest tests/contract/kaggle/test_live_smoke.py -s
Each run starts ONE private kernel (`gpu-router <job_id> 1`, 2xT4) that runs a ~30 s GPU
check (nvidia-smi + torch.cuda.is_available() + a matmul) and uses about 0.05 GPU-hours
of the weekly Kaggle quota. Set LIVE_REPORT_FILE=<file> to get the summary as JSON.

While the kernel runs, it also records (read-only, same kernel) what `kaggle kernels logs`
returns mid-run, with and without `-f`, so NOTES.md can say whether live logs are possible.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import pytest

from gpu_router.adapters.base import AttemptContext, RemotePhase
from gpu_router.clock import SystemClock
from gpu_router.models import Job, JobSpec, JobState, Source
from gpu_router.paths import Paths
from gpu_router.providers.kaggle.cli import find_executable
from tests.contract.kaggle.targets import real_adapter
from tests.unit.packaging.helpers import isolate_git
from tests.unit.providers.kaggle.helpers import make_bundle

pytestmark = [pytest.mark.real_provider("kaggle"), pytest.mark.slow]

GPU_CHECK = r"""
import json, os, pathlib, subprocess, time
import gpu

t0 = time.time()
smi = subprocess.run(
    ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
    capture_output=True, text=True,
)
print("nvidia-smi:", (smi.stdout or smi.stderr).strip().replace("\n", " | "))
import torch
ok = torch.cuda.is_available()
names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if ok else []
print("torch", torch.__version__, "cuda_available", ok, "devices", names)
if ok:
    x = torch.randn(2048, 2048, device="cuda")
    print("matmul ok", round(float((x @ x).abs().sum().item()), 1))
steps = 6
gpu.total_steps(steps)
for i in range(1, steps + 1):
    time.sleep(4)
    gpu.log(step=i, elapsed=round(time.time() - t0, 1))
out = pathlib.Path(os.environ["GPU_OUTPUT_DIR"])
out.mkdir(parents=True, exist_ok=True)
(out / "gpu.json").write_text(json.dumps(
    {"nvidia_smi": smi.stdout.strip(), "cuda": ok, "devices": names, "torch": torch.__version__}
))
print("gpu check done in", round(time.time() - t0, 1), "s")
"""


def _peek_logs(exe: str, ref: str, follow: bool, timeout: float) -> dict[str, Any]:
    argv = [exe, "-W", "kernels", "logs", ref] + (["-f"] if follow else [])
    t0 = time.monotonic()
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
        out, rc, timed_out = proc.stdout, proc.returncode, False
    except subprocess.TimeoutExpired as exc:
        raw = exc.stdout or b""
        out = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
        rc, timed_out = None, True
    return {
        "follow": follow,
        "rc": rc,
        "timed_out": timed_out,
        "seconds": round(time.monotonic() - t0, 1),
        "chars": len(out),
        "lines": len(out.splitlines()),
        "head": out[:300],
    }


def test_live_gpu_smoke(tmp_path: Path, paths: Paths, monkeypatch: pytest.MonkeyPatch) -> None:
    isolate_git(monkeypatch, tmp_path)
    adapter = real_adapter(paths)
    health = adapter.healthcheck()
    assert health.ok, f"{health.reason} ({health.hint})"
    q0 = adapter.quota()

    archive = make_bundle(tmp_path / "proj", paths, script=GPU_CHECK, name="gpu_check.py")
    job_id = os.urandom(6).hex()
    spec = JobSpec(
        project_dir=str(tmp_path / "proj"),
        script="gpu_check.py",
        source=Source.API,
        provider_options={"kaggle": {"timeout_s": 900}},
        checkpoint_interval_min=0,
    )
    now = SystemClock().now()
    job = Job(
        id=job_id,
        short_id=job_id[:4],
        name=spec.display_name(),
        state=JobState.PROVISIONING,
        source=spec.source,
        project_dir=spec.project_dir,
        spec=spec,
        spec_hash="0" * 64,
        provider="kaggle",
        created_at=now,
        updated_at=now,
    )
    ctx = AttemptContext(
        attempt_id=f"{job_id}.1",
        attempt_key=f"gpu-{job_id}-1",
        n=1,
        bundle_archive=archive,
        gpu="T4",
        checkpoint_interval_min=0,
        env={"GPU_ROUTER_JOB_ID": job_id, "GPU_ROUTER_ATTEMPT": "1", "GPU_ROUTER_PROTOCOL": "1"},
    )
    t_submit = time.monotonic()
    ref = adapter.submit(job, ctx)
    assert ref.remote_id.endswith(f"/gpu-router-{job_id}-1")
    again = adapter.submit(job, ctx)  # A4: no second push
    assert again.remote_id == ref.remote_id
    found = adapter.lookup_by_key(ctx.attempt_key)
    assert found is not None
    assert found.remote_id == ref.remote_id

    exe = find_executable()
    assert exe is not None
    phases: list[tuple[float, str]] = []
    peeks: list[dict[str, Any]] = []
    deadline = time.monotonic() + 25 * 60
    st = adapter.status(ref)
    while not st.phase.terminal:
        elapsed = round(time.monotonic() - t_submit, 1)
        if not phases or phases[-1][1] != st.phase:
            phases.append((elapsed, str(st.phase)))
        if st.phase is RemotePhase.RUNNING and not peeks:
            peeks.append(_peek_logs(exe, ref.remote_id, follow=False, timeout=30))
            peeks.append(_peek_logs(exe, ref.remote_id, follow=True, timeout=20))
        assert time.monotonic() < deadline, f"still {st.phase} after 25 min"
        # while running, logs() keeps the cursor and returns nothing (post-run logs only)
        chunks = list(adapter.logs(ref, since="0"))
        assert chunks[-1].eof is False
        time.sleep(15)
        st = adapter.status(ref)
    t_done = round(time.monotonic() - t_submit, 1)
    phases.append((t_done, str(st.phase)))

    lines = [line for c in adapter.logs(ref) for line in c.lines]
    tail = next(adapter.logs(ref, since=str(len(lines))))
    assert tail.lines == []
    assert tail.eof
    dest = tmp_path / "runs" / job_id[:4]
    fetched = adapter.fetch(ref, dest)
    report = json.loads((dest / "gpu.json").read_text()) if (dest / "gpu.json").is_file() else {}
    time.sleep(5)
    q1 = adapter.quota()
    summary = {
        "kernel": ref.remote_id,
        "url": ref.url,
        "final": str(st.phase),
        "exit_code": st.exit_code,
        "phases": phases,
        "wall_s": t_done,
        "gpu_seen": report.get("nvidia_smi"),
        "torch_cuda": report.get("cuda"),
        "devices": report.get("devices"),
        "quota_used_h_before": q0.used,
        "quota_used_h_after": q1.used,
        "gpu_minutes_used": round((q1.used - q0.used) * 60, 1),
        "log_lines": len(lines),
        "log_head": lines[:12],
        "fetched_files": fetched.files,
        "midrun_log_peeks": peeks,
    }
    print(json.dumps(summary, indent=2))
    out = os.environ.get("LIVE_REPORT_FILE")
    if out:
        Path(out).write_text(json.dumps(summary, indent=2))
    assert st.phase is RemotePhase.SUCCEEDED, lines[-30:]
    assert st.exit_code == 0
    assert any("cuda_available True" in line for line in lines), lines[-30:]
    assert fetched.files >= 1
    assert report.get("cuda") is True


SLEEPER = r"""
import time
for i in range(54):
    print("sleeper alive", i * 10, "s", flush=True)
    time.sleep(10)
"""


@pytest.mark.skipif(
    os.environ.get("KAGGLE_CANCEL_PROBE") != "1", reason="set KAGGLE_CANCEL_PROBE=1 to run"
)
def test_live_cancel_probe(tmp_path: Path, paths: Paths, monkeypatch: pytest.MonkeyPatch) -> None:
    """Does cancel (= `kaggle kernels delete`) stop a RUNNING GPU session? One private
    kernel with a 600 s session timeout sleeps for 9 minutes; cancel() runs ~60 s after it
    starts. GPU quota is sampled until well past the 600 s timeout: usage close to the time
    before the delete means the session stopped, ~0.17 h means it ran to the timeout."""
    isolate_git(monkeypatch, tmp_path)
    adapter = real_adapter(paths)
    assert adapter.healthcheck().ok
    q0 = adapter.quota().used
    archive = make_bundle(tmp_path / "proj", paths, script=SLEEPER, name="sleeper.py")
    job_id = os.urandom(6).hex()
    spec = JobSpec(
        project_dir=str(tmp_path / "proj"),
        script="sleeper.py",
        source=Source.API,
        provider_options={"kaggle": {"timeout_s": 600}},
        checkpoint_interval_min=0,
    )
    now = SystemClock().now()
    job = Job(
        id=job_id,
        short_id=job_id[:4],
        name=spec.display_name(),
        state=JobState.PROVISIONING,
        source=spec.source,
        project_dir=spec.project_dir,
        spec=spec,
        spec_hash="0" * 64,
        provider="kaggle",
        created_at=now,
        updated_at=now,
    )
    ctx = AttemptContext(
        attempt_id=f"{job_id}.1",
        attempt_key=f"gpu-{job_id}-1",
        n=1,
        bundle_archive=archive,
        gpu="T4",
        checkpoint_interval_min=0,
        env={"GPU_ROUTER_JOB_ID": job_id, "GPU_ROUTER_ATTEMPT": "1", "GPU_ROUTER_PROTOCOL": "1"},
    )
    t0 = time.monotonic()
    ref = adapter.submit(job, ctx)
    st = adapter.status(ref)
    while st.phase is RemotePhase.PENDING:
        assert time.monotonic() - t0 < 900, "never started"
        time.sleep(10)
        st = adapter.status(ref)
    assert st.phase is RemotePhase.RUNNING, st
    t_running = time.monotonic() - t0
    time.sleep(60)
    samples: list[tuple[float, float]] = [(round(time.monotonic() - t0, 1), adapter.quota().used)]
    adapter.cancel(ref)
    t_cancel = time.monotonic() - t0
    after = adapter.status(ref)
    lookup = adapter.lookup_by_key(ctx.attempt_key)
    while time.monotonic() - t0 < t_running + 600 + 180:
        time.sleep(60)
        samples.append((round(time.monotonic() - t0, 1), adapter.quota().used))
    summary = {
        "kernel": ref.remote_id,
        "running_at_s": round(t_running, 1),
        "cancel_at_s": round(t_cancel, 1),
        "status_after_cancel": str(after.phase),
        "lookup_after_cancel": lookup.remote_id if lookup else None,
        "quota_used_before_h": q0,
        "quota_samples_h": samples,
        "delta_h": round(samples[-1][1] - q0, 2),
        "ran_before_cancel_h": round((t_cancel - t_running) / 3600, 3),
    }
    print(json.dumps(summary, indent=2))
    out = os.environ.get("LIVE_REPORT_FILE")
    if out:
        Path(out).write_text(json.dumps(summary, indent=2))
    assert after.phase is RemotePhase.CANCELLED
