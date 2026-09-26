"""LIVE Lightning AI smoke test: one T4 Studio job end to end through LightningAdapter.

Opt-in only:
    GPU_ROUTER_REAL_PROVIDERS=lightning uv run pytest tests/contract/lightning/test_live_smoke.py -s
Credentials: LIGHTNING_USER_ID + LIGHTNING_API_KEY in the environment, or
~/.lightning/credentials.json (the tests' keyring is in memory, so the Keychain is not
read here; the end-to-end run through the real CLI + daemon reads it). Each run starts ONE
T4 job (`gr-<job_id>-1`) running a ~30 s GPU check (nvidia-smi, torch.cuda, a matmul);
about 0.04 credits (2026-09-25: 0.043, ~2.5 min of T4 incl. Studio snapshot + machine
start). The job is stopped in a finally block whatever happens, and its in-job wall clock
is 15 min (`timeout_s`), so a job orphaned by a killed test process (seen once: the session
dropped mid-test) still ends by itself. Credits used = the balance before minus after
(a job's own `total_cost` lags: 0.0 at completion, 0.022 a minute later, 0.043 final).
Set LIVE_REPORT_FILE=<file> to get the summary as JSON.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import pytest

from gpu_router.adapters.base import AttemptContext, RemotePhase
from gpu_router.clock import SystemClock
from gpu_router.models import Job, JobSpec, JobState, Source
from gpu_router.paths import Paths
from gpu_router.providers.lightning.adapter import LightningAdapter
from tests.contract.lightning.targets import lightning_deps
from tests.unit.packaging.helpers import isolate_git
from tests.unit.providers.kaggle.helpers import make_bundle

pytestmark = [pytest.mark.real_provider("lightning"), pytest.mark.slow]

GPU_CHECK = r"""
import json, os, pathlib, subprocess, time
import gpu

t0 = time.time()
smi = subprocess.run(
    ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
    capture_output=True, text=True,
)
print("nvidia-smi:", (smi.stdout or smi.stderr).strip().replace("\n", " | "))
try:
    import torch
    ok = torch.cuda.is_available()
    names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if ok else []
    print("torch", torch.__version__, "cuda_available", ok, "devices", names)
    if ok:
        x = torch.randn(2048, 2048, device="cuda")
        print("matmul ok", round(float((x @ x).abs().sum().item()), 1))
except ImportError:
    ok, names = False, []
    print("torch not installed in this studio env")
steps = 6
gpu.total_steps(steps)
for i in range(1, steps + 1):
    time.sleep(4)
    gpu.log(step=i, elapsed=round(time.time() - t0, 1))
out = pathlib.Path(os.environ["GPU_OUTPUT_DIR"])
out.mkdir(parents=True, exist_ok=True)
(out / "gpu.json").write_text(json.dumps(
    {"nvidia_smi": smi.stdout.strip(), "cuda": ok, "devices": names}
))
print("gpu check done in", round(time.time() - t0, 1), "s")
"""


def test_one_t4_job_end_to_end(
    paths: Paths, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    isolate_git(monkeypatch, tmp_path)
    adapter = LightningAdapter(lightning_deps(paths))
    health = adapter.healthcheck()
    if not health.ok:
        pytest.skip(f"lightning not healthy: {health.reason}")
    archive = make_bundle(tmp_path / "proj", paths, script=GPU_CHECK, name="gpucheck.py")
    clock = SystemClock()
    spec = JobSpec(
        project_dir=str(tmp_path / "proj"),
        script="gpucheck.py",
        source=Source.API,
        provider_options={"lightning": {"timeout_s": 900}},
    )
    job_id = os.urandom(6).hex()
    now = clock.now()
    job = Job(
        id=job_id,
        short_id=job_id[:4],
        name="gpucheck",
        state=JobState.PROVISIONING,
        source=spec.source,
        project_dir=spec.project_dir,
        spec=spec,
        spec_hash="0" * 64,
        provider="lightning",
        created_at=now,
        updated_at=now,
    )
    ctx = AttemptContext(
        attempt_id=f"{job_id}.1",
        attempt_key=f"gpu-{job_id}-1",
        n=1,
        bundle_archive=archive,
        env={"GPU_ROUTER_JOB_ID": job_id, "GPU_ROUTER_ATTEMPT": "1", "GPU_ROUTER_PROTOCOL": "1"},
        gpu="T4",
        checkpoint_interval_min=0,
    )
    report: dict[str, Any] = {"health": health.detail}
    before = adapter.quota()
    report["balance_before"] = before.detail.get("balance")
    t0 = time.monotonic()
    ref = adapter.submit(job, ctx)
    report.update(job=ref.remote_id, url=ref.url, meta=ref.meta)
    try:
        phases: list[str] = []
        st = adapter.status(ref)
        while not st.phase.terminal and time.monotonic() - t0 < 1800:
            if not phases or phases[-1] != st.phase:
                phases.append(str(st.phase))
            time.sleep(15)
            st = adapter.status(ref)
        report.update(
            phase=str(st.phase), exit_code=st.exit_code, lost=st.lost_reason, phases=phases
        )
        lines = [x for c in adapter.logs(ref) for x in c.lines]
        report["log_tail"] = lines[-25:]
        report["gpu_seen"] = next((x for x in lines if x.startswith("nvidia-smi:")), None)
        if st.phase is RemotePhase.SUCCEEDED:
            res = adapter.fetch(ref, tmp_path / "out")
            report["fetch"] = {"files": res.files, "partial": res.partial, "message": res.message}
            gpu_json = tmp_path / "out" / "gpu.json"
            if gpu_json.exists():
                report["gpu_json"] = json.loads(gpu_json.read_text())
        report["seconds"] = round(time.monotonic() - t0, 1)
    finally:
        adapter.cancel(ref)  # a no-op when finished; stops it otherwise
        final_path = adapter.scratch_dir / "final" / f"{ref.remote_id}.json"
        final = json.loads(final_path.read_text()) if final_path.exists() else {}
        report["job_total_cost_at_finish"] = final.get("total_cost")  # lags, see above
        try:
            time.sleep(90)  # let billing settle the job before reading the balance
            after = adapter.quota()
            report["quota"] = after.model_dump(mode="json")
            b0, b1 = report["balance_before"], after.detail.get("balance")
            if isinstance(b0, float) and isinstance(b1, float):
                report["credits_used"] = round(b0 - b1, 4)
        except Exception as exc:  # report only
            report["quota_error"] = str(exc)
        out = os.environ.get("LIVE_REPORT_FILE")
        if out:
            Path(out).write_text(json.dumps(report, indent=2, default=str))
        print(json.dumps(report, indent=2, default=str))
    assert st.phase is RemotePhase.SUCCEEDED, report
    assert report["gpu_seen"], report
    assert "T4" in report["gpu_seen"], report
