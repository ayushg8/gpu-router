"""Live smoke test on the real Colab account (opt-in: GPU_ROUTER_REAL_PROVIDERS=colab).

One T4 session runs a ~30-second fp16 matmul GPU check end to end through the adapter
(submit, status polls, logs, fetch), then the test confirms the session was stopped. Run:

    GPU_ROUTER_REAL_PROVIDERS=colab uv run pytest tests/unit/providers/colab/test_live.py -s

It uses a private data dir (the tmp GPU_ROUTER_HOME), so its `--config` session file is
not ~/.config/colab-cli/sessions.json and no other tool's session is visible to it.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from gpu_router.adapters.base import AdapterDeps, RemotePhase, RemoteRef
from gpu_router.clock import SystemClock
from gpu_router.config import ProviderSettings
from gpu_router.models import DepsSpec, JobSpec
from gpu_router.packaging.bundle import build_bundle
from gpu_router.paths import Paths
from gpu_router.providers.catalog import load_catalog
from gpu_router.providers.colab.adapter import ColabAdapter
from gpu_router.providers.colab.cli import ColabCli
from tests.contract.harness import ContractTarget, make_ctx, make_job

GPU_CHECK = r"""
import json
import time

import gpu
import torch

assert torch.cuda.is_available(), "no CUDA device visible"
name = torch.cuda.get_device_name(0)
print("device: %s, torch %s, cuda %s" % (name, torch.__version__, torch.version.cuda), flush=True)
n = 4096
a = torch.randn(n, n, device="cuda", dtype=torch.float16)
b = torch.randn(n, n, device="cuda", dtype=torch.float16)
torch.cuda.synchronize()
steps = 30
gpu.total_steps(steps)
t0 = time.time()
count = 0
tflops = 0.0
for step in range(1, steps + 1):
    t_step = time.time()
    while time.time() - t_step < 1.0:
        for _ in range(8):  # CUDA is async: sync per small batch so a step is ~1 s of GPU
            c = a @ b
        torch.cuda.synchronize()
        count += 8
    tflops = count * 2 * n**3 / (time.time() - t0) / 1e12
    gpu.log(step=step, tflops=round(tflops, 2))
result = {
    "device": name,
    "matmuls": count,
    "seconds": round(time.time() - t0, 1),
    "tflops_fp16": round(tflops, 2),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
}
(gpu.output_dir() / "gpu_check.json").write_text(json.dumps(result))
print("gpu check done " + json.dumps(result), flush=True)
"""


@pytest.mark.real_provider("colab")
@pytest.mark.slow
def test_live_t4_gpu_check(paths: Paths, tmp_path: Path) -> None:
    clock = SystemClock()
    adapter = ColabAdapter(
        AdapterDeps(
            name="colab",
            entry=load_catalog(None).get("colab"),
            settings=ProviderSettings(),
            paths=paths,
            clock=clock,
            test_mode=True,  # opted in through GPU_ROUTER_REAL_PROVIDERS
        )
    )
    health = adapter.healthcheck()
    print(f"\nhealthcheck: {health.health} {health.reason or ''} {health.detail}")
    assert health.health.value in ("ok", "degraded"), health

    project = tmp_path / "gpu-check"
    project.mkdir()
    (project / "gpu_check.py").write_text(GPU_CHECK)
    spec = JobSpec(project_dir=str(project), script="gpu_check.py", deps=DepsSpec(kind="none"))
    archive = build_bundle(project, spec, paths=paths).archive
    target = ContractTarget(name="colab", build=lambda: adapter, clock=clock, real=True)
    job = make_job(target, project_dir=str(project), script="gpu_check.py")
    ctx = make_ctx(job).model_copy(update={"bundle_archive": archive, "gpu": "T4"})

    t0 = time.monotonic()
    ref: RemoteRef | None = None
    summary: dict[str, object] = {}
    try:
        ref = adapter.submit(job, ctx)
        print(f"submitted: session {ref.remote_id} in {time.monotonic() - t0:.0f}s")
        deadline = time.monotonic() + 20 * 60
        while True:
            st = adapter.status(ref)
            print(f"  t+{time.monotonic() - t0:4.0f}s {st.phase} gpu={st.gpu} {st.message or ''}")
            if st.phase.terminal:
                break
            assert time.monotonic() < deadline, "live smoke took longer than 20 min"
            time.sleep(15)
        lines = [line for c in adapter.logs(ref) for line in c.lines]
        print("---- remote log (tail)\n" + "\n".join(lines[-12:]) + "\n----")
        assert st.phase is RemotePhase.SUCCEEDED, st
        res = adapter.fetch(ref, tmp_path / "outputs")
        check = json.loads((tmp_path / "outputs" / "gpu_check.json").read_text())
        rec = adapter.store.load(ref.remote_id)
        assert rec is not None
        listed = ColabCli(adapter._cli().prefix, adapter.config_file).run(["sessions"], timeout=60)
        summary = {
            "session": ref.remote_id,
            "gpu_seen": rec.gpu_seen,
            "status_gpu": st.gpu,
            "gpu_check": check,
            "fetched_files": res.files,
            "record_stopped": rec.stopped,
            "session_listed_after": f"[{ref.remote_id}]" in listed.stdout,
            "wall_s": round(time.monotonic() - t0),
        }
        print("LIVE_SMOKE " + json.dumps(summary))
        assert "T4" in str(check["device"])
        assert rec.stopped, "the session must be stopped once the job ended"
        assert not summary["session_listed_after"]
    finally:
        if ref is not None:
            adapter.cancel(ref)  # idempotent; stops the session if anything above failed
