"""Phase-2 end to end through the real `gpu` executable (no CliRunner, no running daemon).

One flow, in a throwaway git project with train.py (`import gpu`) and gpu.yaml:
`gpu run --json` auto-starts a daemon on a private GPU_ROUTER_HOME, the daemon bundles the
project and runs it on the fake provider, outputs land in <project>/runs/<id4>/, then
status / logs / jobs / history / fetch / cancel all answer with --json. Finally the bundle
the daemon materialized is executed through the remote runner (bootstrap.py), which proves
the shipped train.py, `import gpu` and gpu.output_dir() work as a provider would run them.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from gpu_router.paths import Paths
from tests.crash.harness import FAST_CONFIG, FAST_PROVIDERS

pytestmark = pytest.mark.slow

TRAIN_PY = """\
import gpu

gpu.total_steps(3)
for i in range(1, 4):
    gpu.log(step=i, loss=1.0 / i)
(gpu.output_dir() / "result.txt").write_text("trained ok\\n")
print("train.py finished")
"""

GPU_YAML = """\
version: 1
script: train.py
provider_options:
  fake: {duration: 0.5, steps: 3}
"""


def _gpu_argv() -> list[str]:
    exe = Path(sys.executable).parent / "gpu"
    return [str(exe)] if exe.exists() else [sys.executable, "-m", "gpu_router"]


class Gpu:
    def __init__(self, home: Path, project: Path) -> None:
        self.home = home
        self.project = project
        self.env = {k: v for k, v in os.environ.items() if not k.startswith("GPU_ROUTER_")}
        self.env.update(
            {"GPU_ROUTER_HOME": str(home), "GPU_ROUTER_TEST_MODE": "1", "GPU_ROUTER_PORT": "0"}
        )

    def __call__(self, *args: str, timeout: float = 60) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [*_gpu_argv(), *args],
            cwd=self.project,
            env=self.env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    def json(self, *args: str, code: int = 0) -> Any:
        res = self(*args, "--json")
        assert res.returncode == code, (args, res.returncode, res.stdout, res.stderr)
        return json.loads(res.stdout)

    def wait_state(self, job_id: str, states: set[str], timeout_s: float = 30) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_s
        while True:
            job: dict[str, Any] = self.json("status", job_id)["job"]
            if job["state"] in states:
                return job
            assert time.monotonic() < deadline, f"{job_id} stuck in {job['state']}"
            time.sleep(0.2)


@pytest.fixture
def gpu(tmp_path: Path) -> Iterator[Gpu]:
    home = tmp_path / "home"
    home.mkdir()
    (home / "providers.yaml").write_text(FAST_PROVIDERS)
    (home / "config.yaml").write_text(FAST_CONFIG)
    project = tmp_path / "proj"
    project.mkdir()
    (project / "train.py").write_text(TRAIN_PY)
    (project / "gpu.yaml").write_text(GPU_YAML)
    git_env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}
    subprocess.run(["git", "init", "-q", "."], cwd=project, check=True, env=git_env)
    g = Gpu(home, project)
    try:
        yield g
    finally:
        g("daemon", "stop", timeout=30)
        info = Paths(home=home).runtime
        if info.exists():  # stop failed: never leave a daemon behind
            with contextlib.suppress(ProcessLookupError, KeyError, ValueError):
                os.kill(int(json.loads(info.read_text())["pid"]), signal.SIGKILL)


def test_phase2_flow_through_real_executable(gpu: Gpu) -> None:
    # run: auto-starts the daemon (stderr note), detaches with --json
    res = gpu("run", "train.py", "--provider", "fake", "--json")
    assert res.returncode == 0, (res.stdout, res.stderr)
    assert "starting it" in res.stderr
    job = json.loads(res.stdout)["job"]
    job_id, short = job["id"], job["short_id"]
    assert job["spec"]["provider"] == "fake"
    assert job["bundle_sha256"], "daemon did not bundle the project at submit"

    done = gpu.wait_state(job_id, {"done", "failed", "cancelled"})
    assert done["state"] == "done", done["message"]
    outputs = gpu.project / "runs" / short
    assert Path(done["outputs_dir"]).resolve() == outputs.resolve()
    assert outputs.is_dir()
    assert any(outputs.iterdir())

    # status by prefix
    detail = gpu.json("status", job_id[:6])
    assert detail["job"]["id"] == job_id
    assert detail["attempts"][0]["provider"] == "fake"

    # logs: NDJSON records
    res = gpu("logs", short, "--json")
    assert res.returncode == 0, res.stderr
    records = [json.loads(ln) for ln in res.stdout.splitlines() if ln.strip()]
    assert records
    assert all({"attempt", "offset", "line"} <= r.keys() for r in records)

    # fetch again after deleting the outputs
    for f in outputs.iterdir():
        f.unlink()
    fetched = gpu.json("fetch", short)
    assert fetched["fetched"] is True
    assert fetched["files"] >= 1
    assert any(outputs.iterdir())

    # cancel a running job
    gpu.project.joinpath("gpu.yaml").write_text(GPU_YAML.replace("0.5", "60"))
    long_job = gpu.json("run", "train.py", "--provider", "fake")["job"]
    gpu.wait_state(long_job["id"], {"running"})
    assert gpu.json("cancel", long_job["short_id"])["state"] in {"cancelling", "cancelled"}
    assert gpu.wait_state(long_job["id"], {"cancelled"})["state"] == "cancelled"
    assert gpu.json("cancel", long_job["short_id"])["state"] == "cancelled"  # idempotent

    # jobs / history
    assert all(j["id"] != long_job["id"] for j in gpu.json("jobs")["jobs"])
    history = {j["id"]: j["state"] for j in gpu.json("history")["jobs"]}
    assert history == {job_id: "done", long_job["id"]: "cancelled"}
    all_jobs = {j["id"] for j in gpu.json("jobs", "--all")["jobs"]}
    assert all_jobs == {job_id, long_job["id"]}

    # error classes keep their exit codes
    assert gpu.json("status", "ffff", code=4)["error"]["code"] == "job_not_found"
    assert gpu("logs", "zz", "--json").returncode == 2

    # the bundle the daemon materialized runs under the remote runner
    bundle = gpu.home / "jobs" / job_id / "bundle"
    assert (bundle / "code" / "train.py").read_text() == TRAIN_PY
    out_dir = gpu.home.parent / "remote-out"
    run_env = {
        **gpu.env,
        "GPU_ROUTER_PROTOCOL": "1",
        "GPU_ROUTER_JOB_ID": job_id,
        "GPU_ROUTER_ATTEMPT": "1",
        "GPU_OUTPUT_DIR": str(out_dir),
        "GPU_SKIP_INSTALL": "1",
    }
    workdir = gpu.home.parent / "remote-wd"
    proc = subprocess.run(
        [sys.executable, "gpu_runner/bootstrap.py", "--workdir", str(workdir)],
        cwd=bundle,
        env=run_env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (out_dir / "result.txt").read_text() == "trained ok\n"
    assert '::gpu:: {"t":"metric","step":3,"total":3' in proc.stdout
    assert '::gpu:: {"t":"exit","code":0}' in proc.stdout
    assert (workdir / "EXIT").read_text().strip() == "0"
