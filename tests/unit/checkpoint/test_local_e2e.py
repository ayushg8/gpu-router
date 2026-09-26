"""End to end on the real local provider (real launcher, runner and job processes, real
clock): a run whose session dies mid-way is migrated and resumes on a new attempt from the
checkpoint the runner synced into local checkpoint storage, with ckpt seq continuing.

Also: the daemon runtime wires a CheckpointHub into the engine and registers it for the
adapters' side channel."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

from gpu_router.adapters.registry import AdapterRegistry
from gpu_router.checkpoint import active_hub
from gpu_router.checkpoint.hub import CheckpointHub
from gpu_router.clock import SystemClock
from gpu_router.config import Config, DaemonConfig, EngineConfig, ProviderSettings
from gpu_router.engine.calls import AdapterCaller
from gpu_router.engine.deps import EngineDeps
from gpu_router.engine.supervisor import Supervisor
from gpu_router.models import AttemptState, JobSpec, JobState, Reason, Source
from gpu_router.packaging import BundleBuilder
from gpu_router.paths import Paths
from gpu_router.policy import default_policy
from gpu_router.providers.catalog import Catalog
from gpu_router.router.simple import SimpleRouter
from gpu_router.statemachine import is_terminal
from gpu_router.store import Store

pytestmark = pytest.mark.slow

TRAIN = """\
import json, os, signal, time
import gpu

r = gpu.resume_dir()
start = json.loads((r / "state.json").read_text())["step"] if r else 0
print("start=%d resumed=%s" % (start, gpu.is_resumed()), flush=True)
gpu.total_steps(6)
for step in range(start + 1, 7):
    gpu.log(step=step, loss=1.0 / step)
    with gpu.atomic_checkpoint("state.json") as tmp:
        tmp.write_text(json.dumps({"step": step}))
    time.sleep(0.3)
    if step == 3 and not gpu.is_resumed():
        # wait until the runner synced step 3 (interval 1 s, files stable for 2 s)
        root = os.environ["GPU_STORAGE"][len("file://"):]
        latest = os.path.join(root, "jobs", os.environ["GPU_ROUTER_JOB_ID"], "latest.json")
        deadline = time.time() + 30
        while time.time() < deadline:
            try:
                if json.load(open(latest))["step"] == 3:
                    break
            except (OSError, ValueError, KeyError):
                pass
            time.sleep(0.2)
        print("the session dies now", flush=True)
        os.kill(os.getppid(), signal.SIGKILL)  # the runner goes: no EXIT, no final sync
        os.kill(os.getpid(), signal.SIGKILL)
(gpu.output_dir() / "done.txt").write_text("step %d" % step)
print("finished at step %d" % step, flush=True)
"""


def _config() -> Config:
    return Config(
        test_mode=True,
        daemon=DaemonConfig(port=0, shutdown_grace_s=1),
        engine=EngineConfig(
            backoff_base_s=0.2,
            backoff_cap_s=1,
            default_poll_interval_s=0.3,
            max_workers=4,
        ),
        providers={
            "local": ProviderSettings(
                enabled=True, env="system", python=sys.executable, poll_interval_s=0.3
            )
        },
    )


async def test_local_session_death_resumes_from_synced_checkpoint(
    tmp_path: Path, paths: Paths, catalog: Catalog
) -> None:
    clock = SystemClock()
    config = _config()
    # keep the finished job's storage to inspect it (D44 deletes it by default)
    config.checkpoint = config.checkpoint.model_copy(update={"cleanup": False})
    store = Store.open_memory(clock)
    registry = AdapterRegistry.build(config=config, catalog=catalog, paths=paths, clock=clock)
    assert "local" in registry
    hub = CheckpointHub.from_config(config, paths, clock)
    deps = EngineDeps(
        store=store,
        registry=registry,
        router=SimpleRouter(),
        policy=default_policy(),
        caller=AdapterCaller(registry, config.engine),
        clock=clock,
        config=config,
        paths=paths,
        bundler=BundleBuilder(paths),
        checkpoints=hub,
    )
    sup = Supervisor(deps)
    project = tmp_path / "proj"
    project.mkdir()
    (project / "train.py").write_text(TRAIN)
    spec = JobSpec(
        project_dir=str(project),
        script="train.py",
        provider="local",
        source=Source.API,
        env={"GPU_CHECKPOINT_INTERVAL_S": "1"},
    )
    await sup.start()
    try:
        job, _ = await sup.submit(spec, actor="api")
        for _ in range(900):
            if is_terminal(store.get_job(job.id).state):
                break
            await asyncio.sleep(0.1)
        done = store.get_job(job.id)
    finally:
        await sup.stop()
        hub.close()

    logs = {
        n: paths.job_log(job.id, n).read_text() if paths.job_log(job.id, n).exists() else ""
        for n in (1, 2)
    }
    assert done.state is JobState.DONE, (done.message, logs)
    first, second = store.attempts_for(job.id)
    assert first.state is AttemptState.LOST
    assert second.state is AttemptState.SUCCEEDED
    assert '"t":"ckpt_end"' in logs[1]  # synced before the session died
    assert "start=3 resumed=True" in logs[2]  # restored the synced step-3 checkpoint

    ckpts = store.checkpoints_for(job.id)
    seqs = [c.seq for c in ckpts]
    assert seqs == sorted(seqs)
    assert len(set(seqs)) == len(seqs)
    first_seqs = [c.seq for c in ckpts if c.attempt_id == first.id]
    second_seqs = [c.seq for c in ckpts if c.attempt_id == second.id]
    assert first_seqs
    assert second_seqs
    assert second_seqs[0] == max(first_seqs) + 1  # seq continues across attempts
    assert second.resume_checkpoint_id == f"{job.id}.c{max(first_seqs)}"
    root = paths.home / "storage" / "jobs" / job.id
    assert all(c.uri.startswith(root.resolve().as_uri()) for c in ckpts)
    latest = json.loads((root / "latest.json").read_text())
    assert latest["seq"] == max(seqs)
    assert latest["attempt"] == 2
    assert latest["step"] == 6
    assert json.loads((root / "owner.json").read_text())["attempt"] == 2
    assert json.loads((root / f"ckpt-{max(seqs):04d}" / "state.json").read_text()) == {"step": 6}
    assert done.outputs_dir is not None
    assert (Path(done.outputs_dir) / "done.txt").read_text() == "step 6"
    reasons = [e.reason for e in store.events_for(job.id, limit=10_000)]
    assert Reason.SESSION_LOST in reasons
    store.close()


async def test_daemon_runtime_wires_the_hub(paths: Paths) -> None:
    from gpu_router.daemon.runtime import DaemonRuntime

    rt = DaemonRuntime.create(paths, _config(), SystemClock(), configure_logging=False)
    try:
        hub = rt.supervisor.deps.checkpoints
        assert isinstance(hub, CheckpointHub)
        assert active_hub() is hub
    finally:
        await rt.stop()
    assert active_hub() is None
