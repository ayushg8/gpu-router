"""Kaggle + phase 5: secrets through a private dataset (never in run.py), resume through
checkpoint storage, and near-live logs from the runner's log-tail.json."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr

from gpu_router.checkpoint import set_active_hub
from gpu_router.checkpoint.hub import CheckpointHub
from gpu_router.clock import FakeClock
from gpu_router.config import CheckpointConfig
from gpu_router.errors import InvalidJob
from gpu_router.models import Checkpoint
from gpu_router.paths import Paths
from gpu_router.providers.kaggle import remote
from gpu_router.runner import storage as rs
from tests.contract.kaggle.sim import SimKaggle
from tests.unit.engine.conftest import InlineExecutor
from tests.unit.providers.kaggle.test_adapter import (  # fixtures + builders
    JOB_ID,
    MakeAdapter,
    archive,
    make_adapter,
    make_ctx,
    make_job,
    sim,
)

__all__ = ["archive", "make_adapter", "sim"]  # re-exported pytest fixtures

TOKEN = "hf_" + "s" * 34
DATASET = "simuser/gpu-router-secrets"


def _secret_ctx(archive: Path, n: int = 1, token: str = TOKEN, **env: str) -> Any:
    base_env = {"GPU_ROUTER_JOB_ID": JOB_ID, "GPU_ROUTER_ATTEMPT": str(n), **env}
    return make_ctx(archive, n=n, env=base_env, secrets={"GPU_STORAGE_TOKEN": SecretStr(token)})


def test_secrets_ride_in_a_private_dataset_never_in_run_py(
    make_adapter: MakeAdapter, sim: SimKaggle, archive: Path, paths: Paths
) -> None:
    sim.dataset_ready_after = 1
    adapter = make_adapter()
    adapter._sleep = lambda s: None
    adapter.submit(make_job(), _secret_ctx(archive, GPU_STORAGE="hf://buckets/u/gpu-router"))
    ds = sim.datasets[DATASET]
    assert ds.public is False
    payload = json.loads(ds.files[remote.SECRETS_FILE])
    assert payload["values"] == {"GPU_STORAGE_TOKEN": TOKEN}
    kernel = sim.kernels[f"gpu-router-{JOB_ID}-1"]
    assert kernel.metadata["dataset_sources"] == [DATASET]
    assert TOKEN not in kernel.run_py  # version history never sees it
    assert f"SECRETS_DATASET = {DATASET!r}" in kernel.run_py
    assert "GPU_STORAGE" in kernel.run_py  # non-secret env still rides in run.py
    leftovers = [p for p in (paths.provider_dir("kaggle") / "secrets").rglob("*") if p.is_file()]
    assert sorted(p.name for p in leftovers) == ["dataset.json", "salt"]  # values deleted
    assert TOKEN not in (paths.provider_dir("kaggle") / "secrets" / "dataset.json").read_text()

    # same values: no new version; changed values: a new version, old ones deleted
    adapter.submit(make_job(), _secret_ctx(archive, n=2))
    assert ds.version == 1
    assert sim.kernels[f"gpu-router-{JOB_ID}-2"].metadata["dataset_sources"] == [DATASET]
    adapter.submit(make_job(), _secret_ctx(archive, n=3, token="hf_" + "z" * 34))
    assert ds.version == 2
    assert ds.deleted_old == 1


def test_secrets_off_refuses_jobs_that_need_them(
    make_adapter: MakeAdapter, sim: SimKaggle, archive: Path
) -> None:
    adapter = make_adapter(secrets_dataset=False)
    with pytest.raises(InvalidJob, match="secrets"):
        adapter.submit(make_job(), _secret_ctx(archive))
    assert sim.pushes == []
    assert sim.datasets == {}
    adapter.submit(make_job(), make_ctx(archive))  # no secrets: fine
    assert sim.kernels[f"gpu-router-{JOB_ID}-1"].metadata["dataset_sources"] == []


def test_a_storage_checkpoint_is_not_embedded(
    make_adapter: MakeAdapter, sim: SimKaggle, archive: Path, tmp_path: Path
) -> None:
    local = tmp_path / "ckpt.tar.gz"
    local.write_bytes(b"x" * 10)
    ckpt = Checkpoint(
        id=f"{JOB_ID}.c4", job_id=JOB_ID, attempt_id=f"{JOB_ID}.1", seq=4,
        uri="hf://buckets/u/gpu-router/jobs/x/ckpt-0004", created_at=0, recorded_at=0,
    )  # fmt: skip
    ctx = make_ctx(
        archive,
        n=2,
        resume_from=ckpt,
        env={"GPU_RESUME_URI": ckpt.uri, "GPU_ROUTER_JOB_ID": JOB_ID},
    )
    make_adapter().submit(make_job(), ctx)
    run_py = sim.kernels[f"gpu-router-{JOB_ID}-2"].run_py
    assert "RESUME_SHA256 = None" in run_py
    assert "cannot reach kaggle" not in run_py  # the runner downloads it itself
    assert "CKPT_SEQ_START = 5" in run_py
    assert make_adapter().capabilities.resume is True


def test_run_py_loads_secrets_from_the_attached_dataset(tmp_path: Path) -> None:
    """The generated run.py's loader, executed on this Mac against a fake /kaggle/input."""
    inputs = tmp_path / "input"
    text = remote.render_runner(
        attempt_key=f"gpu-{JOB_ID}-1",
        bundle=b"",
        bundle_sha256="0" * 64,
        env={},
        secrets_dataset=DATASET,
        input_root=str(inputs),
    )
    prelude = text[: text.index("def write_blob(")]
    program = prelude + "print(json.dumps(load_secrets()))\n"

    def run() -> str:
        out = subprocess.run(
            [sys.executable, "-c", program], capture_output=True, text=True, check=True
        )
        return out.stdout.splitlines()[-1]

    assert run() == "{}"  # not attached: runs without secrets (with a note)
    mount = inputs / "datasets" / DATASET  # the newer mount layout
    mount.mkdir(parents=True)
    (mount / remote.SECRETS_FILE).write_text(json.dumps({"values": {"A_TOKEN": "v1"}}))
    assert json.loads(run()) == {"A_TOKEN": "v1"}


# --------------------------------------------------------------------------- live logs


@pytest.fixture
def hub(paths: Paths, clock: FakeClock) -> Any:
    h = CheckpointHub(
        CheckpointConfig(),
        paths,
        clock,
        test_mode=True,
        executor=InlineExecutor(),
        bulk_executor=InlineExecutor(),
    )
    set_active_hub(h)
    yield h
    set_active_hub(None)


def test_live_logs_come_from_the_tail_then_the_final_log(
    make_adapter: MakeAdapter, sim: SimKaggle, archive: Path, clock: FakeClock, hub: Any
) -> None:
    local = hub.local()
    root = local.root_uri
    slug = f"gpu-router-{JOB_ID}-1"
    sim.register(slug, {"duration": 100, "steps": 6})
    adapter = make_adapter()
    ref = adapter.submit(make_job(), make_ctx(archive, env={"GPU_STORAGE": root}))
    assert ref.meta["status_uri"] == f"{root}/jobs/{JOB_ID}/attempts/1"
    store = rs.LocalStore(local.raw.root)

    def push(lines: list[str], first: int = 0) -> None:
        store.write_bytes(
            rs.log_tail_key(JOB_ID, 1),
            rs.dumps({"first": first, "total": first + len(lines), "lines": lines}),
        )

    clock.advance(10)
    served: list[str] = []
    chunks = list(adapter.logs(ref))
    assert chunks[0].lines == []
    assert not chunks[0].eof
    cursor = chunks[0].cursor

    kernel = sim.kernels[slug]
    runner_lines = sim._log_lines(kernel, sim._timeline(kernel))[1:]  # the runner's own
    push(runner_lines[:3])
    (chunk,) = adapter.logs(ref, since=cursor)
    assert chunk.lines == runner_lines[:3]
    assert chunk.cursor.startswith("t3:")
    served += chunk.lines
    (again,) = adapter.logs(ref, since=chunk.cursor)
    assert again.lines == []
    assert again.cursor == chunk.cursor

    push(runner_lines[:5])
    (chunk,) = adapter.logs(ref, since=chunk.cursor)
    assert chunk.lines == runner_lines[3:5]
    served += chunk.lines

    clock.advance(200)  # finished: the final log takes over where the tail stopped
    final = list(adapter.logs(ref, since=chunk.cursor))
    rest = [line for c in final for line in c.lines]
    full = json.loads(json.dumps(adapter._read_final(slug)["lines"]))
    hello = next(i for i, line in enumerate(full) if line.startswith('::gpu:: {"t":"hello"'))
    assert rest == full[:hello] + full[hello + 5 :]  # preamble once, then no repeats
    assert final[-1].eof
    assert served + rest[hello:] == full[hello:]


def test_lines_that_scrolled_out_of_the_tail_are_summarised(
    make_adapter: MakeAdapter, sim: SimKaggle, archive: Path, clock: FakeClock, hub: Any
) -> None:
    root = hub.local().root_uri
    sim.register(f"gpu-router-{JOB_ID}-1", {"duration": 100})
    adapter = make_adapter()
    ref = adapter.submit(make_job(), make_ctx(archive, env={"GPU_STORAGE": root}))
    store = rs.LocalStore(hub.local().raw.root)
    clock.advance(5)
    store.write_bytes(
        rs.log_tail_key(JOB_ID, 1),
        rs.dumps({"first": 40, "total": 42, "lines": ["line 40", "line 41"]}),
    )
    (chunk,) = adapter.logs(ref, since="t10:abc")
    assert "30 log lines scrolled past" in chunk.lines[0]
    assert chunk.lines[1:] == ["line 40", "line 41"]
    assert chunk.cursor.startswith("t42:")


def test_no_side_channel_keeps_the_phase3_behaviour(
    make_adapter: MakeAdapter, sim: SimKaggle, archive: Path, clock: FakeClock
) -> None:
    sim.register(f"gpu-router-{JOB_ID}-1", {"duration": 100})
    adapter = make_adapter()
    ref = adapter.submit(make_job(), make_ctx(archive))
    assert "status_uri" not in ref.meta
    clock.advance(5)
    (chunk,) = adapter.logs(ref)
    assert chunk.lines == []
    assert chunk.cursor == "0"
