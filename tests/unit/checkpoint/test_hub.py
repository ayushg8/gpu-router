"""CheckpointHub: backend choice, token handling and degrade messages, per-attempt env,
checkpoint copies between backends, handoff control files, datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from gpu_router import secrets
from gpu_router.checkpoint.hub import STORAGE_TOKEN_ENV, CheckpointHub, StoredCheckpoint
from gpu_router.checkpoint.storage import StorageError
from gpu_router.clock import FakeClock
from gpu_router.config import CheckpointConfig
from gpu_router.models import Checkpoint
from gpu_router.paths import Paths
from gpu_router.runner import storage as rs
from tests.unit.checkpoint.fake_hfapi import FakeHfApi, HTTPError
from tests.unit.engine.conftest import InlineExecutor

TOKEN = "hf_" + "a" * 34
REMOTE = "hf_" + "b" * 34


def make_hub(
    paths: Paths, clock: FakeClock, api: FakeHfApi | None = None, **cfg: Any
) -> CheckpointHub:
    return CheckpointHub(
        CheckpointConfig(**cfg),
        paths,
        clock,
        test_mode=True,
        hf_api=api.factory if api is not None else None,
        executor=InlineExecutor(),
        bulk_executor=InlineExecutor(),
    )


def ckpt(job: str, seq: int, uri: str) -> Checkpoint:
    return Checkpoint(
        id=f"{job}.c{seq}",
        job_id=job,
        attempt_id=f"{job}.1",
        seq=seq,
        uri=uri,
        created_at=0,
        recorded_at=0,
    )


def test_local_kind_uses_the_local_dir(paths: Paths, clock: FakeClock) -> None:
    hub = make_hub(paths, clock)
    store = hub.backend_for("local")
    assert store is not None
    assert store.kind == "local"
    assert store.root_uri == (paths.home / "storage").resolve().as_uri()
    assert hub.backend_for("kaggle") is None  # test mode, nothing injected: hf off
    assert hub.status().reason == "hf storage is off in test mode"
    assert not hub.remote_expected()


def test_no_token_degrades_with_a_login_hint(paths: Paths, clock: FakeClock) -> None:
    api = FakeHfApi()
    hub = make_hub(paths, clock, api)
    assert hub.remote_expected()
    assert hub.backend_for("kaggle") is None
    st = hub.status()
    assert st.reason == "no Hugging Face token in the Keychain"
    assert st.hint is not None
    assert "gpu login hf" in st.hint
    assert api.calls == []  # nothing was tried without a token
    secrets.set_secret("HF_TOKEN", TOKEN)
    assert hub.backend_for("kaggle") is None  # the negative answer is cached for a minute
    clock.advance(61)
    store = hub.backend_for("kaggle")
    assert store is not None
    assert store.root_uri == "hf://buckets/tester/gpu-router"
    assert api.private["tester/gpu-router"] is True
    assert hub.status().reason is None


def test_namespace_is_cached_per_token(paths: Paths, clock: FakeClock) -> None:
    secrets.set_secret("HF_TOKEN", TOKEN)
    api = FakeHfApi()
    assert make_hub(paths, clock, api).hf() is not None
    assert make_hub(paths, clock, api).hf() is not None  # a restarted daemon
    assert len(api.ops("whoami")) == 1
    cache = json.loads((paths.home / "storage" / "hf.json").read_text())
    assert cache["namespace"] == "tester"
    assert TOKEN not in json.dumps(cache)
    secrets.set_secret("HF_TOKEN", REMOTE)  # another token: ask again
    assert make_hub(paths, clock, api).hf() is not None
    assert len(api.ops("whoami")) == 2


def test_explicit_bucket_and_backend_settings(paths: Paths, clock: FakeClock) -> None:
    secrets.set_secret("HF_TOKEN", TOKEN)
    api = FakeHfApi()
    store = make_hub(paths, clock, api, bucket="team/ckpts").hf()
    assert store is not None
    assert store.root_uri == "hf://buckets/team/ckpts"
    assert api.ops("whoami") == []
    assert make_hub(paths, clock, api, backend="local").backend_for("kaggle") is None
    off = make_hub(paths, clock, api, backend="off")
    assert off.backend_for("local") is None
    assert off.prepare_attempt(job_id="j", attempt_n=1, kind="local", resume=None) is None


def test_rejected_token_is_not_retried_every_call(paths: Paths, clock: FakeClock) -> None:
    secrets.set_secret("HF_TOKEN", TOKEN)
    api = FakeHfApi()
    api.fail["whoami"] = [HTTPError(401)]
    hub = make_hub(paths, clock, api)
    assert hub.hf() is None
    assert hub.status().reason == "hugging face rejected the token (whoami)"
    assert hub.hf() is None
    assert len(api.ops("whoami")) == 1


def test_prepare_attempt_for_a_remote_runner(paths: Paths, clock: FakeClock) -> None:
    secrets.set_secret("HF_TOKEN", TOKEN)
    secrets.set_secret("HF_TOKEN_REMOTE", REMOTE)
    api = FakeHfApi()
    hub = make_hub(paths, clock, api, status_push_s=30, keep=2)
    got = hub.prepare_attempt(
        job_id="job1", attempt_n=2, kind="kaggle", resume=None, secret_names=["WANDB_API_KEY"]
    )
    assert got is not None
    assert got.kind == "hf"
    assert got.env["GPU_STORAGE"] == "hf://buckets/tester/gpu-router"
    assert got.env["GPU_STATUS_PUSH_S"] == "30"
    assert got.env["GPU_CKPT_KEEP"] == "2"
    assert got.env["GPU_SECRET_NAMES"] == "WANDB_API_KEY"
    assert "GPU_RESUME_URI" not in got.env
    assert got.secrets[STORAGE_TOKEN_ENV].get_secret_value() == REMOTE  # least privilege
    assert TOKEN not in json.dumps(got.env)
    assert REMOTE not in json.dumps(got.env)
    owner = json.loads(api.files("tester/gpu-router")["jobs/job1/owner.json"])
    assert owner["attempt"] == 2


def test_resume_uri_same_backend_and_copies(paths: Paths, clock: FakeClock, tmp_path: Path) -> None:
    secrets.set_secret("HF_TOKEN", TOKEN)
    secrets.set_secret("HF_TOKEN_REMOTE", REMOTE)
    api = FakeHfApi()
    hub = make_hub(paths, clock, api)
    local = hub.local()
    hf = hub.hf()
    assert local is not None
    assert hf is not None
    src = tmp_path / "c"
    src.mkdir()
    (src / "w.pt").write_text("weights")
    latest = rs.publish_checkpoint(local.raw, "job1", 1, src, [("w.pt", 7, "x")], attempt=1)
    local_ck = ckpt("job1", 1, latest["uri"])

    # a local runner resumes a local checkpoint as is
    got = hub.prepare_attempt(job_id="job1", attempt_n=2, kind="local", resume=local_ck)
    assert got is not None
    assert got.env["GPU_RESUME_URI"] == latest["uri"]
    assert got.notes == []

    # a remote runner gets it copied into the bucket first
    got = hub.prepare_attempt(job_id="job1", attempt_n=3, kind="kaggle", resume=local_ck)
    assert got is not None
    assert got.env["GPU_RESUME_URI"] == "hf://buckets/tester/gpu-router/jobs/job1/ckpt-0001"
    assert "copied checkpoint 1 from local storage to hf storage" in got.notes[0]
    files = api.files("tester/gpu-router")
    assert files["jobs/job1/ckpt-0001/w.pt"] == b"weights"
    assert "jobs/job1/ckpt-0001/.gpu-ckpt.json" in files

    # and back: a checkpoint a remote runner wrote comes down for a local attempt
    stage = tmp_path / "remote"
    stage.mkdir()
    (stage / "w.pt").write_text("newer")
    rlatest = rs.publish_checkpoint(hf.raw, "job1", 2, stage, [("w.pt", 5, "y")], attempt=3)
    got = hub.prepare_attempt(
        job_id="job1", attempt_n=4, kind="local", resume=ckpt("job1", 2, rlatest["uri"])
    )
    assert got is not None
    path = Path(got.env["GPU_RESUME_URI"].removeprefix("file://"))
    assert (path / "w.pt").read_text() == "newer"

    # URIs no backend owns are left to the adapter
    got = hub.prepare_attempt(
        job_id="job1", attempt_n=5, kind="local", resume=ckpt("job1", 3, "fake://x/y")
    )
    assert got is not None
    assert "GPU_RESUME_URI" not in got.env


def test_latest_picks_the_newest_backend(paths: Paths, clock: FakeClock, tmp_path: Path) -> None:
    secrets.set_secret("HF_TOKEN", TOKEN)
    api = FakeHfApi()
    hub = make_hub(paths, clock, api)
    assert hub.latest("job1") is None
    src = tmp_path / "c"
    src.mkdir()
    (src / "a").write_text("1")
    local, hf = hub.local(), hub.hf()
    assert local is not None
    assert hf is not None
    rs.publish_checkpoint(local.raw, "job1", 2, src, [("a", 1, "x")], attempt=1, step=10)
    rs.publish_checkpoint(hf.raw, "job1", 5, src, [("a", 1, "x")], attempt=2, step=50)
    found = hub.latest("job1")
    assert found == StoredCheckpoint(
        seq=5,
        uri="hf://buckets/tester/gpu-router/jobs/job1/ckpt-0005",
        attempt=2,
        step=50,
        size=1,
        sha256=found.sha256 if found else None,
        created_at=found.created_at if found else None,
    )


def test_control_request_and_ack(paths: Paths, clock: FakeClock) -> None:
    hub = make_hub(paths, clock)
    hub.request_checkpoint(
        job_id="j", attempt_n=1, kind="local", request_id="r1", action="handoff",
        wait_s=60, reason="cap",
    )  # fmt: skip
    local = hub.local()
    assert local is not None
    req = local.read_json("jobs/j/attempts/1/control.json")
    assert req is not None
    assert req["id"] == "r1"
    assert req["action"] == "handoff"
    assert hub.read_ack(job_id="j", attempt_n=1, kind="local") is None
    local.write_json(
        "jobs/j/attempts/1/control-ack.json",
        {"id": "r1", "new": True, "seq": 4, "uri": "file:///x", "step": 9},
    )
    ack = hub.read_ack(job_id="j", attempt_n=1, kind="local")
    assert ack is not None
    assert ack.request_id == "r1"
    assert ack.new
    assert ack.checkpoint is not None
    assert ack.checkpoint.seq == 4
    assert ack.checkpoint.step == 9
    with pytest.raises(StorageError):
        hub.request_checkpoint(
            job_id="j", attempt_n=1, kind="kaggle", request_id="r2", action="handoff",
            wait_s=60, reason="cap",
        )  # fmt: skip


def test_datasets_digest_upload_and_presence(
    paths: Paths, clock: FakeClock, tmp_path: Path
) -> None:
    secrets.set_secret("HF_TOKEN", TOKEN)
    api = FakeHfApi()
    hub = make_hub(paths, clock, api)
    data = tmp_path / "ds"
    data.mkdir()
    (data / "train.csv").write_text("a,b\n1,2\n")
    dig = hub.digest(data)
    assert dig.count == 1
    assert dig.size == 8
    assert hub.digest(data) == dig  # from the index the second time
    assert hub.dataset_uri(dig.sha256) is None
    uri = hub.upload_dataset(data, dig)
    assert uri == f"hf://buckets/tester/gpu-router/datasets/{dig.sha256}"
    assert hub.dataset_uri(dig.sha256) == uri
    (data / "train.csv").write_text("a,b\n1,3\n")
    assert hub.digest(data).sha256 != dig.sha256  # content changed


async def test_call_runs_off_loop_and_times_out(paths: Paths, clock: FakeClock) -> None:
    hub = CheckpointHub(CheckpointConfig(), paths, clock, test_mode=True)
    try:
        assert await hub.call(lambda x, y=0: x + y, 1, y=2) == 3

        def slow() -> None:
            import time

            time.sleep(0.5)

        with pytest.raises(StorageError, match="timed out"):
            await hub.call(slow, limit_s=0.05)
    finally:
        hub.close()


def test_a_new_token_is_picked_up_without_a_restart(paths: Paths, clock: FakeClock) -> None:
    secrets.set_secret("HF_TOKEN", TOKEN)
    api = FakeHfApi()
    hub = make_hub(paths, clock, api)
    assert hub.hf() is not None
    assert set(api.tokens) == {TOKEN}
    built = len(api.tokens)
    secrets.set_secret("HF_TOKEN", REMOTE)  # `gpu login hf` with another token
    assert hub.hf() is not None
    assert len(api.tokens) == built  # rechecked only every 5 min
    clock.advance(301)
    assert hub.hf() is not None
    assert api.tokens[-1] == REMOTE
    secrets.delete_secret("HF_TOKEN")
    clock.advance(301)
    assert hub.hf() is None
    assert hub.status().reason == "no Hugging Face token in the Keychain"


def test_remote_data_possible_follows_the_cached_state(paths: Paths, clock: FakeClock) -> None:
    """The routing path's question (D61): may a job's data= reach a remote runner through
    HF? Never decided by a network call, never "no" for trouble that should pass."""
    assert not make_hub(paths, clock, backend="local").remote_data_possible()
    api = FakeHfApi()
    hub = make_hub(paths, clock, api)
    assert hub.remote_data_possible()  # not tried yet: do not rule anything out
    assert hub.hf() is None  # no token: refused
    assert not hub.remote_data_possible()
    clock.advance(61)  # past the refusal's pause: a new `gpu login hf` must count
    assert hub.remote_data_possible()
    hub._hf_failed("hf storage did not answer", None, 60, transient=True)
    assert hub.remote_data_possible()  # a network blip: the placement waits instead
    secrets.set_secret("HF_TOKEN", TOKEN)
    clock.advance(61)
    assert hub.hf() is not None
    assert hub.remote_data_possible()
    # the bucket is fine but remote runners have no token (HF_TOKEN without _REMOTE)
    assert hub.prepare_attempt(job_id="j1", attempt_n=1, kind="kaggle", resume=None) is None
    assert not hub.remote_data_possible()
    secrets.set_secret("HF_TOKEN_REMOTE", REMOTE)
    assert hub.prepare_attempt(job_id="j1", attempt_n=2, kind="kaggle", resume=None) is not None
    assert hub.remote_data_possible()
