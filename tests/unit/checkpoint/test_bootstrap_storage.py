"""bootstrap.py with checkpoint storage (phase 5): publishing to storage, resuming from a
storage URI with a monotonic seq, the status channel (heartbeat + log tail), checkpoint
requests (planned handoff), ownership, secret hygiene and datasets.

Local-directory storage runs the real runner in a subprocess; the HF paths (download of
a resume checkpoint or a dataset) run in-process against a fake HfApi."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from gpu_router import protocol
from gpu_router.models import JobSpec
from gpu_router.packaging import build_bundle
from gpu_router.paths import Paths
from gpu_router.runner import bootstrap
from gpu_router.runner import storage as rs
from tests.unit.checkpoint.fake_hfapi import FakeHfApi, HTTPError
from tests.unit.packaging.helpers import isolate_git, make_project

BOOTSTRAP = Path(bootstrap.__file__).resolve()
JOB = "job000000001"


@pytest.fixture(autouse=True)
def _no_user_git_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    isolate_git(monkeypatch, tmp_path)


def _bundle(tmp_path: Path, paths: Paths, script: str) -> Path:
    project = make_project(tmp_path / "proj", {"train.py": script})
    spec = JobSpec(project_dir=str(project), script="train.py")
    return build_bundle(project, spec, paths=paths).archive


def _run(
    archive: Path, work: Path, env: dict[str, str], *extra: str
) -> subprocess.CompletedProcess[str]:
    full = {k: v for k, v in os.environ.items() if not k.startswith(("GPU_", "PYTHONPATH"))}
    full.update(env)
    return subprocess.run(
        [sys.executable, str(BOOTSTRAP), "--bundle", str(archive), "--workdir", str(work),
         "--skip-install", *extra],
        capture_output=True, text=True, timeout=120, env=full, check=False,
    )  # fmt: skip


def _events(text: str) -> list[protocol.ProtocolEvent]:
    return [ev for line in text.splitlines() if (ev := protocol.parse_line(line)) is not None]


def _env(root: Path, attempt: int, **more: str) -> dict[str, str]:
    base = {
        "GPU_STORAGE": root.as_uri(),
        "GPU_ROUTER_JOB_ID": JOB,
        "GPU_ROUTER_ATTEMPT": str(attempt),
        "GPU_CHECKPOINT_INTERVAL_S": "0.5",
        "GPU_STATUS_PUSH_S": "1",
        "GPU_CONTROL_POLL_S": "0.5",
        "GPU_DATA_DIR": str(root.parent / "data"),
    }
    base.update(more)
    return base


TRAIN = """\
import json, time, gpu
r = gpu.resume_dir()
start = json.loads((r / "state.json").read_text())["step"] if r else 0
print("start=%d" % start)
step = start + 1
gpu.log(step=step, loss=1.0 / step)
with gpu.atomic_checkpoint("state.json") as tmp:
    tmp.write_text(json.dumps({"step": step}))
time.sleep(3.2)
print("done step %d" % step)
"""


def test_publishes_to_storage_and_resumes_with_the_next_seq(tmp_path: Path, paths: Paths) -> None:
    root = tmp_path / "storage"
    archive = _bundle(tmp_path, paths, TRAIN)
    first = _run(archive, tmp_path / "w1", _env(root, 1))
    assert first.returncode == 0, first.stdout + first.stderr
    ends = [e for e in _events(first.stdout) if e.t == "ckpt_end"]
    assert [e.seq for e in ends] == [1]
    assert ends[0].uri == (root / "jobs" / JOB / "ckpt-0001").resolve().as_uri()
    assert ends[0].step == 1
    assert ends[0].size
    assert ends[0].sha256
    latest = json.loads((root / "jobs" / JOB / "latest.json").read_text())
    assert latest["seq"] == 1
    assert latest["attempt"] == 1
    assert json.loads((root / "jobs" / JOB / "ckpt-0001" / "state.json").read_text()) == {"step": 1}
    assert not (tmp_path / "w1" / ".gpu-stage" / "ckpt-0001").exists()  # moved into place
    tail = json.loads((root / "jobs" / JOB / "attempts/1/log-tail.json").read_text())
    assert tail["lines"][0].startswith('::gpu:: {"t":"hello"')
    assert tail["lines"][-1] == '::gpu:: {"t":"exit","code":0}'  # last push after the exit
    assert tail["total"] == len(tail["lines"])
    assert tail["first"] == 0
    beat = json.loads((root / "jobs" / JOB / "attempts/1/heartbeat.json").read_text())
    assert beat["phase"] == "exited"
    assert beat["exit_code"] == 0
    assert beat["step"] == 1
    assert beat["ckpt_seq"] == 1
    assert beat["metrics"] == {"loss": 1.0}

    # attempt 2 resumes from the storage URI; a stale --ckpt-seq-start cannot reuse seq 1
    second = _run(
        archive,
        tmp_path / "w2",
        _env(root, 2, GPU_RESUME_URI=latest["uri"], GPU_CKPT_KEEP="1"),
        "--ckpt-seq-start", "1",
    )  # fmt: skip
    assert second.returncode == 0, second.stdout + second.stderr
    assert "start=1" in second.stdout
    assert [e.seq for e in _events(second.stdout) if e.t == "ckpt_end"] == [2]
    latest = json.loads((root / "jobs" / JOB / "latest.json").read_text())
    assert latest["seq"] == 2
    assert latest["attempt"] == 2
    assert latest["step"] == 2
    assert not (root / "jobs" / JOB / "ckpt-0001").exists()  # keep=1 pruned it


def test_a_superseded_runner_stops_publishing(tmp_path: Path, paths: Paths) -> None:
    root = tmp_path / "storage"
    store = rs.LocalStore(root)
    rs.claim(store, JOB, 3)  # the daemon already moved the job to attempt 3
    archive = _bundle(tmp_path, paths, TRAIN)
    proc = _run(archive, tmp_path / "w", _env(root, 2))
    assert proc.returncode == 0, proc.stdout
    assert not any(e.t in ("ckpt_begin", "ckpt_end") for e in _events(proc.stdout))
    assert "gpu-router moved this job to attempt 3" in proc.stdout
    assert not (root / "jobs" / JOB / "latest.json").exists()


HANDOFF = """\
import time, gpu
deadline = time.time() + 30
while not gpu.checkpoint_requested() and time.time() < deadline:
    time.sleep(0.1)
print("requested=%s" % gpu.checkpoint_requested())
(gpu.checkpoint_dir() / "handoff.pt").write_text("saved for the handoff")
t0 = time.time()
while gpu.checkpoint_requested() and time.time() - t0 < 30:
    time.sleep(0.1)
(gpu.checkpoint_dir() / "later.pt").write_text("after the handoff")
time.sleep(3.2)
"""


def test_checkpoint_request_is_answered_then_nothing_more_is_published(
    tmp_path: Path, paths: Paths
) -> None:
    root = tmp_path / "storage"
    store = rs.LocalStore(root)
    store.write_bytes(
        rs.control_key(JOB, 1),
        rs.dumps({"id": "h1", "action": "handoff", "wait_s": 20, "reason": "session cap"}),
    )
    archive = _bundle(tmp_path, paths, HANDOFF)
    proc = _run(archive, tmp_path / "w", _env(root, 1, GPU_CHECKPOINT_INTERVAL_S="600"))
    assert proc.returncode == 0, proc.stdout
    assert "requested=True" in proc.stdout
    assert "gpu-router asked for a checkpoint (session cap)" in proc.stdout
    ends = [e for e in _events(proc.stdout) if e.t == "ckpt_end"]
    assert [e.seq for e in ends] == [1]  # the handoff one; frozen afterwards (no final sync)
    ack = rs.read_json(store, rs.ack_key(JOB, 1))
    assert ack is not None
    assert ack["id"] == "h1"
    assert ack["new"] is True
    assert ack["seq"] == 1
    assert ack["uri"] == ends[0].uri
    files = sorted(p.name for p in (root / "jobs" / JOB / "ckpt-0001").iterdir())
    assert files == [".gpu-ckpt.json", "handoff.pt"]


SECRETIVE = """\
import os
print("token visible=%s" % ("GPU_STORAGE_TOKEN" in os.environ))
print("secret is " + os.environ["MY_SECRET"])
"""


def test_storage_token_never_reaches_the_job_and_tails_are_masked(
    tmp_path: Path, paths: Paths
) -> None:
    root = tmp_path / "storage"
    archive = _bundle(tmp_path, paths, SECRETIVE)
    token = "hf_" + "t" * 34
    env = _env(
        root,
        1,
        GPU_STORAGE_TOKEN=token,
        MY_SECRET="very-secret-value",
        GPU_SECRET_NAMES="MY_SECRET",
    )
    proc = _run(archive, tmp_path / "w", env)
    assert proc.returncode == 0, proc.stdout
    assert "token visible=False" in proc.stdout
    assert "secret is very-secret-value" in proc.stdout  # the job's own output is untouched
    tail = (root / "jobs" / JOB / "attempts/1/log-tail.json").read_text()
    assert "very-secret-value" not in tail
    assert "secret is ***" in tail
    assert token not in tail


def test_unreachable_resume_uri_starts_fresh(tmp_path: Path, paths: Paths) -> None:
    root = tmp_path / "storage"
    archive = _bundle(tmp_path, paths, TRAIN)
    gone = (tmp_path / "nowhere" / "ckpt-0009").as_uri()
    proc = _run(archive, tmp_path / "w", _env(root, 2, GPU_RESUME_URI=gone))
    assert proc.returncode == 0, proc.stdout
    assert "start=0" in proc.stdout
    assert "is not reachable from here; starting fresh" in proc.stdout


DATA = """\
import gpu
print("data=" + (gpu.data_dir() / "ds" / "rows.csv").read_text().strip())
"""


def test_local_dataset_is_linked_under_the_data_dir(tmp_path: Path, paths: Paths) -> None:
    root = tmp_path / "storage"
    src = tmp_path / "mydata"
    src.mkdir()
    (src / "rows.csv").write_text("1,2,3\n")
    archive = _bundle(tmp_path, paths, DATA)
    items = json.dumps([{"mount": "ds", "local": str(src)}])
    proc = _run(archive, tmp_path / "w", _env(root, 1, GPU_DATA=items))
    assert proc.returncode == 0, proc.stdout
    assert "data=1,2,3" in proc.stdout
    assert (root.parent / "data" / "ds").is_symlink()


def test_missing_dataset_is_an_environment_failure(tmp_path: Path, paths: Paths) -> None:
    root = tmp_path / "storage"
    archive = _bundle(tmp_path, paths, DATA)
    items = json.dumps([{"mount": "ds", "local": str(tmp_path / "missing")}])
    proc = _run(archive, tmp_path / "w", _env(root, 1, GPU_DATA=items))
    assert proc.returncode == bootstrap.INSTALL_FAILED_EXIT  # adapters reroute it
    assert "datasets are not available" in proc.stdout


# --------------------------------------------------------------------------- HF, in-process


class _Tee:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def say(self, text: str) -> None:
        self.lines.append(text)

    def line(self, text: str) -> None:
        self.lines.append(text)


class _Smod:
    """The storage module, with open_store bound to a fake HfApi."""

    def __init__(self, api: FakeHfApi) -> None:
        self._api = api

    def __getattr__(self, name: str) -> Any:
        return getattr(rs, name)

    def open_store(self, uri: str, token: str | None = None) -> Any:
        return rs.open_store(uri, token=token, api=self._api)


def _bucket_with_checkpoint(api: FakeHfApi) -> str:
    api.create_bucket("tester/gpu-router", private=True)
    store = rs.open_store("hf://buckets/tester/gpu-router", api=api)
    store.write_bytes("jobs/j/ckpt-0004/model.pt", b"weights")
    store.write_bytes("jobs/j/ckpt-0004/.gpu-ckpt.json", b"{}")
    return "hf://buckets/tester/gpu-router/jobs/j/ckpt-0004"


def test_resume_download_from_a_bucket(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bootstrap, "_hf_usable", lambda: True)
    monkeypatch.setattr(time, "sleep", lambda s: None)
    api = FakeHfApi()
    uri = _bucket_with_checkpoint(api)
    smod, tee = _Smod(api), _Tee()
    dest = tmp_path / "dl"
    assert bootstrap._fetch_resume(uri, "tok", smod, None, dest, tmp_path, tee)
    assert sorted(p.name for p in dest.iterdir()) == ["model.pt"]  # no manifest
    # gone -> fresh start
    assert not bootstrap._fetch_resume(
        uri.replace("0004", "0005"), "tok", smod, None, dest, tmp_path, tee
    )
    # transient failures are retried, then the attempt ends as an environment failure
    api.fail["list_bucket_tree"] = [HTTPError(503)] * 3
    with pytest.raises(bootstrap.ResumeUnavailable):
        bootstrap._fetch_resume(uri, "tok", smod, None, dest, tmp_path, tee)
    api.fail["list_bucket_tree"] = [HTTPError(503)]
    assert bootstrap._fetch_resume(uri, "tok", smod, None, dest, tmp_path, tee)
    api.fail["list_bucket_tree"] = [HTTPError(403)]
    with pytest.raises(bootstrap.ResumeUnavailable):
        bootstrap._fetch_resume(uri, "tok", smod, None, dest, tmp_path, tee)


def test_bucket_dataset_is_downloaded_once_per_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(bootstrap, "_hf_usable", lambda: True)
    api = FakeHfApi()
    api.create_bucket("tester/gpu-router", private=True)
    store = rs.open_store("hf://buckets/tester/gpu-router", api=api)
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.txt").write_text("A")
    uri = rs.upload_dataset(store, "c0ffee", src, [("a.txt", 1)])
    smod, tee = _Smod(api), _Tee()
    items = [{"mount": "ds", "uri": uri, "sha256": "c0ffee"}]
    data_dir = tmp_path / "data"
    bootstrap._materialize_data(items, data_dir, "tok", smod, tmp_path, tee)
    assert (data_dir / "ds" / "a.txt").read_text() == "A"
    assert not (data_dir / "ds" / ".gpu-data.json").exists()
    downloads = len(api.ops("download_bucket_files"))
    bootstrap._materialize_data(items, data_dir, "tok", smod, tmp_path, tee)
    assert len(api.ops("download_bucket_files")) == downloads  # cached on this machine
    assert any("already on this machine" in line for line in tee.lines)
