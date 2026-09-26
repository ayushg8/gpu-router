"""launch.py for real on this Mac: a real bundle, the real bootstrap, a stand-in drive
folder (no lightning_sdk here, so the SDK upload fails and the folder copy is what fetch
would get), secrets, the storage token file, the wall clock."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any

import pytest

from gpu_router import protocol
from gpu_router.paths import Paths
from gpu_router.providers.lightning import launch
from gpu_router.providers.lightning.adapter import last_exit_code
from tests.unit.providers.kaggle.helpers import make_bundle

ENV_PROBE = """\
import json, os, pathlib
out = pathlib.Path(os.environ["GPU_OUTPUT_DIR"])
out.mkdir(parents=True, exist_ok=True)
(out / "result.txt").write_text("ok\\n")
keys = sorted(os.environ)
(out / "env.json").write_text(json.dumps({
    "keys": keys,
    "secret": os.environ.get("WANDB_API_KEY"),
    "plain": os.environ.get("MY_FLAG"),
}))
print("hello from train.py")
"""

SLEEPER = """\
import time
print("sleeping", flush=True)
time.sleep(120)
"""


def _folder(tmp: Path, archive: Path, *, secrets: dict[str, str] | None = None, **cfg: Any) -> Path:
    folder = tmp / "drive" / "gr-0123456789ab-1"
    folder.mkdir(parents=True)
    shutil.copyfile(launch.__file__, folder / "launch.py")
    shutil.copyfile(archive, folder / "bundle.tar.gz")
    config = {
        "name": "gr-0123456789ab-1",
        "attempt_key": "gpu-0123456789ab-1",
        "teamspace": "me/default",
        "drive_dir": "uploads/gpu-router/gr-0123456789ab-1",
        "env": {"MY_FLAG": "on", "GPU_ROUTER_JOB_ID": "0123456789ab"},
        "ckpt_seq_start": 1,
        "checkpoint_interval_min": None,
        "wall_clock_s": 600,
        "bundle_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "secrets": bool(secrets),
        "workdir": str(tmp / "work"),
        "artifacts_dir": str(tmp / "artifacts"),
        **cfg,
    }
    (folder / "launch.json").write_text(json.dumps(config))
    if secrets:
        (folder / "secrets.json").write_text(json.dumps({"v": 1, "values": secrets}))
    return folder


def _run(folder: Path, tmp: Path, *, timeout: float = 120) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GPU_")}
    env["LIGHTNING_API_KEY"] = "platform-key-0000000"
    env["GPU_SKIP_INSTALL"] = "1"
    return subprocess.run(
        [sys.executable, str(folder / "launch.py")],
        cwd=tmp,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _members(archive: Path) -> dict[str, bytes]:
    with tarfile.open(archive, "r:gz") as tar:
        return {
            m.name: tar.extractfile(m).read()  # type: ignore[union-attr]
            for m in tar.getmembers()
            if m.isfile()
        }


@pytest.mark.slow
def test_a_job_runs_and_its_outputs_are_packed_for_fetch(tmp_path: Path, paths: Paths) -> None:
    archive = make_bundle(tmp_path / "proj", paths, script=ENV_PROBE)
    token = "hf_" + "t" * 34
    folder = _folder(
        tmp_path,
        archive,
        secrets={"WANDB_API_KEY": "wandb-secret-1234", "GPU_STORAGE_TOKEN": token},
    )
    res = _run(folder, tmp_path)
    assert res.returncode == 0, res.stdout + res.stderr
    lines = res.stdout.splitlines()
    assert any("lightning attempt gpu-0123456789ab-1 starting" in x for x in lines)
    assert any(x.endswith("(files: sdk download)") for x in lines)  # not under /teamspace
    assert "hello from train.py" in lines
    assert last_exit_code(lines) == 0
    assert any("outputs: 2 file(s)" in x for x in lines)
    assert any("outputs upload to the drive failed" in x for x in lines)  # no SDK here
    for where in (folder / "out", tmp_path / "artifacts"):
        members = _members(where / "outputs.tar.gz")
        assert members["result.txt"] == b"ok\n"
        probe = json.loads(members["env.json"])
    assert probe["secret"] == "wandb-secret-1234"
    assert probe["plain"] == "on"
    assert "LIGHTNING_API_KEY" not in probe["keys"]
    assert "GPU_STORAGE_TOKEN" not in probe["keys"]
    assert not (folder / "secrets.json").exists()
    assert not (tmp_path / "work" / ".storage-token").exists()  # bootstrap consumed it
    assert token not in res.stdout


@pytest.mark.slow
def test_the_wall_clock_stops_the_runner(tmp_path: Path, paths: Paths) -> None:
    archive = make_bundle(tmp_path / "proj", paths, script=SLEEPER)
    folder = _folder(tmp_path, archive, wall_clock_s=3)
    res = _run(folder, tmp_path, timeout=110)
    assert res.returncode != 0
    lines = res.stdout.splitlines()
    assert any(launch.WALL_MARK in x for x in lines)
    assert "sleeping" in lines
    code = last_exit_code(lines)
    assert code is not None
    assert code != 0
    assert (folder / "out" / "outputs.tar.gz").exists()  # outputs still delivered


def test_a_corrupted_bundle_is_exit_90(tmp_path: Path, paths: Paths) -> None:
    archive = make_bundle(tmp_path / "proj", paths, script=ENV_PROBE)
    folder = _folder(tmp_path, archive, bundle_sha256="0" * 64)
    res = _run(folder, tmp_path)
    assert res.returncode == 90
    assert last_exit_code(res.stdout.splitlines()) == 90
    assert protocol.exit_line(90) in res.stdout


def test_missing_secrets_file_is_a_note_not_a_crash(tmp_path: Path) -> None:
    folder = tmp_path / "f"
    folder.mkdir()
    values = launch.read_secrets(str(folder), {"secrets": True}, {})
    assert values == {}
