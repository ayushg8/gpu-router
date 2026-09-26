"""Kaggle push payload: naming, metadata, and the generated run.py executed for real."""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from gpu_router.paths import Paths
from gpu_router.providers.kaggle import parse, remote
from tests.unit.providers.kaggle.helpers import make_bundle


def test_slug_and_title_follow_the_attempt_key() -> None:
    key = "gpu-0123456789ab-2"
    assert remote.slug_for_key(key) == "gpu-router-0123456789ab-2"
    assert remote.kernel_title(key) == "gpu-router 0123456789ab 2"
    # the CLI's slugify(title) lowercases and joins words with '-' (verified with
    # python-slugify inside the kaggle tool env): title and slug agree
    assert remote.kernel_title(key).replace(" ", "-") == remote.slug_for_key(key)
    assert len(remote.kernel_title(key)) >= 5


@pytest.mark.parametrize("bad", ["gpu-XYZ-1", "fk-gpu-0123456789ab-1", "gpu-0123456789ab-0", ""])
def test_non_keys_have_no_slug(bad: str) -> None:
    assert remote.slug_for_key(bad) is None
    with pytest.raises(ValueError, match="not an attempt key"):
        remote.kernel_title(bad)


def test_metadata_is_private_and_explicit() -> None:
    meta = remote.kernel_metadata(
        owner="me", slug="gpu-router-0123456789ab-1", title="t" * 5, machine_shape="NvidiaTeslaT4"
    )
    assert meta["id"] == "me/gpu-router-0123456789ab-1"
    assert meta["is_private"] is True
    assert meta["enable_gpu"] is True
    assert meta["enable_internet"] is True
    assert meta["machine_shape"] == "NvidiaTeslaT4"
    assert meta["kernel_type"] == "script"
    assert meta["code_file"] == "run.py"
    cpu = remote.kernel_metadata(
        owner="me", slug="s", title="title", machine_shape=None, enable_internet=False
    )
    assert cpu["enable_gpu"] is False
    assert cpu["enable_internet"] is False
    assert "machine_shape" not in cpu


def _render(tmp_path: Path, archive: Path, **kw: object) -> tuple[Path, Path]:
    data = archive.read_bytes()
    out = tmp_path / "kaggle-working" / "outputs"
    script = remote.render_runner(
        attempt_key="gpu-0123456789ab-1",
        bundle=data,
        bundle_sha256=kw.pop("sha", None) or hashlib.sha256(data).hexdigest(),  # type: ignore[arg-type]
        env={"GPU_ROUTER_JOB_ID": "0123456789ab", "GPU_ROUTER_ATTEMPT": "1", "MY_FLAG": "x"},
        workdir=str(tmp_path / "vm-tmp"),
        output_dir=str(out),
        sync_dir=str(tmp_path / "kaggle-working" / ".gpu-router" / "checkpoints"),
        **kw,  # type: ignore[arg-type]
    )
    run_py = tmp_path / "run.py"
    run_py.write_text(script)
    return run_py, out


def _run(run_py: Path) -> subprocess.CompletedProcess[str]:
    env = {"PATH": "/usr/bin:/bin", "GPU_SKIP_INSTALL": "1", "HOME": str(run_py.parent)}
    return subprocess.run(
        [sys.executable, str(run_py)],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
        cwd=run_py.parent,
        check=False,
    )


def test_runner_is_valid_python38_syntax(tmp_path: Path, paths: Paths) -> None:
    archive = make_bundle(tmp_path / "proj", paths)
    run_py, _ = _render(tmp_path, archive)
    ast.parse(run_py.read_text(), feature_version=(3, 8))


def test_runner_runs_the_bundle_and_writes_outputs(tmp_path: Path, paths: Paths) -> None:
    archive = make_bundle(tmp_path / "proj", paths)
    run_py, out = _render(tmp_path, archive)
    proc = _run(run_py)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    lines = proc.stdout.splitlines()
    assert "hello from train.py" in lines
    assert parse.last_exit_code(lines) == 0
    assert (out / "result.txt").read_text() == "ok\n"
    assert (out / "sub" / "nested.txt").read_text() == "nested\n"


def test_runner_reports_nonzero_exit_and_fails_the_kernel(tmp_path: Path, paths: Paths) -> None:
    script = "import sys\nprint('boom')\nsys.exit(3)\n"
    archive = make_bundle(tmp_path / "proj", paths, script=script)
    run_py, _ = _render(tmp_path, archive)
    proc = _run(run_py)
    assert proc.returncode != 0  # the RuntimeError makes Kaggle mark the version ERROR
    assert "job exited with code 3" in proc.stderr
    assert parse.last_exit_code(proc.stdout.splitlines()) == 3


def test_runner_refuses_a_corrupted_bundle(tmp_path: Path, paths: Paths) -> None:
    archive = make_bundle(tmp_path / "proj", paths)
    run_py, out = _render(tmp_path, archive, sha="0" * 64)
    proc = _run(run_py)
    assert proc.returncode != 0
    assert "sha256 mismatch" in proc.stderr
    assert not (out / "result.txt").exists()


def test_runner_resumes_from_an_embedded_checkpoint(tmp_path: Path, paths: Paths) -> None:
    import io
    import tarfile

    script = (
        "import os, pathlib\n"
        "r = os.environ.get('GPU_RESUME_DIR')\n"
        "print('resume', r and pathlib.Path(r, 'state.txt').read_text().strip())\n"
    )
    archive = make_bundle(tmp_path / "proj", paths, script=script)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        body = b"step=7\n"
        info = tarfile.TarInfo("state.txt")
        info.size = len(body)
        tar.addfile(info, io.BytesIO(body))
    ckpt = buf.getvalue()
    run_py, _ = _render(
        tmp_path, archive, resume=ckpt, resume_sha256=hashlib.sha256(ckpt).hexdigest()
    )
    proc = _run(run_py)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "resume step=7" in proc.stdout.splitlines()


def test_env_is_embedded_as_json_and_resume_note_printed(tmp_path: Path, paths: Paths) -> None:
    script = "import os\nprint('flag', os.environ['MY_FLAG'], os.environ['GPU_ROUTER_JOB_ID'])\n"
    archive = make_bundle(tmp_path / "proj", paths, script=script)
    run_py, _ = _render(tmp_path, archive, resume_note="checkpoint 3 cannot reach kaggle yet")
    text = run_py.read_text()
    assert json.dumps({"GPU_ROUTER_ATTEMPT": "1"})[1:-1] in text
    proc = _run(run_py)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "flag x 0123456789ab" in proc.stdout.splitlines()
    assert "gpu-router: checkpoint 3 cannot reach kaggle yet" in proc.stdout


def test_resume_needs_its_sha() -> None:
    with pytest.raises(ValueError, match="go together"):
        remote.render_runner(
            attempt_key="gpu-0123456789ab-1", bundle=b"x", bundle_sha256="0", env={}, resume=b"y"
        )


def test_runner_hands_the_storage_token_over_in_a_file(tmp_path: Path, paths: Paths) -> None:
    """Review finding (D44): the storage token rode in bootstrap's exec environment, which
    the job can read in /proc/<ppid>/environ. run.py now writes it to a 0600 file that
    bootstrap deletes after reading; other secrets still reach the job's environment."""
    script = (
        "import os\n"
        "print('token env', 'GPU_STORAGE_TOKEN' in os.environ,"
        " 'GPU_STORAGE_TOKEN_FILE' in os.environ)\n"
        "print('user secret', os.environ.get('WANDB_API_KEY'))\n"
    )
    archive = make_bundle(tmp_path / "proj", paths, script=script)
    inputs = tmp_path / "input"
    mount = inputs / "datasets" / "me" / "gpu-router-secrets"
    mount.mkdir(parents=True)
    values = {"GPU_STORAGE_TOKEN": "hf_" + "q" * 34, "WANDB_API_KEY": "wk"}
    (mount / remote.SECRETS_FILE).write_text(json.dumps({"values": values}))
    run_py, _ = _render(
        tmp_path, archive, secrets_dataset="me/gpu-router-secrets", input_root=str(inputs)
    )
    proc = _run(run_py)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "token env False False" in proc.stdout
    assert "user secret wk" in proc.stdout
    assert not (tmp_path / "vm-tmp" / ".storage-token").exists()  # read once, deleted
