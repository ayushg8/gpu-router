"""Kaggle blob datasets (2026-10-04 field test): Kaggle's SaveKernel refuses a code file
over ~1 MB, so bundles, resume archives and `data:` datasets that do not fit inline travel
as private content-addressed datasets, uploaded once and reused. Also: the 400 is a
definitive InvalidJob, never "outcome unknown" + a provider cooldown + retries."""

from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from gpu_router.clock import FakeClock
from gpu_router.errors import InvalidJob, RateLimited
from gpu_router.models import Checkpoint
from gpu_router.paths import Paths
from gpu_router.providers.kaggle import adapter as adapter_mod
from gpu_router.providers.kaggle import parse, remote
from gpu_router.providers.kaggle.adapter import KaggleAdapter
from tests.contract.kaggle.sim import SAVEKERNEL_MAX_SOURCE, SimKaggle
from tests.unit.providers.kaggle.helpers import make_bundle
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


@pytest.fixture
def big_archive(tmp_path: Path) -> Path:
    """1.2 MB of incompressible bytes: base64 would push run.py past the 1 MB limit."""
    p = tmp_path / "big.tar.gz"
    p.write_bytes(os.urandom(1_200_000))
    return p


@pytest.fixture
def ticking(make_adapter: MakeAdapter, clock: FakeClock) -> MakeAdapter:
    """Adapters whose pauses move the fake clock (so a submit budget can run out)."""

    def build(**settings: object) -> KaggleAdapter:
        a = make_adapter(**settings)
        a._sleep = clock.advance
        return a

    return build


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_small_bundles_still_ride_inline(ticking: MakeAdapter, sim: SimKaggle, archive: Path):
    ticking().submit(make_job(), make_ctx(archive))
    kernel = sim.kernels[f"gpu-router-{JOB_ID}-1"]
    assert sim.dataset_creates == []
    assert "BUNDLE_INPUT = None" in kernel.run_py
    assert kernel.metadata["dataset_sources"] == []


def test_a_big_bundle_travels_as_a_private_dataset_once(
    ticking: MakeAdapter, sim: SimKaggle, big_archive: Path, paths: Paths
) -> None:
    sim.dataset_hidden_after = 1  # 403 right after the create, as seen live
    sim.dataset_ready_after = 2
    adapter = ticking()
    adapter.submit(make_job(), make_ctx(big_archive))
    sha = _sha(big_archive)
    ref = f"simuser/gpu-router-bundle-{sha[:16]}"
    assert sim.dataset_creates == [ref]
    ds = sim.datasets[ref]
    assert ds.public is False
    assert ds.files == {f"bundle-{sha[:16]}.bin": big_archive.read_bytes()}
    kernel = sim.kernels[f"gpu-router-{JOB_ID}-1"]
    assert kernel.metadata["dataset_sources"] == [ref]
    assert len(kernel.run_py.encode()) < remote.MAX_INLINE_SOURCE < SAVEKERNEL_MAX_SOURCE
    assert f"BUNDLE_INPUT = [{ref!r}, 'bundle-{sha[:16]}.bin']" in kernel.run_py
    leftovers = sorted(p.name for p in (paths.provider_dir("kaggle") / "blobs").iterdir())
    assert leftovers == [f"gpu-router-bundle-{sha[:16]}.json"]  # upload folder removed

    # the retry (and any later job with the same bundle) reuses it
    adapter.submit(make_job(), make_ctx(big_archive, n=2))
    assert sim.dataset_creates == [ref]
    assert sim.kernels[f"gpu-router-{JOB_ID}-2"].metadata["dataset_sources"] == [ref]
    # a fresh daemon without the record still finds it on Kaggle
    (paths.provider_dir("kaggle") / "blobs" / f"gpu-router-bundle-{sha[:16]}.json").unlink()
    ticking().submit(make_job(), make_ctx(big_archive, n=3))
    assert sim.dataset_creates == [ref]
    # deleted on Kaggle meanwhile: uploaded again
    del sim.datasets[ref]
    ticking().submit(make_job(), make_ctx(big_archive, n=4))
    assert sim.dataset_creates == [ref, ref]


def test_a_savekernel_400_is_definitive(
    make_adapter: MakeAdapter,
    sim: SimKaggle,
    big_archive: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The jobs of 2026-10-04: `400 Client Error: Bad Request ... SaveKernel` was "outcome
    unknown", so the provider cooled down and a pinned job retried 6 times."""
    monkeypatch.setattr(remote, "MAX_INLINE_SOURCE", 5_000_000)  # the old inline-only path
    with pytest.raises(InvalidJob, match="400 Bad Request"):
        make_adapter().submit(make_job(), make_ctx(big_archive))
    assert sim.pushes == []


def test_blob_datasets_off_refuses_big_bundles_up_front(
    make_adapter: MakeAdapter, sim: SimKaggle, big_archive: Path
) -> None:
    with pytest.raises(InvalidJob, match="kaggle takes up to"):
        make_adapter(blob_datasets=False).submit(make_job(), make_ctx(big_archive))
    assert sim.calls == []  # nothing sent
    assert make_adapter(blob_datasets=False).capabilities.stage_data is False


def test_still_processing_waits_then_reuses_the_upload(
    ticking: MakeAdapter, sim: SimKaggle, big_archive: Path
) -> None:
    sim.dataset_ready_after = 10_000  # never ready within one submit
    adapter = ticking()
    with pytest.raises(RateLimited, match="still processing") as caught:
        adapter.submit(make_job(), make_ctx(big_archive))
    # definitive (nothing pushed, no ambiguous lookup) and a short pause, not an outage
    assert caught.value.retry_after == adapter_mod.BLOB_RETRY_S
    assert sim.pushes == []  # nothing pushed: safe to submit again
    ref = sim.dataset_creates[0]
    sim.datasets[ref].pending = 0
    adapter.submit(make_job(), make_ctx(big_archive))
    assert sim.dataset_creates == [ref]  # not uploaded twice
    assert len(sim.pushes) == 1


def test_stage_data_uploads_a_dir_as_one_tar_and_a_file_as_is(
    ticking: MakeAdapter, sim: SimKaggle, tmp_path: Path
) -> None:
    from gpu_router.checkpoint.data import digest

    root = tmp_path / "crops"
    (root / "a").mkdir(parents=True)
    (root / "a" / "x.png").write_bytes(b"png-x")
    (root / "y.txt").write_text("why\n")
    dig = digest(root)
    adapter = ticking()
    staged = adapter.stage_data(root, dig.sha256, dig.files)
    slug = f"gpu-router-data-{dig.sha256[:16]}"
    assert staged.uploaded is True
    assert staged.uri == f"kaggle://simuser/{slug}/data-{dig.sha256[:16]}.tar.bin"
    assert staged.where == f"private kaggle dataset simuser/{slug}"
    blob = sim.datasets[f"simuser/{slug}"].files[f"data-{dig.sha256[:16]}.tar.bin"]
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:") as tar:
        assert sorted(tar.getnames()) == ["a/x.png", "y.txt"]
        member = tar.extractfile("a/x.png")
        assert member is not None
        assert member.read() == b"png-x"
    again = adapter.stage_data(root, dig.sha256, dig.files)
    assert again.uploaded is False
    assert again.uri == staged.uri
    assert len(sim.dataset_creates) == 1

    weights = tmp_path / "w.safetensors"
    weights.write_bytes(b"\x00weights")
    wdig = digest(weights)
    staged_file = adapter.stage_data(weights, wdig.sha256, wdig.files)
    assert staged_file.uri.endswith(f"/data-{wdig.sha256[:16]}.bin")
    wref = f"simuser/gpu-router-data-{wdig.sha256[:16]}"
    assert sim.datasets[wref].files == {f"data-{wdig.sha256[:16]}.bin": b"\x00weights"}

    # a submit attaches the staged datasets GPU_DATA names
    gpu_data = json.dumps(
        [
            {"mount": "crops", "uri": staged.uri, "sha256": dig.sha256},
            {"mount": "w", "uri": staged_file.uri, "sha256": wdig.sha256},
        ]
    )
    small = tmp_path / "b.tar.gz"
    small.write_bytes(b"tiny")
    env = {"GPU_ROUTER_JOB_ID": JOB_ID, "GPU_ROUTER_ATTEMPT": "1", "GPU_DATA": gpu_data}
    adapter.submit(make_job(), make_ctx(small, env=env))
    kernel = sim.kernels[f"gpu-router-{JOB_ID}-1"]
    assert kernel.metadata["dataset_sources"] == [f"simuser/{slug}", wref]


def test_only_gpu_router_datasets_are_attached(
    make_adapter: MakeAdapter, sim: SimKaggle, archive: Path
) -> None:
    gpu_data = json.dumps([{"mount": "x", "uri": "kaggle://someone/their-data/x.bin"}])
    env = {"GPU_ROUTER_JOB_ID": JOB_ID, "GPU_ROUTER_ATTEMPT": "1", "GPU_DATA": gpu_data}
    with pytest.raises(InvalidJob, match="not a gpu-router kaggle dataset"):
        make_adapter().submit(make_job(), make_ctx(archive, env=env))


def test_a_big_resume_archive_rides_in_a_ckpt_dataset(
    ticking: MakeAdapter, sim: SimKaggle, archive: Path, tmp_path: Path
) -> None:
    ckpt_file = tmp_path / "ckpt-0003.tar.gz"
    ckpt_file.write_bytes(os.urandom(900_000))
    ckpt = Checkpoint(
        id=f"{JOB_ID}.c3",
        job_id=JOB_ID,
        attempt_id=f"{JOB_ID}.1",
        seq=3,
        uri=ckpt_file.as_uri(),
        created_at=0,
        recorded_at=0,
    )
    ticking().submit(make_job(), make_ctx(archive, n=2, resume_from=ckpt))
    sha = _sha(ckpt_file)
    ref = f"simuser/gpu-router-ckpt-{sha[:16]}"
    kernel = sim.kernels[f"gpu-router-{JOB_ID}-2"]
    assert ref in kernel.metadata["dataset_sources"]
    assert f"RESUME_INPUT = [{ref!r}, 'ckpt-{sha[:16]}.bin']" in kernel.run_py
    assert f"RESUME_SHA256 = {sha!r}" in kernel.run_py
    assert "CKPT_SEQ_START = 4" in kernel.run_py


def test_the_sweep_deletes_only_stale_blobs(
    ticking: MakeAdapter,
    sim: SimKaggle,
    big_archive: Path,
    tmp_path: Path,
    clock: FakeClock,
) -> None:
    from gpu_router.checkpoint.data import digest

    adapter = ticking()
    adapter.submit(make_job(), make_ctx(big_archive))
    clock.advance(4 * 86400)  # bundles are kept 3 days after their last use
    data = tmp_path / "d.bin"
    data.write_bytes(b"d")
    dig = digest(data)
    adapter.stage_data(data, dig.sha256, dig.files)
    deleted = adapter.sweep_blobs()
    assert deleted == [f"simuser/gpu-router-bundle-{_sha(big_archive)[:16]}"]
    assert sim.dataset_deletes == deleted
    clock.advance(31 * 86400)  # data is kept data_keep_days (30)
    assert adapter.sweep_blobs() == [f"simuser/gpu-router-data-{dig.sha256[:16]}"]


def test_the_background_sweep_runs_at_most_every_6h(
    ticking: MakeAdapter, big_archive: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(adapter_mod, "BLOB_SWEEP", True)
    adapter = ticking()
    calls: list[int] = []
    monkeypatch.setattr(adapter, "sweep_blobs", lambda: calls.append(1) or [])
    adapter.submit(make_job(), make_ctx(big_archive))
    assert adapter._sweep_thread is not None
    adapter._sweep_thread.join(5)
    adapter.submit(make_job(), make_ctx(big_archive, n=2))
    assert calls == [1]


# ------------------------------------------------------------------ run.py on this Mac


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


def test_run_py_reads_the_bundle_and_data_from_attached_datasets(
    tmp_path: Path, paths: Paths
) -> None:
    script = (
        "import os, pathlib\n"
        "d = pathlib.Path(os.environ['GPU_DATA_DIR'])\n"
        "print('crop', (d / 'crops' / 'a' / 'x.txt').read_text().strip())\n"
        "print('weights', (d / 'w').read_bytes())\n"
        "out = pathlib.Path(os.environ['GPU_OUTPUT_DIR'])\n"
        "out.mkdir(parents=True, exist_ok=True)\n"
        "(out / 'ok.txt').write_text('ok')\n"
    )
    archive = make_bundle(tmp_path / "proj", paths, script=script)
    sha = _sha(archive)
    inputs = tmp_path / "input" / "datasets" / "me"
    bdir = inputs / f"gpu-router-bundle-{sha[:16]}"
    bdir.mkdir(parents=True)
    (bdir / f"bundle-{sha[:16]}.bin").write_bytes(archive.read_bytes())
    # a directory dataset: one tar; a file dataset: the file
    tar_dir = inputs / f"gpu-router-data-{'1' * 16}"
    tar_dir.mkdir()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        body = b"crop-one\n"
        info = tarfile.TarInfo("a/x.txt")
        info.size = len(body)
        tar.addfile(info, io.BytesIO(body))
    (tar_dir / f"data-{'1' * 16}.tar.bin").write_bytes(buf.getvalue())
    file_dir = inputs / f"gpu-router-data-{'2' * 16}"
    file_dir.mkdir()
    (file_dir / f"data-{'2' * 16}.bin").write_bytes(b"W8")
    gpu_data = [
        {
            "mount": "crops",
            "uri": f"kaggle://me/gpu-router-data-{'1' * 16}/data-{'1' * 16}.tar.bin",
        },
        {"mount": "w", "uri": f"kaggle://me/gpu-router-data-{'2' * 16}/data-{'2' * 16}.bin"},
    ]
    out = tmp_path / "working" / "outputs"
    text = remote.render_runner(
        attempt_key=f"gpu-{JOB_ID}-1",
        bundle=None,
        bundle_input=(f"me/gpu-router-bundle-{sha[:16]}", f"bundle-{sha[:16]}.bin"),
        bundle_sha256=sha,
        env={
            "GPU_ROUTER_JOB_ID": JOB_ID,
            "GPU_ROUTER_ATTEMPT": "1",
            "GPU_DATA": json.dumps(gpu_data),
        },
        input_root=str(tmp_path / "input"),
        workdir=str(tmp_path / "vm-tmp"),
        output_dir=str(out),
        sync_dir=str(tmp_path / "working" / ".gpu-router" / "checkpoints"),
    )
    run_py = tmp_path / "run.py"
    run_py.write_text(text)
    proc = _run(run_py)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    lines = proc.stdout.splitlines()
    assert "crop crop-one" in lines
    assert "weights b'W8'" in lines
    assert parse.last_exit_code(lines) == 0
    assert (out / "ok.txt").read_text() == "ok"


def test_run_py_refuses_a_corrupted_dataset_bundle(tmp_path: Path, paths: Paths) -> None:
    archive = make_bundle(tmp_path / "proj", paths)
    sha = _sha(archive)
    bdir = tmp_path / "input" / "datasets" / "me" / "gpu-router-bundle-x"
    bdir.mkdir(parents=True)
    (bdir / "bundle-x.bin").write_bytes(archive.read_bytes() + b"tampered")
    text = remote.render_runner(
        attempt_key=f"gpu-{JOB_ID}-1",
        bundle=None,
        bundle_input=("me/gpu-router-bundle-x", "bundle-x.bin"),
        bundle_sha256=sha,
        env={},
        input_root=str(tmp_path / "input"),
        workdir=str(tmp_path / "vm-tmp"),
        output_dir=str(tmp_path / "out"),
        sync_dir=str(tmp_path / "sync"),
    )
    run_py = tmp_path / "run.py"
    run_py.write_text(text)
    proc = _run(run_py)
    assert proc.returncode != 0
    assert "sha256 mismatch" in proc.stderr


def test_blobs_off_starts_a_big_resume_fresh_instead_of_excluding_kaggle(
    make_adapter: MakeAdapter, sim: SimKaggle, archive: Path, tmp_path: Path
) -> None:
    ckpt_file = tmp_path / "ckpt-0002.tar.gz"
    ckpt_file.write_bytes(os.urandom(900_000))
    ckpt = Checkpoint(
        id=f"{JOB_ID}.c2",
        job_id=JOB_ID,
        attempt_id=f"{JOB_ID}.1",
        seq=2,
        uri=ckpt_file.as_uri(),
        created_at=0,
        recorded_at=0,
    )
    make_adapter(blob_datasets=False).submit(make_job(), make_ctx(archive, n=2, resume_from=ckpt))
    kernel = sim.kernels[f"gpu-router-{JOB_ID}-2"]
    assert "RESUME_SHA256 = None" in kernel.run_py
    assert "blob_datasets is off; starting fresh" in kernel.run_py
    assert sim.dataset_creates == []


def test_the_sweep_counts_an_upload_in_progress_as_use(
    ticking: MakeAdapter, sim: SimKaggle, paths: Paths, clock: FakeClock
) -> None:
    adapter = ticking()
    blobs = paths.provider_dir("kaggle") / "blobs"
    blobs.mkdir(parents=True, exist_ok=True)
    slug = "gpu-router-data-" + "a" * 16
    (blobs / f"{slug}.json").write_text(
        json.dumps({"ref": f"simuser/{slug}", "kind": "data", "uploading_at": clock.now()})
    )
    clock.advance(86400)
    assert adapter.sweep_blobs() == []  # 1 day into a 30-day keep
    assert (blobs / f"{slug}.json").exists()


def test_a_packing_error_is_a_taxonomy_error(
    ticking: MakeAdapter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A3: an OSError while packing (a file vanished, the disk filled) must not escape as a
    contract violation."""
    from gpu_router.checkpoint.data import digest
    from gpu_router.errors import Unavailable

    root = tmp_path / "d"
    root.mkdir()
    (root / "x.txt").write_text("x")
    dig = digest(root)
    (root / "x.txt").unlink()  # vanished between digest and tar
    with pytest.raises(Unavailable, match="could not pack d for kaggle"):
        ticking().stage_data(root, dig.sha256, dig.files)
