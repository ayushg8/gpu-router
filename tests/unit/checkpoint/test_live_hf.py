"""Live Hugging Face Storage Bucket check (opt-in; never runs by default).

    HF_TOKEN=hf_... GPU_ROUTER_REAL_PROVIDERS=hf \\
        uv run pytest tests/unit/checkpoint/test_live_hf.py -q

Uses a private bucket `<you>/gpu-router-selftest` (created if missing), writes under a
unique prefix and deletes it at the end. The token comes from $HF_TOKEN for this one test
run (tests never read the real Keychain, invariant 20). It exercises exactly the calls the
runner and the daemon make: create_bucket, batch_bucket_files (bytes + files + delete),
get_bucket_paths_info, download_bucket_files, list_bucket_tree.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest

from gpu_router.checkpoint.storage import open_storage
from gpu_router.runner import storage as rs

pytestmark = pytest.mark.real_provider("hf")


def test_bucket_roundtrip(tmp_path: Path) -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        pytest.skip("set HF_TOKEN for the live bucket test")
    from huggingface_hub import HfApi

    user = HfApi(token=token).whoami()["name"]
    store = open_storage(f"hf://buckets/{user}/gpu-router-selftest", token=token)
    store.ensure()
    job = f"selftest-{uuid.uuid4().hex[:8]}"
    try:
        src = tmp_path / "ckpt"
        (src / "sub").mkdir(parents=True)
        (src / "model.pt").write_bytes(os.urandom(4096))
        (src / "sub" / "opt.pt").write_bytes(b"x" * 10)
        files = [("model.pt", 4096, "a"), ("sub/opt.pt", 10, "b")]
        latest = rs.publish_checkpoint(store.raw, job, 1, src, files, attempt=1, step=5)
        assert store.read_latest(job) == latest
        out = tmp_path / "restore"
        assert store.restore_checkpoint(rs.ckpt_key(job, 1), out) == 2
        assert (out / "model.pt").read_bytes() == (src / "model.pt").read_bytes()
        store.write_json(rs.heartbeat_key(job, 1), {"ok": True})
        assert store.read_json(rs.heartbeat_key(job, 1)) == {"ok": True}
        first = store.stat(rs.heartbeat_key(job, 1))
        store.write_json(rs.heartbeat_key(job, 1), {"ok": False})
        assert store.stat(rs.heartbeat_key(job, 1)) != first
    finally:
        store.delete_prefix(f"jobs/{job}")
    assert store.list_files(f"jobs/{job}") == []
