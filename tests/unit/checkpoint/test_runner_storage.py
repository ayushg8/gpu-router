"""gpu_router/runner/storage.py (the shared layout) and its typed daemon facade, on both
backends: a local directory and an HF Storage Bucket over a fake HfApi."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from gpu_router.checkpoint.storage import Storage, StorageError, open_storage
from gpu_router.runner import storage as rs
from tests.unit.checkpoint.fake_hfapi import FakeHfApi, HTTPError

BUCKET = "tester/gpu-router"


@pytest.fixture(params=["local", "hf"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> Storage:
    if request.param == "local":
        return open_storage((tmp_path / "root").as_uri())
    api = FakeHfApi()
    api.create_bucket(BUCKET, private=True)
    return open_storage(f"hf://buckets/{BUCKET}", api=api)


def _tree(tmp_path: Path, files: dict[str, str]) -> Path:
    root = tmp_path / "src"
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    return root


def test_bytes_json_stat_roundtrip(store: Storage) -> None:
    assert store.read_bytes("jobs/j/x.json") is None
    assert store.stat("jobs/j/x.json") is None
    store.write_json("jobs/j/x.json", {"a": 1})
    assert store.read_json("jobs/j/x.json") == {"a": 1}
    size, token = store.stat("jobs/j/x.json") or (0, "")
    assert size == len(b'{"a":1}')
    store.write_json("jobs/j/x.json", {"a": 22})
    assert store.stat("jobs/j/x.json") != (size, token)  # change token moves
    store.write_bytes("jobs/j/bad.json", b"not json")
    assert store.read_json("jobs/j/bad.json") is None


def test_upload_download_list_delete(store: Storage, tmp_path: Path) -> None:
    src = _tree(tmp_path, {"model.pt": "w1", "sub/opt.pt": "o1"})
    store.upload_dir(src, "jobs/j/ckpt-0001", ["model.pt", "sub/opt.pt"])
    assert [k for k, _ in store.list_files("jobs/j")] == [
        "jobs/j/ckpt-0001/model.pt",
        "jobs/j/ckpt-0001/sub/opt.pt",
    ]
    # re-publishing the same key drops files that are not in the new upload
    (src / "sub" / "opt.pt").unlink()
    store.upload_dir(src, "jobs/j/ckpt-0001", ["model.pt"])
    assert [k for k, _ in store.list_files("jobs/j/ckpt-0001")] == ["jobs/j/ckpt-0001/model.pt"]
    out = tmp_path / "out"
    assert store.download_dir("jobs/j/ckpt-0001", out) == 1
    assert (out / "model.pt").read_text() == "w1"
    store.delete_prefix("jobs/j/ckpt-0001")
    assert store.list_files("jobs/j") == []
    with pytest.raises(StorageError) as err:
        store.download_dir("jobs/j/ckpt-0001", out)
    assert err.value.missing


def test_publish_checkpoint_writes_files_manifest_then_pointer(
    store: Storage, tmp_path: Path
) -> None:
    src = _tree(tmp_path, {"last.pt": "abc"})
    latest = rs.publish_checkpoint(
        store.raw, "job1", 3, src, [("last.pt", 3, "sha")], attempt=2, step=120
    )
    assert latest["seq"] == 3
    assert latest["uri"] == store.uri("jobs/job1/ckpt-0003")
    assert latest["attempt"] == 2
    assert latest["step"] == 120
    assert latest["size"] == 3
    assert store.read_latest("job1") == latest
    manifest = store.read_json("jobs/job1/ckpt-0003/.gpu-ckpt.json")
    assert manifest is not None
    assert manifest["files"][0]["path"] == "last.pt"
    dest = tmp_path / "restore"
    assert store.restore_checkpoint("jobs/job1/ckpt-0003", dest) == 1  # manifest skipped
    assert sorted(p.name for p in dest.iterdir()) == ["last.pt"]
    assert store.key_of(latest["uri"]) == "jobs/job1/ckpt-0003"


def test_claim_and_owner(store: Storage) -> None:
    assert rs.read_owner(store.raw, "job1") is None
    store.claim("job1", 4)
    assert rs.read_owner(store.raw, "job1") == 4


def test_datasets_manifest_marks_completion(store: Storage, tmp_path: Path) -> None:
    src = _tree(tmp_path, {"a.csv": "1,2", "b/c.csv": "3"})
    assert not store.dataset_complete("f00d")
    uri = store.upload_dataset("f00d", src, [("a.csv", 3), ("b/c.csv", 1)])
    assert uri == store.uri("datasets/f00d")
    assert store.dataset_complete("f00d")
    single = tmp_path / "one.bin"
    single.write_bytes(b"xyz")
    store.upload_dataset("beef", single, [("one.bin", 3)])
    assert [k for k, _ in store.list_files("datasets/beef")] == [
        "datasets/beef/.gpu-data.json",
        "datasets/beef/one.bin",
    ]


def test_local_store_moves_a_staging_dir_into_place(tmp_path: Path) -> None:
    store = open_storage((tmp_path / "root").as_uri())
    stage = _tree(tmp_path, {"w.pt": "1"})
    store.upload_dir(stage, "jobs/j/ckpt-0002", ["w.pt"], move=True)
    assert not stage.exists()  # renamed, not copied
    assert (tmp_path / "root/jobs/j/ckpt-0002/w.pt").read_text() == "1"


def test_hf_error_mapping(tmp_path: Path) -> None:
    api = FakeHfApi()
    api.create_bucket(BUCKET, private=True)
    store = open_storage(f"hf://buckets/{BUCKET}", api=api)
    api.fail["get_bucket_paths_info"] = [HTTPError(401), HTTPError(503), HTTPError(404)]
    with pytest.raises(StorageError) as err:
        store.read_bytes("x")
    assert not err.value.retryable
    assert "HTTP 401" in err.value.message
    with pytest.raises(StorageError) as err:
        store.read_bytes("x")
    assert err.value.retryable
    with pytest.raises(StorageError) as err:
        store.read_bytes("x")
    assert err.value.missing
    api.fail["batch_bucket_files"] = [ConnectionError("dns")]
    with pytest.raises(StorageError) as err:
        store.write_bytes("x", b"1")
    assert err.value.retryable
    assert "ConnectionError" in err.value.message


def test_hf_ensure_creates_a_private_bucket() -> None:
    api = FakeHfApi()
    store = open_storage("hf://buckets/tester/new-bucket", api=api)
    store.ensure()
    store.ensure()  # idempotent (exist_ok)
    assert api.private["tester/new-bucket"] is True


def test_uri_helpers(tmp_path: Path) -> None:
    assert rs.split_hf_uri("hf://buckets/a/b/jobs/j/ckpt-0001") == (
        "hf://buckets/a/b",
        "jobs/j/ckpt-0001",
    )
    assert rs.split_hf_uri("hf://buckets/a/b") is None
    assert rs.split_hf_uri("file:///x") is None
    local = open_storage(str(tmp_path / "r"))
    assert local.kind == "local"
    assert local.key_of((tmp_path / "r" / "jobs" / "x").as_uri()) == "jobs/x"
    assert local.key_of((tmp_path / "elsewhere").as_uri()) is None
    with pytest.raises(StorageError):
        open_storage("s3://nope")
    with pytest.raises(StorageError):
        local.read_bytes("../escape")


def test_layout_keys_are_stable() -> None:
    """The runner and the daemon must agree on these names (a bundle of today talks to a
    daemon of tomorrow)."""
    assert rs.ckpt_key("j", 7) == "jobs/j/ckpt-0007"
    assert rs.latest_key("j") == "jobs/j/latest.json"
    assert rs.owner_key("j") == "jobs/j/owner.json"
    assert rs.heartbeat_key("j", 2) == "jobs/j/attempts/2/heartbeat.json"
    assert rs.log_tail_key("j", 2) == "jobs/j/attempts/2/log-tail.json"
    assert rs.control_key("j", 2) == "jobs/j/attempts/2/control.json"
    assert rs.ack_key("j", 2) == "jobs/j/attempts/2/control-ack.json"
    assert rs.dataset_key("ab") == "datasets/ab"
    assert json.loads(rs.dumps({"b": 1, "a": 2})) == {"a": 2, "b": 1}


def test_storage_module_parses_as_python38() -> None:
    source = Path(rs.__file__).read_text()
    tree = ast.parse(source, feature_version=(3, 8))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(not a.name.startswith("gpu_router") for a in node.names)
        if isinstance(node, ast.ImportFrom):
            assert node.module is None or not node.module.startswith("gpu_router")
        if isinstance(node, ast.FunctionDef):
            assert node.returns is None
            assert all(a.annotation is None for a in node.args.args)


def test_hf_download_refuses_keys_that_escape_the_destination(tmp_path: Path) -> None:
    api = FakeHfApi()
    api.create_bucket(BUCKET, private=True)
    api.buckets[BUCKET]["jobs/j/ckpt-0001/../../evil"] = b"x"
    store = open_storage(f"hf://buckets/{BUCKET}", api=api)
    with pytest.raises(StorageError, match="unsafe path"):
        store.download_dir("jobs/j/ckpt-0001", tmp_path / "out")
    assert not (tmp_path / "evil").exists()
