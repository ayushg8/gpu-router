"""Phase 4/5 review regressions in the runner (bootstrap.py + storage.py), D44: resume
falls back to storage's latest / an older intact checkpoint instead of starting fresh,
torn checkpoints are never restored, a seq is never reused after an ambiguous publish,
the handoff answer is retried, disk use during restore/sync, Python < 3.10, and the
storage token file channel."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any

import pytest

from gpu_router.paths import Paths
from gpu_router.runner import bootstrap
from gpu_router.runner import storage as rs
from tests.unit.checkpoint.test_bootstrap_storage import (
    JOB,
    TRAIN,
    _bundle,
    _env,
    _events,
    _run,
)


def _publish(store: Any, seq: int, step: int, tmp: Path, attempt: int = 1) -> str:
    src = tmp / f"stage-{seq}"
    src.mkdir(parents=True, exist_ok=True)
    data = json.dumps({"step": step}).encode()
    (src / "state.json").write_bytes(data)
    latest = rs.publish_checkpoint(
        store,
        JOB,
        seq,
        src,
        [("state.json", len(data), hashlib.sha256(data).hexdigest())],
        attempt=attempt,
        step=step,
    )
    return str(latest["uri"])


def test_a_pruned_resume_checkpoint_restores_storages_latest(tmp_path: Path, paths: Paths) -> None:
    """Finding: GPU_RESUME_URI naming a checkpoint storage had pruned (the DB lagged
    storage) started the job fresh, and its pruning then deleted the real progress."""
    root = tmp_path / "storage"
    store = rs.LocalStore(root)
    for seq in (2, 3, 4):
        _publish(store, seq, seq, tmp_path)
    gone = store.uri(rs.ckpt_key(JOB, 1))
    archive = _bundle(tmp_path, paths, TRAIN)
    proc = _run(archive, tmp_path / "w", _env(root, 2, GPU_RESUME_URI=gone, GPU_CKPT_KEEP="3"))
    assert proc.returncode == 0, proc.stdout
    assert "start=4" in proc.stdout  # latest.json's checkpoint, not a fresh start
    assert "resuming from" in proc.stdout
    ends = [e for e in _events(proc.stdout) if e.t == "ckpt_end"]
    assert [e.seq for e in ends] == [5]
    assert rs.list_checkpoints(store, JOB) == [3, 4, 5]  # kept window, nothing lost


def test_a_torn_checkpoint_falls_back_to_the_one_before(tmp_path: Path, paths: Paths) -> None:
    """Finding: restores never checked the manifest, so a torn checkpoint (files from two
    saves, or missing ones) was restored silently."""
    root = tmp_path / "storage"
    store = rs.LocalStore(root)
    _publish(store, 3, 3, tmp_path)
    uri4 = _publish(store, 4, 4, tmp_path)
    (root / "jobs" / JOB / "ckpt-0004" / "state.json").write_text('{"step": 44444}')
    archive = _bundle(tmp_path, paths, TRAIN)
    proc = _run(archive, tmp_path / "w", _env(root, 2, GPU_RESUME_URI=uri4))
    assert proc.returncode == 0, proc.stdout
    assert "is incomplete" in proc.stdout
    assert "start=3" in proc.stdout


def test_missing_checkpoint_and_unreadable_pointer_is_exit_90(tmp_path: Path, paths: Paths) -> None:
    """Missing checkpoint while latest.json cannot be read: exit 90 (the engine tries
    again) rather than discarding progress it cannot see."""
    if os.geteuid() == 0:
        pytest.skip("root reads files regardless of their mode")
    root = tmp_path / "storage"
    store = rs.LocalStore(root)
    _publish(store, 4, 4, tmp_path)
    latest = root / "jobs" / JOB / "latest.json"
    latest.chmod(0)
    try:
        archive = _bundle(tmp_path, paths, TRAIN)
        gone = store.uri(rs.ckpt_key(JOB, 2))
        proc = _run(archive, tmp_path / "w", _env(root, 2, GPU_RESUME_URI=gone))
    finally:
        latest.chmod(stat.S_IRUSR | stat.S_IWUSR)
    assert proc.returncode == bootstrap.INSTALL_FAILED_EXIT, proc.stdout
    assert "start=" not in proc.stdout


class _Tee:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.events: list[tuple[str, dict[str, Any]]] = []

    def say(self, text: str) -> None:
        self.lines.append(text)

    def event(self, t: str, **fields: Any) -> None:
        self.events.append((t, fields))

    def snapshot(self) -> tuple[list[str], int]:
        return list(self.lines), len(self.lines)


class _LatestTimesOut(rs.LocalStore):
    """latest.json is written server-side, then the client sees an error (once)."""

    failures = 1

    def write_bytes(self, key: str, data: bytes) -> None:
        super().write_bytes(key, data)
        if key.endswith("latest.json") and self.failures:
            self.failures -= 1
            raise rs.StorageError("upload latest.json: read timed out")


def _syncer(tmp_path: Path, store: Any, attempt: int = 1) -> tuple[Any, Path, _Tee]:
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir(parents=True)
    tee = _Tee()
    syncer = bootstrap.CheckpointSyncer(
        tee, ckpt, None, 1, 60, store=store, smod=rs, job_id=JOB, attempt=attempt,
        keep=3, stage_root=tmp_path / "stage",
    )  # fmt: skip
    return syncer, ckpt, tee


def _save(ckpt: Path, text: str) -> None:
    """Write a checkpoint file that looks settled (10 s old) to the syncer."""
    path = ckpt / "state.json"
    path.write_text(text)
    st = path.stat()
    os.utime(path, (st.st_atime - 10, st.st_mtime - 10))


def test_a_publish_that_landed_despite_an_error_is_not_published_again(tmp_path: Path) -> None:
    """Finding: after an ambiguous publish failure the retry reused the same seq and
    rewrote ckpt-N in place while latest.json already pointed at it."""
    store = _LatestTimesOut(tmp_path / "root")
    syncer, ckpt, tee = _syncer(tmp_path, store)
    _save(ckpt, '{"step": 1}')
    assert syncer.sync() == 1  # the pointer landed: counted as published
    assert [t for t, _ in tee.events] == ["ckpt_begin", "ckpt_end"]
    _save(ckpt, '{"step": 2}')
    assert syncer.sync() == 2  # the next save gets a new seq
    assert rs.read_latest(store, JOB)["seq"] == 2


def test_a_newer_pointer_from_elsewhere_bumps_the_seq(tmp_path: Path) -> None:
    class Fails(rs.LocalStore):
        def write_bytes(self, key: str, data: bytes) -> None:
            if key.endswith("latest.json"):
                raise rs.StorageError("upload failed")
            super().write_bytes(key, data)

    store = Fails(tmp_path / "root")
    rs.LocalStore.write_bytes(
        store, rs.latest_key(JOB), rs.dumps({"seq": 7, "uri": "file:///x", "attempt": 9})
    )
    syncer, ckpt, _tee = _syncer(tmp_path, store)
    _save(ckpt, '{"step": 1}')
    assert syncer.sync() is None
    assert syncer.seq == 7  # the retry uses 8, never a seq the pointer already names


class _FlakyAck(rs.LocalStore):
    def __init__(self, root: Path, fail: int) -> None:
        super().__init__(root)
        self.fail = fail

    def write_bytes(self, key: str, data: bytes) -> None:
        if key.endswith("control-ack.json") and self.fail:
            self.fail -= 1
            raise rs.StorageError("hugging face returned HTTP 503")
        super().write_bytes(key, data)


def _channel(tmp_path: Path, store: Any) -> tuple[Any, Any, Path]:
    syncer, ckpt, tee = _syncer(tmp_path, store)
    channel = bootstrap.StatusChannel(tee, store, rs, JOB, 1, syncer, ckpt, lambda s: s, 0, 0.1)
    return channel, syncer, ckpt


def _request(store: Any, rid: str) -> None:
    rs.LocalStore.write_bytes(
        store,
        rs.control_key(JOB, 1),
        rs.dumps({"id": rid, "action": "handoff", "wait_s": 0, "reason": "cap"}),
    )


def test_a_failed_handoff_answer_is_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Finding: a failed control-ack write was never re-sent (the unchanged control.json
    matched the remembered token) while the runner stayed frozen."""
    monkeypatch.setattr(bootstrap.time, "sleep", lambda s: None)
    store = _FlakyAck(tmp_path / "root", fail=2)
    channel, syncer, _ckpt = _channel(tmp_path, store)
    _request(store, "r1")
    channel.poll_control()
    channel._handler.join(10)
    assert rs.read_json(store, rs.ack_key(JOB, 1))["id"] == "r1"  # retried with backoff
    assert syncer.frozen == "handed off"

    # every try fails: not frozen, and the same request is answered at the next poll
    store2 = _FlakyAck(tmp_path / "root2", fail=bootstrap.ACK_TRIES)
    channel2, syncer2, _ = _channel(tmp_path / "b", store2)
    _request(store2, "r2")
    channel2.poll_control()
    channel2._handler.join(10)
    assert rs.read_json(store2, rs.ack_key(JOB, 1)) is None
    assert syncer2.frozen is None
    channel2.poll_control()
    channel2._handler.join(10)
    assert rs.read_json(store2, rs.ack_key(JOB, 1))["id"] == "r2"
    assert syncer2.frozen == "handed off"


def test_a_failed_handoff_upload_is_retried_before_answering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Finding: one failed upload during a handoff made the runner answer with the older
    checkpoint right away, losing the save the job made on request."""
    monkeypatch.setattr(bootstrap.time, "sleep", lambda s: None)

    class FailsOnce(rs.LocalStore):
        failures = 1

        def upload_dir(self, src: Any, key: str, files: Any, move: bool = False) -> None:
            if self.failures:
                self.failures -= 1
                raise rs.StorageError("upload failed")
            super().upload_dir(src, key, files, move=move)

    store = FailsOnce(tmp_path / "root")
    channel, syncer, ckpt = _channel(tmp_path, store)
    _save(ckpt, '{"step": 5}')
    syncer.last_sync = 0  # due
    channel.handle_request({"id": "r3", "action": "handoff", "wait_s": 0})
    ack = rs.read_json(store, rs.ack_key(JOB, 1))
    assert ack["new"] is True
    assert ack["seq"] == 1


def test_a_downloaded_checkpoint_is_moved_not_copied(tmp_path: Path) -> None:
    work = tmp_path / "w"
    dl = work / bootstrap.RESUME_DL
    dl.mkdir(parents=True)
    (dl / "big.bin").write_bytes(b"x" * 1024)
    args = type("A", (), {"resume": None, "resume_uri": "hf://buckets/a/b/jobs/j/ckpt-0001"})()
    monkey_src = dl

    def fake_source(*_a: Any, **_k: Any) -> Path:
        return monkey_src

    orig = bootstrap._resume_source
    bootstrap._resume_source = fake_source  # type: ignore[assignment]
    try:
        env: dict[str, str] = {}
        ckpt = work / "checkpoints"
        ckpt.mkdir()
        bootstrap._restore(args, env, work, ckpt, _Tee())
    finally:
        bootstrap._resume_source = orig  # type: ignore[assignment]
    assert not dl.exists()  # renamed into the resume dir: one copy, not two
    assert (Path(env["GPU_RESUME_DIR"]) / "big.bin").stat().st_size == 1024
    assert (ckpt / "big.bin").is_file()  # the checkpoint dir still gets real copies


def test_a_restore_that_hits_the_disk_is_exit_90(tmp_path: Path, paths: Paths) -> None:
    """Finding: an OSError while restoring (a full disk) became 'runner error', exit 1,
    which adapters count as the user's script failing."""
    root = tmp_path / "storage"
    uri = _publish(rs.LocalStore(root), 2, 2, tmp_path)
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("x")
    archive = _bundle(tmp_path, paths, TRAIN)
    env = _env(root, 2, GPU_RESUME_URI=uri, GPU_RESUME_DIR=str(blocker / "resume"))
    proc = _run(archive, tmp_path / "w", env)
    assert proc.returncode == bootstrap.INSTALL_FAILED_EXIT, proc.stdout
    assert "could not restore the checkpoint on this machine" in proc.stdout


def test_low_disk_publishes_straight_from_the_checkpoint_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Finding: every sync staged a full extra copy on the disk the job writes to."""
    import shutil

    real = shutil.disk_usage

    def tiny(path: str) -> Any:
        got = real(path)
        return type(got)(got.total, got.used, 1024)

    monkeypatch.setattr(bootstrap.shutil, "disk_usage", tiny)
    store = rs.LocalStore(tmp_path / "root")
    syncer, ckpt, _tee = _syncer(tmp_path, store)
    _save(ckpt, '{"step": 1}')
    assert syncer.sync() == 1
    assert not (tmp_path / "stage").exists()  # nothing staged
    restored = tmp_path / "restored"
    rs.restore_checkpoint(store, rs.ckpt_key(JOB, 1), restored)
    assert rs.verify_checkpoint(store, rs.ckpt_key(JOB, 1), restored) is None


def test_old_pythons_are_told_hf_storage_needs_3_10(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(bootstrap.sys, "version_info", (3, 9, 18))
    tee = _Tee()
    assert not bootstrap._ensure_hf(rs, tmp_path, tee)
    assert "needs Python 3.10+" in tee.lines[-1]
    store, why = bootstrap._open_storage("hf://buckets/a/b", "tok", rs, tmp_path, tee)
    assert store is None
    assert why == "Hugging Face storage needs Python 3.10+"


TOKEN_JOB = """\
import os
print("env token=%s file=%s" % ("GPU_STORAGE_TOKEN" in os.environ,
                                 "GPU_STORAGE_TOKEN_FILE" in os.environ))
"""


def test_the_storage_token_file_is_read_once_and_deleted(tmp_path: Path, paths: Paths) -> None:
    """Finding: a token in bootstrap's exec environment stays readable in
    /proc/<pid>/environ by the job; launchers now hand it over in a file."""
    root = tmp_path / "storage"
    token = "hf_" + "f" * 34
    tok = tmp_path / ".storage-token"
    tok.write_text(token)
    tok.chmod(0o600)
    archive = _bundle(tmp_path, paths, TOKEN_JOB)
    proc = _run(archive, tmp_path / "w", _env(root, 1, GPU_STORAGE_TOKEN_FILE=str(tok)))
    assert proc.returncode == 0, proc.stdout
    assert "env token=False file=False" in proc.stdout
    assert not tok.exists()
    assert token not in (root / "jobs" / JOB / "attempts/1/log-tail.json").read_text()


def test_python_version_constant_matches_the_requirement() -> None:
    assert rs.HF_REQUIREMENT.startswith("huggingface_hub>=1.")
    assert bootstrap.HF_MIN_PYTHON == (3, 10)
    assert sys.version_info >= bootstrap.HF_MIN_PYTHON  # the daemon itself runs 3.12
