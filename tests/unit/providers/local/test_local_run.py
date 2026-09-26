"""LocalAdapter: a run end to end, logs, fetch, resume (real processes, tmp home)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from gpu_router.adapters.base import RemotePhase
from gpu_router.errors import InvalidJob
from gpu_router.models import Checkpoint
from gpu_router.paths import Paths
from gpu_router.protocol import parse_line
from tests.unit.providers.local.helpers import (
    TERMINAL,
    all_lines,
    build_project,
    drain,
    make_adapter,
    make_bundle,
    make_ctx,
    make_job,
    run_dir,
    wait_for_line,
    wait_phase,
)

TRAIN = """
import os, gpu
gpu.total_steps(3)
for i in range(1, 4):
    gpu.log(step=i, loss=1.0 / i)
    print(f"step {i}/3", flush=True)
(gpu.output_dir() / "result.txt").write_text("ok\\n")
(gpu.output_dir() / "sub").mkdir(exist_ok=True)
(gpu.output_dir() / "sub" / "metrics.json").write_text('{"loss": 0.33}')
with gpu.atomic_checkpoint("last.txt") as tmp:
    tmp.write_text("3")
print("mps fallback", os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"))
"""


def test_run_succeeds_with_protocol_lines_outputs_and_checkpoint(
    paths: Paths, tmp_path: Path
) -> None:
    adapter = make_adapter(paths)
    project = build_project(tmp_path, TRAIN)
    bundle = make_bundle(paths, project)
    job = make_job(project)
    ref = adapter.submit(job, make_ctx(job, bundle))
    assert ref.remote_id == f"gpu-{job.id}-1"
    assert ref.url is not None
    assert ref.url.startswith("file://")
    st = wait_phase(adapter, ref)
    assert st.phase is RemotePhase.SUCCEEDED
    assert st.exit_code == 0
    assert st.gpu == "MPS"
    assert st.ended_at is not None

    lines = all_lines(adapter, ref)
    kinds = [e.t for line in lines if (e := parse_line(line)) is not None]
    assert kinds[0] == "hello"
    assert {"total", "metric", "ckpt_begin", "ckpt_end", "exit"} <= set(kinds)
    assert "step 3/3" in lines
    assert "mps fallback 1" in lines
    ckpt = next(e for line in lines if (e := parse_line(line)) is not None and e.t == "ckpt_end")
    assert ckpt.seq == 1
    assert ckpt.uri.startswith("file://")
    assert Path(ckpt.uri.removeprefix("file://")).is_file()

    dest = tmp_path / "out"
    result = adapter.fetch(ref, dest)
    assert result.files == 2
    assert (dest / "result.txt").read_text() == "ok\n"
    assert (dest / "sub" / "metrics.json").is_file()
    assert not result.partial


def test_bundle_dir_without_archive_runs(paths: Paths, tmp_path: Path) -> None:
    from gpu_router.packaging.bundle import extract_bundle

    adapter = make_adapter(paths)
    project = build_project(tmp_path, "print('from a dir bundle')\n")
    bundle = make_bundle(paths, project)
    extracted = tmp_path / "extracted"
    extract_bundle(bundle.archive, extracted)
    job = make_job(project)
    ref = adapter.submit(job, make_ctx(job, None, bundle_dir=extracted))
    assert wait_phase(adapter, ref).phase is RemotePhase.SUCCEEDED
    assert "from a dir bundle" in all_lines(adapter, ref)


def test_context_without_bundle_paths_uses_the_jobs_materialized_bundle(
    paths: Paths, tmp_path: Path
) -> None:
    from gpu_router.packaging.bundle import materialize

    adapter = make_adapter(paths)
    project = build_project(tmp_path, "print('materialized')\n")
    bundle = make_bundle(paths, project)
    job = make_job(project)
    materialize(paths, bundle.sha256, job.id)
    ref = adapter.submit(job, make_ctx(job, None))
    assert wait_phase(adapter, ref).phase is RemotePhase.SUCCEEDED


def test_no_bundle_is_invalid_job_and_creates_nothing(paths: Paths, tmp_path: Path) -> None:
    adapter = make_adapter(paths)
    job = make_job(build_project(tmp_path, "print(1)\n"))
    ctx = make_ctx(job, None)
    with pytest.raises(InvalidJob) as info:
        adapter.submit(job, ctx)
    assert info.value.hint
    assert adapter.lookup_by_key(ctx.attempt_key) is None


def test_nonzero_exit_is_failed_with_code(paths: Paths, tmp_path: Path) -> None:
    adapter = make_adapter(paths)
    project = build_project(tmp_path, "import sys\nprint('bad things')\nsys.exit(3)\n")
    job = make_job(project)
    ref = adapter.submit(job, make_ctx(job, make_bundle(paths, project)))
    st = wait_phase(adapter, ref)
    assert st.phase is RemotePhase.FAILED
    assert st.exit_code == 3
    result = adapter.fetch(ref, tmp_path / "out")  # the output dir exists, just empty
    assert result.files == 0


def test_log_cursor_resume_never_repeats(paths: Paths, tmp_path: Path) -> None:
    script = """
import sys, time
for i in range(5):
    print(f"line {i}", flush=True)
    time.sleep(0.05)
sys.stdout.write("half a line")
sys.stdout.flush()
time.sleep(0.4)
sys.stdout.write(" and the rest\\nno newline at the end")
sys.stdout.flush()
"""
    adapter = make_adapter(paths)
    project = build_project(tmp_path, script)
    job = make_job(project)
    ref = adapter.submit(job, make_ctx(job, make_bundle(paths, project)))
    seen: list[str] = []
    cursor: str | None = None
    eof = False
    for _ in range(600):
        lines, cursor, eof = drain(list(adapter.logs(ref, since=cursor)))
        seen += lines
        if eof:
            break
        time.sleep(0.02)
    assert eof
    full = all_lines(adapter, ref)
    assert seen == full
    assert "half a line and the rest" in full
    assert "no newline at the end" in full
    again = list(adapter.logs(ref, since=cursor))
    assert all(not c.lines for c in again)
    assert again[-1].eof


def test_chunk_reader_holds_a_partial_line_until_terminal(tmp_path: Path) -> None:
    from gpu_router.providers.local.adapter import _read_chunks

    log = tmp_path / "console.log"
    log.write_bytes(b"one\r\ntwo\nthr")
    running = list(_read_chunks(log, 0, terminal=False))
    lines, cursor, eof = drain(running)
    assert lines == ["one", "two"]
    assert not eof
    assert cursor == str(len(b"one\r\ntwo\n"))
    with log.open("ab") as fh:
        fh.write(b"ee\nfour")
    lines, cursor, eof = drain(list(_read_chunks(log, int(cursor), terminal=False)))
    assert lines == ["three"]
    lines, cursor, eof = drain(list(_read_chunks(log, int(cursor), terminal=True)))
    assert lines == ["four"]
    assert eof
    assert cursor == str(log.stat().st_size)
    assert drain(list(_read_chunks(log, 10_000, terminal=True))) == ([], cursor, True)
    log.write_bytes("caf\u00e9 \xff\n".encode("utf-8", "surrogateescape") + b"\xff\n")
    lines, _, _ = drain(list(_read_chunks(log, 0, terminal=True)))
    assert lines[0].startswith("caf\u00e9")
    assert lines[1] == "\ufffd"


def test_logs_before_the_log_file_exists_is_one_empty_chunk(paths: Paths, tmp_path: Path) -> None:
    adapter = make_adapter(paths)
    project = build_project(tmp_path, "print(1)\n")
    job = make_job(project)
    ref = adapter.submit(job, make_ctx(job, make_bundle(paths, project)))
    wait_phase(adapter, ref)
    (run_dir(adapter, ref) / "console.log").unlink()
    chunks = list(adapter.logs(ref))
    assert len(chunks) == 1
    assert chunks[0].lines == []
    assert chunks[0].eof


def test_follow_logs_streams_until_eof(paths: Paths, tmp_path: Path) -> None:
    script = (
        "import time\nfor i in range(3):\n    print('tick', i, flush=True)\n    time.sleep(0.2)\n"
    )
    adapter = make_adapter(paths)
    project = build_project(tmp_path, script)
    job = make_job(project)
    ref = adapter.submit(job, make_ctx(job, make_bundle(paths, project)))
    chunks = list(adapter.logs(ref, follow=True))
    assert chunks[-1].eof
    lines = [line for c in chunks for line in c.lines]
    assert [line for line in lines if line.startswith("tick")] == ["tick 0", "tick 1", "tick 2"]
    assert adapter.status(ref).phase is RemotePhase.SUCCEEDED


def test_fetch_is_rerunnable_keeps_user_files_and_overwrites_its_own(
    paths: Paths, tmp_path: Path
) -> None:
    adapter = make_adapter(paths)
    project = build_project(tmp_path, TRAIN)
    job = make_job(project)
    ref = adapter.submit(job, make_ctx(job, make_bundle(paths, project)))
    wait_phase(adapter, ref, {RemotePhase.SUCCEEDED})
    dest = tmp_path / "out"
    first = adapter.fetch(ref, dest)
    (dest / "user-note.txt").write_text("keep me")
    (dest / "result.txt").write_text("edited locally")
    second = adapter.fetch(ref, dest)
    assert second.files == first.files
    assert second.bytes == first.bytes
    assert (dest / "user-note.txt").read_text() == "keep me"
    assert (dest / "result.txt").read_text() == "ok\n"  # ours again: size differs


def test_resume_restores_file_checkpoint_and_continues_seq(paths: Paths, tmp_path: Path) -> None:
    adapter = make_adapter(paths)
    first_project = build_project(tmp_path, TRAIN, name="first")
    job = make_job(first_project)
    ref1 = adapter.submit(job, make_ctx(job, make_bundle(paths, first_project)))
    wait_phase(adapter, ref1, {RemotePhase.SUCCEEDED})
    end = next(
        e
        for line in all_lines(adapter, ref1)
        if (e := parse_line(line)) is not None and e.t == "ckpt_end"
    )
    ckpt = Checkpoint(
        id=f"{job.id}.c{end.seq}",
        job_id=job.id,
        attempt_id=f"{job.id}.1",
        seq=end.seq,
        uri=end.uri,
        created_at=1.0,
        recorded_at=1.0,
    )
    resumed_script = """
import gpu
print("resumed", gpu.is_resumed(), (gpu.resume_dir() / "last.txt").read_text())
with gpu.atomic_checkpoint("last.txt") as tmp:
    tmp.write_text("4")
"""
    second_project = build_project(tmp_path, resumed_script, name="second")
    ref2 = adapter.submit(
        job, make_ctx(job, make_bundle(paths, second_project), n=2, resume_from=ckpt)
    )
    assert wait_phase(adapter, ref2).phase is RemotePhase.SUCCEEDED
    lines = all_lines(adapter, ref2)
    assert "resumed True 3" in lines
    seqs = [e.seq for line in lines if (e := parse_line(line)) is not None and e.t == "ckpt_end"]
    assert seqs == [end.seq + 1]


@pytest.mark.parametrize(
    "uri",
    [
        "hf://user/repo/ckpt.tar.gz",
        "file:///nope/ckpt-0001.tar.gz",
        "file:///content/gr/gr-0123456789ab-1/ckpt-sync/ckpt-0001.tar.gz",  # a Colab VM
        "file:///kaggle/working/.gpu-router/checkpoints/ckpt-0001.tar.gz",  # a Kaggle VM
    ],
)
def test_unreadable_resume_checkpoint_starts_fresh_with_a_note(
    paths: Paths, tmp_path: Path, uri: str
) -> None:
    """D34 (review fix): InvalidJob would exclude local for the rest of the job, although
    it can run the job from scratch; the cloud adapters start fresh the same way."""
    adapter = make_adapter(paths)
    project = build_project(tmp_path, "import gpu\nprint('resumed', gpu.is_resumed())\n")
    job = make_job(project)
    ckpt = Checkpoint(
        id=f"{job.id}.c1",
        job_id=job.id,
        attempt_id=f"{job.id}.1",
        seq=1,
        uri=uri,
        created_at=1.0,
        recorded_at=1.0,
    )
    ctx = make_ctx(job, make_bundle(paths, project), resume_from=ckpt)
    ref = adapter.submit(job, ctx)
    assert wait_phase(adapter, ref).phase is RemotePhase.SUCCEEDED
    lines = all_lines(adapter, ref)
    assert any("is not reachable from this Mac; starting fresh" in line for line in lines)
    assert "resumed False" in lines
    record = json.loads((run_dir(adapter, ref) / "run.json").read_text())
    assert record["resume"] == "unavailable"


def test_a_colab_checkpoint_resumes_from_its_mirror_on_this_mac(
    paths: Paths, tmp_path: Path
) -> None:
    """The Colab adapter mirrors small VM checkpoints to <home>/providers/colab/runs/..."""
    import io
    import tarfile

    session = "gr-0123456789ab-1"
    mirror = paths.provider_dir("colab") / "runs" / session / "ckpt" / "ckpt-0003.tar.gz"
    mirror.parent.mkdir(parents=True)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        data = b"step=3"
        info = tarfile.TarInfo("state.txt")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
    mirror.write_bytes(buf.getvalue())
    adapter = make_adapter(paths)
    project = build_project(
        tmp_path,
        "import gpu\nprint('resumed', gpu.is_resumed())\n"
        "print('state', (gpu.resume_dir() / 'state.txt').read_text())\n",
    )
    job = make_job(project)
    ckpt = Checkpoint(
        id=f"{job.id}.c3",
        job_id=job.id,
        attempt_id=f"{job.id}.1",
        seq=3,
        uri=f"file:///content/gr/{session}/ckpt-sync/ckpt-0003.tar.gz",
        created_at=1.0,
        recorded_at=1.0,
    )
    ref = adapter.submit(job, make_ctx(job, make_bundle(paths, project), resume_from=ckpt))
    assert wait_phase(adapter, ref).phase is RemotePhase.SUCCEEDED, all_lines(adapter, ref)
    lines = all_lines(adapter, ref)
    assert "resumed True" in lines
    assert "state step=3" in lines


def test_run_record_and_plan_on_disk(paths: Paths, tmp_path: Path) -> None:
    adapter = make_adapter(paths)
    project = build_project(tmp_path, "print(1)\n")
    job = make_job(project)
    ctx = make_ctx(job, make_bundle(paths, project))
    ref = adapter.submit(job, ctx)
    wait_phase(adapter, ref)
    rd = run_dir(adapter, ref)
    assert rd == paths.home / "local" / ctx.attempt_key
    record = json.loads((rd / "run.json").read_text())
    assert record["attempt_key"] == ctx.attempt_key
    assert record["job_id"] == job.id
    plan = json.loads((rd / "launch.json").read_text())
    assert "--ckpt-seq-start" in plan["bootstrap_args"]
    assert (rd / "phase").read_text().strip() == "run"
    assert (rd / "work" / "EXIT").read_text().strip() == "0"
    assert oct((rd / "run.json").stat().st_mode & 0o777) == "0o600"


def test_all_terminal_phases_have_eof_logs(paths: Paths, tmp_path: Path) -> None:
    adapter = make_adapter(paths)
    project = build_project(tmp_path, "raise SystemExit(5)\n")
    job = make_job(project)
    ref = adapter.submit(job, make_ctx(job, make_bundle(paths, project)))
    assert wait_phase(adapter, ref).phase in TERMINAL
    chunks = list(adapter.logs(ref))
    assert chunks[-1].eof
    wait_for_line(adapter, ref, '"t":"exit","code":5')
