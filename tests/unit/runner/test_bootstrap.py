"""bootstrap.py end to end: a real bundle, a tiny user script, run locally in a tmp dir."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from gpu_router import protocol
from gpu_router.models import JobSpec
from gpu_router.packaging import build_bundle, materialize
from gpu_router.paths import Paths
from gpu_router.runner import bootstrap
from tests.unit.packaging.helpers import make_project

BOOTSTRAP = Path(bootstrap.__file__).resolve()

TRAIN = """\
import sys
import gpu

gpu.total_steps(3)
for step in range(1, 4):
    gpu.log(step=step, loss=1.0 / step)
(gpu.checkpoint_dir() / "last.pt").write_text("step %d" % step)
(gpu.output_dir() / "result.txt").write_text("ok")
print("resumed=%s" % gpu.is_resumed())
latest = gpu.latest_checkpoint()
print("latest=%s" % (latest.name if latest else None))
print("args=%s" % sys.argv[1:])
print("to stderr", file=sys.stderr)
print("loss=0.25 step 3/3")
sys.stdout.write("progress 10%|#   | 1/10\\rprogress 100%|####| 10/10\\n")
sys.exit(int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 0)
"""


def _bundle(tmp_path: Path, paths: Paths, files: dict[str, str | bytes], **spec: object) -> Path:
    project = make_project(tmp_path / "proj", files)
    body: dict[str, object] = {"project_dir": str(project), "script": "train.py"}
    body.update(spec)
    return build_bundle(project, JobSpec.model_validate(body), paths=paths).archive


def _run(
    *args: str, python: str = sys.executable, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    import os

    full_env = {k: v for k, v in os.environ.items() if not k.startswith(("GPU_", "PYTHONPATH"))}
    full_env.update(env or {})
    return subprocess.run(
        [python, str(BOOTSTRAP), *args],
        capture_output=True,
        text=True,
        timeout=120,
        env=full_env,
        check=False,
    )


def _events(text: str) -> list[protocol.ProtocolEvent]:
    return [ev for line in text.splitlines() if (ev := protocol.parse_line(line)) is not None]


def test_end_to_end_success(tmp_path: Path, paths: Paths) -> None:
    archive = _bundle(tmp_path, paths, {"train.py": TRAIN}, args=["--flag"])
    work = tmp_path / "work"
    sync = tmp_path / "sync"
    proc = _run(
        "--bundle", str(archive), "--workdir", str(work), "--skip-install",
        "--checkpoint-sync-dir", str(sync), "--ckpt-seq-start", "4",
    )  # fmt: skip
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (work / "EXIT").read_text() == "0\n"
    log = (work / "job.log").read_text()
    assert log.splitlines() == proc.stdout.splitlines()  # teed byte for byte

    evs = _events(log)
    assert evs[0].t == "hello"
    assert evs[0].runner == bootstrap.RUNNER_VERSION
    assert [e.total for e in evs if e.t == "total"] == [3]
    metrics = [e for e in evs if e.t == "metric"]
    assert [(m.step, m.total, m.metrics["loss"]) for m in metrics] == [
        (1, 3, 1.0),
        (2, 3, 0.5),
        (3, 3, 1.0 / 3),
    ]
    # final checkpoint sync: seq continues from --ckpt-seq-start, file:// uri, step, sha
    begin = [e for e in evs if e.t == "ckpt_begin"]
    end = [e for e in evs if e.t == "ckpt_end"]
    assert [e.seq for e in begin] == [4]
    assert len(end) == 1
    assert end[0].seq == 4
    assert end[0].step == 3
    assert end[0].uri is not None
    assert end[0].uri.startswith("file://")
    assert end[0].sha256 is not None
    with tarfile.open(sync / "ckpt-0004.tar.gz") as tar:
        assert tar.getnames() == ["last.pt"]
    assert evs[-1].t == "exit"
    assert evs[-1].code == 0

    assert "to stderr" in log  # stderr merged
    assert "resumed=False" in log
    assert "args=['--flag']" in log
    assert (work / "outputs" / "result.txt").read_text() == "ok"
    assert (work / "checkpoints" / "last.pt").is_file()


def test_stdout_fallback_still_parses_relayed_output(tmp_path: Path, paths: Paths) -> None:
    archive = _bundle(tmp_path, paths, {"train.py": TRAIN})
    work = tmp_path / "work"
    proc = _run("--bundle", str(archive), "--workdir", str(work), "--skip-install")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    parser = protocol.StdoutMetricParser()
    plain = [parser.feed(line) for line in proc.stdout.splitlines()]
    assert any(r.metrics.get("loss") == 0.25 and (r.step, r.total) == (3, 3) for r in plain)
    # tqdm '\r' updates: the final bar state is what the parser sees
    assert any((r.step, r.total) == (10, 10) for r in plain)
    assert "\r" not in proc.stdout  # '\r' updates become separate, throttled lines


def test_nonzero_exit_is_propagated(tmp_path: Path, paths: Paths) -> None:
    archive = _bundle(tmp_path, paths, {"train.py": TRAIN}, args=["3"])
    work = tmp_path / "work"
    proc = _run("--bundle", str(archive), "--workdir", str(work), "--skip-install")
    assert proc.returncode == 3
    assert (work / "EXIT").read_text() == "3\n"
    assert _events(proc.stdout)[-1].code == 3


def test_resume_restores_checkpoint(tmp_path: Path, paths: Paths) -> None:
    archive = _bundle(tmp_path, paths, {"train.py": TRAIN})
    ckpt_src = tmp_path / "prev"
    ckpt_src.mkdir()
    (ckpt_src / "step-200.pt").write_text("old")
    resume_tar = tmp_path / "ckpt.tar.gz"
    with tarfile.open(resume_tar, "w:gz") as tar:
        tar.add(ckpt_src / "step-200.pt", arcname="step-200.pt")
    work = tmp_path / "work"
    proc = _run(
        "--bundle", str(archive), "--workdir", str(work), "--skip-install",
        "--resume", str(resume_tar),
    )  # fmt: skip
    assert proc.returncode == 0, proc.stdout
    assert "resumed=True" in proc.stdout
    assert "latest=step-200.pt" in proc.stdout
    assert (work / "resume" / "step-200.pt").read_text() == "old"
    assert (work / "checkpoints" / "step-200.pt").is_file()  # seeded for resume-from-dir code


def test_runs_from_materialized_bundle_dir(tmp_path: Path, paths: Paths) -> None:
    archive = _bundle(tmp_path, paths, {"train.py": TRAIN})
    sha = archive.name.removesuffix(".tar.gz")
    bundle_dir, _ = materialize(paths, sha, "abc123def456")
    run_dir = tmp_path / "remote"
    shutil.copytree(bundle_dir, run_dir)
    proc = subprocess.run(
        [sys.executable, str(run_dir / "gpu_runner" / "bootstrap.py"), "--skip-install"],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        cwd=tmp_path,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (run_dir / "EXIT").read_text() == "0\n"
    assert (run_dir / "outputs" / "result.txt").is_file()


def test_missing_bundle_writes_exit_file(tmp_path: Path) -> None:
    work = tmp_path / "work"
    proc = _run("--bundle", str(tmp_path / "nope.tar.gz"), "--workdir", str(work))
    assert proc.returncode == 1
    assert "runner error" in proc.stdout
    assert (work / "EXIT").read_text() == "1\n"
    assert _events(proc.stdout)[-1].code == 1


def test_failed_install_does_not_start_job(tmp_path: Path, paths: Paths) -> None:
    archive = _bundle(tmp_path, paths, {"train.py": TRAIN, "requirements.txt": "numpy\n"})
    fake_python = tmp_path / "fakepy"
    fake_python.write_text('#!/bin/sh\necho "pip says no: $*"\nexit 5\n')
    fake_python.chmod(0o755)
    work = tmp_path / "work"
    proc = _run("--bundle", str(archive), "--workdir", str(work), "--python", str(fake_python))
    # review finding: pip's own code (1/2) looked like the user's script failing, so an
    # offline provider made the job fail permanently instead of rerouting
    assert proc.returncode == bootstrap.INSTALL_FAILED_EXIT
    assert "pip says no: -m pip install" in proc.stdout
    assert "-r" in proc.stdout
    assert "dependency install failed (pip exit 5)" in proc.stdout
    assert "resumed=" not in proc.stdout
    assert (work / "EXIT").read_text() == f"{bootstrap.INSTALL_FAILED_EXIT}\n"
    marker = [ln for ln in proc.stdout.splitlines() if '"install_failed"' in ln]
    assert marker == ['::gpu:: {"t":"install_failed","code":5}']
    assert protocol.parse_line(marker[0]) is None  # ignored by the daemon's parser


def test_install_deps_pyproject_packages(tmp_path: Path) -> None:
    rec = tmp_path / "argv.json"
    fake_python = tmp_path / "fakepy"
    fake_python.write_text(
        f"#!{sys.executable}\nimport json, sys\njson.dump(sys.argv[1:], open({str(rec)!r}, 'w'))\n"
    )
    fake_python.chmod(0o755)
    tee = bootstrap.Tee(None)
    manifest = {"deps": {"kind": "pyproject", "packages": ["torch>=2", "numpy"]}}
    assert bootstrap.install_deps(manifest, tmp_path, str(fake_python), tee) == 0
    argv = json.loads(rec.read_text())
    assert argv[:3] == ["-m", "pip", "install"]
    assert argv[-2:] == ["torch>=2", "numpy"]
    assert bootstrap.install_deps({"deps": {"kind": "none"}}, tmp_path, "/nonexistent", tee) == 0


def test_command_entrypoint(tmp_path: Path, paths: Paths) -> None:
    project = make_project(tmp_path / "proj", {"run.sh": "echo from-shell $1\nexit 7\n"})
    spec = JobSpec(project_dir=str(project), command=["sh", "run.sh"], args=["hi"])
    archive = build_bundle(project, spec, paths=paths).archive
    work = tmp_path / "work"
    proc = _run("--bundle", str(archive), "--workdir", str(work), "--skip-install")
    assert proc.returncode == 7
    assert "from-shell hi" in proc.stdout


@pytest.mark.slow
def test_heartbeat_and_interval_sync(tmp_path: Path, paths: Paths) -> None:
    script = (
        "import time, gpu\n"
        "gpu.log(step=1, loss=1.0)\n"
        "(gpu.checkpoint_dir() / 'a.pt').write_text('1')\n"
        "time.sleep(2.6)\n"
    )
    archive = _bundle(tmp_path, paths, {"train.py": script})
    work = tmp_path / "work"
    proc = _run(
        "--bundle", str(archive), "--workdir", str(work), "--skip-install",
        "--heartbeat-s", "1", "--checkpoint-sync-dir", str(tmp_path / "sync"),
        "--checkpoint-interval-min", "0.02",
    )  # fmt: skip
    assert proc.returncode == 0, proc.stdout
    beats = [
        json.loads(line[len(protocol.PREFIX) :])
        for line in proc.stdout.splitlines()
        if protocol.is_protocol_line(line) and '"heartbeat"' in line
    ]
    assert len(beats) >= 1
    assert beats[0]["elapsed"] > 0
    ends = [e for e in _events(proc.stdout) if e.t == "ckpt_end"]
    # synced during the run (interval 1.2 s), not again at exit because nothing changed
    assert [e.seq for e in ends] == [1]


def _python38() -> str | None:
    uv = shutil.which("uv")
    if uv is None:
        return None
    try:
        out = subprocess.run(
            [uv, "python", "find", "--no-project", "3.8"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    path = out.stdout.strip()
    return path if out.returncode == 0 and path else None


def test_runs_on_python38(tmp_path: Path, paths: Paths) -> None:
    py38 = _python38()
    if py38 is None:
        pytest.skip("no Python 3.8 interpreter (uv python install 3.8)")
    archive = _bundle(tmp_path, paths, {"train.py": TRAIN})
    work = tmp_path / "work"
    proc = _run(
        "--bundle", str(archive), "--workdir", str(work), "--skip-install",
        "--python", py38, "--checkpoint-sync-dir", str(tmp_path / "sync"),
        python=py38,
    )  # fmt: skip
    assert proc.returncode == 0, proc.stdout + proc.stderr
    evs = _events(proc.stdout)
    assert [e.step for e in evs if e.t == "metric"] == [1, 2, 3]
    assert any(e.t == "ckpt_end" for e in evs)
    assert (work / "EXIT").read_text() == "0\n"


def test_device_line_reports_what_nvidia_smi_sees(tmp_path: Path, paths: Paths) -> None:
    """D56: right after hello the runner reports nvidia-smi's GPUs (the engine compares them
    with the GPU the attempt was placed on); no nvidia-smi = no device line."""
    import os

    archive = _bundle(tmp_path, paths, {"train.py": "print('hi')\n"})
    bindir = tmp_path / "bin"
    bindir.mkdir()
    smi = bindir / "nvidia-smi"
    smi.write_text("#!/bin/sh\nprintf 'Tesla T4, 15360 MiB\\n\\nTesla T4, 15360 MiB\\n'\n")
    smi.chmod(0o755)
    path = f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}"
    proc = _run(
        "--bundle", str(archive), "--workdir", str(tmp_path / "w1"), "--skip-install",
        env={"PATH": path},
    )  # fmt: skip
    assert proc.returncode == 0, proc.stdout + proc.stderr
    evs = _events(proc.stdout)
    assert [e.t for e in evs[:2]] == ["hello", "device"]
    assert evs[1].gpus == ("Tesla T4, 15360 MiB", "Tesla T4, 15360 MiB")

    smi.write_text("#!/bin/sh\necho 'NVIDIA-SMI has failed' >&2\nexit 9\n")
    proc = _run(
        "--bundle", str(archive), "--workdir", str(tmp_path / "w2"), "--skip-install",
        env={"PATH": path},
    )  # fmt: skip
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "device" not in [e.t for e in _events(proc.stdout)]
