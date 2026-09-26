"""The `gpu` helper: protocol lines round-trip through protocol.py; plain text outside
gpu-router still feeds the stdout fallback parser; nothing ever raises."""

from __future__ import annotations

import ast
import math
import os
from pathlib import Path

import pytest

from gpu_router import protocol
from gpu_router.runner import bootstrap, gpu

RUNNER_DIR = Path(gpu.__file__).resolve().parent


@pytest.fixture(autouse=True)
def _fresh(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(gpu, "_total", None)
    for var in (
        "GPU_ROUTER_PROTOCOL",
        "GPU_CHECKPOINT_DIR",
        "GPU_OUTPUT_DIR",
        "GPU_RESUME_DIR",
        "GPU_DATA_DIR",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def active(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GPU_ROUTER_PROTOCOL", "1")


def lines(capsys: pytest.CaptureFixture[str]) -> list[str]:
    return capsys.readouterr().out.splitlines()


# --------------------------------------------------------------------------- protocol mode


@pytest.mark.usefixtures("active")
def test_total_steps_round_trips(capsys: pytest.CaptureFixture[str]) -> None:
    gpu.total_steps(1000)
    (line,) = lines(capsys)
    assert line == protocol.total(1000)  # byte-identical to the daemon's formatter
    ev = protocol.parse_line(line)
    assert ev is not None
    assert (ev.t, ev.total) == ("total", 1000)


@pytest.mark.usefixtures("active")
def test_log_round_trips_with_total(capsys: pytest.CaptureFixture[str]) -> None:
    gpu.total_steps(100)
    gpu.log(step=12, loss=0.412, lr=3e-4)
    out = lines(capsys)
    assert out[1] == protocol.metric(12, {"loss": 0.412, "lr": 3e-4}, total_steps=100)
    ev = protocol.parse_line(out[1])
    assert ev is not None
    assert (ev.t, ev.step, ev.total) == ("metric", 12, 100)
    assert ev.metrics == {"loss": 0.412, "lr": 3e-4}


@pytest.mark.usefixtures("active")
def test_log_without_step(capsys: pytest.CaptureFixture[str]) -> None:
    gpu.log(accuracy=0.9)
    ev = protocol.parse_line(lines(capsys)[0])
    assert ev is not None
    assert ev.step is None
    assert ev.metrics == {"accuracy": 0.9}


class _Scalar:
    """Stands in for a 0-d torch tensor / numpy scalar."""

    def __init__(self, v: float) -> None:
        self.v = v

    def item(self) -> float:
        return self.v


@pytest.mark.usefixtures("active")
def test_log_accepts_scalars_and_drops_junk(capsys: pytest.CaptureFixture[str]) -> None:
    gpu.log(
        step=_Scalar(7.0),
        loss=_Scalar(0.5),
        n=3,
        bad="x",
        nan=math.nan,
        inf=math.inf,
        flag=True,
        none=None,
        obj=object(),
    )
    ev = protocol.parse_line(lines(capsys)[0])
    assert ev is not None
    assert ev.step == 7
    assert ev.metrics == {"loss": 0.5, "n": 3.0}


@pytest.mark.usefixtures("active")
def test_invalid_total_is_ignored(capsys: pytest.CaptureFixture[str]) -> None:
    gpu.total_steps(0)
    gpu.total_steps("lots")
    gpu.total_steps(-5)
    assert lines(capsys) == []


@pytest.mark.usefixtures("active")
def test_prefix_matches_protocol() -> None:
    assert gpu.PREFIX == protocol.PREFIX == bootstrap.PREFIX
    assert protocol.PROTOCOL_VERSION == bootstrap.PROTOCOL_VERSION


@pytest.mark.usefixtures("active")
def test_heartbeat_lines_are_ignored_by_parser(capsys: pytest.CaptureFixture[str]) -> None:
    tee = bootstrap.Tee(None)
    tee.event("heartbeat", ts=1.0, elapsed=2.0)
    tee.event("hello", v=1, runner=bootstrap.RUNNER_VERSION)
    tee.event("exit", code=3)
    hb, hello, ex = lines(capsys)
    assert protocol.is_protocol_line(hb)
    assert protocol.parse_line(hb) is None
    h = protocol.parse_line(hello)
    assert h is not None
    assert (h.t, h.version, h.runner) == ("hello", 1, bootstrap.RUNNER_VERSION)
    assert hello == protocol.hello(bootstrap.RUNNER_VERSION)
    e = protocol.parse_line(ex)
    assert e is not None
    assert (e.t, e.code) == ("exit", 3)


# --------------------------------------------------------------------------- plain mode


def test_plain_mode_feeds_stdout_fallback(capsys: pytest.CaptureFixture[str]) -> None:
    assert not gpu.enabled()
    gpu.total_steps(50)
    gpu.log(step=10, loss=0.41, lr=0.001)
    gpu.log(step=11, loss=0.40)
    out = lines(capsys)
    assert not any(protocol.is_protocol_line(line) for line in out)
    parser = protocol.StdoutMetricParser()
    results = [parser.feed(line) for line in out]
    assert results[1].step == 10
    assert results[1].total == 50
    assert results[1].metrics == {"loss": 0.41, "lr": 0.001}
    assert (results[2].step, results[2].total) == (11, 50)


def test_plain_mode_without_total(capsys: pytest.CaptureFixture[str]) -> None:
    gpu.log(step=3, loss=1)
    (line,) = lines(capsys)
    assert line == "step=3 loss=1"
    res = protocol.StdoutMetricParser().feed(line)
    assert (res.step, res.metrics) == (3, {"loss": 1.0})


def test_log_with_nothing_prints_nothing(capsys: pytest.CaptureFixture[str]) -> None:
    gpu.log()
    gpu.log(junk="x")
    assert lines(capsys) == []


def test_broken_stdout_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    class Broken:
        def write(self, _s: str) -> int:
            raise BrokenPipeError

        def flush(self) -> None:
            raise BrokenPipeError

    monkeypatch.setattr("sys.stdout", Broken())
    monkeypatch.setenv("GPU_ROUTER_PROTOCOL", "1")
    gpu.total_steps(5)
    gpu.log(step=1, loss=1.0)
    monkeypatch.setattr("sys.stdout", None)
    gpu.log(step=2, loss=1.0)


# --------------------------------------------------------------------------- dirs


def test_dirs_default_to_cwd(tmp_path: Path) -> None:
    assert gpu.checkpoint_dir() == tmp_path / "checkpoints"
    assert gpu.output_dir() == tmp_path / "outputs"
    assert gpu.checkpoint_dir().is_dir()
    assert gpu.output_dir().is_dir()
    assert gpu.data_dir() == tmp_path / "data"
    assert gpu.resume_dir() is None
    assert not gpu.is_resumed()
    assert gpu.latest_checkpoint() is None


def test_dirs_from_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GPU_CHECKPOINT_DIR", str(tmp_path / "ck"))
    monkeypatch.setenv("GPU_OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setenv("GPU_DATA_DIR", "/data")
    assert gpu.checkpoint_dir() == tmp_path / "ck"
    assert (tmp_path / "ck").is_dir()
    assert gpu.output_dir() == tmp_path / "out"
    assert str(gpu.data_dir()) == "/data"


def test_resume_dir_and_latest_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    resume = tmp_path / "resume"
    monkeypatch.setenv("GPU_RESUME_DIR", str(resume))
    assert gpu.resume_dir() is None  # missing dir = fresh start
    resume.mkdir()
    assert gpu.resume_dir() is None  # empty dir = fresh start
    (resume / "step-100.pt").write_text("a")
    (resume / "step-200.pt").write_text("b")
    (resume / ".hidden").write_text("c")
    os.utime(resume / "step-100.pt", (1000, 1000))
    os.utime(resume / "step-200.pt", (2000, 2000))
    assert gpu.resume_dir() == resume
    assert gpu.is_resumed()
    latest = gpu.latest_checkpoint()
    assert latest is not None
    assert latest.name == "step-200.pt"


def test_latest_checkpoint_falls_back_to_checkpoint_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ck = tmp_path / "ck"
    ck.mkdir()
    (ck / "last.pt").write_text("x")
    monkeypatch.setenv("GPU_CHECKPOINT_DIR", str(ck))
    latest = gpu.latest_checkpoint()
    assert latest is not None
    assert latest.name == "last.pt"
    assert not gpu.is_resumed()


# --------------------------------------------------------------------------- py3.8 compat


@pytest.mark.parametrize("name", ["gpu.py", "bootstrap.py", "__init__.py"])
def test_runner_parses_as_python38(name: str) -> None:
    source = (RUNNER_DIR / name).read_text()
    tree = ast.parse(source, feature_version=(3, 8))
    for node in ast.walk(tree):
        # no gpu_router / third-party imports: the runner ships alone
        if isinstance(node, ast.Import):
            assert all(not a.name.startswith("gpu_router") for a in node.names)
        if isinstance(node, ast.ImportFrom):
            assert node.module is None or not node.module.startswith("gpu_router")
        # no annotations at all: `list[str]` etc. would break at runtime on 3.8
        if isinstance(node, ast.FunctionDef):
            assert node.returns is None
            assert all(a.annotation is None for a in node.args.args)


# --------------------------------------------------------------------------- atomic saves


def test_atomic_checkpoint_renames_into_place_only_when_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GPU_CHECKPOINT_DIR", str(tmp_path / "ck"))
    final = tmp_path / "ck" / "last.pt"
    with gpu.atomic_checkpoint("last.pt") as tmp:
        assert tmp.parent.name == ".gpu-tmp"  # never archived by the runner
        tmp.write_text("v1")
        assert not final.exists()
    assert final.read_text() == "v1"

    def crash_mid_save() -> None:
        with gpu.atomic_checkpoint("last.pt") as tmp:
            tmp.write_text("half")
            raise RuntimeError("OOM mid-save")

    with pytest.raises(RuntimeError):
        crash_mid_save()
    assert final.read_text() == "v1"  # the good checkpoint survives
    assert not list((tmp_path / "ck" / ".gpu-tmp").iterdir())
    assert bootstrap._fingerprint(tmp_path / "ck") == (("last.pt", 2, final.stat().st_mtime_ns),)
    assert gpu.latest_checkpoint() == final
