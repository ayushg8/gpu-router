"""D56: the runner reports the GPU nvidia-smi sees (`device` line); the engine notes one
that differs from the GPU the attempt was placed on and says so when the job ends, instead
of silently claiming the requested GPU (the live Lightning L4 -> T4 report, job e940)."""

from __future__ import annotations

import pytest

from gpu_router import protocol
from gpu_router.adapters.base import LogChunk
from gpu_router.engine.capture import LogCapture, gpu_mismatch
from gpu_router.models import JobState, Reason
from gpu_router.paths import Paths
from tests.unit.engine.conftest import Engine

T4 = "Tesla T4, 15360 MiB"


@pytest.mark.parametrize(
    ("placed", "seen", "expected"),
    [
        ("T4", [T4], None),
        ("t4", [T4], None),
        ("L4", ["NVIDIA L4, 23034 MiB"], None),
        ("L4", [T4], "Tesla T4"),
        ("L4", ["NVIDIA L40S, 46068 MiB"], "NVIDIA L40S"),  # an L40S is not an L4
        ("P100", ["Tesla P100-PCIE-16GB, 16384 MiB"], None),
        ("A100-40GB", ["NVIDIA A100-SXM4-40GB, 40960 MiB"], None),
        ("A100-40GB", ["NVIDIA A100-SXM4-80GB, 81920 MiB"], "NVIDIA A100-SXM4-80GB"),
        ("2xT4", [T4, T4], None),
        ("2xT4", [T4], "Tesla T4"),  # one GPU where two were placed
        ("T4", [T4, T4], None),  # more than placed is fine
        ("T4", [T4, "NVIDIA L4, 23034 MiB"], "Tesla T4 + NVIDIA L4"),
        ("L4", [T4, T4], "2x Tesla T4"),
        (None, [T4], None),  # nothing to compare with
        ("T4", [], None),
        ("T4", ["  "], None),
    ],
)
def test_gpu_mismatch(placed: str | None, seen: list[str], expected: str | None) -> None:
    assert gpu_mismatch(placed, seen) == expected


def test_device_line_round_trips_and_is_sanitized() -> None:
    ev = protocol.parse_line(protocol.device([T4, "NVIDIA L4, 23034 MiB"]))
    assert ev is not None
    assert ev.t == "device"
    assert ev.gpus == (T4, "NVIDIA L4, 23034 MiB")
    ev = protocol.parse_line('::gpu:: {"t":"device","gpus":["A\\u001b[31mB' + "x" * 300 + '", 5]}')
    assert ev is not None
    assert ev.gpus[0].startswith("A[31mB")
    assert len(ev.gpus[0]) == protocol.DEVICE_CHARS
    many = protocol.parse_line(protocol.device([T4] * 40))
    assert many is not None
    assert len(many.gpus) == protocol.MAX_DEVICES
    for bad in ('{"t":"device"}', '{"t":"device","gpus":"T4"}', '{"t":"device","gpus":[1]}'):
        assert protocol.parse_line("::gpu:: " + bad) is None


def test_capture_reports_devices(paths: Paths) -> None:
    cap = LogCapture("j1", 1, paths.job_log("j1", 1), paths.job_metrics("j1"), helper_seen=False)
    cap.open(0)
    try:
        lines = [protocol.hello("bootstrap/0.3"), protocol.device([T4]), "hello from the job"]
        result = cap.ingest([LogChunk(lines=lines, cursor="3")], now=1.0)
    finally:
        cap.close()
    assert result.devices == (T4,)
    assert result.lines_added == 3


async def test_another_gpu_than_placed_is_noted_and_named_when_done(eng: Engine) -> None:
    job = await eng.submit(gpu="A100-40GB", fake={"device": [T4]})
    done = await eng.until_terminal(job.id)
    assert done.state is JobState.DONE
    assert done.gpu == "A100-40GB"  # what it was placed (and priced) on
    notes = [e for e in eng.store.events_for(job.id) if e.reason == Reason.GPU_MISMATCH]
    assert len(notes) == 1  # once per attempt, however many polls saw the line
    (attempt,) = eng.store.attempts_for(job.id)
    assert notes[0].attempt_id == attempt.id
    assert notes[0].detail["placed"] == "A100-40GB"
    assert notes[0].detail["seen"] == [T4]
    assert "gave this run a Tesla T4, not the A100-40GB it was placed on" in notes[0].message
    assert "ran on a Tesla T4, not the A100-40GB it was placed on" in done.message
    assert done.message.startswith(f"finished on {attempt.provider}")


async def test_the_placed_gpu_is_not_noted(eng: Engine) -> None:
    job = await eng.submit(gpu="T4", fake={"device": [T4]})
    done = await eng.until_terminal(job.id)
    assert done.state is JobState.DONE
    assert Reason.GPU_MISMATCH not in eng.reasons(job.id)
    assert "ran on" not in done.message


async def test_no_device_line_no_check(eng: Engine) -> None:
    job = await eng.submit(gpu="A100-40GB")
    done = await eng.until_terminal(job.id)
    assert done.state is JobState.DONE
    assert Reason.GPU_MISMATCH not in eng.reasons(job.id)


async def test_status_detail_shows_the_gpu_seen(eng: Engine) -> None:
    from rich.console import Console

    from gpu_router.api import JobDetail, JobView
    from gpu_router.cli import render

    job = await eng.submit(gpu="A100-40GB", fake={"device": [T4]})
    await eng.until_terminal(job.id)

    def shown(job_id: str) -> str:
        detail = JobDetail(
            job=JobView.of(eng.store.get_job(job_id)),
            attempts=eng.store.attempts_for(job_id),
            checkpoints=[],
            events=eng.store.events_for(job_id),
        )
        console = Console(record=True, width=200)
        render.print_detail(console, detail, eng.clock.now())
        return console.export_text()

    text = shown(job.id)
    (row,) = [line for line in text.splitlines() if line.startswith("gpu seen")]
    assert row.split(None, 2)[2].strip() == "Tesla T4 (not the A100-40GB it was placed on)"
    assert "ran on a Tesla T4" in text  # status row = the done message

    ok = await eng.submit(gpu="T4", fake={"device": [T4]})
    await eng.until_terminal(ok.id)
    assert "gpu seen" not in shown(ok.id)
