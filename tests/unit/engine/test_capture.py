from __future__ import annotations

import json
from pathlib import Path

from gpu_router import protocol
from gpu_router.adapters.base import LogChunk
from gpu_router.engine.capture import LogCapture, read_log_lines


def _cap(tmp_path: Path, *, helper_seen: bool = False) -> LogCapture:
    cap = LogCapture(
        "a" * 12,
        1,
        tmp_path / "logs" / "attempt-1.log",
        tmp_path / "metrics.jsonl",
        helper_seen=helper_seen,
    )
    cap.open(0)
    return cap


def test_protocol_lines_drive_progress_and_checkpoints(tmp_path: Path) -> None:
    cap = _cap(tmp_path)
    lines = [
        protocol.hello("fake/1"),
        protocol.total(100),
        "epoch 1 loss=9.9",  # ignored once the helper spoke
        protocol.metric(10, {"loss": 0.5}),
        protocol.ckpt_begin(1),
        protocol.ckpt_end(1, "fake://x/ckpt-1", step=10),
        protocol.exit_line(0),
    ]
    res = cap.ingest(
        [LogChunk(lines=lines[:3], cursor="3"), LogChunk(lines=lines[3:], cursor="7", eof=True)],
        now=5.0,
    )
    assert res.lines_added == 7
    assert res.cursor == "7"
    assert res.eof
    assert (res.step, res.total, res.progress_source) == (10, 100, "helper")
    assert res.metrics == {"loss": 0.5}
    assert [e.t for e in res.checkpoint_events] == ["ckpt_begin", "ckpt_end"]
    assert res.exit_code == 0
    points = [json.loads(x) for x in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert points == [{"ts": 5.0, "attempt": 1, "step": 10, "metrics": {"loss": 0.5}}]
    cap.close()
    visible = read_log_lines(cap.log_path)
    assert [i for i, _ in visible] == [2]  # protocol lines hidden, indices kept
    assert len(read_log_lines(cap.log_path, include_protocol=True)) == 7


def test_stdout_fallback_until_helper_seen(tmp_path: Path) -> None:
    cap = _cap(tmp_path)
    res = cap.ingest([LogChunk(lines=["step 3/10 loss: 1.25"], cursor="1")], now=1.0)
    assert (res.step, res.total, res.progress_source) == (3, 10, "stdout")
    assert res.metrics == {"loss": 1.25}
    cap.close()
    cap2 = _cap(tmp_path / "b", helper_seen=True)
    res2 = cap2.ingest([LogChunk(lines=["step 3/10 loss: 1.25"], cursor="1")], now=1.0)
    assert not res2.has_progress


def test_lines_are_redacted(tmp_path: Path) -> None:
    cap = _cap(tmp_path)
    token = "hf_" + "a" * 34
    cap.ingest([LogChunk(lines=[f"using token {token}"], cursor="1")], now=0)
    cap.close()
    text = cap.log_path.read_text()
    assert token not in text
    assert "***" in text


def test_open_truncates_to_persisted_lines(tmp_path: Path) -> None:
    cap = _cap(tmp_path)
    cap.ingest([LogChunk(lines=["one", "two", "three"], cursor="3")], now=0)
    cap.close()
    again = LogCapture("a" * 12, 1, cap.log_path, cap.metrics_path, helper_seen=False)
    again.open(2)
    assert again.lines == 2
    again.ingest([LogChunk(lines=["three"], cursor="3")], now=0)
    again.close()
    assert [t for _, t in read_log_lines(cap.log_path)] == ["one", "two", "three"]


def test_read_log_lines_offset_and_missing_file(tmp_path: Path) -> None:
    assert read_log_lines(tmp_path / "nope.log") == []
    p = tmp_path / "x.log"
    p.write_text("a\nb\nc\npartial")
    assert read_log_lines(p, offset=1) == [(1, "b"), (2, "c")]


def test_embedded_newlines_stay_one_line(tmp_path: Path) -> None:
    cap = _cap(tmp_path)
    res = cap.ingest([LogChunk(lines=["a\nb"], cursor="1")], now=0)
    cap.close()
    assert res.lines_added == 1
    assert read_log_lines(cap.log_path) == [(0, "a b")]
