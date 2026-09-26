"""Tail cursors and the tail -> final log alignment (checkpoint/sidechannel.py)."""

from __future__ import annotations

import pytest

from gpu_router.checkpoint import set_active_hub
from gpu_router.checkpoint.sidechannel import (
    align,
    format_cursor,
    line_hash,
    parse_cursor,
    preamble,
    read_tail,
)

HELLO = '::gpu:: {"t":"hello","v":1,"runner":"bootstrap/0.3"}'


def test_cursor_roundtrip() -> None:
    assert parse_cursor(format_cursor(12, "abc")) == (12, "abc")
    assert parse_cursor(format_cursor(0, None)) == (0, None)
    assert parse_cursor("12") is None
    assert parse_cursor(None) is None
    assert parse_cursor("tx:1") is None


def test_align_uses_the_hello_line_and_the_last_line_hash() -> None:
    final = ["kaggle preamble", "resume note", HELLO, "a", "b", "c", "d"]
    assert preamble(final) == ["kaggle preamble", "resume note"]
    assert align(final, 2, line_hash("a")) == 4  # hello + 2 lines served
    # an extra provider line crept in: the hash finds the real position
    shifted = ["kaggle preamble", HELLO, "a", "platform noise", "b", "c"]
    assert align(shifted, 3, line_hash("b")) == 5
    # unknown hash (a redacted or truncated line): fall back to the count
    assert align(final, 3, "0000000000") == 5
    assert align(final, 0, None) == 0
    assert align(["no hello here", "x"], 5, None) == 2  # never past the end


def test_read_tail_without_a_hub_is_none() -> None:
    set_active_hub(None)
    assert read_tail("file:///nowhere/jobs/j/attempts/1", 0) is None


@pytest.mark.parametrize("consumed", [0, 3])
def test_read_tail_serves_only_new_lines(
    consumed: int, paths: object, clock: object, tmp_path: object
) -> None:
    from gpu_router.checkpoint.hub import CheckpointHub
    from gpu_router.config import CheckpointConfig
    from gpu_router.runner import storage as rs

    hub = CheckpointHub(CheckpointConfig(), paths, clock, test_mode=True)  # type: ignore[arg-type]
    set_active_hub(hub)
    try:
        local = hub.local()
        assert local is not None
        lines = [HELLO, "one", "two", "three", '::gpu:: {"t":"exit","code":0}']
        local.write_bytes(
            rs.log_tail_key("j", 1), rs.dumps({"first": 0, "total": 5, "lines": lines})
        )
        got = read_tail(f"{local.root_uri}/jobs/j/attempts/1", consumed)
        assert got is not None
        assert got.lines == lines[consumed:]
        assert got.consumed == 5
        assert got.last_hash == line_hash(lines[-1])
        assert got.ended
        again = read_tail(f"{local.root_uri}/jobs/j/attempts/1", 5, got.last_hash)
        assert again is not None
        assert again.lines == []
        assert read_tail("hf://buckets/x/y/jobs/j/attempts/1", 0) is None  # not ours
    finally:
        set_active_hub(None)
        hub.close()
