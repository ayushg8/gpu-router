"""The ::gpu:: line protocol and the stdout metric fallback (protocol.py). Owner: group B."""

from __future__ import annotations

import pytest

from gpu_router import protocol
from gpu_router.protocol import StdoutMetricParser, parse_line


def test_round_trip_every_event() -> None:
    assert parse_line(protocol.hello("bootstrap/0.1")) == protocol.ProtocolEvent(
        t="hello", version=1, runner="bootstrap/0.1"
    )
    assert parse_line(protocol.total(1000)) == protocol.ProtocolEvent(t="total", total=1000)
    m = parse_line(protocol.metric(120, {"loss": 0.412, "lr": 3e-4}, total_steps=1000))
    assert m is not None
    assert (m.t, m.step, m.total, m.metrics) == ("metric", 120, 1000, {"loss": 0.412, "lr": 3e-4})
    assert parse_line(protocol.ckpt_begin(3)) == protocol.ProtocolEvent(t="ckpt_begin", seq=3)
    end = parse_line(protocol.ckpt_end(3, "hf://x/y", step=1200, size=5, sha256="ab"))
    assert end == protocol.ProtocolEvent(
        t="ckpt_end", seq=3, uri="hf://x/y", step=1200, size=5, sha256="ab"
    )
    assert parse_line(protocol.exit_line(2)) == protocol.ProtocolEvent(t="exit", code=2)


def test_format_is_compact_and_drops_none() -> None:
    line = protocol.metric(None, {"loss": 1.0})
    assert line == '::gpu:: {"t":"metric","metrics":{"loss":1.0}}'
    assert protocol.is_protocol_line(line)
    assert not protocol.is_protocol_line("step 1 loss=1")


@pytest.mark.parametrize(
    "line",
    [
        "loss=0.4",
        "::gpu:: not json",
        "::gpu:: [1,2]",
        '::gpu:: {"t":"future_thing"}',
        '::gpu:: {"t":"total","steps":0}',
        '::gpu:: {"t":"total","steps":true}',
        '::gpu:: {"t":"ckpt_begin","seq":0}',
        '::gpu:: {"t":"ckpt_end","seq":1}',
        '::gpu:: {"t":"ckpt_end","seq":1,"uri":""}',
        '::gpu:: {"t":"exit"}',
        "::gpu::{}",
    ],
)
def test_invalid_or_unknown_lines_are_ignored(line: str) -> None:
    assert parse_line(line) is None


def test_metric_values_are_filtered_to_finite_numbers() -> None:
    e = parse_line(
        '::gpu:: {"t":"metric","step":2.0,"metrics":{"loss":1,"ok":true,"name":"x","big":1e999}}'
    )
    assert e is not None
    assert e.step == 2
    assert e.metrics == {"loss": 1.0}


@pytest.mark.parametrize(
    ("line", "step", "total", "metrics"),
    [
        ("loss=0.41", None, None, {"loss": 0.41}),
        ("Loss: 0.5 val_loss=0.6 lr=1e-4", None, None, {"loss": 0.5, "val_loss": 0.6, "lr": 1e-4}),
        ("step 10/100 loss=1.2", 10, 100, {"loss": 1.2}),
        ("Epoch 3/10", 3, 10, {}),
        ("iter=7", 7, None, {}),
        (" 45%|####      | 45/100 [00:01<00:02]", 45, 100, {}),
        ("old\r 50%|#####     | 50/100", 50, 100, {}),
        ("nothing to see here", None, None, {}),
    ],
)
def test_stdout_parser(
    line: str, step: int | None, total: int | None, metrics: dict[str, float]
) -> None:
    got = StdoutMetricParser().feed(line)
    assert (got.step, got.total, got.metrics) == (step, total, metrics)


def test_stdout_parser_remembers_total_and_skips_protocol_lines() -> None:
    p = StdoutMetricParser()
    p.feed("step 1/50")
    later = p.feed("step=2 loss=0.9")
    assert (later.step, later.total) == (2, 50)
    assert p.feed(protocol.metric(3, {"loss": 1.0})).empty
