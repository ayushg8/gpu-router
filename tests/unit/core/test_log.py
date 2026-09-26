from __future__ import annotations

import json
import logging
from collections.abc import Iterator

import pytest

from gpu_router import secrets
from gpu_router.log import (
    JsonFormatter,
    RedactingFilter,
    get_logger,
    log_event,
    setup_daemon_logging,
)
from gpu_router.paths import Paths


@pytest.fixture
def daemon_logging(paths: Paths) -> Iterator[Paths]:
    setup_daemon_logging(paths, level="DEBUG")
    yield paths
    root = logging.getLogger("gpu_router")
    for h in list(root.handlers):
        root.removeHandler(h)
        h.close()
    root.propagate = True
    root.setLevel(logging.NOTSET)


def _lines(paths: Paths) -> list[dict[str, object]]:
    for h in logging.getLogger("gpu_router").handlers:
        h.flush()
    return [json.loads(line) for line in paths.daemon_log.read_text().splitlines()]


def test_json_shape_and_key_order(daemon_logging: Paths) -> None:
    log = get_logger("gpu_router.engine")
    log_event(
        log,
        "job.transition",
        "a7f2 provisioning -> running",
        job_id="a7f2c19e0b3d",
        provider="fake",
        **{"from": "provisioning"},
        to="running",
        reason="started",
    )
    [rec] = _lines(daemon_logging)
    assert list(rec)[:5] == ["ts", "level", "logger", "event", "msg"]
    assert rec["level"] == "info"
    assert rec["logger"] == "gpu_router.engine"
    assert rec["event"] == "job.transition"
    assert rec["msg"] == "a7f2 provisioning -> running"
    assert rec["from"] == "provisioning"
    assert rec["to"] == "running"
    assert str(rec["ts"]).endswith("Z")


def test_redaction_everywhere(daemon_logging: Paths) -> None:
    secrets.register_for_redaction("my-secret-value")
    log = get_logger("gpu_router.x")
    log_event(
        log,
        "adapter.call",
        "cli said my-secret-value",
        detail={"nested": ["Authorization: Bearer abcdef"]},
        note="hf_" + "z" * 34,
    )
    log.info("plain %s", "my-secret-value")
    try:
        raise ValueError("boom my-secret-value")
    except ValueError:
        log_event(log, "adapter.bug", "failed", level=logging.ERROR, exc_info=True)
    text = daemon_logging.daemon_log.read_text()
    assert "my-secret-value" not in text
    assert "abcdef" not in text
    assert "zzzz" not in text
    recs = _lines(daemon_logging)
    assert recs[0]["detail"] == {"nested": ["Authorization: Bearer ***"]}
    assert recs[1]["msg"] == "plain ***"
    assert "ValueError" in str(recs[2]["exc"])


def test_reserved_field_names_are_prefixed() -> None:
    rec = logging.LogRecord("gpu_router.t", logging.INFO, __file__, 1, "hi", None, None)
    rec.event = "e"
    rec.fields = {"msg": "clash", "level": 3, "obj": object(), "tup": (1, 2)}
    out = json.loads(JsonFormatter().format(rec))
    assert out["msg"] == "hi"
    assert out["field_msg"] == "clash"
    assert out["field_level"] == 3
    assert isinstance(out["obj"], str)
    assert out["tup"] == [1, 2]


def test_filter_survives_bad_format_args() -> None:
    rec = logging.LogRecord("gpu_router.t", logging.INFO, __file__, 1, "%d", ("x",), None)
    assert RedactingFilter().filter(rec)
    assert rec.getMessage() == "%d"


def test_setup_is_idempotent_and_levels(daemon_logging: Paths) -> None:
    setup_daemon_logging(daemon_logging, level="INFO", foreground=True)
    root = logging.getLogger("gpu_router")
    ours = [h for h in root.handlers if getattr(h, "_gpu_router_handler", False)]
    assert len(ours) == 2  # file + stderr, not duplicated
    log_event(get_logger("gpu_router.y"), "api.request", "debug only", level=logging.DEBUG)
    log_event(get_logger("gpu_router.y"), "daemon.start", "started")
    events = [r["event"] for r in _lines(daemon_logging)]
    assert events == ["daemon.start"]
    assert logging.getLogger("uvicorn.access").disabled
