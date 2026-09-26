"""GpuClient against httpx.MockTransport: envelopes, idempotent retry, NDJSON logs,
discovery."""

from __future__ import annotations

import json
import os
from typing import Any

import httpx
import pytest

from gpu_router.api import RuntimeInfo
from gpu_router.client import GpuClient
from gpu_router.errors import (
    ApiError,
    DaemonUnavailable,
    InvalidSpec,
    JobNotFound,
    SubmitUncertain,
)
from gpu_router.models import JobSpec
from gpu_router.paths import Paths

JOB: dict[str, Any] = {
    "id": "a7f2c19e0b3d",
    "short_id": "a7f2",
    "name": "train",
    "state": "queued",
    "source": "cli",
    "project_dir": "/p",
    "spec": {"project_dir": "/p", "script": "train.py"},
    "spec_hash": "x",
    "created_at": "2026-09-23T12:00:00.000Z",
    "updated_at": "2026-09-23T12:00:00.000Z",
}


def _client(handler: Any) -> GpuClient:
    return GpuClient(
        "http://127.0.0.1:1", "tok", client_name="shell", transport=httpx.MockTransport(handler)
    )


def test_headers_and_envelope_errors() -> None:
    seen: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        if req.url.path == "/v1/jobs/ffff":
            return httpx.Response(
                404,
                json={
                    "error": {
                        "code": "job_not_found",
                        "message": "no job ffff",
                        "hint": "see gpu status",
                        "detail": {},
                    }
                },
            )
        if req.url.path == "/v1/jobs/eeee":
            return httpx.Response(
                409, json={"error": {"code": "invalid_transition", "message": "nope", "detail": {}}}
            )
        return httpx.Response(502, text="bad gateway")

    c = _client(handler)
    with pytest.raises(JobNotFound) as nf:
        c.job("ffff")
    assert nf.value.hint == "see gpu status"
    with pytest.raises(ApiError) as it:
        c.job("eeee")
    assert it.value.raw_code == "invalid_transition"
    assert it.value.status == 409
    with pytest.raises(ApiError) as other:
        c.status()
    assert other.value.raw_code == "internal"
    req = seen[0]
    assert req.headers["authorization"] == "Bearer tok"
    assert req.headers["x-gpu-router-client"] == "shell"


def test_submit_sends_key_and_retries_once_with_same_key() -> None:
    keys: list[str] = []
    calls = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        keys.append(req.headers["idempotency-key"])
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("reset")
        body = json.loads(req.content)
        assert body["spec"]["script"] == "train.py"
        return httpx.Response(201, json=JOB)

    job = _client(handler).submit(JobSpec(project_dir="/p", script="train.py"))
    assert job.id == JOB["id"]
    assert len(keys) == 2
    assert keys[0] == keys[1]


def test_submit_uses_long_timeout_and_read_timeout_is_uncertain_not_retried() -> None:
    """Review finding: the daemon bundles inside POST /v1/jobs; a 10 s timeout plus a
    blind retry told the user the daemon was down while the job ran (and a retry with a
    fresh CLI would double-submit). Now: long timeout, and no answer = SubmitUncertain
    naming the idempotency key."""
    seen: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        raise httpx.ReadTimeout("slow bundle", request=req)

    with pytest.raises(SubmitUncertain) as ei:
        _client(handler).submit(JobSpec(project_dir="/p", script="train.py"))
    assert len(seen) == 1  # not retried: a replay would rebuild the bundle and time out too
    assert seen[0].extensions["timeout"]["read"] >= 300
    key = seen[0].headers["idempotency-key"]
    assert ei.value.detail == {"idempotency_key": key, "maybe_submitted": True}
    assert "may have been submitted" in ei.value.message
    assert "gpu jobs" in (ei.value.hint or "")
    assert "gpu daemon run" not in (ei.value.hint or "")


def test_submit_dropped_connection_after_send_is_uncertain() -> None:
    calls = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        raise httpx.RemoteProtocolError("server disconnected", request=req)

    with pytest.raises(SubmitUncertain):
        _client(handler).submit(JobSpec(project_dir="/p", script="train.py"))
    assert calls["n"] == 1


def test_submit_refused_twice_is_daemon_unavailable() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with pytest.raises(DaemonUnavailable) as ei:
        _client(handler).submit(JobSpec(project_dir="/p", script="train.py"))
    assert not isinstance(ei.value, SubmitUncertain)
    assert "gpu daemon start" in (ei.value.hint or "")


def test_timeout_does_not_claim_the_daemon_is_down() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=req)

    with pytest.raises(DaemonUnavailable) as ei:
        _client(handler).status()
    assert "did not answer within" in ei.value.message
    assert "cannot reach" not in ei.value.message
    assert "gpu daemon status" in (ei.value.hint or "")


def test_submit_invalid_spec_surfaces() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={"error": {"code": "invalid_spec", "message": "bad", "detail": {"errors": []}}},
        )

    with pytest.raises(InvalidSpec):
        _client(handler).submit(JobSpec(project_dir="/p", script="t.py"), idempotency_key="k")


def test_logs_filter_heartbeats() -> None:
    lines = [
        {"attempt": 1, "offset": 0, "line": "hi"},
        {"heartbeat": True},
        {"eof": True, "state": "done"},
    ]

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.params["follow"] == "true"
        body = "".join(json.dumps(x) + "\n" for x in lines)
        return httpx.Response(200, text=body, headers={"content-type": "application/x-ndjson"})

    recs = list(_client(handler).logs("a7f2", follow=True))
    assert [r.line for r in recs] == ["hi", None]
    assert recs[-1].eof
    assert recs[-1].state == "done"


def test_transport_error_is_daemon_unavailable() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with pytest.raises(DaemonUnavailable, match="cannot reach"):
        _client(handler).health()


def test_other_endpoints_parse() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/v1/jobs":
            assert req.url.params.get_list("state") == ["queued", "running"]
            return httpx.Response(200, json={"jobs": [JOB], "next_before": None})
        if req.url.path == "/v1/quota":
            return httpx.Response(200, json=[])
        if req.url.path == "/v1/events":
            assert req.url.params["timeout"] == "1.0"
            return httpx.Response(200, json={"events": [], "next": 7})
        if req.url.path == "/v1/daemon/shutdown":
            return httpx.Response(202, json={"ok": True})
        return httpx.Response(200, json=JOB)

    from gpu_router.models import JobState

    c = _client(handler)
    assert c.jobs(states=[JobState.QUEUED, JobState.RUNNING]).jobs[0].short_id == "a7f2"
    assert c.cancel("a7f2").id == JOB["id"]
    assert c.approve("a7f2", reason="ok").id == JOB["id"]
    assert c.quota() == []
    assert c.wait_events(after=3, timeout_s=1.0).next == 7
    assert c.shutdown() is None


def test_from_env_discovery(paths: Paths) -> None:
    with pytest.raises(DaemonUnavailable, match="not running"):
        GpuClient.from_env(paths)
    dead = RuntimeInfo(pid=999_999, port=1234, version="0.1.0", started_at=0)
    paths.runtime.write_text(dead.model_dump_json())
    with pytest.raises(DaemonUnavailable, match="999999"):
        GpuClient.from_env(paths)
    live = RuntimeInfo(pid=os.getpid(), port=1234, version="0.1.0", started_at=0)
    paths.runtime.write_text(live.model_dump_json())
    with pytest.raises(DaemonUnavailable, match="token"):
        GpuClient.from_env(paths)
    from gpu_router.daemon.auth import ensure_token

    ensure_token(paths)
    client = GpuClient.from_env(paths)
    assert client.base_url == "http://127.0.0.1:1234"
