"""HTTP API v1 through the ASGI app: auth, envelope, idempotency, jobs, logs, providers."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import httpx

from gpu_router import __version__
from gpu_router.api import VERSION_HEADER, EventList, JobDetail, JobList, JobView, StatusView
from gpu_router.daemon import routes
from tests.api.conftest import BASE_URL, Api

# --------------------------------------------------------------------------- guard


async def test_health_is_public_and_versioned(api: Api) -> None:
    resp = await api.client.get("/v1/health", headers={"Authorization": ""})
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"]
    assert body["ready"]
    assert body["test_mode"]
    assert body["pid"] == os.getpid()
    assert body["started_at"].endswith("Z")
    assert resp.headers[VERSION_HEADER] == __version__
    # the daemon's own notification backend (doctor reports it; review fix)
    assert body["notifications"] in (None, "off: running under pytest")


async def test_missing_or_bad_token_is_401(api: Api) -> None:
    for auth in ("", "Bearer nope", "Basic abc", f"bearer {api.runtime.token}x"):
        resp = await api.client.get("/v1/status", headers={"Authorization": auth})
        assert resp.status_code == 401, auth
        assert resp.json()["error"]["code"] == "unauthorized"
        assert resp.headers[VERSION_HEADER]


async def test_token_file_is_0600(api: Api) -> None:
    mode = stat.S_IMODE(api.runtime.paths.token.stat().st_mode)
    assert mode == 0o600
    assert api.runtime.paths.token.read_text().strip() == api.runtime.token


async def test_foreign_host_and_browser_origin_are_403(api: Api) -> None:
    resp = await api.client.get("/v1/health", headers={"Host": "evil.example:47291"})
    assert resp.status_code == 403
    assert resp.json()["error"]["code"] == "forbidden"
    resp = await api.client.get("/v1/status", headers={"Origin": "http://evil.example"})
    assert resp.status_code == 403
    resp = await api.client.get("/v1/health", headers={"Host": "localhost:47291"})
    assert resp.status_code == 200
    resp = await api.client.get("/v1/health", headers={"Host": "127.0.0.1:1234"})
    assert resp.status_code == 403


async def test_mutations_are_503_while_not_ready(api: Api) -> None:
    api.runtime.supervisor.ready = False
    try:
        resp = await api.client.post("/v1/jobs", json={"spec": api.spec()})
        assert resp.status_code == 503
        assert resp.json()["error"]["code"] == "not_ready"
        assert (await api.client.get("/v1/status")).status_code == 200
    finally:
        api.runtime.supervisor.ready = True


async def test_unknown_route_uses_envelope(api: Api) -> None:
    resp = await api.client.get("/v1/nope")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "invalid_request"


# --------------------------------------------------------------------------- submit


async def test_submit_runs_to_done_and_lists(api: Api) -> None:
    job = JobView.model_validate(await api.submit())
    assert job.state == "queued"
    await api.drive(lambda: api.state(job.id) == "done")

    detail = JobDetail.model_validate((await api.client.get(f"/v1/jobs/{job.short_id}")).json())
    assert detail.job.state == "done"
    assert detail.attempts[0].state == "succeeded"
    assert detail.route is not None
    assert detail.route.chosen is not None
    assert detail.route.chosen.provider == "fake"
    assert [e.reason for e in detail.events][:2] == ["submitted", "routing_started"]
    assert (Path(api.project) / "runs" / job.id[:4] / "result.json").exists()

    listing = JobList.model_validate((await api.client.get("/v1/jobs")).json())
    assert [j.id for j in listing.jobs] == [job.id]
    done_only = await api.client.get("/v1/jobs", params={"state": "done,failed"})
    assert len(done_only.json()["jobs"]) == 1
    none = await api.client.get("/v1/jobs", params=[("state", "queued")])
    assert none.json()["jobs"] == []
    bad = await api.client.get("/v1/jobs", params={"state": "bogus"})
    assert bad.status_code == 400

    events = EventList.model_validate(
        (await api.client.get(f"/v1/jobs/{job.id}/events", params={"after": 0})).json()
    )
    assert events.events[-1].reason == "completed"
    assert events.next == events.events[-1].seq
    again = await api.client.get(f"/v1/jobs/{job.id}/events", params={"after": events.next})
    assert again.json() == {"events": [], "next": events.next}


async def test_submit_actor_from_client_header(api: Api) -> None:
    resp = await api.client.post(
        "/v1/jobs", json={"spec": api.spec()}, headers={"X-Gpu-Router-Client": "mcp"}
    )
    job_id = resp.json()["id"]
    first = api.runtime.store.events_for(job_id)[0]
    assert first.actor == "agent"
    assert routes.actor_for("shell") == "user:shell"
    assert routes.actor_for(None) == "api"
    assert routes.actor_for("Bad Header!") == "api"


async def test_idempotency_key_replays(api: Api) -> None:
    headers = {"Idempotency-Key": "k-1"}
    first = await api.client.post("/v1/jobs", json={"spec": api.spec()}, headers=headers)
    second = await api.client.post("/v1/jobs", json={"spec": api.spec()}, headers=headers)
    assert (first.status_code, second.status_code) == (201, 200)
    assert first.json()["id"] == second.json()["id"]
    other = await api.client.post(
        "/v1/jobs", json={"spec": api.spec(script="other.py")}, headers=headers
    )
    assert other.status_code == 409
    assert other.json()["error"]["code"] == "conflict"
    assert "different job" in other.json()["error"]["message"]
    assert len(api.runtime.store.list_jobs()) == 1


async def test_invalid_spec_and_request(api: Api) -> None:
    resp = await api.client.post("/v1/jobs", json={"spec": api.spec(project_dir="relative")})
    assert resp.status_code == 400
    err = resp.json()["error"]
    assert err["code"] == "invalid_spec"
    assert "project_dir" in err["message"]
    assert err["detail"]["errors"]
    resp = await api.client.post("/v1/jobs", json={"spec": api.spec(env={"API_TOKEN": "x"})})
    assert resp.json()["error"]["code"] == "invalid_spec"
    resp = await api.client.post("/v1/jobs", json={"nope": 1})
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_request"
    resp = await api.client.post(
        "/v1/jobs", content=b"{not json", headers={"Content-Type": "application/json"}
    )
    assert resp.status_code == 400


async def test_job_ref_errors(api: Api) -> None:
    resp = await api.client.get("/v1/jobs/ffff")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "job_not_found"
    resp = await api.client.get("/v1/jobs/xyz!")
    assert resp.status_code == 400


# --------------------------------------------------------------------------- actions


async def test_cancel_approve_deny_fetch(api: Api) -> None:
    long = await api.submit(provider_options={"fake": {"duration": 100}})
    await api.drive(lambda: api.state(long["id"]) == "running")
    resp = await api.client.post(f"/v1/jobs/{long['id']}/cancel")
    assert resp.status_code == 200
    assert resp.json()["state"] == "cancelling"
    await api.drive(lambda: api.state(long["id"]) == "cancelled")
    again = await api.client.post(f"/v1/jobs/{long['id']}/cancel")
    assert again.json()["state"] == "cancelled"

    gated = await api.submit(requires_approval=True)
    await api.drive(lambda: api.state(gated["id"]) == "awaiting_approval")
    ok = await api.client.post(f"/v1/jobs/{gated['id']}/approve", json={"reason": "fine"})
    assert ok.status_code == 200
    assert ok.json()["approved_by"] == "user:cli"
    await api.drive(lambda: api.state(gated["id"]) == "done")
    bad = await api.client.post(f"/v1/jobs/{gated['id']}/approve")
    assert bad.status_code == 409
    assert bad.json()["error"]["code"] == "invalid_transition"

    fetched = await api.client.post(f"/v1/jobs/{gated['id']}/fetch")
    assert fetched.status_code == 200
    await api.drive(
        lambda: any(
            e.reason == "fetched" and e.actor == "engine"
            for e in api.runtime.store.events_for(gated["id"])[-2:]
        )
    )

    denied = await api.submit(requires_approval=True)
    await api.drive(lambda: api.state(denied["id"]) == "awaiting_approval")
    resp = await api.client.post(f"/v1/jobs/{denied['id']}/deny", json={"reason": "no"})
    assert resp.json()["state"] == "denied"
    resp = await api.client.post(f"/v1/jobs/{long['id']}/fetch")
    assert resp.status_code == 409


# --------------------------------------------------------------------------- logs


async def test_logs_stream_and_resume(api: Api) -> None:
    job = await api.submit(provider_options={"fake": {"duration": 3, "steps": 3}})
    await api.drive(lambda: api.state(job["id"]) == "done")
    resp = await api.client.get(f"/v1/jobs/{job['id']}/logs")
    assert resp.headers["content-type"].startswith("application/x-ndjson")
    records = [json.loads(line) for line in resp.text.splitlines()]
    assert records
    assert all(r["attempt"] == 1 for r in records)
    assert not any(r["line"].startswith("::gpu::") for r in records)
    offsets = [r["offset"] for r in records]
    assert offsets == sorted(offsets)
    resumed = await api.client.get(
        f"/v1/jobs/{job['id']}/logs", params={"attempt": 1, "offset": offsets[-1]}
    )
    assert [json.loads(x)["offset"] for x in resumed.text.splitlines()] == [offsets[-1]]
    with_proto = await api.client.get(f"/v1/jobs/{job['id']}/logs", params={"protocol": True})
    assert any(json.loads(x)["line"].startswith("::gpu::") for x in with_proto.text.splitlines())
    follow = await api.client.get(f"/v1/jobs/{job['id']}/logs", params={"follow": True})
    last = json.loads(follow.text.splitlines()[-1])
    assert last == {"eof": True, "state": "done"}
    missing = await api.client.get(f"/v1/jobs/{job['id']}/logs", params={"attempt": 9})
    assert missing.status_code == 400


async def test_logs_follow_streams_live_until_eof(api: Api, monkeypatch: object) -> None:
    import asyncio

    import pytest

    mp = pytest.MonkeyPatch()
    mp.setattr(routes, "HEARTBEAT_S", 0.05)
    try:
        job = await api.submit(provider_options={"fake": {"duration": 4, "steps": 4}})
        lines: list[dict[str, object]] = []

        async def reader() -> None:
            async with api.client.stream(
                "GET", f"/v1/jobs/{job['id']}/logs", params={"follow": True}
            ) as resp:
                async for line in resp.aiter_lines():
                    if line:
                        lines.append(json.loads(line))

        task = asyncio.create_task(reader())
        await api.drive(lambda: api.state(job["id"]) == "done")
        await asyncio.wait_for(task, 10)
        assert lines[-1] == {"eof": True, "state": "done"}
        assert any("line" in rec for rec in lines)
    finally:
        mp.undo()


# --------------------------------------------------------------------------- misc


async def test_status_route_providers_quota_events(api: Api) -> None:
    await api.submit(provider_options={"fake": {"duration": 100}})
    status = StatusView.model_validate((await api.client.get("/v1/status")).json())
    assert status.ready
    assert len(status.active) == 1
    assert {p.name for p in status.providers} >= {"fake", "fake-b", "kaggle"}

    route = await api.client.post("/v1/route", json={"spec": api.spec(vram_gb=30)})
    assert route.status_code == 200
    assert route.json()["chosen"]["provider"] == "fake-b"
    assert len(api.runtime.store.list_jobs()) == 1

    providers = (await api.client.get("/v1/providers")).json()
    assert {"fake", "fake-b"} <= {p["name"] for p in providers if p["enabled"]}
    one = await api.client.get("/v1/providers/fake")
    assert one.json()["kind"] == "fake"
    assert (await api.client.get("/v1/providers/nope")).status_code == 404
    hc = await api.client.post("/v1/providers/fake/healthcheck")
    assert hc.json()["health"] == "ok"
    assert (await api.client.post("/v1/providers/kaggle/healthcheck")).status_code == 404

    quota = (await api.client.get("/v1/quota")).json()
    assert {q["provider"] for q in quota} == {"fake", "fake-b"}
    assert all(q["source"] == "live" for q in quota)

    feed = (await api.client.get("/v1/events", params={"after": 0})).json()
    assert feed["events"]
    assert feed["next"] == feed["events"][-1]["seq"]
    empty = await api.client.get(
        "/v1/events", params={"after": feed["next"] + 10_000, "timeout": 0.05}
    )
    assert empty.json()["events"] == []


async def test_long_poll_wakes_on_new_event(api: Api) -> None:
    import asyncio

    after = api.runtime.store.max_event_seq()
    poll = asyncio.create_task(api.client.get("/v1/events", params={"after": after, "timeout": 10}))
    await asyncio.sleep(0.05)
    assert not poll.done()
    await api.submit()
    resp = await asyncio.wait_for(poll, 5)
    assert resp.json()["events"][0]["reason"] == "submitted"


async def test_shutdown_route_calls_hook(api: Api) -> None:
    called: list[bool] = []
    api.runtime.shutdown_hook = lambda: called.append(True)
    resp = await api.client.post("/v1/daemon/shutdown")
    assert resp.status_code == 202
    assert called == [True]


async def test_internal_error_is_enveloped(api: Api) -> None:
    def boom() -> object:
        raise RuntimeError("kaboom")

    api.runtime.supervisor.provider_views = boom  # type: ignore[method-assign]
    resp = await api.client.get("/v1/providers")
    assert resp.status_code == 500
    assert resp.json()["error"] == {
        "code": "internal",
        "message": "internal error; see the daemon log",
        "hint": None,
        "detail": {"error": "RuntimeError"},
    }


async def test_second_runtime_is_refused(api: Api) -> None:
    import pytest

    from gpu_router.clock import FakeClock
    from gpu_router.errors import DaemonAlreadyRunning

    with pytest.raises(DaemonAlreadyRunning):
        type(api.runtime).create(
            api.runtime.paths, api.runtime.config, FakeClock(), configure_logging=False
        )


def test_base_url_constant() -> None:
    assert httpx.URL(BASE_URL).host == "127.0.0.1"
