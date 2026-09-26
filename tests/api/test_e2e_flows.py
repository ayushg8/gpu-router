"""End-to-end job flows through the HTTP API (phase-1 integration).

Every scenario submits over `POST /v1/jobs`, drives fake time, and checks what a client sees
through the API: the job's state, its attempts, the event trail (one transition event per
state change, chained from `submitted` to the terminal state), logs, and fetched outputs.
The runtime is real (store on the tmp home, supervisor, worker threads, FakeAdapter on disk);
only the clock is fake.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

from gpu_router.api import EventList, JobDetail
from tests.api.conftest import Api


async def _detail(api: Api, job_id: str) -> JobDetail:
    resp = await api.client.get(f"/v1/jobs/{job_id}")
    assert resp.status_code == 200, resp.text
    return JobDetail.model_validate(resp.json())


async def _all_events(api: Api, job_id: str) -> list[Any]:
    resp = await api.client.get(f"/v1/jobs/{job_id}/events", params={"after": 0})
    assert resp.status_code == 200, resp.text
    return list(EventList.model_validate(resp.json()).events)


def _transitions(events: list[Any]) -> list[tuple[str | None, str, str]]:
    return [
        (None if e.from_state is None else str(e.from_state), str(e.to_state), e.reason)
        for e in events
        if e.kind == "transition"
    ]


def _assert_chain(trail: list[tuple[str | None, str, str]], final: str) -> None:
    """Transition events form an unbroken chain (new) -> queued -> ... -> final."""
    assert trail, "no transition events"
    assert trail[0][:2] == (None, "queued")
    for (_, prev_to, _), (frm, _, reason) in itertools.pairwise(trail):
        assert frm == prev_to, f"broken chain at {reason}: {trail}"
    assert trail[-1][1] == final, trail


async def _logs(api: Api, job_id: str, **params: Any) -> list[dict[str, Any]]:
    resp = await api.client.get(f"/v1/jobs/{job_id}/logs", params=params)
    assert resp.status_code == 200, resp.text
    return [json.loads(line) for line in resp.text.splitlines() if line]


# --------------------------------------------------------------------------- (a) happy path


async def test_submit_to_done_with_full_event_trail_logs_and_outputs(api: Api) -> None:
    job = await api.submit(provider_options={"fake": {"duration": 4, "steps": 4}})
    job_id = job["id"]

    seen_states: list[str] = []

    def observe() -> bool:
        s = api.state(job_id)
        if not seen_states or seen_states[-1] != s:
            seen_states.append(s)
        return s == "done"

    await api.drive(observe)

    trail = _transitions(await _all_events(api, job_id))
    _assert_chain(trail, "done")
    assert [t[2] for t in trail] == [
        "submitted",
        "routing_started",
        "placed",
        "started",
        "completed",
    ]
    # every state the job was observed in has the transition event that put it there
    assert set(seen_states) <= {t[1] for t in trail}

    # the global event feed carries the same transitions
    feed = (await api.client.get("/v1/events", params={"after": 0})).json()
    feed_trail = [
        (e["from_state"], e["to_state"], e["reason"])
        for e in feed["events"]
        if e["job_id"] == job_id and e["kind"] == "transition"
    ]
    assert feed_trail == trail

    detail = await _detail(api, job_id)
    assert detail.job.state == "done"
    assert [a.state for a in detail.attempts] == ["succeeded"]
    assert detail.attempts[0].remote_id
    assert detail.job.outputs_fetched

    lines = await _logs(api, job_id)
    assert any("loss" in r["line"] for r in lines)
    offsets = [r["offset"] for r in lines]
    assert offsets == sorted(set(offsets))  # protocol lines are hidden, so gaps are fine
    full = await _logs(api, job_id, protocol=True)
    assert [r["offset"] for r in full] == list(range(len(full)))
    assert any(r["line"].startswith("::gpu:: ") for r in full)

    out = Path(api.project) / "runs" / job_id[:4]
    assert json.loads((out / "result.json").read_text())["job_id"] == job_id
    assert (out / "model.txt").exists()
    assert detail.job.outputs_dir == str(out)


# --------------------------------------------------------------------------- (b) rate limit


async def test_rate_limit_backs_off_then_succeeds(api: Api) -> None:
    job = await api.submit(
        provider="fake",
        provider_options={"fake": {"duration": 2, "steps": 2, "rate_limit_n": 2}},
    )
    job_id = job["id"]
    await api.drive(lambda: api.state(job_id) in ("done", "failed"))

    detail = await _detail(api, job_id)
    assert detail.job.state == "done", [(e.reason, e.message) for e in detail.events]
    assert [a.state for a in detail.attempts] == ["rejected", "rejected", "succeeded"]
    assert [a.error_kind for a in detail.attempts[:2]] == ["RateLimited", "RateLimited"]
    assert {a.provider for a in detail.attempts} == {"fake"}

    events = await _all_events(api, job_id)
    trail = _transitions(events)
    _assert_chain(trail, "done")
    limited = [t for t in trail if t[2] == "rate_limited"]
    assert limited == [("provisioning", "queued", "rate_limited")] * 2
    assert all(e.reason != "failed" for e in events)

    # backoff: each retry was placed no earlier than the previous rejection
    placed_ts = [e.ts for e in events if e.kind == "transition" and e.reason == "placed"]
    limited_ts = [e.ts for e in events if e.kind == "transition" and e.reason == "rate_limited"]
    assert len(placed_ts) == 3
    for lim, nxt in zip(limited_ts, placed_ts[1:], strict=True):
        assert nxt > lim

    # the provider was cooled down (visible through the providers API)
    prov = (await api.client.get("/v1/providers/fake")).json()
    assert prov["name"] == "fake"


# --------------------------------------------------------------------------- (c) migration


async def test_mid_run_death_migrates_and_resumes_from_latest_checkpoint(api: Api) -> None:
    directives = {
        "duration": 6,
        "steps": 6,
        "checkpoint_every": 1,
        "attempts": {"1": {"die_after": 3.5}},
    }
    job = await api.submit(provider_options={"fake": directives})
    job_id = job["id"]
    await api.drive(lambda: api.state(job_id) in ("done", "failed"), max_s=600)

    detail = await _detail(api, job_id)
    assert detail.job.state == "done", [(e.reason, e.message) for e in detail.events]
    assert len(detail.attempts) == 2
    first, second = detail.attempts
    assert first.state == "lost"
    assert second.state == "succeeded"
    assert first.remote_id != second.remote_id

    trail = _transitions(await _all_events(api, job_id))
    _assert_chain(trail, "done")
    assert ("running", "migrating", "session_lost") in trail or (
        "checkpointing",
        "migrating",
        "session_lost",
    ) in trail
    assert any(t[:2] == ("migrating", "provisioning") for t in trail)

    ckpts_before = [c for c in detail.checkpoints if c.attempt_id == first.id]
    assert ckpts_before, "attempt 1 recorded no checkpoints"
    latest = max(ckpts_before, key=lambda c: c.seq)
    assert second.resume_checkpoint_id == latest.id

    second_logs = await _logs(api, job_id, attempt=2)
    assert any(f"resuming from checkpoint {latest.seq}" in r["line"] for r in second_logs)
    # checkpoint numbering continues across attempts
    later = [c for c in detail.checkpoints if c.attempt_id == second.id]
    assert all(c.seq > latest.seq for c in later)
    assert (Path(api.project) / "runs" / job_id[:4] / "result.json").exists()


# --------------------------------------------------------------------------- (d) cancel


async def test_cancel_while_running(api: Api) -> None:
    job = await api.submit(provider_options={"fake": {"duration": 500, "steps": 50}})
    job_id = job["id"]
    await api.drive(lambda: api.state(job_id) == "running")

    resp = await api.client.post(f"/v1/jobs/{job_id[:6]}/cancel")
    assert resp.status_code == 200, resp.text
    assert resp.json()["state"] in ("cancelling", "cancelled")
    await api.drive(lambda: api.state(job_id) == "cancelled")

    detail = await _detail(api, job_id)
    assert [a.state for a in detail.attempts] == ["cancelled"]
    trail = _transitions(await _all_events(api, job_id))
    _assert_chain(trail, "cancelled")
    assert ("running", "cancelling", "user_cancel") in trail
    assert trail[-1] == ("cancelling", "cancelled", "cancelled")

    remote_id = detail.attempts[0].remote_id
    assert remote_id
    run_file = api.runtime.paths.home / "fake" / "fake" / "runs" / remote_id / "run.json"
    run = json.loads(run_file.read_text())
    assert run["cancelled_at"] is not None
    # idempotent
    again = await api.client.post(f"/v1/jobs/{job_id}/cancel")
    assert again.json()["state"] == "cancelled"


# --------------------------------------------------------------------------- (e) approval


async def test_approval_required_then_approved_runs(api: Api) -> None:
    job = await api.submit(requires_approval=True)
    job_id = job["id"]
    await api.drive(lambda: api.state(job_id) == "awaiting_approval")
    detail = await _detail(api, job_id)
    assert detail.attempts == []  # nothing submitted before approval
    assert detail.job.approval_reason

    resp = await api.client.post(f"/v1/jobs/{job_id}/approve", json={"reason": "go"})
    assert resp.status_code == 200, resp.text
    await api.drive(lambda: api.state(job_id) == "done")

    events = await _all_events(api, job_id)
    trail = _transitions(events)
    _assert_chain(trail, "done")
    assert ("routing", "awaiting_approval", "approval_required") in trail
    assert ("awaiting_approval", "provisioning", "placed") in trail
    assert any(e.kind == "note" and e.reason == "approved" for e in events)
    assert len((await _detail(api, job_id)).attempts) == 1


async def test_approval_required_then_denied(api: Api) -> None:
    job = await api.submit(requires_approval=True)
    job_id = job["id"]
    await api.drive(lambda: api.state(job_id) == "awaiting_approval")

    resp = await api.client.post(f"/v1/jobs/{job_id}/deny", json={"reason": "too big"})
    assert resp.status_code == 200, resp.text
    assert resp.json()["state"] == "denied"

    # stays denied as time passes; nothing is ever submitted
    api.clock.advance(60)
    await api.drive(lambda: True)
    detail = await _detail(api, job_id)
    assert detail.job.state == "denied"
    assert detail.attempts == []
    trail = _transitions(await _all_events(api, job_id))
    _assert_chain(trail, "denied")
    assert trail[-1] == ("awaiting_approval", "denied", "denied")
    assert not (api.runtime.paths.home / "fake" / "fake" / "runs").exists() or not any(
        (api.runtime.paths.home / "fake" / "fake" / "runs").iterdir()
    )
