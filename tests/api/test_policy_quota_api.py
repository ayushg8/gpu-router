"""Phase 5 over HTTP: /v1/policy (rules drive approvals, persisted to config.yaml), the
quota ledger behind /v1/quota, and the scoring router's estimate in /v1/route."""

from __future__ import annotations

from pathlib import Path

import yaml

from gpu_router.api import PolicyView
from gpu_router.policy import PolicyConfig
from gpu_router.statemachine import Reason
from tests.api.conftest import Api


def _config_text(api: Api) -> str:
    path = Path(api.runtime.paths.config)
    return path.read_text() if path.exists() else ""


def _config_mode(api: Api) -> int:
    return Path(api.runtime.paths.config).stat().st_mode & 0o777


def _write(path: Path, text: str) -> None:
    path.write_text(text)


async def test_get_policy_defaults(api: Api) -> None:
    resp = await api.client.get("/v1/policy")
    assert resp.status_code == 200
    view = PolicyView.model_validate(resp.json())
    assert (view.name, view.editable) == ("rules", True)
    assert view.policy == PolicyConfig()
    assert view.defaults == PolicyConfig()
    assert view.config_path == str(api.runtime.paths.config)


async def test_put_policy_persists_and_applies(api: Api) -> None:
    doc = PolicyConfig().model_dump(mode="json")
    doc["agent"]["auto_max_hours"] = 3
    resp = await api.client.put("/v1/policy", json=doc)
    assert resp.status_code == 200, resp.text
    assert resp.json()["policy"]["agent"]["auto_max_hours"] == 3
    saved = yaml.safe_load(_config_text(api))
    assert saved["policy"]["agent"]["auto_max_hours"] == 3
    assert saved["version"] == 1
    assert _config_mode(api) == 0o600
    again = await api.client.get("/v1/policy")
    assert again.json()["policy"]["agent"]["auto_max_hours"] == 3


async def test_put_policy_rejects_bad_documents(api: Api) -> None:
    bad = PolicyConfig().model_dump(mode="json")
    bad["agent"]["auto_max_hourz"] = 3
    resp = await api.client.put("/v1/policy", json=bad)
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] in {"invalid_request", "invalid_spec"}
    assert "auto_max_hourz" not in _config_text(api)
    unauth = await api.client.put(
        "/v1/policy", json=PolicyConfig().model_dump(mode="json"), headers={"Authorization": ""}
    )
    assert unauth.status_code == 401


async def test_agent_rules_gate_long_jobs_and_follow_edits(api: Api) -> None:
    long_agent = await api.submit(source="agent", hours=2)
    await api.drive(lambda: api.state(long_agent["id"]) == "awaiting_approval")
    job = api.runtime.store.get_job(long_agent["id"])
    assert job.approval_reason is not None
    assert "over the 1h auto limit" in job.approval_reason
    events = api.runtime.store.events_for(long_agent["id"])
    asked = [e for e in events if e.reason == Reason.APPROVAL_REQUIRED]
    assert asked[-1].detail["rule"] == "hours"

    # the same 2h job from the user runs without asking
    mine = await api.submit(hours=2)
    await api.drive(lambda: api.state(mine["id"]) in {"provisioning", "running", "done"})

    # raise the agent limit: the next 2h agent job runs automatically
    doc = PolicyConfig().model_dump(mode="json")
    doc["agent"]["auto_max_hours"] = 3
    assert (await api.client.put("/v1/policy", json=doc)).status_code == 200
    next_agent = await api.submit(source="agent", hours=2)
    await api.drive(lambda: api.state(next_agent["id"]) in {"provisioning", "running", "done"})


async def test_quota_share_rule_asks_even_for_user_jobs(api: Api) -> None:
    # fake has a 30h weekly quota: a 20h job would use two thirds of what is left
    job = await api.submit(hours=20)
    await api.drive(lambda: api.state(job["id"]) == "awaiting_approval")
    reason = api.runtime.store.get_job(job["id"]).approval_reason or ""
    assert "would use 67% of fake's remaining 30h quota" in reason


async def test_quota_endpoint_serves_the_ledger(api: Api) -> None:
    resp = await api.client.get("/v1/quota")
    assert resp.status_code == 200
    rows = {q["provider"]: q for q in resp.json()}
    assert set(rows) == {"fake", "fake-b"}
    fake = rows["fake"]
    assert fake["source"] == "live"  # fake reports live quota; fetched on this request
    assert fake["detail"]["basis"] == "live"
    assert "checked" in fake["detail"]["note"]
    assert fake["detail"]["window"]["kind"] in {"rolling", "fixed"}
    # within the TTL no new adapter call: the cached reading is served
    before = len(api.runtime.store.latest_quota_snapshots())
    again = await api.client.get("/v1/quota")
    assert again.status_code == 200
    assert len(api.runtime.store.latest_quota_snapshots()) == before
    forced = await api.client.get("/v1/quota", params={"refresh": "true"})
    assert forced.status_code == 200
    # provider views and the routing context see the same ledger numbers
    providers = {p["name"]: p for p in (await api.client.get("/v1/providers")).json()}
    assert providers["fake"]["quota"]["detail"]["basis"] in {"live", "live+history"}


async def test_route_uses_the_project_estimate(api: Api) -> None:
    _write(
        Path(api.project) / "train.py",
        "from peft import LoraConfig\nmodel = 'mistralai/mistral-7b-v0.1'\n",
    )
    resp = await api.client.post("/v1/route", json={"spec": api.spec()})
    assert resp.status_code == 200, resp.text
    d = resp.json()
    assert d["router"] == "scoring"
    assert (d["hours"], d["hours_source"]) == (3.0, "heuristic")
    # ~21GB estimated for a 7B LoRA in fp16: the 40GB fake-b is preferred, the 16GB fake
    # stays a candidate (a heuristic never hard-blocks)
    assert d["chosen"]["provider"] == "fake-b"
    fake = next(c for c in d["candidates"] if c["provider"] == "fake")
    assert "may not fit 16GB" in fake["reason"]


async def test_placement_records_the_scoring_decision(api: Api) -> None:
    _write(Path(api.project) / "train.py", "print('hi')\n")
    job = await api.submit(hours=0.5)
    await api.drive(lambda: api.state(job["id"]) in {"provisioning", "running", "done"})
    placed = [e for e in api.runtime.store.events_for(job["id"]) if e.reason == Reason.PLACED]
    assert placed
    detail = placed[0].detail
    assert detail["router"] == "scoring"
    assert detail["hours"] == 0.5
    assert detail["chosen"]["provider"] == "fake"
    assert detail["chosen"]["reason"].startswith("fake: fits 16GB")
