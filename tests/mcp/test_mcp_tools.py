"""MCP tools (phase 6) through FastMCP's in-memory client against a real in-process daemon
on the fake providers: submit -> status -> logs(tail) -> fetch, approval, cancel, quota,
route, bad refs, and the stdio transport through the real `gpu mcp` entry point."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from gpu_router.mcp import tools
from gpu_router.mcp.server import INSTRUCTIONS
from tests.mcp.conftest import call, call_error, write_gpu_yaml
from tests.shell.conftest import InProcDaemon, wait_for

SPEC_TOOLS = {
    "gpu_submit",
    "gpu_status",
    "gpu_logs",
    "gpu_fetch",
    "gpu_cancel",
    "gpu_quota",
    "gpu_route",
}
#: phase 7b: gpu_infer is additive to the spec's seven (decision in CLAUDE.md)
TOOLS = SPEC_TOOLS | {"gpu_infer"}


async def _until_finished(mcp: Any, ref: str, timeout_s: float = 30) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        st = await call(mcp, "gpu_status", ref=ref, wait_s=2)
        if st["guidance"]["finished"]:
            return st
    raise AssertionError(f"job {ref} did not finish")


async def _until_state(mcp: Any, ref: str, states: set[str], timeout_s: float = 20) -> dict:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        st = await call(mcp, "gpu_status", ref=ref)
        if st["job"]["state"] in states:
            return st
        await asyncio.sleep(0.05)
    raise AssertionError(f"job {ref} never reached {states}")


# --------------------------------------------------------------------------- surface


async def test_exactly_the_spec_tools_and_no_approve(mcp: Any) -> None:
    listed = await mcp.list_tools()
    names = {t.name for t in listed}
    assert names == TOOLS
    assert not any("approve" in n or "deny" in n for n in names)
    by_name = {t.name: t for t in listed}
    submit = by_name["gpu_submit"]
    props = submit.input_schema["properties"]
    assert submit.input_schema["required"] == ["project_dir"]
    assert "Absolute path" in props["project_dir"]["description"]
    assert "relative to project_dir" in props["script"]["description"]
    assert "tell_user" in submit.description
    assert "runs/<id>/" in submit.description
    assert "untrusted" in by_name["gpu_logs"].description
    assert "wait_s" in by_name["gpu_status"].description
    for name in ("gpu_status", "gpu_logs", "gpu_quota", "gpu_route"):
        assert by_name[name].annotations.read_only_hint is True
    assert by_name["gpu_cancel"].annotations.destructive_hint is True
    assert mcp.instructions == INSTRUCTIONS
    assert "never call the kaggle, colab or lightning" in INSTRUCTIONS.lower()
    assert "modal" not in INSTRUCTIONS.lower()  # phase 7b: dropped, never offered
    assert "untrusted" in by_name["gpu_infer"].description


# --------------------------------------------------------------------------- happy path


async def test_submit_status_logs_fetch(daemon: InProcDaemon, project: Path, mcp: Any) -> None:
    sub = await call(
        mcp, "gpu_submit", project_dir=str(project), script="train.py", hours=0.1, name="t1"
    )
    assert sub["submitted"] is True
    job = sub["job"]
    ref = job["short_id"]
    assert job["source"] == "agent"
    assert job["spec"]["labels"]["via"] == "mcp"
    assert job["spec"]["script"] == "train.py"
    assert job["spec"]["provider_options"]["fake"]["steps"] == 40  # gpu.yaml was read
    assert "approval_reason" not in job  # nulls dropped by default
    assert sub["guidance"].get("needs_approval") is None
    assert job["state"] not in ("queued", "routing")  # settled before returning

    final = await _until_finished(mcp, ref)
    assert final["job"]["state"] == "done"
    g = final["guidance"]
    assert g["outputs_dir"] == str(project / "runs" / job["id"][:4])
    assert g["route"].startswith("fake")
    assert len(final["events"]) <= tools.DEFAULT_EVENTS
    assert set(final) == {"job", "attempts", "checkpoints", "events", "guidance", "untrusted"}

    full = await call(mcp, "gpu_status", ref=ref, verbose=True)
    assert "route" in full
    assert len(full["events"]) > tools.DEFAULT_EVENTS
    assert full["job"]["approval_reason"] is None

    tail = await call(mcp, "gpu_logs", ref=ref, tail=5)
    assert len(tail["lines"]) == 5
    assert tail["older_lines_not_shown"] > 0
    assert tail["attempt"] == 1
    assert tail["provider"] == "fake"
    assert tail["finished"] is True
    assert "untrusted" in tail["note"]
    assert not any(line.startswith("::gpu::") for line in tail["lines"])
    assert tail["progress"]["total"] == 40

    again = await call(mcp, "gpu_logs", ref=ref, since=tail["next_since"])
    assert again["lines"] == []
    assert again["empty"]
    assert again["next_since"] == tail["next_since"]

    head = await call(mcp, "gpu_logs", ref=ref, since="1:0", tail=1000)
    assert head["older_lines_not_shown"] == 0
    assert len(head["lines"]) > 40  # every human line from the start
    assert head["lines"][-5:] == tail["lines"]

    fetched = await call(mcp, "gpu_fetch", ref=ref)
    assert fetched["fetched"] is True
    assert fetched["message"] == "outputs are already on disk"
    names = {f["path"] for f in fetched["outputs"]["listing"]}
    assert names  # the fake's own output files
    assert fetched["files"] == fetched["outputs"]["count"]
    assert fetched["outputs_dir"] == g["outputs_dir"]

    refetched = await call(mcp, "gpu_fetch", ref=ref, refetch=True, wait_s=20)
    assert refetched["fetched"] is True, refetched["message"]
    assert {f["path"] for f in refetched["outputs"]["listing"]} == names


async def test_status_overview_is_compact(daemon: InProcDaemon, project: Path, mcp: Any) -> None:
    empty = await call(mcp, "gpu_status")
    assert empty["active"] == []
    assert empty["recent"] == []
    names = {p["name"] for p in empty["providers"]}
    assert {"fake", "fake-b"} <= names
    assert "capabilities" not in empty["providers"][0]  # compact rows
    await call(mcp, "gpu_submit", project_dir=str(project), hours=0.1)
    over = await call(mcp, "gpu_status")
    rows = over["active"] + over["recent"]
    assert len(rows) == 1
    assert set(rows[0]) >= {"id", "short_id", "state"}
    full = await call(mcp, "gpu_status", verbose=True)
    assert "capabilities" in full["providers"][0]


# --------------------------------------------------------------------------- approval


async def test_agent_job_without_hours_waits_and_says_what_to_tell_the_user(
    daemon: InProcDaemon, project: Path, mcp: Any
) -> None:
    sub = await call(mcp, "gpu_submit", project_dir=str(project), script="train.py")
    job = sub["job"]
    sid = job["short_id"]
    assert job["state"] == "awaiting_approval"
    g = sub["guidance"]
    assert g["needs_approval"] is True
    assert "runtime" in g["why"]
    assert g["route"].startswith("fake")  # the one-line route reason
    assert f"/gpu-approve {sid}" in g["tell_user"]
    assert f"gpu approve {sid}" in g["tell_user"]
    assert "do not approve it yourself" in g["rules"]

    over = await call(mcp, "gpu_status")
    row = next(r for r in over["active"] if r["short_id"] == sid)
    assert row["needs_approval"] is True
    assert f"/gpu-approve {sid}" in row["tell_user"]

    # the actor on the submit event is the agent
    full = await call(mcp, "gpu_status", ref=sid, verbose=True)
    assert full["events"][0]["actor"] == "agent"

    # a human approves (never the agent): the job then runs to done
    with daemon.client() as human:
        human.approve(job["id"])
    final = await _until_finished(mcp, sid)
    assert final["job"]["state"] == "done"
    assert final["job"]["approved_by"] != "agent"


# --------------------------------------------------------------------------- cancel


async def test_cancel_is_idempotent(daemon: InProcDaemon, project: Path, mcp: Any) -> None:
    write_gpu_yaml(project, duration=120, steps=10)
    sub = await call(mcp, "gpu_submit", project_dir=str(project), hours=0.1)
    ref = sub["job"]["short_id"]
    await _until_state(mcp, ref, {"running"})
    res = await call(mcp, "gpu_cancel", ref=ref)
    assert res["job"]["state"] in ("cancelling", "cancelled")
    assert "outputs written so far are kept" in res["message"] or "cancelled" in res["message"]
    final = await _until_finished(mcp, ref)
    assert final["job"]["state"] == "cancelled"
    assert final["guidance"]["next"] == "nothing more happens to this job"
    again = await call(mcp, "gpu_cancel", ref=ref)
    assert again["job"]["state"] == "cancelled"
    assert "already finished" in again["message"]
    nothing = await call(mcp, "gpu_fetch", ref=ref, wait_s=1)
    assert nothing["fetched"] is False
    assert "without a successful run" in nothing["message"]


async def test_fetch_before_the_end_says_where_outputs_land(
    daemon: InProcDaemon, project: Path, mcp: Any
) -> None:
    write_gpu_yaml(project, duration=120, steps=10)
    sub = await call(mcp, "gpu_submit", project_dir=str(project), hours=0.1)
    ref = sub["job"]["short_id"]
    res = await call(mcp, "gpu_fetch", ref=ref)
    assert res["fetched"] is False
    assert res["files"] == 0
    assert str(project / "runs") in res["message"]
    await call(mcp, "gpu_cancel", ref=ref)


# --------------------------------------------------------------------------- quota / route


async def test_quota(daemon: InProcDaemon, mcp: Any) -> None:
    res = await call(mcp, "gpu_quota")
    providers = {q["provider"] for q in res["quota"]}
    assert {"fake", "fake-b"} <= providers
    assert len(res["summary"]) == len(res["quota"])
    assert all(":" in line and ("live" in line or "estimate" in line) for line in res["summary"])


async def test_route_and_approval_preview(daemon: InProcDaemon, project: Path, mcp: Any) -> None:
    short = await call(mcp, "gpu_route", project_dir=str(project), script="train.py", hours=0.1)
    assert short["route"]["outcome"] == "place"
    assert short["route"]["chosen"]["provider"] in ("fake", "fake-b")
    assert short["spec"]["source"] == "agent"
    assert short["approval"]["would_ask"] is False
    assert "reason" not in short["approval"]
    # the runtime the rules judged, where it came from, and when it is enforced (D48)
    assert short["approval"]["hours"] == 0.1
    assert short["approval"]["hours_source"] == "spec"
    assert short["approval"]["stopped_for_approval_after_h"] == 0.35
    assert "would run on" in short["guidance"]["summary"]

    unknown = await call(mcp, "gpu_route", project_dir=str(project))
    assert unknown["approval"]["would_ask"] is True
    assert "runtime" in unknown["approval"]["reason"]

    long = await call(mcp, "gpu_route", project_dir=str(project), hours=3)
    assert long["approval"]["would_ask"] is True
    assert "auto limit" in long["approval"]["reason"]

    nofit = await call(mcp, "gpu_route", project_dir=str(project), hours=0.1, vram_gb=600)
    assert nofit["route"]["outcome"] == "no_fit"
    assert nofit["approval"] is None
    assert "no provider fits" in nofit["guidance"]["summary"]

    # a dry run submits nothing
    over = await call(mcp, "gpu_status")
    assert over["active"] == []
    assert over["recent"] == []


# --------------------------------------------------------------------------- bad input


@pytest.mark.parametrize(
    "ref",
    [
        "../daemon/shutdown",
        "a7f2/../../daemon/shutdown",
        "a7f2#x",
        "a7f2?x=1",
        "zz",
        "%2e%2e",
        "a" * 13,
    ],
)
async def test_bad_refs_are_rejected_before_any_request(
    daemon: InProcDaemon, mcp: Any, ref: str
) -> None:
    for tool in ("gpu_status", "gpu_logs", "gpu_fetch", "gpu_cancel"):
        err = await call_error(mcp, tool, ref=ref)
        assert err["code"] == "invalid_request", (tool, err)
        assert "job id" in err["message"] or "hex" in err["message"]
    with daemon.client() as c:
        assert c.health().ready  # nothing reached /v1/daemon/shutdown


async def test_bad_ref_never_needs_the_daemon(no_daemon: Path, mcp: Any) -> None:
    err = await call_error(mcp, "gpu_cancel", ref="../daemon/shutdown")
    assert err["code"] == "invalid_request"
    err = await call_error(mcp, "gpu_status", ref="a7f2")
    assert err["code"] == "daemon_unavailable"
    assert "gpu daemon start" in (err["hint"] or "")


async def test_unknown_job_and_bad_cursor(daemon: InProcDaemon, project: Path, mcp: Any) -> None:
    err = await call_error(mcp, "gpu_status", ref="abcdef123456")
    assert err["code"] == "job_not_found"
    sub = await call(mcp, "gpu_submit", project_dir=str(project), hours=0.1)
    err = await call_error(mcp, "gpu_logs", ref=sub["job"]["short_id"], since="x:y")
    assert err["code"] == "invalid_request"


async def test_bad_project_and_spec_inputs(daemon: InProcDaemon, project: Path, mcp: Any) -> None:
    err = await call_error(mcp, "gpu_submit", project_dir="relative/dir", script="train.py")
    assert err["code"] == "invalid_request"
    assert "absolute" in err["message"]
    err = await call_error(mcp, "gpu_submit", project_dir=str(project / "missing"))
    assert err["code"] == "invalid_request"
    for script in ("../escape.py", "/etc/passwd", "-rf"):
        err = await call_error(mcp, "gpu_submit", project_dir=str(project), script=script)
        assert err["code"] == "invalid_request", script
    err = await call_error(
        mcp, "gpu_submit", project_dir=str(project), hours=0.1, env={"HF_TOKEN": "abc"}
    )
    assert err["code"] == "invalid_spec"
    err = await call_error(mcp, "gpu_submit", project_dir=str(project), provider="kagle")
    assert err["code"] == "provider_not_found"
    over = await call(mcp, "gpu_status")
    assert over["active"] == []
    assert over["recent"] == []


# --------------------------------------------------------------------------- migration


async def test_logs_across_attempts_are_marked(
    daemon: InProcDaemon, project: Path, mcp: Any
) -> None:
    write_gpu_yaml(project, duration=1.0, steps=20, attempts={"1": {"die_after": 0.4}})
    sub = await call(mcp, "gpu_submit", project_dir=str(project), hours=0.1)
    ref = sub["job"]["short_id"]
    final = await _until_finished(mcp, ref, timeout_s=40)
    assert final["job"]["state"] == "done"
    assert final["job"]["attempt_count"] >= 2
    logs = await call(mcp, "gpu_logs", ref=ref, since="1:0", tail=1000)
    markers = [ln for ln in logs["lines"] if ln.startswith("── attempt ")]
    assert markers
    assert markers[0].startswith("── attempt 1 on ")
    assert logs["latest_attempt"] >= 2
    one = await call(mcp, "gpu_logs", ref=ref, attempt=1, tail=3)
    assert one["attempt"] == 1
    assert len(one["lines"]) <= 3


# --------------------------------------------------------------------------- pure helpers


def test_long_lines_are_cut() -> None:
    line = "x" * (tools.MAX_LINE_CHARS + 500)
    cut = tools._cut(line)
    assert cut.startswith("x" * tools.MAX_LINE_CHARS)
    assert cut.endswith("[500 more chars cut]")
    assert tools._cut("short") == "short"


def test_check_ref_normalizes() -> None:
    assert tools.check_ref(" A7F2 ") == "a7f2"
    from gpu_router.errors import InvalidRequest

    with pytest.raises(InvalidRequest):
        tools.check_ref("a7f2/x")


# --------------------------------------------------------------------------- stdio


def test_gpu_mcp_help() -> None:
    import subprocess

    out = subprocess.run(
        [sys.executable, "-m", "gpu_router", "mcp", "--help"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert out.returncode == 0
    assert "gpu_submit" in out.stdout
    bad = subprocess.run(
        [sys.executable, "-m", "gpu_router", "mcp", "--nope"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert bad.returncode == 2
    assert "unexpected arguments" in bad.stderr


async def test_stdio_transport_through_the_gpu_entry_point(
    daemon: InProcDaemon, project: Path
) -> None:
    """`python -m gpu_router mcp` (= `gpu mcp`) speaks MCP on stdout, finds the running
    daemon through GPU_ROUTER_HOME, and nothing else writes to stdout."""
    from fastmcp import Client
    from fastmcp.client.transports import StdioTransport

    env = {k: v for k, v in os.environ.items() if not k.startswith("GPU_ROUTER_")}
    env.update(
        GPU_ROUTER_HOME=str(daemon.home),
        GPU_ROUTER_TEST_MODE="1",
        GPU_ROUTER_NO_AUTOSTART="1",
    )
    transport = StdioTransport(
        command=sys.executable, args=["-m", "gpu_router", "mcp"], env=env, cwd=str(project)
    )
    async with Client(transport) as client:
        names = {t.name for t in await client.list_tools()}
        assert names == TOOLS
        res = await call(client, "gpu_quota")
        assert {"fake", "fake-b"} <= {q["provider"] for q in res["quota"]}
        sub = await call(client, "gpu_submit", project_dir=str(project), hours=0.1)
        ref = sub["job"]["short_id"]
    wait_for(lambda: daemon.job(ref).state == "done", what="stdio-submitted job done")
    assert json.loads(daemon.job(ref).model_dump_json())["source"] == "agent"


async def test_datasets_and_roots_that_would_leak_are_refused(
    daemon: InProcDaemon, project: Path, mcp: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_home = tmp_path / "home"
    (fake_home / ".ssh").mkdir(parents=True)
    (fake_home / ".ssh" / "id_ed25519").write_text("not a key")
    (fake_home / "data").mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    for item in (f"keys={fake_home / '.ssh'}", str(fake_home / ".ssh" / "id_ed25519")):
        err = await call_error(mcp, "gpu_submit", project_dir=str(project), hours=0.1, data=[item])
        assert err["code"] == "invalid_request", item
        assert "credential" in err["message"]
    err = await call_error(
        mcp, "gpu_submit", project_dir=str(project), hours=0.1, data=[f"all={fake_home}"]
    )
    assert err["code"] == "invalid_request"
    assert "home directory" in err["message"]
    (fake_home / "train.py").write_text("print(1)\n")
    err = await call_error(mcp, "gpu_route", project_dir=str(fake_home), script="train.py")
    assert err["code"] == "invalid_request"
    assert "home directory" in err["message"]
    ok = await call(
        mcp,
        "gpu_route",
        project_dir=str(project),
        hours=0.1,
        data=[f"d={fake_home / 'data'}"],
    )
    assert ok["spec"]["data"][0]["mount"] == "d"
