"""Polling results stay small (2026-10-04 field test: every gpu_status repeated the full
spec and two hashes, ~25% of each poll) and the poll hint never exceeds the wait_s cap."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from gpu_router.mcp import tools
from tests.mcp.conftest import call
from tests.shell.conftest import InProcDaemon


async def test_polls_carry_a_brief_spec(daemon: InProcDaemon, project: Path, mcp: Any) -> None:
    sub = await call(
        mcp,
        "gpu_submit",
        project_dir=str(project),
        script="train.py",
        args=["--lr", "3"],
        hours=0.1,
    )
    assert "spec_hash" in sub["job"]  # the submit result keeps the whole document
    assert sub["job"]["spec"]["labels"]["via"] == "mcp"
    ref = sub["job"]["short_id"]
    st = await call(mcp, "gpu_status", ref=ref)
    job = st["job"]
    assert job["spec"] == {"script": "train.py", "args": ["--lr", "3"], "hours": 0.1}
    assert "spec_hash" not in job
    assert "bundle_sha256" not in job
    if st["guidance"].get("follow") == "wait":
        assert st["guidance"]["poll_every_s"] <= tools.MAX_WAIT_S
    full = await call(mcp, "gpu_status", ref=ref, verbose=True)
    assert full["job"]["spec_hash"]
    assert full["job"]["spec"]["labels"]["via"] == "mcp"
