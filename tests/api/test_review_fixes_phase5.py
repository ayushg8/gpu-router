"""Phase 4/5 review regressions through a real DaemonRuntime (bundler, scoring router,
rules policy), D44."""

from __future__ import annotations

from pathlib import Path

from gpu_router.statemachine import Reason
from tests.api.conftest import Api


async def test_an_agent_job_without_hours_asks_first_with_a_real_bundle(api: Api) -> None:
    """Finding: the bundle estimate always supplies a 1h guess, equal to the agent limit,
    so 'unknown runtime asks' never fired for a real submission."""
    (Path(api.project) / "train.py").write_text("for i in range(10):\n    print(i)\n")
    job = await api.submit(source="agent")
    await api.drive(lambda: api.state(job["id"]) == "awaiting_approval", max_s=30)
    stored = api.runtime.store.get_job(job["id"])
    assert (stored.approval_reason or "").startswith("runtime not given (guess 1h); set --hours")
    asked = [
        e for e in api.runtime.store.events_for(job["id"]) if e.reason == Reason.APPROVAL_REQUIRED
    ]
    assert asked[-1].detail["rule"] == "unknown_hours"
    assert asked[-1].detail["hours_source"] == "heuristic"
    # the same submission from the user runs, and an agent job with --hours 1 runs too
    mine = await api.submit()
    await api.drive(lambda: api.state(mine["id"]) in {"provisioning", "running", "done"})
    timed = await api.submit(source="agent", hours=1)
    await api.drive(lambda: api.state(timed["id"]) in {"provisioning", "running", "done"})
