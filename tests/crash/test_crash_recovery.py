"""Kill the daemon at every named crash point, restart it, and check the job finishes
exactly once (CLAUDE.md "Testing strategy": crash)."""

from __future__ import annotations

from typing import Any

import pytest

from gpu_router.engine.crashpoints import CRASH_POINTS
from tests.crash.harness import DaemonProc, wait_for

pytestmark = [pytest.mark.crash, pytest.mark.slow]

#: crash point -> (fake directives, action after submit, expected final state)
SCENARIOS: dict[str, tuple[dict[str, Any], str, str]] = {
    "after_place_commit": ({"duration": 0.5, "steps": 3}, "none", "done"),
    "after_submit_return": ({"duration": 0.5, "steps": 3}, "none", "done"),
    "after_running": ({"duration": 1.0, "steps": 3}, "none", "done"),
    "mid_checkpoint": ({"duration": 3.0, "steps": 5, "checkpoint_every": 0.5}, "none", "done"),
    "before_fetch": ({"duration": 0.5, "steps": 3}, "none", "done"),
    "after_fetch": ({"duration": 0.5, "steps": 3}, "none", "done"),
    "during_cancel": ({"duration": 60, "steps": 3}, "cancel", "cancelled"),
}


def test_every_crash_point_has_a_scenario() -> None:
    assert set(SCENARIOS) == set(CRASH_POINTS)


@pytest.mark.parametrize("point", sorted(SCENARIOS))
def test_crash_and_recover(daemon: DaemonProc, point: str) -> None:
    directives, action, expected = SCENARIOS[point]
    client = daemon.start(crash_at=point)
    job = client.submit(daemon.spec(directives))
    if action == "cancel":
        wait_for(lambda: client.job(job.id).job.state == "running", what="running")
        client.cancel(job.id)
    code = daemon.wait_exit()
    client.close()
    assert code == 137, daemon.logs[-1].read_text()

    client = daemon.start()
    try:
        wait_for(
            lambda: client.job(job.id).job.state in ("done", "failed", "cancelled"),
            timeout_s=60,
            what=f"{point}: terminal state",
        )
        detail = client.job(job.id)
        assert detail.job.state == expected, [
            (e.reason, e.message) for e in client.events(job.id).events
        ]
        runs = daemon.fake_runs()
        assert len(runs) == 1, runs  # exactly one remote run: no double submit
        assert any(e.reason == "recovered" for e in client.events(job.id).events)
        if expected == "done":
            assert detail.job.outputs_fetched
            assert (daemon.project / "runs" / job.id[:4] / "result.json").exists()
            assert [a.state for a in detail.attempts] == ["succeeded"]
        else:
            assert runs[0]["cancelled_at"] is not None
    finally:
        client.close()
