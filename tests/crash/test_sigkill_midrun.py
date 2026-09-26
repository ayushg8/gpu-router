"""SIGKILL a real daemon (no crash-point injection) while a ~10s fake job is running or
still provisioning, restart it, and check the job completes on the SAME remote run:
recovery reattaches instead of submitting again (invariants 6 and 11)."""

from __future__ import annotations

import pytest

from gpu_router.api import JobDetail
from tests.crash.harness import DaemonProc, wait_for

pytestmark = [pytest.mark.crash, pytest.mark.slow]


def _remote_id(detail: JobDetail) -> str | None:
    return detail.attempts[0].remote_id if detail.attempts else None


def _finish_after_restart(daemon: DaemonProc, job_id: str, remote_id: str) -> None:
    client = daemon.start()
    try:
        wait_for(
            lambda: client.job(job_id).job.state in ("done", "failed", "cancelled"),
            timeout_s=40,
            what="terminal state after restart",
        )
        detail = client.job(job_id)
        events = client.events(job_id).events
        assert detail.job.state == "done", [(e.reason, e.message) for e in events]
        assert len(detail.attempts) == 1, detail.attempts
        assert detail.attempts[0].remote_id == remote_id
        assert detail.attempts[0].state == "succeeded"
        runs = daemon.fake_runs()
        assert [r["remote_id"] for r in runs] == [remote_id], runs  # no double submit
        assert any(e.reason == "recovered" for e in events)
        assert (daemon.project / "runs" / job_id[:4] / "result.json").exists()
    finally:
        client.close()


def test_sigkill_mid_run_reattaches_same_remote(daemon: DaemonProc) -> None:
    client = daemon.start()
    job = client.submit(daemon.spec({"duration": 10, "steps": 10}))
    wait_for(lambda: client.job(job.id).job.state == "running", timeout_s=15, what="running")
    wait_for(
        lambda: (client.job(job.id).job.progress.step or 0) >= 2,
        timeout_s=15,
        what="some progress",
    )
    remote_id = _remote_id(client.job(job.id))
    assert remote_id
    client.close()

    daemon.kill()  # SIGKILL: no shutdown path runs
    assert daemon.proc is not None
    assert daemon.proc.returncode == -9

    _finish_after_restart(daemon, job.id, remote_id)


def test_sigkill_during_provisioning_reattaches_same_remote(daemon: DaemonProc) -> None:
    client = daemon.start()
    job = client.submit(daemon.spec({"pending_s": 4, "duration": 2, "steps": 4}))
    wait_for(
        lambda: _remote_id(client.job(job.id)) is not None,
        timeout_s=15,
        what="submitted attempt",
    )
    detail = client.job(job.id)
    assert detail.job.state == "provisioning", detail.job.state
    remote_id = _remote_id(detail)
    assert remote_id
    client.close()

    daemon.kill()
    _finish_after_restart(daemon, job.id, remote_id)
