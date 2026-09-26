"""Phase-1 acceptance: a real daemon process accepts a fake job over HTTP and it reaches
`done` with outputs in runs/<id4>/; `gpu daemon status/stop` work against it."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from tests.crash.harness import DaemonProc, wait_for

pytestmark = pytest.mark.slow


def _gpu(daemon: DaemonProc, *args: str) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GPU_ROUTER_")}
    env["GPU_ROUTER_HOME"] = str(daemon.home)
    return subprocess.run(
        [sys.executable, "-m", "gpu_router", *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def test_daemon_runs_fake_job_end_to_end(daemon: DaemonProc) -> None:
    client = daemon.start()
    try:
        job = client.submit(daemon.spec({"duration": 0.5, "steps": 5}))
        records = list(client.logs(job.id, follow=True))
        assert records[-1].eof
        assert records[-1].state == "done"
        assert any(r.line and "loss" in r.line for r in records)
        detail = client.job(job.id)
        assert detail.job.state == "done"
        out = daemon.project / "runs" / job.id[:4]
        assert json.loads((out / "result.json").read_text())["job_id"] == job.id
        state = json.loads(daemon.paths.state.read_text())
        assert state["schema"] == 1
        # D55: finished jobs leave the status line at once (finished_visible_s defaults to
        # 0), so the done job must drop out of state.json's active list, not linger.
        wait_for(
            lambda: all(
                r["id"] != job.id for r in json.loads(daemon.paths.state.read_text())["active"]
            ),
            what="done job out of state.json active",
        )
        assert json.loads(daemon.paths.state.read_text())["finished_visible_s"] == 0
    finally:
        client.close()

    status = _gpu(daemon, "daemon", "status", "--json")
    assert status.returncode == 0, status.stderr
    info = json.loads(status.stdout)
    assert info["running"]
    assert info["ready"]
    assert info["test_mode"] is True

    second = _gpu(daemon, "daemon", "run", "--port", "0")
    assert second.returncode == 3
    assert "already running" in second.stderr.lower()

    stop = _gpu(daemon, "daemon", "stop")
    assert stop.returncode == 0, stop.stderr
    assert daemon.wait_exit() == 0
    assert not daemon.paths.runtime.exists()

    after = _gpu(daemon, "daemon", "status")
    assert after.returncode == 3
    assert "not running" in after.stderr
