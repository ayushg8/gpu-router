"""`gpu` CLI end to end: CliRunner -> GpuClient -> real daemon subprocess -> fake provider."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from gpu_router.cli import exitcodes
from tests.cli.conftest import Cli

# --------------------------------------------------------------------------- run


def test_run_waits_streams_logs_and_fetches_outputs(cli: Cli) -> None:
    res = cli("run")
    assert res.exit_code == 0, res.output
    out = res.stdout
    assert "submitted job" in out
    assert "running on fake" in out
    assert "step 3/3" in out  # log lines streamed
    assert "✓ train done" in out
    runs = list((cli.project / "runs").iterdir())
    assert len(runs) == 1
    assert any(runs[0].iterdir())
    # the final log lines print before the "finished" line
    assert out.index("step 3/3") < out.index("✓ train done")


def test_run_json_defaults_to_detach(cli: Cli) -> None:
    data, res = cli.json("run")
    assert res.exit_code == 0
    job = data["job"]
    assert set(data) == {"job"}
    assert job["spec"]["script"] == "train.py"
    assert job["spec"]["source"] == "cli"
    assert job["state"] not in ("done", "failed")
    assert len(job["short_id"]) >= 4


def test_run_json_wait_returns_final_job(cli: Cli) -> None:
    data, res = cli.json("run", "--wait")
    assert res.exit_code == 0
    assert data["job"]["state"] == "done"
    assert data["job"]["outputs_fetched"] is True


def test_run_failed_script_exit_code(cli: Cli) -> None:
    cli.fake(exit_code=3)
    res = cli("run")
    assert res.exit_code == exitcodes.JOB_FAILED
    assert "✗ train failed" in res.stdout
    assert "gpu logs" in res.stdout


def test_run_args_passthrough_gpu_options_before_script(cli: Cli) -> None:
    """D23: gpu options go before the script; after it only --json/--wait/--detach/
    --dry-run/--vram/--hours/--provider are gpu's. Everything else reaches the script."""
    data, res = cli.json(
        "run", "--name", "exp1", "train.py", "--epochs", "3", "--vram", "8",
        "--name", "run7", "--", "--name", "x",
    )  # fmt: skip
    assert res.exit_code == 0, res.output
    spec = data["job"]["spec"]
    assert spec["args"] == ["--epochs", "3", "--name", "run7", "--name", "x"]
    assert spec["vram_gb"] == 8  # tail gpu option still read
    assert spec["name"] == "exp1"  # from before the script only
    assert "`--name` after the script is passed to the script" in res.stderr


def test_run_script_short_flags_and_common_names_reach_the_script(cli: Cli) -> None:
    """Review finding: `-wd 0.01` was read as --wait --detach, `-e 10` as --env, and
    `--gpu 0` / `--name exp1` were swallowed by gpu. After the script they are script args."""
    data, res = cli.json(
        "run", "--dry-run", "train.py", "-wd", "0.01", "--lr", "3", "-e", "10",
        "--gpu", "0", "--name", "exp1",
    )  # fmt: skip
    assert res.exit_code == 0, res.output
    spec = data["spec"]
    assert spec["args"] == ["-wd", "0.01", "--lr", "3", "-e", "10", "--gpu", "0", "--name", "exp1"]
    assert spec["gpu"] is None
    assert spec["name"] is None
    assert spec["env"] == {}
    assert data["dry_run"] is True  # tail --dry-run after `--json` still gpu's


def test_run_script_args_without_script_use_gpu_yaml_script(cli: Cli) -> None:
    data, res = cli.json("run", "--epochs", "2")
    assert res.exit_code == 0, res.output
    assert data["job"]["spec"]["script"] == "train.py"
    assert data["job"]["spec"]["args"] == ["--epochs", "2"]


def test_flags_override_gpu_yaml(cli: Cli) -> None:
    cli.write_yaml(
        "script: train.py\nvram: 8\nhours: 1\nenv: {A: file, B: file}\n"
        "provider_options:\n  fake: {duration: 0.3}\n"
    )
    data, res = cli.json("run", "--dry-run", "--vram", "12", "--env", "B=flag", "-e", "C=3")
    assert res.exit_code == 0, res.output
    spec = data["spec"]
    assert data["dry_run"] is True
    assert spec["vram_gb"] == 12  # flag wins
    assert spec["hours"] == 1  # file value kept
    assert spec["env"] == {"A": "file", "B": "flag", "C": "3"}
    assert data["route"]["outcome"] == "place"
    assert data["route"]["chosen"]["provider"] == "fake"


def test_dry_run_submits_nothing_and_previews_bundle(cli: Cli) -> None:
    before, _ = cli.json("jobs", "--all", "--limit", "500")
    data, res = cli.json("run", "--dry-run")
    assert res.exit_code == 0
    bundle = data["bundle"]
    assert bundle is None or bundle["file_count"] >= 1
    after, _ = cli.json("jobs", "--all", "--limit", "500")
    assert len(after["jobs"]) == len(before["jobs"])
    human = cli("run", "--dry-run")
    assert "dry run: nothing submitted" in human.stdout
    assert "fake" in human.stdout


def test_run_command_for_non_python_entrypoint(cli: Cli) -> None:
    data, res = cli.json("run", "bash", "run.sh", "--fast", "--dry-run")
    assert res.exit_code == 0, res.output
    assert data["spec"]["command"] == ["bash", "run.sh", "--fast"]
    assert data["spec"]["script"] is None


# --------------------------------------------------------------------------- errors


def test_invalid_gpu_yaml_exit_2_with_line_and_suggestion(cli: Cli) -> None:
    cli.write_yaml("version: 1\nscirpt: train.py\n")
    res = cli("run")
    assert res.exit_code == exitcodes.USAGE
    assert "gpu.yaml:2" in res.stderr
    assert "did you mean `script`?" in res.stderr
    data, res = cli.json("run")
    assert res.exit_code == exitcodes.USAGE
    assert data["error"]["code"] == "invalid_spec"
    assert data["error"]["detail"]["key"] == "scirpt"
    assert data["error"]["detail"]["line"] == 2


def test_bad_flag_value_exit_2(cli: Cli) -> None:
    data, res = cli.json("run", "--hours", "-1")
    assert res.exit_code == exitcodes.USAGE
    assert "--hours" in data["error"]["message"]
    data, res = cli.json("run", "--env", "NOEQUALS")
    assert res.exit_code == exitcodes.USAGE
    assert "NAME=VALUE" in data["error"]["message"]


def test_missing_script_exit_2(cli: Cli) -> None:
    res = cli("run", "nope.py")
    assert res.exit_code == exitcodes.USAGE
    assert "nope.py not found" in res.stderr


def test_unknown_job_exit_4(cli: Cli) -> None:
    data, res = cli.json("status", "fffffff")
    assert res.exit_code == exitcodes.NOT_FOUND
    assert data["error"]["code"] == "job_not_found"
    res = cli("logs", "zz")
    assert res.exit_code == exitcodes.USAGE  # not hex at all
    assert "not a job id" in res.stderr


def test_approve_finished_job_is_conflict(cli: Cli) -> None:
    job = cli.json("run", "--wait")[0]["job"]
    data, res = cli.json("approve", job["short_id"])
    assert res.exit_code == exitcodes.CONFLICT
    assert data["error"]["code"] == "invalid_transition"


def test_daemon_down_without_autostart_exit_3(
    cli: Cli, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GPU_ROUTER_HOME", str(tmp_path / "empty-home"))
    data, res = cli.json("status")
    assert res.exit_code == exitcodes.DAEMON
    assert data["error"]["code"] == "daemon_unavailable"
    res = cli("jobs")
    assert res.exit_code == exitcodes.DAEMON
    assert "gpu daemon start" in res.stderr


# --------------------------------------------------------------------------- status / jobs


def test_status_overview_and_detail_by_prefix(cli: Cli) -> None:
    job = cli.json("run", "--wait")[0]["job"]
    view, res = cli.json("status")
    assert res.exit_code == 0
    assert {"ready", "counts", "active", "recent", "providers"} <= set(view)
    detail, res = cli.json("status", job["id"][:6])
    assert res.exit_code == 0
    assert detail["job"]["id"] == job["id"]
    assert detail["attempts"][0]["provider"] == "fake"
    assert any(e["reason"] == "placed" for e in detail["events"])
    human = cli("status", job["short_id"])
    assert f"job {job['short_id']}" in human.stdout
    assert "timeline" in human.stdout
    assert "fake · T4" in human.stdout


def test_jobs_and_history(cli: Cli) -> None:
    cli.fake(duration=30)
    running = cli.json("run")[0]["job"]
    try:
        jobs, res = cli.json("jobs")
        assert res.exit_code == 0
        ids = [j["id"] for j in jobs["jobs"]]
        assert running["id"] in ids
        assert all(
            j["state"] not in ("done", "failed", "cancelled", "denied") for j in jobs["jobs"]
        )
        human = cli("jobs")
        assert running["short_id"] in human.stdout
    finally:
        cli("cancel", running["id"])
    _wait_state(cli, running["id"], {"cancelled"})
    hist, res = cli.json("history")
    assert res.exit_code == 0
    assert running["id"] in [j["id"] for j in hist["jobs"]]
    assert all(j["state"] in ("done", "failed", "cancelled", "denied") for j in hist["jobs"])
    failed, _ = cli.json("history", "--failed")
    assert all(j["state"] == "failed" for j in failed["jobs"])


def test_jobs_here_filters_by_project(cli: Cli) -> None:
    data, res = cli.json("jobs", "--all", "--here")
    assert res.exit_code == 0
    assert data["jobs"] == []  # fresh project dir per test
    human = cli("jobs", "--here")
    # helpful empty state: quota plus an example run, never blank
    assert "no running or queued jobs" in human.stdout
    assert "gpu run train.py" in human.stdout
    assert "fake" in human.stdout


# --------------------------------------------------------------------------- logs


def test_logs_json_is_ndjson_ending_in_eof(cli: Cli) -> None:
    job = cli.json("run", "--wait")[0]["job"]
    res = cli("logs", job["short_id"], "--follow", "--json")
    assert res.exit_code == 0
    records = [json.loads(ln) for ln in res.stdout.splitlines() if ln.strip()]
    lines = [r for r in records if "line" in r]
    assert lines
    assert all({"attempt", "offset", "line"} <= set(r) for r in lines)
    assert records[-1] == {"eof": True, "state": "done"}
    assert not any("::gpu::" in r["line"] for r in lines)  # protocol hidden


def test_logs_human_and_follow_exit_code(cli: Cli) -> None:
    cli.fake(exit_code=2)
    job = cli.json("run")[0]["job"]
    res = cli("logs", job["short_id"], "-f")
    assert res.exit_code == exitcodes.JOB_FAILED
    assert "step" in res.stdout
    assert "failed" in res.stdout


# --------------------------------------------------------------------------- actions


def test_cancel_running_job(cli: Cli) -> None:
    cli.fake(duration=60)
    job = cli.json("run")[0]["job"]
    _wait_state(cli, job["id"], {"running"})
    data, res = cli.json("cancel", job["short_id"])
    assert res.exit_code == 0
    assert data["state"] in ("cancelling", "cancelled")
    _wait_state(cli, job["id"], {"cancelled"})
    again = cli("cancel", job["short_id"])
    assert again.exit_code == 0
    assert "nothing to cancel" in again.stdout


def test_approve_flow(cli: Cli) -> None:
    cli.write_yaml(
        "script: train.py\nrequires_approval: true\nprovider_options:\n  fake: {duration: 0.3}\n"
    )
    job = cli.json("run")[0]["job"]
    _wait_state(cli, job["id"], {"awaiting_approval"})
    human = cli("status")
    assert "needs approval" in human.stdout
    detail = cli("status", job["short_id"])
    assert f"gpu approve {job['short_id']}" in detail.stdout
    _data, res = cli.json("approve", job["short_id"], "--reason", "looks fine")
    assert res.exit_code == 0
    _wait_state(cli, job["id"], {"done"})


def test_deny_flow(cli: Cli) -> None:
    cli.write_yaml("script: train.py\nrequires_approval: true\n")
    job = cli.json("run")[0]["job"]
    _wait_state(cli, job["id"], {"awaiting_approval"})
    res = cli("deny", job["short_id"], "--reason", "too long")
    assert res.exit_code == 0
    assert "denied" in res.stdout
    data, _ = cli.json("status", job["short_id"])
    assert data["job"]["state"] == "denied"


def test_run_wait_on_denied_job_exit_11(cli: Cli) -> None:
    """`gpu run` waits through approval; a deny from elsewhere ends it with exit 11.
    The deny goes through GpuClient (CliRunner swaps sys.stdout, so it is not
    thread-safe)."""
    import threading

    from gpu_router.client import GpuClient
    from gpu_router.paths import Paths

    cli.write_yaml("script: train.py\nrequires_approval: true\n")
    denied: list[str] = []

    def deny_soon() -> None:
        with GpuClient.from_env(Paths(cli.home)) as client:
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                found = client.jobs(project_dir=str(cli.project.resolve()), limit=5).jobs
                waiting = [j for j in found if j.state == "awaiting_approval"]
                if waiting:
                    client.deny(waiting[0].id)
                    denied.append(waiting[0].id)
                    return
                time.sleep(0.1)

    t = threading.Thread(target=deny_soon, daemon=True)
    t.start()
    res = cli("run")
    t.join(20)
    assert denied
    assert res.exit_code == exitcodes.JOB_STOPPED
    assert "denied" in res.stdout


def test_fetch_with_dest(cli: Cli, tmp_path: Path) -> None:
    job = cli.json("run", "--wait")[0]["job"]
    dest = tmp_path / "copy"
    data, res = cli.json("fetch", job["short_id"], "--dest", str(dest))
    assert res.exit_code == 0, res.output
    assert data["fetched"] is True
    assert data["dest"] == str(dest.resolve())
    assert data["files"] >= 1
    assert any(dest.iterdir())


def test_fetch_unfinished_job_is_conflict(cli: Cli) -> None:
    cli.fake(duration=60)
    job = cli.json("run")[0]["job"]
    try:
        data, res = cli.json("fetch", job["short_id"])
        assert res.exit_code == exitcodes.CONFLICT
        assert data["error"]["code"] == "invalid_transition"
    finally:
        cli("cancel", job["id"])


# --------------------------------------------------------------------------- route / providers


def test_route_explains(cli: Cli) -> None:
    data, res = cli.json("route")
    assert res.exit_code == 0
    assert data["route"]["outcome"] == "place"
    assert [c["provider"] for c in data["route"]["candidates"]] == ["fake", "fake-b"]
    human = cli("route", "--vram", "24")
    assert "fake-b" in human.stdout
    assert "ruled out" in human.stdout


def test_route_no_fit_exit_12(cli: Cli) -> None:
    data, res = cli.json("route", "--vram", "600")
    assert res.exit_code == exitcodes.NO_FIT
    assert data["route"]["outcome"] == "no_fit"


def test_quota_and_providers(cli: Cli) -> None:
    q, res = cli.json("quota")
    assert res.exit_code == 0
    assert {x["provider"] for x in q["quota"]} >= {"fake", "fake-b"}
    p, res = cli.json("providers")
    assert res.exit_code == 0
    names = {x["name"]: x for x in p["providers"]}
    assert names["fake"]["enabled"] is True
    human = cli("providers")
    assert "fake-b" in human.stdout
    assert "A100-40GB" in human.stdout
    human = cli("quota")
    assert "/30h" in human.stdout


# --------------------------------------------------------------------------- daemon


def test_daemon_status_and_start_when_running(cli: Cli) -> None:
    res = cli("daemon", "status", "--json")
    assert res.exit_code == 0
    assert json.loads(res.stdout)["running"] is True
    res = cli("daemon", "start", "--json")
    assert res.exit_code == 0
    info = json.loads(res.stdout)
    assert info["running"] is True
    assert info["started"] is False


def test_install_launchd_print_does_not_touch_launchctl(
    cli: Cli, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr("gpu_router.daemon.launchd._launchctl", lambda *a: calls.append(a))
    res = cli("daemon", "install-launchd", "--print")
    assert res.exit_code == 0
    assert "dev.gpu-router.daemon" in res.stdout
    assert "<string>--launchd</string>" in res.stdout
    assert calls == []


def test_install_launchd_writes_plist(
    cli: Cli, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import subprocess as sp

    calls: list[tuple[str, ...]] = []

    def fake_launchctl(*args: str) -> sp.CompletedProcess[str]:
        calls.append(args)
        return sp.CompletedProcess(["launchctl", *args], 0, "", "")

    monkeypatch.setenv("HOME", str(tmp_path / "userhome"))
    monkeypatch.setattr("gpu_router.daemon.launchd._launchctl", fake_launchctl)
    plist = tmp_path / "userhome" / "Library" / "LaunchAgents" / "dev.gpu-router.daemon.plist"
    # the test's GPU_ROUTER_HOME is a custom dir: the global agent is refused (D54) ...
    res = cli("daemon", "install-launchd")
    assert res.exit_code != 0
    assert "--this-home" in res.output
    assert not plist.exists()
    assert not calls
    # ... unless the dir takes it on purpose
    res = cli("daemon", "install-launchd", "--this-home")
    assert res.exit_code == 0, res.output
    assert plist.exists()
    assert calls
    assert calls[-1][0] == "bootstrap"


def test_autostart_spawns_daemon_in_background(
    gpu_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No daemon running: the CLI starts one (same GPU_ROUTER_HOME), says so on stderr,
    and answers. Never touches launchd."""
    from typer.testing import CliRunner

    from gpu_router.cli.app import app

    home = tmp_path / "auto-home"
    home.mkdir()
    (home / "config.yaml").write_text("version: 1\ndaemon:\n  shutdown_grace_s: 1\n")
    monkeypatch.setenv("GPU_ROUTER_HOME", str(home))
    monkeypatch.setenv("GPU_ROUTER_PORT", "0")
    monkeypatch.setattr("gpu_router.daemon.launchd.is_installed", lambda home=None: False)
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    try:
        res = runner.invoke(app, ["status", "--json"], catch_exceptions=False)
        assert res.exit_code == 0, res.output
        assert "starting it in the background" in res.stderr
        assert "daemon started" in res.stderr
        view = json.loads(res.stdout)
        assert view["ready"] is True
        # second call reuses it silently
        res = runner.invoke(app, ["status", "--json"], catch_exceptions=False)
        assert res.exit_code == 0
        assert "starting" not in res.stderr
    finally:
        stop = runner.invoke(app, ["daemon", "stop"], catch_exceptions=False)
        assert stop.exit_code == 0, stop.output


# --------------------------------------------------------------------------- entry wiring


def test_console_entry_dispatches_to_cli(cli: Cli) -> None:
    env = {**os.environ, "GPU_ROUTER_HOME": str(cli.home), "GPU_ROUTER_NO_AUTOSTART": "1"}
    proc = subprocess.run(
        [sys.executable, "-m", "gpu_router", "status", "--json"],
        capture_output=True,
        text=True,
        env=env,
        cwd=cli.project,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    assert "providers" in json.loads(proc.stdout)
    proc = subprocess.run(
        [sys.executable, "-m", "gpu_router", "--help"],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )
    assert proc.returncode == 0
    for cmd in ("run", "route", "status", "logs", "cancel", "fetch", "approve", "quota"):
        assert cmd in proc.stdout


def test_status_line_fast_path_untouched(cli: Cli) -> None:
    env = {**os.environ, "GPU_ROUTER_HOME": str(cli.home)}
    start = time.monotonic()
    proc = subprocess.run(
        [sys.executable, "-m", "gpu_router", "status", "--line"],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )
    assert proc.returncode == 0
    assert "Traceback" not in proc.stderr
    assert time.monotonic() - start < 5  # interpreter start dominates; no daemon call


# --------------------------------------------------------------------------- helpers


def _wait_state(cli: Cli, job_id: str, states: set[str], timeout: float = 20) -> None:
    deadline = time.monotonic() + timeout
    state = None
    while time.monotonic() < deadline:
        state = cli.json("status", job_id)[0]["job"]["state"]
        if state in states:
            return
        time.sleep(0.1)
    raise AssertionError(f"job {job_id} stuck in {state}, wanted {states}")
