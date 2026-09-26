"""Phase-2 review fixes for the CLI: usage errors through the real executable, the JSON
contract of `run --json --wait`, fetch --dest failures, unknown providers, paging, secrets,
daemon --json, and spawn's early-exit detection."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from gpu_router.cli import exitcodes
from gpu_router.errors import DaemonUnavailable
from tests.cli.conftest import Cli

# --------------------------------------------------------------------------- usage errors


def _gpu(args: list[str], home: Path, cwd: Path) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GPU_ROUTER_")}
    env.update({"GPU_ROUTER_HOME": str(home), "GPU_ROUTER_NO_AUTOSTART": "1"})
    return subprocess.run(
        [sys.executable, "-m", "gpu_router", *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=cwd,
        timeout=60,
        check=False,
    )


@pytest.mark.parametrize(
    "args",
    [
        ["foo"],
        ["status", "--bogus"],
        ["logs"],
        ["jobs", "-n", "0"],
        ["run", "train.py", "--vram", "abc"],
        ["run", "train.py", "--vram"],
    ],
)
def test_usage_errors_exit_2_without_traceback(args: list[str], tmp_path: Path) -> None:
    """Review finding: typer 0.27 vendors click, so `main()` never matched the usage
    exceptions and every usage error was a traceback + exit 1."""
    (tmp_path / "train.py").write_text("print(1)\n")
    human = _gpu(args, tmp_path / "home", tmp_path)
    assert human.returncode == exitcodes.USAGE, human.stderr
    assert "Traceback" not in human.stderr
    assert human.stderr.startswith("gpu: ")
    assert "--help" in human.stderr
    assert human.stdout == ""

    as_json = _gpu([*args, "--json"], tmp_path / "home", tmp_path)
    assert as_json.returncode == exitcodes.USAGE, as_json.stderr
    assert "Traceback" not in as_json.stderr
    body = json.loads(as_json.stdout)
    assert body["error"]["code"] == "invalid_request"
    assert body["error"]["message"]
    assert body["error"]["detail"]["usage"] is True


def test_run_usage_error_hints_at_argument_order(tmp_path: Path) -> None:
    (tmp_path / "train.py").write_text("print(1)\n")
    res = _gpu(["run", "train.py", "--vram", "abc"], tmp_path / "home", tmp_path)
    assert "gpu options go before the script" in res.stderr


def test_help_still_exits_0(tmp_path: Path) -> None:
    res = _gpu(["run", "--help"], tmp_path / "home", tmp_path)
    assert res.returncode == 0
    assert "gpu options go before the script" in res.stdout


# --------------------------------------------------------------------------- run --json --wait


def test_run_json_wait_error_after_submit_names_the_job(
    cli: Cli, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review finding: an error while waiting was a bare envelope with no job id, so an
    agent could not tell a job exists and would resubmit."""
    cli.fake(duration=30)

    def boom(self: Any, ref: str, *, after: int = 0) -> Any:
        raise DaemonUnavailable("lost the daemon", hint="start it")

    monkeypatch.setattr("gpu_router.client.GpuClient.events", boom)
    data, res = cli.json("run", "--wait")
    assert res.exit_code == exitcodes.DAEMON
    detail = data["error"]["detail"]
    assert len(detail["job_id"]) == 12
    assert detail["short_id"]
    assert detail["state"]
    human = cli("run")
    assert "was submitted and keeps running" in human.stderr
    monkeypatch.undo()
    for ref in (detail["job_id"],):
        cli("cancel", ref)


def test_run_json_wait_ctrl_c_reports_detached_job(
    cli: Cli, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli.fake(duration=30)

    def interrupt(self: Any, ref: str, *, after: int = 0) -> Any:
        raise KeyboardInterrupt

    monkeypatch.setattr("gpu_router.client.GpuClient.events", interrupt)
    data, res = cli.json("run", "--wait")
    assert res.exit_code == exitcodes.INTERRUPTED
    assert data["detached"] is True
    assert len(data["job"]["id"]) == 12
    monkeypatch.undo()
    cli("cancel", data["job"]["id"])


def test_run_json_wait_says_on_stderr_when_approval_is_needed(cli: Cli) -> None:
    """Review finding: `run --json --wait` on a job awaiting approval sat silent."""
    from gpu_router.client import GpuClient
    from gpu_router.paths import Paths

    cli.write_yaml("script: train.py\nrequires_approval: true\n")

    def deny_soon() -> None:
        with GpuClient.from_env(Paths(cli.home)) as client:
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                found = client.jobs(project_dir=str(cli.project.resolve()), limit=5).jobs
                waiting = [j for j in found if j.state == "awaiting_approval"]
                if waiting:
                    time.sleep(0.5)  # let the CLI poll see the state first
                    client.deny(waiting[0].id)
                    return
                time.sleep(0.1)

    t = threading.Thread(target=deny_soon, daemon=True)
    t.start()
    data, res = cli.json("run", "--wait")
    t.join(20)
    assert res.exit_code == exitcodes.JOB_STOPPED
    assert data["job"]["state"] == "denied"
    assert "waiting for approval: gpu approve" in res.stderr


# --------------------------------------------------------------------------- guarded / fetch


def test_unexpected_exception_is_an_internal_envelope(
    cli: Cli, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(self: Any) -> Any:
        raise RuntimeError("kaboom")

    monkeypatch.setattr("gpu_router.client.GpuClient.status", broken)
    data, res = cli.json("status")
    assert res.exit_code == exitcodes.ERROR
    assert data["error"]["code"] == "internal"
    assert "kaboom" in data["error"]["message"]
    res = cli("status")
    assert res.exit_code == exitcodes.ERROR
    assert "internal error: RuntimeError: kaboom" in res.stderr


def test_fetch_dest_equal_to_outputs_dir_is_fine(cli: Cli) -> None:
    job = cli.json("run", "--wait")[0]["job"]
    outputs = job["outputs_dir"]
    data, res = cli.json("fetch", job["short_id"], "--dest", outputs)
    assert res.exit_code == 0, res.output
    assert data["fetched"] is True
    assert data["dest"] == str(Path(outputs).resolve())


def test_fetch_dest_inside_outputs_or_unwritable_reports_it(cli: Cli, tmp_path: Path) -> None:
    job = cli.json("run", "--wait")[0]["job"]
    inside = Path(job["outputs_dir"]) / "sub"
    data, res = cli.json("fetch", job["short_id"], "--dest", str(inside))
    assert res.exit_code == exitcodes.ERROR
    assert data["fetched"] is True
    assert data["dest"] is None
    assert "inside the outputs dir" in data["message"]

    ro = tmp_path / "ro"
    ro.mkdir()
    ro.chmod(0o500)
    try:
        data, res = cli.json("fetch", job["short_id"], "--dest", str(ro / "copy"))
    finally:
        ro.chmod(0o700)
    assert res.exit_code == exitcodes.ERROR
    assert data["fetched"] is True
    assert data["dest"] is None
    assert "copied nothing" in data["message"]


# --------------------------------------------------------------------------- providers


def test_unknown_provider_is_exit_4_with_suggestion(cli: Cli) -> None:
    before, _ = cli.json("jobs", "--all", "--limit", "500")
    data, res = cli.json("run", "-p", "fakee")
    assert res.exit_code == exitcodes.NOT_FOUND
    assert data["error"]["code"] == "provider_not_found"
    assert "did you mean fake?" in data["error"]["hint"]
    after, _ = cli.json("jobs", "--all", "--limit", "500")
    assert len(after["jobs"]) == len(before["jobs"])  # nothing submitted
    res = cli("route", "-p", "kagle")
    assert res.exit_code == exitcodes.NOT_FOUND
    assert "no provider named 'kagle'" in res.stderr
    # gpu.yaml provider, mixed case, is lowercased and accepted
    cli.write_yaml("script: train.py\nprovider: FAKE\nprovider_options:\n  fake: {duration: 0.3}\n")
    data, res = cli.json("run", "--dry-run")
    assert res.exit_code == 0, res.output
    assert data["spec"]["provider"] == "fake"


# --------------------------------------------------------------------------- lists


def test_empty_states_match_the_filter(cli: Cli) -> None:
    res = cli("history", "--failed", "--here")
    assert "no failed jobs from this project." in res.stdout
    res = cli("history", "--here")
    assert "no past jobs from this project." in res.stdout
    res = cli("jobs", "--all", "--here")
    assert "no jobs from this project." in res.stdout


def test_jobs_are_paged_not_silently_cut(cli: Cli) -> None:
    first = cli.json("run", "--wait")[0]["job"]
    second = cli.json("run", "--wait")[0]["job"]
    page, res = cli.json("history", "--here", "-n", "1")
    assert res.exit_code == 0
    assert [j["id"] for j in page["jobs"]] == [second["id"]]
    assert page["next_before"]
    rest, res = cli.json("history", "--here", "-n", "1", "--before", page["next_before"])
    assert res.exit_code == 0
    assert [j["id"] for j in rest["jobs"]] == [first["id"]]
    human = cli("history", "--here", "-n", "1")
    assert "… more: gpu history --here -n 2" in human.stdout
    assert "--before" in human.stdout
    human = cli("jobs", "--all", "--here", "-n", "1")
    assert "… more: gpu jobs --all --here -n 2" in human.stdout


def test_ambiguous_prefix_lists_candidates() -> None:
    from gpu_router.cli.app import Out
    from gpu_router.errors import AmbiguousJobRef

    ids = [f"a{i:011x}" for i in range(7)]
    exc = AmbiguousJobRef("'a' matches more than one job", detail={"matches": ids})
    lines = Out(False)._matches(exc)
    assert lines[:5] == ids[:5]
    assert lines[5] == "... and 2 more"


# --------------------------------------------------------------------------- secrets


def test_secrets_set_list_rm(cli: Cli) -> None:
    """The hint for secret-looking env vars points at `gpu secrets set`; it exists now."""
    from gpu_router.cli.app import app

    res = cli.runner.invoke(
        app,
        ["secrets", "set", "API_TOKEN", "--stdin", "--json"],
        input="s3cr3t-value-123\n",
        catch_exceptions=False,
    )
    assert res.exit_code == 0, res.output
    assert json.loads(res.stdout) == {"secret": "API_TOKEN", "stored": True}
    assert "s3cr3t" not in res.output
    from gpu_router import secrets

    assert secrets.get_secret("API_TOKEN") == "s3cr3t-value-123"
    res = cli("secrets", "list", "--json")
    assert "API_TOKEN" in json.loads(res.stdout)["secrets"]
    res = cli("secrets", "rm", "API_TOKEN", "--json")
    assert json.loads(res.stdout) == {"secret": "API_TOKEN", "removed": True}
    assert secrets.get_secret("API_TOKEN") is None
    res = cli("secrets", "set", "bad-name", "--json")
    assert res.exit_code == exitcodes.USAGE
    assert json.loads(res.stdout)["error"]["code"] == "invalid_request"


# --------------------------------------------------------------------------- daemon --json


def test_daemon_stop_and_install_take_json(cli: Cli, tmp_path: Path) -> None:
    res = cli("daemon", "install-launchd", "--print", "--json")
    assert res.exit_code == 0
    body = json.loads(res.stdout)
    assert body["installed"] is False
    assert "dev.gpu-router.daemon" in body["plist"]
    env_home = tmp_path / "nobody-home"
    res = _gpu(["daemon", "stop", "--json"], env_home, tmp_path)
    assert res.returncode == 3
    assert json.loads(res.stdout)["running"] is False
    res = _gpu(["daemon", "stop", "--bogus", "--json"], env_home, tmp_path)
    assert res.returncode == 2
    assert json.loads(res.stdout)["error"]["code"] == "invalid_request"


# --------------------------------------------------------------------------- spawn


def test_autostart_reports_a_child_that_dies_right_away(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review finding: a daemon that could not listen made the CLI wait the full 20 s and
    say only "did not answer"; now it fails fast with the child's own message."""
    from gpu_router.daemon.spawn import connect
    from gpu_router.paths import Paths

    home = tmp_path / "home"
    home.mkdir()
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen()
    port = sock.getsockname()[1]
    monkeypatch.setenv("GPU_ROUTER_HOME", str(home))
    monkeypatch.setenv("GPU_ROUTER_PORT", str(port))
    monkeypatch.setattr("gpu_router.daemon.launchd.is_installed", lambda home=None: False)
    start = time.monotonic()
    try:
        with pytest.raises(DaemonUnavailable) as ei:
            connect(Paths(home), wait_s=20)
    finally:
        sock.close()
    assert time.monotonic() - start < 15
    assert "exited right away" in ei.value.message
    assert "cannot listen" in ei.value.message


def test_started_only_when_the_ready_daemon_is_our_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A CLI that lost the start race (its child exits 3: lock held) keeps waiting for the
    winner's daemon and reports started=False."""
    from gpu_router.api import RuntimeInfo
    from gpu_router.client import GpuClient
    from gpu_router.daemon import spawn as spawn_mod
    from gpu_router.paths import Paths

    loser = subprocess.Popen([sys.executable, "-c", "import sys; sys.exit(3)"])
    loser.wait()
    calls = {"n": 0}
    ready = GpuClient("http://127.0.0.1:1", "tok")

    def fake_try(paths: Paths) -> tuple[GpuClient | None, bool]:
        calls["n"] += 1
        return (ready, True) if calls["n"] >= 4 else (None, False)

    monkeypatch.setattr(spawn_mod, "_try_connect", fake_try)
    monkeypatch.setattr(spawn_mod, "spawn", lambda p: spawn_mod.Spawned("process", loser, 0))
    monkeypatch.setattr(
        spawn_mod,
        "read_runtime_info",
        lambda p: RuntimeInfo.model_validate(
            {"pid": loser.pid + 100000, "port": 1, "version": "x", "started_at": 0}
        ),
    )
    monkeypatch.setenv("GPU_ROUTER_NO_AUTOSTART", "0")
    conn = spawn_mod.connect(Paths(tmp_path), wait_s=5)
    assert conn.started is False
