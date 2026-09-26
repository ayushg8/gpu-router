"""`gpu run` from an AI agent counts as an agent job (D48, finding 2): the agent approval
rules and intake checks apply whether it comes through MCP or the CLI."""

from __future__ import annotations

from pathlib import Path

import pytest

from gpu_router.agent import agent_marker
from tests.cli.conftest import Cli


def test_markers() -> None:
    assert agent_marker({}) is None
    assert agent_marker({"CLAUDECODE": "1"}) == "CLAUDECODE=1"
    assert agent_marker({"CODEX_SANDBOX": "seatbelt"}) == "CODEX_SANDBOX"
    assert agent_marker({"GPU_ROUTER_AGENT": "1"}) == "GPU_ROUTER_AGENT=1"
    assert agent_marker({"CLAUDECODE": "0", "CODEX_HOME": "/x"}) is None


@pytest.mark.parametrize("marker", ["CLAUDECODE", "GPU_ROUTER_AGENT"])
def test_gpu_run_inside_an_agent_is_an_agent_job(
    cli: Cli, monkeypatch: pytest.MonkeyPatch, marker: str
) -> None:
    monkeypatch.setenv(marker, "1")
    data, res = cli.json("run")
    assert res.exit_code == 0
    job = data["job"]
    assert job["source"] == "agent"
    assert job["spec"]["labels"]["via"] == "cli-agent"
    assert "submitting as an agent job" in res.stderr


def test_as_agent_flag_before_or_after_the_script(cli: Cli) -> None:
    data, _res = cli.json("run", "--as-agent", "train.py")
    assert data["job"]["source"] == "agent"
    data, _res = cli.json("run", "train.py", "--as-agent")
    assert data["job"]["source"] == "agent"
    assert data["job"]["spec"]["args"] == []  # not passed to the script


def test_plain_gpu_run_stays_a_user_job(cli: Cli) -> None:
    data, _res = cli.json("run")
    assert data["job"]["source"] == "cli"


def test_agent_cli_jobs_get_the_intake_checks(
    cli: Cli, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    home = tmp_path / "fakehome"
    (home / ".config" / "gh").mkdir(parents=True)
    (home / ".config" / "gh" / "hosts.yml").write_text("oauth_token: x")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("CLAUDECODE", "1")
    res = cli("run", "--json", "--data", f"c={home / '.config'}", "train.py")
    assert res.exit_code != 0
    assert "credential store" in res.stdout + res.stderr
