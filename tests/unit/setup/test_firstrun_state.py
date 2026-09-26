"""First-run detection (stdlib only), setup.json (atomic, resumable) and the bare-`gpu`
dispatch in entry.py."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from gpu_router import entry
from gpu_router.paths import Paths
from gpu_router.setup import firstrun
from gpu_router.setup.state import SetupState
from tests.unit.setup.conftest import Sandbox, ScriptedUi


def _env(home: Path, **extra: str) -> dict[str, str]:
    return {"GPU_ROUTER_HOME": str(home), **extra}


def test_firstrun_module_is_stdlib_only() -> None:
    code = (
        "import sys; import gpu_router.setup.firstrun as f; "
        "bad = [m for m in ('pydantic', 'typer', 'rich', 'httpx', 'textual', 'fastapi') "
        "if m in sys.modules]; print(','.join(bad))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == ""


def test_first_run_mode_cases(paths: Paths) -> None:
    home = paths.home
    assert firstrun.first_run_mode(_env(home)) == "new"
    assert firstrun.first_run_mode(_env(home, GPU_ROUTER_TEST_MODE="1")) is None
    assert firstrun.first_run_mode(_env(home, GPU_ROUTER_NO_SETUP="yes")) is None
    f = home / "setup.json"
    f.write_text(json.dumps({"run": {"started_at": 1.0, "finished_at": None}}))
    assert firstrun.first_run_mode(_env(home)) == "resume"
    f.write_text(json.dumps({"run": {"started_at": 1.0, "finished_at": 2.0}}))
    assert firstrun.first_run_mode(_env(home)) is None
    f.write_text(json.dumps({"completed_at": 3.0, "run": {"started_at": 4.0}}))
    assert firstrun.first_run_mode(_env(home)) is None
    f.write_text(json.dumps({"dismissed_at": 3.0}))
    assert firstrun.first_run_mode(_env(home)) is None
    f.write_text("{broken")  # a damaged file never loops the prompt
    assert firstrun.first_run_mode(_env(home)) is None


def test_state_round_trip_is_private_and_atomic(paths: Paths) -> None:
    s = SetupState.load(paths.home)
    s.begin(100.0)
    s.record_answer("integration.plugin", False)
    s.record_item("tools.kaggle", "done", "installed", 101.0)
    s.record_smoke("kaggle", {"ok": True, "gpu": "Tesla T4"})
    f = paths.home / "setup.json"
    assert f.stat().st_mode & 0o777 == 0o600
    again = SetupState.load(paths.home)
    assert again.answer("integration.plugin") == "no"
    assert again.items["tools.kaggle"]["outcome"] == "done"
    assert again.smoke["kaggle"]["gpu"] == "Tesla T4"
    assert again.interrupted
    assert not list(paths.home.glob(".setup.json.*"))


def test_resume_keeps_answers_until_the_run_finishes(paths: Paths) -> None:
    s = SetupState.load(paths.home)
    s.begin(1.0)
    run_id = s.run["id"]
    s.record_answer("launchd", False)
    resumed = SetupState.load(paths.home)
    resumed.begin(2.0)
    assert resumed.resumed
    assert resumed.run["id"] == run_id
    assert resumed.answer("launchd") == "no"
    resumed.finish(3.0, complete=True)
    assert resumed.completed_at == 3.0
    fresh = SetupState.load(paths.home)
    fresh.begin(4.0)
    assert not fresh.resumed
    assert fresh.run["id"] != run_id
    assert fresh.answer("launchd") is None


def test_subset_runs_neither_resume_nor_end_the_recorded_run(paths: Paths) -> None:
    s = SetupState.load(paths.home)
    s.begin(1.0)
    s.record_answer("launchd", False)
    sub = SetupState.load(paths.home)
    sub.begin_subset()
    assert sub.answer("launchd") is None  # --only asks again
    sub.record_answer("launchd", True)
    sub.finish(2.0, complete=False)
    after = SetupState.load(paths.home)
    assert after.interrupted
    assert after.answer("launchd") == "no"
    assert after.completed_at is None


# --------------------------------------------------------------------------- entry.py


def _dispatch(monkeypatch: pytest.MonkeyPatch, *, first_run: Any) -> list[str]:
    calls: list[str] = []
    monkeypatch.setattr(sys, "argv", ["gpu"])
    monkeypatch.setattr(entry, "_is_tty", lambda: True)
    monkeypatch.setattr("gpu_router.setup.wizard.first_run", first_run)
    monkeypatch.setattr("gpu_router.shell.app.run", lambda: calls.append("shell") or 0)
    with pytest.raises(SystemExit) as info:
        entry.main()
    calls.append(f"exit {info.value.code}")
    return calls


def test_bare_gpu_offers_the_wizard_first_then_opens_the_shell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GPU_ROUTER_TEST_MODE")
    seen: list[str] = []

    def wizard(mode: str) -> bool:
        seen.append(mode)
        return True

    assert _dispatch(monkeypatch, first_run=wizard) == ["shell", "exit 0"]
    assert seen == ["new"]


def test_bare_gpu_does_not_open_the_shell_when_the_user_says_no(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GPU_ROUTER_TEST_MODE")
    assert _dispatch(monkeypatch, first_run=lambda _m: False) == ["exit 0"]


def test_a_broken_wizard_never_blocks_the_shell(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("GPU_ROUTER_TEST_MODE")

    def boom(_mode: str) -> bool:
        raise RuntimeError("kaboom")

    assert _dispatch(monkeypatch, first_run=boom) == ["shell", "exit 0"]
    assert "gpu setup" in capsys.readouterr().err


def test_test_mode_and_finished_setups_go_straight_to_the_shell(
    monkeypatch: pytest.MonkeyPatch, paths: Paths
) -> None:
    def never(_mode: str) -> bool:
        raise AssertionError("wizard offered")

    assert _dispatch(monkeypatch, first_run=never) == ["shell", "exit 0"]  # test mode
    monkeypatch.delenv("GPU_ROUTER_TEST_MODE")
    (paths.home / "setup.json").write_text(json.dumps({"completed_at": 5.0}))
    assert _dispatch(monkeypatch, first_run=never) == ["shell", "exit 0"]


def test_first_run_not_now_is_remembered(
    monkeypatch: pytest.MonkeyPatch, paths: Paths, sandbox: Sandbox
) -> None:
    from gpu_router.setup import wizard

    monkeypatch.delenv("GPU_ROUTER_TEST_MODE")
    ui = ScriptedUi([False])
    monkeypatch.setattr("gpu_router.setup.ui.TerminalUi", lambda *a, **k: ui)
    assert wizard.first_run("new") is True  # dismissed: the shell still opens
    assert "not set up yet" in ui.questions[0]
    assert firstrun.first_run_mode() is None
    assert SetupState.load(paths.home).dismissed_at is not None
