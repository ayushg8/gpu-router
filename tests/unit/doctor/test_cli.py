"""`gpu doctor` through the real CLI entry (phase 8a). Only checks that stay inside the tmp
home run here (`--only`): the full run is covered with fakes in test_runner_catalog.py, so
no test probes a real provider, ~/.claude or the Keychain."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from gpu_router.cli.app import Out, main
from gpu_router.doctor import cli as doctor_cli
from gpu_router.doctor.model import DriftItem, Report
from gpu_router.paths import Paths
from tests.shell.conftest import InProcDaemon

SAFE = ["--only", "daemon", "--only", "local.config", "--only", "local.perms"]


@pytest.fixture
def daemon(gpu_home: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[InProcDaemon]:
    monkeypatch.setenv("GPU_ROUTER_NO_AUTOSTART", "1")
    d = InProcDaemon(home=gpu_home)
    d.start()
    try:
        yield d
    finally:
        d.stop()


def run_json(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, dict[str, Any]]:
    code = main(["doctor", *argv, "--json"])
    return code, json.loads(capsys.readouterr().out)


def test_json_with_the_daemon_down(
    capsys: pytest.CaptureFixture[str], paths: Paths, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GPU_ROUTER_NO_AUTOSTART", "1")
    code, doc = run_json(capsys, *SAFE)
    assert code == 0
    assert doc["ok"] is True
    by_id = {c["id"]: c for c in doc["checks"]}
    assert by_id["daemon.running"]["status"] == "warn"
    assert by_id["daemon.running"]["fix"] == "gpu daemon start"
    assert by_id["local.perms"]["status"] == "ok"
    assert set(doc["counts"]) == {"ok", "warn", "fail", "skip"}
    assert not paths.runtime.exists()  # doctor never started a daemon


def test_a_failing_check_exits_1(
    capsys: pytest.CaptureFixture[str], paths: Paths, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GPU_ROUTER_NO_AUTOSTART", "1")
    paths.config.write_text("daemon: [\n")
    code, doc = run_json(capsys, *SAFE)
    assert code == 1
    assert doc["ok"] is False
    assert {c["id"]: c["status"] for c in doc["checks"]}["local.config"] == "fail"


def test_against_a_running_daemon(capsys: pytest.CaptureFixture[str], daemon: InProcDaemon) -> None:
    code, doc = run_json(capsys, "--only", "daemon", "--only", "provider.fake")
    assert code == 0, doc
    by_id = {c["id"]: c for c in doc["checks"]}
    assert by_id["daemon.running"]["status"] == "ok"
    assert "test mode" in by_id["daemon.running"]["summary"]
    assert by_id["daemon.version"]["status"] == "ok"
    assert by_id["daemon.state"]["status"] == "ok"
    assert by_id["provider.fake.live"]["status"] == "ok"
    assert by_id["provider.fake-b.live"]["status"] == "ok"


def test_human_output(capsys: pytest.CaptureFixture[str], daemon: InProcDaemon) -> None:
    assert main(["doctor", "--only", "daemon", "-v"]) == 0
    out = capsys.readouterr().out
    assert "gpu doctor" in out
    assert "running" in out
    assert "ms" in out
    assert "ok" in out.splitlines()[-1]


def _report(drift: list[DriftItem]) -> Report:
    return Report(
        version="0.1.0",
        home="/x",
        checked_at=0,
        elapsed_ms=1,
        ok=True,
        counts={"ok": 1, "warn": 0, "fail": 0, "skip": 0},
        checks=[],
        drift=drift,
    )


DRIFT = [DriftItem(provider="kaggle", key="quota.limit", catalog=30, live=29, note="n")]


def test_update_catalog_refuses_without_a_terminal_or_yes(paths: Paths) -> None:
    out = Out(True)
    result, refused = doctor_cli._update_catalog(out, paths, _report(DRIFT), yes=False)
    assert refused
    assert result["changed"] is False
    assert not paths.user_providers.exists()


def test_update_catalog_with_yes_writes(paths: Paths) -> None:
    result, refused = doctor_cli._update_catalog(Out(True), paths, _report(DRIFT), yes=True)
    assert not refused
    assert result["changed"] is True
    assert result["keys"] == ["kaggle.quota.limit"]
    assert "limit: 29" in paths.user_providers.read_text()


def test_update_catalog_without_drift(paths: Paths) -> None:
    result, refused = doctor_cli._update_catalog(Out(True), paths, _report([]), yes=True)
    assert not refused
    assert result == {
        "changed": False,
        "path": str(paths.user_providers),
        "reason": "no drift",
    }


def test_doctor_is_in_help(capsys: pytest.CaptureFixture[str]) -> None:
    main(["--help"])
    out = capsys.readouterr().out
    assert "doctor" in out
    assert "notify" in out
