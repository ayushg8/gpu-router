"""`gpu notify [status|test]` (phase 8a). Never shows a real notification: under pytest the
backend is the null one, and the sending test swaps in a recorder."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gpu_router.cli.app import main
from gpu_router.notify import cli as notify_cli
from gpu_router.notify.backends import NotifyError, NullBackend
from gpu_router.notify.format import Notification
from gpu_router.paths import Paths


def run_json(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, dict[str, object]]:
    code = main([*argv, "--json"])
    out = capsys.readouterr().out
    return code, json.loads(out)


def test_status_shows_settings_and_why_nothing_would_show(
    capsys: pytest.CaptureFixture[str], paths: Paths
) -> None:
    code, doc = run_json(capsys, "notify")
    assert code == 0
    info = doc["notifications"]
    assert isinstance(info, dict)
    assert info["backend"] == "none"
    assert "pytest" in str(info["why_none"])
    assert info["events"] == {"finished": True, "failed": True, "approval": True, "migrated": True}


def test_status_reads_config_yaml(capsys: pytest.CaptureFixture[str], paths: Paths) -> None:
    Path(paths.config).write_text("notifications:\n  events: {finished: false}\n  sound: false\n")
    code, doc = run_json(capsys, "notify", "status")
    info = doc["notifications"]
    assert isinstance(info, dict)
    assert code == 0
    assert info["events"]["finished"] is False
    assert info["sound"] is False


def test_bad_config_is_a_usage_style_error(
    capsys: pytest.CaptureFixture[str], paths: Paths
) -> None:
    Path(paths.config).write_text("notifications:\n  backend: growl\n")
    code = main(["notify", "--json"])
    doc = json.loads(capsys.readouterr().out)
    assert code != 0
    assert "notifications" in doc["error"]["message"]


def test_test_says_nothing_was_sent_under_pytest(
    capsys: pytest.CaptureFixture[str], paths: Paths
) -> None:
    code, doc = run_json(capsys, "notify", "test")
    assert code == 1
    assert doc["sent"] is False
    assert "pytest" in str(doc["why"])


class Rec:
    name = "osascript"

    def __init__(self, fail: bool = False) -> None:
        self.sent: list[Notification] = []
        self.fail = fail

    def send(self, n: Notification) -> None:
        if self.fail:
            raise NotifyError("osascript exited 1: Not authorized")
        self.sent.append(n)


def test_test_sends_one_harmless_notification(
    capsys: pytest.CaptureFixture[str], paths: Paths, monkeypatch: pytest.MonkeyPatch
) -> None:
    rec = Rec()
    monkeypatch.setattr(notify_cli, "_backend", lambda: rec)
    code, doc = run_json(capsys, "notify", "test")
    assert code == 0
    assert doc["sent"] is True
    assert [n.subtitle for n in rec.sent] == ["gpu-router test notification"]


def test_test_reports_a_failing_backend(
    capsys: pytest.CaptureFixture[str], paths: Paths, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(notify_cli, "_backend", lambda: Rec(fail=True))
    code, doc = run_json(capsys, "notify", "test")
    assert code == 1
    assert "Not authorized" in str(doc["why"])


def test_human_status(capsys: pytest.CaptureFixture[str], paths: Paths) -> None:
    assert main(["notify"]) == 0
    out = capsys.readouterr().out
    assert "notifications off" in out
    assert "gpu notify test" in out


def test_null_backend_type_is_what_status_reports() -> None:
    assert NullBackend().name == "none"
