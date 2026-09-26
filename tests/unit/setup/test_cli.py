"""The Typer front ends: `gpu setup`, bare `gpu login`, `gpu login kaggle|colab`. The
SetupEnv is the sandbox's (monkeypatched `setup.cli._env`), so nothing real is touched."""

from __future__ import annotations

import json
from typing import Any

import pytest
from typer.testing import CliRunner

from gpu_router import secrets
from gpu_router.cli.app import app
from gpu_router.setup import cli as setup_cli
from gpu_router.setup import logins
from tests.unit.setup.conftest import KAGGLE_KEY, Sandbox, fail

KAGGLE_DOC = json.dumps({"username": "demo-user", "key": KAGGLE_KEY})


@pytest.fixture(autouse=True)
def _sandboxed(sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(setup_cli, "_env", lambda: sandbox.env)


def _invoke(args: list[str], **kw: Any) -> Any:
    return CliRunner().invoke(app, args, catch_exceptions=False, **kw)


def test_help_lists_setup_and_every_login() -> None:
    assert "setup" in _invoke(["--help"]).output
    res = _invoke(["login", "--help"])
    for name in ("kaggle", "colab", "lightning", "hf", "groq", "gemini", "cloudflare"):
        assert name in res.output
    assert "--stdin" in _invoke(["login", "kaggle", "--help"]).output


def test_setup_dry_run_json_keeps_stdout_one_document(sandbox: Sandbox) -> None:
    res = _invoke(["setup", "--dry-run", "--json"])
    doc = json.loads(res.stdout)
    assert {i["id"] for i in doc["items"]} >= {"tools.kaggle", "launchd", "check.summary"}
    assert "gpu-router setup" in res.stderr
    assert not (sandbox.paths.home / "setup.json").exists()


def test_setup_only_bad_value_is_a_usage_error() -> None:
    res = _invoke(["setup", "--only", "bogus", "--json"])
    assert res.exit_code == 2
    assert json.loads(res.stdout)["error"]["code"] == "invalid_request"


def test_setup_only_one_item(sandbox: Sandbox) -> None:
    sandbox.tools["codex"] = "/fake/bin/codex"
    res = _invoke(["setup", "--only", "codex", "--yes", "--json"])
    doc = json.loads(res.stdout)
    assert [i["id"] for i in doc["items"]] == ["integration.codex"]
    assert doc["items"][0]["outcome"] == "done"
    assert (sandbox.user_home / ".codex" / "config.toml").is_file()


def test_bare_login_lists_every_login(sandbox: Sandbox) -> None:
    sandbox.write(".kaggle/kaggle.json", KAGGLE_DOC)
    res = _invoke(["login"])
    assert res.exit_code == 0
    assert "kaggle" in res.output
    assert "gpu login lightning --browser" in res.output
    doc = json.loads(_invoke(["login", "--json"]).stdout)
    rows = {r["provider"]: r for r in doc["logins"]}
    assert rows["kaggle"]["status"] == "ok"
    assert rows["colab"]["status"] == "missing"
    assert KAGGLE_KEY not in res.output


def test_login_kaggle_stdin_and_file(sandbox: Sandbox, tmp_path: Any) -> None:
    res = _invoke(["login", "kaggle", "--stdin", "--json"], input=KAGGLE_DOC)
    assert res.exit_code == 0, res.output
    doc = json.loads(res.stdout)
    assert doc == {
        "secret": "kaggle",
        "stored": True,
        "user": "demo-user",
        "verified": True,
        "source": "stdin",
        "notes": [],
    }
    assert KAGGLE_KEY not in res.output
    secrets.delete_secret("kaggle")
    f = tmp_path / "kaggle.json"
    f.write_text(KAGGLE_DOC)
    res = _invoke(["login", "kaggle", "--file", str(f)])
    assert res.exit_code == 0
    assert "stored kaggle for demo-user" in res.output
    assert json.loads(secrets.get_secret("kaggle") or "{}")["key"] == KAGGLE_KEY


def test_login_kaggle_rejected(sandbox: Sandbox) -> None:
    sandbox.run.on(r"kaggle -W quota", fail(1, err="401 Unauthorized"))
    res = _invoke(["login", "kaggle", "--stdin", "--json"], input=KAGGLE_DOC)
    assert res.exit_code == 2
    assert "rejected" in json.loads(res.stdout)["error"]["message"]
    assert secrets.get_secret("kaggle") is None


def test_login_colab_reports_and_names_the_command(sandbox: Sandbox) -> None:
    res = _invoke(["login", "colab", "--json"])
    assert res.exit_code == 1
    doc = json.loads(res.stdout)
    assert doc["ok"] is False
    assert doc["command"] == logins.ADC_LOGIN
    assert not doc["ran_gcloud"]
    sandbox.write(".config/gcloud/application_default_credentials.json", "{}")
    ok_res = _invoke(["login", "colab", "--json"])
    assert ok_res.exit_code == 0
    assert json.loads(ok_res.stdout)["ok"] is True


def test_login_colab_run_uses_gcloud(sandbox: Sandbox) -> None:
    sandbox.tools["gcloud"] = "/fake/bin/gcloud"
    sandbox.on_attached = lambda _argv: (
        sandbox.write(".config/gcloud/application_default_credentials.json", "{}") and 0
    )
    res = _invoke(["login", "colab", "--run", "--json"])
    assert res.exit_code == 0, res.output
    assert json.loads(res.stdout)["ran_gcloud"] is True
    assert sandbox.attached
    assert sandbox.attached[0][0] == "/fake/bin/gcloud"
