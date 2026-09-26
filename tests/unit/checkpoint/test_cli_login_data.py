"""`gpu login hf` (token into the Keychain, never argv) and `gpu run --data` (spec merge)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from gpu_router import secrets
from gpu_router.cli import app as cli_app
from gpu_router.cli import login as login_mod
from gpu_router.cli.app import app, split_run_argv
from gpu_router.errors import InvalidRequest
from gpu_router.paths import Paths

TOKEN = "hf_" + "L" * 34


def _invoke(*args: str, stdin: str | None = None) -> Any:
    return CliRunner().invoke(app, list(args), input=stdin, catch_exceptions=False)


def test_login_hf_from_stdin_stores_the_token_and_the_owner(
    paths: Paths, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(login_mod, "_whoami", lambda token: ("tester", None))
    res = _invoke("login", "hf", "--stdin", "--json", stdin=TOKEN + "\n")
    assert res.exit_code == 0, res.output
    out = json.loads(res.stdout)
    assert out == {
        "secret": "HF_TOKEN",
        "stored": True,
        "user": "tester",
        "verified": True,
        "source": "stdin",
    }
    assert secrets.get_secret("HF_TOKEN") == TOKEN
    assert TOKEN not in res.output
    cache = json.loads((paths.home / "storage" / "hf.json").read_text())
    assert cache["namespace"] == "tester"
    assert TOKEN not in json.dumps(cache)


def test_login_hf_remote_token(paths: Paths, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(login_mod, "_whoami", lambda token: ("tester", None))
    res = _invoke("login", "hf", "--remote", "--stdin", stdin=TOKEN)
    assert res.exit_code == 0, res.output
    assert secrets.get_secret("HF_TOKEN_REMOTE") == TOKEN
    assert secrets.get_secret("HF_TOKEN") is None
    assert "remote runtimes" in res.stdout


@pytest.mark.parametrize(
    ("value", "why"),
    [("", "empty"), ("not-a-token", "start with hf_"), ("hf_ab cd" + "x" * 20, "whitespace")],
)
def test_login_hf_refuses_what_is_not_a_token(paths: Paths, value: str, why: str) -> None:
    res = _invoke("login", "hf", "--stdin", "--no-check", "--json", stdin=value)
    assert res.exit_code == 2
    assert why in json.loads(res.stdout)["error"]["message"]
    assert secrets.get_secret("HF_TOKEN") is None


def test_login_hf_rejected_token_is_not_stored(
    paths: Paths, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        login_mod, "_whoami", lambda token: (None, "hugging face rejected this token")
    )
    res = _invoke("login", "hf", "--stdin", "--json", stdin=TOKEN)
    assert res.exit_code == 2
    assert "rejected" in json.loads(res.stdout)["error"]["message"]
    assert secrets.get_secret("HF_TOKEN") is None


def test_login_hf_offline_stores_anyway(paths: Paths, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(login_mod, "_whoami", lambda token: (None, None))
    res = _invoke("login", "hf", "--stdin", stdin=TOKEN)
    assert res.exit_code == 0
    assert "could not reach hugging face" in res.stdout
    assert secrets.get_secret("HF_TOKEN") == TOKEN


def test_login_hf_import(paths: Paths, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(login_mod, "_whoami", lambda token: ("tester", None))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "nohf"))
    monkeypatch.delenv("HF_TOKEN_PATH", raising=False)
    res = _invoke("login", "hf", "--import", "--json")
    assert res.exit_code == 2
    assert "no token found" in json.loads(res.stdout)["error"]["message"]
    token_file = tmp_path / "nohf" / "token"
    token_file.parent.mkdir()
    token_file.write_text(TOKEN + "\n")
    res = _invoke("login", "hf", "--import", "--json")
    assert res.exit_code == 0, res.output
    assert json.loads(res.stdout)["source"] == str(token_file)
    assert secrets.get_secret("HF_TOKEN") == TOKEN


def test_login_help_never_takes_the_token_as_an_argument() -> None:
    res = _invoke("login", "hf", "--help")
    assert res.exit_code == 0
    assert "TOKEN" not in res.stdout.split("Options")[0]  # no positional token


# --------------------------------------------------------------------------- --data


def test_data_flag_merges_into_the_spec(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = tmp_path / "proj"
    (project / "data" / "coco").mkdir(parents=True)
    (project / "train.py").write_text("print(1)\n")
    (project / "gpu.yaml").write_text(
        "script: train.py\ndata:\n  - {mount: coco, path: data/old}\n  - {mount: keep, uri: hf://datasets/a/b}\n"
    )
    other = tmp_path / "My Data.v2"
    other.mkdir()
    monkeypatch.chdir(project)
    spec = cli_app._build(
        None,
        [],
        vram=None,
        hours=None,
        provider=None,
        gpu=None,
        name=None,
        env=None,
        project=None,
        data=["coco=data/coco", str(other), "wiki=hf://datasets/org/wiki@main/en"],
    )
    by_mount = {d.mount: d for d in spec.data}
    assert set(by_mount) == {"coco", "keep", "my-data.v2", "wiki"}
    assert by_mount["coco"].path == str((project / "data" / "coco").resolve())  # flag wins
    assert by_mount["keep"].uri == "hf://datasets/a/b"
    assert by_mount["my-data.v2"].path == str(other.resolve())
    assert by_mount["wiki"].uri == "hf://datasets/org/wiki@main/en"


def test_data_flag_missing_path_is_a_usage_error(tmp_path: Path) -> None:
    with pytest.raises(InvalidRequest, match="does not exist"):
        cli_app._with_data(
            cli_app.JobSpec(project_dir=str(tmp_path), script="t.py"), ["nope"], tmp_path
        )


def test_data_is_a_gpu_option_before_the_script() -> None:
    head, script_args, _ = split_run_argv(["--data", "d", "train.py", "--data", "x"])
    assert head == ["--data", "d", "train.py"]
    assert script_args == ["--data", "x"]  # after the script it belongs to the script
