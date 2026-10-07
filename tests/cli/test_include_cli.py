"""D60 through the real CLI against a daemon subprocess: `gpu run --include` (before the
script) in a dry run and a real submit, and the dry run's left-out lines."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.cli.conftest import Cli
from tests.unit.packaging.helpers import git, isolate_git, write_files


@pytest.fixture
def repo(cli: Cli, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Cli:
    isolate_git(monkeypatch, tmp_path)
    (cli.project / ".gitignore").write_text("third_party/\ndata/\n")
    write_files(cli.project, {"third_party/foo/m.py": "x = 1\n", "data/d.bin": b"0" * 10})
    git(cli.project, "init", "-q")
    return cli


def test_dry_run_with_include(repo: Cli) -> None:
    data, res = repo.json("run", "--dry-run", "--include", "third_party/")
    assert res.exit_code == 0, res.output
    assert data["spec"]["include"] == ["third_party/"]
    assert data["bundle"]["included"] == {"files": 1, "bytes": 6}
    assert data["bundle"]["left_out"] == ["data/ (10 B, ignored)"]
    human = repo("run", "--dry-run", "train.py")
    assert "left out: data/ (10 B, ignored), third_party/ (6 B, ignored)" in human.stdout


def test_submit_with_include_records_it(repo: Cli) -> None:
    data, res = repo.json("run", "--detach", "--include", "third_party/", "train.py")
    assert res.exit_code == 0, res.output
    assert data["job"]["spec"]["include"] == ["third_party/"]
    plain, _ = repo.json("run", "--detach", "train.py", "--include", "x")
    assert "include" not in plain["job"]["spec"]  # after the script it is the script's
    assert plain["job"]["spec"]["args"] == ["--include", "x"]
