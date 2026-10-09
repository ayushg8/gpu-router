"""A whole command line passed as the script (2026-10-08, job ca1e: an agent's
`script="bash .../build.sh"` became the one-word command "bash .../build.sh", exec'd as a
program of that name: exit 127) is split like a shell would."""

from __future__ import annotations

from pathlib import Path

import pytest

from gpu_router.errors import InvalidSpec
from gpu_router.jobspec import Flags, build_spec
from tests.unit.packaging.helpers import isolate_git, make_project


@pytest.fixture
def proj(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    isolate_git(monkeypatch, tmp_path)
    p = make_project(
        tmp_path / "proj",
        {"train.py": "print(1)\n", "jobs/build.sh": "echo hi\n", "my run.sh": "echo 2\n"},
    )
    monkeypatch.chdir(p)
    return p


def test_a_command_line_becomes_argv(proj: Path) -> None:
    spec, _, _ = build_spec(Flags(script="bash jobs/build.sh --fast", args=["x"]), cwd=proj)
    assert spec.script is None
    assert spec.command == ["bash", "jobs/build.sh", "--fast", "x"]
    spec, _, _ = build_spec(Flags(script="bash 'my run.sh'"), cwd=proj)
    assert spec.command == ["bash", "my run.sh"]


def test_a_python_command_line_keeps_its_arguments(proj: Path) -> None:
    spec, _, _ = build_spec(Flags(script="train.py --epochs 3", args=["--lr", "0.1"]), cwd=proj)
    assert spec.script == "train.py"
    assert spec.args == ["--epochs", "3", "--lr", "0.1"]
    spec, _, _ = build_spec(Flags(script="train.py", args=["--epochs", "3"]), cwd=proj)
    assert spec.args == ["--epochs", "3"]


def test_a_file_whose_name_has_a_space_stays_one_word(proj: Path) -> None:
    spec, _, _ = build_spec(Flags(script="my run.sh"), cwd=proj)
    assert spec.command == ["my run.sh"]


def test_an_unbalanced_quote_is_a_clear_error(proj: Path) -> None:
    with pytest.raises(InvalidSpec) as info:
        build_spec(Flags(script="bash 'jobs/build.sh"), cwd=proj)
    assert "cannot split the command line" in info.value.message
    assert info.value.hint is not None
