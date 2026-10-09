"""Arguments naming paths the job will not find (2026-10-08, real agent jobs): job 6124 passed
`--list experiments/math/out/list.txt`, a git-ignored file, and failed on FileNotFoundError;
its retry passed absolute paths on this Mac, which only a local run can see. The bundle
summary an agent gets from gpu_submit / gpu_route now names both."""

from __future__ import annotations

from pathlib import Path

import pytest

from gpu_router.models import DataRef, JobSpec
from gpu_router.packaging.bundle import bundle_summary
from tests.unit.packaging.helpers import isolate_git, make_project


@pytest.fixture(autouse=True)
def _git(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    isolate_git(monkeypatch, tmp_path)


def _project(tmp_path: Path) -> Path:
    project = make_project(
        tmp_path / "proj",
        {"read.py": "print(1)\n", "lists/kept.txt": "a\n", ".gitignore": "out/\nweights/\n"},
    )
    (project / "out").mkdir()
    (project / "out" / "list.txt").write_text("x\n")
    (project / "weights").mkdir()
    (project / "weights" / "w.bin").write_bytes(b"w")
    return project


def _warnings(project: Path, *args: str, **kw: object) -> list[str]:
    spec = JobSpec(project_dir=str(project), script="read.py", args=list(args), **kw)  # type: ignore[arg-type]
    return list(bundle_summary(spec).get("warnings", []))


def test_an_ignored_input_is_named_with_the_fix(tmp_path: Path) -> None:
    project = _project(tmp_path)
    (warning,) = _warnings(project, "--list", "out/list.txt")
    assert warning.startswith("argument out/list.txt: git ignores it")
    assert 'include=["out/list.txt"]' in warning
    (warning,) = _warnings(project, "--list=out/list.txt")  # --flag=value too
    assert "out/list.txt" in warning


def test_shipped_missing_and_data_paths_say_nothing(tmp_path: Path) -> None:
    project = _project(tmp_path)
    assert _warnings(project, "lists/kept.txt", "lists", "out/not-yet.jsonl", "-v", "3") == []
    data = [DataRef(mount="w", path="weights")]
    assert _warnings(project, "weights/w.bin", data=data) == []


def test_absolute_paths_on_this_mac_are_named(tmp_path: Path) -> None:
    project = _project(tmp_path)
    inside = str(project / "lists" / "kept.txt")
    (warning,) = _warnings(project, "--list", inside)
    assert warning.startswith("pass lists/kept.txt (relative to the project) instead of an ")
    assert "absolute path on this Mac" in warning
    assert "include=" not in warning  # it ships; only the path is wrong
    ignored = str(project / "out" / "list.txt")
    (warning,) = _warnings(project, ignored)
    assert warning.startswith('pass out/list.txt (relative to the project) and add include=["')
    outside = tmp_path / "elsewhere.txt"
    outside.write_text("z\n")
    (warning,) = _warnings(project, str(outside))
    assert warning.startswith("pass it as data= or move it into the project: an argument outside")


def test_argument_warnings_come_before_the_untracked_file_list(tmp_path: Path) -> None:
    project = _project(tmp_path)
    for n in range(8):  # untracked, not ignored: each one is named in a warning
        (project / f"new{n}.py").write_text("x\n")
    warnings = _warnings(project, "out/list.txt")
    assert warnings[0].startswith("argument out/list.txt")
