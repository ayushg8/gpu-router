"""The bundle summary does not tell an agent a path "does not ship" when the job already
passes it as data= (2026-10-04 field test: `data/ (36 B, ignored)` next to `data/rows`)."""

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
        tmp_path / "proj", {"train.py": "print(1)\n", ".gitignore": "data/\nweights/\n"}
    )
    (project / "data" / "rows").mkdir(parents=True)
    (project / "data" / "rows" / "a.csv").write_text("a\n")
    (project / "data" / "other.csv").write_text("b\n")
    (project / "weights").mkdir()
    (project / "weights" / "w.bin").write_bytes(b"w")
    return project


def test_a_data_path_is_not_left_out(tmp_path: Path) -> None:
    project = _project(tmp_path)
    plain = bundle_summary(JobSpec(project_dir=str(project), script="train.py"))
    assert any(x.startswith("data/") for x in plain["left_out"])
    spec = JobSpec(
        project_dir=str(project),
        script="train.py",
        data=[DataRef(mount="w", path="weights"), DataRef(mount="rows", path="data/rows")],
    )
    out = bundle_summary(spec)
    assert not any(x.startswith("weights/") for x in out["left_out"])
    (data,) = [x for x in out["left_out"] if x.startswith("data/")]
    assert data.endswith("ignored; data/rows/ is passed as data=)")


def test_nothing_left_means_no_hint(tmp_path: Path) -> None:
    project = _project(tmp_path)
    spec = JobSpec(
        project_dir=str(project),
        script="train.py",
        data=[DataRef(mount="d", path="data"), DataRef(mount="w", path=str(project / "weights"))],
    )
    out = bundle_summary(spec)
    assert "left_out" not in out
    assert "hint" not in out
