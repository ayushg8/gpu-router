from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import BaseModel, ValidationError

from gpu_router.models import (
    DataRef,
    JobPatch,
    JobSpec,
    Progress,
    Timestamp,
    to_iso,
)


class _T(BaseModel):
    ts: Timestamp


def test_timestamp_forms() -> None:
    assert _T(ts=1_790_000_000).ts == 1_790_000_000.0
    assert _T(ts="2026-09-21T14:13:20Z").ts == 1_790_000_000.0
    assert _T(ts=datetime(2026, 9, 21, 14, 13, 20, tzinfo=UTC)).ts == 1_790_000_000.0
    assert _T(ts=1_790_000_000).model_dump(mode="json") == {"ts": "2026-09-21T14:13:20.000Z"}
    assert _T(ts=1.5).model_dump() == {"ts": 1.5}
    assert to_iso(0) == "1970-01-01T00:00:00.000Z"
    for bad in (True, "2026-09-21T14:13:20", datetime(2026, 1, 1)):
        with pytest.raises(ValidationError):
            _T(ts=bad)


def test_jobspec_minimal_and_display_name() -> None:
    s = JobSpec(project_dir="/p/", script="./src/train.py")
    assert s.project_dir == "/p"
    assert s.script == "src/train.py"
    assert s.display_name() == "train"
    assert JobSpec(project_dir="/p", command=["python", "-m", "x"]).display_name() == "python"
    assert JobSpec(project_dir="/p", script="t.py", name="n").display_name() == "n"


@pytest.mark.parametrize(
    "data",
    [
        {"project_dir": "rel", "script": "t.py"},
        {"project_dir": "/p"},
        {"project_dir": "/p", "script": "t.py", "command": ["x"]},
        {"project_dir": "/p", "command": []},
        {"project_dir": "/p", "script": "../t.py"},
        {"project_dir": "/p", "script": "/abs/t.py"},
        {"project_dir": "/p", "script": "t.py", "env": {"HF_TOKEN": "x"}},
        {"project_dir": "/p", "script": "t.py", "env": {"MY_API_KEY": "x"}},
        {"project_dir": "/p", "script": "t.py", "env": {"1BAD": "x"}},
        {"project_dir": "/p", "script": "t.py", "secrets": ["bad-name"]},
        {"project_dir": "/p", "script": "t.py", "provider": "Kaggle!"},
        {"project_dir": "/p", "script": "t.py", "vram_gb": 0},
        {"project_dir": "/p", "script": "t.py", "unknown": 1},
    ],
)
def test_jobspec_rejects(data: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        JobSpec.model_validate(data)


def test_jobspec_is_frozen_and_roundtrips() -> None:
    s = JobSpec(project_dir="/p", script="t.py", env={"BATCH": "8"}, secrets=["HF_TOKEN"])
    with pytest.raises(ValidationError):
        s.name = "x"  # type: ignore[misc]
    assert JobSpec.model_validate_json(s.model_dump_json()) == s


def test_dataref_needs_one_source() -> None:
    DataRef(mount="d", path="data/")
    DataRef(mount="d", uri="hf://datasets/a/b")
    with pytest.raises(ValidationError):
        DataRef(mount="d")
    with pytest.raises(ValidationError):
        DataRef(mount="d", path="x", uri="y")


def test_progress_fraction() -> None:
    assert Progress(step=5, total=10).fraction == 0.5
    assert Progress(step=15, total=10).fraction == 1.0
    assert Progress(step=5).fraction is None
    assert Progress(step=5, total=0).fraction is None


def test_patch_fields_set_semantics() -> None:
    assert JobPatch().model_fields_set == set()
    assert JobPatch(provider=None).model_fields_set == {"provider"}
    with pytest.raises(ValidationError):
        JobPatch(state="done")  # type: ignore[call-arg]
