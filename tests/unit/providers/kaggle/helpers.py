"""Helpers for Kaggle adapter tests: real bundles from throwaway projects."""

from __future__ import annotations

from pathlib import Path

from gpu_router.models import JobSpec
from gpu_router.packaging.bundle import build_bundle
from gpu_router.paths import Paths
from tests.unit.packaging.helpers import make_project

TRAIN_OK = """\
import os, pathlib
out = pathlib.Path(os.environ["GPU_OUTPUT_DIR"])
out.mkdir(parents=True, exist_ok=True)
(out / "result.txt").write_text("ok\\n")
(out / "sub").mkdir(exist_ok=True)
(out / "sub" / "nested.txt").write_text("nested\\n")
print("hello from train.py")
"""


def make_bundle(
    root: Path,
    paths: Paths,
    *,
    script: str = TRAIN_OK,
    name: str = "train.py",
    extra: dict[str, str | bytes] | None = None,
) -> Path:
    """Build a real job bundle (git project, content-addressed cache) and return the
    archive path."""
    project = make_project(root, {name: script, **(extra or {})})
    spec = JobSpec(project_dir=str(project), script=name, checkpoint_interval_min=0)
    return build_bundle(project, spec, paths=paths).archive
