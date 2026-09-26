"""Deps auto-detection and VRAM/hours heuristics."""

from __future__ import annotations

from pathlib import Path

import pytest

from gpu_router.models import DepsSpec, JobSpec
from gpu_router.packaging import BundleError, build_bundle
from gpu_router.packaging.deps import DepsError, detect_deps
from gpu_router.packaging.estimate import DEFAULT_HOURS, DEFAULT_VRAM_GB, estimate
from gpu_router.packaging.files import select_files
from gpu_router.paths import Paths
from tests.unit.packaging.helpers import make_project, write_files

PYPROJECT = """\
[project]
name = "demo"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["torch>=2.2", "transformers==4.44.0"]

[dependency-groups]
dev = ["pytest"]
"""


def _detect(project: Path, spec: DepsSpec | None = None):  # type: ignore[no-untyped-def]
    shipped = {f.rel for f in select_files(project).files}
    return detect_deps(project, spec or DepsSpec(), shipped)


def test_auto_prefers_requirements(tmp_path: Path) -> None:
    project = make_project(
        tmp_path / "p",
        {
            "requirements.txt": "# pinned\ntorch==2.3.0\nnumpy  # math\n\n",
            "pyproject.toml": PYPROJECT,
        },
    )
    info = _detect(project)
    assert info.kind == "requirements"
    assert info.file == "requirements.txt"
    assert info.packages == ["torch==2.3.0", "numpy"]


def test_auto_uses_pyproject_dependencies(tmp_path: Path) -> None:
    project = make_project(tmp_path / "p", {"pyproject.toml": PYPROJECT})
    info = _detect(project)
    assert info.kind == "pyproject"
    assert info.packages == ["torch>=2.2", "transformers==4.44.0"]  # dev group excluded
    assert info.python_requires == ">=3.10"
    assert info.to_manifest()["kind"] == "pyproject"


def test_auto_pyproject_without_deps_is_none(tmp_path: Path) -> None:
    project = make_project(tmp_path / "p", {"pyproject.toml": "[project]\nname='x'\n"})
    assert _detect(project).kind == "none"


def test_auto_nothing_is_none(tmp_path: Path) -> None:
    project = make_project(tmp_path / "p", {"train.py": ""})
    assert _detect(project).kind == "none"


def test_auto_invalid_toml_warns(tmp_path: Path) -> None:
    project = make_project(tmp_path / "p", {"pyproject.toml": "[project\n"})
    info = _detect(project)
    assert info.kind == "none"
    assert info.warnings
    assert "not valid TOML" in info.warnings[0]


def test_explicit_pyproject_invalid_toml_errors(tmp_path: Path) -> None:
    project = make_project(tmp_path / "p", {"pyproject.toml": "[project\n"})
    with pytest.raises(DepsError):
        _detect(project, DepsSpec(kind="pyproject"))


def test_explicit_none(tmp_path: Path) -> None:
    project = make_project(tmp_path / "p", {"requirements.txt": "torch\n"})
    assert _detect(project, DepsSpec(kind="none")).kind == "none"


def test_explicit_custom_requirements_file(tmp_path: Path) -> None:
    project = make_project(tmp_path / "p", {"reqs/gpu.txt": "bitsandbytes\n"})
    info = _detect(project, DepsSpec(kind="requirements", file="reqs/gpu.txt"))
    assert (info.kind, info.file, info.packages) == (
        "requirements",
        "reqs/gpu.txt",
        ["bitsandbytes"],
    )


def test_ignored_requirements_file_is_an_error(tmp_path: Path) -> None:
    project = make_project(tmp_path / "p", {".gitignore": "gpu-reqs.txt\n"})
    write_files(project, {"gpu-reqs.txt": "torch\n"})
    with pytest.raises(DepsError, match="ignored by git"):
        _detect(project, DepsSpec(kind="requirements", file="gpu-reqs.txt"))


def test_ignored_requirements_txt_is_not_auto_detected(tmp_path: Path) -> None:
    project = make_project(tmp_path / "p", {".gitignore": "requirements.txt\n"})
    write_files(project, {"requirements.txt": "torch\n"})
    assert _detect(project).kind == "none"


def test_bundle_records_deps_in_manifest(tmp_path: Path, paths: Paths) -> None:
    project = make_project(tmp_path / "p", {"train.py": "", "requirements.txt": "numpy\n"})
    bundle = build_bundle(
        project, JobSpec(project_dir=str(project), script="train.py"), paths=paths
    )
    assert bundle.manifest["deps"] == {
        "kind": "requirements",
        "file": "requirements.txt",
        "packages": ["numpy"],
        "python_requires": None,
    }


def test_bundle_wraps_deps_error(tmp_path: Path, paths: Paths) -> None:
    project = make_project(tmp_path / "p", {"train.py": ""})
    spec = JobSpec(project_dir=str(project), script="train.py", deps=DepsSpec(kind="pyproject"))
    with pytest.raises(BundleError) as err:
        build_bundle(project, spec, paths=paths)
    assert err.value.hint is not None


# --------------------------------------------------------------------------- estimate


def _estimate(tmp_path: Path, source: str, **spec_fields: object):  # type: ignore[no-untyped-def]
    project = make_project(tmp_path / "p", {"train.py": source})
    body: dict[str, object] = {"project_dir": str(project), "script": "train.py"}
    body.update(spec_fields)
    return estimate(JobSpec.model_validate(body), select_files(project).files)


def test_estimate_defaults_without_hints(tmp_path: Path) -> None:
    est = _estimate(tmp_path, "import torch\nprint('hello')\n")
    assert (est.vram_gb, est.hours) == (DEFAULT_VRAM_GB, DEFAULT_HOURS)
    assert est.vram_source == est.hours_source == "heuristic"
    assert est.mode == "unknown"
    assert est.reasons


def test_spec_values_win(tmp_path: Path) -> None:
    est = _estimate(tmp_path, "m = 'meta-llama/Llama-3-70B'\n", vram_gb=40, hours=2)
    assert (est.vram_gb, est.vram_source, est.hours, est.hours_source) == (40, "spec", 2, "spec")


def test_inference_fp16_7b(tmp_path: Path) -> None:
    est = _estimate(
        tmp_path, "pipe = pipeline('text-generation', model='mistralai/Mistral-7B-v0.3')\n"
    )
    assert est.mode == "inference"
    assert est.model_params_b == 7
    assert est.vram_gb == 18  # 7 * 2 * 1.2 + 1 = 17.8 -> 18


def test_qlora_8b_fits_a_t4(tmp_path: Path) -> None:
    src = (
        "from peft import LoraConfig, get_peft_model\n"
        "model = AutoModelForCausalLM.from_pretrained(\n"
        "    'meta-llama/Llama-3.1-8B', load_in_4bit=True)\n"
    )
    est = _estimate(tmp_path, src)
    assert est.mode == "lora"
    assert est.vram_gb is not None
    assert est.vram_gb <= 16
    assert est.hours == 3.0


def test_full_training_small_model(tmp_path: Path) -> None:
    src = "model = load('qwen2.5-0.5b')\nloss.backward()\noptimizer.step()\n"
    est = _estimate(tmp_path, src)
    assert est.mode == "train"
    assert est.model_params_b == 0.5
    assert est.vram_gb == 8  # 0.5 * 16
    assert est.hours == DEFAULT_HOURS


def test_huge_model_is_capped_and_flagged(tmp_path: Path) -> None:
    est = _estimate(tmp_path, "m = 'llama-70b'\nloss.backward()\n")
    assert est.vram_gb == 80
    assert any("capped" in r for r in est.reasons)


def test_estimate_in_manifest(tmp_path: Path, paths: Paths) -> None:
    project = make_project(tmp_path / "p", {"train.py": "m='gpt2-1.5b'\n"})
    bundle = build_bundle(
        project, JobSpec(project_dir=str(project), script="train.py"), paths=paths
    )
    assert bundle.manifest["estimate"]["model_params_b"] == 1.5
    assert bundle.manifest["estimate"]["vram_source"] == "heuristic"
