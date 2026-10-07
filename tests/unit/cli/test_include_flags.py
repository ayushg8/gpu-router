"""D60: gpu.yaml `include:`, `gpu run --include` (before the script, D23) and the dry-run
bundle preview's include / left-out facts."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from rich.console import Console

from gpu_router.cli import render
from gpu_router.cli.app import split_run_argv
from gpu_router.cli.bundling import preview
from gpu_router.errors import InvalidSpec
from gpu_router.jobspec import Flags, GpuYamlError, build_spec
from tests.unit.packaging.helpers import isolate_git, make_project, write_files


@pytest.fixture
def proj(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    isolate_git(monkeypatch, tmp_path)
    p = make_project(
        tmp_path / "proj", {"train.py": "print(1)\n", ".gitignore": "third_party/\ndata/\n"}
    )
    write_files(p, {"third_party/foo/model.py": "x = 1\n", "data/big.bin": b"0" * 2048})
    monkeypatch.chdir(p)
    return p


def test_gpu_yaml_include_string_or_list(proj: Path) -> None:
    (proj / "gpu.yaml").write_text("script: train.py\ninclude: third_party/\n")
    spec, _, _ = build_spec(Flags(), cwd=proj)
    assert spec.include == ["third_party/"]
    (proj / "gpu.yaml").write_text('script: train.py\ninclude: [third_party/, "x/*.png"]\n')
    spec, _, _ = build_spec(Flags(), cwd=proj)
    assert spec.include == ["third_party/", "x/*.png"]


def test_include_flag_adds_to_gpu_yaml(proj: Path) -> None:
    (proj / "gpu.yaml").write_text("script: train.py\ninclude: [third_party/]\n")
    spec, _, _ = build_spec(Flags(include=["data/*.bin", "third_party/"]), cwd=proj)
    assert spec.include == ["third_party/", "data/*.bin"]


def test_bad_include_names_the_file_line_or_the_flag(proj: Path) -> None:
    (proj / "gpu.yaml").write_text("script: train.py\n\ninclude: [../secrets]\n")
    with pytest.raises(GpuYamlError) as ei:
        build_spec(Flags(include=["ok/"]), cwd=proj)
    assert "gpu.yaml:3: `include`" in ei.value.message
    assert "'../secrets'" in ei.value.message
    (proj / "gpu.yaml").write_text("script: train.py\ninclude: [ok/]\n")
    with pytest.raises(InvalidSpec) as e2:
        build_spec(Flags(include=["/etc"]), cwd=proj)
    assert e2.value.message.startswith("--include: ")
    (proj / "gpu.yaml").write_text("script: train.py\ninclude: {a: 1}\n")
    with pytest.raises(GpuYamlError):
        build_spec(Flags(), cwd=proj)


def test_include_is_a_gpu_option_only_before_the_script() -> None:
    click, script_args, clashes = split_run_argv(
        ["--include", "third_party/", "train.py", "--include", "x"]
    )
    assert click == ["--include", "third_party/", "train.py"]
    assert script_args == ["--include", "x"]
    assert clashes == ["--include"]


def test_dry_run_preview_shows_included_and_left_out(proj: Path) -> None:
    spec, _, _ = build_spec(Flags(script="train.py", include=["third_party/"]), cwd=proj)
    bundle = preview(spec)
    assert bundle is not None
    assert bundle["included"] == {"files": 1, "bytes": 6}
    assert bundle["left_out"] == ["data/ (2.0 KB, ignored)"]
    assert "include:" in bundle["hint"]
    buf = io.StringIO()
    render.print_bundle(Console(file=buf, width=200, color_system=None), bundle)
    text = buf.getvalue()
    assert "left out: data/ (2.0 KB, ignored)" in text
    assert "data= (datasets)" in text
