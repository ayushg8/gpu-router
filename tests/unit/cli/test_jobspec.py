"""gpu.yaml parsing/validation and the gpu.yaml + flags merge (jobspec.py)."""

from __future__ import annotations

from pathlib import Path

import pytest

from gpu_router.errors import InvalidSpec
from gpu_router.jobspec import (
    GPU_YAML_VERSION,
    Flags,
    GpuYamlError,
    build_spec,
    find_project_root,
    load_gpu_yaml,
    parse_env_pairs,
)


@pytest.fixture
def proj(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    p = tmp_path / "proj"
    p.mkdir()
    (p / "train.py").write_text("print(1)\n")
    monkeypatch.chdir(p)
    return p


def write(p: Path, text: str) -> None:
    (p / "gpu.yaml").write_text(text)


def err(p: Path, text: str) -> GpuYamlError:
    write(p, text)
    with pytest.raises(GpuYamlError) as ei:
        build_spec(Flags(), cwd=p, project_root=p)
    return ei.value


# --------------------------------------------------------------------------- project root


def test_project_root_is_nearest_git_ancestor(tmp_path: Path) -> None:
    (tmp_path / "repo" / ".git").mkdir(parents=True)
    deep = tmp_path / "repo" / "src" / "pkg"
    deep.mkdir(parents=True)
    assert find_project_root(deep) == (tmp_path / "repo").resolve()


def test_project_root_git_file_worktree(tmp_path: Path) -> None:
    (tmp_path / "wt").mkdir()
    (tmp_path / "wt" / ".git").write_text("gitdir: /elsewhere\n")
    assert find_project_root(tmp_path / "wt") == (tmp_path / "wt").resolve()


def test_project_root_falls_back_to_cwd(tmp_path: Path) -> None:
    d = tmp_path / "plain"
    d.mkdir()
    assert find_project_root(d) == d.resolve()


# --------------------------------------------------------------------------- loading


def test_missing_gpu_yaml_is_empty(proj: Path) -> None:
    doc = load_gpu_yaml(proj)
    assert doc.path is None
    assert doc.values == {}


def test_empty_gpu_yaml_is_empty(proj: Path) -> None:
    write(proj, "# nothing yet\n")
    assert load_gpu_yaml(proj).values == {}


def test_full_schema_round_trip(proj: Path) -> None:
    (proj / "reqs.txt").write_text("torch\n")
    write(
        proj,
        """
version: 1
name: yolo
script: train.py
args: "--epochs 3 --lr 0.1"
vram: 16
hours: 2.5
provider: Kaggle
gpu: T4
env: {WANDB_MODE: offline, SEED: 7, FLAG: true}
secrets: [WANDB_API_KEY]
deps: reqs.txt
data:
  - {mount: coco, path: data/coco}
  - {mount: wiki, uri: "hf://datasets/me/wiki"}
checkpoint_interval_min: 0
interactive: false
requires_approval: true
max_attempts: 3
labels: {team: vision}
provider_options:
  fake: {duration: 2}
""",
    )
    spec, doc, root = build_spec(Flags(provider="kaggle"), cwd=proj, project_root=proj)
    assert root == proj.resolve()
    assert doc.lines["vram_gb"] == 6
    assert spec.name == "yolo"
    assert spec.script == "train.py"
    assert spec.args == ["--epochs", "3", "--lr", "0.1"]
    assert spec.vram_gb == 16
    assert spec.hours == 2.5
    assert spec.provider == "kaggle"
    assert spec.env == {"WANDB_MODE": "offline", "SEED": "7", "FLAG": "true"}
    assert spec.secrets == ["WANDB_API_KEY"]
    assert spec.deps.kind == "requirements"
    assert spec.deps.file == "reqs.txt"
    assert [d.mount for d in spec.data] == ["coco", "wiki"]
    assert spec.checkpoint_interval_min == 0
    assert spec.requires_approval is True
    assert spec.max_attempts == 3
    assert spec.labels == {"team": "vision"}
    assert spec.provider_options == {"fake": {"duration": 2}}
    assert spec.project_dir == str(proj.resolve())


def test_unknown_key_suggests_and_names_line(proj: Path) -> None:
    e = err(proj, "version: 1\nscript: train.py\nvarm: 16\n")
    assert "gpu.yaml:3" in e.message
    assert "`varm`" in e.message
    assert e.hint == "did you mean `vram`?"
    assert e.detail["line"] == 3
    assert e.detail["key"] == "varm"


def test_unknown_key_without_close_match_lists_keys(proj: Path) -> None:
    e = err(proj, "zzzzzz: 1\n")
    assert e.hint is not None
    assert "known keys:" in e.hint
    assert "script" in e.hint


def test_wrong_type_says_what_is_expected(proj: Path) -> None:
    e = err(proj, "script: train.py\nhours: two\n")
    assert "gpu.yaml:2: `hours` is 'two'" in e.message
    assert e.hint == "`hours` must be a number of hours greater than 0, e.g. 2.5"


@pytest.mark.parametrize(
    ("text", "key"),
    [
        ("vram: true\n", "vram"),
        ("vram: -1\n", "vram"),
        ("interactive: yes please\n", "interactive"),
        ("max_attempts: 2.5\n", "max_attempts"),
        ("command: []\n", "command"),
        ("args: {a: 1}\n", "args"),
        ("env: [A]\n", "env"),
        ("env: {A: {nested: 1}}\n", "env"),
        ("provider_options: {fake: 1}\n", "provider_options"),
        ("deps: setup.py\n", "deps"),
        ("data: coco\n", "data"),
        ("name: ''\n", "name"),
    ],
)
def test_shape_errors(proj: Path, text: str, key: str) -> None:
    e = err(proj, "script: train.py\n" + text if key != "command" else text)
    assert e.detail["key"] == key
    assert f"`{key}`" in e.message


def test_script_and_command_together(proj: Path) -> None:
    e = err(proj, "script: train.py\ncommand: [bash, x.sh]\n")
    assert "either `script` or `command`" in e.message


def test_version_too_new(proj: Path) -> None:
    e = err(proj, f"version: {GPU_YAML_VERSION + 1}\nscript: train.py\n")
    assert "upgrade gpu-router" in (e.hint or "")


def test_bad_version(proj: Path) -> None:
    e = err(proj, "version: one\n")
    assert e.detail["key"] == "version"


def test_yaml_syntax_error_has_line(proj: Path) -> None:
    e = err(proj, "script: train.py\nenv: {A: 1\nvram: 2\n")
    assert e.detail["line"] is not None
    assert "gpu.yaml:" in e.message


def test_not_a_mapping(proj: Path) -> None:
    e = err(proj, "- train.py\n")
    assert "expected a mapping" in e.message


def test_secret_looking_env_rejected_without_generic_hint(proj: Path) -> None:
    e = err(proj, "script: train.py\nenv: {HF_TOKEN: abc}\n")
    assert "looks like a secret" in e.message
    assert e.hint is None
    assert "abc" not in e.message


def test_data_entry_error_names_index(proj: Path) -> None:
    e = err(proj, "script: train.py\ndata:\n  - {mount: ok, path: a}\n  - {mount: BAD!, path: b}\n")
    assert "entry 2" in e.message


def test_data_entry_needs_one_source(proj: Path) -> None:
    e = err(proj, "script: train.py\ndata:\n  - {mount: x}\n")
    assert "exactly one of path or uri" in e.message


def test_deps_forms(proj: Path) -> None:
    for text, kind in (
        ("auto", "auto"),
        ("none", "none"),
        ("pyproject.toml", "pyproject"),
        ("{kind: requirements, file: r.txt}", "requirements"),
    ):
        write(proj, f"script: train.py\ndeps: {text}\n")
        spec, _, _ = build_spec(Flags(), cwd=proj, project_root=proj)
        assert spec.deps.kind == kind


def test_deps_mapping_invalid_kind(proj: Path) -> None:
    e = err(proj, "script: train.py\ndeps: {kind: conda}\n")
    assert e.detail["key"] == "deps"


def test_gpu_yaml_script_must_exist(proj: Path) -> None:
    e = err(proj, "script: missing.py\n")
    assert "missing.py" in e.message
    assert "does not exist" in e.message


# --------------------------------------------------------------------------- merge


def test_flags_override_file(proj: Path) -> None:
    write(proj, "script: train.py\nvram: 8\nhours: 1\nprovider: fake\nargs: [--a]\n")
    spec, _, _ = build_spec(
        Flags(vram_gb=24, provider="fake-b", name="n"), cwd=proj, project_root=proj
    )
    assert spec.vram_gb == 24
    assert spec.provider == "fake-b"
    assert spec.name == "n"
    assert spec.hours == 1
    assert spec.args == ["--a"]


def test_env_flags_merge_per_key(proj: Path) -> None:
    write(proj, "script: train.py\nenv: {A: '1', B: '2'}\n")
    spec, _, _ = build_spec(Flags(env={"B": "x", "C": "y"}), cwd=proj, project_root=proj)
    assert spec.env == {"A": "1", "B": "x", "C": "y"}


def test_cli_script_replaces_file_script_and_args(proj: Path) -> None:
    (proj / "eval.py").write_text("")
    write(proj, "script: train.py\nargs: [--epochs, '9']\n")
    spec, _, _ = build_spec(Flags(script="eval.py", args=[]), cwd=proj, project_root=proj)
    assert spec.script == "eval.py"
    assert spec.args == []


def test_cli_args_alone_replace_file_args(proj: Path) -> None:
    write(proj, "script: train.py\nargs: [--epochs, '9']\n")
    spec, _, _ = build_spec(Flags(args=["--epochs", "1"]), cwd=proj, project_root=proj)
    assert spec.script == "train.py"
    assert spec.args == ["--epochs", "1"]


def test_cli_command_replaces_file_script(proj: Path) -> None:
    write(proj, "script: train.py\nargs: [--x]\n")
    spec, _, _ = build_spec(
        Flags(script="bash", args=["run.sh", "-v"]), cwd=proj, project_root=proj
    )
    assert spec.script is None
    assert spec.command == ["bash", "run.sh", "-v"]
    assert spec.args == []


def test_script_from_subdirectory_is_root_relative(proj: Path) -> None:
    sub = proj / "src"
    sub.mkdir()
    (sub / "t.py").write_text("")
    spec, _, _ = build_spec(Flags(script="t.py"), cwd=sub, project_root=proj)
    assert spec.script == "src/t.py"
    assert spec.project_dir == str(proj.resolve())


def test_script_outside_project(proj: Path, tmp_path: Path) -> None:
    (tmp_path / "outside.py").write_text("")
    with pytest.raises(InvalidSpec, match="outside the project"):
        build_spec(Flags(script=str(tmp_path / "outside.py")), cwd=proj, project_root=proj)


def test_missing_cli_script(proj: Path) -> None:
    with pytest.raises(InvalidSpec, match=r"nope\.py not found"):
        build_spec(Flags(script="nope.py"), cwd=proj, project_root=proj)


def test_nothing_to_run(proj: Path) -> None:
    with pytest.raises(InvalidSpec, match="nothing to run") as ei:
        build_spec(Flags(), cwd=proj, project_root=proj)
    assert "gpu.yaml" in (ei.value.hint or "")


def test_bad_flag_value_names_the_flag(proj: Path) -> None:
    with pytest.raises(InvalidSpec) as ei:
        build_spec(Flags(script="train.py", hours=-2), cwd=proj, project_root=proj)
    assert ei.value.message.startswith("--hours:")
    assert not isinstance(ei.value, GpuYamlError)


def test_bad_provider_flag(proj: Path) -> None:
    with pytest.raises(InvalidSpec, match="--provider"):
        build_spec(Flags(script="train.py", provider="No Such!"), cwd=proj, project_root=proj)


def test_parse_env_pairs() -> None:
    assert parse_env_pairs(["A=1", "B=x=y", "C="]) == {"A": "1", "B": "x=y", "C": ""}
    with pytest.raises(InvalidSpec, match="NAME=VALUE"):
        parse_env_pairs(["oops"])
    with pytest.raises(InvalidSpec):
        parse_env_pairs(["=1"])


def test_default_root_discovery(proj: Path) -> None:
    (proj / ".git").mkdir()
    sub = proj / "a"
    sub.mkdir()
    write(proj, "script: train.py\n")
    spec, doc, root = build_spec(Flags(), cwd=sub)
    assert root == proj.resolve()
    assert doc.path == proj / "gpu.yaml"
    assert spec.script == "train.py"


# --------------------------------------------------------------------------- review fixes


def test_gpu_yaml_script_checked_even_with_script_args(proj: Path) -> None:
    """Review finding: `gpu run --epochs 3` skipped the existence check of gpu.yaml's
    script, so a typo'd `script:` was submitted and failed remotely."""
    write(proj, "script: trian.py\n")
    with pytest.raises(GpuYamlError, match=r"trian\.py, which does not exist") as ei:
        build_spec(Flags(args=["--epochs", "3"]), cwd=proj, project_root=proj)
    assert ei.value.detail["key"] == "script"


def test_file_name_dropped_when_cli_replaces_entrypoint(proj: Path) -> None:
    (proj / "eval.py").write_text("print(2)\n")
    write(proj, "name: train-yolo\nscript: train.py\n")
    spec, _, _ = build_spec(Flags(script="eval.py"), cwd=proj, project_root=proj)
    assert spec.name is None
    assert spec.display_name() == "eval"
    # same entrypoint typed out: the file's name still describes it
    spec, _, _ = build_spec(Flags(script="train.py", args=["-x"]), cwd=proj, project_root=proj)
    assert spec.name == "train-yolo"
    # --name always wins
    spec, _, _ = build_spec(Flags(script="eval.py", name="ev"), cwd=proj, project_root=proj)
    assert spec.name == "ev"
    # a command replacing a script entrypoint drops it too
    spec, _, _ = build_spec(Flags(script="bash", args=["x.sh"]), cwd=proj, project_root=proj)
    assert spec.name is None


def test_gpu_yaml_provider_is_lowercased_like_the_flag(proj: Path) -> None:
    write(proj, "script: train.py\nprovider: Kaggle\n")
    spec, _, _ = build_spec(Flags(), cwd=proj, project_root=proj)
    assert spec.provider == "kaggle"


def test_env_error_blames_the_file_when_the_bad_key_is_from_the_file(proj: Path) -> None:
    write(proj, "script: train.py\nenv: {API_TOKEN: x}\n")
    with pytest.raises(GpuYamlError) as ei:
        build_spec(Flags(env={"FOO": "1"}), cwd=proj, project_root=proj)
    assert ei.value.detail["key"] == "env"
    assert "gpu.yaml" in ei.value.message
    assert "--env" not in ei.value.message
    assert "gpu secrets set API_TOKEN" in ei.value.message
    # and the flag when the flag holds it
    write(proj, "script: train.py\nenv: {FOO: x}\n")
    with pytest.raises(InvalidSpec) as ei2:
        build_spec(Flags(env={"MY_SECRET": "1"}), cwd=proj, project_root=proj)
    assert ei2.value.detail["flag"] == "--env"
