"""Phase-2 review fixes for file selection, deps and the bundle cache: credentials never
ship (broad deny list, symlink targets, untracked files named), git failures fail closed,
walk-mode dir symlinks are reported, uv sources are honoured, a truncated cache heals."""

from __future__ import annotations

import os
import tarfile
from pathlib import Path

import pytest

from gpu_router.models import DepsSpec, JobSpec
from gpu_router.packaging import BundleError, build_bundle
from gpu_router.packaging.bundle import cached_archive
from gpu_router.packaging.deps import DepsError, DepsInfo, detect_deps
from gpu_router.packaging.files import looks_secret, select_files
from gpu_router.paths import Paths
from tests.unit.packaging.helpers import git, make_project, write_files


def spec_for(project: Path) -> JobSpec:
    return JobSpec(project_dir=str(project), script="train.py")


# --------------------------------------------------------------------------- credentials

LEAKED = {
    ".envrc": "export HF_TOKEN=x\n",
    "prod.env": "KEY=x\n",
    ".aws/credentials": "[default]\n",
    ".huggingface/token": "hf_x\n",
    ".streamlit/secrets.toml": "k = 'x'\n",
    "client_secret_1234.apps.googleusercontent.com.json": "{}",
    "token.json": "{}",
    "application_default_credentials.json": "{}",
    ".npmrc": "//registry.npmjs.org/:_authToken=x\n",
    "keystore.p12": b"\0",
    "conf/.config/gcloud/creds.db": b"\0",
}


def test_credential_files_never_ship_tracked_or_not(tmp_path: Path) -> None:
    """Review finding: all of these shipped (only .env was caught)."""
    project = make_project(tmp_path / "proj", {"train.py": "", "x.env.example": ""})
    write_files(project, LEAKED)  # untracked, not ignored
    sel = select_files(project)
    shipped = {f.rel for f in sel.files}
    assert shipped == {"train.py", "x.env.example"}
    assert set(sel.excluded_secrets) == set(LEAKED)
    git(project, "add", "-A")  # tracked: still excluded
    assert {f.rel for f in select_files(project).files} == {"train.py", "x.env.example"}


@pytest.mark.parametrize(
    ("rel", "secret"),
    [
        ("train.py", False),
        ("config/model.yaml", False),
        ("tokenizer.json", False),
        ("envs/base.yaml", False),
        (".env.example", False),
        ("prod.env", True),
        ("PROD.ENV", True),
        ("sub/.ssh/config", True),
        ("x/.kaggle/whatever.txt", True),
        ("/Users/me/.kaggle/kaggle.json", True),
        ("a/.cache/huggingface/token", True),
        ("service-account-prod.json", True),
    ],
)
def test_looks_secret(rel: str, secret: bool) -> None:
    assert looks_secret(rel) is secret


def test_symlink_target_is_checked_not_just_its_name(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "kaggle.json").write_text('{"key": "abc"}')
    project = make_project(tmp_path / "proj", {"train.py": "", "real/kaggle.json": "{}"})
    (project / "config.json").symlink_to(outside / "kaggle.json")  # innocent name, outside
    (project / "settings.json").symlink_to(project / "real" / "kaggle.json")  # inside
    git(project, "add", "-A")
    sel = select_files(project)
    shipped = {f.rel for f in sel.files}
    assert "config.json" not in shipped
    assert "settings.json" not in shipped
    assert {"config.json", "settings.json"} <= set(sel.excluded_secrets)


def test_untracked_files_that_ship_are_named(tmp_path: Path, paths: Paths) -> None:
    project = make_project(tmp_path / "proj", {"train.py": ""})
    write_files(project, {"new_model.py": "", "notes/idea.md": ""})
    bundle = build_bundle(project, spec_for(project), paths=paths)
    assert bundle.manifest["files"]["untracked"] == 2
    msg = next(w for w in bundle.warnings if "untracked" in w)
    assert "new_model.py" in msg
    assert "notes/idea.md" in msg


# --------------------------------------------------------------------------- git failures


def _fake_git(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stderr: str, code: int) -> None:
    bindir = tmp_path / "fakebin"
    bindir.mkdir()
    git_exe = bindir / "git"
    git_exe.write_text(f"#!/bin/sh\necho {stderr!r} >&2\nexit {code}\n")
    git_exe.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")


def test_git_failure_in_a_repo_fails_closed(
    tmp_path: Path, paths: Paths, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review finding: any git failure fell back to the walk, which ignores .gitignore and
    shipped gitignored secrets, with a wrong 'not a git repository' warning."""
    project = make_project(
        tmp_path / "proj",
        {".gitignore": "local_config.py\n", "train.py": "", "local_config.py": "PW = 1\n"},
    )
    _fake_git(
        tmp_path,
        monkeypatch,
        "xcrun: error: invalid active developer path (/Library/Developer/CommandLineTools)",
        1,
    )
    with pytest.raises(BundleError) as ei:
        build_bundle(project, spec_for(project), paths=paths)
    assert "git ls-files" in ei.value.message
    assert "xcode-select --install" in (ei.value.hint or "")


def test_dubious_ownership_hint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from gpu_router.packaging.files import GitError

    project = make_project(tmp_path / "proj", {"train.py": ""})
    _fake_git(tmp_path, monkeypatch, "fatal: detected dubious ownership in repository", 128)
    with pytest.raises(GitError) as ei:
        select_files(project)
    assert "safe.directory" in ei.value.hint


def test_plain_dir_walks_without_calling_git(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = make_project(tmp_path / "plain", {"train.py": ""}, use_git=False)
    _fake_git(tmp_path, monkeypatch, "xcrun: error", 1)  # would fail if it were called
    sel = select_files(project)
    assert sel.source == "walk"
    assert [f.rel for f in sel.files] == ["train.py"]


def test_walk_reports_directory_symlinks(tmp_path: Path) -> None:
    shared = tmp_path / "shared" / "common"
    shared.mkdir(parents=True)
    (shared / "util.py").write_text("")
    project = make_project(tmp_path / "plain", {"train.py": ""}, use_git=False)
    (project / "common").symlink_to(shared, target_is_directory=True)
    sel = select_files(project)
    assert [f.rel for f in sel.files] == ["train.py"]
    assert any("common" in w and "directory symlinks" in w for w in sel.warnings)


# --------------------------------------------------------------------------- uv sources


def _pyproject(tmp_path: Path, body: str, extra: dict[str, str] | None = None) -> Path:
    files = {"train.py": "", "pyproject.toml": body, **(extra or {})}
    return make_project(tmp_path / "proj", files)


def _deps(project: Path) -> DepsInfo:
    shipped = {f.rel for f in select_files(project).files}
    return detect_deps(project, DepsSpec(), shipped)


def test_uv_git_and_url_sources_become_direct_references(tmp_path: Path) -> None:
    project = _pyproject(
        tmp_path,
        """
[project]
name = "x"
dependencies = ["mylib[fast]>=1; python_version >= '3.8'", "other", "wheelpkg", "numpy"]

[tool.uv.sources]
mylib = { git = "https://github.com/me/mylib", tag = "v1.2" }
Other = { git = "https://github.com/me/other", subdirectory = "pkg" }
wheelpkg = { url = "https://example.com/wheelpkg-1.0-py3-none-any.whl" }
""",
    )
    info = _deps(project)
    assert info.packages == [
        "mylib[fast] @ git+https://github.com/me/mylib@v1.2 ; python_version >= '3.8'",
        "other @ git+https://github.com/me/other#subdirectory=pkg",
        "wheelpkg @ https://example.com/wheelpkg-1.0-py3-none-any.whl",
        "numpy",
    ]


def test_uv_path_source_inside_project_ships_and_installs_locally(tmp_path: Path) -> None:
    project = _pyproject(
        tmp_path,
        """
[project]
name = "x"
dependencies = ["mylib"]

[tool.uv.sources]
mylib = { path = "libs/mylib", editable = true }
""",
        {"libs/mylib/pyproject.toml": "[project]\nname = 'mylib'\n"},
    )
    assert _deps(project).packages == ["./libs/mylib"]


@pytest.mark.parametrize(
    ("source", "phrase"),
    [
        ('{ path = "../elsewhere/mylib" }', "outside the project"),
        ("{ workspace = true }", "workspace member"),
        ('[{ git = "https://a" }, { git = "https://b" }]', "several sources"),
    ],
)
def test_uv_sources_pip_cannot_reproduce_are_refused(
    tmp_path: Path, source: str, phrase: str
) -> None:
    project = _pyproject(
        tmp_path,
        f"""
[project]
name = "x"
dependencies = ["mylib"]

[tool.uv.sources]
mylib = {source}
""",
    )
    with pytest.raises(DepsError, match=phrase) as ei:
        _deps(project)
    assert "uv export" in str(ei.value)


def test_uv_index_source_warns(tmp_path: Path) -> None:
    project = _pyproject(
        tmp_path,
        """
[project]
name = "x"
dependencies = ["torch"]

[[tool.uv.index]]
name = "pytorch"
url = "https://download.pytorch.org/whl/cu121"

[tool.uv.sources]
torch = { index = "pytorch" }
""",
    )
    info = _deps(project)
    assert info.packages == ["torch"]
    assert any("download.pytorch.org" in w for w in info.warnings)


def test_refused_uv_source_is_a_bundle_error(tmp_path: Path, paths: Paths) -> None:
    project = _pyproject(
        tmp_path,
        '[project]\nname = "x"\ndependencies = ["m"]\n'
        "[tool.uv.sources]\nm = { workspace = true }\n",
    )
    with pytest.raises(BundleError, match="workspace member"):
        build_bundle(project, spec_for(project), paths=paths)


# --------------------------------------------------------------------------- cache


def test_truncated_cache_entry_is_replaced_on_the_next_build(tmp_path: Path, paths: Paths) -> None:
    """Review finding: an existing cache entry was trusted on exists() alone, so a file
    truncated by a crash was reused by every later identical build."""
    project = make_project(tmp_path / "proj", {"train.py": "print(1)\n"})
    first = build_bundle(project, spec_for(project), paths=paths)
    entry = cached_archive(paths, first.sha256)
    good = entry.read_bytes()
    entry.write_bytes(good[: len(good) // 2])
    again = build_bundle(project, spec_for(project), paths=paths)
    assert again.cached is True
    assert entry.read_bytes() == good
    with tarfile.open(entry) as tar:
        assert "manifest.json" in tar.getnames()
