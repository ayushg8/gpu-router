"""build_bundle: file selection, determinism, cache, size guard, manifest, materialize."""

from __future__ import annotations

import gzip
import io
import json
import os
import tarfile
from pathlib import Path

import pytest

from gpu_router.errors import InvalidSpec
from gpu_router.models import DepsSpec, JobSpec
from gpu_router.packaging import BundleError, BundleTooLarge, build_bundle, materialize
from gpu_router.packaging.bundle import FIXED_MTIME, bundles_dir, cached_archive
from gpu_router.packaging.files import select_files
from gpu_router.paths import Paths
from tests.unit.packaging.helpers import git, make_project, write_files


def spec_for(project: Path, **fields: object) -> JobSpec:
    body: dict[str, object] = {"project_dir": str(project), "script": "train.py"}
    body.update(fields)
    return JobSpec.model_validate(body)


def members(archive: Path) -> dict[str, tarfile.TarInfo]:
    with tarfile.open(archive, "r:gz") as tar:
        return {m.name: m for m in tar.getmembers()}


def read_member(archive: Path, name: str) -> bytes:
    with tarfile.open(archive, "r:gz") as tar:
        fh = tar.extractfile(name)
        assert fh is not None
        return fh.read()


def code_files(archive: Path) -> set[str]:
    return {n.removeprefix("code/") for n in members(archive) if n.startswith("code/")}


# --------------------------------------------------------------------------- selection


def test_git_project_respects_gitignore_and_includes_untracked(
    tmp_path: Path, paths: Paths
) -> None:
    project = make_project(
        tmp_path / "proj",
        {
            ".gitignore": "data/\n*.ckpt\nsecret_notes.txt\n",
            "train.py": "print('hi')\n",
            "model/net.py": "x = 1\n",
            "data/train.bin": b"\0" * 100,
            "weights.ckpt": b"\1" * 100,
            "secret_notes.txt": "ignored\n",
        },
    )
    # untracked but not ignored: must ship; untracked and ignored: must not
    write_files(project, {"new_module.py": "y = 2\n", "later.ckpt": b"z"})
    bundle = build_bundle(project, spec_for(project), paths=paths)
    assert code_files(bundle.archive) == {".gitignore", "train.py", "model/net.py", "new_module.py"}
    assert bundle.manifest["files"]["source"] == "git"
    assert bundle.file_count == 4


def test_tracked_file_that_is_now_ignored_still_ships(tmp_path: Path, paths: Paths) -> None:
    project = make_project(tmp_path / "proj", {"train.py": "", "config.yaml": "a: 1\n"})
    write_files(project, {".gitignore": "config.yaml\n"})
    bundle = build_bundle(project, spec_for(project), paths=paths)
    # git semantics: tracked files are never ignored
    assert "config.yaml" in code_files(bundle.archive)


def test_deleted_tracked_file_is_skipped(tmp_path: Path, paths: Paths) -> None:
    project = make_project(tmp_path / "proj", {"train.py": "", "gone.py": ""})
    (project / "gone.py").unlink()
    bundle = build_bundle(project, spec_for(project), paths=paths)
    assert code_files(bundle.archive) == {"train.py"}


def test_subdirectory_of_a_repo_bundles_only_that_subtree(tmp_path: Path, paths: Paths) -> None:
    repo = make_project(tmp_path / "repo", {"top.py": "", "sub/train.py": "", "sub/lib/u.py": ""})
    bundle = build_bundle(repo / "sub", spec_for(repo / "sub"), paths=paths)
    assert code_files(bundle.archive) == {"train.py", "lib/u.py"}


def test_secret_looking_files_never_ship_even_when_tracked(tmp_path: Path, paths: Paths) -> None:
    project = make_project(
        tmp_path / "proj",
        {
            "train.py": "",
            ".env": "HF_TOKEN=abc\n",
            ".env.example": "HF_TOKEN=\n",
            "kaggle.json": "{}",
            "certs/server.pem": "x",
        },
    )
    bundle = build_bundle(project, spec_for(project), paths=paths)
    shipped = code_files(bundle.archive)
    assert shipped == {"train.py", ".env.example"}
    assert any("look like credentials" in w for w in bundle.warnings)


def test_non_git_project_falls_back_to_walk_with_warning(tmp_path: Path, paths: Paths) -> None:
    project = make_project(
        tmp_path / "plain",
        {
            "train.py": "",
            "utils/io.py": "",
            ".venv/lib/site.py": "",
            "__pycache__/train.cpython-312.pyc": b"\0",
            "runs/abcd/out.txt": "",
            "node_modules/x/index.js": "",
            ".DS_Store": b"\0",
        },
        use_git=False,
    )
    bundle = build_bundle(project, spec_for(project), paths=paths)
    assert code_files(bundle.archive) == {"train.py", "utils/io.py"}
    assert bundle.manifest["files"]["source"] == "walk"
    assert any("not a git repository" in w for w in bundle.warnings)


def test_symlink_to_file_is_dereferenced_and_dir_symlink_skipped(
    tmp_path: Path, paths: Paths
) -> None:
    outside = tmp_path / "shared.py"
    outside.write_text("SHARED = 1\n")
    project = make_project(
        tmp_path / "proj", {"train.py": "", "pkg/a.py": "", "common/util.py": "U = 1\n"},
        track=False,
    )  # fmt: skip
    (project / "util.py").symlink_to(project / "common" / "util.py")
    (project / "shared.py").symlink_to(outside)
    (project / "pkg_link").symlink_to(project / "pkg")
    git(project, "add", "-A")
    bundle = build_bundle(project, spec_for(project), paths=paths)
    assert read_member(bundle.archive, "code/util.py") == b"U = 1\n"  # inside: dereferenced
    shipped = code_files(bundle.archive)
    assert "pkg_link" not in shipped
    # review finding: links resolving outside the project are refused (bundles go to
    # third-party providers), with a warning naming them
    assert "shared.py" not in shipped
    assert any("point outside the project: shared.py" in w for w in bundle.warnings)


def test_select_files_missing_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        select_files(tmp_path / "nope")


# --------------------------------------------------------------------------- determinism + cache


def test_same_input_same_sha_and_cache_hit(tmp_path: Path, paths: Paths) -> None:
    project = make_project(tmp_path / "proj", {"train.py": "print(1)\n", "b.py": "", "a/c.py": ""})
    first = build_bundle(project, spec_for(project), paths=paths)
    second = build_bundle(project, spec_for(project), paths=paths)
    assert first.sha256 == second.sha256
    assert not first.cached
    assert second.cached
    assert first.archive == second.archive == cached_archive(paths, first.sha256)
    assert sorted(p.name for p in bundles_dir(paths).iterdir()) == [f"{first.sha256}.tar.gz"]


def test_sha_ignores_mtimes_and_owner(tmp_path: Path, paths: Paths) -> None:
    project = make_project(tmp_path / "proj", {"train.py": "print(1)\n"})
    first = build_bundle(project, spec_for(project), paths=paths)
    os.utime(project / "train.py", (1_000_000, 1_000_000))
    again = build_bundle(project, spec_for(project), paths=paths)
    assert again.sha256 == first.sha256
    for info in members(first.archive).values():
        assert info.mtime == FIXED_MTIME
        assert (info.uid, info.gid, info.uname, info.gname) == (0, 0, "", "")
    raw = first.archive.read_bytes()
    assert raw[4:8] == b"\0\0\0\0"  # gzip MTIME field is zero


def test_same_content_in_two_directories_gives_same_sha(tmp_path: Path, paths: Paths) -> None:
    files: dict[str, str | bytes] = {"train.py": "print(1)\n", "lib/x.py": "x = 1\n"}
    a = make_project(tmp_path / "a", files)
    b = make_project(tmp_path / "b", files)
    assert build_bundle(a, spec_for(a), paths=paths).sha256 == (
        build_bundle(b, spec_for(b), paths=paths).sha256
    )


def test_content_change_changes_sha(tmp_path: Path, paths: Paths) -> None:
    project = make_project(tmp_path / "proj", {"train.py": "print(1)\n"})
    first = build_bundle(project, spec_for(project), paths=paths)
    (project / "train.py").write_text("print(2)\n")
    assert build_bundle(project, spec_for(project), paths=paths).sha256 != first.sha256


def test_executable_bit_is_normalised_and_kept(tmp_path: Path, paths: Paths) -> None:
    project = make_project(tmp_path / "proj", {"train.py": "", "run.sh": "#!/bin/sh\n"})
    (project / "run.sh").chmod(0o700)
    (project / "train.py").chmod(0o600)
    m = members(build_bundle(project, spec_for(project), paths=paths).archive)
    assert m["code/run.sh"].mode == 0o755
    assert m["code/train.py"].mode == 0o644


# --------------------------------------------------------------------------- layout + manifest


def test_bundle_contains_runner_and_manifest(tmp_path: Path, paths: Paths) -> None:
    project = make_project(tmp_path / "proj", {"train.py": ""})
    bundle = build_bundle(project, spec_for(project, args=["--lr", "3e-4"]), paths=paths)
    names = set(members(bundle.archive))
    assert {"manifest.json", "gpu_runner/gpu.py", "gpu_runner/bootstrap.py"} <= names
    manifest = json.loads(read_member(bundle.archive, "manifest.json"))
    assert manifest == bundle.manifest
    assert manifest["manifest_version"] == 1
    assert manifest["entrypoint"] == {
        "script": "train.py",
        "command": None,
        "args": ["--lr", "3e-4"],
    }
    assert manifest["runner"]["helper"] == "gpu_runner/gpu.py"
    assert manifest["checkpoint_interval_min"] == 20
    assert len(manifest["files"]["tree_sha256"]) == 64
    assert "created" not in json.dumps(manifest)  # no timestamps: determinism


def test_runner_in_bundle_is_the_packaged_source(tmp_path: Path, paths: Paths) -> None:
    project = make_project(tmp_path / "proj", {"train.py": ""})
    bundle = build_bundle(project, spec_for(project), paths=paths)
    src = Path(__file__).resolve().parents[3] / "src" / "gpu_router" / "runner" / "gpu.py"
    assert read_member(bundle.archive, "gpu_runner/gpu.py") == src.read_bytes()


def test_command_entrypoint(tmp_path: Path, paths: Paths) -> None:
    project = make_project(tmp_path / "proj", {"run.sh": "echo hi\n"})
    spec = JobSpec(project_dir=str(project), command=["bash", "run.sh"])
    bundle = build_bundle(project, spec, paths=paths)
    assert bundle.manifest["entrypoint"]["command"] == ["bash", "run.sh"]
    assert bundle.warnings == []


def test_missing_entry_script_is_a_warning(tmp_path: Path, paths: Paths) -> None:
    project = make_project(tmp_path / "proj", {"other.py": ""})
    bundle = build_bundle(project, spec_for(project), paths=paths)
    assert any("train.py is not in the project" in w for w in bundle.warnings)
    assert bundle.manifest["warnings"] == bundle.warnings


def test_ignored_entry_script_is_called_out(tmp_path: Path, paths: Paths) -> None:
    project = make_project(tmp_path / "proj", {".gitignore": "train.py\n"})
    write_files(project, {"train.py": ""})
    bundle = build_bundle(project, spec_for(project), paths=paths)
    assert any("ignored by git" in w for w in bundle.warnings)


# --------------------------------------------------------------------------- guards


def test_size_guard_names_biggest_files(tmp_path: Path, paths: Paths) -> None:
    project = make_project(
        tmp_path / "proj", {"train.py": "", "big.bin": b"\0" * (2 * 1024 * 1024), "s.bin": b"1"}
    )
    with pytest.raises(BundleTooLarge) as err:
        build_bundle(project, spec_for(project), paths=paths, max_mb=1)
    assert isinstance(err.value, InvalidSpec)  # API: 400 invalid_spec
    assert "big.bin (2.0 MB)" in err.value.message
    assert "over the 1 MB limit" in err.value.message
    assert err.value.hint is not None
    assert ".gitignore" in err.value.hint
    assert err.value.detail["biggest"][0]["path"] == "big.bin"
    assert not bundles_dir(paths).exists() or not list(bundles_dir(paths).iterdir())


def test_missing_project_dir(tmp_path: Path, paths: Paths) -> None:
    missing = tmp_path / "nope"
    with pytest.raises(BundleError, match="does not exist"):
        build_bundle(missing, spec_for(missing), paths=paths)


def test_explicit_missing_requirements_file(tmp_path: Path, paths: Paths) -> None:
    project = make_project(tmp_path / "proj", {"train.py": ""})
    spec = spec_for(project, deps=DepsSpec(kind="requirements", file="reqs/gpu.txt"))
    with pytest.raises(BundleError, match="not found"):
        build_bundle(project, spec, paths=paths)


def test_failed_build_leaves_no_temp_files(
    tmp_path: Path, paths: Paths, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = make_project(tmp_path / "proj", {"train.py": ""})
    import gpu_router.packaging.bundle as mod

    def boom() -> dict[str, bytes]:
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(mod, "runner_sources", boom)
    with pytest.raises(RuntimeError):
        build_bundle(project, spec_for(project), paths=paths)
    assert list(bundles_dir(paths).iterdir()) == []


# --------------------------------------------------------------------------- materialize


def test_materialize_links_archive_and_extracts(tmp_path: Path, paths: Paths) -> None:
    project = make_project(tmp_path / "proj", {"train.py": "print(1)\n"})
    bundle = build_bundle(project, spec_for(project), paths=paths)
    bundle_dir, archive = materialize(paths, bundle.sha256, "abc123def456")
    assert archive == paths.job_bundle_archive("abc123def456")
    assert bundle_dir == paths.job_bundle_dir("abc123def456")
    assert archive.read_bytes() == bundle.archive.read_bytes()
    assert (bundle_dir / "code" / "train.py").read_text() == "print(1)\n"
    assert (bundle_dir / "gpu_runner" / "gpu.py").is_file()
    assert json.loads((bundle_dir / "manifest.json").read_text()) == bundle.manifest
    # idempotent
    assert materialize(paths, bundle.sha256, "abc123def456") == (bundle_dir, archive)


def test_materialize_unknown_sha(paths: Paths) -> None:
    with pytest.raises(BundleError, match="missing from the cache"):
        materialize(paths, "0" * 64, "abc123def456")


def test_gzip_stream_is_plain_tar(tmp_path: Path, paths: Paths) -> None:
    project = make_project(tmp_path / "proj", {"train.py": ""})
    bundle = build_bundle(project, spec_for(project), paths=paths)
    raw = gzip.decompress(bundle.archive.read_bytes())
    with tarfile.open(fileobj=io.BytesIO(raw)) as tar:
        assert "manifest.json" in tar.getnames()
