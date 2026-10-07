"""D60: `include:` ships git-ignored paths on purpose (credential, symlink and size rules
still apply), and the bundle summary names what git ignores and so never ships."""

from __future__ import annotations

import os
import tarfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from gpu_router.models import JobSpec
from gpu_router.packaging import BundleError, build_bundle, bundle_summary
from gpu_router.packaging import files as files_mod
from gpu_router.packaging.bundle import LEFT_OUT_HINT
from gpu_router.packaging.files import (
    IncludeError,
    include_regex,
    left_out,
    normalize_include,
    select_files,
)
from tests.unit.packaging.helpers import git, make_project, write_files

IGNORE = "data/\nthird_party/\nout/\n*.log\n.env\n__pycache__/\n.venv/\n"


def spec_for(project: Path, include: list[str] | None = None) -> JobSpec:
    return JobSpec(project_dir=str(project), script="train.py", include=include or [])


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A git project the way the field test's projects looked: ignored data/, a cloned
    (nested git) repo under an ignored third_party/, ignored experiment outputs."""
    proj = make_project(tmp_path / "proj", {"train.py": "print(1)\n", ".gitignore": IGNORE})
    write_files(
        proj,
        {
            "data/crops/a.png": b"\x89PNG-a",
            "data/crops/b.png": b"\x89PNG-b",
            "data/raw/big.bin": b"0" * 4096,
            "experiments/a/out/r.json": "{}",
            "experiments/b/c/out/x.json": "{}",
            "experiments/a/run.log": "log\n",
            "experiments/a/eval.py": "print(2)\n",  # not ignored: git lists a/ by entry
            "top.log": "log\n",
            ".env": "HF_TOKEN=x\n",
            "__pycache__/m.cpython-312.pyc": b"\0",
            ".venv/bin/python": "",
        },
    )
    foo = proj / "third_party" / "foo"
    foo.mkdir(parents=True)
    git(foo, "init", "-q")
    write_files(foo, {"model.py": "x = 1\n", "weights/tiny.pt": b"w", "pkg/__pycache__/a.pyc": ""})
    git(foo, "add", "-A")
    return proj


def shipped(sel: files_mod.FileSelection) -> set[str]:
    return {f.rel for f in sel.files}


# --------------------------------------------------------------------------- include


def test_ignored_files_ship_when_included(project: Path) -> None:
    before = select_files(project)
    assert not any(r.startswith("data/") for r in shipped(before))
    spec = spec_for(project, ["data/crops/*.png"])
    bundle = build_bundle(project, spec)
    sel = select_files(project, spec.include)
    assert shipped(sel) - shipped(before) == {"data/crops/a.png", "data/crops/b.png"}
    assert sel.included == ["data/crops/a.png", "data/crops/b.png"]
    assert bundle.manifest["files"]["included"] == {"count": 2, "bytes": 12}
    assert any(w.startswith("include: shipping 2 file(s)") for w in bundle.warnings)
    with tarfile.open(bundle.archive, "r:gz") as tar:
        names = tar.getnames()
    assert "code/data/crops/a.png" in names
    assert "code/data/raw/big.bin" not in names


def test_a_nested_git_repo_ships_without_its_git_dir(project: Path) -> None:
    sel = select_files(project, ["third_party/"])
    assert {"third_party/foo/model.py", "third_party/foo/weights/tiny.pt"} <= shipped(sel)
    assert not any("/.git/" in r or r.endswith(".pyc") for r in shipped(sel))


def test_a_nested_repo_git_lists_only_as_a_whole(project: Path) -> None:
    """Not ignored, but git never descends into it: it used to vanish with a warning."""
    vendor = project / "vendor" / "bar"
    vendor.mkdir(parents=True)
    git(vendor, "init", "-q")
    write_files(vendor, {"lib.py": "y = 2\n"})
    sel = select_files(project)
    assert "vendor/bar/" in sel.skipped
    assert any("name a nested repo in `include:`" in w for w in sel.warnings)
    items, _ = left_out(project, sel)
    assert "vendor/bar/ (6 B, nested git repo)" in [i.text() for i in items]
    sel = select_files(project, ["vendor/bar"])
    assert "vendor/bar/lib.py" in shipped(sel)
    assert sel.skipped == []
    assert not any("not regular files" in w for w in sel.warnings)
    assert "vendor/bar/" not in [i.path for i in left_out(project, sel, ["vendor/bar"])[0]]


def test_credentials_under_an_include_still_never_ship(project: Path, tmp_path: Path) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("secret-ish")
    write_files(
        project,
        {
            "third_party/foo/.env": "K=v\n",
            "third_party/foo/kaggle.json": "{}",
            "third_party/foo/.aws/credentials": "[default]\n",
            "data/crops/token.json": "{}",
            "data/crops/real.key": "k",
        },
    )
    os.symlink(outside, project / "third_party" / "foo" / "escape.txt")
    os.symlink(project / "data" / "crops" / "real.key", project / "data" / "crops" / "x.png")
    sel = select_files(project, ["third_party", "data/crops"])
    names = shipped(sel)
    assert "third_party/foo/model.py" in names
    for rel in (
        "third_party/foo/.env",
        "third_party/foo/kaggle.json",
        "third_party/foo/.aws/credentials",
        "data/crops/token.json",
        "data/crops/real.key",
        "data/crops/x.png",  # an innocent name for a key file
        "third_party/foo/escape.txt",  # points outside the project
    ):
        assert rel not in names, rel
    assert "third_party/foo/.aws/" in sel.excluded_secrets
    assert {"third_party/foo/kaggle.json", "data/crops/x.png"} <= set(sel.excluded_secrets)
    assert any("point outside the project" in w for w in sel.warnings)


@pytest.mark.parametrize("bad", ["../x", "a/../../b", "/etc/passwd", "~/.ssh", ".", "**", ""])
def test_patterns_outside_the_project_are_refused(project: Path, bad: str) -> None:
    with pytest.raises(ValidationError) as ei:
        spec_for(project, [bad])
    assert "include" in str(ei.value)
    with pytest.raises(IncludeError):
        select_files(project, [bad])


def test_patterns_are_normalized_and_deduplicated(project: Path) -> None:
    spec = spec_for(project, ["./third_party//", "third_party/", " data/crops/*.png "])
    assert spec.include == ["third_party/", "data/crops/*.png"]
    assert normalize_include("a/./b/") == "a/b/"
    assert JobSpec(project_dir="/p", script="t.py").model_dump().get("include") is None
    assert '"include"' not in JobSpec(project_dir="/p", script="t.py").model_dump_json()


def test_zero_match_and_symlinked_patterns_warn(project: Path) -> None:
    os.symlink(project / "third_party", project / "linked")
    sel = select_files(project, ["nope/", "*.xyz", "linked/foo"])
    assert any(w.startswith("include matched no files: *.xyz, nope/") for w in sel.warnings)
    assert any("include `linked/foo` goes through the symlink linked" in w for w in sel.warnings)
    assert not sel.included


def test_globs_match_files_and_whole_directories(project: Path) -> None:
    sel = select_files(project, ["experiments/**/out"])
    assert sel.included == ["experiments/a/out/r.json", "experiments/b/c/out/x.json"]
    sel = select_files(project, ["experiments/*/out/"])  # one level, directories only
    assert sel.included == ["experiments/a/out/r.json"]
    sel = select_files(project, ["*.log"])  # top level only: * never crosses "/"
    assert sel.included == ["top.log"]
    sel = select_files(project, ["**/*.log"])
    assert sel.included == ["experiments/a/run.log", "top.log"]
    rx = include_regex("data/[!r]*/*.p?g")
    assert rx.fullmatch("data/crops/a.png")
    assert not rx.fullmatch("data/raw/a.png")
    assert not rx.fullmatch("data/c/d/a.png")


def test_a_glob_only_walks_directories_that_can_match(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_files(project, {f"data/many/{i}.bin": b"" for i in range(30)})
    monkeypatch.setattr(files_mod, "MAX_INCLUDE_FILES", 12)
    assert select_files(project, ["*.log"]).included == ["top.log"]  # data/ never entered
    with pytest.raises(IncludeError) as ei:
        select_files(project, ["**/*.log"])
    assert "data:" in ei.value.hint
    with pytest.raises(BundleError) as be:
        build_bundle(project, spec_for(project, ["data/"]))
    assert "more than 12 files" in be.value.message
    assert "data:" in (be.value.hint or "")


def test_included_bytes_count_toward_the_size_cap(project: Path) -> None:
    write_files(project, {"data/raw/huge.bin": b"0" * 300_000})
    with pytest.raises(BundleError) as ei:
        build_bundle(project, spec_for(project, ["data/raw"]), max_mb=0.1)
    assert "`include:` adds" in (ei.value.hint or "")


def test_include_bundles_are_deterministic(project: Path) -> None:
    a = build_bundle(project, spec_for(project, ["third_party/", "data/crops/*.png"]))
    b = build_bundle(project, spec_for(project, ["data/crops/*.png", "third_party/"]))
    assert a.sha256 == b.sha256
    with tarfile.open(a.archive, "r:gz") as tar:
        code = [n for n in tar.getnames() if n.startswith("code/")]
        mtimes = {m.mtime for m in tar.getmembers()}
    assert code == sorted(code)
    assert mtimes == {315532800}
    # a spec without include keeps its old shape: no `include` key, same manifest keys
    plain = build_bundle(project, spec_for(project))
    assert plain.manifest["files"]["included"] == {"count": 0, "bytes": 0}


# --------------------------------------------------------------------------- left out


def test_left_out_names_ignored_paths_not_junk_or_credentials(project: Path) -> None:
    items, more = left_out(project, select_files(project))
    texts = [i.text() for i in items]
    assert more == 0
    assert [i.path for i in items] == [
        "data/",
        "experiments/a/out/",
        "experiments/a/run.log",
        "experiments/b/",
        "third_party/",
        "top.log",
    ]
    assert "data/ (4.0 KB, ignored)" in texts
    assert any(t.startswith("third_party/ (") and t.endswith(", ignored)") for t in texts)
    joined = " ".join(texts)
    for hidden in (".env", "__pycache__", ".venv", ".git"):
        assert hidden not in joined


def test_left_out_drops_what_include_ships(project: Path) -> None:
    inc = ["third_party/", "data/crops/*.png", "experiments/**/out", "*.log"]
    sel = select_files(project, inc)
    items, _ = left_out(project, sel, inc)
    texts = {i.path: i.text() for i in items}
    assert "third_party/" not in texts
    assert "experiments/a/out/" not in texts
    assert "top.log" not in texts
    assert texts["data/"] == "data/ (4.0 KB, ignored; part ships via include)"


def test_left_out_sizes_stay_within_a_budget(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(files_mod, "LEFT_OUT_SIZE_BUDGET", 3)
    items, _ = left_out(project, select_files(project))
    data = next(i for i in items if i.path == "data/")
    assert data.bytes is None
    assert data.files_at_least
    assert data.text() == f"data/ (ignored, {data.files}+ files)"


def test_many_left_out_paths_are_grouped_by_top_dir(project: Path) -> None:
    write_files(project, {f"experiments/e{i}/out/r.json": "{}" for i in range(12)})
    items, more = left_out(project, select_files(project), limit=3)
    assert len(items) == 3
    assert more >= 1
    exp = next(i for i in items if i.path == "experiments/")
    assert exp.why.endswith("paths inside left out")


def test_left_out_in_a_plain_directory(tmp_path: Path) -> None:
    proj = make_project(
        tmp_path / "plain",
        {"train.py": "", "checkpoints/w.pt": b"w" * 10, ".venv/x": "", "outputs/o.txt": ""},
        use_git=False,
    )
    items, _ = left_out(proj, select_files(proj))
    assert [i.text() for i in items] == [
        "checkpoints/ (10 B, skipped by default)",
        "outputs/ (0 B, skipped by default)",
    ]
    sel = select_files(proj, ["checkpoints"])
    assert "checkpoints/w.pt" in shipped(sel)


def test_bundle_summary_shape(project: Path) -> None:
    out = bundle_summary(spec_for(project, ["third_party/"]))
    assert out["files"] == len(select_files(project, ["third_party/"]).files)
    assert out["included"]["files"] == 2
    assert out["hint"] == LEFT_OUT_HINT
    assert "data/ (4.0 KB, ignored)" in out["left_out"]
    assert not any(x.startswith("third_party/") for x in out["left_out"])
    assert all(len(w) <= 300 for w in out.get("warnings", []))
    plain = bundle_summary(spec_for(project))
    assert "included" not in plain
    tiny = bundle_summary(spec_for(project, ["data/"]), max_mb=0.001)
    assert tiny["warnings"][0].startswith("over the 0.001 MB bundle limit, so the submit")
    assert "`include:` adds" in tiny["warnings"][0]


def test_bundle_summary_reports_errors_instead_of_raising(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(files_mod, "MAX_INCLUDE_FILES", 1)
    out = bundle_summary(spec_for(project, ["data/"]))
    assert set(out) == {"error", "hint"}
    assert "more than 1 files" in out["error"]
    gone = bundle_summary(JobSpec(project_dir=str(project / "nope"), script="t.py"))
    assert "does not exist" in gone["error"]


def test_entry_warning_points_at_include(project: Path) -> None:
    write_files(project, {"data/run_eval.py": "print(1)\n"})
    spec = JobSpec(project_dir=str(project), script="data/run_eval.py")
    out = bundle_summary(spec)
    assert any("add it to gpu.yaml `include:`" in w for w in out["warnings"])
    spec = JobSpec(project_dir=str(project), script="data/run_eval.py", include=["data/*.py"])
    assert not any("ignored by git" in w for w in bundle_summary(spec).get("warnings", []))


def test_venvs_and_vcs_metadata_never_ship_from_an_include(project: Path) -> None:
    """Field test: a cloned repo under third_party/ had its own .venv (50k+ files)."""
    write_files(
        project,
        {
            "third_party/foo/.venv/lib/site.py": "",
            "third_party/foo/myenv/pyvenv.cfg": "home = /usr/bin\n",
            "third_party/foo/myenv/lib/x.py": "",
            "third_party/foo/node_modules/a/index.js": "",
        },
    )
    sel = select_files(project, ["third_party/", ".git/config", "third_party/foo/.git"])
    assert sel.included == ["third_party/foo/model.py", "third_party/foo/weights/tiny.pt"]
    assert any("include `.git/config` names .git/, which never ships" in w for w in sel.warnings)
    assert any("include `third_party/foo/.git` names .git/" in w for w in sel.warnings)
