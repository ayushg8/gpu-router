"""D60 through the MCP tools: `include` on gpu_submit / gpu_route, and the `bundle` summary
that tells an agent what shipped and which git-ignored paths did not."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from gpu_router.mcp.server import INSTRUCTIONS
from gpu_router.packaging.bundle import LEFT_OUT_HINT
from gpu_router.paths import Paths
from tests.mcp.conftest import call, call_error
from tests.shell.conftest import InProcDaemon
from tests.unit.packaging.helpers import git, isolate_git, write_files


@pytest.fixture
def repo(project: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The MCP suite's project as a git repo with an ignored clone and an ignored dataset."""
    isolate_git(monkeypatch, tmp_path)
    (project / ".gitignore").write_text("third_party/\ndata/\n")
    write_files(
        project,
        {
            "third_party/foo/model.py": "x = 1\n",
            "third_party/foo/.env": "K=v\n",
            "data/crops/a.png": b"\x89PNG" * 512,
        },
    )
    git(project / "third_party" / "foo", "init", "-q")
    git(project, "init", "-q")
    git(project, "add", "-A")
    return project


async def test_tools_take_include_and_explain_it(mcp: Any) -> None:
    listed = {t.name: t for t in await mcp.list_tools()}
    for name in ("gpu_submit", "gpu_route"):
        prop = listed[name].input_schema["properties"]["include"]
        assert "git ignores" in prop["description"]
        assert "bundle.left_out" in listed[name].description
    assert "include=[...]" in INSTRUCTIONS
    assert 'provider="local"' in INSTRUCTIONS


async def test_route_says_what_is_left_out(daemon: InProcDaemon, repo: Path, mcp: Any) -> None:
    out = await call(mcp, "gpu_route", project_dir=str(repo), script="train.py", hours=0.1)
    bundle = out["bundle"]
    assert bundle["files"] >= 3  # train.py, gpu.yaml, .gitignore
    assert "included" not in bundle
    assert bundle["left_out"] == ["data/ (2.0 KB, ignored)", "third_party/ (6 B, ignored)"]
    assert bundle["hint"] == LEFT_OUT_HINT

    out = await call(
        mcp,
        "gpu_route",
        project_dir=str(repo),
        script="train.py",
        hours=0.1,
        include=["third_party/"],
    )
    assert out["spec"]["include"] == ["third_party/"]
    bundle = out["bundle"]
    assert bundle["included"] == {"files": 1, "bytes": 6}  # .env never ships
    assert bundle["left_out"] == ["data/ (2.0 KB, ignored)"]
    assert any("look like credentials" in w for w in bundle["warnings"])


async def test_submit_ships_included_paths(daemon: InProcDaemon, repo: Path, mcp: Any) -> None:
    sub = await call(
        mcp,
        "gpu_submit",
        project_dir=str(repo),
        script="train.py",
        hours=0.1,
        include=["third_party/"],
    )
    assert sub["submitted"] is True
    job = sub["job"]
    assert job["spec"]["include"] == ["third_party/"]
    assert sub["bundle"]["included"]["files"] == 1
    assert sub["bundle"]["left_out"] == ["data/ (2.0 KB, ignored)"]
    code = Paths.from_env().job_bundle_dir(job["id"]) / "code"
    assert (code / "third_party" / "foo" / "model.py").is_file()
    assert not (code / "third_party" / "foo" / ".env").exists()
    assert not (code / "data").exists()

    again = await call(
        mcp,
        "gpu_submit",
        project_dir=str(repo),
        script="train.py",
        hours=0.1,
        include=["third_party/"],
    )
    assert again["submitted"] is False  # the duplicate carries no fresh bundle summary
    assert "bundle" not in again


async def test_bad_include_is_refused_before_submit(
    daemon: InProcDaemon, repo: Path, mcp: Any
) -> None:
    err = await call_error(
        mcp, "gpu_submit", project_dir=str(repo), script="train.py", include=["../other"]
    )
    assert err["code"] == "invalid_spec"
    assert "include" in err["message"]
    assert "'../other'" in err["message"]
