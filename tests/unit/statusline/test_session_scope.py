"""D56: each Claude Code status line shows only the jobs launched from its own session."""

from __future__ import annotations

import io
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_router import statefile
from gpu_router.origin import claude_origin, with_origin
from gpu_router.statusline import fast, samples
from gpu_router.statusline.samples import NOW, snapshot

S1, S2 = "11111111-aaaa-4bbb-8ccc-000000000001", "22222222-aaaa-4bbb-8ccc-000000000002"


def _origin(session: str | None, pid: str | None) -> dict[str, str]:
    out: dict[str, str] = {}
    if session:
        out["claude_session"] = session
    if pid:
        out["claude_pid"] = pid
    return out


def _mine(job: dict[str, Any], session: str | None, pid: str | None) -> dict[str, Any]:
    return dict(job, origin=_origin(session, pid))


def _two_sessions() -> dict[str, Any]:
    a = _mine(samples._job(name="alpha_train"), S1, "1001")
    b = _mine(samples._job(id="bbbbbbbbbbbb", short_id="bbbb", name="beta_eval"), S2, "2002")
    return snapshot([a, b])


def _render(snap: dict[str, Any], tmp_path: Path, payload: Any, env: dict[str, str]) -> str:
    (tmp_path / "state.json").write_text(
        json.dumps(dict(snap, daemon_pid=__import__("os").getpid()))
    )
    stdin = io.StringIO(json.dumps(payload))
    return fast.render(
        ["--line", "--stdin", "--plain", "--now", str(NOW)],
        environ={"GPU_ROUTER_HOME": str(tmp_path), **env},
        stdin=stdin,
    )


def test_two_sessions_each_see_only_their_own_job(tmp_path: Path) -> None:
    snap = _two_sessions()
    one = _render(snap, tmp_path, {"session_id": S1}, {"CLAUDE_PID": "1001"})
    two = _render(snap, tmp_path, {"session_id": S2}, {"CLAUDE_PID": "2002"})
    assert "alpha_train" in one
    assert "beta_eval" not in one
    assert "beta_eval" in two
    assert "alpha_train" not in two
    assert "+1" not in one  # the other session's job is not counted
    assert "+1" not in two


def test_pid_match_keeps_rows_after_the_session_id_changes(tmp_path: Path) -> None:
    snap = _two_sessions()
    after_clear = "33333333-aaaa-4bbb-8ccc-000000000003"
    out = _render(snap, tmp_path, {"session_id": after_clear}, {"CLAUDE_PID": "1001"})
    assert "alpha_train" in out
    assert "beta_eval" not in out


def test_other_session_with_no_pid_sees_nothing(tmp_path: Path) -> None:
    out = _render(_two_sessions(), tmp_path, {"session_id": "not-a-launcher"}, {})
    assert out == ""


def test_env_session_id_also_matches(tmp_path: Path) -> None:
    out = _render(
        _two_sessions(), tmp_path, {"session_id": "fresh"}, {"CLAUDE_CODE_SESSION_ID": S2}
    )
    assert "beta_eval" in out
    assert "alpha_train" not in out


def test_job_without_origin_is_hidden_when_session_info_is_present(tmp_path: Path) -> None:
    snap = snapshot([samples._job(name="plain_terminal")])
    assert _render(snap, tmp_path, {"session_id": S1}, {"CLAUDE_PID": "1001"}) == ""


def test_no_session_info_shows_every_job(tmp_path: Path) -> None:
    snap = _two_sessions()
    snap["active"].append(samples._job(id="cccccccccccc", short_id="cccc", name="no_origin"))
    out = _render(snap, tmp_path, {"workspace": {"current_dir": "/tmp"}}, {"CLAUDE_PID": "1001"})
    assert "alpha_train" in out
    assert "+2 running" in out


def test_approval_row_is_scoped_and_counts_only_this_session() -> None:
    mine = _mine(samples.APPROVAL, S1, "1001")
    theirs = _mine(
        dict(samples.APPROVAL, id="e5e5e5e5e5e5", short_id="e5e5", script="bench.py"), S2, "2002"
    )
    viewer = (frozenset({S1}), "1001")
    rows = fast.rows_text(
        snapshot([mine, theirs]),
        now=NOW,
        color=False,
        pid_alive=lambda _p: True,
        viewer=viewer,
    )
    assert len(rows) == 1
    assert rows[0].endswith("/gpu-approve")  # one approval here, so no id is needed
    assert "bench.py" not in rows[0]


def test_recent_rows_are_scoped_too() -> None:
    snap = snapshot(
        [],
        recent=[
            dict(samples._recent(state="failed", failure_kind="internal"), origin=_origin(S2, None))
        ],
    )
    viewer = (frozenset({S1}), "1001")
    assert (
        fast.rows_text(snap, now=NOW, color=False, pid_alive=lambda _p: True, viewer=viewer) == []
    )
    viewer2 = (frozenset({S2}), None)
    rows = fast.rows_text(snap, now=NOW, color=False, pid_alive=lambda _p: True, viewer=viewer2)
    assert rows
    assert "internal error" in rows[0]


def test_daemon_down_hint_only_for_this_sessions_jobs() -> None:
    snap = _two_sessions()
    other = (frozenset({"someone-else"}), None)
    assert fast.build_rows(snap, now=NOW, pid_alive=lambda _p: False, viewer=other) == []
    mine = (frozenset({S1}), None)
    rows = fast.build_rows(snap, now=NOW, pid_alive=lambda _p: False, viewer=mine)
    assert len(rows) == 1


# --------------------------------------------------------------------------- origin tagging


def test_claude_origin_from_env() -> None:
    env = {"CLAUDE_CODE_SESSION_ID": S1, "CLAUDE_PID": "1281"}
    assert claude_origin(env) == {"claude_session": S1, "claude_pid": "1281"}
    assert claude_origin({}) == {}
    assert claude_origin({}, fallback_pid=999) == {}  # not a Claude Code child: no origin
    assert claude_origin({"CLAUDE_CODE_SESSION_ID": S1}, fallback_pid=999) == {
        "claude_session": S1,
        "claude_pid": "999",
    }
    assert claude_origin({"CLAUDE_CODE_SESSION_ID": "bad id; rm", "CLAUDE_PID": "x"}) == {}


def test_with_origin_adds_labels_and_keeps_others() -> None:
    from gpu_router.models import JobSpec

    spec = JobSpec(project_dir="/tmp/p", script="t.py", labels={"via": "mcp"})
    tagged = with_origin(spec, {"CLAUDE_CODE_SESSION_ID": S1, "CLAUDE_PID": "42"})
    assert tagged.labels == {"via": "mcp", "claude_session": S1, "claude_pid": "42"}
    assert with_origin(spec, {}) is spec


def test_mcp_build_spec_records_the_origin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from gpu_router.mcp import tools

    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "train.py").write_text("print(1)\n")
    monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", S1)
    monkeypatch.setenv("CLAUDE_PID", "4242")
    spec = tools.build_spec(str(proj), "train.py")
    assert spec.labels["claude_session"] == S1
    assert spec.labels["claude_pid"] == "4242"
    assert spec.labels["via"] == "mcp"


def test_statefile_carries_the_origin() -> None:
    job = SimpleNamespace(spec=SimpleNamespace(labels={"claude_session": S1, "claude_pid": "7"}))
    assert statefile._origin(job) == {"claude_session": S1, "claude_pid": "7"}  # type: ignore[arg-type]
    none = SimpleNamespace(spec=SimpleNamespace(labels={"via": "mcp"}))
    assert statefile._origin(none) is None  # type: ignore[arg-type]


# --------------------------------------------------------------------------- timing

GPU = Path(sys.executable).with_name("gpu")


@pytest.mark.skipif(not GPU.is_file(), reason="no gpu console script in this venv")
def test_scoped_render_under_50ms(tmp_path: Path) -> None:
    snap = _two_sessions()
    (tmp_path / "state.json").write_text(json.dumps(dict(snap, daemon_pid=1)))
    cmd = [str(GPU), "status", "--line", "--stdin", "--now", str(NOW)]
    env = {"GPU_ROUTER_HOME": str(tmp_path), "PATH": "/usr/bin:/bin", "CLAUDE_PID": "1001"}
    payload = json.dumps({"session_id": S1, "workspace": {"current_dir": "/tmp"}}).encode()
    first = subprocess.run(cmd, input=payload, capture_output=True, env=env, check=True)
    assert b"alpha_train" in first.stdout
    assert b"beta_eval" not in first.stdout
    wall = []
    for _ in range(25):
        t0 = time.perf_counter()
        subprocess.run(cmd, input=payload, capture_output=True, env=env, check=True)
        wall.append((time.perf_counter() - t0) * 1000)
    p50 = statistics.median(wall)
    assert p50 < 50, f"p50 {p50:.1f} ms"
