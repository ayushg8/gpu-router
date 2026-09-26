"""The gpu rows copy a reference Claude Code status line: its tokens, its grid, and its
real rendered widths. The reference is checked in (tests/fixtures/statusline/statusline.sh,
a trimmed copy of a real ~/.claude/statusline.sh), so this runs on any machine with jq; it
is copied into a tmp HOME before it runs. GPU_ROUTER_TEST_STATUSLINE points it at another
script (e.g. your own ~/.claude/statusline.sh) to check the rows against that instead."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from gpu_router.clock import SystemClock
from gpu_router.statusline import fast, samples

FIXTURE = Path(__file__).parents[2] / "fixtures" / "statusline" / "statusline.sh"
USER_SCRIPT = Path(os.environ.get("GPU_ROUTER_TEST_STATUSLINE") or FIXTURE)
ANSI = re.compile(r"\x1b\[[0-9;]*m")
GPU = Path(sys.executable).with_name("gpu")
WRAPPER = Path(__file__).parents[3] / "plugin" / "statusline" / "gpu-statusline.sh"

pytestmark = [
    pytest.mark.skipif(not USER_SCRIPT.is_file(), reason=f"no status line script at {USER_SCRIPT}"),
    pytest.mark.skipif(shutil.which("jq") is None, reason="the status line script needs jq"),
]


def _script() -> str:
    return USER_SCRIPT.read_text(encoding="utf-8")


def _token(name: str) -> str:
    m = re.search(rf"^{name}=\$'([^']*)'", _script(), re.MULTILINE)
    assert m, f"{name}= not found in {USER_SCRIPT}"
    return m.group(1).replace("\\033", "\x1b")


def test_colour_tokens_are_the_scripts() -> None:
    assert _token("fg") == fast.FG
    assert _token("dim") == fast.DIM
    assert _token("warn") == fast.WARN
    assert _token("crit") == fast.CRIT
    assert _token("off") == fast.OFF
    assert _token("accent") == fast.ACCENT


def test_grid_and_meter_are_the_scripts() -> None:
    text = _script()
    col = re.search(r"^COL=(\d+)", text, re.MULTILINE)
    assert col
    assert int(col.group(1)) == fast.COL
    cells = re.search(r"cells=(\d+)", text)
    assert cells
    assert int(cells.group(1)) == fast.CELLS
    assert "(pct * cells + 99) / 100" in text  # ceil, like fast._meter
    assert f"printf '{fast.FILL}'" in text
    assert f"printf '{fast.EMPTY}'" in text
    assert fast.RESET in text
    assert "-ge 90" in text
    assert "-ge 70" in text
    assert "tr 'AMP' 'amp'" in text
    assert "79200" in text  # when(): 22 h


def _payload(project: Path) -> str:
    now = SystemClock().now()
    return json.dumps(
        {
            "model": {"display_name": "Opus 5.5 (1M context)"},
            "context_window": {"used_percentage": 12},
            "workspace": {"current_dir": str(project), "project_dir": str(project)},
            "effort": {"level": "medium"},
            "rate_limits": {
                "five_hour": {"used_percentage": 20, "resets_at": int(now + 3 * 3600)},
                "seven_day": {"used_percentage": 43, "resets_at": int(now + 5 * 86400)},
            },
            "session_id": "not-a-uuid",
        }
    )


def _run_users_line(home: Path, payload: str, *, wrapper_env: dict[str, str] | None = None) -> str:
    env = {
        "HOME": str(home),
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "TZ": "America/Los_Angeles",
    }
    if wrapper_env is not None:
        env.update(wrapper_env)
        cmd = ["bash", str(WRAPPER)]
    else:
        cmd = ["bash", str(home / ".claude" / "statusline.sh")]
    proc = subprocess.run(
        cmd, input=payload.encode(), capture_output=True, timeout=30, check=True, env=env
    )
    return proc.stdout.decode()


@pytest.fixture
def sandbox(tmp_path: Path) -> tuple[Path, Path]:
    home = tmp_path / "home"
    (home / ".claude").mkdir(parents=True)
    shutil.copy(USER_SCRIPT, home / ".claude" / "statusline.sh")
    project = tmp_path / "proj"
    project.mkdir()
    return home, project


def test_real_rendered_row_two_puts_week_on_our_grid(sandbox: tuple[Path, Path]) -> None:
    home, project = sandbox
    out = _run_users_line(home, _payload(project))
    lines = [ANSI.sub("", ln) for ln in out.split("\n")]
    assert lines[1].startswith("session ")
    week = lines[1].index("week")
    assert week == fast.COL
    assert lines[1][week - 1] == " "
    # our rows put column 2 at the same place
    for sample in samples.SAMPLES:
        for row in sample.rows(color=False):
            assert row[week - 1] == " ", row
            assert row[week] != " ", row
    # the script's row 2 and our mirror of it render the same shape
    mine = samples.user_rows(color=False)[1]
    assert re.sub(r"\d", "0", mine.split("↻")[0]) == re.sub(r"\d", "0", lines[1].split("↻")[0])


@pytest.mark.skipif(not GPU.is_file(), reason="no gpu console script in this venv")
def test_wrapper_keeps_the_users_lines_byte_for_byte(sandbox: tuple[Path, Path]) -> None:
    home, project = sandbox
    gpu_home = home / "gpuhome"
    from tests.unit.statusline.test_wrapper import live_state

    live_state(gpu_home)
    # D56: rows are scoped to the launching session, so the job must carry this payload's
    # session id or the status line (correctly) hides it.
    state_path = gpu_home / "state.json"
    snap = json.loads(state_path.read_text())
    for job in snap["active"]:
        job["origin"] = {"claude_session": "not-a-uuid"}
    state_path.write_text(json.dumps(snap))
    payload = _payload(project)
    alone_before = _run_users_line(home, payload)
    wrapped = _run_users_line(
        home,
        payload,
        wrapper_env={
            "GPU_STATUSLINE_ORIGINAL": 'bash "$HOME/.claude/statusline.sh"',
            "GPU_ROUTER_BIN": str(GPU),
            "GPU_ROUTER_HOME": str(gpu_home),
        },
    )
    alone_after = _run_users_line(home, payload)
    prefix = wrapped[: wrapped.rfind("\n", 0, wrapped.rfind("\n"))]  # minus 2 gpu rows
    assert prefix in (alone_before, alone_after)
    gpu_part = wrapped[len(prefix) + 1 :]
    assert re.match(r"gpu ████░░░░░░ 38% 1:5\d left  kaggle ", ANSI.sub("", gpu_part))
    assert len(wrapped.split("\n")) == len(alone_before.split("\n")) + 2
