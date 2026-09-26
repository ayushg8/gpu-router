"""plugin/statusline/gpu-statusline.sh: the user's command runs first, unchanged, with the
same stdin bytes; `gpu status --line` rows follow; nothing breaks when gpu is absent."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from gpu_router.clock import SystemClock
from gpu_router.statusline import fast, samples

REPO = Path(__file__).parents[3]
WRAPPER = REPO / "plugin" / "statusline" / "gpu-statusline.sh"
GPU = Path(sys.executable).with_name("gpu")
ANSI = re.compile(r"\x1b\[[0-9;]*m")
PAYLOAD = json.dumps({"model": {"display_name": "Opus"}, "workspace": {"current_dir": "/x"}})

pytestmark = pytest.mark.skipif(not GPU.is_file(), reason="no gpu console script in this venv")


def live_state(home: Path, *, active: bool = True) -> None:
    """A state.json whose times are relative to the real clock (pid 1 = alive)."""
    now = SystemClock().now()
    shift = now - samples.NOW

    def moved(d: dict[str, Any]) -> dict[str, Any]:
        keys = ("created_at", "started_at", "last_checkpoint_at", "finished_at", "resets_at")
        return {k: (v + shift if k in keys and isinstance(v, float) else v) for k, v in d.items()}

    snap = samples.snapshot(
        [moved(samples._job())] if active else [], daemon_pid=1, written_at=now - 5
    )
    snap["providers"] = [moved(p) for p in snap["providers"]]
    home.mkdir(parents=True, exist_ok=True)
    (home / "state.json").write_text(json.dumps(snap))


def run(
    tmp_path: Path, env: dict[str, str], stdin: str = PAYLOAD, script: Path = WRAPPER
) -> subprocess.CompletedProcess[bytes]:
    base = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "TZ": "America/Los_Angeles"}
    return subprocess.run(
        ["bash", str(script)],
        input=stdin.encode(),
        capture_output=True,
        timeout=30,
        check=False,
        env={**base, **env},
    )


def gpu_rows(home: Path) -> str:
    proc = subprocess.run(
        [str(GPU), "status", "--line", "--stdin"],
        input=PAYLOAD.encode(),
        capture_output=True,
        timeout=30,
        check=True,
        env={"GPU_ROUTER_HOME": str(home), "TZ": "America/Los_Angeles", "PATH": "/usr/bin:/bin"},
    )
    return proc.stdout.decode().rstrip("\n")


def test_original_first_then_gpu_rows(tmp_path: Path) -> None:
    home = tmp_path / "gpuhome"
    live_state(home)
    env = {
        "GPU_STATUSLINE_ORIGINAL": "printf 'Opus 5.5 1M\\nsession 20%%'",
        "GPU_ROUTER_BIN": str(GPU),
        "GPU_ROUTER_HOME": str(home),
    }
    proc = run(tmp_path, env)
    assert proc.returncode == 0
    out = proc.stdout.decode()
    rows = gpu_rows(home)
    assert rows.count("\n") == 1
    assert out == "Opus 5.5 1M\nsession 20%\n" + rows
    assert "\x1b[38;5;243mgpu\x1b[0m" in out  # colour codes pass through untouched
    assert not out.endswith("\n")  # like the script: no trailing newline


def test_stdin_reaches_the_original_byte_for_byte(tmp_path: Path) -> None:
    dump = tmp_path / "seen.json"
    payload = PAYLOAD + "\n\n"  # trailing newlines must survive too
    env = {"GPU_STATUSLINE_ORIGINAL": f"cat > '{dump}'; printf mine", "GPU_ROUTER_BIN": "/nope"}
    proc = run(tmp_path, env, stdin=payload)
    assert proc.returncode == 0
    assert dump.read_bytes() == payload.encode()
    assert proc.stdout == b"mine"


def test_idle_gpu_leaves_the_users_line_alone(tmp_path: Path) -> None:
    home = tmp_path / "gpuhome"
    live_state(home, active=False)
    env = {
        "GPU_STATUSLINE_ORIGINAL": "printf 'a\\nb\\nc'",
        "GPU_ROUTER_BIN": str(GPU),
        "GPU_ROUTER_HOME": str(home),
    }
    assert run(tmp_path, env).stdout == b"a\nb\nc"


def test_no_gpu_anywhere_is_just_the_users_line(tmp_path: Path) -> None:
    env = {"GPU_STATUSLINE_ORIGINAL": "printf 'a\\nb'", "GPU_ROUTER_BIN": ""}
    proc = run(tmp_path, env)
    assert proc.returncode == 0
    assert proc.stdout == b"a\nb"
    env["GPU_ROUTER_BIN"] = str(tmp_path / "missing-gpu")
    assert run(tmp_path, env).stdout == b"a\nb"


def test_failing_original_still_exits_zero(tmp_path: Path) -> None:
    home = tmp_path / "gpuhome"
    live_state(home)
    env = {
        "GPU_STATUSLINE_ORIGINAL": "printf partial; exit 3",
        "GPU_ROUTER_BIN": str(GPU),
        "GPU_ROUTER_HOME": str(home),
    }
    proc = run(tmp_path, env)
    assert proc.returncode == 0
    assert proc.stdout.decode().startswith("partial\n")


def test_no_original_prints_only_gpu_rows(tmp_path: Path) -> None:
    home = tmp_path / "gpuhome"
    live_state(home)
    env = {"GPU_STATUSLINE_ORIGINAL": "", "GPU_ROUTER_BIN": str(GPU), "GPU_ROUTER_HOME": str(home)}
    assert run(tmp_path, env).stdout.decode() == gpu_rows(home)


def test_installed_layout_reads_its_data_files(tmp_path: Path) -> None:
    """What `gpu statusline install` writes: the wrapper beside original-command, gpu-bin,
    gpu-home; no environment variables at all (Claude Code passes none)."""
    home = tmp_path / "Application Support" / "gpu-router"
    live_state(home)
    folder = home / "statusline"
    folder.mkdir()
    shutil.copy(WRAPPER, folder / "gpu-statusline.sh")
    (folder / "original-command").write_text("printf 'line1\\nline2'")
    (folder / "gpu-bin").write_text(str(GPU))
    (folder / "gpu-home").write_text(str(home))
    proc = run(tmp_path, {}, script=folder / "gpu-statusline.sh")
    assert proc.returncode == 0
    assert proc.stdout.decode() == "line1\nline2\n" + gpu_rows(home)


def test_a_status_line_pointing_back_at_the_wrapper_does_not_recurse(tmp_path: Path) -> None:
    home = tmp_path / "gpuhome"
    live_state(home)
    env = {
        "GPU_STATUSLINE_ORIGINAL": f"bash '{WRAPPER}'",
        "GPU_ROUTER_BIN": str(GPU),
        "GPU_ROUTER_HOME": str(home),
    }
    proc = run(tmp_path, env)
    assert proc.returncode == 0
    assert proc.stdout.decode() == gpu_rows(home)


def test_rows_align_under_the_users_grid(tmp_path: Path) -> None:
    home = tmp_path / "gpuhome"
    live_state(home)
    mine = samples.user_rows(color=True)
    env = {
        "GPU_STATUSLINE_ORIGINAL": "printf '%s\\n%s' " + " ".join(f"'{r}'" for r in mine),
        "GPU_ROUTER_BIN": str(GPU),
        "GPU_ROUTER_HOME": str(home),
    }
    lines = [ANSI.sub("", ln) for ln in run(tmp_path, env).stdout.decode().split("\n")]
    assert len(lines) == 4
    week = lines[1].index("week")
    assert week == fast.COL
    for row in lines[2:]:
        assert row[week - 1] == " "
        assert row[week] != " "


def test_wrapper_overhead_is_small(tmp_path: Path) -> None:
    home = tmp_path / "gpuhome"
    live_state(home)
    env = {
        "GPU_STATUSLINE_ORIGINAL": "true",
        "GPU_ROUTER_BIN": str(GPU),
        "GPU_ROUTER_HOME": str(home),
    }
    run(tmp_path, env)  # warm the page cache
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        run(tmp_path, env)
        times.append(time.perf_counter() - t0)
    assert sorted(times)[2] < 0.25, times  # generous: CI boxes and parallel agents


def test_syntax_and_mode() -> None:
    assert subprocess.run(["bash", "-n", str(WRAPPER)], check=False).returncode == 0
    assert os.access(WRAPPER, os.X_OK)
