# ruff: noqa: RUF001  (goldens hold the real glyphs, the multiplication sign included)
"""`gpu status --line` rendering (statusline/fast.py): goldens for every row state, the
grid, the 2-row cap, stale / missing / broken files."""

from __future__ import annotations

import io
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from gpu_router import statefile
from gpu_router.statusline import fast, samples
from gpu_router.statusline.samples import NOW, SAMPLES, snapshot

ANSI = re.compile(r"\x1b\[[0-9;]*m")
G, D, W, C, Z, OK = fast.FG, fast.DIM, fast.WARN, fast.CRIT, fast.OFF, fast.OK  # Z = reset

#: Plain text of every sample (TZ=America/Los_Angeles, clock at samples.NOW).
GOLDEN: dict[str, list[str]] = {
    "running": [
        "gpu ████░░░░░░ 38% 1:50 left  kaggle ████████░░ 73% ↻Sat 5pm",
        "train_yolo · 2×T4             loss 0.412 ↓  ckpt 3m ago",
    ],
    "running-elapsed": [
        "gpu ██░░░░░░░░ (1:42 of 12h)  colab ██░░░░░░░░ ~14% ↻4pm",
        "train_lm · T4                 loss 2.31 →  ckpt 12m ago",
    ],
    "approval": ["gpu ⏸ eval.py → colab T4      ~20m  /gpu-approve"],
    "finished": ["gpu ✓ train_yolo · 3h12m      → ./runs/a7f2"],
    "migrated": [
        "gpu ↪ train_yolo              colab → kaggle  resumed · ckpt 4",
        "gpu █████░░░░░ 42% 1:30 left  kaggle ████████░░ 73% ↻Sat 5pm",
    ],
    "migrating": ["gpu ↪ train_yolo  from colab  resuming · ckpt 4"],
    "failed": ["gpu ✗ train_yolo · 14m        exit 1 · gpu logs a7f2"],
    "starting": ["gpu train_yolo → kaggle 2×T4  starting"],
    "several": [
        "gpu ⏸ eval.py → colab T4      ~20m  /gpu-approve",
        "gpu ████░░░░░░ 38% 1:50 left  kaggle ████████░░ 73% ↻Sat 5pm  +2 queued",
    ],
    "two-running": [
        "gpu ████░░░░░░ 38% 1:50 left  kaggle ████████░░ 73% ↻Sat 5pm",
        "train_yolo · 2×T4             loss 0.412 ↓  ckpt 3m ago  +1 running · +1 queued",
    ],
    "local": [
        "gpu ████░░░░░░ 38% 1:50 left  local unlimited",
        "train_yolo · MPS              loss 0.412 ↓  ckpt 3m ago",
    ],
    "daemon-down": ["gpu daemon not running        gpu daemon start"],
    "idle": [],
}


def _sample(key: str) -> samples.Sample:
    return next(s for s in SAMPLES if s.key == key)


def _rows(snap: dict[str, Any], now: float = NOW, **kw: Any) -> list[str]:
    kw.setdefault("pid_alive", lambda _pid: True)
    kw.setdefault("cwd", samples.PROJECT)
    return fast.rows_text(snap, now=now, color=False, **kw)


def _right_col(row: str) -> int:
    """Visible column where the right column starts (first char after the grid gap)."""
    plain = ANSI.sub("", row)
    assert len(plain) > fast.COL, row
    assert plain[fast.COL - 1] == " ", f"left cell spills into column 2: {plain!r}"
    assert plain[fast.COL] != " ", f"right column does not start at {fast.COL}: {plain!r}"
    return fast.COL


# --------------------------------------------------------------------------- goldens


def test_every_sample_has_a_golden() -> None:
    assert sorted(GOLDEN) == sorted(s.key for s in SAMPLES)


@pytest.mark.parametrize("key", list(GOLDEN))
def test_golden_plain(key: str) -> None:
    assert _sample(key).rows(color=False) == GOLDEN[key]


@pytest.mark.parametrize("key", list(GOLDEN))
def test_ansi_stripped_equals_plain(key: str) -> None:
    colored = _sample(key).rows(color=True)
    assert [ANSI.sub("", r) for r in colored] == GOLDEN[key]


def test_running_raw_ansi_matches_the_scripts_codes() -> None:
    bar, detail = _sample("running").rows(color=True)
    assert bar == (
        f"{D}gpu{Z} {G}████{D}░░░░░░{Z} {G}38%{Z} {G}1:50{Z} {D}left{Z}  "
        f"{D}kaggle{Z} {W}████████{D}░░{Z} {W}73%{Z} {D}↻Sat 5pm{Z}"
    )
    assert detail == (
        f"{G}train_yolo{Z} {D}·{Z} {D}2×T4{Z}             "
        f"{D}loss{Z} {G}0.412{Z} {G}↓{Z}  {D}ckpt{Z} {G}3m ago{Z}"
    )
    assert "\x1b[1m" not in bar  # the script's % is 38;5;252, not bold


def test_state_icons_are_the_only_colour_besides_quota_thresholds() -> None:
    (approval,) = _sample("approval").rows(color=True)
    (finished,) = _sample("finished").rows(color=True)
    (failed,) = _sample("failed").rows(color=True)
    (migrating,) = _sample("migrating").rows(color=True)
    assert f"{W}⏸{Z}" in approval
    assert f"{OK}✓{Z}" in finished
    assert f"{C}✗{Z}" in failed
    assert f"{G}↪{Z}" in migrating  # ↪ is not a coloured state
    colours = {W, C, OK, fast.ACCENT}
    for row in (approval, finished, failed, migrating):
        found = [c for c in colours if c in row]
        assert len(found) <= 1, row
    for sample in SAMPLES:
        for row in sample.rows(color=True):
            assert fast.ACCENT not in row  # blue belongs to the model name only


def test_progress_bar_is_never_threshold_coloured() -> None:
    snap = snapshot([samples._job(step=950, eta_s=300.0)])
    (bar, _detail) = fast.rows_text(snap, now=NOW, pid_alive=lambda _p: True)
    assert bar.startswith(f"{D}gpu{Z} {G}██████████{D}{Z} {G}95%{Z}")


@pytest.mark.parametrize(
    ("remaining", "colour", "shown"),
    [(29.0, G, "3%"), (8.0, W, "73%"), (2.0, C, "93%"), (29.9, G, "<1%")],
)
def test_quota_meter_uses_the_scripts_thresholds(remaining: float, colour: str, shown: str) -> None:
    snap = snapshot([samples._job()])
    snap["providers"][0]["remaining"] = remaining
    snap["providers"][0]["used"] = 30.0 - remaining
    bar = fast.rows_text(snap, now=NOW, pid_alive=lambda _p: True)[0]
    assert f"{colour}{shown}{Z}" in bar


# --------------------------------------------------------------------------- grid


@pytest.mark.parametrize("key", [k for k, rows in GOLDEN.items() if rows])
def test_right_column_starts_on_the_grid(key: str) -> None:
    for row in _sample(key).rows(color=True):
        _right_col(row)


def test_grid_matches_the_users_row_two() -> None:
    mine = samples.user_rows(color=False)[1]
    assert mine.index("week") == fast.COL
    for sample in SAMPLES:
        for row in sample.rows(color=False):
            assert _right_col(row) == mine.index("week")


def test_long_names_are_cut_with_an_ellipsis_and_keep_the_grid() -> None:
    long = "a_really_long_experiment_name_for_the_sweep"
    snap = snapshot([samples._job(name=long)], recent=[samples._recent(name=long)])
    rows = _rows(snap)
    assert rows[1].startswith("a_really_long_experiment_na…")
    assert fast.width(rows[1][: fast.COL].rstrip()) <= fast.COL - fast.GUTTER
    for row in rows:
        _right_col(row)
    (done,) = _rows(snapshot(recent=[samples._recent(name=long)]))
    assert done.startswith("gpu ✓ a_really_long_experim…  3h12m")
    assert "3h12m  → ./runs/a7f2" in done  # the duration moved right, nothing lost


def test_wide_characters_count_two_cells() -> None:
    snap = snapshot([samples._job(name="学習ジョブ_long_name_x")])
    detail = _rows(snap)[1]
    assert fast.width(detail.split("  ")[0]) <= fast.COL - fast.GUTTER
    _right_col(
        detail.replace("学", "xx")
        .replace("習", "xx")
        .replace("ジ", "xx")
        .replace("ョ", "xx")
        .replace("ブ", "xx")
    )


def test_ten_hours_left_drops_the_word_left() -> None:
    snap = snapshot([samples._job(eta_s=11 * 3600 + 20.0)])
    bar = _rows(snap)[0]
    assert bar.startswith("gpu ████░░░░░░ 38% 11:00  ")


def test_elapsed_fallback_labels_the_session_cap() -> None:
    snap = snapshot([samples._job(total_steps=None, step=None, eta_s=None, started_at=NOW - 3600)])
    assert _rows(snap)[0].startswith("gpu ▌".replace("▌", "█") + "░" * 9 + " (1:00 of 12h)")


def test_no_cap_and_no_steps_shows_elapsed() -> None:
    snap = snapshot([samples._job(total_steps=None, step=None, eta_s=None, session_cap_s=None)])
    assert _rows(snap)[0].startswith("gpu 1:02 elapsed")


# --------------------------------------------------------------------------- composition


def test_at_most_two_rows_with_counts() -> None:
    jobs = [samples._job(), samples.APPROVAL] + [samples._queued(i) for i in range(5)]
    jobs.append(samples._job(id="bbbbbbbbbbbb", short_id="bbbb", name="other"))
    rows = _rows(snapshot(jobs, recent=[samples._recent()]))
    assert len(rows) == 2
    assert rows[1].endswith("+1 running · +5 queued · +1 done")


def test_failure_outranks_a_running_job() -> None:
    snap = snapshot(
        [samples._job()], recent=[samples._recent(state="failed", failure_kind="internal")]
    )
    rows = _rows(snap)
    assert rows[0].startswith("gpu ✗ train_yolo")
    assert "internal error · gpu logs a7f2" in rows[0]
    assert rows[1].startswith("gpu ████")


def test_two_approvals_name_the_job() -> None:
    second = dict(samples.APPROVAL, id="e5e5e5e5e5e5", short_id="e5e5", script="bench.py")
    rows = _rows(snapshot([samples.APPROVAL, second]))
    assert rows[0].endswith("/gpu-approve c19e")
    assert rows[1].endswith("/gpu-approve e5e5")


def test_finished_rows_expire_after_the_window() -> None:
    snap = snapshot(recent=[samples._recent(finished_at=NOW - 599)])
    assert _rows(snap)
    snap = snapshot(recent=[samples._recent(finished_at=NOW - 601)])
    assert _rows(snap) == []
    snap = snapshot(recent=[samples._recent(finished_at=NOW - 601)], finished_visible_s=900.0)
    assert _rows(snap)


def test_zero_window_hides_finished_rows() -> None:
    # the default since 2026-09-25: rows only while a job is in use
    snap = snapshot(recent=[samples._recent(finished_at=NOW - 1)], finished_visible_s=0.0)
    assert _rows(snap) == []
    snap = snapshot([samples._job()], recent=[samples._recent()], finished_visible_s=0.0)
    assert len(_rows(snap)) == 2  # the running job's two rows, no finished row


def test_config_default_shows_rows_only_while_in_use() -> None:
    from gpu_router.config import StatusLineConfig

    assert StatusLineConfig().finished_visible_s == 0


def test_cancelled_and_denied_jobs_print_nothing() -> None:
    snap = snapshot(recent=[samples._recent(state="cancelled"), samples._recent(state="denied")])
    assert _rows(snap) == []


def test_migrated_row_expires_back_to_running_rows() -> None:
    later = NOW + 601
    migrated = dict(_sample("migrated").snapshot, written_at=later - 10)
    rows = _rows(migrated, now=later)
    assert rows[0].startswith("gpu █")
    assert rows[1].startswith("train_yolo · 2×T4")


def test_outputs_outside_the_project_name_it() -> None:
    (row,) = _rows(snapshot(recent=[samples._recent()]), cwd="/somewhere/else")
    assert row.endswith("→ yolo/runs/a7f2")
    (row,) = _rows(snapshot(recent=[samples._recent()]), cwd=samples.PROJECT + "/src")
    assert row.endswith("→ ./runs/a7f2")


def test_outputs_not_fetched_points_at_fetch() -> None:
    (row,) = _rows(snapshot(recent=[samples._recent(outputs_fetched=False)]))
    assert row.endswith("gpu fetch a7f2")


def test_queued_with_backoff_says_when() -> None:
    job = samples._queued(1)
    job["not_before"] = NOW + 125
    (row,) = _rows(snapshot([job]))
    assert row == "gpu sweep_1                   queued · retry in 2m"


def test_older_daemon_file_without_phase6b_fields() -> None:
    old_keys = {  # ActiveJob as phase 1 wrote it
        "id",
        "short_id",
        "name",
        "state",
        "provider",
        "gpu",
        "created_at",
        "started_at",
        "session_cap_s",
        "step",
        "total_steps",
        "progress_source",
        "eta_s",
        "metric",
        "last_checkpoint_at",
        "checkpoint_seq",
        "route_summary",
        "approval_reason",
        "not_before",
    }
    job = {k: v for k, v in samples._job().items() if k in old_keys}
    appr = {k: v for k, v in samples.APPROVAL.items() if k in old_keys}
    snap = snapshot([job, appr])
    for key in ("finished_visible_s", "migrated_visible_s", "heartbeat_s"):
        snap.pop(key)
    for p in snap["providers"]:
        p.pop("unlimited")
        p.pop("remaining")
    rows = _rows(snap)
    assert rows[0] == "gpu ⏸ eval → colab T4 · ~20m  /gpu-approve"  # name, route_summary
    assert rows[1].startswith("gpu ████░░░░░░ 38% 1:50 left  kaggle ████████░░ 73%")


# --------------------------------------------------------------------------- stale / broken


def test_daemon_gone_prints_nothing_when_idle() -> None:
    idle = snapshot(recent=[samples._recent()])
    assert fast.rows_text(idle, now=NOW, pid_alive=lambda _p: False) == []


def test_real_pid_check(tmp_path: Path) -> None:
    proc = subprocess.run(
        [sys.executable, "-c", "import os; print(os.getpid())"],
        capture_output=True,
        text=True,
        check=True,
    )
    dead = int(proc.stdout)
    snap = snapshot([samples._job()], daemon_pid=dead)
    assert fast.rows_text(snap, now=NOW, color=False) == [
        "gpu daemon not running        gpu daemon start"
    ]
    import os

    snap["daemon_pid"] = os.getpid()
    assert len(fast.rows_text(snap, now=NOW, color=False)) == 2
    snap["daemon_pid"] = 1  # launchd: alive, not ours (EPERM) -> alive
    assert len(fast.rows_text(snap, now=NOW, color=False)) == 2


def test_overdue_heartbeat_is_a_dim_hint() -> None:
    snap = snapshot([samples._job()], written_at=NOW - 301)
    assert _rows(snap) == ["gpu daemon not responding     gpu daemon status"]
    raw = fast.rows_text(snap, now=NOW, pid_alive=lambda _p: True)[0]
    assert raw == f"{D}gpu{Z} {D}daemon not responding{Z}     {D}gpu daemon status{Z}"
    assert len(_rows(snapshot([samples._job()], written_at=NOW - 299))) == 2
    # an idle daemon writes no heartbeat: an old file is fine
    assert _rows(snapshot(recent=[samples._recent()], written_at=NOW - 110)) == [
        "gpu ✓ train_yolo · 3h12m      → ./runs/a7f2"
    ]


def _write(home: Path, doc: Any) -> None:
    home.mkdir(parents=True, exist_ok=True)
    text = doc if isinstance(doc, str) else json.dumps(doc)
    (home / "state.json").write_text(text)


def test_render_missing_file(tmp_path: Path) -> None:
    assert fast.render(["--line"], environ={"GPU_ROUTER_HOME": str(tmp_path / "nope")}) == ""


@pytest.mark.parametrize(
    "doc",
    [
        "",
        "{not json",
        "[1, 2]",
        {"schema": 99, "active": [samples._job()], "daemon_pid": 1},
        {"schema": 1, "active": "oops", "recent": None, "daemon_pid": 1, "providers": 7},
        {
            "schema": 1,
            "active": [{"state": "running", "step": "x", "total_steps": {}}],
            "daemon_pid": 1,
        },
        {"schema": 1, "active": [None, 3, "s"], "recent": [{"state": "done"}], "daemon_pid": 1},
    ],
)
def test_render_never_raises_on_broken_files(tmp_path: Path, doc: Any) -> None:
    _write(tmp_path, doc)
    out = fast.render(["--line"], environ={"GPU_ROUTER_HOME": str(tmp_path)}, now=NOW)
    assert isinstance(out, str)
    assert "Traceback" not in out


def test_render_reads_the_file_and_stdin_cwd(tmp_path: Path) -> None:
    snap = snapshot(recent=[samples._recent()], daemon_pid=1)
    _write(tmp_path, snap)
    env = {"GPU_ROUTER_HOME": str(tmp_path)}
    payload = json.dumps({"workspace": {"current_dir": "/elsewhere"}, "model": {}})
    out = fast.render(
        ["--line", "--stdin", "--plain"], environ=env, now=NOW, stdin=io.StringIO(payload)
    )
    assert out == "gpu ✓ train_yolo · 3h12m      → yolo/runs/a7f2"
    out = fast.render(["--line", "--plain", "--cwd", samples.PROJECT], environ=env, now=NOW)
    assert out == "gpu ✓ train_yolo · 3h12m      → ./runs/a7f2"
    bad = fast.render(
        ["--line", "--stdin", "--plain", "--cwd", samples.PROJECT],
        environ=env,
        now=NOW,
        stdin=io.StringIO("not json"),
    )
    assert bad.endswith("→ ./runs/a7f2")  # unreadable stdin: the --cwd fallback stands
    assert (
        fast.render(
            [
                "--line",
                "--home",
                str(tmp_path),
                "--now",
                str(NOW),
                "--plain",
                "--cwd",
                samples.PROJECT,
            ],
            environ={},
        )
        == bad
    )


def test_oversized_file_is_ignored(tmp_path: Path) -> None:
    snap = snapshot(recent=[samples._recent()], daemon_pid=1, pad="x" * (fast.MAX_FILE_BYTES))
    _write(tmp_path, snap)
    assert fast.render(["--line"], environ={"GPU_ROUTER_HOME": str(tmp_path)}, now=NOW) == ""


# --------------------------------------------------------------------------- formatting units


@pytest.mark.parametrize(
    ("offset_h", "text"),
    [(3, "1pm"), (14, "12am"), (21.9, "7am"), (22, "Fri 8am"), (80, "Sun 6pm"), (96, "mon 10am")],
)
def test_when_mirrors_the_scripts_reset_format(offset_h: float, text: str) -> None:
    assert fast._when(NOW + offset_h * 3600, NOW) == text  # NOW = Thu 10am PDT


def test_when_in_the_past_is_omitted() -> None:
    assert fast._when(NOW - 1, NOW) is None
    assert fast._when(None, NOW) is None


@pytest.mark.parametrize(
    ("gpu", "shown"),
    [
        ("2xT4", "2×T4"),
        ("T4", "T4"),
        ("P100", "P100"),
        ("8xA100", "8×A100"),
        ("x2", "x2"),
        ("2x", "2x"),
        (None, ""),
    ],
)
def test_gpu_names(gpu: str | None, shown: str) -> None:
    assert fast._gpu(gpu) == shown


@pytest.mark.parametrize("pct", [0, 1, 9, 10, 11, 50, 99, 100, 140])
def test_meter_is_ceil_of_tenths(pct: int) -> None:
    plain, raw = fast._meter(pct, fast.FG)
    filled = min(10, (pct * 10 + 99) // 100)
    assert plain == "█" * filled + "░" * (10 - filled)
    assert raw == f"{G}{'█' * filled}{D}{'░' * (10 - filled)}{Z}"


def test_schema_constant_matches_the_writer() -> None:
    assert fast.STATE_SCHEMA == statefile.STATE_SCHEMA


def test_fast_path_imports_stdlib_only() -> None:
    code = (
        "import sys\n"
        "from gpu_router.statusline.fast import render\n"
        "render(['--line'])\n"
        "heavy = {'pydantic', 'fastapi', 'httpx', 'typer', 'textual', 'uvicorn', 'yaml',\n"
        "         'rich', 'click', 'gpu_router.statefile', 'gpu_router.paths', 'pathlib',\n"
        "         'typing', 'dataclasses'}\n"
        "print(sorted(m for m in heavy if m in sys.modules))\n"
    )
    out = subprocess.run(
        [sys.executable, "-I", "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env={"GPU_ROUTER_HOME": "/nonexistent", "PYTHONPATH": str(Path(fast.__file__).parents[2])},
    ).stdout.strip()
    assert out == "[]"
