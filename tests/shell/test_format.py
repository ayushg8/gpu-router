"""Pure formatting of the shell: metric history, panel rows, footer, chart."""

from __future__ import annotations

from rich.cells import cell_len

from gpu_router import protocol
from gpu_router.models import JobState, ProviderHealth
from gpu_router.shell.chart import area_rows, chart_lines, resample, stretch
from gpu_router.shell.metrics import MetricHistory, fmt_value, sparkline, trend
from gpu_router.shell.panel import (
    empty_lines,
    fit,
    footer_text,
    icon_cell,
    notice_line,
    panel_lines,
    quota_short,
)
from gpu_router.shell.state import Conn, ConnStatus, JobMetric, Snapshot
from tests.shell.helpers import NOW, job, mockup_providers, provider, quota

UP = Conn(ConnStatus.UP, last_ok=NOW)


def plain(lines: list) -> list[str]:
    return [t.plain for t in lines]


# --------------------------------------------------------------------------- metrics


def test_stdout_fallback_builds_history() -> None:
    h = MetricHistory()
    for i, loss in enumerate([2.0, 1.5, 1.1], start=1):
        assert h.feed(f"step {i}/10 loss={loss}")
    assert h.values("loss") == [2.0, 1.5, 1.1]
    assert (h.step, h.total) == (3, 10)
    assert h.primary() == "loss"


def test_helper_metrics_win_over_stdout() -> None:
    h = MetricHistory()
    h.feed("loss=9.0")  # a guess from the fallback, before the helper speaks
    h.feed(protocol.total(100))
    h.feed(protocol.metric(1, {"loss": 2.0}))
    h.feed("step 1/100 loss=2.0")  # the helper's own human line: ignored now
    h.feed(protocol.metric(2, {"loss": 1.8, "acc": 0.5}))
    assert h.values("loss") == [2.0, 1.8]
    assert h.values("acc") == [0.5]
    assert h.helper
    assert h.total == 100
    assert h.step == 2


def test_history_thins_instead_of_dropping_the_start() -> None:
    from gpu_router.shell import metrics

    h = MetricHistory()
    for i in range(metrics.MAX_POINTS + 10):
        h.feed(protocol.metric(i, {"loss": float(i)}))
    values = h.values("loss")
    assert len(values) <= metrics.MAX_POINTS
    assert values[0] == 0.0  # the run's start is still there (D44: never thinned away)


def test_non_finite_and_protocol_noise_are_ignored() -> None:
    h = MetricHistory()
    assert not h.feed('::gpu:: {"t":"heartbeat","n":1}')
    assert not h.feed("::gpu:: not json")
    assert not h.feed("hello world")
    assert h.series == {}


def test_sparkline_and_trend() -> None:
    falling = [2.0, 1.8, 1.5, 1.2, 1.0, 0.8, 0.6, 0.5, 0.45, 0.41]
    spark = sparkline(falling, 12)
    assert len(spark) == len(falling)
    assert spark[0] == "█"
    assert spark[-1] == "▁"
    assert trend(falling) == "↓"
    assert trend(list(reversed(falling))) == "↑"
    assert trend([1.0, 1.0, 1.0, 1.0, 1.0]) == "→"
    assert trend([1.0]) == ""
    assert sparkline([], 8) == ""
    assert sparkline([3.0, 3.0], 8) == "▄▄"
    assert len(sparkline(list(range(100)), 12)) == 12  # the newest 12


def test_fmt_value() -> None:
    assert fmt_value(0.41234) == "0.412"
    assert fmt_value(12.345) == "12.3"
    assert fmt_value(1234.5) == "1234"
    assert fmt_value(0.00123) == "0.00123"
    assert fmt_value(3e-5) == "3e-05"
    assert fmt_value(0) == "0"


# --------------------------------------------------------------------------- panel rows


def test_icons_take_two_cells_so_columns_line_up() -> None:
    # ⚡ is double width, ⏸ single: both slots are 3 cells (icon + padding + space)
    for state in (JobState.RUNNING, JobState.AWAITING_APPROVAL, JobState.DONE, JobState.FAILED):
        assert cell_len(icon_cell(state).plain) == 3
    snap = Snapshot(
        conn=UP,
        active=(
            job(
                "a7f2",
                "train_yolo.py",
                progress={"step": 45, "total": 100},
                last_metrics={"loss": 0.412},
            ),
            job(
                "c19e",
                "eval.py",
                JobState.AWAITING_APPROVAL,
                provider="colab",
                gpu="T4",
                route_reason="colab: kaggle saved for long jobs",
                approval_reason="over 1 GPU-hr · colab T4",
            ),
        ),
        metrics={"a7f2000000000"[:12]: JobMetric("loss", (0.9, 0.7, 0.5, 0.412), 45, 100)},
        ckpt_where={},
    )
    lines = plain(panel_lines(snap, 76, 12, now=NOW, example="train.py"))
    assert len(lines) == 4
    col = cell_len(lines[0][: lines[0].index("train_yolo.py")])
    assert col == cell_len(lines[2][: lines[2].index("eval.py")])  # same name column (cells)
    assert lines[0].startswith("⚡ job a7f2")
    assert lines[0].endswith("01:42:10")  # elapsed h:mm:ss, right-aligned
    assert "kaggle · 2xT4" in lines[0]
    assert lines[1].strip().startswith("step 45/100")
    assert "loss 0.412" in lines[1]
    assert "↓" in lines[1]
    assert lines[2].startswith("⏸  job c19e")
    assert "needs approval" in lines[2]
    assert lines[2].endswith("/approve c19e")
    assert lines[3].strip() == "route → colab T4 (kaggle saved for long jobs) · over 1 GPU-hr"
    assert all(cell_len(ln) <= 76 for ln in lines)


def test_colour_only_on_icon_and_state_word() -> None:
    snap = Snapshot(conn=UP, active=(job(progress={"step": 3}),))
    first = panel_lines(snap, 80, 12, now=NOW, example="train.py")[0]
    styled = {first.plain[s.start : s.end]: str(s.style) for s in first.spans}
    assert styled["⚡"] == "green"
    assert "green" not in {v for k, v in styled.items() if k != "⚡"}
    assert first.style == ""  # no base style that would bleed into the whole row


def test_checkpoint_age_and_destination() -> None:
    running = job(checkpoint_count=3, last_checkpoint_at=NOW - 180, last_metrics={"loss": 0.4})
    snap = Snapshot(conn=UP, active=(running,), ckpt_where={running.id: "HF Hub"})
    line2 = plain(panel_lines(snap, 90, 12, now=NOW, example="train.py"))[1]
    assert "ckpt 3m00s ago → HF Hub" in line2


def test_narrow_width_truncates_instead_of_wrapping() -> None:
    long_name = "a_really_long_training_script_name_for_experiments.py"
    snap = Snapshot(conn=UP, active=(job(name=long_name, last_metrics={"loss": 0.4}),))
    for width in (40, 60, 80):
        lines = panel_lines(snap, width, 12, now=NOW, example="train.py")
        assert all(t.cell_len <= width for t in lines), width
    assert "…" in panel_lines(snap, 60, 12, now=NOW, example="x")[0].plain


def test_too_many_jobs_collapse_and_keep_approvals() -> None:
    running = tuple(job(f"{i:04x}", f"sweep_{i}.py", last_metrics={"loss": 0.5}) for i in range(8))
    waiting = job("c19e", "eval.py", JobState.AWAITING_APPROVAL, provider="colab")
    snap = Snapshot(conn=UP, active=(*running, waiting))
    lines = plain(panel_lines(snap, 90, 6, now=NOW, example="train.py"))
    assert len(lines) == 6
    assert any("c19e" in ln for ln in lines)  # the approval is never hidden
    assert lines[-1].strip().startswith("+4 more (4 running)")
    assert "/jobs" in lines[-1]


def test_recent_finished_jobs_show_one_line() -> None:
    done = job(
        "b3d1",
        "prep.py",
        JobState.DONE,
        provider="colab",
        gpu="T4",
        started_at=NOW - 200,
        finished_at=NOW - 8,
        outputs_dir="/tmp/proj/runs/b3d1",
    )
    failed = job("e5a0", "bad.py", JobState.FAILED, message="exit code 1", finished_at=NOW - 3)
    snap = Snapshot(conn=UP, recent=(done, failed))
    lines = plain(panel_lines(snap, 90, 12, now=NOW, example="train.py"))
    assert lines[0].startswith("✓  job b3d1")
    assert "done in 3m12s on colab" in lines[0]
    assert lines[0].endswith("runs/b3d1")
    assert lines[1].startswith("✗  job e5a0")
    assert "failed: exit code 1" in lines[1]


def test_empty_state_shows_quota_left_and_an_example_run() -> None:
    lines = plain(empty_lines(mockup_providers(), 100, now=NOW, example="train_yolo.py"))
    assert lines[0] == "no jobs running"
    assert lines[1].startswith("free GPU time  kaggle     8h left ↻")
    assert lines[2].strip() == "colab      ● up"
    assert lines[3].strip().startswith("lightning  6h left")
    assert not any("modal" in ln for ln in lines)  # phase 7b: dropped, never shown
    assert "/run train_yolo.py" in lines[4]
    assert "best free GPU" in lines[4]
    assert "/route train_yolo.py" in lines[5]
    # one name column, one value column
    assert {ln.index("left") for ln in (lines[1], lines[3])} == {lines[1].index("left")}


def test_empty_state_flows_many_providers_and_fits_narrow_screens() -> None:
    many = [provider(f"p{i}", q=quota(f"p{i}", 1, 10)) for i in range(9)]
    lines = empty_lines(many, 60, now=NOW, example="train.py")
    assert all(t.cell_len <= 60 for t in lines)
    assert len(lines) < 9 + 3  # several providers per row
    assert empty_lines([], 60, now=NOW, example="x")[1].plain.endswith("/providers")


def test_connection_states() -> None:
    connecting = Snapshot(conn=Conn(ConnStatus.CONNECTING))
    assert "connecting" in plain(panel_lines(connecting, 80, 12, now=NOW, example="x"))[0]
    starting = Snapshot(conn=Conn(ConnStatus.STARTING))
    assert (
        "starting it in the background"
        in plain(panel_lines(starting, 80, 12, now=NOW, example="x"))[0]
    )
    down = Snapshot(
        conn=Conn(
            ConnStatus.DOWN,
            message="the gpu-router daemon is not running",
            hint="start it with `gpu daemon start`",
            retry_s=3,
        )
    )
    lines = plain(panel_lines(down, 80, 12, now=NOW, example="x"))
    assert lines[0].startswith("✗")
    assert "not running" in lines[0]
    assert "gpu daemon start" in lines[1]
    assert "retrying every 3s" in lines[2]
    assert "/doctor" in lines[2]


def test_notice_line_is_a_sentence() -> None:
    done = job(
        "b3d1",
        "prep.py",
        JobState.DONE,
        provider="colab",
        gpu="T4",
        started_at=NOW - 200,
        finished_at=NOW - 8,
        outputs_dir="/elsewhere/runs/b3d1",
    )
    assert notice_line(done, NOW).plain == (
        "✓  job b3d1 prep.py done in 3m12s on colab · T4 → /elsewhere/runs/b3d1"
    )
    failed = job("e5a0", "bad.py", JobState.FAILED, message="exit code 1")
    assert notice_line(failed, NOW).plain.endswith("failed: exit code 1  /logs e5a0")


# --------------------------------------------------------------------------- footer


def test_footer_matches_the_mockup() -> None:
    snap = Snapshot(
        conn=UP,
        providers=tuple(mockup_providers()),
        active=(job(), job("c19e", "eval.py", JobState.AWAITING_APPROVAL)),
    )
    text = footer_text(snap, 120, now=NOW).plain
    assert text.startswith("kaggle 22/30h ↻")
    assert "│ colab ● up │" in text
    assert "lightning 14/20h" in text
    assert "modal" not in text  # phase 7b: dropped (needs a card), never in the footer
    assert text.endswith("│ 1 running")


def test_footer_fits_narrow_widths() -> None:
    snap = Snapshot(conn=UP, providers=tuple(mockup_providers()), active=(job(),))
    wide = footer_text(snap, 200, now=NOW).plain
    mid = footer_text(snap, 60, now=NOW).plain
    tiny = footer_text(snap, 30, now=NOW).plain
    assert "↻" in wide
    assert "↻" not in mid
    assert cell_len(mid) <= 60
    assert mid.endswith("1 running")
    assert cell_len(tiny) <= 30
    assert tiny.endswith("1 running")
    assert "+" in tiny


def test_footer_states() -> None:
    down = Snapshot(conn=Conn(ConnStatus.DOWN, retry_s=3))
    assert footer_text(down, 80, now=NOW).plain == "✗ daemon down │ retrying every 3s"
    idle = Snapshot(conn=UP, providers=(provider("kaggle", q=quota("kaggle", 1, 30)),))
    assert footer_text(idle, 80, now=NOW).plain.endswith("│ idle")
    stale = Snapshot(conn=Conn(ConnStatus.UP, last_ok=NOW - 40), providers=())
    assert "stale 40s" in footer_text(stale, 80, now=NOW).plain
    login = Snapshot(conn=UP, providers=(provider("colab", health=ProviderHealth.AUTH_REQUIRED),))
    assert "colab ● login needed" in footer_text(login, 80, now=NOW).plain


def test_quota_short_labels_estimates() -> None:
    q = quota("colab", 2.5, 12, source="estimate", resets_in=None)
    assert quota_short(q, NOW) == "~2.5/12h"


# --------------------------------------------------------------------------- chart


def test_resample_keeps_the_whole_run() -> None:
    values = [float(i) for i in range(1000)]
    out = resample(values, 50)
    assert len(out) == 50
    assert out[0] < 20
    assert out[-1] > 980
    assert resample([1.0, 2.0], 50) == [1.0, 2.0]


def test_area_rows_draw_a_falling_curve() -> None:
    rows, lo, hi = area_rows([2.0, 1.5, 1.0, 0.5], 20, 4)
    assert len(rows) == 4
    assert all(r.cell_len == 20 for r in rows)
    assert (lo, hi) == (0.5, 2.0)
    top, bottom = rows[0].plain, rows[-1].plain
    assert top[0] == "█"
    assert top[-1] == " "  # high on the left, empty top-right
    assert bottom[-1] != " "  # the lowest value still gets a sliver
    # the value edge is plain ink, the fill under it dim
    first_col_styles = [
        next((str(sp.style) for sp in r.spans if sp.start <= 0 < sp.end), "") for r in rows
    ]
    assert first_col_styles[0] == ""
    assert set(first_col_styles[1:]) == {"grey30"}


def test_stretch_interpolates_short_runs() -> None:
    assert stretch([0.0, 1.0], 5) == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert len(stretch([float(i) for i in range(500)], 40)) == 40
    assert stretch([3.0], 3) == [3.0, 3.0, 3.0]


def test_chart_lines_title_axes_and_empty() -> None:
    lines = chart_lines(
        "loss", [2.0, 1.2, 0.8, 0.41], width=60, height=6, step=40, total=100, first_step=10
    )
    text = plain(lines)
    assert text[0].startswith("loss  0.410 ↓   step 40/100   min 0.410 · max 2.00")
    assert text[1].lstrip().startswith("2.00 ┤")
    assert text[6].lstrip().startswith("0.410 ┤")
    assert text[-1].strip().startswith("step 10")
    assert text[-1].endswith("step 40")
    assert all(cell_len(t) <= 60 for t in text)
    empty = plain(chart_lines("loss", [], width=60, height=6))
    assert "no points yet" in empty[0]
    assert fit("abcdef", 4) == "abc…"


def test_checkpoint_where_names_the_storage_backend() -> None:
    from gpu_router.shell.feed import checkpoint_where

    stored = "file:///Users/me/Library/Application%20Support/gpu-router/storage"
    assert checkpoint_where(f"{stored}/jobs/0123456789ab/ckpt-0003", "kaggle") == "local storage"
    assert checkpoint_where("hf://buckets/me/gpu-router/jobs/0123456789ab/ckpt-0003", "colab") == (
        "HF Hub"
    )
    assert checkpoint_where("file:///content/gr/s/ckpt-sync/c.tar.gz", "colab") == "colab disk"
    assert checkpoint_where("file:///tmp/run/ckpt-sync/c.tar.gz", "local") == "this Mac"
