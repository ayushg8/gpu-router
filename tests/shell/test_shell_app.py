"""GpuShell driven by Textual's pilot against a real in-process daemon on the fake provider:
typing commands, the / popup, tab completion, approve/deny, live logs and watch blocks,
CLI-backed commands, notices, empty and daemon-down states, history and quitting."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from gpu_router.shell.app import GpuShell
from gpu_router.shell.widgets import CommandBlock, LiveBlock, LogsBlock, WatchBlock
from gpu_router.statemachine import JobState, is_terminal
from tests.shell.conftest import InProcDaemon, settle, write_gpu_yaml

SIZE = (110, 34)
LONG = {"duration": 120, "steps": 1200, "checkpoint_every": 3}


def shell(**kw: Any) -> GpuShell:
    kw.setdefault("poll_s", 0.2)
    kw.setdefault("quota_s", 2.0)
    kw.setdefault("retry_s", 0.3)
    return GpuShell(**kw)


def transcript(app: GpuShell) -> str:
    return "\n".join(b.plain for b in app.query(CommandBlock))


def last_block(app: GpuShell) -> CommandBlock:
    return [b for b in app.query(CommandBlock) if b.line is not None][-1]


async def type_line(pilot: Any, app: GpuShell, line: str) -> CommandBlock:
    """Type `line` into the prompt, press enter, wait for the command to finish."""
    app.prompt.value = line
    app.prompt.cursor_position = len(line)
    app.popup.hide()
    before = len([b for b in app.query(CommandBlock) if b.line is not None])
    await pilot.press("enter")
    await settle(
        pilot,
        lambda: (
            len([b for b in app.query(CommandBlock) if b.line is not None]) > before
            and not last_block(app).running
        ),
        what=f"{line!r} to finish",
    )
    return last_block(app)


async def ready(pilot: Any, app: GpuShell, pred: Any = None) -> None:
    await settle(
        pilot,
        lambda: app.snap is not None and app.snap.conn.status == "up",
        what="the feed to connect",
    )
    if pred is not None:
        await settle(pilot, pred, what="the panel state")


# --------------------------------------------------------------------------- states


async def test_empty_state_shows_quota_and_an_example_run(
    daemon: InProcDaemon, project: Path
) -> None:
    app = shell()
    async with app.run_test(size=SIZE) as pilot:
        await ready(pilot, app, lambda: "fake" in app.footer.plain)
        text = app.panel.plain
        assert "no jobs running" in text
        assert "free GPU time" in text
        assert "fake" in text
        assert "/run train.py" in text  # the project's own script
        assert "/route train.py" in text
        assert app.footer.plain.endswith("idle")
        assert "fake-b" in app.footer.plain


async def test_daemon_down_shows_a_clear_error_then_recovers(
    no_daemon: Path, project: Path
) -> None:
    app = shell()
    async with app.run_test(size=SIZE) as pilot:
        await settle(
            pilot,
            lambda: app.snap is not None and app.snap.conn.status == "down",
            what="the down state",
        )
        text = app.panel.plain
        assert text.startswith("✗")
        assert "not running" in text
        assert "gpu daemon start" in text
        assert "retrying every 0.3s" in text
        assert app.footer.plain.startswith("✗ daemon down")
        block = await type_line(pilot, app, "/jobs")
        assert "✗ the gpu-router daemon is not running" in block.plain
        assert "gpu daemon start" in block.plain
        # a daemon appears: the feed reconnects on its own
        d = InProcDaemon(home=no_daemon)
        d.start()
        try:
            await settle(
                pilot,
                lambda: app.snap is not None and app.snap.conn.status == "up",
                what="reconnect",
            )
            await settle(pilot, lambda: "no jobs running" in app.panel.plain, what="empty")
        finally:
            d.stop()


async def test_too_many_jobs_fold_into_a_summary(daemon: InProcDaemon, project: Path) -> None:
    for i in range(9):
        daemon.submit(project, LONG, name=f"sweep_{i}.py")
    app = shell()
    async with app.run_test(size=(110, 20)) as pilot:
        await ready(pilot, app, lambda: app.snap is not None and len(app.snap.active) == 9)
        await settle(pilot, lambda: "more (" in app.panel.plain, what="overflow line")
        lines = app.panel.plain.splitlines()
        assert len(lines) <= app.panel.max_rows
        assert "/jobs lists them all" in lines[-1]


# --------------------------------------------------------------------------- prompt


async def test_slash_opens_the_command_popup_and_tab_completes(
    daemon: InProcDaemon, project: Path
) -> None:
    app = shell()
    async with app.run_test(size=SIZE) as pilot:
        await ready(pilot, app)
        await pilot.press("slash")
        await settle(pilot, lambda: app.popup.display, what="popup")
        from gpu_router.shell.commands import COMMANDS

        assert len(app.popup.candidates) == len(COMMANDS)  # 21 since phase 7b's /infer
        await pilot.press("a", "p")
        assert [c.value for c in app.popup.candidates] == ["/approve"]
        await pilot.press("tab")
        assert app.prompt.value == "/approve "
        await pilot.press("escape")
        assert not app.popup.display


async def test_enter_on_the_popup_runs_a_command_without_arguments(
    daemon: InProcDaemon, project: Path
) -> None:
    app = shell()
    async with app.run_test(size=SIZE) as pilot:
        await ready(pilot, app)
        await pilot.press("slash", "q", "u")
        assert app.popup.current is not None
        assert app.popup.current.value == "/quota"
        await pilot.press("enter")
        await settle(
            pilot,
            lambda: any(b.line == "/quota" and not b.running for b in app.query(CommandBlock)),
            what="/quota",
        )
        assert "provider" in transcript(app)
        assert "fake" in transcript(app)


async def test_popup_arrow_keys_pick_a_candidate(daemon: InProcDaemon, project: Path) -> None:
    app = shell()
    async with app.run_test(size=SIZE) as pilot:
        await ready(pilot, app)
        await pilot.press("slash", "l", "o")
        assert [c.value for c in app.popup.candidates] == ["/logs", "/login"]
        await pilot.press("down")
        await pilot.press("tab")
        assert app.prompt.value == "/login "
        assert [c.value for c in app.popup.candidates] == ["fake", "fake-b"]


async def test_tab_completes_scripts(daemon: InProcDaemon, project: Path) -> None:
    app = shell()
    async with app.run_test(size=SIZE) as pilot:
        await ready(pilot, app)
        app.prompt.value = "/route mo"
        app.prompt.cursor_position = len(app.prompt.value)
        await pilot.press("tab")
        assert app.prompt.value == "/route models/"
        await pilot.press("tab")
        assert app.prompt.value == "/route models/yolo.py "


async def test_unknown_commands_and_bad_arguments(daemon: InProcDaemon, project: Path) -> None:
    app = shell()
    async with app.run_test(size=SIZE) as pilot:
        await ready(pilot, app)
        block = await type_line(pilot, app, "/aprove 1234")
        assert "unknown command: /aprove" in block.plain
        assert "did you mean /approve" in block.plain
        block = await type_line(pilot, app, "/logs")
        assert "needs a job id" in block.plain
        block = await type_line(pilot, app, "/logs zz")
        assert "is not a job id" in block.plain
        block = await type_line(pilot, app, "/run --vram lots train.py")
        assert "✗" in block.plain
        assert "--vram" in block.plain
        assert "internal error" not in transcript(app)


async def test_history_walks_back_and_persists(daemon: InProcDaemon, project: Path) -> None:
    app = shell()
    async with app.run_test(size=SIZE) as pilot:
        await ready(pilot, app)
        await type_line(pilot, app, "/help")
        await type_line(pilot, app, "/providers")
        await pilot.press("up")
        assert app.prompt.value == "/providers"
        await pilot.press("up")
        assert app.prompt.value == "/help"
        await pilot.press("down", "down")
        assert app.prompt.value == ""
    saved = (daemon.home / "shell_history").read_text().splitlines()
    assert saved[-2:] == ["/help", "/providers"]
    assert oct((daemon.home / "shell_history").stat().st_mode & 0o777) == "0o600"


# --------------------------------------------------------------------------- approvals


async def test_approve_flow_with_job_id_completion(daemon: InProcDaemon, project: Path) -> None:
    waiting = daemon.submit(
        project, {"duration": 0.5, "steps": 5}, name="eval.py", requires_approval=True
    )
    app = shell()
    async with app.run_test(size=SIZE) as pilot:
        await ready(pilot, app, lambda: "needs approval" in app.panel.plain)
        panel = app.panel.plain
        assert f"/approve {waiting.short_id}" in panel
        assert "route → fake" in panel
        await pilot.press("slash", "a", "p", "tab")
        assert app.prompt.value == "/approve "
        assert [c.value for c in app.popup.candidates] == [waiting.short_id]
        await pilot.press("tab")
        assert app.prompt.value == f"/approve {waiting.short_id} "
        await pilot.press("enter")
        await settle(pilot, lambda: is_terminal(daemon.job(waiting.id).state), what="job ends")
        assert daemon.job(waiting.id).state is JobState.DONE
        assert f"job {waiting.short_id} approved" in transcript(app)
        await settle(pilot, lambda: "done in" in app.panel.plain, what="recent row")


async def test_deny_stops_a_waiting_job(daemon: InProcDaemon, project: Path) -> None:
    waiting = daemon.submit(project, {"duration": 0.5}, name="eval.py", requires_approval=True)
    app = shell()
    async with app.run_test(size=SIZE) as pilot:
        await ready(pilot, app, lambda: "needs approval" in app.panel.plain)
        block = await type_line(pilot, app, f"/deny {waiting.short_id} not now")
        assert f"job {waiting.short_id} denied" in block.plain
        assert daemon.job(waiting.id).state is JobState.DENIED


async def test_a_new_approval_request_is_announced(daemon: InProcDaemon, project: Path) -> None:
    app = shell()
    async with app.run_test(size=SIZE) as pilot:
        await ready(pilot, app)
        await pilot.pause(0.5)  # a first snapshot without the job
        job = daemon.submit(
            project, {"duration": 0.5}, name="agent_eval.py", requires_approval=True
        )
        await settle(pilot, lambda: "needs approval  /approve" in transcript(app), what="notice")
        assert f"/approve {job.short_id}" in transcript(app)


# --------------------------------------------------------------------------- run / logs / watch


async def test_run_submits_and_streams_logs_until_done(daemon: InProcDaemon, project: Path) -> None:
    write_gpu_yaml(project, duration=0.6, steps=4)
    app = shell()
    async with app.run_test(size=SIZE) as pilot:
        await ready(pilot, app)
        app.prompt.value = "/run train.py"
        await pilot.press("enter")
        await settle(pilot, lambda: bool(app.query(LogsBlock)), what="logs block")
        logs = app.query_one(LogsBlock)
        await settle(pilot, lambda: logs.finished, what="the job to finish")
        assert "submitted job" in transcript(app)
        lines = [str(line.text) for line in logs.output.lines]
        text = "\n".join(lines)
        assert "step 4/4" in text
        assert logs.job.state is JobState.DONE
        assert "done" in logs.foot_text


async def test_run_detached_returns_right_away(daemon: InProcDaemon, project: Path) -> None:
    write_gpu_yaml(project, duration=30, steps=10)
    app = shell()
    async with app.run_test(size=SIZE) as pilot:
        await ready(pilot, app)
        block = await type_line(pilot, app, "/run -d train.py")
        assert "submitted job" in block.plain
        assert "/logs" in block.plain
        assert not app.query(LogsBlock)
        await settle(
            pilot, lambda: app.snap is not None and len(app.snap.active) == 1, what="panel row"
        )


async def test_logs_stream_live_and_escape_detaches(daemon: InProcDaemon, project: Path) -> None:
    job = daemon.submit(project, LONG, name="train_yolo.py")
    app = shell()
    async with app.run_test(size=SIZE) as pilot:
        await ready(pilot, app, lambda: app.snap is not None and app.snap.running == 1)
        app.prompt.value = f"/logs {job.short_id}"
        await pilot.press("enter")
        await settle(pilot, lambda: bool(app.query(LogsBlock)), what="logs block")
        logs = app.query_one(LogsBlock)
        await settle(pilot, lambda: logs.count >= 3, what="first lines")
        first = logs.count
        await settle(pilot, lambda: logs.count > first, what="more lines (live)")
        await pilot.press("escape")
        assert not logs.live
        assert "keeps running" in logs.foot_text
        assert daemon.job(job.id).state in (JobState.RUNNING, JobState.CHECKPOINTING)


async def test_watch_draws_a_live_loss_chart(daemon: InProcDaemon, project: Path) -> None:
    job = daemon.submit(project, LONG, name="train_yolo.py")
    app = shell()
    async with app.run_test(size=SIZE) as pilot:
        await ready(pilot, app, lambda: app.snap is not None and app.snap.running == 1)
        block = await type_line(pilot, app, f"/watch {job.short_id}")
        assert "✗" not in block.plain
        await settle(pilot, lambda: bool(app.query(WatchBlock)), what="watch block")
        watch = app.query_one(WatchBlock)
        await settle(pilot, lambda: len(watch.state.series.get("loss", [])) >= 5, what="points")
        n = len(watch.state.series["loss"])
        await settle(pilot, lambda: len(watch.state.series["loss"]) > n, what="chart grows")
        chart = str(watch.chart.render())
        assert watch.state.series["loss"][-1] < watch.state.series["loss"][0]
        assert chart  # rendered
        # the panel's sparkline comes from the same history
        assert "loss " in app.panel.plain
        assert "↓" in app.panel.plain
        # a new live view replaces the old one
        await type_line(pilot, app, f"/logs {job.short_id}")
        await settle(pilot, lambda: not watch.live, what="watch stops")
        assert sum(1 for b in app.query(LiveBlock) if b.live) == 1


# --------------------------------------------------------------------------- CLI-backed commands


async def test_commands_reuse_the_cli_renderers(daemon: InProcDaemon, project: Path) -> None:
    job = daemon.submit(project, LONG, name="train_yolo.py")
    app = shell()
    async with app.run_test(size=SIZE) as pilot:
        await ready(pilot, app, lambda: app.snap is not None and app.snap.running == 1)
        expect = {
            "/jobs": ["train_yolo.py", job.short_id],
            f"/status {job.short_id}": ["train_yolo.py", "timeline"],
            "/route train.py": ["→ fake", "candidates"],
            "/quota": ["provider", "fake"],
            "/providers": ["fake", "● up"],
            "/history": ["no past jobs"],
            "/help": ["/run", "/watch", "/doctor"],
            "/policy": ["agent jobs", "your jobs", "no jobs waiting for approval"],
            "/policy set agent.auto_max_hours 2": ["agent.auto_max_hours = 2"],
            "/policy reset": ["approval rules reset"],
            "/config": ["effective settings", "engine:", "backoff_base_s"],
            "/config path": ["config.yaml"],
            "/login fake": ["fake", "● up", "nothing to do"],
            # phase 8a: the doctor module; limited to checks that stay inside the tmp home
            "/doctor --only daemon --only provider.fake": [
                "running",
                "test mode",
                "fake live",
                "healthcheck ok",
            ],
            "gpu jobs --all": ["train_yolo.py"],  # a pasted CLI line
        }
        for line, fragments in expect.items():
            block = await type_line(pilot, app, line)
            for frag in fragments:
                assert frag in block.plain, (line, frag, block.plain)
            assert "internal error" not in block.plain, block.plain
        # CLI hints are rewritten into shell commands
        assert "gpu logs" not in transcript(app)


async def test_phase5_flags_reach_the_router_and_the_ledger(
    daemon: InProcDaemon, project: Path
) -> None:
    """/route --smoke and --hours show what the scoring router assumed; /quota --refresh
    re-reads live providers; /run --data merges a dataset (dry run: nothing submitted)."""
    (project / "data").mkdir()
    (project / "data" / "rows.csv").write_text("a,b\n1,2\n")
    app = shell()
    async with app.run_test(size=SIZE) as pilot:
        await ready(pilot, app)
        block = await type_line(pilot, app, "/route train.py --smoke")
        assert "smoke test" in block.plain, block.plain
        block = await type_line(pilot, app, "/route --hours 6 train.py")
        assert "6h runtime" in block.plain, block.plain
        block = await type_line(pilot, app, "/quota --refresh")
        assert "asking every live provider" in block.plain
        assert "fake" in block.plain, block.plain
        assert "live" in block.plain, block.plain
        block = await type_line(pilot, app, "/run --dry-run --data ds=./data train.py")
        assert "dry run: nothing submitted" in block.plain, block.plain
        block = await type_line(pilot, app, "/run --dry-run --data ds=./nope train.py")
        assert "does not exist" in block.plain, block.plain
        with daemon.client() as c:
            assert c.jobs().jobs == []


async def test_cancel_and_fetch(daemon: InProcDaemon, project: Path) -> None:
    long = daemon.submit(project, LONG, name="train_yolo.py")
    short = daemon.submit(project, {"duration": 0.3, "steps": 2}, name="prep.py")
    app = shell()
    async with app.run_test(size=SIZE) as pilot:
        await ready(pilot, app)
        await settle(pilot, lambda: daemon.job(short.id).state is JobState.DONE, what="done")
        block = await type_line(pilot, app, f"/cancel {long.short_id}")
        assert f"job {long.short_id} cancel" in block.plain
        await settle(
            pilot, lambda: daemon.job(long.id).state is JobState.CANCELLED, what="cancelled"
        )
        block = await type_line(pilot, app, f"/fetch {short.short_id} --dest out")
        assert "files →" in block.plain
        assert (project / "out").is_dir()


async def test_a_finished_job_is_announced(daemon: InProcDaemon, project: Path) -> None:
    job = daemon.submit(project, {"duration": 1.5, "steps": 3}, name="prep.py")
    app = shell()
    async with app.run_test(size=SIZE) as pilot:
        await ready(pilot, app, lambda: app.snap is not None and len(app.snap.active) == 1)
        await settle(
            pilot, lambda: f"job {job.short_id} prep.py done" in transcript(app), what="done notice"
        )
        assert f"runs/{job.id[:4]}" in transcript(app)


# --------------------------------------------------------------------------- leaving


async def test_ctrl_d_quits_and_jobs_keep_running(daemon: InProcDaemon, project: Path) -> None:
    job = daemon.submit(project, LONG, name="train_yolo.py")
    app = shell()
    async with app.run_test(size=SIZE) as pilot:
        await ready(pilot, app)
        await pilot.press("ctrl+d")
        await pilot.pause(0.2)
    assert app.return_value == 0
    assert app.feed.stopped
    assert daemon.job(job.id).state in (
        JobState.RUNNING,
        JobState.CHECKPOINTING,
        JobState.PROVISIONING,
    )


async def test_ctrl_c_twice_quits(daemon: InProcDaemon, project: Path) -> None:
    app = shell()
    async with app.run_test(size=SIZE) as pilot:
        await ready(pilot, app)
        await pilot.press("ctrl+c")
        assert "press ctrl+c again" in transcript(app)
        await pilot.press("ctrl+c")
        await pilot.pause(0.2)
    assert app.return_value == 0


@pytest.mark.parametrize("line", ["/exit", "quit"])
async def test_exit_commands(daemon: InProcDaemon, project: Path, line: str) -> None:
    app = shell()
    async with app.run_test(size=SIZE) as pilot:
        await ready(pilot, app)
        app.prompt.value = line
        await pilot.press("enter")
        await pilot.pause(0.2)
    assert app.return_value == 0


# --------------------------------------------------------------------------- auto-start


async def test_daemon_down_is_started_like_the_cli_does(
    no_daemon: Path, project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No daemon + auto-start on: the shell spawns one (private home, port 0, test mode),
    says so, and connects; the test shuts that daemon down again."""
    from gpu_router.client import GpuClient
    from gpu_router.paths import Paths

    monkeypatch.delenv("GPU_ROUTER_NO_AUTOSTART", raising=False)
    monkeypatch.setenv("GPU_ROUTER_PORT", "0")
    app = shell()
    try:
        async with app.run_test(size=SIZE) as pilot:
            await settle(
                pilot,
                lambda: app.snap is not None and app.snap.conn.status == "up",
                timeout_s=40,
                what="auto-started daemon",
            )
            await settle(
                pilot,
                lambda: "daemon started in the background" in transcript(app),
                what="start notice",
            )
            await settle(pilot, lambda: "no jobs running" in app.panel.plain, what="empty")
    finally:
        try:
            with GpuClient.from_env(Paths(no_daemon), timeout_s=5) as c:
                c.shutdown()
        except Exception:
            pass


async def test_config_edit_without_a_terminal_says_so(
    daemon: InProcDaemon, project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EDITOR", "true")
    app = shell()
    async with app.run_test(size=SIZE) as pilot:
        await ready(pilot, app)
        block = await type_line(pilot, app, "/config edit")
        assert "could not open 'true' here" in block.plain  # headless: nothing to suspend to
        assert "config.yaml" in block.plain


async def test_status_overview_login_list_and_fetch(daemon: InProcDaemon, project: Path) -> None:
    short = daemon.submit(project, {"duration": 0.3, "steps": 2}, name="prep.py")
    app = shell()
    async with app.run_test(size=SIZE) as pilot:
        await ready(pilot, app)
        await settle(pilot, lambda: daemon.job(short.id).state is JobState.DONE, what="done")
        block = await type_line(pilot, app, "/status")
        assert "finished recently" in block.plain
        assert "prep.py" in block.plain
        block = await type_line(pilot, app, "/login")
        assert "fake" in block.plain
        assert "/login <provider>" in block.plain
        block = await type_line(pilot, app, f"/fetch {short.short_id}")
        assert "files →" in block.plain
        block = await type_line(pilot, app, "/login nope")
        assert "no provider named 'nope'" in block.plain


def test_feed_reads_helper_metrics_through_the_protocol_log(
    daemon: InProcDaemon, project: Path
) -> None:
    """GpuClient.logs(protocol=True) returns the `::gpu::` lines, so the metric history
    is the helper's, not the stdout fallback's (D42)."""
    from gpu_router.shell.feed import read_new_lines
    from gpu_router.shell.metrics import MetricHistory
    from tests.shell.conftest import wait_for

    job = daemon.submit(project, {"duration": 0.8, "steps": 8}, name="train.py")
    wait_for(lambda: daemon.job(job.id).state is JobState.DONE, what="job done")
    hist = MetricHistory()
    with daemon.client() as client:
        plain = list(client.logs(job.id, attempt=1))
        read_new_lines(client, job.id, 1, hist)
        again = read_new_lines(client, job.id, 1, hist)  # nothing new: no duplicates
    assert not any((r.line or "").startswith("::gpu::") for r in plain)
    assert hist.helper
    assert hist.total == 8
    assert len(hist.values("loss")) == 8
    assert not again
