"""Phase 4/5 review regressions for the shell (D44): what Enter takes from the popup,
bounded /logs, quitting while a command waits, notices (double finish, daemon started,
outages), /jobs paging, transcript growth, scrolling a log, the quota thread, metric
thinning, completion speed, history drafts and one visual language."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from rich.text import Text

from gpu_router import protocol
from gpu_router.models import JobState
from gpu_router.shell import app as app_mod
from gpu_router.shell import commands as cmds
from gpu_router.shell.app import GpuShell
from gpu_router.shell.complete import Known, _scripts, complete
from gpu_router.shell.feed import FIRST_READ_LINES, Feed
from gpu_router.shell.metrics import MAX_POINTS, MetricHistory
from gpu_router.shell.panel import job_line2, notice_line, panel_lines
from gpu_router.shell.state import Conn, ConnStatus, Snapshot
from gpu_router.shell.widgets import (
    TAIL_LINES,
    WRITE_CHUNK,
    CommandBlock,
    LiveBlock,
    LogsBlock,
)
from tests.shell.conftest import InProcDaemon, settle
from tests.shell.helpers import NOW, job
from tests.shell.test_shell_app import LONG, SIZE, ready, shell, transcript, type_line

# --------------------------------------------------------------------------- Enter + popup


async def test_tab_then_enter_never_runs_on_a_candidate_nobody_picked(
    daemon: InProcDaemon, project: Path
) -> None:
    """Finding: `/can` + tab + enter cancelled the first job in the popup; the same keys
    denied or approved the first waiting job, and `/status ` showed a job, not the
    overview."""
    running = daemon.submit(project, LONG, name="train_yolo.py")
    waiting = daemon.submit(project, {"duration": 0.5}, name="eval.py", requires_approval=True)
    app = shell()
    async with app.run_test(size=SIZE) as pilot:
        await ready(pilot, app, lambda: app.snap is not None and len(app.snap.active) == 2)
        for typed, command in (("can", "/cancel "), ("den", "/deny "), ("appr", "/approve ")):
            await pilot.press("slash", *typed, "tab")
            assert app.prompt.value == command
            assert app.popup.display  # the job list is showing, row 0 highlighted
            await pilot.press("enter")
            await settle(
                pilot,
                lambda c=command: any(
                    b.line == c.strip() and not b.running for b in app.query(CommandBlock)
                ),
                what=f"{command!r} as typed",
            )
            assert "needs a job id" in transcript(app)
        assert daemon.job(running.id).state in (JobState.RUNNING, JobState.CHECKPOINTING)
        assert daemon.job(waiting.id).state is JobState.AWAITING_APPROVAL
        # /status + space + enter is the overview
        await pilot.press("slash", *"status", "space")
        assert app.popup.display
        await pilot.press("enter")
        await settle(
            pilot,
            lambda: any(b.line == "/status" and not b.running for b in app.query(CommandBlock)),
            what="/status",
        )
        # moving the highlight is a choice: enter then takes the job it points at
        await pilot.press("slash", *"status", "space", "down", "up")
        assert app.popup.moved
        await pilot.press("enter")
        await settle(
            pilot,
            lambda: any(
                (b.line or "").startswith("/status ") and not b.running
                for b in app.query(CommandBlock)
            ),
            what="/status <picked id>",
        )


async def test_policy_set_keeps_editing_until_it_has_a_value(
    daemon: InProcDaemon, project: Path
) -> None:
    """Finding: enter on `set` or on a rule key ran `/policy set` without its arguments
    and cleared the prompt."""
    app = shell()
    async with app.run_test(size=SIZE) as pilot:
        await ready(pilot, app)
        before = len(list(app.query(CommandBlock)))
        await pilot.press("slash", *"policy", "space", "s", "e")
        await pilot.press("enter")
        assert app.prompt.value == "/policy set "
        await pilot.press(*"agent.auto_m")
        await pilot.press("enter")
        assert app.prompt.value == "/policy set agent.auto_max_hours "
        assert len(list(app.query(CommandBlock))) == before  # nothing ran
        await pilot.press("2", "enter")
        await settle(
            pilot,
            lambda: "agent.auto_max_hours = 2" in transcript(app),
            what="/policy set ran with its value",
        )


async def test_history_navigation_keeps_the_line_being_typed(
    daemon: InProcDaemon, project: Path
) -> None:
    app = shell()
    async with app.run_test(size=SIZE) as pilot:
        await ready(pilot, app)
        await type_line(pilot, app, "/help")
        draft = "/run train.py --hours 6 --vram 24"
        app.prompt.value = draft
        app.popup.hide()
        await pilot.press("up")
        assert app.prompt.value == "/help"
        await pilot.press("down")
        assert app.prompt.value == draft


# --------------------------------------------------------------------------- bounded /logs


@dataclass
class _LogClient:
    """Enough of GpuClient for LogsBlock.loop on a job that printed `lines` lines."""

    view: Any
    lines: int
    offsets: list[int] = field(default_factory=list)

    def job(self, _ref: str) -> Any:
        att = SimpleNamespace(n=1, log_lines=self.lines, provider="fake")
        return SimpleNamespace(job=self.view, attempts=[att])

    def events(self, _ref: str, after: int = 0) -> Any:
        return SimpleNamespace(next=0, events=[])

    def logs(self, _ref: str, attempt: int, offset: int = 0, protocol: bool = False) -> Any:
        self.offsets.append(offset)
        for i in range(offset, self.lines):
            yield SimpleNamespace(line=f"line {i}", offset=i)


def test_logs_of_a_huge_job_start_at_the_tail_and_write_in_chunks() -> None:
    """Finding: /logs read every line from offset 0 and wrote them all in one UI call
    (100k lines = a 26 s freeze esc could not stop)."""
    done = job("a7f2", "train.py", JobState.DONE, finished_at=NOW)
    client = _LogClient(done, 100_000)
    block = LogsBlock(done, lambda: client)  # type: ignore[arg-type,return-value]
    posted: list[tuple[str, int]] = []

    def post(fn: Any, *args: Any) -> bool:
        if fn.__name__ == "write":
            posted.append(("write", len(args[1])))
            if len(posted) == 1:
                assert "earlier lines of attempt 1 not shown" in args[1][0].plain
        return True

    block.post = post  # type: ignore[method-assign]
    block.loop(client)  # type: ignore[arg-type]
    assert client.offsets == [100_000 - TAIL_LINES]  # the rest is never read
    sizes = [n for kind, n in posted if kind == "write"]
    assert max(sizes) <= WRITE_CHUNK
    assert sum(sizes) == TAIL_LINES + 1  # the kept lines + the "not shown" line


def test_the_feed_reads_only_the_tail_of_a_job_it_sees_first() -> None:
    running = job("a7f2", "train.py", JobState.RUNNING)
    client = _LogClient(running, 200_000)
    feed = Feed(paths=None, publish=lambda s: None)  # type: ignore[arg-type]
    feed._update_metrics(client, running)  # type: ignore[arg-type]
    assert client.offsets == [200_000 - FIRST_READ_LINES]


# --------------------------------------------------------------------------- quitting


async def test_quitting_does_not_wait_for_a_command_still_running(
    daemon: InProcDaemon, project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Finding: commands ran as Textual thread workers, and App.run() waited for them:
    ctrl+d during a slow /fetch left the process hanging for up to 10 minutes."""
    release = threading.Event()
    started = threading.Event()

    def slow(ctx: Any, args: list[str]) -> None:
        started.set()
        release.wait(60)

    monkeypatch.setitem(cmds.HANDLERS, "fetch", slow)
    app = shell()
    t0 = 0.0
    try:
        async with app.run_test(size=SIZE) as pilot:
            await ready(pilot, app)
            app.prompt.value = "/fetch a7f2"
            await pilot.press("enter")
            await settle(pilot, started.is_set, what="the command to start")
            assert app.busy()
            t0 = time.monotonic()
            await pilot.press("ctrl+d")
        assert time.monotonic() - t0 < 5
        assert app.return_value == 0
    finally:
        release.set()


def test_run_exits_hard_when_a_command_is_still_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = threading.Event()
    exits: list[int] = []

    def fake_run(self: GpuShell) -> int:
        t = threading.Thread(target=release.wait, args=(30,), daemon=True)
        self._command_threads.add(t)
        t.start()
        return 0

    monkeypatch.setattr(GpuShell, "run", fake_run)
    monkeypatch.setattr(app_mod.os, "_exit", lambda code: exits.append(code))
    try:
        assert app_mod.run() == 0
        assert exits == [0]
    finally:
        release.set()


# --------------------------------------------------------------------------- notices


class _Followed(LiveBlock):
    """A live block for a job without a poll thread (notices only look at its flags)."""

    def on_mount(self) -> None:
        self.update_head(self.job)


def _up(active: tuple[Any, ...] = (), recent: tuple[Any, ...] = (), pid: int | None = None) -> Any:
    return Snapshot(
        conn=Conn(ConnStatus.UP, last_ok=NOW, started_pid=pid),
        active=active,
        recent=recent,
        taken_at=NOW,
    )


DOWN = Snapshot(conn=Conn(ConnStatus.DOWN, message="down", retry_s=3), taken_at=NOW)


def _notes(app: GpuShell) -> list[str]:
    return [b.plain for b in app.query(CommandBlock) if b.line is None]


async def test_a_finish_the_live_block_showed_gets_no_second_notice(no_daemon: Path) -> None:
    """Finding: a /logs block that had already shown '✓ done' stopped counting as live,
    so the feed's next snapshot added a second finish notice (and /watch always did)."""
    app = shell()
    async with app.run_test(size=SIZE) as pilot:
        app.feed.stop()
        await pilot.pause(0.1)
        run = job("75a4", "train.py", JobState.RUNNING)
        done = job("75a4", "train.py", JobState.DONE, finished_at=NOW, outputs_dir="/p/runs/75a4")
        block = _Followed(run, lambda: None)  # type: ignore[arg-type,return-value]
        await app.transcript.mount(block)
        block.finish(done, [Text("✓ train done")])  # the block saw the end itself
        app._apply(_up(active=(run,)))
        app._apply(_up(recent=(done,)))
        assert not any("75a4 train.py done" in n for n in _notes(app))
        # stopped by the user before the end: the notice is the only word on it
        other = job("b3d1", "prep.py", JobState.RUNNING)
        finished = job("b3d1", "prep.py", JobState.DONE, finished_at=NOW)
        stopped = _Followed(other, lambda: None)  # type: ignore[arg-type,return-value]
        await app.transcript.mount(stopped)
        stopped.stop()
        app._apply(_up(active=(other,)))
        app._apply(_up(recent=(finished,)))
        assert any("b3d1 prep.py done" in n for n in _notes(app))


async def test_daemon_started_is_announced_once(no_daemon: Path) -> None:
    """Finding: every DOWN -> UP repeated 'daemon started (pid 4242)'."""
    app = shell()
    async with app.run_test(size=SIZE) as pilot:
        app.feed.stop()
        await pilot.pause(0.1)
        for snap in (_up(pid=4242), DOWN, _up(pid=4242), DOWN, _up(pid=4242)):
            app._apply(snap)
        started = [n for n in _notes(app) if "daemon started in the background" in n]
        assert len(started) == 1


async def test_changes_during_an_outage_are_announced(no_daemon: Path) -> None:
    """Finding: a DOWN snapshot in between reset the baseline, so jobs that finished or
    started waiting for approval while the daemon restarted were never announced."""
    app = shell()
    async with app.run_test(size=SIZE) as pilot:
        app.feed.stop()
        await pilot.pause(0.1)
        queued = job("c19e", "eval.py", JobState.QUEUED)
        running = job("a7f2", "train.py", JobState.RUNNING)
        waiting = job("c19e", "eval.py", JobState.AWAITING_APPROVAL)
        done = job("a7f2", "train.py", JobState.DONE, finished_at=NOW)
        app._apply(_up(active=(queued, running)))
        app._apply(DOWN)
        app._apply(_up(active=(waiting,), recent=(done,)))
        text = "\n".join(_notes(app))
        assert "job c19e eval.py needs approval" in text
        assert "job a7f2 train.py done" in text


# --------------------------------------------------------------------------- /jobs paging


async def test_jobs_paging_hint_works_in_the_shell(daemon: InProcDaemon, project: Path) -> None:
    """Finding: the shell's own copy of the paging hint suggested
    `/jobs --all --before <ts>`, which its parser rejected, and hard-coded -n 200."""
    for i in range(3):
        daemon.submit(project, {"duration": 0.2, "steps": 1}, name=f"sweep_{i}.py")
    app = shell()
    async with app.run_test(size=SIZE) as pilot:
        await ready(pilot, app)
        block = await type_line(pilot, app, "/jobs --all -n 2")
        hint = next(line for line in block.plain.splitlines() if "… more" in line)
        assert "/jobs --all -n 4" in hint
        assert "gpu jobs" not in hint
        before = hint.split("--before ")[1].split()[0]
        page2 = await type_line(pilot, app, f"/jobs --all --before {before}")
        assert "unrecognized arguments" not in page2.plain
        assert "sweep_0.py" in page2.plain
        await settle(pilot, lambda: app.snap is not None and not app.snap.active, what="all done")
        here = await type_line(pilot, app, "/jobs --here")  # the CLI's empty-state words
        assert "no running or queued jobs from this project" in here.plain


# --------------------------------------------------------------------------- transcript


async def test_the_transcript_is_bounded(no_daemon: Path) -> None:
    """Finding: nothing ever left the transcript; each tick re-laid out every block."""
    app = shell()
    async with app.run_test(size=SIZE) as pilot:
        app.feed.stop()
        await pilot.pause(0.1)
        logs = []
        for i in range(5):
            view = job(f"{i:04d}", f"j{i}.py", JobState.DONE)
            block = LogsBlock(view, lambda: None)  # type: ignore[arg-type,return-value]
            block.stop_event.set()
            await app.transcript.mount(block)
            block.write(view, [Text(f"line {n}") for n in range(50)])
            logs.append(block)
        for i in range(app_mod.MAX_BLOCKS + 20):
            app._note([Text(f"note {i}")])
        await pilot.pause(0.2)
        assert len(app.transcript.children) <= app_mod.MAX_BLOCKS
        kept = [b for b in app.query(LogsBlock) if not b.forgotten]
        assert len(kept) <= app_mod.KEEP_LOG_OUTPUT


async def test_pageup_scrolls_the_newest_log_first(no_daemon: Path) -> None:
    """Finding: pgup/pgdn scrolled only the transcript, so a keyboard user could not read
    the start of a traceback longer than the /logs block."""
    app = shell()
    async with app.run_test(size=SIZE) as pilot:
        app.feed.stop()
        await pilot.pause(0.1)
        view = job("e5a0", "bad.py", JobState.FAILED)
        block = LogsBlock(view, lambda: None)  # type: ignore[arg-type,return-value]
        block.stop_event.set()
        block.max_rows = 10
        await app.transcript.mount(block)
        block.write(view, [Text(f"Traceback line {n}") for n in range(43)])
        out = block.output
        # fixed pauses read too early on a slow CI runner (2026-10-09: "33 < 33"), so wait
        await settle(
            pilot,
            lambda: out.max_scroll_y > 0 and out.scroll_y == out.max_scroll_y,
            timeout_s=5,
            what="the log to render and follow its end",
        )
        top = out.scroll_y
        calls: list[tuple[int, bool]] = []
        real_scroll_page = block.scroll_page

        def spy(direction: int) -> bool:
            calls.append((direction, real_scroll_page(direction)))
            return calls[-1][1]

        block.scroll_page = spy  # type: ignore[method-assign]
        await pilot.press("pageup")
        try:
            await settle(pilot, lambda: out.scroll_y < top, timeout_s=5, what="pgup to scroll")
        except AssertionError as exc:
            state = (
                f"scroll_page calls {calls}, y {out.scroll_y} of {out.max_scroll_y}, "
                f"height {out.size.height}, focused {app.focused!r}, "
                f"{len(app.query(LogsBlock))} logs blocks"
            )
            raise AssertionError(f"{exc}: {state}") from None
        from gpu_router.shell.widgets import _plain

        words = " ".join(_plain(cmds.help_renderable()).split())
        assert "pgup/pgdn scroll (the newest /logs output first)" in words


# --------------------------------------------------------------------------- feed quota thread


def test_the_quota_thread_is_restarted_whenever_it_ends(
    daemon: InProcDaemon, project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Finding: the quota thread ended on DaemonUnavailable and was restarted only after
    the status loop itself reconnected; otherwise live quota never refreshed again."""
    starts: list[float] = []

    def quick(self: Feed) -> None:
        starts.append(time.monotonic())  # ends at once, like a quota call that failed

    monkeypatch.setattr(Feed, "_quota_loop", quick)
    feed = Feed(paths=daemon.paths, publish=lambda s: None, poll_s=0.05, autostart=False)
    thread = threading.Thread(target=feed.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while len(starts) < 3 and time.monotonic() < deadline:
        time.sleep(0.05)
    feed.stop()
    thread.join(5)
    assert len(starts) >= 3


# --------------------------------------------------------------------------- metrics, completion


def test_thinning_keeps_the_first_point() -> None:
    """Finding: `del points[::2]` dropped index 0 at every thinning, so the chart's start
    drifted to step 2**k and the first loss left the min/max title."""
    h = MetricHistory()
    for i in range(MAX_POINTS * 5):
        h.feed(protocol.metric(i, {"loss": 10.0 - i / 1000}))
    assert h.series["loss"][0] == (0, 10.0)
    assert len(h.series["loss"]) <= MAX_POINTS


def test_script_completion_stays_fast_in_a_huge_folder(tmp_path: Path) -> None:
    """Finding: completion listed, stat'ed and sorted the whole directory on every
    keystroke (50k files: 0.3-0.4 s per key)."""
    folder = tmp_path / "data"
    folder.mkdir()
    for i in range(30_000):
        (folder / f"img_{i:05d}.png").touch()
    (folder / "train.py").write_text("x")
    (folder / "tools").mkdir()
    t0 = time.perf_counter()
    got = complete("/run t", Known(cwd=folder))
    took = time.perf_counter() - t0
    assert got is not None
    assert [c.value for c in got.candidates] == ["tools/", "train.py"]
    assert took < 0.2, took
    t0 = time.perf_counter()
    everything = _scripts("", folder)
    assert time.perf_counter() - t0 < 0.2
    assert [c.value for c in everything][:2] == ["tools/", "train.py"]


# --------------------------------------------------------------------------- visual language


def test_one_word_for_approval_and_slash_syntax_in_daemon_text() -> None:
    """Finding: 'waiting for your approval' (panel), 'needs approval' (tables) and
    'awaiting approval' (live heads) for one state, and daemon text with `gpu x` next to
    the shell's /x hints."""
    waiting = job(
        "c19e",
        "eval.py",
        JobState.AWAITING_APPROVAL,
        provider="kaggle",
        route_reason="kaggle: fits 16GB, colab needs login (run `gpu login colab`)",
        approval_reason="over the 1h auto limit · kaggle T4",
    )
    panel = [t.plain for t in panel_lines(_up(active=(waiting,)), 110, 12, now=NOW, example="t")]
    assert "needs approval" in panel[0]
    assert "gpu login" not in panel[1]
    assert "/login colab" in panel[1]
    failed = job(
        "a7f2",
        "train.py",
        JobState.FAILED,
        message="script exited with 1 on kaggle; see `gpu logs a7f2`",
    )
    line = notice_line(failed, NOW).plain
    assert "gpu logs" not in line
    assert line.count("/logs a7f2") == 1
    moving = job("b3d1", "prep.py", JobState.QUEUED, message="retry: run `gpu status b3d1`")
    l2 = job_line2(moving, None, None, 110, NOW)
    assert l2 is not None
    assert "/status b3d1" in l2.plain


async def test_live_heads_use_the_same_state_words(no_daemon: Path) -> None:
    app = shell()
    async with app.run_test(size=SIZE) as pilot:
        app.feed.stop()
        await pilot.pause(0.1)
        view = job("c19e", "eval.py", JobState.AWAITING_APPROVAL)
        block = _Followed(view, lambda: None)  # type: ignore[arg-type,return-value]
        await app.transcript.mount(block)
        await pilot.pause(0.1)
        assert "needs approval" in str(block.head.render())
        assert "awaiting approval" not in str(block.head.render())
