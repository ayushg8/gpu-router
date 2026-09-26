"""Regressions from driving the installed `gpu` shell in a real pseudo-terminal (pexpect +
pyte, 120x36 and 80x24, resizes): /help was unreadable below ~110 columns and ate its own
optional arguments, the /run usage taught a flag order that sends `--smoke` to the script,
and the transcript stopped following new output once the popup opened (or the terminal was
resized) over a full screen."""

from __future__ import annotations

import io
import re
from pathlib import Path

import pytest
from rich.console import Console
from rich.text import Text

from gpu_router.cli.app import (
    RUN_TAIL_FLAG_OPTS,
    RUN_TAIL_VALUE_OPTS,
    split_run_argv,
)
from gpu_router.shell import commands as cmds
from gpu_router.shell.app import GpuShell
from gpu_router.shell.widgets import CommandBlock
from tests.shell.conftest import InProcDaemon, settle
from tests.shell.test_shell_app import ready, shell, type_line

# --------------------------------------------------------------------------- /help


def _render(renderable: object, width: int) -> str:
    console = Console(width=width, record=True, color_system=None, file=io.StringIO())
    console.print(renderable)
    return console.export_text()


@pytest.mark.parametrize("width", [120, 100, 80, 64, 63, 56, 40])
def test_help_shows_every_usage_and_summary_at_any_width(width: int) -> None:
    """At 80 columns the usage column (sized to the longest usage, never wrapped) left the
    summaries no room at all, and `[id]` / `[reason]` / `[edit|path]` were parsed as rich
    markup and vanished."""
    text = _render(cmds.help_renderable(), width)
    lines = text.splitlines()
    assert all(len(line.rstrip()) <= width for line in lines)
    assert "…" not in text  # nothing cut
    tokens = set(text.split())
    for c in cmds.COMMANDS:
        for word in f"/{c.name} {c.usage}".split():
            assert word in tokens, (width, c.name, word)
        for word in c.summary.split():
            assert word in tokens, (width, c.name, word)
    for eaten in ("[id]", "[reason]", "[edit|path]", "[command]", "[metric]"):
        assert eaten in tokens


def test_help_is_laid_out_again_for_the_width_it_is_drawn_at() -> None:
    """One renderable, drawn after a resize: two columns when wide, stacked when narrow."""
    help_list = cmds.HelpList()
    wide = _render(help_list, 120).splitlines()
    narrow = _render(help_list, 50).splitlines()
    status_wide = next(line for line in wide if line.startswith("/status [id]"))
    assert "one job in detail" in status_wide  # same row
    i = next(n for n, line in enumerate(narrow) if line.startswith("/status [id]"))
    assert "one job in detail" not in narrow[i]
    assert narrow[i + 1].startswith("  one job in detail")  # indented under it


# --------------------------------------------------------------------------- /run usage


def test_run_usage_lists_only_tail_options_after_the_script() -> None:
    """`/run <script> ... [--smoke] [--data ...]` (the old usage) put --smoke after the
    script, where split_run_argv (D23) hands it to the script: typing /help's own order ran
    `train.py --smoke` on a remote GPU instead of a smoke test on this Mac."""
    usage = cmds.BY_NAME["run"].usage
    before, _, after = usage.partition("<script>")
    assert after, usage
    tail_flags = re.findall(r"(?<![\w-])(--?[a-z][\w-]*)", after)
    assert tail_flags
    assert set(tail_flags) <= RUN_TAIL_VALUE_OPTS | RUN_TAIL_FLAG_OPTS
    assert "--smoke" in before
    assert "--data" in before
    # typed exactly in the usage's order, nothing goes to the script
    argv = ["--smoke", "--data", "d", "train.py", "--vram", "8", "--hours", "1", "-p", "local"]
    click_args, script_args, clashes = split_run_argv(argv, frozenset({"T4"}))
    assert script_args == []
    assert clashes == []
    assert "train.py" in click_args
    assert "--smoke" in click_args


# --------------------------------------------------------------------------- following output


def _at_end(app: GpuShell) -> bool:
    t = app.transcript
    return t.max_scroll_y > 0 and t.scroll_y >= t.max_scroll_y - 1


async def _fill(pilot: object, app: GpuShell) -> None:
    await type_line(pilot, app, "/help")
    await type_line(pilot, app, "/help")
    await settle(pilot, lambda: _at_end(app), what="a transcript taller than the screen")


async def test_output_is_followed_after_the_popup_opened_over_a_full_transcript(
    daemon: InProcDaemon, project: Path
) -> None:
    """The popup takes rows from the transcript; the old "was it at the end?" check then
    said no, the new block was not scrolled to, and no later output was ever followed."""
    app = shell()
    async with app.run_test(size=(80, 24)) as pilot:
        await ready(pilot, app)
        await _fill(pilot, app)
        await pilot.press("slash")  # every command: the popup takes 9 rows
        await settle(pilot, lambda: bool(app.popup.display), what="the popup")
        await pilot.pause(0.2)
        assert _at_end(app)  # the newest output stays right above the prompt
        await pilot.press("down", "down", "enter")  # picks and runs /jobs
        await settle(
            pilot,
            lambda: any(b.line == "/jobs" and not b.running for b in app.query(CommandBlock)),
            what="/jobs from the popup",
        )
        await settle(pilot, lambda: _at_end(app), what="the /jobs output in view")
        await pilot.pause(0.3)
        app._note([Text("a notice")])
        await settle(pilot, lambda: _at_end(app), what="the notice in view")


async def test_a_resize_keeps_the_newest_output_in_view(
    daemon: InProcDaemon, project: Path
) -> None:
    app = shell()
    async with app.run_test(size=(80, 24)) as pilot:
        await ready(pilot, app)
        await _fill(pilot, app)
        for size in ((120, 36), (60, 20), (80, 24)):
            await pilot.resize_terminal(*size)
            await pilot.pause(0.3)
            await settle(pilot, lambda: _at_end(app), what=f"the end in view at {size}")


async def test_scrolling_back_is_left_alone_until_a_command_runs(
    daemon: InProcDaemon, project: Path
) -> None:
    app = shell()
    async with app.run_test(size=(80, 24)) as pilot:
        await ready(pilot, app)
        await _fill(pilot, app)
        await pilot.press("pageup")
        await pilot.pause(0.2)
        top = app.transcript.scroll_y
        assert not _at_end(app)
        app._note([Text("a notice while reading")])
        await pilot.pause(0.3)
        assert app.transcript.scroll_y == top  # the reader keeps their place
        await pilot.press("pagedown", "pagedown", "pagedown")
        await pilot.pause(0.2)
        assert _at_end(app)
        app._note([Text("followed again")])
        await settle(pilot, lambda: _at_end(app), what="following again at the end")
        await pilot.pause(0.3)
        await pilot.press("pageup")
        await pilot.pause(0.3)
        assert not _at_end(app)
        await type_line(pilot, app, "/help jobs")  # a command the user ran: show it
        await settle(pilot, lambda: _at_end(app), what="the command's output")


# --------------------------------------------------------------------------- /logs width


async def test_logs_lines_fit_the_view_at_80_columns_and_after_a_resize(no_daemon: Path) -> None:
    """RichLog wrapped every line for at least 78 cells (its min_width), wider than the view
    in an 80-column terminal: the right edge was cut and a horizontal scrollbar covered the
    newest line. Lines also kept their first wrap after a resize."""
    from gpu_router.models import JobState
    from gpu_router.shell.widgets import LogsBlock
    from tests.shell.helpers import job

    app = shell()
    async with app.run_test(size=(80, 24)) as pilot:
        app.feed.stop()
        await pilot.pause(0.1)
        view = job("a5af", "hello.py", JobState.DONE)
        block = LogsBlock(view, lambda: None)  # type: ignore[arg-type,return-value]
        block.stop_event.set()
        await app.transcript.mount(block)
        await pilot.pause(0.2)
        path = "/private/tmp/" + "-".join(f"segment{n:02d}" for n in range(14)) + "/runs/a5af"
        block.write(view, [Text("outputs saved to"), Text(path), Text("the last line")])
        await pilot.pause(0.3)
        out = block.output

        def fits() -> bool:
            view_w = out.scrollable_content_region.width
            return 0 < out.virtual_size.width <= view_w and out.max_scroll_x == 0

        assert fits()
        for size in ((60, 20), (100, 30)):
            await pilot.resize_terminal(*size)
            await pilot.pause(LogsBlock.REWRAP_DELAY_S + 0.3)
            assert fits(), size
            assert block.wrap_width == out.scrollable_content_region.width
        assert list(block.kept)[-1].plain == "the last line"
