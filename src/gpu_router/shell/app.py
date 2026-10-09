"""GpuShell: the interactive Textual shell (phase 4), opened by bare `gpu` in a terminal.

Layout, top to bottom (spec "Interactive shell" mockup):

    ╭─ gpu-router ──────────────────────────────╮   JobPanel: live jobs, 2 rows each
    │ ⚡ job a7f2  train_yolo.py  kaggle · 2xT4 01:42:10 │
    │    step 45/100   loss 0.412 ▇▆▅▄▃▂ ↓   ckpt 3m ago │
    ╰───────────────────────────────────────────╯
    > /jobs                                         transcript: one block per command,
    ...                                             live /logs and /watch blocks
    > _                                             prompt, then the / popup under it
                                                    (spacer)
    ──────────────────────────────────────────────
    kaggle 22/30h ↻Sat │ colab ● up │ 1 running    StatusFooter

The transcript is only as tall as its content (up to the space left), so the prompt sits
right under the panel on an empty screen, like the mockup, and moves down as output
arrives. Direction spec: operator console / live job monitor archetype, dense (one row
per fact), colour only for job state, 0 ms motion; states: connecting, starting,
daemon down (retrying), empty (quota left + example /run), too many jobs (+N more).
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import threading
from collections.abc import Callable
from pathlib import Path

from rich.console import RenderableType
from rich.terminal_theme import TerminalTheme
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.theme import Theme
from textual.widget import Widget
from textual.widgets import Input, Label, Rule

from gpu_router.api import JobView
from gpu_router.cli import render
from gpu_router.client import GpuClient
from gpu_router.clock import SystemClock
from gpu_router.paths import Paths
from gpu_router.shell import commands as cmds
from gpu_router.shell.complete import Candidate, Completion, Known, complete
from gpu_router.shell.feed import Feed
from gpu_router.shell.panel import notice_line, seg, shell_words, sort_active
from gpu_router.shell.state import ConnStatus, Snapshot
from gpu_router.shell.widgets import (
    CommandBlock,
    CommandPopup,
    JobPanel,
    LiveBlock,
    LogsBlock,
    Prompt,
    StatusFooter,
    WatchBlock,
)
from gpu_router.statemachine import JobState, is_terminal

_clock = SystemClock()
HISTORY_FILE = "shell_history"
HISTORY_MAX = 500
MAX_BLOCKS = 300  # transcript blocks kept; older ones are dropped (a shell open for days)
KEEP_LOG_OUTPUT = 3  # stopped /logs blocks that keep their rendered lines
#: commands whose argument Enter never takes from a default popup highlight: a job id the
#: user did not pick must not be cancelled, denied or approved (spending quota)
NO_DEFAULT_PICK = frozenset({"cancel", "deny", "approve"})

# Tokens. Neutral greys carry all chrome (focus, highlight, borders); the only hues are the
# state colours, shared with the CLI's visual language: green running/done, yellow
# waiting, red failed. The ANSI theme maps rich's named colours onto the same values.
BG = "#121212"
FG = "#d4d4d4"
MUTED = "#8a8a8a"
BORDER = "#3a3a3a"
BOOST = "#2e2e2e"
CURSOR = "#333333"  # the highlighted popup row
GREEN = "#6cc07a"
YELLOW = "#e0b55e"
RED = "#e06c6c"

THEME = Theme(
    name="gpu-router",
    primary=MUTED,
    secondary=MUTED,
    accent=MUTED,
    warning=YELLOW,
    error=RED,
    success=GREEN,
    foreground=FG,
    background=BG,
    surface=BG,
    panel=BOOST,
    boost=BOOST,
    dark=True,
    variables={
        "panel-border": BORDER,
        "popup-cursor": CURSOR,
        "text-muted": MUTED,
        "block-cursor-background": BOOST,
        "block-cursor-blurred-background": BOOST,
        "block-cursor-blurred-foreground": FG,
        "block-cursor-blurred-text-style": "bold",
        "block-hover-background": BG,
        "input-cursor-background": FG,
        "input-cursor-foreground": BG,
        "input-selection-background": "#3a3a3a",
        "scrollbar": BORDER,
        "scrollbar-background": BG,
        "scrollbar-hover": MUTED,
        "scrollbar-active": MUTED,
        "footer-background": BG,
    },
)

ANSI = TerminalTheme(
    (0x12, 0x12, 0x12),
    (0xD4, 0xD4, 0xD4),
    [
        (0x12, 0x12, 0x12),
        (0xE0, 0x6C, 0x6C),
        (0x6C, 0xC0, 0x7A),
        (0xE0, 0xB5, 0x5E),
        (0x6F, 0x9F, 0xD8),
        (0xB8, 0x8C, 0xD8),
        (0x6C, 0xB8, 0xC0),
        (0xD4, 0xD4, 0xD4),
    ],
    [
        (0x5A, 0x5A, 0x5A),
        (0xF0, 0x80, 0x80),
        (0x80, 0xD0, 0x8C),
        (0xF0, 0xC8, 0x70),
        (0x88, 0xB0, 0xE8),
        (0xC8, 0xA0, 0xE8),
        (0x80, 0xC8, 0xD0),
        (0xF0, 0xF0, 0xF0),
    ],
)


def example_script(cwd: Path) -> str:
    """A real script name for the empty state's example: gpu.yaml's script, else a .py
    file in the project root (train*.py first), else train.py."""
    try:
        from gpu_router.jobspec import find_project_root

        root = find_project_root(cwd)
    except Exception:
        root = cwd
    try:
        import yaml

        doc = yaml.safe_load((root / "gpu.yaml").read_text()) or {}
        script = doc.get("script") if isinstance(doc, dict) else None
        if isinstance(script, str) and script:
            return script
    except (OSError, ValueError, yaml.YAMLError):
        pass
    try:
        pys = sorted(p.name for p in cwd.glob("*.py") if p.is_file())
    except OSError:
        pys = []
    trains = [p for p in pys if p.startswith("train")]
    return (trains or pys or ["train.py"])[0]


class GpuShell(App[int]):
    """`gpu` with no arguments. See the module docstring."""

    TITLE = "gpu-router"
    ENABLE_COMMAND_PALETTE = False
    CSS = """
    Screen { background: $background; color: $foreground; layout: vertical;
             overflow: hidden hidden; }
    #transcript { height: auto; scrollbar-size-vertical: 1; }
    #prompt-row { height: 1; padding: 0 1; }
    #prompt-mark { width: 2; color: $text-muted; }
    #bottom { dock: bottom; height: 2; }
    #popup { scrollbar-size-vertical: 1; }
    #popup > .option-list--option-highlighted { background: $popup-cursor; text-style: bold; }
    #rule { color: $panel-border; margin: 0; }
    """
    BINDINGS = [
        Binding("ctrl+c", "interrupt", "interrupt", show=False, priority=True),
        Binding("ctrl+d", "eof", "quit", show=False, priority=True),
        Binding("ctrl+q", "quit_shell", "quit", show=False, priority=True),
        Binding("ctrl+l", "clear_screen", "clear", show=False),
        Binding("pageup", "scroll_transcript(-1)", show=False, priority=True),
        Binding("pagedown", "scroll_transcript(1)", show=False, priority=True),
    ]

    def __init__(
        self,
        *,
        paths: Paths | None = None,
        cwd: Path | None = None,
        autostart: bool = True,
        poll_s: float = 1.0,
        quota_s: float = 60.0,
        retry_s: float = 3.0,
        history_file: Path | None = None,
    ) -> None:
        super().__init__()
        self.paths = paths or Paths.from_env()
        self.cwd = (cwd or Path.cwd()).resolve()
        self.autostart = autostart
        self.feed = Feed(
            paths=self.paths,
            publish=self._publish,
            poll_s=poll_s,
            quota_s=quota_s,
            retry_s=retry_s,
            autostart=autostart,
        )
        self.history_path = history_file or (self.paths.home / HISTORY_FILE)
        self.history: list[str] = []
        self.history_pos: int | None = None
        self._draft = ""  # the line being typed when history navigation started
        self.last_interrupt = 0.0
        self.snap: Snapshot | None = None
        self._last_up: Snapshot | None = None  # baseline for notices, across outages
        self._announced_pid: int | None = None  # the auto-started daemon already announced
        self._command_threads: set[threading.Thread] = set()
        self._feed_thread: threading.Thread | None = None
        self.panel = JobPanel(example_script(self.cwd), id="panel")
        self.transcript = VerticalScroll(id="transcript")
        self.prompt = Prompt(
            placeholder=f"/run {self.panel.example} · / for commands · tab completes", id="prompt"
        )
        self.popup = CommandPopup(id="popup")
        self.footer = StatusFooter(id="footer")
        self.register_theme(THEME)
        self.theme = THEME.name
        self.ansi_theme_dark = ANSI

    # ------------------------------------------------------------------ layout

    def compose(self) -> ComposeResult:
        yield self.panel
        yield self.transcript
        with Horizontal(id="prompt-row"):
            yield Label(">", id="prompt-mark")
            yield self.prompt
        yield self.popup
        with Vertical(id="bottom"):
            yield Rule(id="rule")
            yield self.footer

    def on_mount(self) -> None:
        self._load_history()
        self.prompt.focus()
        # Follow the newest output through every layout change: a block that grows, the
        # popup opening (the transcript loses rows), a terminal resize. A geometric "was it
        # at the end?" check alone breaks as soon as the viewport shrinks under it, and the
        # shell then never followed again. pgup / the wheel release the anchor, scrolling
        # back to the end restores it (Textual's anchor), a submitted command re-anchors.
        self.transcript.anchor()
        self._feed_thread = threading.Thread(
            target=self.feed.run, name="gpu-shell-feed", daemon=True
        )
        self._feed_thread.start()
        self.set_interval(1.0, self._tick)
        self.call_after_refresh(self._fit)

    def on_resize(self) -> None:
        self.call_after_refresh(self._fit)

    def _fit(self) -> None:
        """Give the transcript whatever height the fixed rows leave (it is height:auto, so
        a short transcript keeps the prompt right under the panel)."""
        h = self.size.height
        self.panel.max_rows = max(3, int(h * 0.4) - 2)
        fixed = self.panel.outer_size.height + 1 + 2  # prompt row, rule, footer
        if self.popup.display:
            fixed += self.popup.outer_size.height
        room = max(3, h - fixed)
        self.transcript.styles.max_height = room
        for block in self.query(LiveBlock):
            block.max_rows = max(4, room - 4)  # head, foot, margin, the command line

    async def on_unmount(self) -> None:
        self.feed.stop()
        for block in self.query(LiveBlock):
            block.stop_event.set()

    def _tick(self) -> None:
        """Once a second: elapsed clocks, 'ago' labels and staleness move without a poll."""
        self.panel.refresh(layout=True)
        self.footer.refresh()
        self.call_after_refresh(self._fit)

    # ------------------------------------------------------------------ feed

    def _publish(self, snap: Snapshot) -> None:
        """Called from the feed thread."""
        try:
            self.call_from_thread(self._apply, snap)
        except Exception:
            self.feed.stop()

    def _apply(self, snap: Snapshot) -> None:
        self.snap = snap
        self.panel.show(snap)
        self.footer.show(snap)
        if snap.conn.status is ConnStatus.UP:
            # compare with the last UP snapshot, whatever DOWN/STARTING came in between:
            # a daemon restarted by launchd may recover jobs that finished or now need
            # approval, and those are exactly the changes worth a line (D44)
            if self._last_up is not None:
                self._notices(self._last_up, snap)
            self._last_up = snap
        pid = snap.conn.started_pid
        if pid and pid != self._announced_pid:
            self._announced_pid = pid  # once per daemon this shell started, not per reconnect
            self._note(
                [
                    Text(
                        f"daemon started in the background (pid {snap.conn.started_pid}); "
                        "`gpu daemon install-launchd` starts it at login",
                        style="dim",
                    )
                ]
            )
        self.call_after_refresh(self._fit)

    def _notices(self, prev: Snapshot, snap: Snapshot) -> None:
        """One transcript line when a job finishes, fails, needs approval or moves. Jobs a
        live /logs block follows get none (it shows every event); a job whose finish a
        /logs or /watch block showed itself gets no second finish line (D44)."""
        logs_live = {b.job.id for b in self.query(LogsBlock) if b.live}
        ended_shown = {b.job.id for b in self.query(LiveBlock) if b.live or b.saw_end}
        now_active = {j.id: j for j in snap.active}
        recent = {j.id: j for j in snap.recent}
        lines: list[RenderableType] = []
        for old in prev.active:
            new = now_active.get(old.id)
            if new is None:
                done = recent.get(old.id)
                if old.id not in ended_shown and done is not None and is_terminal(done.state):
                    lines.append(notice_line(done, _clock.now()))
                continue
            if old.id in logs_live or new.state is old.state:
                continue
            if new.state is JobState.AWAITING_APPROVAL:
                lines.append(_approval_notice(new))
            elif new.state is JobState.MIGRATING:
                line = seg(f"{render.ICON_MIGRATED} ", "yellow")
                line.append(f"job {new.short_id} {new.name} is moving: ")
                line.append(shell_words(new.message or ""))
                lines.append(line)
        for j in snap.active:
            if j.id not in {o.id for o in prev.active} and j.state is JobState.AWAITING_APPROVAL:
                lines.append(_approval_notice(j))
        if lines:
            self._note(lines)

    def _note(self, items: list[RenderableType]) -> None:
        block = CommandBlock(None, classes="notice")
        block.items = list(items)
        block.running = False
        self._mount_block(block)

    # ------------------------------------------------------------------ transcript

    def _mount_block(self, block: Widget) -> None:
        at_end = self.transcript.max_scroll_y - self.transcript.scroll_y <= 1
        self.transcript.mount(block)
        self._prune_transcript()
        if at_end:
            self.call_after_refresh(lambda: self.transcript.scroll_end(animate=False))
        self.call_after_refresh(self._fit)

    def _prune_transcript(self) -> None:
        """Bound what a shell left open for days holds (D44): the oldest blocks past
        MAX_BLOCKS go (never a live view), and stopped /logs blocks other than the newest
        KEEP_LOG_OUTPUT drop their rendered lines (up to 5000 each)."""
        children = list(self.transcript.children)
        extra = len(children) - MAX_BLOCKS
        if extra > 0:
            gone = [c for c in children if not (isinstance(c, LiveBlock) and c.live)][:extra]
            for child in gone:
                if isinstance(child, LiveBlock):
                    child.stop_event.set()
                child.remove()
        stopped = [
            b
            for b in self.transcript.children
            if isinstance(b, LogsBlock) and not b.live and not b.forgotten
        ]
        for block in stopped[:-KEEP_LOG_OUTPUT]:
            block.forget_output()

    def action_scroll_transcript(self, direction: int) -> None:
        """pgup/pgdn: the newest /logs block first while it has lines hidden that way
        (it scrolls inside the transcript), then the transcript (D44)."""
        logs = list(self.query(LogsBlock))
        if logs and logs[-1].scroll_page(direction):
            return
        page = max(1, self.transcript.size.height - 2)
        self.transcript.scroll_relative(y=direction * page, animate=False, immediate=True)

    def action_clear_screen(self) -> None:
        self.clear_transcript_now()

    def clear_transcript_now(self) -> None:
        for block in self.query(LiveBlock):
            block.stop()
        self.transcript.remove_children()
        self.call_after_refresh(self._fit)

    # ------------------------------------------------------------------ prompt

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input is not self.prompt:
            return
        self._refresh_popup()

    def _completion(self) -> Completion | None:
        snap = self.snap
        jobs: list[JobView] = []
        providers = []
        if snap is not None:
            jobs = [*sort_active(snap.active), *snap.recent]
            providers = list(snap.providers)
        return complete(self.prompt.value, Known(jobs=jobs, providers=providers, cwd=self.cwd))

    def _refresh_popup(self) -> None:
        comp = self._completion()
        if comp is None or not comp.candidates:
            self.popup.hide()
        elif len(comp.candidates) == 1 and comp.candidates[0].value == comp.word:
            self.popup.hide()  # already complete
        else:
            self.popup.show_candidates(comp.candidates)
        self.call_after_refresh(self._fit)

    def _accept(self, value: str, final: bool) -> None:
        comp = self._completion()
        start = comp.start if comp is not None else len(self.prompt.value)
        text = self.prompt.value[:start] + value + (" " if final else "")
        self.prompt.value = text
        self.prompt.cursor_position = len(text)

    def action_prompt_complete(self) -> None:
        current = self.popup.current
        if current is not None:
            self._accept(current.value, current.final)
            self._refresh_popup()
            return
        comp = self._completion()
        if comp is None:
            return
        if len(comp.candidates) == 1:
            c = comp.candidates[0]
            self._accept(c.value, c.final)
        else:
            prefix = comp.common_prefix()
            if len(prefix) > len(comp.word):
                self._accept(prefix, False)
        self._refresh_popup()

    def action_prompt_move(self, delta: int) -> None:
        if self.popup.display:
            self.popup.move(delta)
            return
        if not self.history:
            return
        if self.history_pos is None:
            if delta > 0:
                return
            self.history_pos = len(self.history)
            self._draft = self.prompt.value  # the line being typed comes back on the way down
        self.history_pos = max(0, min(len(self.history), self.history_pos + delta))
        if self.history_pos < len(self.history):
            value = self.history[self.history_pos]
        else:
            value = self._draft
            self.history_pos = None
        self.prompt.value = value
        self.prompt.cursor_position = len(value)
        self.popup.hide()

    def action_prompt_back(self) -> None:
        if self.popup.display:
            self.popup.hide()
            self.call_after_refresh(self._fit)
            return
        live = [b for b in self.query(LiveBlock) if b.live]
        if live:
            for b in live:
                b.stop()
            return
        self.prompt.value = ""

    def action_prompt_enter(self) -> None:
        current = self.popup.current
        if current is not None:
            comp = self._completion()
            word = comp.word if comp else ""
            if current.value != word and self._may_pick(word, current):
                self._accept(current.value, current.final)
                self.popup.hide()
                if not self._run_after_accept(current):
                    self._refresh_popup()
                    return
        line = self.prompt.value.strip()
        self.prompt.value = ""
        self.popup.hide()
        self.history_pos = None
        if not line:
            return
        self._remember(line)
        self.execute(line)

    def _may_pick(self, word: str, current: Candidate) -> bool:
        """Whether Enter takes the highlighted candidate (D44): only when the user moved
        the highlight or typed part of the word. Right after tab or a space the word is
        empty and the highlight is just row 0: Enter runs the line as typed (`/status `
        is the overview, `/cancel ` asks for an id). A default highlight never supplies
        the job id of /cancel, /deny or /approve."""
        if self.popup.moved:
            return True
        if not word:
            return False
        if current.value.startswith("/"):
            return True  # a command name being typed
        words = self.prompt.value.split()
        cmd = cmds.lookup(words[0]) if words else None
        return cmd is None or cmd.name not in NO_DEFAULT_PICK

    def _run_after_accept(self, current: Candidate) -> bool:
        """After Enter picked a candidate: run the line now (a command without required
        arguments, a job id, a provider), or keep editing (a command that needs an
        argument, a directory, a script: flags may follow and a job costs quota; a
        candidate that needs a further argument, like `/policy set KEY`)."""
        if not current.final or current.more:
            return False
        words = self.prompt.value.split()
        cmd = cmds.lookup(words[0]) if words else None
        if cmd is None:
            return False
        if current.value.startswith("/"):
            return "<" not in cmd.usage
        return cmd.arg != "script"

    def action_interrupt(self) -> None:
        live = [b for b in self.query(LiveBlock) if b.live]
        if live:
            for b in live:
                b.stop("detached")
            return
        if self.prompt.value:
            self.prompt.value = ""
            self.popup.hide()
            return
        import time

        now = time.monotonic()
        if now - self.last_interrupt < 2.0:
            self.action_quit_shell()
            return
        self.last_interrupt = now
        self._note([Text("press ctrl+c again (or ctrl+d) to quit; jobs keep running", "dim")])

    def action_eof(self) -> None:
        if not self.prompt.value:
            self.action_quit_shell()

    def action_quit_shell(self) -> None:
        self.feed.stop()
        for block in self.query(LiveBlock):
            block.stop_event.set()
        self.exit(0)

    # ------------------------------------------------------------------ commands

    def execute(self, line: str) -> CommandBlock:
        """Run one prompt line; returns its transcript block."""
        self.transcript.anchor()  # the user wants to see what they just ran
        block = CommandBlock(line)
        try:
            cmd, word, args = cmds.split_line(line)
        except cmds.UsageError as exc:
            block.items = cmds.error_renderables(exc)
            block.running = False
            self._mount_block(block)
            return block
        if cmd is None:
            guess = cmds.suggest(word) if word else None
            block.items = [
                Text(f"{render.ICON_FAILED} unknown command: {word}", style="red"),
                Text(
                    ("  did you mean /" + guess + "? " if guess else "  ")
                    + "/help lists every command",
                    style="dim",
                ),
            ]
            block.running = False
            self._mount_block(block)
            return block
        self._mount_block(block)
        if cmd.name == "clear":
            self.clear_transcript_now()
            return block
        if cmd.name == "exit":
            self.action_quit_shell()
            return block
        # a plain daemon thread, not a Textual worker: App.run() waits for its workers,
        # so quitting during a 10 min /fetch would leave the process hanging (D44)
        thread = threading.Thread(
            target=self._run_command,
            args=(cmd, args, block),
            name=f"gpu-shell-cmd-{cmd.name}",
            daemon=True,
        )
        self._command_threads.add(thread)
        thread.start()
        return block

    def busy(self) -> bool:
        """A command is still running (waiting on the daemon or a provider)."""
        return any(t.is_alive() for t in list(self._command_threads))

    def _run_command(self, cmd: cmds.Command, args: list[str], block: CommandBlock) -> None:
        """Worker thread: run the handler, stream its output into `block`."""

        def sink(items: list[RenderableType]) -> None:
            with contextlib.suppress(Exception):
                self.call_from_thread(block.add, items)

        ctx = cmds.Ctx(host=self, sink=sink)
        try:
            cmds.HANDLERS[cmd.name](ctx, args)
        except BaseException as exc:
            if isinstance(exc, KeyboardInterrupt | SystemExit):
                raise
            sink(cmds.error_renderables(exc))
        finally:
            ctx.close()
            with contextlib.suppress(Exception):
                self.call_from_thread(block.done)
            self.feed.poke()
            self._command_threads.discard(threading.current_thread())

    # ------------------------------------------------------------------ Host (commands.Host)

    def connect(self, note: Callable[[str], None]) -> GpuClient:
        from gpu_router.daemon.spawn import connect

        conn = connect(
            self.paths,
            start=self.autostart,
            on_start=lambda: note("the daemon is not running; starting it in the background"),
        )
        if conn.started:
            note(
                f"daemon started (pid {conn.pid}); `gpu daemon install-launchd` starts it at login"
            )
            self.feed.poke()
        conn.client.client_name = "shell"
        return conn.client

    def _client_factory(self) -> GpuClient:
        return self.connect(lambda _m: None)

    def open_logs(self, job: JobView, attempt: int | None) -> None:
        self.call_from_thread(
            self._open_live, LogsBlock(job, self._client_factory, attempt=attempt)
        )

    def open_watch(self, job: JobView, metric: str | None) -> None:
        self.call_from_thread(self._open_live, WatchBlock(job, self._client_factory, metric=metric))

    def _open_live(self, block: LiveBlock) -> None:
        for other in self.query(LiveBlock):
            if other.live:
                other.stop("replaced")
        self._mount_block(block)
        self.call_after_refresh(lambda: self.transcript.scroll_end(animate=False))

    def run_suspended(self, argv: list[str]) -> int | None:
        """Hand the terminal to `argv` (the editor) and take it back; None when that is
        impossible (headless, no such program)."""
        result: dict[str, int] = {}

        def run() -> None:
            try:
                with self.suspend():
                    result["rc"] = subprocess.call(argv)
            except Exception:  # SuspendNotSupported, FileNotFoundError, ...
                return

        self.call_from_thread(run)
        return result.get("rc")

    def clear_transcript(self) -> None:
        self.call_from_thread(self.clear_transcript_now)

    def quit_shell(self) -> None:
        self.call_from_thread(self.action_quit_shell)

    def poke(self) -> None:
        self.feed.poke()

    def snapshot(self) -> Snapshot | None:
        return self.snap

    def output_width(self) -> int:
        return max(40, self.transcript.content_size.width - 2)

    # ------------------------------------------------------------------ history

    def _load_history(self) -> None:
        try:
            lines = self.history_path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            return
        self.history = [ln for ln in lines if ln.strip()][-HISTORY_MAX:]

    def _remember(self, line: str) -> None:
        if self.history and self.history[-1] == line:
            return
        self.history.append(line)
        del self.history[:-HISTORY_MAX]
        try:
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(self.history_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write("\n".join(self.history) + "\n")
        except OSError:
            pass


def _approval_notice(job: JobView) -> Text:
    line = seg(f"{render.ICON_WAITING} ", "yellow")
    line.append(f"job {job.short_id} {job.name} {render.state_label(job.state)}  ")
    line.append(f"/approve {job.short_id} · /deny {job.short_id}", style="dim")
    return line


def run() -> int:
    """Entry point for bare `gpu` in a terminal (entry.py)."""
    app = GpuShell()
    code = int(app.run() or 0)
    if app.busy():
        # A command still waits (a /fetch up to 10 min, a submit, healthchecks): its thread
        # and a healthcheck pool would keep the process alive with the terminal already
        # restored. Nothing it does matters once the user left; jobs run in the daemon.
        with contextlib.suppress(Exception):
            sys.stdout.flush()
            sys.stderr.flush()
        os._exit(code)
    return code


__all__ = ["GpuShell", "run"]
