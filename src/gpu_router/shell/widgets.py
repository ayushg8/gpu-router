"""Textual widgets of the shell (phase 4): job panel, transcript blocks, live /logs and
/watch blocks, the completion popup, the prompt and the footer.

Live blocks poll the daemon from their own thread (stop = a threading.Event checked at
every wait, so Esc and quitting return within one poll) and hand results to the UI thread
with `app.call_from_thread`.
"""

from __future__ import annotations

import io
import threading
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Any, cast

from rich.console import Group, RenderableType
from rich.text import Text
from textual.binding import Binding
from textual.containers import ScrollableContainer, Vertical
from textual.timer import Timer
from textual.widget import Widget
from textual.widgets import Input, OptionList, RichLog, Static
from textual.widgets.option_list import Option

from gpu_router.api import JobView
from gpu_router.cli import render
from gpu_router.clock import SystemClock
from gpu_router.errors import GpuRouterError
from gpu_router.shell.chart import chart_lines, mini_line
from gpu_router.shell.complete import Candidate
from gpu_router.shell.feed import read_new_lines
from gpu_router.shell.metrics import MetricHistory, primary_metric
from gpu_router.shell.panel import footer_text, icon_cell, panel_lines
from gpu_router.shell.state import Snapshot
from gpu_router.statemachine import JobState, is_terminal

if TYPE_CHECKING:
    from gpu_router.client import GpuClient

_clock = SystemClock()
LOG_POLL_S = 0.5
WATCH_POLL_S = 1.0
MAX_LOG_LINES = 5000  # lines a LogsBlock keeps; never more are read or rendered (D44)
TAIL_LINES = MAX_LOG_LINES - 50  # read per attempt at most (room for events, headers)
WRITE_CHUNK = 250  # lines per UI-thread call, so keys (esc) are handled in between


def follow_tail(widget: Widget, change: Callable[[], None]) -> None:
    """Apply `change` to a transcript block; if the transcript was scrolled to its end,
    keep it there after the block grows (scrolled up to read = left alone)."""
    scroller = widget.parent
    at_end = isinstance(scroller, ScrollableContainer) and (
        scroller.max_scroll_y - scroller.scroll_y <= 1
    )
    change()
    if at_end and isinstance(scroller, ScrollableContainer):
        widget.call_after_refresh(lambda: scroller.scroll_end(animate=False))


# --------------------------------------------------------------------------- panel + footer


class JobPanel(Static):
    """The bordered live panel on top (spec mockup), redrawn from the latest Snapshot."""

    DEFAULT_CSS = """
    JobPanel {
        height: auto;
        border: round $panel-border;
        border-title-color: $text-muted;
        border-subtitle-color: $text-muted;
        padding: 0 1;
    }
    """

    def __init__(self, example: str, **kw: Any) -> None:
        super().__init__(**kw)
        self.snap: Snapshot | None = None
        self.example = example
        self.max_rows = 12
        self.border_title = "gpu-router"

    def show(self, snap: Snapshot) -> None:
        self.snap = snap
        self.refresh(layout=True)

    def lines(self) -> list[Text]:
        if self.snap is None:
            return [Text("connecting to the gpu-router daemon…", style="dim")]
        width = max(20, self.content_size.width or (self.app.size.width - 4))
        return panel_lines(self.snap, width, self.max_rows, now=_clock.now(), example=self.example)

    def render(self) -> RenderableType:
        lines = self.lines()
        return Group(*lines) if lines else Text("")

    @property
    def plain(self) -> str:
        return "\n".join(t.plain for t in self.lines())


class StatusFooter(Static):
    DEFAULT_CSS = """
    StatusFooter { height: 1; padding: 0 1; }
    """

    def __init__(self, **kw: Any) -> None:
        super().__init__(**kw)
        self.snap: Snapshot | None = None

    def show(self, snap: Snapshot) -> None:
        self.snap = snap
        self.refresh()

    def render(self) -> Text:
        if self.snap is None:
            return Text("connecting…", style="dim")
        width = max(20, self.content_size.width or self.app.size.width - 2)
        return footer_text(self.snap, width, now=_clock.now())

    @property
    def plain(self) -> str:
        return self.render().plain


# --------------------------------------------------------------------------- transcript


class CommandBlock(Static):
    """One command and its output: `> /jobs`, then whatever the command printed."""

    DEFAULT_CSS = """
    CommandBlock { height: auto; padding: 0 1; margin-bottom: 1; }
    """

    def __init__(self, line: str | None, **kw: Any) -> None:
        super().__init__(**kw)
        self.line = line
        self.items: list[RenderableType] = []
        self.running = line is not None

    def add(self, items: Sequence[RenderableType]) -> None:
        self.items.extend(items)
        follow_tail(self, self._redraw)

    def done(self) -> None:
        self.running = False
        follow_tail(self, self._redraw)

    def on_mount(self) -> None:
        self._redraw()

    def _redraw(self) -> None:
        parts: list[RenderableType] = []
        if self.line is not None:
            parts.append(Text.assemble(("> ", "dim"), (self.line, "dim")))
        parts.extend(self.items)
        if self.running:
            parts.append(Text("…", style="dim"))
        self.update(Group(*parts))

    @property
    def plain(self) -> str:
        """Text of the block (tests)."""
        out: list[str] = [f"> {self.line}"] if self.line else []
        for item in self.items:
            out.append(_plain(item))
        return "\n".join(out)


def _plain(item: RenderableType) -> str:
    if isinstance(item, Text):
        return item.plain
    if isinstance(item, str):
        return item
    from rich.console import Console

    console = Console(width=120, color_system=None, record=True, file=io.StringIO())
    console.print(item)
    return console.export_text()


class LiveBlock(Vertical):
    """Base for blocks that keep updating until the job finishes or the user stops them."""

    DEFAULT_CSS = """
    LiveBlock { height: auto; padding: 0 1; margin-bottom: 1; }
    LiveBlock > .live-head { height: auto; }
    LiveBlock > .live-foot { height: auto; color: $text-muted; }
    """

    def __init__(self, job: JobView, connect: Callable[[], GpuClient], **kw: Any) -> None:
        super().__init__(**kw)
        self.job = job
        self._connect = connect
        self.stop_event = threading.Event()
        self.finished = False
        self.saw_end = False  # showed the job's terminal state itself (not stopped, no error)
        self.max_rows = 20  # set by the app to what the transcript can show
        self.head = Static(classes="live-head")
        self.foot = Static(classes="live-foot")
        self.foot_text = ""  # plain text of the footer line(s), for tests

    @property
    def live(self) -> bool:
        return not self.finished and not self.stop_event.is_set()

    def stop(self, why: str = "stopped") -> None:
        if self.stop_event.is_set():
            return
        self.stop_event.set()
        if not self.finished:
            self._set_foot(Text(f"{why}; job {self.job.short_id} keeps running", style="dim"))

    def on_mount(self) -> None:
        self.update_head(self.job)
        self._set_foot(Text("esc stops · the job keeps running", style="dim"))
        threading.Thread(target=self._thread, name=f"gpu-shell-{self.kind}", daemon=True).start()

    kind = "live"

    def _set_foot(self, *lines: Text) -> None:
        self.foot_text = "\n".join(t.plain for t in lines)
        self.foot.update(Group(*lines) if lines else Text(""))

    def update_head(self, job: JobView) -> None:
        self.job = job
        head = icon_cell(job.state)
        head.append(f"{self.kind} ", style="dim")
        head.append(job.short_id, style="bold")
        head.append(f"  {job.name}  ")
        _, style = render.state_style(job.state)
        head.append(render.state_label(job.state), style=style)
        if job.provider:
            head.append(f" · {render.where(job)}", style="dim")
        self.head.update(head)

    def post(self, fn: Callable[..., None], *args: Any) -> bool:
        """Run fn on the UI thread; False when the app is gone (the thread then ends)."""
        if self.stop_event.is_set():
            return False
        try:
            self.app.call_from_thread(follow_tail, self, partial(fn, *args))
        except Exception:
            self.stop_event.set()
            return False
        return True

    def _thread(self) -> None:
        client: GpuClient | None = None
        try:
            client = self._connect()
            self.loop(client)
        except GpuRouterError as exc:
            self.post(self._error, exc.message)
        except Exception as exc:  # never let a poll thread die silently
            self.post(self._error, f"internal error: {type(exc).__name__}: {exc}")
        finally:
            if client is not None:
                client.close()

    def _error(self, message: str) -> None:
        self.finished = True
        self._set_foot(Text(f"{render.ICON_FAILED} {message}", style="red"))

    def loop(self, client: GpuClient) -> None:
        raise NotImplementedError

    def finish(self, job: JobView, lines: list[Text]) -> None:
        self.finished = True
        self.saw_end = True
        self.update_head(job)
        self._set_foot(*lines)


class LogsBlock(LiveBlock):
    """`/logs <id>`: the job's output as it arrives (all attempts, with transitions), the
    CLI's `gpu logs -f` / `gpu run` view (cli.app.follow_job) in the transcript."""

    # overflow-x hidden: a line wrapped for a wider view (before a resize or before the
    # transcript's scrollbar took a column) is clipped until the re-wrap, never given a
    # horizontal scrollbar that would cover the newest line
    DEFAULT_CSS = """
    LogsBlock > RichLog {
        height: 1; background: $background; scrollbar-size-vertical: 0; overflow-x: hidden;
    }
    """
    kind = "logs"
    REWRAP_DELAY_S = 0.15  # a dragged window sends a burst of resizes: re-wrap once, after

    def __init__(
        self, job: JobView, connect: Callable[[], GpuClient], attempt: int | None = None, **kw: Any
    ) -> None:
        super().__init__(job, connect, **kw)
        self.attempt = attempt
        # min_width: RichLog's default (78) wraps every line for 78 cells, wider than the
        # view in an 80-column terminal; lines are wrapped for the view's own width
        self.output = RichLog(
            max_lines=MAX_LOG_LINES,
            min_width=20,
            wrap=True,
            markup=False,
            highlight=False,
            auto_scroll=True,
        )
        self.count = 0
        self.rows = 0
        self.forgotten = False  # forget_output() dropped the rendered lines
        self.kept: deque[Text] = deque(maxlen=MAX_LOG_LINES)  # the lines shown, to re-wrap
        self.wrap_width = 0  # the view width the shown lines were wrapped for
        self._rewrap_timer: Timer | None = None

    def compose(self) -> Any:
        yield self.head
        yield self.output
        yield self.foot

    def write(self, job: JobView, lines: list[Text]) -> None:
        self.update_head(job)
        width = max(20, self.output.content_size.width or self.app.size.width - 6)
        # new lines keep the view at the end, unless pgup scrolled it back to read
        follow = self.output.scroll_y >= self.output.max_scroll_y - 1
        for line in lines:
            self.output.write(line, scroll_end=follow)
            self.kept.append(line)
            self.count += 1
            self.rows += max(1, -(-line.cell_len // width))  # wrapped rows
        if self.output.scrollable_content_region.width:  # else RichLog defers the render
            self.wrap_width = self.output.scrollable_content_region.width
        self.output.styles.height = min(self.max_rows, max(1, self.rows))

    def on_resize(self) -> None:
        if self._rewrap_timer is not None:
            self._rewrap_timer.stop()
        self._rewrap_timer = self.set_timer(self.REWRAP_DELAY_S, self.rewrap)

    def rewrap(self) -> None:
        """Wrap the shown lines again for the view's current width (RichLog keeps each line
        as wrapped when written: a narrower terminal would cut them, a wider one keep them
        folded). Keeps following the end when the reader was there."""
        self._rewrap_timer = None
        out = self.output
        width = out.scrollable_content_region.width
        if self.forgotten or not self.kept or not width or width == self.wrap_width:
            return
        follow = out.scroll_y >= out.max_scroll_y - 1
        out.clear()
        for line in self.kept:
            out.write(line, scroll_end=False)
        self.wrap_width = width
        self.rows = max(1, len(out.lines))
        out.styles.height = min(self.max_rows, self.rows)
        if follow:
            out.scroll_end(animate=False)

    def _event(self, ev: Any, now: float) -> Text:
        from gpu_router.shell.commands import shellify

        return cast(Text, shellify(render.event_text(ev, now)))

    def _skipped(self, n: int, count: int) -> Text:
        """What stands for lines never read: the RichLog would drop them anyway."""
        return Text(
            f"… {count} earlier lines of attempt {n} not shown here; "
            f"`gpu logs {self.job.short_id}` in a terminal prints them all",
            style="dim",
        )

    def loop(self, client: GpuClient) -> None:
        from gpu_router.cli.app import _split_events

        cursor = 0
        positions: dict[int, int] = {}
        shown: int | None = None
        while not self.stop_event.is_set():
            detail = client.job(self.job.id)
            evs = client.events(self.job.id, after=cursor)
            cursor = evs.next
            early, late = _split_events(list(evs.events))
            now = _clock.now()
            batch: list[Text] = [self._event(ev, now) for ev in early]
            for att in detail.attempts:
                if self.attempt is not None and att.n != self.attempt:
                    continue
                pos = positions.get(att.n, 0)
                if att.log_lines <= pos:
                    continue
                if shown != att.n:
                    if shown is not None or att.n > 1:
                        batch.append(Text(f"── attempt {att.n} on {att.provider} ──", "dim"))
                    shown = att.n
                # a job that already printed 100k lines (or prints faster than we poll):
                # start MAX_LOG_LINES from its end, never reading what cannot be kept
                skip = att.log_lines - pos - TAIL_LINES
                if skip > 0:
                    batch.append(self._skipped(att.n, skip))
                    pos += skip
                for rec in client.logs(self.job.id, attempt=att.n, offset=pos):
                    if rec.line is None or rec.offset is None:
                        continue
                    batch.append(Text(rec.line))
                    pos = rec.offset + 1
                positions[att.n] = max(pos, att.log_lines)
            batch += [self._event(ev, now) for ev in late]
            if len(batch) > MAX_LOG_LINES:  # several attempts at once: keep the newest
                dropped = len(batch) - MAX_LOG_LINES
                note = Text(f"… {dropped} lines not shown here", style="dim")
                batch = [note, *batch[-MAX_LOG_LINES:]]
            for i in range(0, len(batch), WRITE_CHUNK):
                if not self.post(self.write, detail.job, batch[i : i + WRITE_CHUNK]):
                    return
            if is_terminal(detail.job.state):
                self.post(self.finish, detail.job, final_lines(detail.job, self.count))
                return
            self.stop_event.wait(LOG_POLL_S)

    def scroll_page(self, direction: int) -> bool:
        """pgup/pgdn: scroll the output when it has lines hidden that way (the block is a
        scroller inside the transcript); False when there is nothing more to show."""
        out = self.output
        if out.max_scroll_y <= 0:
            return False
        if direction < 0 and out.scroll_y <= 0:
            return False
        if direction > 0 and out.scroll_y >= out.max_scroll_y:
            return False
        page = max(1, out.size.height - 1)
        # immediate: a key press scrolls now, not after a refresh that a busy screen can
        # keep putting off (Textual defers scroll_to without it; seen on CI, 2026-10-09)
        out.scroll_relative(y=direction * page, animate=False, immediate=True)
        return True

    def forget_output(self) -> None:
        """Drop the rendered lines of a stopped block (the transcript keeps its head and
        foot): old /logs blocks must not hold 5000 lines each for the whole session."""
        if self.live or self.forgotten:
            return
        self.forgotten = True
        self.kept.clear()
        self.output.clear()
        self.output.write(
            Text(f"(output cleared to save memory; /logs {self.job.short_id} shows it again)"),
        )
        self.output.styles.height = 1
        self.rows = 1


def final_lines(job: JobView, count: int) -> list[Text]:
    """The CLI's closing line (cli.app._print_final) plus the next step, shellified."""
    from gpu_router.cli.app import Out, _print_final
    from gpu_router.shell.commands import Collector

    items: list[RenderableType] = []
    col = Collector(items.extend, width=120)
    out = Out(False)
    out.console = col
    out.err = col
    _print_final(out, job)
    if count == 0 and job.state is not JobState.DONE:
        items.append(Text("this job produced no output", style="dim"))
    return [i for i in items if isinstance(i, Text)]


@dataclass
class WatchState:
    job: JobView
    names: list[str] = field(default_factory=list)
    series: dict[str, list[float]] = field(default_factory=dict)
    step: int | None = None
    total: int | None = None
    first_step: int | None = None


class WatchBlock(LiveBlock):
    """`/watch <id> [metric]`: a live chart of the job's loss (or the named metric) with
    one-line sparklines for its other metrics."""

    DEFAULT_CSS = """
    WatchBlock > .chart { height: auto; }
    """
    kind = "watch"

    def __init__(
        self, job: JobView, connect: Callable[[], GpuClient], metric: str | None = None, **kw: Any
    ) -> None:
        super().__init__(job, connect, **kw)
        self.metric = metric
        self.state = WatchState(job)
        self.chart = Static(classes="chart")
        self.chart_height = 10

    def compose(self) -> Any:
        yield self.head
        yield self.chart
        yield self.foot

    def on_resize(self) -> None:
        self.redraw()

    def show(self, state: WatchState) -> None:
        self.state = state
        self.update_head(state.job)
        self.redraw()

    def redraw(self) -> None:
        st = self.state
        width = max(30, (self.content_size.width or self.app.size.width - 4))
        name = self.metric if self.metric in st.series else primary_metric(st.series)
        if self.metric and self.metric not in st.series and st.series:
            lines: list[Text] = [
                Text(f"no metric {self.metric!r} yet; showing {name}", style="dim")
            ]
        else:
            lines = []
        lines += chart_lines(
            name or self.metric or "loss",
            st.series.get(name or "", []),
            width=width,
            height=self.chart_height,
            step=st.step,
            total=st.total,
            first_step=st.first_step,
        )
        others = [n for n in st.series if n != name]
        if others:
            lines.append(Text(""))
            row = Text()
            for other in others[:6]:
                piece = mini_line(other, st.series[other], 40)
                if row.plain and row.cell_len + 3 + piece.cell_len > width:
                    lines.append(row)
                    row = Text()
                if row.plain:
                    row.append("   ")
                row.append_text(piece)
            lines.append(row)
        self.chart.update(Group(*lines))

    def loop(self, client: GpuClient) -> None:
        hist = MetricHistory()
        while not self.stop_event.is_set():
            detail = client.job(self.job.id)
            for att in detail.attempts:
                if att.log_lines > hist.positions.get(att.n, 0):
                    read_new_lines(client, self.job.id, att.n, hist)
            job = detail.job
            first = None
            name = primary_metric(hist.series)
            if name and hist.series[name]:
                first = hist.series[name][0][0]
            state = WatchState(
                job=job,
                names=hist.names(),
                series={k: hist.values(k) for k in hist.series},
                step=hist.step if hist.step is not None else job.progress.step,
                total=hist.total or job.progress.total,
                first_step=first,
            )
            if not self.post(self.show, state):
                return
            if is_terminal(job.state):
                self.post(self.finish, job, final_lines(job, 1))
                return
            self.stop_event.wait(WATCH_POLL_S)


# --------------------------------------------------------------------------- prompt + popup


class CommandPopup(OptionList):
    """Candidates for the word being typed; never takes focus (the prompt keeps it)."""

    DEFAULT_CSS = """
    CommandPopup {
        height: auto;
        max-height: 9;
        border: none;
        padding: 0 1;
        background: $background;
        display: none;
    }
    CommandPopup > .option-list--option-highlighted {
        background: $boost;
        color: $text;
        text-style: bold;
    }
    CommandPopup:focus > .option-list--option-highlighted { background: $boost; }
    """
    can_focus = False

    def __init__(self, **kw: Any) -> None:
        super().__init__(**kw)
        self.candidates: tuple[Candidate, ...] = ()
        self.moved = False  # the user moved the highlight since these candidates showed

    def show_candidates(self, candidates: Sequence[Candidate]) -> None:
        self.candidates = tuple(candidates)
        self.moved = False
        self.clear_options()
        self.add_options([Option(c.label) for c in self.candidates])
        self.display = bool(self.candidates)
        if self.candidates:
            self.highlighted = 0

    def hide(self) -> None:
        self.candidates = ()
        self.moved = False
        self.display = False

    @property
    def current(self) -> Candidate | None:
        if not self.display or not self.candidates:
            return None
        i = self.highlighted if self.highlighted is not None else 0
        return self.candidates[i] if 0 <= i < len(self.candidates) else None

    def move(self, delta: int) -> None:
        if not self.candidates:
            return
        i = self.highlighted if self.highlighted is not None else 0
        self.highlighted = (i + delta) % len(self.candidates)
        self.moved = True


class Prompt(Input):
    """The `>` prompt: tab completes, ↑↓ walk the popup or the history, esc backs out."""

    DEFAULT_CSS = """
    Prompt { border: none; height: 1; padding: 0; background: $background; }
    Prompt:focus { border: none; background: $background; background-tint: 0%; }
    Prompt > .input--placeholder { color: $text-muted; }
    """
    BINDINGS = [
        Binding("tab", "complete", "complete", show=False),
        Binding("up", "up", show=False),
        Binding("down", "down", show=False),
        Binding("escape", "back", show=False),
    ]

    def action_complete(self) -> None:
        self.app.action_prompt_complete()  # type: ignore[attr-defined]

    def action_up(self) -> None:
        self.app.action_prompt_move(-1)  # type: ignore[attr-defined]

    def action_down(self) -> None:
        self.app.action_prompt_move(1)  # type: ignore[attr-defined]

    def action_back(self) -> None:
        self.app.action_prompt_back()  # type: ignore[attr-defined]

    async def action_submit(self) -> None:
        self.app.action_prompt_enter()  # type: ignore[attr-defined]
