"""The live job panel and the footer status bar (phase 4). Pure formatting: Snapshot in,
rich Text lines out, so every state is unit-testable without a terminal.

Visual language (spec UX 6, shared with cli/render.py): colour only on state icons and
state words (green running, yellow waiting, red failed, dim idle); labels dim, values
plain; direction as a glyph (↓ ↑ →), never as a colour. Icons are padded to two cells
because ⚡ is double width and ⏸ ✓ ✗ ↪ are single, so the columns line up.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from rich.cells import cell_len
from rich.text import Text

from gpu_router.api import JobView, ProviderView
from gpu_router.cli import render
from gpu_router.models import ProviderHealth, QuotaSnapshot
from gpu_router.shell.metrics import fmt_value, sparkline, trend
from gpu_router.shell.state import Conn, ConnStatus, JobMetric, Snapshot
from gpu_router.statemachine import JobState

NAME_MAX = 28
EMPTY_TABLE_MAX = 6  # providers listed one per row in the empty state
SPARK_W = 12
GAP = "  "
INDENT = "   "  # line 2 starts under "job"

_ORDER = {
    JobState.RUNNING: 0,
    JobState.CHECKPOINTING: 0,
    JobState.AWAITING_APPROVAL: 1,
    JobState.MIGRATING: 2,
    JobState.PROVISIONING: 3,
    JobState.ROUTING: 4,
    JobState.QUEUED: 5,
    JobState.CANCELLING: 6,
}
_WAIT_WORD = {
    JobState.QUEUED: "queued",
    JobState.ROUTING: "routing",
    JobState.PROVISIONING: "starting",
    JobState.MIGRATING: "moving",
    JobState.CANCELLING: "cancelling",
    JobState.AWAITING_APPROVAL: render.state_label(JobState.AWAITING_APPROVAL),  # one word
}
_UNIT = {"gpu_hours": "h", "credits": " cr", "usd": ""}


# --------------------------------------------------------------------------- cells


def fit(text: str, width: int) -> str:
    """`text` cut to `width` cells with an ellipsis (no padding)."""
    if width <= 0:
        return ""
    if cell_len(text) <= width:
        return text
    out = ""
    for ch in text:
        if cell_len(out + ch) > width - 1:
            break
        out += ch
    return out + "…"


def pad(text: str, width: int) -> str:
    """`text` fitted, then padded with spaces to exactly `width` cells."""
    t = fit(text, width)
    return t + " " * max(0, width - cell_len(t))


def shell_words(text: str) -> str:
    """Daemon text for the shell: CLI hints (`gpu logs a7f2`) become slash commands, so
    the panel, notices and live views never mix the two syntaxes (D44)."""
    from gpu_router.shell.commands import shellify_text

    return shellify_text(text)


def seg(text: str, style: str = "") -> Text:
    """A Text whose style covers only this text: rich's `Text(s, style=x)` makes x the
    BASE style, which everything appended later inherits (a green icon would turn the
    whole row green)."""
    out = Text()
    out.append(text, style=style)
    return out


def icon_cell(state: JobState, *, migrated: bool = False) -> Text:
    """The state icon in a 2-cell slot plus one space (3 cells)."""
    icon, style = render.state_style(state)
    if migrated:
        icon = render.ICON_MIGRATED
    out = seg(icon, style)
    out.append(" " * max(0, 2 - cell_len(icon)) + " ")
    return out


def _label(out: Text, label: str, value: str | Text) -> None:
    out.append(label + " ", style="dim")
    if isinstance(value, Text):
        out.append_text(value)
    else:
        out.append(value)


def _right(line: Text, right: Text, width: int) -> Text:
    """`line` + spaces + `right`, right-aligned at `width`; `right` is dropped (not the
    line) when both do not fit."""
    used = line.cell_len
    if right.plain and used + 2 + right.cell_len <= width:
        line.append(" " * (width - used - right.cell_len))
        line.append_text(right)
    line.truncate(width, overflow="ellipsis")
    return line


# --------------------------------------------------------------------------- job rows


@dataclass(frozen=True)
class Cols:
    """Column widths shared by every job row in one render, so the panel reads as a table."""

    sid: int
    name: int
    mid: int


def _mid(job: JobView) -> Text:
    """Line 1's middle column: where it runs, or the waiting state word (yellow)."""
    if job.state in (JobState.RUNNING, JobState.CHECKPOINTING):
        return Text(render.where(job))
    word = _WAIT_WORD.get(job.state)
    if word is None:
        return Text(str(job.state).replace("_", " "))
    _, style = render.state_style(job.state)
    out = seg(word, style)
    if job.state is JobState.PROVISIONING and job.provider:
        out.append(f" on {render.where(job)}")
    elif job.state is JobState.MIGRATING and job.provider:
        out.append(f" from {job.provider}")
    return out


def _right_col(job: JobView, now: float) -> Text:
    if job.state is JobState.AWAITING_APPROVAL:
        return Text(f"/approve {job.short_id}", style="dim")
    took = render.elapsed(job, now)
    if job.state in (JobState.RUNNING, JobState.CHECKPOINTING) and took is not None:
        return Text(render.clock_time(took))
    return Text(render.clock_time(now - job.created_at), style="dim")


def columns(jobs: Sequence[JobView], width: int) -> Cols:
    sid = max((cell_len(j.short_id) for j in jobs), default=4)
    name = min(NAME_MAX, max((cell_len(j.name) for j in jobs), default=8))
    mid = max(
        (_mid(j).cell_len for j in jobs if j.state is not JobState.AWAITING_APPROVAL), default=0
    )
    # 3 icon + "job " + sid + gaps + right column (~10) must fit: shrink name, then mid
    budget = width - (3 + 4 + sid + len(GAP) * 3 + 10)
    mid = min(mid, max(12, budget - name))
    name = max(8, min(name, budget - mid))
    return Cols(sid=sid, name=name, mid=mid)


def job_line1(job: JobView, cols: Cols, width: int, now: float) -> Text:
    line = icon_cell(job.state)
    line.append("job ", style="dim")
    line.append(pad(job.short_id, cols.sid), style="bold")
    line.append(GAP)
    line.append(pad(job.name, cols.name))
    line.append(GAP)
    mid = _mid(job)
    if job.state is JobState.AWAITING_APPROVAL:
        line.append_text(mid)  # the state word may run past the column; nothing follows
    else:
        mid.truncate(cols.mid, overflow="ellipsis", pad=True)
        line.append_text(mid)
    return _right(line, _right_col(job, now), width)


def _ckpt(job: JobView, where: str | None, now: float) -> Text | None:
    if job.state is JobState.CHECKPOINTING:
        return Text("saving checkpoint…", style="dim")
    if not job.checkpoint_count or job.last_checkpoint_at is None:
        return None
    out = Text()
    _label(out, "ckpt", f"{render.duration(now - job.last_checkpoint_at)} ago")
    if where:
        out.append(f" → {where}", style="dim")
    return out


def _progress(job: JobView, metric: JobMetric | None) -> Text | None:
    step = job.progress.step if job.progress.step is not None else (metric and metric.step)
    total = job.progress.total or (metric.total if metric else None)
    if step is None:
        return None
    out = Text()
    _label(out, "step", f"{step}/{total}" if total else str(step))
    return out


def _metric(job: JobView, metric: JobMetric | None, spark_w: int) -> Text | None:
    name = metric.name if metric and metric.name else None
    values: Sequence[float] = metric.values if metric else ()
    if name is None:
        if not job.last_metrics:
            return None
        name = render_primary(job.last_metrics)
        if name is None:
            return None
    latest = values[-1] if values else job.last_metrics.get(name)
    if latest is None:
        return None
    out = Text()
    _label(out, name, fmt_value(latest))
    spark = sparkline(values, spark_w)
    if spark:
        out.append(f" {spark}")
    arrow = trend(values)
    if arrow:
        out.append(f" {arrow}")
    return out


def render_primary(metrics: dict[str, float]) -> str | None:
    from gpu_router.shell.metrics import primary_metric

    return primary_metric(metrics)


def _join(parts: Iterable[Text | None], sep: str = "   ") -> Text:
    out = Text()
    for p in parts:
        if p is None or not p.plain:
            continue
        if out.plain:
            out.append(sep)
        out.append_text(p)
    return out


def job_line2(
    job: JobView, metric: JobMetric | None, where: str | None, width: int, now: float
) -> Text | None:
    """The detail line under a job, or None when there is nothing worth a row."""
    avail = width - len(INDENT)
    if job.state in (JobState.RUNNING, JobState.CHECKPOINTING):
        prog = _progress(job, metric)
        ckpt = _ckpt(job, where, now)
        for spark_w, with_ckpt in ((SPARK_W, True), (6, True), (6, False), (0, False)):
            body = _join([prog, _metric(job, metric, spark_w), ckpt if with_ckpt else None])
            if body.cell_len <= avail:
                break
        if not body.plain:
            body = Text(f"no metrics yet · /watch {job.short_id} once it logs loss=…", "dim")
    elif job.state is JobState.AWAITING_APPROVAL:
        body = Text(style="dim")
        target = " ".join(x for x in (job.provider, job.gpu) if x)
        why = shell_words(job.route_reason or "")
        if job.provider and why.startswith(f"{job.provider}: "):
            why = why[len(job.provider) + 2 :]
        body.append(f"route → {target or 'the first free provider'}")
        if why:
            body.append(f" · {why}" if "(" in why else f" ({why})")
        rule = (job.approval_reason or "").split(" · ")[0]
        if rule and body.cell_len + 3 + cell_len(rule) <= avail:
            body.append(f" · {rule}")
    elif job.message:
        body = Text(shell_words(job.message), style="dim")
    else:
        return None
    line = Text(INDENT)
    line.append_text(body)
    line.truncate(width, overflow="ellipsis")
    return line


def recent_line(job: JobView, cols: Cols, width: int, now: float) -> Text:
    """One row for a job that finished recently (✓ done, ✗ failed, ✗ cancelled...)."""
    line = icon_cell(job.state, migrated=False)
    line.append("job ", style="dim")
    line.append(pad(job.short_id, cols.sid), style="bold")
    line.append(GAP)
    line.append(pad(job.name, cols.name))
    line.append(GAP)
    _, style = render.state_style(job.state)
    right = Text(render.ago(job.finished_at, now), style="dim")
    if job.state is JobState.DONE:
        line.append("done", style=style)
        took = render.elapsed(job, now)
        if took is not None:
            line.append(f" in {render.duration(took)}")
        if job.provider:
            line.append(f" on {job.provider}")
        if job.outputs_dir:
            right = Text(f"→ {render.rel_path(job.outputs_dir)}", style="dim")
    else:
        line.append(render.state_label(job.state), style=style)
        if job.message and job.state is JobState.FAILED:
            line.append(f": {shell_words(job.message)}", style="dim")
    return _right(line, right, width)


def notice_line(job: JobView, now: float) -> Text:
    """A finished job, for the transcript: `✓ job a7f2 train.py done in 3m12s on kaggle →
    ./runs/a7f2` (no column padding: it is a sentence, not a table row)."""
    line = icon_cell(job.state)
    line.append("job ", style="dim")
    line.append(job.short_id, style="bold")
    line.append(f" {job.name} ")
    _, style = render.state_style(job.state)
    if job.state is JobState.DONE:
        line.append("done", style=style)
        took = render.elapsed(job, now)
        if took is not None:
            line.append(f" in {render.duration(took)}")
        if job.provider:
            line.append(f" on {render.where(job)}")
        if job.outputs_dir:
            line.append(f" → {render.rel_path(job.outputs_dir)}", style="dim")
        return line
    line.append(render.state_label(job.state), style=style)
    message = shell_words(job.message or "")
    if message:
        line.append(f": {message}", style="dim")
    if job.state is JobState.FAILED and f"/logs {job.short_id}" not in message:
        line.append(f"  /logs {job.short_id}", style="dim")
    return line


def sort_active(jobs: Iterable[JobView]) -> list[JobView]:
    return sorted(jobs, key=lambda j: (_ORDER.get(j.state, 9), j.created_at))


# --------------------------------------------------------------------------- panel


def panel_lines(
    snap: Snapshot, width: int, max_rows: int, *, now: float, example: str
) -> list[Text]:
    """Every line inside the panel border for this snapshot (never more than max_rows)."""
    conn = snap.conn
    if conn.status is not ConnStatus.UP and not snap.active and not snap.recent:
        return conn_lines(conn, width)
    active = sort_active(snap.active)
    recent = list(snap.recent)[:3]
    if not active and not recent:
        return empty_lines(snap.providers, width, now=now, example=example)
    cols = columns([*active, *recent], width)
    rows: list[tuple[JobView, Text, Text | None]] = []
    for job in active:
        l2 = job_line2(job, snap.metrics.get(job.id), snap.ckpt_where.get(job.id), width, now)
        rows.append((job, job_line1(job, cols, width, now), l2))
    full = sum(1 + (l2 is not None) for _, _, l2 in rows)
    out: list[Text] = []
    if full <= max_rows:
        for _, l1, l2 in rows:
            out.append(l1)
            if l2 is not None:
                out.append(l2)
        for job in recent:
            if len(out) >= max_rows:
                break
            out.append(recent_line(job, cols, width, now))
        return out
    # too many: one row per job, approvals always shown, then a summary of the rest
    keep = max(1, max_rows - 1)
    chosen = [j for j, _, _ in rows if j.state is JobState.AWAITING_APPROVAL][:keep]
    for j, _, _ in rows:
        if len(chosen) >= keep:
            break
        if j not in chosen:
            chosen.append(j)
    shown = [r for r in rows if r[0] in chosen]
    out = [l1 for _, l1, _ in shown]
    rest = [j for j, _, _ in rows if j not in chosen]
    if rest:
        out.append(overflow_line(rest, width))
    return out


def overflow_line(rest: Sequence[JobView], width: int) -> Text:
    counts: dict[str, int] = {}
    for j in rest:
        if j.state in (JobState.RUNNING, JobState.CHECKPOINTING):
            word = "running"
        elif j.state is JobState.AWAITING_APPROVAL:
            word = "need approval"  # "+3 more (1 running, 2 need approval)"
        else:
            word = _WAIT_WORD.get(j.state, str(j.state)).split(" ")[0]
        counts[word] = counts.get(word, 0) + 1
    detail = ", ".join(f"{n} {w}" for w, n in counts.items())
    line = Text(f"{INDENT}+{len(rest)} more ({detail}) · /jobs lists them all", style="dim")
    line.truncate(width, overflow="ellipsis")
    return line


def conn_lines(conn: Conn, width: int) -> list[Text]:
    if conn.status is ConnStatus.CONNECTING:
        return [Text("connecting to the gpu-router daemon…", style="dim")]
    if conn.status is ConnStatus.STARTING:
        return [Text("the daemon is not running; starting it in the background…", style="dim")]
    head = seg(f"{render.ICON_FAILED}  ", "red")
    head.append(conn.message or "cannot reach the gpu-router daemon")
    lines = [head]
    if conn.hint:
        lines.append(Text(f"{INDENT}{conn.hint}", style="dim"))
    retry = f"retrying every {conn.retry_s:g}s · " if conn.retry_s else ""
    start = (
        "any command tries to start it"
        if conn.autostart
        else "auto-start is off (GPU_ROUTER_NO_AUTOSTART)"
    )
    lines.append(Text(f"{INDENT}{retry}{start} · /doctor checks everything", "dim"))
    for t in lines:
        t.truncate(width, overflow="ellipsis")
    return lines


def left_text(p: ProviderView, now: float) -> Text:
    """Quota left for the empty state: '8h left ↻Sat', else the health word."""
    q = p.quota
    if (
        q is not None
        and q.limit is not None
        and p.health
        in (
            ProviderHealth.OK,
            ProviderHealth.UNKNOWN,
        )
    ):
        unit = _UNIT.get(str(q.unit), f" {q.unit}")
        money = "$" if str(q.unit) == "usd" else ""
        est = "~" if q.source == "estimate" else ""
        left = f"{est}{money}{render.fmt_num(max(0.0, q.limit - q.used))}{unit} left"
        reset = render.reset_label(q.resets_at, now)
        return Text(left + (f" {reset}" if reset else ""))
    return render.health_text(p.health)


def empty_lines(
    providers: Sequence[ProviderView], width: int, *, now: float, example: str
) -> list[Text]:
    """Spec UX 7: no jobs shows quota left plus an example /run, never a blank panel.
    One provider per row (name column, then what is left), like `gpu status`'s empty
    state; more than EMPTY_TABLE_MAX providers flow several to a row instead."""
    label_w = 15
    lines = [Text("no jobs running", style="dim")]
    items = [(p.name, left_text(p, now)) for p in providers if p.enabled]
    if not items:
        row = seg(pad("free GPU time", label_w), "dim")
        row.append("no providers connected yet · /providers", style="dim")
        lines.append(row)
    elif len(items) <= EMPTY_TABLE_MAX:
        name_w = max(cell_len(n) for n, _ in items) + 2
        for i, (name, value) in enumerate(items):
            row = seg(pad("free GPU time" if i == 0 else "", label_w), "dim")
            row.append(pad(name, name_w))
            row.append_text(value)
            lines.append(row)
    else:
        row = seg(pad("free GPU time", label_w), "dim")
        for i, (name, value) in enumerate(items):
            piece = Text(name + " ")
            piece.append_text(value)
            sep = "   " if i else ""
            if row.cell_len + len(sep) + piece.cell_len > width and i:
                lines.append(row)
                row = Text(" " * label_w)
                sep = ""
            row.append(sep)
            row.append_text(piece)
        lines.append(row)
    run = f"/run {example}"
    route = f"/route {example}"
    cmd_w = max(cell_len(run), cell_len(route)) + 3
    first = seg(pad("try", label_w), "dim")
    first.append(pad(run, cmd_w))
    first.append("run it on the best free GPU", style="dim")
    second = Text(" " * label_w)
    second.append(pad(route, cmd_w))
    second.append("see where it would go and why", style="dim")
    lines += [first, second]
    for t in lines:
        t.truncate(width, overflow="ellipsis")
    return lines


# --------------------------------------------------------------------------- footer


def quota_short(q: QuotaSnapshot, now: float, *, resets: bool = True) -> str:
    """22/30h ↻Sat, ~3/15 cr, $24/30 (used/limit; ~ = estimated)."""
    unit = _UNIT.get(str(q.unit), f" {q.unit}")
    money = "$" if str(q.unit) == "usd" else ""
    est = "~" if q.source == "estimate" else ""
    if q.limit is None:
        body = f"{est}{money}{render.fmt_num(q.used)}{unit} used"
    else:
        body = f"{est}{money}{render.fmt_num(q.used)}/{render.fmt_num(q.limit)}{unit}"
    reset = render.reset_label(q.resets_at, now) if resets else ""
    return body + (f" {reset}" if reset else "")


def _provider_piece(p: ProviderView, now: float, *, resets: bool) -> Text:
    out = Text(p.name + " ")
    healthy = p.health in (ProviderHealth.OK, ProviderHealth.UNKNOWN)
    if healthy and p.quota is not None and p.quota.limit is not None:
        out.append(quota_short(p.quota, now, resets=resets), style="dim")
    elif p.health is ProviderHealth.UNKNOWN:
        out.append("○ unchecked", style="dim")
    else:
        out.append_text(render.health_text(p.health))
    return out


def footer_text(snap: Snapshot, width: int, *, now: float) -> Text:
    """`kaggle 22/30h ↻Sat │ colab ● up │ lightning ~3/15 cr │ 1 running` (enabled providers
    only: manual, verify-at-signup and excluded ones such as modal never show here),
    shortened (resets dropped, then providers folded into +N) to fit `width`."""
    sep = Text(" │ ", style="dim")
    conn = snap.conn
    if conn.status is ConnStatus.DOWN:
        out = seg(f"{render.ICON_FAILED} ", "red")
        out.append("daemon down")
        out.append_text(sep)
        out.append(f"retrying every {conn.retry_s:g}s" if conn.retry_s else "not retrying", "dim")
        return out
    if conn.status in (ConnStatus.CONNECTING, ConnStatus.STARTING):
        return Text(
            "connecting…" if conn.status is ConnStatus.CONNECTING else "starting daemon…",
            style="dim",
        )
    tail = Text(
        f"{snap.running} running" if snap.running else "idle", style="" if snap.running else "dim"
    )
    if conn.last_ok is not None and now - conn.last_ok > 10:
        tail.append(" · ", style="dim")
        tail.append(f"stale {render.duration(now - conn.last_ok)}", style="yellow")
    enabled = [p for p in snap.providers if p.enabled]
    for resets in (True, False):
        pieces = [_provider_piece(p, now, resets=resets) for p in enabled]
        for keep in range(len(pieces), -1, -1):
            out = Text()
            for piece in pieces[:keep]:
                out.append_text(piece)
                out.append_text(sep)
            hidden = len(pieces) - keep
            if hidden:
                out.append(f"+{hidden}", style="dim")
                out.append_text(sep)
            out.append_text(tail)
            if out.cell_len <= width:
                return out
            if resets and keep == len(pieces):
                break  # first try again without reset labels before folding providers
    out = tail.copy()
    out.truncate(width, overflow="ellipsis")
    return out
