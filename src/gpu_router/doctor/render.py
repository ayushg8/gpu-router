"""Human rendering of a doctor Report (phase 8a), shared by `gpu doctor` and `/doctor`.

One visual language (spec UX 6): colour only on the status mark and word; labels dim.
Marks: ✓ ok (green), ! warn (yellow), ✗ fail (red), · skip (dim). Fix commands print
verbatim (dim `fix:` + the command), so they can be copied into a terminal.
"""

from __future__ import annotations

from pathlib import Path

from rich.console import RenderableType
from rich.table import Table
from rich.text import Text

from gpu_router.doctor.model import Report, Status
from gpu_router.doctor.probe import home_label

__all__ = ["MARKS", "render_report"]

MARKS: dict[Status, tuple[str, str]] = {
    Status.OK: ("✓", "green"),
    Status.WARN: ("!", "yellow"),
    Status.FAIL: ("✗", "red"),
    Status.SKIP: ("·", "dim"),
}


def render_report(
    report: Report, *, user_home: Path | None = None, verbose: bool = False
) -> list[RenderableType]:
    """Renderables for the whole report. `verbose` adds per-check timings."""
    home = user_home or Path.home()
    out: list[RenderableType] = []
    head = Text("gpu doctor", style="bold")
    head.append(
        f"  v{report.version} · data dir {home_label(report.home, home)} · "
        f"{report.elapsed_ms / 1000:.1f}s",
        style="dim",
    )
    out.append(head)
    for group, rows in report.by_group():
        out.append(Text(""))
        out.append(Text(group, style="dim"))
        width = max(len(c.title) for c in rows)
        for c in rows:
            mark, style = MARKS[c.status]
            skip = c.status is Status.SKIP
            # one grid per row: the summary wraps under itself (hanging indent) ...
            grid = Table.grid(padding=(0, 1), expand=False)
            grid.add_column(no_wrap=True, width=3)
            grid.add_column(no_wrap=True, width=width)
            grid.add_column(overflow="fold")
            cells: list[RenderableType] = [
                Text(f"  {mark}", style=style),
                Text(c.title, style="dim" if skip else ""),
                Text(c.summary, style="dim" if skip else ""),
            ]
            if verbose:
                grid.add_column(no_wrap=True, justify="right")
                cells.append(Text(f"{c.elapsed_ms}ms", style="dim"))
            grid.add_row(*cells)
            out.append(grid)
            # ... and the fix is one unbroken line, so it copies into a terminal whole
            if c.fix and c.status is not Status.OK:
                line = Text(" " * (width + 5) + "fix: ", style="dim")
                line.append(c.fix)
                line.no_wrap = False
                out.append(line)
    if report.drift:
        out.append(Text(""))
        out.append(Text("limits drifted from providers.yaml", style="yellow"))
        for d in report.drift:
            out.append(Text(f"  {d.provider}: {d.note}"))
        out.append(
            Text(
                "  gpu doctor --update-catalog writes the live numbers to "
                f"{home_label(Path(report.home) / 'providers.yaml', home)} (shows the diff, "
                "asks first)",
                style="dim",
            )
        )
    out.append(Text(""))
    out.append(summary_line(report))
    return out


def summary_line(report: Report) -> Text:
    c = report.counts
    line = Text()
    parts: list[tuple[str, str]] = [
        (f"✓ {c.get('ok', 0)} ok", "green"),
        (f"! {c.get('warn', 0)} warn", "yellow" if c.get("warn") else "dim"),
        (f"✗ {c.get('fail', 0)} fail", "red" if c.get("fail") else "dim"),
        (f"{c.get('skip', 0)} skipped", "dim"),
    ]
    for i, (text, style) in enumerate(parts):
        if i:
            line.append(" · ", style="dim")
        line.append(text, style=style)
    if not report.ok:
        line.append("   fix the ✗ rows first; each says how", style="dim")
    elif report.unknown:
        # crashed / unfinished rows verified nothing: never "everything works"
        line.append(
            f"   {report.unknown} check(s) crashed or did not finish, so they verified "
            "nothing; see their rows",
            style="yellow",
        )
    elif c.get("warn"):
        line.append("   everything works; the ! rows are worth a look", style="dim")
    else:
        line.append("   all good", style="dim")
    return line
