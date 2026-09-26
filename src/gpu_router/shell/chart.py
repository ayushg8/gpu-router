"""The /watch chart (phase 4): one metric as an eighth-block area chart. Pure formatting.

One series, so no legend (the title names it); one y axis with its top and bottom value in
dim ink; the latest value labelled directly in the title; no colour (spec UX 6: colour is
for job state only). The value edge is plain ink and the area under it dim, so the mark
reads as a line; a 10-row chart has 80 levels. The whole run spans the width (bucket
means when there are more points than columns, interpolation when fewer). Braille was
tried first: in real fonts its dots render as a faint dotted trail.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from rich.text import Text

from gpu_router.shell.metrics import fmt_value, sparkline, trend


def resample(values: Sequence[float], width: int) -> list[float]:
    """At most `width` points: the raw values when they fit, else the mean of each of
    `width` equal buckets (the whole run stays visible, oldest on the left)."""
    vals = [v for v in values if math.isfinite(v)]
    if width <= 0 or not vals:
        return []
    if len(vals) <= width:
        return vals
    out: list[float] = []
    n = len(vals)
    for i in range(width):
        a = i * n // width
        b = max(a + 1, (i + 1) * n // width)
        chunk = vals[a:b]
        out.append(sum(chunk) / len(chunk))
    return out


EIGHTHS = " ▁▂▃▄▅▆▇█"  # index = eighths of a cell filled from the bottom
FILL = "grey30"  # the area under the edge: recessive grey, not a hue (spec UX 6)


def stretch(values: Sequence[float], width: int) -> list[float]:
    """Exactly `width` points: bucket means when there are more values, linear
    interpolation when there are fewer (the run always spans the whole chart)."""
    pts = resample(values, width)
    n = len(pts)
    if n == 0 or n >= width:
        return pts
    if n == 1:
        return pts * width
    out = []
    for x in range(width):
        pos = x * (n - 1) / (width - 1)
        i = min(n - 2, int(pos))
        out.append(pts[i] + (pts[i + 1] - pts[i]) * (pos - i))
    return out


def area_rows(values: Sequence[float], width: int, height: int) -> tuple[list[Text], float, float]:
    """(rows top to bottom, y min, y max): an area chart in eighth blocks whose top edge
    (the value) is drawn in plain ink and the fill under it dim, so the curve reads as a
    line with a quiet area below it."""
    pts = stretch(values, width)
    if not pts or width <= 0 or height <= 0:
        return [Text(" " * max(0, width)) for _ in range(max(0, height))], 0.0, 0.0
    lo, hi = min(pts), max(pts)
    if hi - lo <= abs(hi) * 1e-9:
        pad = abs(hi) * 0.5 or 1.0
        lo, hi = lo - pad, hi + pad
    levels = height * 8
    heights = [max(1, round((v - lo) / (hi - lo) * levels)) for v in pts]
    tops = [(h - 1) // 8 for h in heights]  # row index from the bottom holding the edge
    rows: list[Text] = []
    for r in range(height):
        level = height - 1 - r  # this row's index from the bottom
        row = Text()
        for h, top in zip(heights, tops, strict=True):
            fill = min(8, max(0, h - level * 8))
            row.append(EIGHTHS[fill], style="" if level == top else FILL)
        rows.append(row)
    return rows, lo, hi


def chart_lines(
    name: str,
    values: Sequence[float],
    *,
    width: int,
    height: int,
    step: int | None = None,
    total: int | None = None,
    first_step: int | None = None,
) -> list[Text]:
    """Title, `height` chart rows with a left y axis, and an x axis line."""
    if not values:
        return [
            Text(f"{name}  no points yet", style="dim"),
            Text(
                "waiting for the job to log metrics (gpu.log(step=i, loss=l) or loss=… lines)",
                style="dim",
            ),
        ]
    title = Text()
    title.append(name, style="bold")
    title.append(f"  {fmt_value(values[-1])}")
    arrow = trend(values)
    if arrow:
        title.append(f" {arrow}")
    if step is not None:
        title.append(f"   step {step}" + (f"/{total}" if total else ""), style="dim")
    title.append(f"   min {fmt_value(min(values))} · max {fmt_value(max(values))}", style="dim")
    title.append(f" · {len(values)} points", style="dim")
    title.truncate(width, overflow="ellipsis")
    axis_w = max(len(fmt_value(max(values))), len(fmt_value(min(values)))) + 1
    plot_w = max(8, width - axis_w - 2)
    rows, lo, hi = area_rows(values, plot_w, height)
    out = [title]
    for i, row in enumerate(rows):
        label = fmt_value(hi) if i == 0 else fmt_value(lo) if i == height - 1 else ""
        line = Text()
        line.append(label.rjust(axis_w), style="dim")
        line.append(" ┤" if label else " │", style="dim")
        line.append_text(row)
        out.append(line)
    base = Text(" " * axis_w, style="dim")
    base.append(" └" + "─" * plot_w, style="dim")
    out.append(base)
    start = f"step {first_step}" if first_step is not None else "start"
    end = f"step {step}" if step is not None else f"{len(values)} points"
    xl = Text(" " * (axis_w + 2) + start, style="dim")
    gap = plot_w - len(start) - len(end)
    if gap >= 2:
        xl.append(" " * gap + end, style="dim")
    out.append(xl)
    return out


def mini_line(name: str, values: Sequence[float], width: int) -> Text:
    """`val_loss 0.52 ▇▆▅▄▃ ↓` for the secondary metrics under the chart."""
    out = Text()
    out.append(f"{name} ", style="dim")
    out.append(fmt_value(values[-1]) if values else "-")
    spark = sparkline(values, max(0, min(24, width - out.cell_len - 4)))
    if spark:
        out.append(f" {spark}")
    arrow = trend(values)
    if arrow:
        out.append(f" {arrow}")
    return out
