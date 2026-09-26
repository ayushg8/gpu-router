"""Metric history for the job panel sparkline and /watch (phase 4).

The daemon keeps only a job's latest metrics (`JobView.last_metrics`); the history lives in
its captured log. The shell reads that log through the API with `protocol=true` and parses
it the way the engine does (engine/capture.py): `::gpu::` metric lines from the helper, and
the stdout fallback parser only while no helper metric has been seen for the job (D42).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from gpu_router.protocol import StdoutMetricParser, is_protocol_line, parse_line

SPARK = "▁▂▃▄▅▆▇█"
#: Which metric the panel shows when a job reports several.
PRIMARY = ("loss", "train_loss", "val_loss", "acc", "accuracy")
#: Points kept per metric; older points are thinned (every other one dropped), so the
#: whole run keeps its shape at half the resolution instead of losing its start.
MAX_POINTS = 4000


@dataclass
class MetricHistory:
    """Metric points of one job, fed log line by log line (all attempts, in order)."""

    series: dict[str, list[tuple[int | None, float]]] = field(default_factory=dict)
    step: int | None = None
    total: int | None = None
    helper: bool = False  # a `::gpu::` metric line was seen: stdout results are ignored
    positions: dict[int, int] = field(default_factory=dict)  # attempt n -> next log offset
    _stdout: StdoutMetricParser = field(default_factory=StdoutMetricParser, repr=False)

    def feed(self, line: str) -> bool:
        """Parse one log line; True when it added a point or changed progress."""
        if is_protocol_line(line):
            ev = parse_line(line)
            if ev is None:
                return False
            if ev.t == "total" and ev.total:
                self.total = ev.total
                return True
            if ev.t != "metric":
                return False
            if not self.helper:
                self.helper = True
                self.series.clear()  # helper beats stdout: drop what the fallback guessed
            if ev.total:
                self.total = ev.total
            return self._add(ev.step, ev.metrics)
        if self.helper:
            return False
        got = self._stdout.feed(line)
        if got.empty:
            return False
        if got.total:
            self.total = got.total
        return self._add(got.step, got.metrics) or got.step is not None

    def _add(self, step: int | None, metrics: dict[str, float]) -> bool:
        if step is not None:
            self.step = step
        added = False
        for name, value in metrics.items():
            if not math.isfinite(value):
                continue
            points = self.series.setdefault(name, [])
            points.append((step, value))
            if len(points) > MAX_POINTS:
                del points[1::2]  # every other point, but never the first (its start)
            added = True
        return added

    def values(self, name: str) -> list[float]:
        return [v for _, v in self.series.get(name, [])]

    def names(self) -> list[str]:
        return list(self.series)

    def primary(self) -> str | None:
        return primary_metric(self.series)


def primary_metric(names: Iterable[str]) -> str | None:
    keys = list(names)
    for want in PRIMARY:
        if want in keys:
            return want
    return keys[0] if keys else None


def sparkline(values: Sequence[float], width: int) -> str:
    """The last `width` values as block characters scaled to their own min..max."""
    tail = [v for v in values[-width:] if math.isfinite(v)] if width > 0 else []
    if not tail:
        return ""
    lo, hi = min(tail), max(tail)
    if hi - lo <= abs(hi) * 1e-9:
        return SPARK[3] * len(tail)
    top = len(SPARK) - 1
    return "".join(SPARK[round((v - lo) / (hi - lo) * top)] for v in tail)


def trend(values: Sequence[float]) -> str:
    """↓ falling, ↑ rising, → flat: the mean of the newest fifth (max 20 points) against
    the fifth before it, relative to the range seen. A glyph, so direction never depends
    on colour."""
    n = len(values)
    if n < 2:
        return ""
    k = max(1, min(20, n // 5))
    recent = values[-k:]
    before = values[-2 * k : -k] or values[:1]
    a = sum(before) / len(before)
    b = sum(recent) / len(recent)
    span = max(values) - min(values)
    if span <= 0 or abs(b - a) < span * 0.02:
        return "→"
    return "↓" if b < a else "↑"


def fmt_value(v: float) -> str:
    """0.412, 12.3, 1234, 0.00123 (three significant digits, no exponent in 1e-3..1e5)."""
    if v == 0:
        return "0"
    mag = abs(v)
    if 1e-3 <= mag < 1e5:
        decimals = max(0, 2 - math.floor(math.log10(mag)))
        return f"{v:.{decimals}f}"
    return f"{v:.3g}"
