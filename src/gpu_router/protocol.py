"""The `::gpu::` line protocol and the stdout metric fallback (phase 1; real code; owner: B).

Remote side (runner/bootstrap.py, runner/gpu.py helper, and the fake adapter) prints
protocol lines to stdout; the engine (engine/capture.py) parses every captured log line
with `parse_line`, and runs `StdoutMetricParser` on the rest (spec decision 3: helper
preferred, stdout parsing as zero-code-change fallback).

Line format: the prefix `::gpu:: ` followed by one compact JSON object with key "t":

    ::gpu:: {"t":"hello","v":1,"runner":"bootstrap/0.1"}
    ::gpu:: {"t":"total","steps":1000}
    ::gpu:: {"t":"metric","step":120,"metrics":{"loss":0.412,"lr":0.0003}}
    ::gpu:: {"t":"ckpt_begin","seq":3}
    ::gpu:: {"t":"ckpt_end","seq":3,"uri":"hf://...","step":1200,"size":123456,"sha256":"..."}
    ::gpu:: {"t":"exit","code":0}
    ::gpu:: {"t":"device","gpus":["Tesla T4, 15360 MiB"]}   (nvidia-smi, D56)

Unknown "t" values and malformed JSON are ignored (returned as None) so newer runners work
with older daemons. This module is stdlib-only so runner code can mirror it byte for byte.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Literal

PREFIX = "::gpu:: "
PROTOCOL_VERSION = 1

#: Metric names the daemon keeps (D48): printable ASCII without spaces, quotes or escape
#: characters, at most 64 long. Anything else is dropped: names reach the status line, the
#: shell and agents' context verbatim.
METRIC_NAME = re.compile(r"^[A-Za-z0-9_.:/@%+-]{1,64}$")
#: At most this many metrics per line (and per job, engine/driver.py): the primary ones
#: first, then the first ones reported.
MAX_METRICS = 32
PRIMARY_METRICS = ("loss", "train_loss", "val_loss", "acc", "accuracy")

EventType = Literal["hello", "total", "metric", "ckpt_begin", "ckpt_end", "exit", "device"]
#: a device line keeps at most this many GPUs, each at most DEVICE_CHARS printable chars
MAX_DEVICES = 16
DEVICE_CHARS = 120


@dataclass(frozen=True, slots=True)
class ProtocolEvent:
    t: EventType
    step: int | None = None
    total: int | None = None  # "total" events and optional on "metric"
    metrics: dict[str, float] = field(default_factory=dict)
    seq: int | None = None  # ckpt_begin / ckpt_end
    uri: str | None = None  # ckpt_end
    size: int | None = None
    sha256: str | None = None
    code: int | None = None  # exit
    version: int | None = None  # hello
    runner: str | None = None  # hello
    gpus: tuple[str, ...] = ()  # device: one "name, memory" per GPU nvidia-smi sees


# --------------------------------------------------------------------------- format


def format_event(t: EventType, **fields: Any) -> str:
    """Render one protocol line (no trailing newline). None-valued fields are dropped."""
    body: dict[str, Any] = {"t": t}
    body.update({k: v for k, v in fields.items() if v is not None})
    return PREFIX + json.dumps(body, separators=(",", ":"), sort_keys=False)


def hello(runner: str) -> str:
    return format_event("hello", v=PROTOCOL_VERSION, runner=runner)


def total(steps: int) -> str:
    return format_event("total", steps=int(steps))


def metric(step: int | None, metrics: dict[str, float], total_steps: int | None = None) -> str:
    return format_event("metric", step=step, total=total_steps, metrics=metrics)


def ckpt_begin(seq: int) -> str:
    return format_event("ckpt_begin", seq=seq)


def ckpt_end(
    seq: int,
    uri: str,
    *,
    step: int | None = None,
    size: int | None = None,
    sha256: str | None = None,
) -> str:
    return format_event("ckpt_end", seq=seq, uri=uri, step=step, size=size, sha256=sha256)


def exit_line(code: int) -> str:
    return format_event("exit", code=code)


def device(gpus: list[str]) -> str:
    return format_event("device", gpus=list(gpus))


# --------------------------------------------------------------------------- parse


def is_protocol_line(line: str) -> bool:
    return line.startswith(PREFIX)


def _int(v: Any) -> int | None:
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float) and v.is_integer():
        return int(v)
    return None


def _metrics(v: Any) -> dict[str, float]:
    out: dict[str, float] = {}
    if not isinstance(v, dict):
        return out
    for k, x in v.items():
        if not isinstance(k, str) or METRIC_NAME.match(k) is None:
            continue
        if isinstance(x, int | float) and not isinstance(x, bool):
            fx = float(x)
            if math.isfinite(fx):
                out[k] = fx
    return cap_metrics(out)


def cap_metrics(metrics: dict[str, float], limit: int = MAX_METRICS) -> dict[str, float]:
    """At most `limit` metrics: the primary ones, then the others in the order given."""
    if len(metrics) <= limit:
        return metrics
    keep = [k for k in PRIMARY_METRICS if k in metrics]
    keep += [k for k in metrics if k not in PRIMARY_METRICS][: max(0, limit - len(keep))]
    return {k: metrics[k] for k in metrics if k in set(keep[:limit])}


def merge_metrics(
    current: dict[str, float], new: dict[str, float], limit: int = MAX_METRICS
) -> dict[str, float]:
    """A job's latest value per metric (`jobs.last_metrics`): known names are updated,
    new ones added while there is room (a primary one takes the place of the newest
    other), so the dict stays at most `limit` names however many a job reports."""
    out = dict(current)
    for k, x in new.items():
        if k in out or len(out) < limit:
            out[k] = x
        elif k in PRIMARY_METRICS:
            others = [n for n in out if n not in PRIMARY_METRICS]
            if others:
                del out[others[-1]]
                out[k] = x
    return out


def parse_line(line: str) -> ProtocolEvent | None:
    """Parse one captured line. None if it is not a (valid, known) protocol line."""
    if not line.startswith(PREFIX):
        return None
    try:
        body = json.loads(line[len(PREFIX) :])
    except (ValueError, RecursionError):
        return None
    if not isinstance(body, dict):
        return None
    t = body.get("t")
    if t == "hello":
        return ProtocolEvent(
            t="hello",
            version=_int(body.get("v")),
            runner=str(body["runner"]) if "runner" in body else None,
        )
    if t == "total":
        steps = _int(body.get("steps"))
        return ProtocolEvent(t="total", total=steps) if steps is not None and steps > 0 else None
    if t == "metric":
        return ProtocolEvent(
            t="metric",
            step=_int(body.get("step")),
            total=_int(body.get("total")),
            metrics=_metrics(body.get("metrics")),
        )
    if t == "ckpt_begin":
        seq = _int(body.get("seq"))
        return ProtocolEvent(t="ckpt_begin", seq=seq) if seq is not None and seq >= 1 else None
    if t == "ckpt_end":
        seq = _int(body.get("seq"))
        uri = body.get("uri")
        if seq is None or seq < 1 or not isinstance(uri, str) or not uri:
            return None
        sha = body.get("sha256")
        return ProtocolEvent(
            t="ckpt_end",
            seq=seq,
            uri=uri,
            step=_int(body.get("step")),
            size=_int(body.get("size")),
            sha256=sha if isinstance(sha, str) else None,
        )
    if t == "exit":
        code = _int(body.get("code"))
        return ProtocolEvent(t="exit", code=code) if code is not None else None
    if t == "device":
        raw = body.get("gpus")
        if not isinstance(raw, list):
            return None
        gpus = tuple(
            name
            for name in (
                "".join(ch for ch in str(g) if ch.isprintable())[:DEVICE_CHARS].strip()
                for g in raw[:MAX_DEVICES]
                if isinstance(g, str)
            )
            if name
        )
        return ProtocolEvent(t="device", gpus=gpus) if gpus else None
    return None


# --------------------------------------------------------------------------- stdout fallback

_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_METRIC_RE = re.compile(
    rf"(?<![A-Za-z0-9_])(?P<name>loss|val_loss|train_loss|acc|accuracy|val_acc|lr|"
    rf"learning_rate|perplexity|ppl)\s*[=:]\s*(?P<val>{_NUM})",
    re.IGNORECASE,
)
_STEP_OF_RE = re.compile(
    r"(?<![A-Za-z])(?:step|iter|iteration)\s*[=:]?\s*(\d+)\s*/\s*(\d+)", re.IGNORECASE
)
_STEP_RE = re.compile(
    r"(?<![A-Za-z])(?:step|iter|iteration)\s*[=:]?\s*(\d+)(?!\s*/)", re.IGNORECASE
)
_EPOCH_RE = re.compile(r"(?<![A-Za-z])epoch\s*[=:]?\s*(\d+)\s*/\s*(\d+)", re.IGNORECASE)
_TQDM_RE = re.compile(r"(\d{1,3})%\|[^|]*\|\s*(\d+)/(\d+)")


@dataclass(frozen=True, slots=True)
class StdoutMetrics:
    step: int | None = None
    total: int | None = None
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def empty(self) -> bool:
        return self.step is None and self.total is None and not self.metrics


class StdoutMetricParser:
    """Best-effort metric extraction from plain training output (no code changes needed).

    Recognises `loss=0.41`, `loss: 0.41`, `step 10/100`, `step=10`, `Epoch 3/10` and tqdm
    bars `45%|####      | 45/100 [...]`. Precedence for progress within one line:
    tqdm > step N/M > epoch N/M > bare step. Stateful only to remember the last total so a
    later bare `step=11` still has a total. A tqdm carriage-return line is split on '\\r'
    and the last segment wins.
    """

    def __init__(self) -> None:
        self.last_total: int | None = None

    def feed(self, line: str) -> StdoutMetrics:
        if line.startswith(PREFIX):
            return StdoutMetrics()
        if "\r" in line:
            line = line.rsplit("\r", 1)[-1] or line.replace("\r", "")
        metrics: dict[str, float] = {}
        for m in _METRIC_RE.finditer(line):
            try:
                v = float(m.group("val"))
            except ValueError:
                continue
            if math.isfinite(v):
                metrics[m.group("name").lower()] = v
        step: int | None = None
        tot: int | None = None
        tq = _TQDM_RE.search(line)
        so = None if tq else _STEP_OF_RE.search(line)
        ep = None if tq or so else _EPOCH_RE.search(line)
        if tq is not None:
            step, tot = int(tq.group(2)), int(tq.group(3))
        elif so is not None:
            step, tot = int(so.group(1)), int(so.group(2))
        elif ep is not None:
            step, tot = int(ep.group(1)), int(ep.group(2))
        elif (bare := _STEP_RE.search(line)) is not None:
            step = int(bare.group(1))
        if tot is not None and tot > 0:
            self.last_total = tot
        elif step is not None:
            tot = self.last_total
        return StdoutMetrics(step=step, total=tot, metrics=metrics)
