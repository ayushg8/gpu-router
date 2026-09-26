"""Log capture for one attempt (phase 1; owner: group C).

The driver calls `LogCapture.ingest(chunks)` after each `AdapterCaller.logs()`; capture:

1. redacts every line (secrets.redact) and appends it to
   paths.job_log(job_id, n) (the log file is the truth; attempts.log_lines is a cache);
2. parses each line with protocol.parse_line; non-protocol lines go through
   protocol.StdoutMetricParser, whose results are used only while no protocol metric has
   been seen for this job (progress_source "helper" beats "stdout");
3. appends parsed metric points to paths.job_metrics(job_id) as JSON lines
   {"ts", "attempt", "step", "metrics"};
4. returns a `CaptureResult` the driver turns into store writes on the event-loop thread:
   JobPatch(progress_*, last_metrics), checkpoint begin/end events, exit code, the GPUs
   the runner saw (`device` line, D56: the driver notes one that differs from the GPU the
   attempt was placed on, see `gpu_mismatch`), and the new log cursor + line count for
   AttemptPatch(log_cursor, log_lines).

Order matters for crash safety: lines are appended and fsynced BEFORE the driver persists the
new cursor, so a crash can duplicate at most one chunk in the file, never lose lines. On
reattach, `LogCapture.open()` truncates the file to the persisted `log_lines` count so the
duplicate is dropped.

File writes happen on the event-loop thread; they are small appends to a local file.
Embedded newlines / carriage-return-only progress lines inside one adapter line are kept as
one captured line (newlines replaced by spaces) so line indices stay stable.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from io import BufferedWriter
from pathlib import Path

from gpu_router.adapters.base import LogChunk
from gpu_router.protocol import ProtocolEvent, StdoutMetricParser, is_protocol_line, parse_line
from gpu_router.secrets import redact


@dataclass(slots=True)
class CaptureResult:
    lines_added: int = 0
    cursor: str | None = None  # cursor after the last ingested chunk
    step: int | None = None
    total: int | None = None
    progress_source: str | None = None  # "helper" | "stdout"
    metrics: dict[str, float] = field(default_factory=dict)  # latest value per name
    checkpoint_events: list[ProtocolEvent] = field(default_factory=list)  # ckpt_begin/end
    exit_code: int | None = None
    devices: tuple[str, ...] = ()  # the runner's nvidia-smi view, when it reported one
    eof: bool = False

    @property
    def has_progress(self) -> bool:
        return self.step is not None or self.total is not None or bool(self.metrics)


_TOKEN = re.compile(r"[^A-Z0-9]+")
_COUNT = re.compile(r"^\s*(\d+)\s*[xX]\s*(.+)$")


def gpu_mismatch(placed: str | None, seen: Sequence[str]) -> str | None:
    """What the runner saw, when it is not the GPU the attempt was placed on (`placed` is
    a catalog label such as "T4", "2xT4", "L4", "A100-40GB"); None when it matches or
    either side is unknown. A match needs every token of the placed model among the tokens
    of each seen GPU name ("Tesla T4" is a T4, "NVIDIA L40S" is not an L4) and at least as
    many GPUs as placed."""
    names = [s.split(",", 1)[0].strip() for s in seen if s.strip()]
    if not placed or not names:
        return None
    count, model = 1, placed.strip()
    m = _COUNT.match(model)
    if m is not None:
        count, model = int(m.group(1)), m.group(2)
    want = {t for t in _TOKEN.split(model.upper()) if t}
    if not want:
        return None
    ok = all(want <= {t for t in _TOKEN.split(n.upper()) if t} for n in names)
    if ok and len(names) >= count:
        return None
    kinds = list(dict.fromkeys(names))
    label = kinds[0] if len(kinds) == 1 else " + ".join(kinds)
    return f"{len(names)}x {label}" if len(names) > 1 and len(kinds) == 1 else label


def _clean(line: str) -> str:
    line = line.rstrip("\r\n")
    if "\n" in line:
        line = line.replace("\r\n", " ").replace("\n", " ")
    return line


class LogCapture:
    def __init__(
        self, job_id: str, attempt_n: int, log_path: Path, metrics_path: Path, *, helper_seen: bool
    ) -> None:
        """`helper_seen`: the job already reported protocol metrics (progress_source ==
        "helper"), so stdout parsing stays off after a restart."""
        self.job_id = job_id
        self.attempt_n = attempt_n
        self.log_path = log_path
        self.metrics_path = metrics_path
        self.helper_seen = helper_seen
        self._parser = StdoutMetricParser()
        self._fh: BufferedWriter | None = None
        self.lines = 0

    def open(self, persisted_lines: int) -> None:
        """Create parent dirs; truncate the log file to `persisted_lines` lines (drops a
        chunk written before a crash but whose cursor was never persisted)."""
        self.log_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.metrics_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        keep = max(0, persisted_lines)
        if self.log_path.exists():
            offset = 0
            count = 0
            with self.log_path.open("rb") as fh:
                while count < keep:
                    raw = fh.readline()
                    if not raw:
                        break
                    offset += len(raw)
                    count += 1
            if self.log_path.stat().st_size != offset:
                with self.log_path.open("r+b") as fh:
                    fh.truncate(offset)
            self.lines = count
        else:
            self.lines = 0
        self._fh = self.log_path.open("ab")

    def ingest(self, chunks: list[LogChunk], *, now: float) -> CaptureResult:
        if self._fh is None:
            raise RuntimeError("LogCapture.open() must be called before ingest()")
        result = CaptureResult()
        buf: list[bytes] = []
        points: list[str] = []
        for chunk in chunks:
            result.cursor = chunk.cursor
            result.eof = result.eof or chunk.eof
            for raw in chunk.lines:
                line = redact(_clean(raw))
                buf.append(line.encode("utf-8", "replace") + b"\n")
                result.lines_added += 1
                point = self._parse(line, result)
                if point is not None:
                    points.append(
                        json.dumps(
                            {
                                "ts": now,
                                "attempt": self.attempt_n,
                                "step": point[0],
                                "metrics": point[1],
                            },
                            separators=(",", ":"),
                        )
                    )
        if buf:
            self._fh.write(b"".join(buf))
            self._fh.flush()
            os.fsync(self._fh.fileno())
            self.lines += result.lines_added
        if points:
            with self.metrics_path.open("a", encoding="utf-8") as mf:
                mf.write("\n".join(points) + "\n")
        return result

    def _parse(
        self, line: str, result: CaptureResult
    ) -> tuple[int | None, dict[str, float]] | None:
        """Update `result` from one line; return a metric point to record, if any."""
        if is_protocol_line(line):
            ev = parse_line(line)
            if ev is None:
                return None
            if ev.t == "total":
                self.helper_seen = True
                result.total = ev.total
                result.progress_source = "helper"
            elif ev.t == "metric":
                self.helper_seen = True
                result.progress_source = "helper"
                if ev.step is not None:
                    result.step = ev.step
                if ev.total is not None:
                    result.total = ev.total
                result.metrics.update(ev.metrics)
                if ev.metrics:
                    return ev.step, dict(ev.metrics)
            elif ev.t in ("ckpt_begin", "ckpt_end"):
                result.checkpoint_events.append(ev)
            elif ev.t == "exit":
                result.exit_code = ev.code
            elif ev.t == "device":
                result.devices = ev.gpus
            return None
        if self.helper_seen:
            return None
        sm = self._parser.feed(line)
        if sm.empty:
            return None
        result.progress_source = "stdout"
        if sm.step is not None:
            result.step = sm.step
        if sm.total is not None:
            result.total = sm.total
        result.metrics.update(sm.metrics)
        if sm.metrics:
            return sm.step, dict(sm.metrics)
        return None

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            finally:
                self._fh = None


def read_log_lines(
    log_path: Path, *, offset: int = 0, include_protocol: bool = False
) -> list[tuple[int, str]]:
    """(line index, line) pairs of a captured attempt log from line index `offset` (used by
    the logs API). Indices count every line in the file, protocol lines included, so a
    client can resume with offset = last index + 1; protocol lines are only omitted from
    the output unless include_protocol."""
    out: list[tuple[int, str]] = []
    try:
        fh = log_path.open("rb")
    except FileNotFoundError:
        return out
    with fh:
        for i, raw in enumerate(fh):
            if i < offset:
                continue
            if not raw.endswith(b"\n"):
                break  # a line still being written; it is returned on the next read
            line = raw[:-1].decode("utf-8", "replace")
            if not include_protocol and is_protocol_line(line):
                continue
            out.append((i, line))
    return out
