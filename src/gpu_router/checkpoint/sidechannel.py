"""Near-live logs from checkpoint storage for providers without live logs (phase 5).

The runner pushes `jobs/<job>/attempts/<n>/log-tail.json` every GPU_STATUS_PUSH_S:
`{"first": i, "total": n, "lines": [...]}` = its last lines with absolute line numbers
(line 0 is the runner's `hello`). An adapter that can only read the provider's log after
the run (Kaggle) serves these while the run is live, with cursors `t<consumed>:<hash>`
(tail lines consumed + a short hash of the last one), and switches to the provider's
final log when the run ends: `align()` maps the tail cursor onto the final log, whose
lines are the runner's lines behind a short provider preamble (Kaggle's run.py prints a
line or two before the runner starts). Lines that scrolled out of the tail between two
reads are replaced by one note and never repeated (A7).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from gpu_router.checkpoint import active_hub
from gpu_router.checkpoint.storage import StorageError

__all__ = ["TailRead", "align", "format_cursor", "parse_cursor", "preamble", "read_tail"]

TAIL_FILE = "log-tail.json"
HELLO_PREFIX = '::gpu:: {"t":"hello"'
ALIGN_WINDOW = 200


def line_hash(line: str) -> str:
    return hashlib.sha256(line.encode("utf-8", "replace")).hexdigest()[:10]


def format_cursor(consumed: int, last_hash: str | None) -> str:
    return f"t{consumed}:{last_hash or ''}"


def parse_cursor(since: str | None) -> tuple[int, str | None] | None:
    """`t<n>:<hash>` -> (n, hash or None); None when `since` is not a tail cursor."""
    if not since or not since.startswith("t"):
        return None
    head, _, tail = since[1:].partition(":")
    try:
        n = int(head)
    except ValueError:
        return None
    return max(0, n), (tail or None)


@dataclass(frozen=True, slots=True)
class TailRead:
    lines: list[str]  # to yield (a gap note first when lines scrolled away)
    consumed: int  # tail lines consumed after these (the cursor number)
    last_hash: str | None  # hash of the last consumed tail line
    total: int  # lines the runner had written at push time
    ended: bool  # the tail ends with the runner's exit line


def read_tail(status_uri: str, consumed: int, last_hash: str | None = None) -> TailRead | None:
    """New tail lines after `consumed`, or None when there is no readable side channel
    (no active hub, a URI no backend of ours owns, nothing pushed yet, storage trouble)."""
    hub = active_hub()
    if hub is None:
        return None
    try:
        opened = hub.open_status_uri(status_uri)
        if opened is None:
            return None
        store, key = opened
        data = store.read_json(f"{key}/{TAIL_FILE}")
    except StorageError:
        return None
    if not data:
        return None
    try:
        first = int(data.get("first", 0))
        total = int(data.get("total", 0))
    except (TypeError, ValueError):
        return None
    raw = data.get("lines")
    lines = [str(x) for x in raw] if isinstance(raw, list) else []
    ended = bool(lines) and lines[-1].startswith('::gpu:: {"t":"exit"')
    if total <= consumed:
        return TailRead([], consumed, last_hash, total, ended)
    start = max(consumed, first)
    out: list[str] = []
    if start > consumed:
        out.append(
            f"gpu-router: {start - consumed} log lines scrolled past before they could be "
            "shown live; the full log arrives when the run ends"
        )
    fresh = lines[start - first :]
    out.extend(fresh)
    new_hash = line_hash(fresh[-1]) if fresh else last_hash
    return TailRead(out, total, new_hash, total, ended)


def preamble(final_lines: list[str]) -> list[str]:
    """The provider-side lines before the runner's hello (never part of the tail)."""
    for i, line in enumerate(final_lines):
        if line.startswith(HELLO_PREFIX):
            return final_lines[:i]
    return []


def align(final_lines: list[str], consumed: int, last_hash: str | None) -> int:
    """Index into the provider's final log right after the last tail line served."""
    if consumed <= 0:
        return 0
    hello = next((i for i, line in enumerate(final_lines) if line.startswith(HELLO_PREFIX)), 0)
    expected = min(hello + consumed, len(final_lines))
    if not last_hash:
        return expected
    for delta in range(ALIGN_WINDOW + 1):
        for cand in (expected - delta, expected + delta):
            if 0 < cand <= len(final_lines) and line_hash(final_lines[cand - 1]) == last_hash:
                return cand
    return expected
