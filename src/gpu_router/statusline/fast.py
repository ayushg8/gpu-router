"""`gpu status --line`: gpu rows for the Claude Code status line (phase 6b).

Reads `<home>/state.json` (written by the daemon, shape in `gpu_router.statefile`) and returns
0-2 rows drawn in the user's own status-line style (CLAUDE.md "Claude Code status line": the
30-column grid, the 10-cell meter, the exact 256-colour codes of the reference status line,
tests/fixtures/statusline/statusline.sh).

Rules (invariants 14 and 16):

- STDLIB ONLY: json, os, sys, time (+ math, unicodedata when needed). No other gpu_router
  module is imported, not even `statefile` or `paths` (pathlib alone costs ~3 ms).
- Never calls the daemon or a provider; never raises; prints nothing when idle, when the
  file is missing, unreadable, or from an unknown schema.
- A daemon that is not running (its pid is gone) or whose heartbeat is overdue makes the
  file stale: nothing is printed, or one dim hint row when the file still lists active jobs
  (their remote runs keep going, invariant 11).
- Every duration is computed here from the absolute timestamps in the file, so the daemon
  never has to rewrite it just because a second passed.

Row states (docs/spec.md "GPU rows by state"): running (2 rows: bar + detail), needs approval,
just finished (visible `finished_visible_s`, default 0: hidden), migrated (visible
`migrated_visible_s`), failed, starting/queued; nothing active prints nothing. At most 2
rows: the most important job first (approval > failed > running/migrated > done >
starting), a second job's first row if room is left, and "+N running · +N queued" for the
rest at the end of the last row.

Per session (D56): with Claude Code's stdin JSON (`--stdin`, the wrapper), only jobs whose
origin (state.json `origin`, set by `gpu run` / the MCP server from CLAUDE_CODE_SESSION_ID
and CLAUDE_PID) matches this session are drawn: the stdin `session_id` (or this process's
CLAUDE_CODE_SESSION_ID) equals `claude_session`, or this process's CLAUDE_PID equals
`claude_pid` (keeps the rows after /clear, which changes the session id). Jobs with no
origin show in no Claude status line. Without a stdin session id (`gpu status --line` by
hand) every job is drawn, as before, so the command stays useful for debugging.
"""

from __future__ import annotations

import json
import os
import sys
import time

TYPE_CHECKING = False
if TYPE_CHECKING:  # no runtime import of typing/collections.abc: every ms counts here
    from collections.abc import Callable, Mapping, Sequence
    from typing import Any, TextIO

    Part = tuple[str, str]  # (plain text, the same text with its ANSI codes)
    Build = Callable[[str], list[Part]]
    Row = tuple[list[Part], list[Part]]  # (left cell, right column)
    Viewer = tuple[frozenset[str], str | None]  # (session ids, claude pid), D56

# --------------------------------------------------------------------------- tokens
# Exact copies of the reference status line's tokens (tests/fixtures/statusline/statusline.sh;
# tests/unit/statusline/test_user_style.py checks).
FG = "\033[38;5;252m"  # values (the spec's "bold white %" is this: light grey, not bold)
DIM = "\033[38;5;243m"  # labels, separators, resets, anything secondary
WARN = "\033[38;5;179m"  # 70%+ ... and the ⏸ icon
CRIT = "\033[38;5;174m"  # 90%+ ... and the ✗ icon
OFF = "\033[0m"
ACCENT = "\033[38;5;75m"  # the model name's blue: never used by gpu rows
OK = "\033[38;5;108m"  # the ✓ icon (the script has no green; muted like WARN/CRIT)
COL = 30  # column width: column 2 starts here on every row
GUTTER = 2  # a gpu left cell keeps at least this many spaces before column 2 (like row 1)
CELLS = 10  # meter cells
FILL = "█"  # █
EMPTY = "░"  # ░
RESET = "↻"  # ↻
ICON_WAIT = "⏸"  # ⏸
ICON_DONE = "✓"  # ✓
ICON_FAIL = "✗"  # ✗
ICON_MOVED = "↪"  # ↪
ARROW = "→"  # →
DOT = "·"  # ·
ELLIPSIS = "…"  # …

MAX_ROWS = 2
STATE_SCHEMA = 1  # == statefile.STATE_SCHEMA (test_fast.py asserts it)
STATE_FILE = "state.json"
DEFAULT_HOME = "~/Library/Application Support/gpu-router"
FINISHED_VISIBLE_S = 0.0  # defaults when an older daemon's file lacks the fields
MIGRATED_VISIBLE_S = 600.0
STALE_MIN_S = 300.0  # heartbeat overdue after max(5 x heartbeat_s, this)
MAX_FILE_BYTES = 1 << 20
MAX_STDIN_BYTES = 1 << 18

SP: Part = (" ", " ")
GAP: Part = ("  ", "  ")  # between two facts in one column (spec: "loss 0.412 ↓  ckpt 3m ago")
_NARROW = frozenset(
    FILL
    + EMPTY
    + RESET
    + ICON_WAIT
    + ICON_DONE
    + ICON_FAIL
    + ICON_MOVED
    + ARROW
    + DOT
    + ELLIPSIS
    + "\u2191\u2193\u00d7"
)
_TREND = {"down": "↓", "up": "↑", "flat": ARROW}
_DAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
_AMP = str.maketrans("AMP", "amp")  # the script pipes times through `tr 'AMP' 'amp'`
_RUNNING = ("running", "checkpointing")
_FAILURES = {
    "user_error": "script failed",
    "no_provider": "no provider fits",
    "provider_error": "provider error",
    "invalid_job": "rejected by providers",
    "internal": "internal error",
}
_PHASES = {
    "provisioning": "starting",
    "migrating": "moving",
    "routing": "routing",
    "queued": "queued",
    "cancelling": "cancelling",
}
#: Words of the "+N ..." summary, in display order.
_COUNT_WORDS = ("running", "to approve", "starting", "queued", "cancelling", "failed", "done")
_COUNT_OF = {
    "running": "running",
    "checkpointing": "running",
    "awaiting_approval": "to approve",
    "provisioning": "starting",
    "migrating": "starting",
    "routing": "queued",
    "queued": "queued",
    "cancelling": "cancelling",
    "failed": "failed",
    "done": "done",
}


# --------------------------------------------------------------------------- parts


def _dim(s: str) -> Part:
    return (s, DIM + s + OFF)


def _fg(s: str) -> Part:
    return (s, FG + s + OFF)


def _paint(code: str, s: str) -> Part:
    return (s, code + s + OFF)


def _state_color(pct: int) -> str:
    """The script's state_color: crit at 90%+, warn at 70%+, else the value colour."""
    if pct >= 90:
        return CRIT
    if pct >= 70:
        return WARN
    return FG


def _meter(pct: int, color: str) -> Part:
    """10 cells; filled = ceil(pct/10) in `color`, the rest dim (the script's meter())."""
    filled = max(0, min(CELLS, (pct * CELLS + 99) // 100))
    empty = CELLS - filled
    return (FILL * filled + EMPTY * empty, color + FILL * filled + DIM + EMPTY * empty + OFF)


def width(s: str) -> int:
    """Terminal cells of `s` (East Asian wide = 2, combining = 0)."""
    if s.isascii():
        return len(s)
    w = 0
    for ch in s:
        if ch < "ᄀ" or ch in _NARROW:
            w += 1
            continue
        import unicodedata

        if unicodedata.combining(ch):
            continue
        w += 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
    return w


def _pw(parts: Sequence[Part]) -> int:
    return sum(width(p[0]) for p in parts)


def _cut(text: str, room: int) -> str:
    """`text` fitted into `room` cells, ending in … when cut (the script's row-3 rule)."""
    if width(text) <= room:
        return text
    if room <= 0:
        return ""
    out, used = [], 0
    for ch in text:
        w = width(ch)
        if used + w > room - 1:
            break
        out.append(ch)
        used += w
    return "".join(out) + ELLIPSIS


def _fit(options: Sequence[tuple[Build, list[Part]]], name: str) -> tuple[list[Part], list[Part]]:
    """First option whose left cell fits in COL-GUTTER cells, plus the parts it moved to
    the right column; if none fits, the last option with `name` cut to the room left
    (the script's row-3 rule: drop the secondary half first, then cut with …)."""
    for build, moved in options:
        parts = build(name)
        if _pw(parts) <= COL - GUTTER:
            return parts, moved
    build, moved = options[-1]
    room = COL - GUTTER - _pw(build(""))
    return build(_cut(name, room)), moved


def join_row(left: Sequence[Part], right: Sequence[Part], *, color: bool = True) -> str:
    """Left cell padded to the grid (at least 1 space, like the script's cell()), then the
    right column. No trailing padding when the right column is empty."""
    i = 1 if color else 0
    out = "".join(p[i] for p in left)
    if right:
        out += " " * max(1, COL - _pw(left)) + "".join(p[i] for p in right)
    return out


# --------------------------------------------------------------------------- values


def _num(v: Any) -> float | None:
    if isinstance(v, bool) or not isinstance(v, int | float):
        return None
    return float(v)


def _int(v: Any) -> int | None:
    if isinstance(v, bool) or not isinstance(v, int | float):
        return None
    return int(v)


#: C0/C1 controls, DEL, bidi overrides: never printed (a job name with an ESC in it would
#: be an escape sequence in the user's terminal, D48). The daemon strips them too.
_CONTROLS = frozenset(
    [chr(c) for c in range(0x20)]
    + [chr(c) for c in range(0x7F, 0xA0)]
    + [chr(c) for c in (*range(0x202A, 0x202F), *range(0x2066, 0x206A))]
)


def _str(v: Any) -> str | None:
    if not isinstance(v, str) or not v:
        return None
    if v.isprintable():
        return v
    clean = "".join(
        " " if ch in "\t\n\r" else ch for ch in v if ch not in _CONTROLS or ch in "\t\n\r"
    )
    return clean or None


def _hm(seconds: float, *, up: bool = False) -> str:
    """h:mm (`up` rounds a remaining time up to the next minute)."""
    minutes = int(-(-seconds // 60)) if up else int(seconds // 60)
    minutes = max(0, minutes)
    return f"{minutes // 60}:{minutes % 60:02d}"


def _dur(seconds: float) -> str:
    """3h12m, 1h05m, 12m, 40s."""
    s = max(0, int(seconds))
    if s < 60:
        return f"{s}s"
    m = s // 60
    if m < 60:
        return f"{m}m"
    return f"{m // 60}h{m % 60:02d}m"


def _ago(seconds: float) -> str:
    s = max(0, int(seconds))
    if s < 60:
        return "<1m ago"
    m = s // 60
    if m < 60:
        return f"{m}m ago"
    if m < 48 * 60:
        return f"{m // 60}h ago"
    return f"{m // 1440}d ago"


def _hours_text(hours: float) -> str:
    """Route estimate: ~20m, ~2h, ~1.5h."""
    if hours < 1:
        return f"~{max(1, round(hours * 60))}m"
    return f"~{round(hours, 1):g}h"


def _cap_text(seconds: float) -> str:
    return f"{round(seconds / 3600, 1):g}h"


def _when(ts: float | None, now: float) -> str | None:
    """The script's when(): a reset inside 22 h reads as a time (1pm), further out it needs
    the weekday (Sat 5pm); both pass through tr 'AMP' 'amp' (so Monday prints 'mon')."""
    if ts is None or ts <= now:
        return None
    lt = time.localtime(ts)
    hour = lt.tm_hour % 12 or 12
    text = f"{hour}{'AM' if lt.tm_hour < 12 else 'PM'}"
    if ts - now >= 79200:
        text = f"{_DAYS[lt.tm_wday]} {text}"
    return text.translate(_AMP)


def _gpu(g: Any) -> str:
    """2xT4 -> 2\u00d7T4 (the multiplication sign of the spec); other names unchanged."""
    s = _str(g) or ""
    i = 0
    while i < len(s) and s[i].isdigit():
        i += 1
    if 0 < i < len(s) - 1 and s[i] in "xX":
        return f"{s[:i]}\u00d7{s[i + 1 :]}"
    return s


def _fmt_value(v: float) -> str:
    """0.412, 12.3, 1234, 0.00123 (three significant digits; shell/metrics.fmt_value)."""
    import math

    if v == 0 or not math.isfinite(v):
        return "0" if v == 0 else str(v)
    mag = abs(v)
    if 1e-3 <= mag < 1e5:
        decimals = max(0, 2 - math.floor(math.log10(mag)))
        return f"{v:.{decimals}f}"
    return f"{v:.3g}"


# --------------------------------------------------------------------------- context


class _Ctx:
    __slots__ = ("approvals", "cwd", "finished_s", "migrated_s", "now", "providers", "written")

    def __init__(self, snap: Mapping[str, Any], now: float, cwd: str | None) -> None:
        self.now = now
        self.written = _num(snap.get("written_at")) or now
        self.cwd = cwd.rstrip("/") if cwd else None
        provs = snap.get("providers")
        self.providers: dict[str, Mapping[str, Any]] = {
            p["name"]: p
            for p in (provs if isinstance(provs, list) else [])
            if isinstance(p, dict) and _str(p.get("name"))
        }
        fin = _num(snap.get("finished_visible_s"))  # 0 is valid: finished rows hidden
        mig = _num(snap.get("migrated_visible_s"))
        self.finished_s = FINISHED_VISIBLE_S if fin is None else fin
        self.migrated_s = MIGRATED_VISIBLE_S if mig is None else mig
        self.approvals = 0


def _name(job: Mapping[str, Any]) -> str:
    return _str(job.get("name")) or _str(job.get("short_id")) or "job"


def _short(job: Mapping[str, Any]) -> str:
    return _str(job.get("short_id")) or (_str(job.get("id")) or "")[:4]


def _label() -> list[Part]:
    return [_dim("gpu"), SP]


# --------------------------------------------------------------------------- rows


def _quota(provider: str | None, ctx: _Ctx) -> list[Part]:
    """Right column of the running row, mirroring `week █████░░░░░ 43% ↻Tue 4pm`."""
    if not provider:
        return []
    parts = [_dim(provider)]
    p = ctx.providers.get(provider)
    if p is None:
        return parts
    if p.get("unlimited") is True:
        return [*parts, SP, _dim("unlimited")]
    limit = _num(p.get("limit"))
    if not limit or limit <= 0:
        return parts
    left, used = _num(p.get("remaining")), _num(p.get("used"))
    if left is not None:
        raw = (limit - left) / limit * 100
    elif used is not None:
        raw = used / limit * 100
    else:
        return parts
    raw = max(0.0, raw)
    pct = int(raw + 0.5)
    show = "<1" if 0 < raw < 1 else str(pct)
    if p.get("source") == "estimate":
        show = "~" + show  # the ledger's "(est)": estimated from job history, not live
    color = _state_color(pct)
    parts += [SP, _meter(pct, color), SP, _paint(color, f"{show}%")]
    reset = _when(_num(p.get("resets_at")), ctx.now)
    if reset:
        parts += [SP, _dim(RESET + reset)]
    return parts


def _remaining(job: Mapping[str, Any], ctx: _Ctx) -> float | None:
    """The daemon's `eta_s` (computed at written_at from this attempt's own step rate)
    minus the file's age. No local fallback: `elapsed / step` from the file is wrong for a
    resumed attempt (job-level steps, attempt-level elapsed; D48), and None means the
    daemon does not know yet."""
    eta = _num(job.get("eta_s"))
    if eta is not None:
        return max(60.0, eta - max(0.0, ctx.now - ctx.written))
    return None


def _bar_row(job: Mapping[str, Any], ctx: _Ctx) -> Row:
    """`gpu ███░░░░░░░ 38% 1:50 left` | `kaggle ████████░░ 73% ↻Sat 5pm`.

    The spec's `~1:50 left` loses its tilde: with it a 2-digit percentage leaves a
    1-space gutter before column 2; `1:50` alone is kept for 10h+ remaining."""
    step, total = _int(job.get("step")), _int(job.get("total_steps"))
    started = _num(job.get("started_at"))
    elapsed = max(0.0, ctx.now - started) if started is not None else None
    cap = _num(job.get("session_cap_s"))
    right = _quota(_str(job.get("provider")), ctx)
    if total and total > 0:  # real steps from the helper (or a parsed "step 10/100")
        pct = max(0, min(100, int((step or 0) * 100 / total)))
        base = [*_label(), _meter(pct, FG), SP, _fg(f"{pct}%")]
        rem = _remaining(job, ctx)
        options: list[tuple[Build, list[Part]]] = [(lambda _n: base, [])]
        if rem is not None:
            t = _hm(rem, up=True)
            options = [
                (lambda _n: [*base, SP, _fg(t), SP, _dim("left")], []),
                (lambda _n: [*base, SP, _fg(t)], []),
                *options,
            ]
        left, _ = _fit(options, "")
        return left, right
    if elapsed is not None and cap:  # elapsed vs the session cap, labelled
        pct = max(0, min(100, int(elapsed * 100 / cap)))
        base = [*_label(), _meter(pct, FG), SP]
        e = _hm(elapsed)
        left, _ = _fit(
            [
                (lambda _n: [*base, _dim("("), _fg(e), _dim(f" of {_cap_text(cap)})")], []),
                (lambda _n: [*base, _dim("("), _fg(e), _dim(")")], []),
            ],
            "",
        )
        return left, right
    if elapsed is not None:
        return [*_label(), _fg(_hm(elapsed)), SP, _dim("elapsed")], right
    return [*_label(), _dim("starting")], right


def _detail_row(job: Mapping[str, Any], ctx: _Ctx) -> Row:
    """`train_yolo · 2xT4` | `loss 0.412 ↓  ckpt 3m ago` (mirrors the script's row 3)."""
    gpu = _gpu(job.get("gpu"))
    options: list[tuple[Build, list[Part]]] = [(lambda n: [_fg(n)], [])]
    if gpu:
        options.insert(0, (lambda n: [_fg(n), SP, _dim(DOT), SP, _dim(gpu)], []))
    left, _ = _fit(options, _name(job))
    right: list[Part] = []
    metric = job.get("metric")
    if isinstance(metric, dict):
        mname, value = _str(metric.get("name")), _num(metric.get("value"))
        if mname and value is not None:
            right += [_dim(mname), SP, _fg(_fmt_value(value))]
            arrow = _TREND.get(str(metric.get("trend")))
            if arrow:
                right += [SP, _fg(arrow)]
    ckpt: list[Part] = []
    if job.get("state") == "checkpointing":
        ckpt = [_dim("ckpt"), SP, _fg("saving")]
    else:
        last = _num(job.get("last_checkpoint_at"))
        if last is not None:
            ckpt = [_dim("ckpt"), SP, _fg(_ago(ctx.now - last))]
    if ckpt:
        right += [GAP, *ckpt] if right else ckpt
    step = _int(job.get("step"))
    if not right and step is not None and not _int(job.get("total_steps")):
        right = [_dim("step"), SP, _fg(str(step))]
    return left, right


def _approval_row(job: Mapping[str, Any], ctx: _Ctx) -> Row:
    """`gpu ⏸ eval.py → colab T4 · ~20m` | `/gpu-approve`."""
    provider = _str(job.get("provider"))
    route = " ".join(x for x in (provider, _gpu(job.get("gpu"))) if x)
    hours = _num(job.get("route_hours"))
    eta = _hours_text(hours) if hours else None
    summary = _str(job.get("route_summary"))  # older daemons: "colab T4 · ~20m"
    if not eta and summary and f" {DOT} " in summary:
        eta = summary.rsplit(f" {DOT} ", 1)[1]
    icon = _paint(WARN, ICON_WAIT)

    def head(n: str) -> list[Part]:
        return [*_label(), icon, SP, _fg(n)]

    to = [SP, _dim(ARROW), SP, _fg(route)]
    options: list[tuple[Build, list[Part]]] = []
    if route and eta:
        options.append((lambda n: [*head(n), *to, SP, _dim(DOT), SP, _fg(eta)], []))
        options.append((lambda n: [*head(n), *to], [_fg(eta), GAP]))
        options.append((head, [_dim(ARROW), SP, _fg(route), SP, _dim(DOT), SP, _fg(eta), GAP]))
    elif route:
        options.append((lambda n: [*head(n), *to], []))
        options.append((head, [_dim(ARROW), SP, _fg(route), GAP]))
    else:
        options.append((head, [_fg(eta), GAP] if eta else []))
    left, moved = _fit(options, _str(job.get("script")) or _name(job))
    cmd = "/gpu-approve" + (f" {_short(job)}" if ctx.approvals > 1 else "")
    return left, [*moved, _fg(cmd)]


def _migrated_row(job: Mapping[str, Any], ctx: _Ctx) -> Row:
    """`gpu ↪ train_yolo  colab → kaggle` | `resumed · ckpt 4`."""
    frm = _str(job.get("migrated_from")) or "?"
    to = _str(job.get("provider"))
    running = job.get("state") in _RUNNING
    move = (
        [_dim(frm), SP, _dim(ARROW), SP, _fg(to)]
        if to and to != frm
        else [_dim("from"), SP, _fg(frm)]
    )

    def head(n: str) -> list[Part]:
        return [*_label(), _fg(ICON_MOVED), SP, _fg(n)]

    left, moved = _fit([(lambda n: [*head(n), GAP, *move], []), (head, [*move, GAP])], _name(job))
    seq = _int(job.get("resumed_from_seq")) if running else _int(job.get("checkpoint_seq"))
    if running and seq is None and "resumed_from_seq" not in job:  # older daemon
        seq = _int(job.get("checkpoint_seq"))
    if seq:
        tail = [_fg("resumed" if running else "resuming"), SP, _dim(DOT), SP, _fg(f"ckpt {seq}")]
    else:
        tail = [_fg("restarted" if running else "restarting"), SP, _dim(DOT), SP, _dim("no ckpt")]
    return left, [*moved, *tail]


def _starting_row(job: Mapping[str, Any], ctx: _Ctx) -> Row:
    """`gpu train_yolo → kaggle 2xT4` | `starting` (provisioning, queued, routing...)."""
    route = " ".join(x for x in (_str(job.get("provider")), _gpu(job.get("gpu"))) if x)

    def head(n: str) -> list[Part]:
        return [*_label(), _fg(n)]

    options: list[tuple[Build, list[Part]]] = [(head, [])]
    if route:
        options = [
            (lambda n: [*head(n), SP, _dim(ARROW), SP, _fg(route)], []),
            (head, [_dim(ARROW), SP, _fg(route), GAP]),
        ]
    left, moved = _fit(options, _name(job))
    state = str(job.get("state"))
    phase = [_dim(_PHASES.get(state) or _str(state) or "")]
    retry = _num(job.get("not_before"))
    if state == "queued" and retry is not None and retry > ctx.now:
        wait = retry - ctx.now
        when = f"in {_dur(wait)}" if wait < 3600 else (_when(retry, ctx.now) or "later")
        phase += [SP, _dim(DOT), SP, _dim(f"retry {when}")]
    return left, [*moved, *phase]


def _outputs(rec: Mapping[str, Any], ctx: _Ctx) -> str | None:
    """./runs/a7f2 when the session is in the job's project, else <project>/runs/a7f2."""
    proj = (_str(rec.get("project_dir")) or "").rstrip("/")
    path = _str(rec.get("outputs_path"))
    if proj and path and path.startswith(proj + "/"):
        sub = path[len(proj) + 1 :]
        if ctx.cwd and (ctx.cwd == proj or ctx.cwd.startswith(proj + "/")):
            return f"./{sub}"
        return f"{os.path.basename(proj)}/{sub}"
    return _str(rec.get("outputs_dir")) or path


def _finished_row(rec: Mapping[str, Any], ctx: _Ctx) -> Row:
    """`gpu ✓ train_yolo · 3h12m` | `→ ./runs/a7f2`."""
    dur_s = _num(rec.get("duration_s"))
    dur = _dur(dur_s) if dur_s is not None else None
    icon = _paint(OK, ICON_DONE)

    def head(n: str) -> list[Part]:
        return [*_label(), icon, SP, _fg(n)]

    options: list[tuple[Build, list[Part]]] = [(head, [])]
    if dur:
        options = [(lambda n: [*head(n), SP, _dim(DOT), SP, _fg(dur)], []), (head, [_fg(dur), GAP])]
    left, moved = _fit(options, _name(rec))
    out = _outputs(rec, ctx)
    if rec.get("outputs_fetched") is False:
        tail = [_dim(f"gpu fetch {_short(rec)}")]
    elif out:
        tail = [_dim(ARROW), SP, _fg(out)]
    else:
        tail = [_dim("done")]
    return left, [*moved, *tail]


def _failed_row(rec: Mapping[str, Any], ctx: _Ctx) -> Row:
    """`gpu ✗ train_yolo · 1h02m` | `exit 1 · gpu logs a7f2`."""
    dur_s = _num(rec.get("duration_s"))
    icon = _paint(CRIT, ICON_FAIL)

    def head(n: str) -> list[Part]:
        return [*_label(), icon, SP, _fg(n)]

    options: list[tuple[Build, list[Part]]] = [(head, [])]
    if dur_s is not None:
        d = _dur(dur_s)
        options = [(lambda n: [*head(n), SP, _dim(DOT), SP, _fg(d)], []), (head, [_fg(d), GAP])]
    left, moved = _fit(options, _name(rec))
    kind, code = _str(rec.get("failure_kind")), _int(rec.get("exit_code"))
    reason = _FAILURES.get(kind or "", "failed")
    if kind == "user_error" and code:
        reason = f"exit {code}"
    return left, [*moved, _fg(reason), SP, _dim(DOT), SP, _dim(f"gpu logs {_short(rec)}")]


def _hint_row(text: str, command: str) -> Row:
    return [*_label(), _dim(text)], [_dim(command)]


# --------------------------------------------------------------------------- assembly


def _blocks(snap: Mapping[str, Any], ctx: _Ctx) -> list[tuple[int, str, list[Row]]]:
    """(priority, count word, rows) per job worth showing, most important first."""
    active = [j for j in snap.get("active") or [] if isinstance(j, dict)]
    ctx.approvals = sum(1 for j in active if j.get("state") == "awaiting_approval")
    out: list[tuple[int, int, str, list[Row]]] = []
    for i, job in enumerate(active):
        state = str(job.get("state"))
        word = _COUNT_OF.get(state, "queued")
        migrated_at = _num(job.get("migrated_at"))
        recently_moved = (
            _str(job.get("migrated_from")) is not None
            and migrated_at is not None
            and ctx.now - migrated_at <= ctx.migrated_s
        )
        if state == "awaiting_approval":
            out.append((0, i, word, [_approval_row(job, ctx)]))
        elif recently_moved and state != "cancelling":
            rows = [_migrated_row(job, ctx)]
            if state in _RUNNING:
                rows.append(_bar_row(job, ctx))
            out.append((2, i, word, rows))
        elif state in _RUNNING:
            out.append((2, i, word, [_bar_row(job, ctx), _detail_row(job, ctx)]))
        else:
            out.append((4, i, word, [_starting_row(job, ctx)]))
    recent = [r for r in snap.get("recent") or [] if isinstance(r, dict)]
    for i, rec in enumerate(recent, start=len(active)):
        finished = _num(rec.get("finished_at"))
        if finished is None or ctx.now - finished > ctx.finished_s:
            continue
        outcome = rec.get("state")
        if outcome == "failed":
            out.append((1, i, "failed", [_failed_row(rec, ctx)]))
        elif outcome == "done":
            out.append((3, i, "done", [_finished_row(rec, ctx)]))
        # cancelled / denied: the user did that; nothing to report
    out.sort(key=lambda b: (b[0], b[1]))
    return [(p, w, rows) for p, _, w, rows in out]


def _pid_alive(pid: Any) -> bool:
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _mine(job: Mapping[str, Any], viewer: Viewer) -> bool:
    """True when `job` came from the viewing session (its origin matches, D56)."""
    origin = job.get("origin")
    if not isinstance(origin, dict):
        return False
    sessions, pid = viewer
    sid = origin.get("claude_session")
    if isinstance(sid, str) and sid and sid in sessions:
        return True
    opid = origin.get("claude_pid")
    return bool(pid) and isinstance(opid, str) and opid == pid


def _scoped(snap: Mapping[str, Any], viewer: Viewer | None) -> Mapping[str, Any]:
    """`snap` with only the viewing session's active and recent jobs (all without a viewer)."""
    if viewer is None:
        return snap
    out = dict(snap)
    for key in ("active", "recent"):
        jobs = snap.get(key)
        out[key] = (
            [j for j in jobs if isinstance(j, dict) and _mine(j, viewer)]
            if (isinstance(jobs, list))
            else []
        )
    return out


def build_rows(
    snap: Mapping[str, Any],
    *,
    now: float,
    cwd: str | None = None,
    pid_alive: Callable[[Any], bool] = _pid_alive,
    viewer: Viewer | None = None,
) -> list[Row]:
    """The (left, right) parts of each row to print, at most MAX_ROWS. Pure given `now`
    and `pid_alive`. `viewer` (session ids, claude pid) keeps only that session's jobs."""
    if not isinstance(snap, dict) or snap.get("schema") != STATE_SCHEMA:
        return []
    snap = _scoped(snap, viewer)
    ctx = _Ctx(snap, now, cwd)
    has_active = any(isinstance(j, dict) for j in snap.get("active") or [])
    if not pid_alive(snap.get("daemon_pid")):
        return [_hint_row("daemon not running", "gpu daemon start")] if has_active else []
    heartbeat = _num(snap.get("heartbeat_s"))
    if heartbeat and has_active and now - ctx.written > max(5 * heartbeat, STALE_MIN_S):
        return [_hint_row("daemon not responding", "gpu daemon status")]
    rows: list[Row] = []
    hidden: dict[str, int] = {}
    for _prio, word, block in _blocks(snap, ctx):
        room = MAX_ROWS - len(rows)
        if room <= 0:
            hidden[word] = hidden.get(word, 0) + 1
            continue
        rows.extend(block[:room])
    if rows and hidden:
        summary = f" {DOT} ".join(f"+{hidden[w]} {w}" for w in _COUNT_WORDS if hidden.get(w))
        left, right = rows[-1]
        rows[-1] = (left, [*right, GAP, _dim(summary)] if right else [_dim(summary)])
    return rows


def rows_text(
    snap: Mapping[str, Any],
    *,
    now: float,
    cwd: str | None = None,
    color: bool = True,
    pid_alive: Callable[[Any], bool] = _pid_alive,
    viewer: Viewer | None = None,
) -> list[str]:
    """build_rows() joined into printable lines."""
    return [
        join_row(left, right, color=color)
        for left, right in build_rows(snap, now=now, cwd=cwd, pid_alive=pid_alive, viewer=viewer)
    ]


# --------------------------------------------------------------------------- entry


def state_path(environ: Mapping[str, str], home: str | None = None) -> str:
    base = home or environ.get("GPU_ROUTER_HOME") or DEFAULT_HOME
    return os.path.join(os.path.expanduser(base), STATE_FILE)


def load_snapshot(path: str) -> dict[str, Any] | None:
    """The parsed state.json, or None when missing, too big, unreadable or not an object."""
    try:
        with open(path, "rb") as fh:
            data = fh.read(MAX_FILE_BYTES + 1)
    except OSError:
        return None
    if len(data) > MAX_FILE_BYTES:
        return None
    try:
        snap = json.loads(data)
    except ValueError:
        return None
    return snap if isinstance(snap, dict) else None


def _stdin_info(stream: TextIO) -> tuple[str | None, str | None]:
    """(cwd, session id) from Claude Code's status-line JSON: workspace.current_dir (or
    project_dir / cwd) and session_id."""
    try:
        payload = json.loads(stream.read(MAX_STDIN_BYTES) or "null")
    except (ValueError, OSError):
        return None, None
    if not isinstance(payload, dict):
        return None, None
    session = _str(payload.get("session_id"))
    ws = payload.get("workspace")
    if isinstance(ws, dict):
        for key in ("current_dir", "project_dir"):
            if _str(ws.get(key)):
                return str(ws[key]), session
    return _str(payload.get("cwd")), session


def _stdin_cwd(stream: TextIO) -> str | None:
    """workspace.current_dir (or project_dir / cwd) from Claude Code's status-line JSON."""
    return _stdin_info(stream)[0]


def _viewer(session: str | None, env: Mapping[str, str]) -> Viewer | None:
    """The session drawing the rows (D56), or None (no session id: show every job)."""
    if not session:
        return None
    sessions = {session}
    own = env.get("CLAUDE_CODE_SESSION_ID")
    if own:
        sessions.add(own)
    pid = env.get("CLAUDE_PID") or ""
    return frozenset(sessions), (pid if pid.isdigit() else None)


def render(
    argv: Sequence[str] = (),
    *,
    environ: Mapping[str, str] | None = None,
    now: float | None = None,
    stdin: TextIO | None = None,
) -> str:
    """`gpu status --line [--stdin] [--plain] [--home DIR] [--cwd DIR] [--now EPOCH]
    [--session ID]`: the rows joined by newlines ("" = print nothing). Never raises.
    A session id (stdin JSON, else --session) shows only that session's jobs (D56)."""
    try:
        env = os.environ if environ is None else environ
        home = cwd = session = None
        use_stdin, color = False, True
        args = list(argv)
        while args:
            a = args.pop(0)
            if a == "--stdin":
                use_stdin = True
            elif a in ("--plain", "--no-color"):
                color = False
            elif a in ("--home", "--cwd", "--now", "--session") and args:
                value = args.pop(0)
                if a == "--home":
                    home = value
                elif a == "--cwd":
                    cwd = value
                elif a == "--session":
                    session = value
                else:
                    now = float(value)
            # anything else (--line itself, future flags) is ignored: never fail
        if use_stdin:
            in_cwd, in_session = _stdin_info(stdin if stdin is not None else sys.stdin)
            cwd, session = in_cwd or cwd, in_session or session
        snap = load_snapshot(state_path(env, home))
        if snap is None:
            return ""
        if cwd is None:
            try:
                cwd = os.getcwd()
            except OSError:
                cwd = None
        t = time.time() if now is None else now
        viewer = _viewer(session, env)
        return "\n".join(rows_text(snap, now=t, cwd=cwd, color=color, viewer=viewer))
    except Exception:
        return ""


if __name__ == "__main__":  # python -m gpu_router.statusline.fast --line ...
    out = render(sys.argv[1:])
    if out:
        sys.stdout.write(out + "\n")
