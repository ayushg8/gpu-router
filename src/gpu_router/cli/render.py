"""Human output for the CLI (phase 2): one visual language (spec UX principle 6).

green running, yellow waiting, red failed, dim idle/finished-by-user. Icons: ⚡ running,
⏸ waiting (queued, routing, provisioning, approval, cancelling), ✓ done, ✗ failed /
cancelled / denied, ↪ migrating or handed off. Colour goes on the icon and state word
only; everything else is plain or dim.

Pure formatting: functions take API models and return rich renderables or strings; no I/O
besides the Console passed in.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime

from rich.console import Console
from rich.table import Table
from rich.text import Text

from gpu_router.api import JobDetail, JobView, PolicyView, ProviderView, RouteDecision
from gpu_router.models import FailureKind, JobEvent, JobSpec, ProviderHealth, QuotaSnapshot
from gpu_router.router.base import RouteOutcome
from gpu_router.statemachine import JobState

ICON_RUNNING = "⚡"
ICON_WAITING = "⏸"
ICON_DONE = "✓"
ICON_FAILED = "✗"
ICON_MIGRATED = "↪"

_WAITING = {
    JobState.QUEUED,
    JobState.ROUTING,
    JobState.PROVISIONING,
    JobState.AWAITING_APPROVAL,
    JobState.CANCELLING,
}

_LABEL = {
    JobState.AWAITING_APPROVAL: "needs approval",
    JobState.PROVISIONING: "starting",
}


def state_style(state: JobState) -> tuple[str, str]:
    """(icon, rich style) for a job state."""
    if state in (JobState.RUNNING, JobState.CHECKPOINTING):
        return ICON_RUNNING, "green"
    if state is JobState.MIGRATING:
        return ICON_MIGRATED, "yellow"
    if state in _WAITING:
        return ICON_WAITING, "yellow"
    if state is JobState.DONE:
        return ICON_DONE, "green"
    if state is JobState.FAILED:
        return ICON_FAILED, "red"
    return ICON_FAILED, "dim"  # cancelled, denied


def state_label(state: JobState) -> str:
    """The word for a state in tables, the shell's panel, popup and live views: one word
    per state everywhere ("needs approval", never "awaiting approval" in one place and
    "waiting for your approval" in another)."""
    return _LABEL.get(state, str(state).replace("_", " "))


def state_text(state: JobState) -> Text:
    icon, style = state_style(state)
    return Text(f"{icon} {state_label(state)}", style=style)


def icon_text(state: JobState) -> Text:
    icon, style = state_style(state)
    return Text(icon, style=style)


# --------------------------------------------------------------------------- formatting


def duration(seconds: float | None) -> str:
    """42s, 3m12s, 1h05m, 2d3h."""
    if seconds is None:
        return "-"
    s = max(0, int(seconds))
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    if h < 48:
        return f"{h}h{m:02d}m"
    d, h = divmod(h, 24)
    return f"{d}d{h}h"


def clock_time(seconds: float | None) -> str:
    """Elapsed as h:mm:ss (the spec's job panel style)."""
    if seconds is None:
        return "-"
    s = max(0, int(seconds))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def local_dt(ts: float) -> datetime:
    return datetime.fromtimestamp(ts, UTC).astimezone()


def reset_label(ts: float | None, now: float) -> str:
    """↻Sat within a week, ↻3h within a day, ↻Oct 4 further out."""
    if ts is None:
        return ""
    delta = ts - now
    if delta <= 0:
        return "↻now"
    if delta < 86400:
        return f"↻{duration(delta)}"
    dt = local_dt(ts)
    if delta < 7 * 86400:
        return f"↻{dt.strftime('%a')}"
    return f"↻{dt.strftime('%b')} {dt.day}"


def ago(ts: float | None, now: float) -> str:
    if ts is None:
        return "-"
    return f"{duration(now - ts)} ago"


def rel_path(path: str | None) -> str:
    """./runs/a7f2 when under the cwd, else the absolute path."""
    if not path:
        return "-"
    try:
        rel = os.path.relpath(path)
    except ValueError:
        return path
    return path if rel.startswith("..") else f"./{rel}" if rel != "." else "."


def fmt_num(v: float) -> str:
    """16, 2.5, 0.1 (one decimal at most; 0.0004 -> 0)."""
    r = round(v, 1)
    return str(int(r)) if r == int(r) else f"{r:.1f}"


_UNIT = {"gpu_hours": "h", "credits": " credits", "usd": " USD"}


def quota_text(q: QuotaSnapshot | None, now: float) -> str:
    if q is None:
        return "quota unknown"
    unit = _UNIT.get(str(q.unit), f" {q.unit}")
    used = fmt_num(q.used)
    body = f"{used}/{fmt_num(q.limit)}{unit}" if q.limit is not None else f"{used}{unit} used"
    est = " est" if q.source == "estimate" else ""
    reset = reset_label(q.resets_at, now)
    return f"{body}{est}" + (f" {reset}" if reset else "")


def health_text(health: ProviderHealth) -> Text:
    if health is ProviderHealth.OK:
        return Text("● up", style="green")
    if health is ProviderHealth.DEGRADED:
        return Text("● degraded", style="yellow")
    if health is ProviderHealth.AUTH_REQUIRED:
        return Text("● login needed", style="yellow")
    if health is ProviderHealth.UNAVAILABLE:
        return Text("● down", style="red")
    if health is ProviderHealth.DISABLED:
        return Text("○ off", style="dim")
    return Text("○ unchecked", style="dim")


def elapsed(job: JobView, now: float) -> float | None:
    if job.started_at is None:
        return None
    end = job.finished_at if job.finished_at is not None else now
    return end - job.started_at


def where(job: JobView) -> str:
    if not job.provider:
        return "-"
    return f"{job.provider} · {job.gpu}" if job.gpu else job.provider


def progress_text(job: JobView) -> str:
    parts: list[str] = []
    p = job.progress
    if p.step is not None and p.total:
        parts.append(f"step {p.step}/{p.total}")
    elif p.step is not None:
        parts.append(f"step {p.step}")
    for key in ("loss", "val_loss", "acc", "accuracy"):
        if key in job.last_metrics:
            parts.append(f"{key} {job.last_metrics[key]:.4g}")
            break
    else:
        if job.last_metrics:
            k, v = next(iter(job.last_metrics.items()))
            parts.append(f"{k} {v:.4g}")
    if job.checkpoint_count:
        parts.append(f"ckpt {job.checkpoint_count}")
    return "  ".join(parts)


def bar(fraction: float | None, width: int = 12) -> str:
    if fraction is None:
        return ""
    filled = round(max(0.0, min(1.0, fraction)) * width)
    return "█" * filled + "░" * (width - filled) + f" {round(fraction * 100):>3d}%"


def next_step(job: JobView) -> str:
    """One hint line for what the user can do next."""
    sid = job.short_id
    if job.state is JobState.AWAITING_APPROVAL:
        return f"gpu approve {sid}  or  gpu deny {sid}"
    if job.state is JobState.DONE:
        return f"outputs in {rel_path(job.outputs_dir)}"
    if job.state is JobState.FAILED:
        if job.failure_kind is FailureKind.NO_PROVIDER and job.accepted_attempts == 0:
            # nothing ever ran, so there are no logs: show why nothing fits instead
            return "gpu route to see which providers fit and why"
        return f"gpu logs {sid} to see what happened"
    if job.state in (JobState.CANCELLED, JobState.DENIED):
        return ""
    return f"gpu logs {sid} --follow"


# --------------------------------------------------------------------------- tables


def jobs_table(jobs: Sequence[JobView], now: float, *, finished: bool = False) -> Table:
    t = Table(box=None, pad_edge=False, show_edge=False, header_style="dim")
    t.add_column("", no_wrap=True)
    t.add_column("id", no_wrap=True)
    t.add_column("name", overflow="ellipsis", max_width=28)
    t.add_column("state", no_wrap=True)
    t.add_column("where", no_wrap=True)
    t.add_column("time" if not finished else "took", no_wrap=True, justify="right")
    t.add_column("progress / note" if not finished else "result", overflow="fold")
    for job in jobs:
        icon, style = state_style(job.state)
        took = elapsed(job, now)
        if finished:
            note = rel_path(job.outputs_dir) if job.state is JobState.DONE else job.message
            if job.accepted_attempts > 1:
                note = f"{ICON_MIGRATED} {job.accepted_attempts} runs  " + note
            when = duration(took)
        else:
            note = progress_text(job) or (job.message if job.state in _WAITING else "")
            when = clock_time(took) if took is not None else duration(now - job.created_at)
        t.add_row(
            Text(icon, style=style),
            job.short_id,
            job.name,
            Text(_LABEL.get(job.state, str(job.state).replace("_", " ")), style=style),
            where(job),
            when,
            Text(note, style="dim") if not finished else note,
        )
    return t


def providers_footer(providers: Iterable[ProviderView], now: float, running: int) -> Text:
    """kaggle 22/30h ↻Sat │ colab ● up │ 1 running (the shell footer's language)."""
    out = Text()
    first = True
    for p in providers:
        if not p.enabled:
            continue
        if not first:
            out.append(" │ ", style="dim")
        first = False
        out.append(p.name)
        out.append(" ")
        if p.health in (ProviderHealth.OK, ProviderHealth.UNKNOWN) and p.quota is not None:
            out.append(quota_text(p.quota, now), style="dim")
        else:
            out.append_text(health_text(p.health))
    if not first:
        out.append(" │ ", style="dim")
    out.append(f"{running} running" if running else "idle", style="" if running else "dim")
    return out


def providers_table(providers: Sequence[ProviderView], now: float) -> Table:
    t = Table(box=None, pad_edge=False, show_edge=False, header_style="dim")
    for col in ("provider", "status", "gpus", "session", "running", "quota", "note"):
        t.add_column(col, no_wrap=col not in ("gpus", "note"), overflow="fold")
    for p in providers:
        t.add_row(
            p.name if p.enabled else Text(p.name, style="dim"),
            health_text(p.health if p.enabled else ProviderHealth.DISABLED),
            ", ".join(p.gpus) or "-",
            f"{fmt_num(p.session_hours)}h" if p.session_hours else "-",
            str(p.live_attempts),
            quota_text(p.quota, now) if p.quota else "-",
            Text(p.health_reason or "", style="dim"),
        )
    return t


def quota_table(
    quotas: Sequence[QuotaSnapshot], providers: Sequence[ProviderView], now: float
) -> Table:
    t = Table(box=None, pad_edge=False, show_edge=False, header_style="dim")
    for col in ("provider", "used", "left", "resets", "source"):
        t.add_column(col, no_wrap=True)
    t.add_column("how", overflow="fold")  # phase 5: the ledger's one-line basis
    by_name = {q.provider: q for q in quotas}
    names = [p.name for p in providers if p.enabled] or list(by_name)
    for name in names:
        q = by_name.get(name)
        if q is None:
            t.add_row(name, "-", "-", "-", Text("unknown", style="dim"), "")
            continue
        unit = _UNIT.get(str(q.unit), f" {q.unit}")
        left = f"{fmt_num(max(0.0, q.limit - q.used))}{unit}" if q.limit is not None else "-"
        limit = f"/{fmt_num(q.limit)}" if q.limit is not None else ""
        t.add_row(
            name,
            f"{fmt_num(q.used)}{limit}{unit}",
            left,
            reset_label(q.resets_at, now).lstrip("↻") or "-",
            Text("live" if q.source == "live" else "estimate", style="dim"),
            Text(str(q.detail.get("note") or ""), style="dim"),
        )
    return t


# --------------------------------------------------------------------------- screens


def print_empty_state(
    console: Console, providers: Sequence[ProviderView], now: float, *, what: str = "no jobs"
) -> None:
    """Spec UX 7: never a blank screen. Quota left plus an example run."""
    console.print(Text(f"{what}.", style="dim"))
    ready = [p for p in providers if p.enabled]
    if ready:
        console.print()
        console.print(Text("free GPU time", style="dim"))
        for p in ready:
            line = Text("  ")
            line.append(f"{p.name:<12}")
            line.append_text(health_text(p.health))
            line.append("  ")
            line.append(quota_text(p.quota, now) if p.quota else "quota unknown", style="dim")
            console.print(line)
    console.print()
    console.print(Text("try", style="dim"))
    console.print("  gpu run train.py              run it on the best free GPU")
    console.print("  gpu route train.py            see where it would go and why")


def print_status(
    console: Console,
    status_active: Sequence[JobView],
    recent: Sequence[JobView],
    providers: Sequence[ProviderView],
    now: float,
) -> None:
    if not status_active and not recent:
        print_empty_state(console, providers, now, what="nothing running")
        return
    if status_active:
        console.print(jobs_table(status_active, now))
    if recent:
        if status_active:
            console.print()
        console.print(Text("finished recently", style="dim"))
        console.print(jobs_table(recent, now, finished=True))
    running = sum(1 for j in status_active if j.state in (JobState.RUNNING, JobState.CHECKPOINTING))
    console.print()
    console.print(providers_footer(providers, now, running))


def print_detail(console: Console, detail: JobDetail, now: float) -> None:
    job = detail.job
    head = Text()
    head.append_text(icon_text(job.state))
    head.append(f" job {job.short_id}  ", style="bold")
    head.append(job.name, style="bold")
    head.append("  ")
    _, style = state_style(job.state)
    head.append(_LABEL.get(job.state, str(job.state).replace("_", " ")), style=style)
    console.print(head)

    rows: list[tuple[str, str | Text]] = [("id", job.id)]
    spec = job.spec
    rows.append(("run", entry_text(spec)))
    rows.append(("where", where(job)))
    seen = gpu_seen(detail)
    if seen:
        rows.append(("gpu seen", Text(seen, style="yellow")))
    if job.route_reason:
        rows.append(("why", job.route_reason))
    if job.approval_reason and job.state is JobState.AWAITING_APPROVAL:
        rows.append(("approval", job.approval_reason))
    took = elapsed(job, now)
    if took is not None:
        rows.append(("time", clock_time(took)))
    frac = job.progress.fraction
    prog = progress_text(job)
    if frac is not None:
        rows.append(("progress", f"{bar(frac)}  {prog}"))
    elif prog:
        rows.append(("progress", prog))
    if job.last_checkpoint_at is not None:
        rows.append(
            ("checkpoint", f"{job.checkpoint_count} saved, last {ago(job.last_checkpoint_at, now)}")
        )
    if job.outputs_dir:
        fetched = "" if job.outputs_fetched else " (not fetched yet)"
        rows.append(("outputs", rel_path(job.outputs_dir) + fetched))
    if job.message:
        rows.append(("status", job.message))
    if job.exit_code is not None:
        rows.append(("exit code", str(job.exit_code)))
    rows.append(("submitted", f"{ago(job.created_at, now)} via {job.source}"))

    t = Table(box=None, show_header=False, pad_edge=False)
    t.add_column(style="dim", no_wrap=True)
    t.add_column(overflow="fold")
    for k, v in rows:
        t.add_row(k, v)
    console.print(t)

    if len(detail.attempts) > 0:
        console.print()
        console.print(Text("attempts", style="dim"))
        at = Table(box=None, show_header=False, pad_edge=False)
        for _ in range(4):
            at.add_column(no_wrap=True)
        at.add_column(overflow="fold")
        for a in detail.attempts:
            end = a.ended_at if a.ended_at is not None else now
            start = a.started_at or a.submitted_at or a.created_at
            note = a.error_message or a.lost_reason or a.remote_message or ""
            at.add_row(
                f"  {a.n}",
                a.provider,
                a.gpu or "-",
                str(a.state),
                f"{duration(end - start)}  " + note,
            )
        console.print(at)

    if detail.events:
        console.print()
        console.print(Text("timeline", style="dim"))
        for ev in detail.events[-12:]:
            console.print(event_text(ev, now, with_time=True), soft_wrap=True)

    hint = next_step(job)
    if hint:
        console.print()
        console.print(Text(hint, style="dim"))


def gpu_seen(detail: JobDetail) -> str | None:
    """'Tesla T4 (not the L4 it was placed on)' when the runner of the job's latest attempt
    reported another GPU (a gpu_mismatch note, D56), else None."""
    latest = detail.job.current_attempt_id or (detail.attempts[-1].id if detail.attempts else None)
    for ev in reversed(detail.events):
        if ev.kind == "note" and ev.reason == "gpu_mismatch" and ev.attempt_id == latest:
            seen = [str(s).split(",", 1)[0].strip() for s in ev.detail.get("seen") or []]
            placed = ev.detail.get("placed")
            label = " + ".join(dict.fromkeys(seen)) or "another GPU"
            return f"{label} (not the {placed} it was placed on)" if placed else label
    return None


def entry_text(spec: JobSpec) -> str:
    import shlex

    argv = [spec.script] if spec.script else list(spec.command or [])
    return shlex.join([*argv, *spec.args])


def event_text(ev: JobEvent, now: float, *, with_time: bool = False) -> Text:
    out = Text("  ")
    if with_time:
        out.append(f"{local_dt(ev.ts).strftime('%H:%M:%S')}  ", style="dim")
    if ev.kind == "transition" and ev.to_state is not None:
        icon, style = state_style(ev.to_state)
        if ev.reason in ("session_lost", "status_lost", "handoff", "quota_exhausted"):
            icon = ICON_MIGRATED
        out.append(f"{icon} ", style=style)
        out.append(ev.message)
    else:
        out.append("· ", style="dim")
        out.append(ev.message, style="dim")
    return out


def print_route(console: Console, spec: JobSpec, decision: RouteDecision) -> None:
    head = Text()
    if decision.outcome is RouteOutcome.PLACE and decision.chosen is not None:
        c = decision.chosen
        head.append(f"{ICON_RUNNING} ", style="green")
        head.append(f"{entry_text(spec)} → {c.provider}" + (f" {c.gpu}" if c.gpu else ""))
    elif decision.outcome is RouteOutcome.WAIT:
        head.append(f"{ICON_WAITING} ", style="yellow")
        head.append(f"{entry_text(spec)} would wait: nothing is free right now")
    else:
        head.append(f"{ICON_FAILED} ", style="red")
        head.append(f"{entry_text(spec)} cannot run anywhere")
    console.print(head)
    console.print(Text(f"  {decision.reason}", style="dim"))
    facts = route_facts(decision)
    if facts:
        console.print(Text(f"  {facts}", style="dim"))
    if decision.candidates:
        console.print()
        console.print(Text("candidates, best first", style="dim"))
        for i, c in enumerate(decision.candidates, 1):
            console.print(f"  {i}. {c.reason}")
    if decision.rejected:
        console.print()
        console.print(Text("ruled out", style="dim"))
        for r in decision.rejected:
            console.print(Text(f"  {ICON_FAILED} {r.reason}", style="dim"))


def _mb(n: int) -> str:
    return f"{n / 1e6:.1f} MB" if n >= 100_000 else f"{max(1, round(n / 1000))} KB"


def print_bundle(console: Console, bundle: dict[str, object]) -> None:
    """Dry-run summary of what would ship (cli/bundling.preview shape)."""
    deps = bundle.get("deps")
    est = bundle.get("estimate")
    deps = deps if isinstance(deps, dict) else {}
    est = est if isinstance(est, dict) else {}
    console.print()
    console.print(Text("would ship", style="dim"))
    size = bundle.get("size_bytes")
    code = bundle.get("code_bytes")
    console.print(
        f"  {bundle.get('file_count')} files, "
        f"{_mb(int(code)) if isinstance(code, int) else '?'} "
        f"({_mb(int(size)) if isinstance(size, int) else '?'} compressed)"
    )
    kind = deps.get("kind")
    if kind and kind != "none":
        pkgs = deps.get("packages") or []
        n = len(pkgs) if isinstance(pkgs, list) else 0
        console.print(f"  deps from {deps.get('file') or kind} ({n} packages)")
    else:
        console.print("  no dependency file found (requirements.txt or pyproject.toml)")
    vram, hours = est.get("vram_gb"), est.get("hours")
    if vram is not None or hours is not None:
        parts = []
        if isinstance(vram, int | float):
            parts.append(f"~{fmt_num(float(vram))} GB VRAM ({est.get('vram_source')})")
        if isinstance(hours, int | float):
            parts.append(f"~{fmt_num(float(hours))}h ({est.get('hours_source')})")
        console.print("  estimate: " + ", ".join(parts))
    warnings = bundle.get("warnings") or []
    if isinstance(warnings, list):
        for w in warnings:
            console.print(Text(f"  ! {w}", style="yellow"))


def route_facts(decision: RouteDecision) -> str:
    """What the router assumed about the job (phase 5): "assuming ~1h (estimated), smoke"."""
    parts: list[str] = []
    if decision.hours is not None:
        h = decision.hours
        length = f"{max(1, round(h * 60))}m" if h < 1 else f"{fmt_num(h)}h"
        if decision.hours_source == "heuristic":
            parts.append(f"assuming ~{length} (estimated; set --hours to be exact)")
        else:
            parts.append(f"{length} runtime")
    if decision.smoke:
        parts.append("smoke test")
    return ", ".join(parts)


def print_policy(console: Console, view: PolicyView) -> None:
    """`gpu policy`: the approval rules per audience, in words."""
    from gpu_router.policy import describe_rules

    if not view.editable or view.policy is None:
        console.print(Text(f"approval policy: {view.name} (no editable rules)", style="dim"))
        return
    for title, rules in (("agent jobs", view.policy.agent), ("your jobs", view.policy.user)):
        console.print(Text(title, style="bold"))
        for label, text in describe_rules(rules):
            line = Text("  ")
            line.append(f"{label:<20}", style="dim")
            line.append(text)
            console.print(line)
    where = f" in {view.config_path}" if view.config_path else ""
    console.print()
    console.print(
        Text(f"saved under `policy:`{where}. change: gpu policy set agent.auto_max_hours 2", "dim")
    )
