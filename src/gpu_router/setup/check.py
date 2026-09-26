"""Step 5, check: `gpu doctor`, a 10-second GPU smoke test per ready provider, then the
headline "N providers ready, ~X free hrs/month" (phase 8b).

check.doctor   runs doctor in-process against the daemon (the wizard starts it like any
               command; a dry run only looks for a running one) and prints the counts plus
               the rows that need something, each with its fix.
check.smoke    one tiny job per ready provider, through the daemon like any job (so the
               quota ledger sees it, invariant 2): `gpu_smoke.py` prints which GPU it
               sees (nvidia-smi, a CUDA or MPS matmul when torch is there) and writes
               outputs/gpu.json. Asks first (it spends a little quota); providers that
               passed before are not re-run (`--again` re-runs them). Jobs still not
               finished at the deadline are cancelled; one waiting for approval is left
               for the user and named (`manual`, never a failure: the provider stays
               ready). Each job id is recorded in setup.json right after its submit, so
               a run cut short (Ctrl-C) or a later run reattaches to a smoke job still
               in flight instead of submitting a second one, and one still waiting for
               approval is not submitted again.
check.summary  ready = enabled + healthy in the daemon (+ its smoke test, when one ran, did
               not fail). Free hours per month = each ready provider's quota limit (the
               daemon's ledger when it answers, else providers.yaml) in GPU hours, times
               resets per month (weekly x 30.44/7, monthly x 1, daily x 30.44). Unknown
               limits (colab) and this Mac are named, never counted.
"""

from __future__ import annotations

import contextlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from gpu_router.setup.base import Ctx, Outcome
from gpu_router.setup.ui import Mark

if TYPE_CHECKING:
    from gpu_router.api import ProviderView
    from gpu_router.client import GpuClient
    from gpu_router.doctor.model import Report
    from gpu_router.doctor.probe import DaemonInfo
    from gpu_router.models import QuotaSnapshot
    from gpu_router.providers.catalog import ProviderEntry

__all__ = [
    "ITEMS",
    "SMOKE_SCRIPT",
    "FreeHours",
    "free_hours",
    "run",
    "smoke_spec",
    "summary_line",
]

ITEMS = ("check.doctor", "check.smoke", "check.summary")
MONTH_DAYS = 30.44
PER_MONTH = {"weekly": MONTH_DAYS / 7, "monthly": 1.0, "daily": MONTH_DAYS}
SMOKE_NAME = "gpu_smoke.py"
SMOKE_DEADLINE_S = 15 * 60.0
SMOKE_MARKER = "gpu-smoke: "

SMOKE_SCRIPT = '''"""gpu-router setup smoke test: which GPU does this runtime see? (~10 seconds)

Written by `gpu setup`; safe to delete. Prints one `gpu-smoke: {json}` line and writes
outputs/gpu.json. Exit 0 when a GPU answered (or on this Mac), 3 when none did.
"""
import json
import os
import platform
import shutil
import subprocess
import sys
import time

t0 = time.time()
info = {"python": platform.python_version(), "machine": platform.machine(),
        "system": platform.system()}
gpu = None
smi = shutil.which("nvidia-smi")
if smi:
    try:
        out = subprocess.run([smi, "--query-gpu=name,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30).stdout.strip()
        if out:
            info["gpus"] = out.splitlines()
            gpu = info["gpus"][0]
    except Exception as exc:
        info["nvidia_smi_error"] = str(exc)[:200]
try:
    import torch
    info["torch"] = torch.__version__
    if torch.cuda.is_available():
        x = torch.randn(1024, 1024, device="cuda")
        float((x @ x).sum())
        torch.cuda.synchronize()
        info["cuda_matmul"] = "ok"
        gpu = gpu or torch.cuda.get_device_name(0)
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        x = torch.randn(512, 512, device="mps")
        float((x @ x).sum())
        info["mps_matmul"] = "ok"
        gpu = gpu or "Apple MPS"
except Exception as exc:
    info["torch_error"] = type(exc).__name__
local = os.environ.get("GPU_SMOKE_LOCAL") == "1"
if gpu is None and local:
    gpu = "this Mac (%s; MPS not checked: no torch in the job env)" % platform.machine()
info["gpu"] = gpu
info["seconds"] = round(time.time() - t0, 2)
out_dir = os.environ.get("GPU_OUTPUT_DIR") or "outputs"
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, "gpu.json"), "w") as fh:
    json.dump(info, fh, indent=2)
print("gpu-smoke: " + json.dumps(info), flush=True)
sys.exit(0 if gpu else 3)
'''


# =========================================================================== doctor


def _connect(ctx: Ctx) -> tuple[GpuClient | None, DaemonInfo]:
    """A ready daemon (started like any command; a dry run only looks for a running one)."""
    from gpu_router.doctor.probe import DaemonInfo, daemon_from_client, probe_daemon
    from gpu_router.errors import GpuRouterError

    if ctx.opts.dry_run:
        info = probe_daemon(ctx.env.paths)
        return (info.client if info.up else None), info
    try:
        client = ctx.env.client()
    except GpuRouterError as exc:
        return None, DaemonInfo(error=exc.message, hint=exc.hint)
    return client, daemon_from_client(client)


def _doctor(ctx: Ctx, daemon: DaemonInfo) -> Report | None:
    from gpu_router.doctor.model import Status
    from gpu_router.doctor.render import MARKS
    from gpu_router.doctor.runner import run_doctor

    item = "check.doctor"
    probe = ctx.env.probe(daemon=daemon, deadline_s=45.0)
    ctx.ui.say("running gpu doctor ...", "dim")
    report = run_doctor(probe)
    for c in report.checks:
        if c.status in (Status.FAIL, Status.WARN):
            ctx.ui.item(MARKS[c.status], f"{c.title}: {c.summary}", c.fix)
    n = report.counts
    summary = (
        f"doctor: {len(report.checks)} checks, {n.get('ok', 0)} ok, {n.get('warn', 0)} warn, "
        f"{n.get('fail', 0)} fail, {n.get('skip', 0)} skipped (gpu doctor shows every row)"
    )
    ctx.done(
        item,
        Outcome.DONE if report.ok else Outcome.FAILED,
        summary,
        None if report.ok else "gpu doctor",
    )
    return report


# =========================================================================== smoke


def smoke_dir(ctx: Ctx) -> Path:
    return ctx.env.paths.home / "setup" / "smoke"


def write_smoke_project(ctx: Ctx) -> Path:
    d = smoke_dir(ctx)
    for part in (d.parent, d):  # <home>/setup and its smoke dir: private, like the data dir
        part.mkdir(mode=0o700, exist_ok=True)
        part.chmod(0o700)
    script = d / SMOKE_NAME
    if not script.is_file() or script.read_text(encoding="utf-8") != SMOKE_SCRIPT:
        script.write_text(SMOKE_SCRIPT, encoding="utf-8")
    return d


def smoke_spec(project: Path, view: ProviderView) -> Any:
    from gpu_router.models import JobSpec, Source

    env = {"GPU_SMOKE_LOCAL": "1"} if view.kind == "local" else {}
    return JobSpec(
        name=f"setup smoke {view.name}",
        project_dir=str(project),
        script=SMOKE_NAME,
        provider=view.name,
        hours=0.1,
        env=env,
        checkpoint_interval_min=0,
        max_attempts=1,
        source=Source.CLI,
        labels={"via": "gpu setup"},
    )


def _cost_note(entry: ProviderEntry | None, view: ProviderView) -> str:
    if view.kind == "local":
        return "free"
    if entry is None:
        return "a few minutes of its quota"
    unit = str(entry.quota.unit)
    if unit == "credits":
        rate = entry.options.get("quota_per_gpu_hour")
        per = (
            f" (~{float(rate) * 0.1:.2f} credits at most)" if isinstance(rate, int | float) else ""
        )
        return f"a few minutes of credits{per}"
    if entry.quota.limit is None:
        return "one short session of its dynamic quota"
    return f"a few minutes of {entry.quota.limit:g}h/{entry.quota.reset.removesuffix('ly')}"


def _entries(ctx: Ctx) -> dict[str, ProviderEntry]:
    from gpu_router.setup.providers import catalog_or_none

    catalog = catalog_or_none(ctx.env)
    return {e.name: e for e in catalog.ordered()} if catalog is not None else {}


def _views(client: GpuClient) -> list[ProviderView]:
    """Enabled providers; one whose health the daemon has not checked yet is checked now
    (a fresh daemon may not have run its health loop)."""
    from gpu_router.errors import GpuRouterError
    from gpu_router.models import ProviderHealth

    out: list[ProviderView] = []
    for v in client.providers():
        if not v.enabled:
            continue
        if v.health is ProviderHealth.UNKNOWN:
            with contextlib.suppress(GpuRouterError):
                v = client.healthcheck(v.name)
        out.append(v)
    return out


def _ready_views(client: GpuClient) -> list[ProviderView]:
    from gpu_router.models import ProviderHealth

    return [v for v in _views(client) if v.health is ProviderHealth.OK]


def _smoke_candidates(
    ctx: Ctx, client: GpuClient
) -> tuple[list[ProviderView], list[str], dict[str, str]]:
    """(to test, notes about the ones not tested, provider -> job id of an earlier smoke
    job still in flight to reattach to)."""
    from gpu_router.errors import GpuRouterError
    from gpu_router.models import ProviderHealth
    from gpu_router.statemachine import JobState, is_terminal

    test, notes = [], []
    reattach: dict[str, str] = {}
    for v in _views(client):
        if v.health is not ProviderHealth.OK:
            why = v.health_reason or str(v.health)
            notes.append(f"{v.name}: not tested ({why})")
            continue
        prev = ctx.state.smoke.get(v.name)
        if prev and prev.get("pending") and prev.get("job"):
            try:
                job = client.job(str(prev["job"])).job
            except GpuRouterError:
                job = None
            if job is not None and job.state is JobState.AWAITING_APPROVAL:
                notes.append(
                    f"{v.name}: smoke job {job.short_id} still waits for your approval: "
                    f"gpu approve {job.short_id}"
                )
                continue
            if job is not None and not is_terminal(job.state):
                reattach[v.name] = job.id
                test.append(v)
                continue
            if job is not None and job.state is JobState.DONE:
                _record(ctx, v.name, ok=True, job=job.id, gpu=job.gpu, summary="passed")
                notes.append(f"{v.name}: passed (smoke job {job.short_id})")
                continue
        if prev and prev.get("ok") and not (ctx.opts.again or ctx.explicit("check.smoke")):
            notes.append(f"{v.name}: passed before ({prev.get('gpu') or 'ok'}); --again re-runs it")
            continue
        test.append(v)
    return test, notes, reattach


def _record(
    ctx: Ctx,
    provider: str,
    *,
    ok: bool | None,
    job: str | None,
    gpu: str | None = None,
    summary: str = "",
) -> None:
    """setup.json's smoke record: ok True/False, or None + pending while the job is in
    flight or waits for approval."""
    rec: dict[str, Any] = {
        "ok": ok,
        "at": ctx.env.clock.now(),
        "job": job,
        "gpu": gpu,
        "summary": summary,
    }
    if ok is None:
        rec["pending"] = True
    ctx.state.record_smoke(provider, rec)


@dataclass
class _Run:
    view: ProviderView
    job_id: str | None = None
    state: str = "submitting"
    started: float = 0.0
    done: bool = False
    ok: bool = False
    pending: bool = False  # waits for approval: neither passed nor failed
    short: str | None = None  # the job's short id once seen
    summary: str = ""
    gpu: str | None = None


def _smoke_line(client: GpuClient, job_id: str) -> dict[str, Any] | None:
    from gpu_router.errors import GpuRouterError

    try:
        for rec in client.logs(job_id):
            line = rec.line or ""
            if line.startswith(SMOKE_MARKER):
                try:
                    doc = json.loads(line[len(SMOKE_MARKER) :])
                except ValueError:
                    return None
                return doc if isinstance(doc, dict) else None
    except GpuRouterError:
        return None
    return None


def _smoke(ctx: Ctx, client: GpuClient | None) -> dict[str, bool]:
    """provider -> passed, for the providers tested now."""
    from gpu_router.errors import GpuRouterError
    from gpu_router.statemachine import JobState

    item = "check.smoke"
    if client is None:
        ctx.done(item, Outcome.SKIPPED, "smoke test: no daemon to run it through")
        return {}
    if not ctx.opts.smoke:
        ctx.done(item, Outcome.SKIPPED, "smoke test: skipped (--no-smoke)")
        return {}
    try:
        views, notes, reattach = _smoke_candidates(ctx, client)
    except GpuRouterError as exc:
        ctx.done(item, Outcome.FAILED, f"smoke test: cannot list providers: {exc.message}")
        return {}
    for note in notes:
        ctx.ui.item(Mark.SKIP, note)
    if not views:
        ctx.done(item, Outcome.SKIPPED, "smoke test: nothing new to test")
        return {}
    entries = _entries(ctx)
    ctx.ui.say("a 10-second job on each (plus the provider's start-up, 1-5 min):", "dim")
    for v in views:
        gpus = "/".join(v.gpus) or "?"
        ctx.ui.say(f"  {v.name:<10} {gpus:<12} {_cost_note(entries.get(v.name), v)}", "dim")
    fresh = [v for v in views if v.name not in reattach]
    if ctx.opts.dry_run:
        ctx.dry(item, "run the smoke test on " + ", ".join(v.name for v in views))
        return {}
    answer: bool | None = True
    if fresh:
        answer = ctx.confirm(
            item, "run the smoke test on " + ", ".join(v.name for v in fresh) + "?", default=True
        )
    if answer is None:
        ctx.not_asked(item, "smoke test: not run", "gpu setup --only check.smoke")
        return {}
    if not answer:
        ctx.declined(item, "smoke test not run", "gpu setup --only check.smoke")
        return {}
    clock = ctx.env.clock
    runs = [_Run(v) for v in views]
    project = write_smoke_project(ctx) if fresh else None
    for r in runs:
        if r.view.name in reattach:
            r.job_id, r.started = reattach[r.view.name], clock.now()
            ctx.ui.item(Mark.WAIT, f"{r.view.name}: back to smoke job {r.job_id[:4]}")
            continue
        assert project is not None
        try:
            job = client.submit(smoke_spec(project, r.view))
        except GpuRouterError as exc:
            r.done, r.summary = True, f"could not submit: {exc.message}"
            continue
        r.job_id, r.state, r.started = job.id, str(job.state), clock.now()
        # recorded at once: a run cut short here reattaches instead of submitting again
        _record(ctx, r.view.name, ok=None, job=job.id, summary="submitted")
        ctx.ui.item(Mark.WAIT, f"{r.view.name}: job {job.short_id} submitted")
    deadline = time.monotonic() + SMOKE_DEADLINE_S
    while any(not r.done for r in runs) and time.monotonic() < deadline:
        for r in runs:
            if r.done or r.job_id is None:
                continue
            try:
                job = client.job(r.job_id).job
            except GpuRouterError:
                continue
            state = str(job.state)
            if state != r.state:
                r.state = state
                if job.state is JobState.RUNNING:
                    ctx.ui.say(f"{r.view.name}: running on {job.gpu or '?'}", "dim")
            r.short = job.short_id
            if job.state is JobState.AWAITING_APPROVAL:
                r.done, r.pending = True, True
                r.summary = (
                    f"waits for your approval ({job.approval_reason or 'policy'}): "
                    f"gpu approve {job.short_id}"
                )
            elif job.state is JobState.DONE:
                doc = _smoke_line(client, r.job_id) or {}
                r.gpu = str(doc.get("gpu") or job.gpu or "?")
                secs = (job.finished_at or clock.now()) - (job.created_at or r.started)
                r.done, r.ok = True, True
                r.summary = f"saw {r.gpu} ({int(secs // 60)}m{int(secs % 60):02d}s end to end)"
            elif job.state in (JobState.FAILED, JobState.CANCELLED, JobState.DENIED):
                doc = _smoke_line(client, r.job_id) or {}
                why = job.message or str(job.failure_kind or job.state)
                if job.exit_code == 3:
                    why = "the job ran but saw no GPU"
                r.done, r.summary = True, f"{state}: {why}"
                r.gpu = doc.get("gpu")
        if any(not r.done for r in runs):
            time.sleep(ctx.env.smoke_poll_s)
    for r in runs:
        if not r.done and r.job_id is not None:
            with contextlib.suppress(GpuRouterError):
                client.cancel(r.job_id)
            r.summary = (
                f"not finished after {int(SMOKE_DEADLINE_S // 60)}m (was {r.state}); cancelled"
            )
    passed: dict[str, bool] = {}
    waiting: list[_Run] = []
    for r in runs:
        if r.pending:
            waiting.append(r)
            ctx.ui.item(Mark.WAIT, f"{r.view.name}: {r.summary}")
            _record(ctx, r.view.name, ok=None, job=r.job_id, summary=r.summary)
            continue
        passed[r.view.name] = r.ok
        mark = Mark.DONE if r.ok else Mark.FAIL
        ctx.ui.item(mark, f"{r.view.name}: {r.summary}")
        _record(ctx, r.view.name, ok=r.ok, job=r.job_id, gpu=r.gpu, summary=r.summary)
    good = [n for n, ok in passed.items() if ok]
    bad = [n for n, ok in passed.items() if not ok]
    summary = f"smoke test: {len(good)} of {len(passed)} passed"
    if bad:
        summary += f" (failed: {', '.join(bad)}; gpu logs <id> shows why)"
    if waiting:
        summary += f"; {', '.join(r.view.name for r in waiting)} waiting for your approval"
    if bad:
        ctx.done(item, Outcome.FAILED, summary)
    elif waiting:
        fix = " && ".join(f"gpu approve {r.short or r.job_id}" for r in waiting)
        ctx.done(item, Outcome.MANUAL, summary, fix)
    else:
        ctx.done(item, Outcome.DONE, summary)
    return passed


# =========================================================================== summary


@dataclass(frozen=True)
class FreeHours:
    provider: str
    hours: float | None  # per month; None = not counted
    note: str


def free_hours(entries: list[ProviderEntry], quotas: dict[str, QuotaSnapshot]) -> list[FreeHours]:
    from gpu_router.quota.ledger import to_gpu_hours

    out: list[FreeHours] = []
    for entry in entries:
        if entry.quota.reset == "none":
            out.append(FreeHours(entry.name, None, "unlimited, for smoke tests"))
            continue
        q = quotas.get(entry.name)
        limit = q.limit if q is not None and q.limit is not None else entry.quota.limit
        source = (
            "live" if q is not None and q.limit is not None and q.source == "live" else "catalog"
        )
        if limit is None:
            out.append(FreeHours(entry.name, None, "dynamic, not counted"))
            continue
        factor = PER_MONTH.get(entry.quota.reset)
        hours = to_gpu_hours(entry, float(limit))
        if factor is None or hours is None:
            out.append(FreeHours(entry.name, None, "limit in an unknown unit or window"))
            continue
        per = entry.quota.reset.removesuffix("ly")
        unit = {"credits": " credits", "usd": " USD"}.get(str(entry.quota.unit), "h")
        note = f"{limit:g}{unit}/{per}"
        if source == "live":
            note += ", live"
        out.append(FreeHours(entry.name, hours * factor, note))
    return out


def summary_line(ready: list[str], hours: list[FreeHours]) -> str:
    counted = [h for h in hours if h.hours is not None]
    total = sum(h.hours or 0.0 for h in counted)
    n = len(ready)
    line = f"{n} provider{'s' if n != 1 else ''} ready, ~{total:.0f} free GPU hrs/month"
    parts = [f"{h.provider} ~{h.hours:.0f}h ({h.note})" for h in counted]
    parts += [f"{h.provider}: {h.note}" for h in hours if h.hours is None]
    return line + (f" ({'; '.join(parts)})" if parts else "")


def _summary(
    ctx: Ctx,
    client: GpuClient | None,
    report: Report | None,
    passed: dict[str, bool],
) -> None:
    from gpu_router.doctor.model import Status
    from gpu_router.errors import GpuRouterError
    from gpu_router.setup.providers import enabled_entries

    item = "check.summary"
    entries = {e.name: e for e in enabled_entries(ctx.env)}
    ready: list[str] = []
    quotas: dict[str, QuotaSnapshot] = {}
    if client is not None:  # the daemon's registry is the truth when it answers
        entries = _entries(ctx)
        try:
            ready = [v.name for v in _ready_views(client)]
            quotas = {q.provider: q for q in client.quota()}
        except GpuRouterError:
            ready = []
    elif report is not None:  # no daemon: providers whose login row is fine
        logins = {c.id: c.status for c in report.checks}
        ready = [
            n
            for n, e in entries.items()
            if e.kind == "local" or logins.get(f"provider.{n}.login") is Status.OK
        ]
    smoke_failed = {n for n, ok in passed.items() if not ok}
    for name, rec in ctx.state.smoke.items():
        if name not in passed and rec.get("ok") is False:
            smoke_failed.add(name)
    ready = [n for n in ready if n in entries and n not in smoke_failed]
    hours = free_hours([entries[n] for n in ready], quotas)
    line = summary_line(ready, hours)
    ctx.done(item, Outcome.DONE if ready else Outcome.FAILED, line, quiet=True)


# =========================================================================== step


def run(ctx: Ctx) -> None:
    wanted = [i for i in ITEMS if ctx.selected(i)]
    if not wanted:
        return
    client, daemon = _connect(ctx)
    if client is None:
        ctx.ui.item(Mark.SKIP, f"daemon: {daemon.error or 'not running'}")
    try:
        report = _doctor(ctx, daemon) if ctx.selected("check.doctor") else None
        passed = _smoke(ctx, client) if ctx.selected("check.smoke") else {}
        if ctx.selected("check.summary"):
            _summary(ctx, client, report, passed)
    finally:
        if client is not None:
            client.close()
