"""`gpu doctor` (Typer) and `/doctor` (shell) front ends (phase 8a).

    gpu doctor [--json] [--timeout S] [--only GROUP ...] [--start] [-v]
    gpu doctor --update-catalog [--yes]

Exit 0 when no check failed (warnings allowed), 1 when one did or when a check crashed or
did not finish by the deadline (it verified nothing; `Report.unknown`); `--update-catalog`
without a terminal and without `--yes` exits 2 and changes nothing. The daemon is never
started unless `--start` is given (a diagnostic observes); `/doctor` in the shell uses the
shell's own connection, which already started it.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Annotated, Any

import typer
from rich.text import Text

if TYPE_CHECKING:
    from gpu_router.doctor.model import Report
    from gpu_router.paths import Paths

__all__ = ["doctor", "register", "shell_doctor"]

JsonOpt = Annotated[bool, typer.Option("--json", help="Machine-readable JSON on stdout.")]
DEFAULT_TIMEOUT_S = 20.0


def doctor(
    as_json: JsonOpt = False,
    timeout: Annotated[
        float, typer.Option("--timeout", min=1, help="Seconds for all checks together (20).")
    ] = DEFAULT_TIMEOUT_S,
    only: Annotated[
        list[str] | None,
        typer.Option(
            "--only",
            help="Only these groups or check ids (daemon, providers, storage, inference, limits, "
            "local, integration, provider.kaggle, ...). Repeatable.",
        ),
    ] = None,
    start: Annotated[
        bool, typer.Option("--start", help="Start the daemon first if it is not running.")
    ] = False,
    update_catalog: Annotated[
        bool,
        typer.Option(
            "--update-catalog",
            help="Write live limits that differ from providers.yaml into your providers.yaml "
            "(shows the diff, asks first).",
        ),
    ] = False,
    yes: Annotated[
        bool, typer.Option("--yes", "-y", help="Do not ask (--update-catalog).")
    ] = False,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Show per-check timings.")
    ] = False,
) -> None:
    """Check every CLI, login, the daemon, the data dir and the Claude Code/Codex setup;
    compare real limits with providers.yaml. Each finding comes with the exact fix."""
    from gpu_router.cli import exitcodes
    from gpu_router.cli.app import Out, guarded
    from gpu_router.doctor.probe import ProbeEnv
    from gpu_router.doctor.render import render_report
    from gpu_router.doctor.runner import run_doctor
    from gpu_router.paths import Paths

    out = Out(as_json)
    with guarded(out):
        paths = Paths.from_env()
        started = False
        if start:
            started = _start_daemon(out)
        env = ProbeEnv.real(paths, deadline_s=timeout)
        env.daemon.started = started
        try:
            report = run_doctor(env, only=set(only) if only else None)
        finally:
            if env.daemon.client is not None:
                env.daemon.client.close()
        update: dict[str, Any] | None = None
        # a check that crashed or did not finish verified nothing: a gate must not pass
        code = exitcodes.OK if report.ok and not report.unknown else exitcodes.ERROR
        if not as_json:
            for item in render_report(report, user_home=env.user_home, verbose=verbose):
                out.console.print(item)
        if update_catalog:
            update, refused = _update_catalog(out, paths, report, yes=yes)
            if refused:
                code = exitcodes.USAGE
        if as_json:
            doc = report.model_dump(mode="json")
            if update is not None:
                doc["catalog_update"] = update
            out.emit(doc)
        raise typer.Exit(code)


def _start_daemon(out: Any) -> bool:
    from gpu_router.daemon.spawn import connect as spawn_connect

    conn = spawn_connect(on_start=lambda: out.note("the daemon is not running; starting it"))
    conn.client.close()
    if conn.started:
        out.note(f"daemon started (pid {conn.pid})")
    return conn.started


def _update_catalog(
    out: Any, paths: Paths, report: Report, *, yes: bool
) -> tuple[dict[str, Any], bool]:
    """(result for --json, refused)."""
    from gpu_router.doctor.catalog_fix import apply_plan, plan_update

    target = paths.user_providers
    if not report.drift:
        if not out.json:
            out.console.print(Text("no drift: providers.yaml is unchanged", style="dim"))
        return {"changed": False, "path": str(target), "reason": "no drift"}, False
    plan = plan_update(target, report.drift, now=report.checked_at)
    if not plan.changed:
        if not out.json:
            out.console.print(
                Text("nothing doctor can write automatically; see the rows above", style="dim")
            )
        return {"changed": False, "path": str(target), "reason": "not auto-fixable"}, False
    if not out.json:
        out.console.print(Text(""))
        out.console.print(Text(plan.diff().rstrip("\n")))
    if not yes:
        if out.json or not (sys.stdin.isatty() and sys.stdout.isatty()):
            msg = "refusing to change providers.yaml without --yes (no terminal to ask)"
            if not out.json:
                out.err.print(Text(f"gpu: {msg}", style="red"))
            return {"changed": False, "path": str(target), "reason": msg}, True
        answer = input(f"write these limits to {target}? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            out.console.print(Text("nothing changed", style="dim"))
            return {"changed": False, "path": str(target), "reason": "declined"}, False
    apply_plan(plan)
    if not out.json:
        out.console.print(Text(f"✓ wrote {target}", style="green"))
        out.console.print(
            Text(
                "  the daemon reads it at start: gpu daemon stop && gpu daemon start",
                style="dim",
            )
        )
    return {
        "changed": True,
        "path": str(target),
        "keys": [f"{d.provider}.{d.key}" for d in plan.items],
        "restart": "gpu daemon stop && gpu daemon start",
    }, False


def register(app: typer.Typer) -> None:
    """Add `gpu doctor` to the main Typer app (called from cli/app.py)."""
    app.command(name="doctor")(doctor)


# --------------------------------------------------------------------------- shell


def shell_doctor(ctx: Any, args: list[str]) -> None:
    """`/doctor [--only GROUP] [--update-catalog [--yes]] [--timeout S] [-v]`: the same checks
    and rendering, through the shell's daemon connection. Output goes to the transcript
    unshellified: fix lines are terminal commands (`gpu login hf` is not `/login`)."""
    from gpu_router.clock import SystemClock
    from gpu_router.doctor.catalog_fix import apply_plan, plan_update
    from gpu_router.doctor.probe import DaemonInfo, ProbeEnv, daemon_from_client
    from gpu_router.doctor.render import render_report
    from gpu_router.doctor.runner import run_doctor
    from gpu_router.errors import GpuRouterError

    update = "--update-catalog" in args
    yes = "--yes" in args or "-y" in args
    verbose = "-v" in args or "--verbose" in args
    timeout = DEFAULT_TIMEOUT_S
    only: set[str] = set()
    for i, word in enumerate(args):
        value = args[i + 1] if i + 1 < len(args) else ""
        if word == "--timeout":
            try:
                timeout = max(1.0, float(value))
            except ValueError:
                timeout = DEFAULT_TIMEOUT_S
        elif word == "--only" and value:
            only.add(value)
    paths = ctx.host.paths
    env = ProbeEnv(paths=paths, clock=SystemClock(), deadline_s=timeout)
    try:
        client = ctx.client()
    except GpuRouterError as exc:
        env.daemon = DaemonInfo(error=exc.message, hint=exc.hint)
    else:
        env.daemon = daemon_from_client(client)
    ctx.sink([Text(f"checking… (up to {timeout:g}s)", style="dim")])
    report = run_doctor(env, only=only or None)
    ctx.sink(render_report(report, user_home=env.user_home, verbose=verbose))
    if not update:
        return
    if not report.drift:
        ctx.sink([Text("no drift: providers.yaml is unchanged", style="dim")])
        return
    plan = plan_update(paths.user_providers, report.drift, now=report.checked_at)
    if not plan.changed:
        ctx.sink([Text("nothing doctor can write automatically", style="dim")])
        return
    ctx.sink([Text(plan.diff().rstrip("\n"))])
    if not yes:
        ctx.sink([Text("/doctor --update-catalog --yes writes it", style="dim")])
        return
    apply_plan(plan)
    ctx.sink(
        [
            Text(f"✓ wrote {paths.user_providers}", style="green"),
            Text("  the daemon reads it at start: gpu daemon stop && gpu daemon start", "dim"),
        ]
    )
