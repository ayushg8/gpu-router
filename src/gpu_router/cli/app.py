"""Typer app for `gpu <command>` (phase 2). See cli/__init__.py and docs/cli.md.

Conventions:
- Every command takes `--json`: exactly one JSON document on stdout (NDJSON for
  `logs --json`), errors as {"error": {code, message, hint, detail}} on stdout, progress
  notes (daemon auto-start) on stderr, so stdout always parses. That includes usage errors
  (`main` catches them) and unexpected exceptions (`guarded`: code `internal`).
- Human output goes through rich (render.py); errors print `gpu: <message>` + hint on stderr.
- Exit codes: exitcodes.py.
- Commands that need the daemon start it in the background if it is not running (spawn.py)
  and say so on stderr.
- `gpu run` argv: gpu options go before the script; after it only RUN_TAIL_* options are
  still gpu's, plus `--gpu <a GPU type the catalog knows>` (D56); everything else belongs
  to the script (split_run_argv, D23).
"""

from __future__ import annotations

import difflib
import json
import re
import shutil
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Any, NoReturn

import click
import typer
from rich.console import Console
from rich.text import Text
from typer.core import TyperCommand

from gpu_router.api import JobDetail, JobView, PolicyView, ProviderView
from gpu_router.cli import exitcodes, render
from gpu_router.client import GpuClient
from gpu_router.clock import SystemClock
from gpu_router.errors import (
    AmbiguousJobRef,
    ApiError,
    GpuRouterError,
    InvalidRequest,
    ProviderNotFound,
)
from gpu_router.models import JobEvent, JobSpec, QuotaSnapshot
from gpu_router.router.base import RouteOutcome
from gpu_router.statemachine import TERMINAL_STATES, JobState, is_terminal


def _click_classes() -> tuple[
    tuple[type[Exception], ...], tuple[type[BaseException], ...], tuple[type[Exception], ...]
]:
    """(usage errors, aborts, exits) from BOTH click copies: typer >= 0.27 vendors its own
    click as `typer._click`, whose exceptions do not subclass the `click` package's."""
    usage: list[type[Exception]] = [click.ClickException]
    aborts: list[type[BaseException]] = [click.exceptions.Abort, typer.Abort]
    exits: list[type[Exception]] = [click.exceptions.Exit, typer.Exit]
    try:
        from typer._click import exceptions as vendored
    except ImportError:  # pragma: no cover - a typer that uses the click package again
        pass
    else:
        usage.append(vendored.ClickException)
    return tuple(usage), tuple(aborts), tuple(exits)


CLICK_USAGE_ERRORS, CLICK_ABORTS, CLICK_EXITS = _click_classes()

HELP = """Run scripts on free cloud GPUs.

Examples:
  gpu setup                        first time: CLIs, logins, launchd, Claude Code, a GPU test
  gpu run train.py                 run on the best free GPU and stream logs
  gpu run --vram 24 -d train.py    need 24GB, don't wait (gpu options go first)
  gpu route train.py               where would it go, and why
  gpu status                       what is running, where, and quota left
  gpu logs a7f2 -f                 follow a job's output
"""

app = typer.Typer(
    name="gpu",
    help=HELP,
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode=None,
    pretty_exceptions_enable=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)

JsonOpt = Annotated[bool, typer.Option("--json", help="Machine-readable JSON on stdout.")]
RefArg = Annotated[str, typer.Argument(help="Job id or unique prefix, e.g. a7f2.")]

POLL_S = 0.5
_clock = SystemClock()


# --------------------------------------------------------------------------- output


class Out:
    """Where output goes for one invocation."""

    def __init__(self, as_json: bool) -> None:
        self.json = as_json
        self.console = Console(highlight=False, soft_wrap=True)
        self.err = Console(stderr=True, highlight=False, soft_wrap=True)
        self.client: GpuClient | None = None  # set by connect()
        self.job: JobView | None = None  # set once `run` has submitted: errors name it

    def emit(self, data: Any) -> None:
        """One JSON document on stdout."""
        sys.stdout.write(json.dumps(data, ensure_ascii=False) + "\n")
        sys.stdout.flush()

    def note(self, message: str) -> None:
        """Progress notes: stderr, so stdout stays machine-readable."""
        self.err.print(Text(f"gpu: {message}", style="dim"), soft_wrap=True)

    def error(self, exc: GpuRouterError) -> None:
        job = self.job
        if self.json:
            body = exc.to_body()
            if isinstance(exc, ApiError):
                body["code"] = exc.raw_code  # e.g. invalid_transition from the daemon
            if job is not None:
                # the job exists and keeps running: an agent must not resubmit blindly
                body["detail"] = {
                    **(body.get("detail") or {}),
                    "job_id": job.id,
                    "short_id": job.short_id,
                    "state": str(job.state),
                }
            self.emit({"error": body})
            return
        self.err.print(Text(f"gpu: {exc.message}", style="red"), soft_wrap=True)
        if exc.hint:
            self.err.print(Text(f"  {exc.hint}", style="dim"), soft_wrap=True)
        if isinstance(exc, AmbiguousJobRef):
            for line in self._matches(exc):
                self.err.print(Text(f"  {line}", style="dim"), soft_wrap=True)
        if job is not None:
            self.err.print(
                Text(
                    f"  job {job.short_id} was submitted and keeps running; "
                    f"gpu status {job.short_id} shows where it is",
                    style="dim",
                ),
                soft_wrap=True,
            )

    def _matches(self, exc: AmbiguousJobRef) -> list[str]:
        """Up to 5 candidate jobs for an ambiguous prefix (id, name, state)."""
        matches = [str(m) for m in (exc.detail.get("matches") or [])]
        lines: list[str] = []
        for job_id in matches[:5]:
            label = job_id
            if self.client is not None:
                try:
                    view = self.client.job(job_id).job
                    label = f"{job_id}  {view.name}  {str(view.state).replace('_', ' ')}"
                except (GpuRouterError, ValueError):
                    pass
            lines.append(label)
        if len(matches) > 5:
            lines.append(f"... and {len(matches) - 5} more")
        return lines

    def internal(self, exc: BaseException) -> None:
        """An exception gpu-router did not expect: still one parseable document."""
        self.error(
            GpuRouterError(
                f"internal error: {type(exc).__name__}: {exc}",
                hint="this is a gpu-router bug; the daemon log may say more "
                "(`gpu daemon status` shows the data dir)",
                detail={"exception": type(exc).__name__},
            )
        )


def _model(obj: Any) -> Any:
    return obj.model_dump(mode="json")


@contextmanager
def guarded(out: Out) -> Iterator[None]:
    """Turn gpu-router errors into messages + exit codes; Ctrl-C into 130; anything else
    into an `internal` error (exit 1), so `--json` stdout always parses."""
    try:
        yield
    except CLICK_EXITS:
        raise
    except CLICK_USAGE_ERRORS:
        raise
    except GpuRouterError as exc:
        out.error(exc)
        raise typer.Exit(exitcodes.for_error(exc)) from None
    except KeyboardInterrupt:
        raise typer.Exit(exitcodes.INTERRUPTED) from None
    except Exception as exc:
        out.internal(exc)
        raise typer.Exit(exitcodes.ERROR) from None


def _exit(code: int) -> NoReturn:
    raise typer.Exit(code)


def connect(out: Out) -> GpuClient:
    """Client for a ready daemon; starts one in the background if needed (spawn.py)."""
    from gpu_router.daemon.spawn import connect as spawn_connect

    conn = spawn_connect(
        on_start=lambda: out.note("the daemon is not running; starting it in the background")
    )
    if conn.started:
        out.note(
            f"daemon started (pid {conn.pid}); `gpu daemon install-launchd` starts it at login"
        )
    out.client = conn.client
    return conn.client


# --------------------------------------------------------------------------- run / route


def _build(
    script: str | None,
    args: list[str],
    *,
    vram: float | None,
    hours: float | None,
    provider: str | None,
    gpu: str | None,
    name: str | None,
    env: list[str] | None,
    project: Path | None,
    smoke: bool = False,
    data: list[str] | None = None,
) -> JobSpec:
    from gpu_router.jobspec import Flags, build_spec, find_project_root, parse_env_pairs

    cwd = Path.cwd()
    root = find_project_root(project.resolve()) if project else None
    if project is not None:
        cwd = project.resolve()
    flags = Flags(
        script=script,
        args=args if (script is not None or args) else None,
        vram_gb=vram,
        hours=hours,
        provider=provider.strip().lower() if provider else None,
        gpu=gpu,
        name=name,
        env=parse_env_pairs(env or []),
        smoke=True if smoke else None,
    )
    spec, _doc, _root = build_spec(flags, cwd=cwd, project_root=root)
    if data:
        spec = _with_data(spec, data, cwd)
    return spec


def _data_mount(raw: str) -> str:
    """A mount name from a path's basename: lowercase, [a-z0-9._-], <= 64 chars."""
    import re

    name = re.sub(r"[^a-z0-9._-]+", "-", raw.lower()).strip("-.")
    return (name or "data")[:64]


def _with_data(spec: JobSpec, items: list[str], cwd: Path) -> JobSpec:
    """`--data [MOUNT=]PATH|URI` (repeatable) merged into gpu.yaml's `data:` by mount; the
    flag wins for a mount both name (phase 5)."""
    from gpu_router.models import DataRef

    merged = {ref.mount: ref for ref in spec.data}
    for item in items:
        mount, eq, target = item.partition("=")
        if not eq or "/" in mount or ":" in mount:
            mount, target = "", item
        target = target.strip()
        if not target:
            raise InvalidRequest(f"--data {item!r} names nothing", hint="--data [NAME=]PATH")
        if target.startswith("hf://"):
            name = mount or _data_mount(target.rstrip("/").rsplit("/", 1)[-1])
            ref = DataRef(mount=name, uri=target)
        else:
            path = Path(target).expanduser()
            path = (path if path.is_absolute() else cwd / path).resolve()
            if not path.exists():
                raise InvalidRequest(
                    f"--data {target}: {path} does not exist",
                    hint="give a file or directory on this Mac (or an hf://datasets/... URI)",
                    detail={"path": str(path)},
                )
            name = mount or _data_mount(path.name)
            try:
                ref = DataRef(mount=name, path=str(path))
            except ValueError:
                raise InvalidRequest(
                    f"--data: {name!r} is not a usable mount name",
                    hint="use NAME=PATH with NAME in [a-z0-9._-]",
                ) from None
        merged[ref.mount] = ref
    return JobSpec.model_validate(
        {**spec.model_dump(), "data": [ref.model_dump() for ref in merged.values()]}
    )


# gpu run options (must match the run() signature below). Before the script every one of
# them is gpu's; after it only RUN_TAIL_* are (plus `--help`), the rest goes to the script.
RUN_VALUE_OPTS = frozenset(
    {
        "--vram",
        "--hours",
        "--provider",
        "-p",
        "--gpu",
        "--name",
        "--env",
        "-e",
        "--project",
        "-C",
        "--data",
    }
)
RUN_FLAG_OPTS = frozenset(
    {
        "--wait",
        "-w",
        "--detach",
        "-d",
        "--dry-run",
        "--smoke",
        "--json",
        "--help",
        "-h",
        "--as-agent",
    }
)
# `-p` stays gpu's after the script: a value that is not a provider name fails loudly
# (exit 2/4), unlike `-d`/`-w`/`-e`, which would silently change what runs.
RUN_TAIL_VALUE_OPTS = frozenset({"--vram", "--hours", "--provider", "-p"})
RUN_TAIL_FLAG_OPTS = frozenset(
    {"--json", "--wait", "--detach", "--dry-run", "--help", "--as-agent"}
)
# `--gpu VALUE` after the script is gpu's only when VALUE names a GPU type the catalog
# offers (`--gpu L4`); `--gpu 0` / `--gpu cuda:0` stay the script's (D56, refines D23).
RUN_TAIL_GPU_OPT = "--gpu"
_SHORT_FLAGS = frozenset(o[1] for o in RUN_FLAG_OPTS if len(o) == 2)
_SHORT_VALUES = frozenset(o[1] for o in RUN_VALUE_OPTS if len(o) == 2)
_META_ARGS = "gpu_run.script_args"
_META_CLASHES = "gpu_run.clashes"
#: GPU types every packaged catalog has offered, used when the catalog cannot be read
_FALLBACK_GPU_TYPES = frozenset({"T4", "L4", "P100", "MPS"})
#: a GPU model name even when no provider offers it (L4, A100-40GB, H100, RTX4090): such a
#: value is gpu's, so the router refuses it plainly instead of the script getting it
_GPU_MODEL = re.compile(r"^(?:[1-9]\d?X)?(?!GPU)[A-Z]{1,3}\d{1,4}[A-Z]{0,2}(?:-\d{1,3}GB)?$")


def catalog_gpu_types() -> frozenset[str]:
    """Upper-cased GPU type names of the catalog the daemon uses (packaged + the user's
    providers.yaml); the packaged defaults if it cannot be read."""
    try:
        from gpu_router.cli.lanes import load_listing_catalog

        catalog = load_listing_catalog()
    except Exception:  # argv parsing must never fail on a catalog problem
        catalog = None
    if catalog is None:
        return _FALLBACK_GPU_TYPES
    names = {g.name.upper() for e in catalog.providers.values() for g in e.gpus}
    return frozenset(names) or _FALLBACK_GPU_TYPES


def split_run_argv(
    argv: list[str], gpu_types: frozenset[str] | None = None
) -> tuple[list[str], list[str], list[str]]:
    """Split `gpu run` argv into (argv for click, script args, gpu flags that were passed
    to the script although gpu also has them).

    - Before the script: gpu options, long or short (clusters like `-wd` included).
    - The first token that is not a gpu option is the script, or, if it looks like an
      option (`--epochs`), the first script arg for gpu.yaml's script.
    - After the script: only `--json --wait --detach --dry-run --help --vram --hours
      --provider/-p` are gpu's, plus `--gpu X` when X is a GPU type in `gpu_types`
      (default: the catalog's, `catalog_gpu_types`), so `train.py -p lightning --gpu L4`
      asks for the L4 (D56); every other token (other short flags, `--name`, `--gpu 0`, ...)
      goes to the script. `--` ends gpu parsing: everything after it goes to the script
      verbatim.
    """
    head: list[str] = []
    script: list[str] = []
    tail: list[str] = []
    verbatim: list[str] = []
    i, n = 0, len(argv)
    while i < n:
        tok = argv[i]
        if tok == "--":
            rest = argv[i + 1 :]
            if rest and not rest[0].startswith("-"):
                script, verbatim = [rest[0]], rest[1:]
            else:
                verbatim = rest
            break
        if tok.startswith("--"):
            name, eq, _ = tok.partition("=")
            if name in RUN_FLAG_OPTS and not eq:
                head.append(tok)
                i += 1
                continue
            if name in RUN_VALUE_OPTS:
                head.append(tok)
                i += 1
                if not eq and i < n:
                    head.append(argv[i])
                    i += 1
                continue
            tail = argv[i:]  # unknown option: script args for gpu.yaml's script
            break
        if tok.startswith("-") and len(tok) > 1:
            takes_value, known = False, True
            for j, ch in enumerate(tok[1:], start=1):
                if ch in _SHORT_FLAGS:
                    continue
                if ch in _SHORT_VALUES:
                    takes_value = j == len(tok) - 1  # else the value is attached: -pkaggle
                    break
                known = False
                break
            if not known:
                tail = argv[i:]
                break
            head.append(tok)
            i += 1
            if takes_value and i < n:
                head.append(argv[i])
                i += 1
            continue
        script = [tok]
        tail = argv[i + 1 :]
        break

    tail_opts: list[str] = []
    script_args: list[str] = []
    clashes: list[str] = []
    i, n = 0, len(tail)
    while i < n:
        tok = tail[i]
        if tok == "--":
            verbatim = tail[i + 1 :] + verbatim
            break
        name, eq, _ = tok.partition("=")
        if name in RUN_TAIL_FLAG_OPTS and not eq:
            tail_opts.append(tok)
            i += 1
            continue
        if name in RUN_TAIL_VALUE_OPTS:
            tail_opts.append(tok)
            i += 1
            if not eq and i < n:
                tail_opts.append(tail[i])
                i += 1
            continue
        if name == RUN_TAIL_GPU_OPT:
            value = tok[len(name) + 1 :] if eq else (tail[i + 1] if i + 1 < n else None)
            if gpu_types is None:
                gpu_types = catalog_gpu_types()
            if value is not None and (
                value.upper() in gpu_types or _GPU_MODEL.match(value.upper()) is not None
            ):
                tail_opts.extend([tok] if eq else [tok, value])
                i += 1 if eq else 2
                continue
        if name in RUN_VALUE_OPTS or name in RUN_FLAG_OPTS:
            clashes.append(name)
        script_args.append(tok)
        i += 1
    return [*head, *script, *tail_opts], [*script_args, *verbatim], clashes


class _RunCommand(TyperCommand):
    """`gpu run` parses its own argv split (split_run_argv); click sees only gpu's part."""

    def parse_args(self, ctx: typer.Context, args: list[str]) -> list[str]:  # type: ignore[override]
        click_args, script_args, clashes = split_run_argv(list(args))
        ctx.meta[_META_ARGS] = script_args
        ctx.meta[_META_CLASHES] = clashes
        return super().parse_args(ctx, click_args)


@app.command(cls=_RunCommand)
def run(
    ctx: typer.Context,
    script: Annotated[
        str | None,
        typer.Argument(help="Script (.py) or command to run; default from gpu.yaml."),
    ] = None,
    vram: Annotated[float | None, typer.Option("--vram", help="GB of GPU memory needed.")] = None,
    hours: Annotated[float | None, typer.Option("--hours", help="Expected runtime, hours.")] = None,
    provider: Annotated[
        str | None, typer.Option("--provider", "-p", help="Run only on this provider.")
    ] = None,
    gpu: Annotated[str | None, typer.Option("--gpu", help="GPU type, e.g. T4.")] = None,
    name: Annotated[str | None, typer.Option("--name", help="Job name.")] = None,
    env: Annotated[
        list[str] | None, typer.Option("--env", "-e", help="NAME=VALUE (repeatable).")
    ] = None,
    data: Annotated[
        list[str] | None,
        typer.Option(
            "--data",
            help="Dataset [NAME=]PATH or hf://datasets/... (repeatable): uploaded once, "
            "cached by content, at $GPU_DATA_DIR/NAME on the GPU.",
        ),
    ] = None,
    wait: Annotated[
        bool | None,
        typer.Option(
            "--wait/--detach",
            "-w/-d",
            help="Stream logs until it finishes (default), or return right away "
            "(default with --json).",
        ),
    ] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Show the job and where it would go; submit nothing.")
    ] = False,
    smoke: Annotated[
        bool, typer.Option("--smoke", help="Quick smoke test: prefer this Mac (MPS).")
    ] = False,
    project: Annotated[
        Path | None, typer.Option("--project", "-C", help="Run as if started in this directory.")
    ] = None,
    as_agent: Annotated[
        bool,
        typer.Option(
            "--as-agent",
            help="Submit as an AI agent's job (agent approval rules). Automatic when "
            "CLAUDECODE=1, CODEX_SANDBOX or GPU_ROUTER_AGENT=1 is set.",
        ),
    ] = False,
    as_json: JsonOpt = False,
) -> None:
    """Submit a job: package the project, pick a free GPU, stream logs.

    gpu options go before the script: gpu run --name exp1 -p kaggle train.py --lr 3e-4.
    After the script only --json, --wait/--detach, --dry-run, --vram, --hours,
    --provider/-p and --gpu with a GPU type (T4, L4, ...) are still read as gpu options;
    everything else goes to the script, and everything after `--` goes to it verbatim.
    """
    out = Out(as_json)
    extra = list(ctx.meta.get(_META_ARGS, []))
    for flag in dict.fromkeys(ctx.meta.get(_META_CLASHES, [])):
        out.note(
            f"`{flag}` after the script is passed to the script; gpu options go before it "
            f"(gpu run {flag} ... <script>), or after `--` to silence this"
        )
    with guarded(out):
        spec = _build(
            script,
            extra,
            vram=vram,
            hours=hours,
            provider=provider,
            gpu=gpu,
            name=name,
            env=env,
            project=project,
            smoke=smoke,
            data=data,
        )
        spec = _agent_spec(spec, out, as_agent=as_agent)
        import os

        from gpu_router.origin import with_origin

        spec = with_origin(spec, os.environ)  # the Claude Code session it came from, D56
        client = connect(out)
        _check_provider(client, spec)
        if dry_run:
            from gpu_router.cli.bundling import preview

            bundle = preview(spec)
            decision = client.route(spec)
            if as_json:
                out.emit(
                    {
                        "dry_run": True,
                        "spec": _model(spec),
                        "route": _model(decision),
                        "bundle": bundle,
                    }
                )
            else:
                render.print_route(out.console, spec, decision)
                if bundle is not None:
                    render.print_bundle(out.console, bundle)
                out.console.print()
                out.console.print(Text("dry run: nothing submitted", style="dim"))
            _exit(exitcodes.NO_FIT if decision.outcome is RouteOutcome.NO_FIT else exitcodes.OK)

        job = client.submit(spec)
        out.job = job
        should_wait = wait if wait is not None else not as_json
        if not should_wait:
            if as_json:
                out.emit({"job": _model(job)})
            else:
                _print_submitted(out, job)
                out.console.print(
                    Text(f"  gpu logs {job.short_id} -f  ·  gpu status {job.short_id}", style="dim")
                )
            _exit(exitcodes.OK)
        if not as_json:
            _print_submitted(out, job)
        try:
            final = follow_job(client, job, out, stream=not as_json)
        except KeyboardInterrupt:
            last = out.job or job
            if as_json:
                # Ctrl-C detaches, never cancels: say which job keeps running
                out.emit({"job": _model(last), "detached": True})
            else:
                out.console.print()
                out.console.print(
                    Text(
                        f"detached; job {job.short_id} keeps running. "
                        f"gpu logs {job.short_id} -f to reattach, gpu cancel {job.short_id} "
                        "to stop it",
                        style="yellow",
                    )
                )
            raise
        if as_json:
            out.emit({"job": _model(final)})
        else:
            _print_final(out, final)
        _exit(exitcodes.for_state(final.state))


def _agent_spec(spec: JobSpec, out: Out, *, as_agent: bool) -> JobSpec:
    """`gpu run` from an AI agent (--as-agent, or its environment says so) submits an
    agent job: the agent approval rules and intake checks apply, as through the MCP
    server (agent.py, D48). Otherwise the spec is unchanged."""
    from gpu_router.agent import agent_marker, check_agent_spec, mark_agent

    marker = "--as-agent" if as_agent else agent_marker()
    if marker is None:
        return spec
    check_agent_spec(spec)
    out.note(f"submitting as an agent job ({marker}): the agent approval rules apply")
    return mark_agent(spec, "cli-agent")


def _check_provider(client: GpuClient, spec: JobSpec) -> None:
    """A misspelled --provider / gpu.yaml provider is exit 4 with a suggestion, not a job
    that fails a second later with no_provider_fits."""
    if spec.provider is None:
        return
    names = [p.name for p in client.providers()]
    if spec.provider in names:
        return
    close = difflib.get_close_matches(spec.provider, names, n=1, cutoff=0.5)
    known = ", ".join(names) or "none"
    raise ProviderNotFound(
        f"there is no provider named {spec.provider!r}",
        hint=(f"did you mean {close[0]}? " if close else "") + f"known providers: {known}",
        detail={"provider": spec.provider, "known": names},
    )


def _print_submitted(out: Out, job: JobView) -> None:
    line = Text()
    line.append(f"{render.ICON_WAITING} ", style="yellow")
    line.append(f"submitted job {job.short_id}", style="bold")
    line.append(f"  {render.entry_text(job.spec)}")
    out.console.print(line)


def _print_final(out: Out, job: JobView) -> None:
    now = _clock.now()
    icon, style = render.state_style(job.state)
    line = Text()
    line.append(f"{icon} ", style=style)
    took = render.elapsed(job, now)
    if job.state is JobState.DONE:
        line.append(f"{job.name} done", style="bold")
        line.append(f" in {render.duration(took)} on {render.where(job)}")
        line.append(f"  → {render.rel_path(job.outputs_dir)}")
    elif job.state is JobState.FAILED:
        line.append(f"{job.name} failed", style="bold red")
        if job.message:
            line.append(f": {job.message}")
    else:
        line.append(f"{job.name} {job.state}", style="bold")
        if job.message:
            line.append(f": {job.message}")
    out.console.print(line)
    hint = render.next_step(job)
    if hint and job.state is not JobState.DONE:
        out.console.print(Text(f"  {hint}", style="dim"))


_LATE_REASONS = frozenset({"fetched", "fetch_failed", "outputs_kept"})


def _split_events(events: list[JobEvent]) -> tuple[list[JobEvent], list[JobEvent]]:
    """(events to print before this tick's log lines, events to print after them).
    A terminal transition and the output notes around it go after the logs, so the
    final lines of output appear before "finished"."""
    for i, ev in enumerate(events):
        terminal = ev.kind == "transition" and ev.to_state is not None and is_terminal(ev.to_state)
        if terminal or ev.reason in _LATE_REASONS:
            return events[:i], events[i:]
    return events, []


def follow_job(client: GpuClient, job: JobView, out: Out, *, stream: bool) -> JobView:
    """Poll the job until it is terminal, printing transitions/notes and new log lines
    when `stream`. Without `stream` (`run --json --wait`) state changes go to stderr as
    notes, so a job waiting for approval is never silent. Keeps `out.job` current (errors
    and Ctrl-C report it). Returns the final JobView. Ctrl-C propagates (the job keeps
    running)."""
    cursor = 0
    positions: dict[int, int] = {}
    shown_attempt: int | None = None
    now = _clock.now
    sid = job.short_id
    while True:
        detail: JobDetail = client.job(job.id)
        out.job = detail.job
        evs = client.events(job.id, after=cursor)
        cursor = evs.next
        early, late = _split_events(list(evs.events))
        if stream:
            for ev in early:
                out.console.print(render.event_text(ev, now()), soft_wrap=True)
        else:
            for ev in evs.events:
                if ev.kind != "transition":
                    continue
                out.note(f"job {sid}: {ev.message}")
                if ev.to_state is JobState.AWAITING_APPROVAL:
                    out.note(
                        f"job {sid} is waiting for approval: gpu approve {sid}  or  gpu deny {sid}"
                    )
        for attempt in detail.attempts:
            pos = positions.get(attempt.n, 0)
            if attempt.log_lines <= pos:
                continue
            for rec in client.logs(job.id, attempt=attempt.n, offset=pos):
                if rec.line is None or rec.offset is None:
                    continue
                if stream:
                    if shown_attempt != attempt.n:
                        if shown_attempt is not None or attempt.n > 1:
                            out.console.print(
                                Text(f"── attempt {attempt.n} on {attempt.provider} ──", "dim")
                            )
                        shown_attempt = attempt.n
                    out.console.print(Text(rec.line), soft_wrap=True)
                pos = rec.offset + 1
            # trailing protocol lines are never returned; skip past them too
            positions[attempt.n] = max(pos, attempt.log_lines)
        if stream:
            for ev in late:
                out.console.print(render.event_text(ev, now()), soft_wrap=True)
        if is_terminal(detail.job.state):
            return detail.job
        time.sleep(POLL_S)


@app.command()
def route(
    script: Annotated[
        str | None, typer.Argument(help="Script (.py) or command; default from gpu.yaml.")
    ] = None,
    vram: Annotated[float | None, typer.Option("--vram", help="GB of GPU memory needed.")] = None,
    hours: Annotated[float | None, typer.Option("--hours", help="Expected runtime, hours.")] = None,
    provider: Annotated[
        str | None, typer.Option("--provider", "-p", help="Only this provider.")
    ] = None,
    gpu: Annotated[str | None, typer.Option("--gpu", help="GPU type, e.g. T4.")] = None,
    smoke: Annotated[
        bool, typer.Option("--smoke", help="Quick smoke test: prefer this Mac (MPS).")
    ] = False,
    project: Annotated[
        Path | None, typer.Option("--project", "-C", help="As if started in this directory.")
    ] = None,
    as_json: JsonOpt = False,
) -> None:
    """Dry run: where a job would go, the candidates and why each was ruled in or out."""
    out = Out(as_json)
    with guarded(out):
        spec = _build(
            script,
            [],
            vram=vram,
            hours=hours,
            provider=provider,
            gpu=gpu,
            name=None,
            env=None,
            project=project,
            smoke=smoke,
        )
        client = connect(out)
        _check_provider(client, spec)
        decision = client.route(spec)
        if as_json:
            out.emit({"spec": _model(spec), "route": _model(decision)})
        else:
            render.print_route(out.console, spec, decision)
        _exit(exitcodes.NO_FIT if decision.outcome is RouteOutcome.NO_FIT else exitcodes.OK)


# --------------------------------------------------------------------------- status / jobs


@app.command()
def status(
    ref: Annotated[str | None, typer.Argument(help="Job id or prefix; omit for overview.")] = None,
    line: Annotated[bool, typer.Option("--line", hidden=True)] = False,
    as_json: JsonOpt = False,
) -> None:
    """What is running, where, and quota left; or one job in detail."""
    if line:  # normally handled by entry.py before typer loads (stdlib fast path)
        from gpu_router.entry import _status_line

        _exit(_status_line(["status", "--line"]))
    out = Out(as_json)
    with guarded(out):
        client = connect(out)
        now = _clock.now()
        if ref is None:
            view = client.status()
            if as_json:
                out.emit(_model(view))
            else:
                provs = list(view.providers)
                if not view.active and not view.recent:
                    provs = _with_quota(client, provs)
                render.print_status(out.console, view.active, view.recent, provs, now)
            return
        detail = client.job(ref)
        if as_json:
            out.emit(_model(detail))
        else:
            render.print_detail(out.console, detail, now)


_ACTIVE = [s for s in JobState if s not in TERMINAL_STATES]
_FINISHED = sorted(TERMINAL_STATES, key=str)


BeforeOpt = Annotated[
    str | None,
    typer.Option(
        "--before",
        help="Only jobs created before this time: the `next_before` of a previous --json "
        "page (ISO-8601) or epoch seconds.",
    ),
]


@app.command()
def jobs(
    show_all: Annotated[bool, typer.Option("--all", "-a", help="Include finished jobs.")] = False,
    here: Annotated[bool, typer.Option("--here", help="Only jobs from this project.")] = False,
    limit: Annotated[int, typer.Option("--limit", "-n", min=1, max=500)] = 50,
    before: BeforeOpt = None,
    as_json: JsonOpt = False,
) -> None:
    """Running and queued jobs (--all adds finished ones)."""
    out = Out(as_json)
    with guarded(out):
        client = connect(out)
        list_jobs(out, client, finished=False, wide=show_all, here=here, limit=limit, before=before)


@app.command()
def history(
    limit: Annotated[int, typer.Option("--limit", "-n", min=1, max=500)] = 20,
    failed: Annotated[bool, typer.Option("--failed", help="Only failed jobs.")] = False,
    here: Annotated[bool, typer.Option("--here", help="Only jobs from this project.")] = False,
    before: BeforeOpt = None,
    as_json: JsonOpt = False,
) -> None:
    """Past jobs: results, failures and handoffs (↪ = moved between providers)."""
    out = Out(as_json)
    with guarded(out):
        client = connect(out)
        list_jobs(out, client, finished=True, wide=failed, here=here, limit=limit, before=before)


def list_jobs(
    out: Out,
    client: GpuClient,
    *,
    finished: bool,
    wide: bool,
    here: bool,
    limit: int,
    before: str | None,
) -> None:
    """`gpu jobs` (finished=False; wide = --all) and `gpu history` (finished=True; wide =
    --failed): the page, its empty state and the "… more" hint. The shell's /jobs and
    /history call this too, so both say the same thing (D44)."""
    if finished:
        states: list[JobState] | None = [JobState.FAILED] if wide else _FINISHED
    else:
        states = None if wide else _ACTIVE
    found = client.jobs(
        states=states, project_dir=_here() if here else None, limit=limit, before=before
    )
    if out.json:
        out.emit(_model(found))
        return
    now = _clock.now()
    if not found.jobs:
        if finished:
            what = "no failed jobs" if wide else "no past jobs"
        else:
            what = "no jobs" if wide else "no running or queued jobs"
        if here:
            what += " from this project"
        if before is not None:
            what += " before that time"
        render.print_empty_state(
            out.console, _with_quota(client, client.providers()), now, what=what
        )
        return
    out.console.print(render.jobs_table(found.jobs, now, finished=finished))
    command = "history" if finished else "jobs"
    flag = "--failed" if finished else "--all"
    _print_more(out, command, found.next_before, limit, flags=_flags(wide, flag, here))


def _flags(on: bool, flag: str, here: bool) -> str:
    return "".join(f" {f}" for f, set_ in ((flag, on), ("--here", here)) if set_)


def _print_more(
    out: Out, command: str, next_before: float | None, limit: int, *, flags: str
) -> None:
    """The list stopped at -n: say so, and how to see the rest (never a silent cut)."""
    if next_before is None:
        return
    from gpu_router.models import to_iso

    out.console.print(
        Text(
            f"… more: gpu {command}{flags} -n {min(500, limit * 2)}, or "
            f"gpu {command}{flags} --before {to_iso(next_before)}",
            style="dim",
        )
    )


def _with_quota(client: GpuClient, providers: list[ProviderView]) -> list[ProviderView]:
    """Fill in quota the daemon has not observed yet (empty states show quota left).
    Short timeout; on any error the providers are returned unchanged."""
    if all(p.quota is not None for p in providers if p.enabled):
        return providers
    try:
        raw = client.request("GET", "/quota", timeout_s=5.0)
        quotas = {q.provider: q for q in (QuotaSnapshot.model_validate(x) for x in raw or [])}
    except (GpuRouterError, ValueError):
        return providers
    return [
        p.model_copy(update={"quota": quotas[p.name]})
        if p.quota is None and p.name in quotas
        else p
        for p in providers
    ]


def _here() -> str:
    from gpu_router.jobspec import find_project_root

    return str(find_project_root())


# --------------------------------------------------------------------------- logs


@app.command()
def logs(
    ref: RefArg,
    follow: Annotated[
        bool, typer.Option("--follow", "-f", help="Keep streaming until the job finishes.")
    ] = False,
    attempt: Annotated[
        int | None, typer.Option("--attempt", min=1, help="Only this attempt.")
    ] = None,
    as_json: JsonOpt = False,
) -> None:
    """A job's output (all attempts, in order). --json prints NDJSON log records."""
    out = Out(as_json)
    with guarded(out):
        client = connect(out)
        detail = client.job(ref)
        job = detail.job
        providers = {a.n: a.provider for a in detail.attempts}
        current: int | None = None
        final_state: JobState | None = None
        count = 0
        for rec in client.logs(job.id, attempt=attempt, follow=follow):
            if as_json:
                out.emit(rec.model_dump(mode="json", exclude_none=True))
            if rec.eof:
                final_state = rec.state
                continue
            if rec.line is None or as_json:
                continue
            count += 1
            if rec.attempt != current:
                if current is not None or (rec.attempt or 1) > 1:
                    prov = providers.get(rec.attempt or 0) or "provider"
                    out.console.print(Text(f"── attempt {rec.attempt} on {prov} ──", "dim"))
                current = rec.attempt
            out.console.print(Text(rec.line), markup=False, highlight=False)
        if as_json:
            _exit(exitcodes.for_state(final_state) if final_state else exitcodes.OK)
        if count == 0 and not follow:
            msg = "no output yet" if not is_terminal(job.state) else "this job produced no output"
            out.err.print(Text(f"gpu: {msg} ({job.state})", style="dim"))
        if final_state is not None:
            _print_final(out, client.job(job.id).job)
            _exit(exitcodes.for_state(final_state))


# --------------------------------------------------------------------------- actions


def _action_result(out: Out, job: JobView, verb: str) -> None:
    if out.json:
        out.emit(_model(job))
        return
    line = Text()
    line.append_text(render.icon_text(job.state))
    line.append(f" job {job.short_id} {verb}", style="bold")
    line.append(f"  ({str(job.state).replace('_', ' ')})")
    out.console.print(line)
    if job.message:
        out.console.print(Text(f"  {job.message}", style="dim"))


@app.command()
def cancel(ref: RefArg, as_json: JsonOpt = False) -> None:
    """Stop a job (idempotent). Remote runs are stopped; outputs so far are kept."""
    out = Out(as_json)
    with guarded(out):
        client = connect(out)
        before = client.job(ref).job
        job = client.cancel(before.id)
        if is_terminal(before.state) and not out.json:
            out.console.print(
                Text(
                    f"job {job.short_id} already finished ({job.state}); nothing to cancel",
                    style="dim",
                )
            )
            return
        _action_result(
            out, job, "cancel requested" if job.state is JobState.CANCELLING else "cancelled"
        )


@app.command()
def approve(
    ref: RefArg,
    reason: Annotated[str | None, typer.Option("--reason", help="Note to record.")] = None,
    as_json: JsonOpt = False,
) -> None:
    """Approve a job that is waiting for your approval."""
    out = Out(as_json)
    with guarded(out):
        job = connect(out).approve(ref, reason=reason)
        _action_result(out, job, "approved")


@app.command()
def deny(
    ref: RefArg,
    reason: Annotated[str | None, typer.Option("--reason", help="Why (recorded).")] = None,
    as_json: JsonOpt = False,
) -> None:
    """Deny a job that is waiting for your approval. Nothing runs."""
    out = Out(as_json)
    with guarded(out):
        job = connect(out).deny(ref, reason=reason)
        _action_result(out, job, "denied")


@app.command()
def fetch(
    ref: RefArg,
    dest: Annotated[Path | None, typer.Option("--dest", help="Also copy the outputs here.")] = None,
    timeout: Annotated[
        float, typer.Option("--timeout", help="Seconds to wait for the download.")
    ] = 600.0,
    as_json: JsonOpt = False,
) -> None:
    """Download a finished job's outputs again (into ./runs/<id>/, or --dest too)."""
    out = Out(as_json)
    with guarded(out):
        client = connect(out)
        detail = client.job(ref)
        job = detail.job
        after = max((e.seq for e in detail.events), default=0)
        client.fetch(job.id)
        if not as_json:
            out.note(f"fetching outputs of job {job.short_id}...")
        result = _wait_fetch(client, job.id, after, timeout)
        job = client.job(job.id).job
        files = int(result.get("files", 0)) if result else 0
        ok = result is not None and bool(result["ok"])
        message = result["message"] if result else f"no answer within {timeout:g}s"
        copied: str | None = None
        copy_error: str | None = None
        if ok and dest is not None and job.outputs_dir:
            copied, copy_error = _copy_outputs(Path(job.outputs_dir), dest)
            if copy_error is not None:
                message = f"{message}; {copy_error}"
        payload = {
            "job": _model(job),
            "fetched": ok,
            "outputs_dir": job.outputs_dir,
            "dest": copied,
            "files": files,
            "message": message,
        }
        if as_json:
            out.emit(payload)
        elif ok:
            where = render.rel_path(copied or job.outputs_dir)
            out.console.print(Text(f"{render.ICON_DONE} {files} files → {where}", style="green"))
            if copy_error is not None:
                out.err.print(Text(f"gpu: {copy_error}", style="red"))
        else:
            out.err.print(Text(f"gpu: {payload['message']}", style="red"))
        _exit(exitcodes.OK if ok and copy_error is None else exitcodes.ERROR)


def _copy_outputs(src: Path, dest: Path) -> tuple[str | None, str | None]:
    """Copy fetched outputs to --dest. Returns (dest path or None, error or None); never
    raises, the download itself already succeeded."""
    target = dest.expanduser().resolve()
    src = src.resolve()
    if target == src:
        return str(target), None  # the outputs are already there
    if target.is_relative_to(src):
        return None, f"copied nothing: {target} is inside the outputs dir {src}"
    try:
        if src.exists():
            shutil.copytree(src, target, dirs_exist_ok=True)
        else:
            target.mkdir(parents=True, exist_ok=True)
    except shutil.Error as exc:
        problems = exc.args[0] if exc.args and isinstance(exc.args[0], list) else []
        why = str(problems[0][2]) if problems and len(problems[0]) == 3 else str(exc)
        return None, f"could not copy everything to {target}: {why}"
    except OSError as exc:
        return None, f"copied nothing to {target}: {exc.strerror or exc}"
    return str(target), None


def _wait_fetch(
    client: GpuClient, job_id: str, after: int, timeout: float
) -> dict[str, Any] | None:
    deadline = time.monotonic() + timeout
    cursor = after
    while time.monotonic() < deadline:
        evs = client.events(job_id, after=cursor)
        cursor = evs.next
        for ev in evs.events:
            if ev.reason == "fetched":
                return {"ok": True, "message": ev.message, "files": ev.detail.get("files", 0)}
            if ev.reason == "fetch_failed":
                return {"ok": False, "message": ev.message}
        time.sleep(POLL_S / 2)
    return None


# --------------------------------------------------------------------------- providers


@app.command()
def quota(
    refresh: Annotated[
        bool, typer.Option("--refresh", help="Ask every live provider again right now.")
    ] = False,
    as_json: JsonOpt = False,
) -> None:
    """Free quota per provider: used, left, when it resets, live or estimated."""
    out = Out(as_json)
    with guarded(out):
        client = connect(out)
        quotas = client.quota(refresh=refresh)
        providers = client.providers()
        from gpu_router.cli.lanes import quota_extras  # phase 7b: the inference lane

        extra, shown = quota_extras(client)
        if as_json:
            out.emit({"quota": [_model(q) for q in quotas], **extra})
            return
        if not providers and not quotas:
            out.console.print(Text("no providers connected yet.", style="dim"))
        else:
            out.console.print(render.quota_table(quotas, providers, _clock.now()))
        for item in shown:
            out.console.print(item)


@app.command()
def providers(as_json: JsonOpt = False) -> None:
    """Connected providers: health, GPUs, session cap, running jobs, quota; then the ones
    gpu-router lists but never routes to (manual, verify at signup, excluded) and the
    inference lane."""
    out = Out(as_json)
    with guarded(out):
        client = connect(out)
        found = client.providers()
        from gpu_router.cli.lanes import provider_extras  # phase 7b: not routed + inference

        extra, shown = provider_extras(client)
        if as_json:
            out.emit({"providers": [_model(p) for p in found], **extra})
            return
        if not found:
            out.console.print(Text("no providers connected yet.", style="dim"))
        else:
            out.console.print(render.providers_table(found, _clock.now()))
        for item in shown:
            out.console.print(item)


# --------------------------------------------------------------------------- policy

policy_app = typer.Typer(
    name="policy",
    help="Approval rules: which jobs run automatically and which ask first.",
    invoke_without_command=True,
    add_completion=False,
    rich_markup_mode=None,
    pretty_exceptions_enable=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)
app.add_typer(policy_app)


def _print_policy(out: Out, view: PolicyView) -> None:
    if out.json:
        out.emit(_model(view))
    else:
        render.print_policy(out.console, view)


@policy_app.callback()
def policy_main(ctx: typer.Context, as_json: JsonOpt = False) -> None:
    """Approval rules (same as `gpu policy show`). Edit: gpu policy set KEY VALUE."""
    if ctx.invoked_subcommand is not None:
        return
    out = Out(as_json)
    with guarded(out):
        _print_policy(out, connect(out).policy())


@policy_app.command("show")
def policy_show(as_json: JsonOpt = False) -> None:
    """Show the approval rules for agent jobs and for your own jobs."""
    out = Out(as_json)
    with guarded(out):
        _print_policy(out, connect(out).policy())


@policy_app.command("set")
def policy_set(
    key: Annotated[
        str,
        typer.Argument(help="Rule, e.g. agent.auto_max_hours (no agent./user. prefix = both)."),
    ],
    value: Annotated[str, typer.Argument(help="New value, e.g. 2, null, 50%, lightning.")],
    as_json: JsonOpt = False,
) -> None:
    """Change one rule; the daemon saves it to config.yaml and uses it right away."""
    from gpu_router.policy import apply_policy_setting

    out = Out(as_json)
    with guarded(out):
        client = connect(out)
        view = client.policy()
        if not view.editable or view.policy is None:
            raise InvalidRequest(
                f"the daemon's approval policy ({view.name}) has no editable rules",
                hint="restart the daemon (gpu daemon stop; gpu daemon start)",
            )
        updated = client.set_policy(apply_policy_setting(view.policy, key, value))
        if not as_json:
            out.console.print(Text(f"{render.ICON_DONE} {key} = {value}", style="green"))
        _print_policy(out, updated)


@policy_app.command("reset")
def policy_reset(as_json: JsonOpt = False) -> None:
    """Back to the default rules (spec defaults)."""
    from gpu_router.policy import PolicyConfig

    out = Out(as_json)
    with guarded(out):
        updated = connect(out).set_policy(PolicyConfig())
        if not as_json:
            out.console.print(Text(f"{render.ICON_DONE} approval rules reset", style="green"))
        _print_policy(out, updated)


# --------------------------------------------------------------------------- secrets

secrets_app = typer.Typer(
    name="secrets",
    help="Keychain secrets that jobs list under `secrets:` in gpu.yaml (values never shown).",
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode=None,
    pretty_exceptions_enable=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)
app.add_typer(secrets_app)


def _register_login() -> None:
    from gpu_router.cli.login import login_app  # phase 5: `gpu login hf`

    app.add_typer(login_app)


_register_login()


def _register_infer() -> None:
    from gpu_router.cli.infer import register  # phase 7b: gpu infer, gpu login groq|gemini|...

    register(app)


_register_infer()


def _register_doctor_notify() -> None:
    from gpu_router.doctor.cli import register  # phase 8a: `gpu doctor`
    from gpu_router.notify.cli import notify_app  # phase 8a: `gpu notify [status|test]`

    register(app)
    app.add_typer(notify_app)


_register_doctor_notify()


def _register_setup() -> None:
    from gpu_router.setup.cli import register  # phase 8b: gpu setup, gpu login kaggle|colab

    register(app)


_register_setup()

SecretNameArg = Annotated[str, typer.Argument(help="Secret name, e.g. HF_TOKEN.")]


def _secret_name(name: str) -> str:
    from gpu_router.models import _ENV_NAME

    clean = name.strip()
    if not _ENV_NAME.match(clean):
        raise InvalidRequest(
            f"{clean!r} is not a secret name",
            hint="use an env-var style name such as HF_TOKEN or WANDB_API_KEY",
            detail={"name": clean},
        )
    return clean


@secrets_app.command("set")
def secrets_set(
    name: SecretNameArg,
    stdin: Annotated[
        bool, typer.Option("--stdin", help="Read the value from stdin instead of a prompt.")
    ] = False,
    as_json: JsonOpt = False,
) -> None:
    """Store a secret in the Keychain (prompted without echo, or read from stdin)."""
    from gpu_router import secrets as secret_store

    out = Out(as_json)
    with guarded(out):
        key = _secret_name(name)
        if stdin or not sys.stdin.isatty():
            value = sys.stdin.read().rstrip("\r\n")
        else:
            import getpass

            value = getpass.getpass(f"value for {key} (not shown): ")
        if not value:
            raise InvalidRequest(f"no value given for {key}; nothing stored")
        secret_store.set_secret(key, value)
        if as_json:
            out.emit({"secret": key, "stored": True})
        else:
            out.console.print(
                Text(f"{render.ICON_DONE} stored {key} in the Keychain", style="green")
            )
            from gpu_router.models import is_reserved_secret

            if is_reserved_secret(key):
                out.console.print(Text(f"  gpu-router uses {key} itself; jobs never get it", "dim"))
            else:
                out.console.print(Text(f"  list it under `secrets: [{key}]` in gpu.yaml", "dim"))


@secrets_app.command("list")
def secrets_list(as_json: JsonOpt = False) -> None:
    """Names of stored secrets (never values)."""
    from gpu_router import secrets as secret_store

    out = Out(as_json)
    with guarded(out):
        names = secret_store.secret_names()
        if as_json:
            out.emit({"secrets": names})
        elif not names:
            out.console.print(Text("no secrets stored. gpu secrets set HF_TOKEN", style="dim"))
        else:
            for n in names:
                out.console.print(n)


@secrets_app.command("rm")
def secrets_rm(name: SecretNameArg, as_json: JsonOpt = False) -> None:
    """Delete a secret from the Keychain (no-op if absent)."""
    from gpu_router import secrets as secret_store

    out = Out(as_json)
    with guarded(out):
        key = _secret_name(name)
        existed = key in secret_store.secret_names()
        secret_store.delete_secret(key)
        if as_json:
            out.emit({"secret": key, "removed": existed})
        else:
            out.console.print(f"removed {key}" if existed else f"{key} was not stored")


# --------------------------------------------------------------------------- daemon


@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    add_help_option=False,
)
def daemon(ctx: typer.Context) -> None:
    """Manage the daemon: run | start | stop | status | install-launchd | uninstall."""
    from gpu_router.daemon.__main__ import main as daemon_main

    _exit(daemon_main(list(ctx.args)))


@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    add_help_option=False,
)
def mcp(ctx: typer.Context) -> None:
    """Serve the MCP tools for agents on stdio (Claude Code, Codex). Phase 6."""
    from gpu_router.mcp.server import main as mcp_main  # entry.py normally gets here first

    _exit(mcp_main(list(ctx.args)))


@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    add_help_option=False,
)
def statusline(ctx: typer.Context) -> None:
    """Claude Code status line: install | uninstall | preview | status. Phase 6b."""
    from gpu_router.statusline.cli import main as statusline_main  # entry.py gets here first

    _exit(statusline_main(list(ctx.args)))


# --------------------------------------------------------------------------- entry


def _wants_json(args: list[str]) -> bool:
    """`--json` among gpu's own arguments (before any `--`)."""
    head = args[: args.index("--")] if "--" in args else args
    return "--json" in head


def _usage_error(exc: Exception, args: list[str]) -> int:
    """A click usage error (unknown command/option, bad value): short message + hint on
    stderr, or the {"error": ...} envelope on stdout with --json. Exit 2."""
    message = exc.format_message() if hasattr(exc, "format_message") else str(exc)
    ctx = getattr(exc, "ctx", None)
    command = ctx.command_path if ctx is not None else "gpu"
    hint = f"see `{command} --help`"
    if ctx is not None and ctx.command.name == "run":
        hint += "; gpu options go before the script, script arguments after it (or after `--`)"
    if _wants_json(args):
        Out(True).emit(
            {
                "error": {
                    "code": "invalid_request",
                    "message": message,
                    "hint": hint,
                    "detail": {"usage": True, "command": command},
                }
            }
        )
        return exitcodes.USAGE
    if type(exc).__name__ == "NoArgsIsHelpError":
        exc.show()  # type: ignore[attr-defined]  # prints the help text
        return exitcodes.USAGE
    err = Console(stderr=True, highlight=False, soft_wrap=True)
    err.print(Text(f"gpu: {message}", style="red"), soft_wrap=True)
    err.print(Text(f"  {hint}", style="dim"), soft_wrap=True)
    return exitcodes.USAGE


def main(argv: list[str] | None = None) -> int:
    """Run the CLI on argv (without the program name); return the exit code. Never lets a
    traceback out: usage errors exit 2, anything unexpected exits 1 (JSON envelope with
    --json)."""
    args = sys.argv[1:] if argv is None else list(argv)
    try:
        rv = app(args=args, prog_name="gpu", standalone_mode=False)
    except CLICK_EXITS as exc:
        return int(getattr(exc, "exit_code", 0))
    except CLICK_USAGE_ERRORS as exc:
        return _usage_error(exc, args)
    except CLICK_ABORTS:
        return exitcodes.INTERRUPTED
    except KeyboardInterrupt:
        return exitcodes.INTERRUPTED
    except GpuRouterError as exc:  # pragma: no cover - commands guard their own errors
        Out(_wants_json(args)).error(exc)
        return exitcodes.for_error(exc)
    except Exception as exc:  # pragma: no cover - guarded() catches these inside commands
        Out(_wants_json(args)).internal(exc)
        return exitcodes.ERROR
    return rv if isinstance(rv, int) else exitcodes.OK


__all__ = ["app", "main"]
