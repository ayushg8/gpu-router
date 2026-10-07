"""Slash commands of the shell (phase 4).

Each command runs in a worker thread with a `Ctx` and prints rich renderables into its
transcript block. Business logic is the CLI's: specs are built with `cli.app._build`,
providers checked with `_check_provider`, fetches awaited with `_wait_fetch`, and output is
rendered by `cli.render` (and the CLI's own `_print_*` helpers, through an `Out` whose
consoles are a `Collector`). CLI hints such as `gpu approve a7f2` are rewritten to
`/approve a7f2` on the way into the transcript (`shellify`).

Every command also answers without the slash (`jobs`, `gpu jobs`), like the spec's "every
slash command also works as a plain command".
"""

from __future__ import annotations

import argparse
import difflib
import io
import os
import re
import shlex
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from rich.cells import cell_len
from rich.console import Console, ConsoleOptions, Group, RenderableType, RenderResult
from rich.padding import Padding
from rich.table import Table
from rich.text import Text

from gpu_router.api import JobView
from gpu_router.cli import render
from gpu_router.client import GpuClient
from gpu_router.clock import SystemClock
from gpu_router.errors import GpuRouterError, InvalidRequest
from gpu_router.models import ProviderHealth, Source
from gpu_router.paths import Paths
from gpu_router.shell.infer import cmd_infer  # phase 7b
from gpu_router.shell.panel import seg
from gpu_router.shell.state import Snapshot
from gpu_router.statemachine import JobState, is_terminal

ArgKind = Literal["script", "job", "approval", "provider", "command"]
_clock = SystemClock()


# --------------------------------------------------------------------------- registry


@dataclass(frozen=True)
class Command:
    name: str
    usage: str
    summary: str
    arg: ArgKind | None = None
    needs_daemon: bool = True
    short: str | None = None  # usage in the popup when the full one is long

    @property
    def hint(self) -> str:
        return self.usage if self.short is None else self.short


COMMANDS: tuple[Command, ...] = (
    Command(
        "run",
        # the order `split_run_argv` (D23) honours: after the script only --vram/--hours/-p
        # (and --json/-w/-d/--dry-run) stay gpu's; --smoke or --data there go to the script
        "[--smoke] [--data [NAME=]PATH] <script> [--vram GB] [--hours H] [-p provider]",
        "submit a job and stream its logs",
        "script",
        short="<script> [flags]",
    ),
    Command(
        "route",
        "<script> [--vram GB] [--hours H] [-p provider] [--smoke]",
        "dry run: where it would go and why",
        "script",
        short="<script> [flags]",
    ),
    Command("jobs", "[--all] [-n N] [--before T]", "running and queued jobs", short="[--all]"),
    Command("status", "[id]", "one job in detail, or the overview", "job"),
    Command("logs", "<id>", "live log stream", "job"),
    Command("watch", "<id> [metric]", "live loss / metric chart", "job"),
    Command("cancel", "<id>", "stop a job (outputs so far are kept)", "job"),
    Command("fetch", "<id> [--dest dir]", "download a job's outputs again", "job"),
    Command("approve", "<id> [reason]", "let a waiting job run", "approval"),
    Command("deny", "<id> [reason]", "refuse a waiting job; nothing runs", "approval"),
    Command("quota", "[--refresh]", "free quota per provider, with reset times"),
    Command(
        "history",
        "[--failed] [-n N] [--before T]",
        "past jobs, failures and handoffs",
        short="[--failed]",
    ),
    Command("providers", "", "connected providers and their health"),
    Command(
        "infer",
        "-m MODEL [-p provider] <prompt> | --file evals.jsonl [-o out] | --list",
        "LLM call or JSONL eval on free inference APIs (not a GPU job)",
        short="-m MODEL <prompt>",
    ),
    Command("login", "<provider>", "check a provider's login and how to fix it", "provider"),
    Command(
        "doctor",
        "[--update-catalog [--yes]] [-v]",
        "CLIs, logins, daemon, data dir, plugin; limits vs providers.yaml",
        short="[--update-catalog]",
    ),
    Command("policy", "[set KEY VALUE | reset]", "approval rules; change one", short="[set|reset]"),
    Command("config", "[edit|path]", "settings (edit opens $EDITOR)", needs_daemon=False),
    Command("help", "[command]", "commands and keys", "command", needs_daemon=False),
    Command("clear", "", "clear the transcript", needs_daemon=False),
    Command("exit", "", "leave the shell (jobs keep running)", needs_daemon=False),
)
BY_NAME = {c.name: c for c in COMMANDS}
ALIASES = {"quit": "exit", "q": "exit", "ls": "jobs", "ps": "jobs", "?": "help"}


class UsageError(Exception):
    """Bad arguments to a slash command: message + the usage line."""


def lookup(word: str) -> Command | None:
    name = word.lstrip("/").lower()
    return BY_NAME.get(ALIASES.get(name, name))


def suggest(word: str) -> str | None:
    names = [c.name for c in COMMANDS]
    close = difflib.get_close_matches(word.lstrip("/").lower(), names, n=1, cutoff=0.5)
    return close[0] if close else None


def split_line(line: str) -> tuple[Command | None, str, list[str]]:
    """(command or None, the word typed, its args). A leading `gpu ` is ignored so CLI
    lines paste in. Raises UsageError on unbalanced quotes."""
    try:
        words = shlex.split(line)
    except ValueError as exc:
        raise UsageError(f"cannot parse that line: {exc}") from None
    if words[:1] == ["gpu"]:
        words = words[1:]
    if not words:
        return None, "", []
    return lookup(words[0]), words[0], words[1:]


# --------------------------------------------------------------------------- output


_CLI_HINT = re.compile(
    r"\bgpu (" + "|".join(sorted((c.name for c in COMMANDS), key=len, reverse=True)) + r")\b"
)


def shellify_text(text: str) -> str:
    """`gpu approve a7f2` -> `/approve a7f2` in a plain string (daemon messages, reasons)."""
    return _CLI_HINT.sub(r"/\1", text)


def shellify(obj: Any, width: int | None = None) -> Any:
    """`gpu approve a7f2` -> `/approve a7f2` in text from the CLI's renderers; tables get
    the same treatment cell by cell and, given the transcript `width`, list tables are
    fitted to it (`fit_table`)."""
    if isinstance(obj, Table):
        return fit_table(obj, width)
    if isinstance(obj, str):
        return Text(_CLI_HINT.sub(r"/\1", obj))
    if isinstance(obj, Text):
        matches = list(_CLI_HINT.finditer(obj.plain))
        if not matches:
            return obj
        out = obj.copy()
        for m in reversed(matches):
            replacement = out[m.start() : m.end()]
            replacement = Text("/" + m.group(1), style=replacement.style)
            out = out[: m.start()] + replacement + out[m.end() :]
        return out
    return obj


NOTE_MIN = 16  # cells a list table's last (free-text) column keeps before names shrink


def _cell_width(cell: Any) -> int:
    if isinstance(cell, Text):
        return cell.cell_len
    return cell_len(str(cell)) if isinstance(cell, str) else 0


def fit_table(table: Table, width: int | None) -> Table:
    """Shellify every cell; a list table (4+ columns, e.g. `render.jobs_table`) keeps one
    row per record at `width`: its last column (notes) is cut with an ellipsis to the room
    left, and columns the CLI already caps (names, `max_width`) give up cells first when
    that room is under NOTE_MIN. Rich would otherwise fold the notes into a word-per-line stack."""
    cols = table.columns
    for col in cols:
        cells = getattr(col, "_cells", None)  # rich keeps a column's cells here
        if isinstance(cells, list):
            cells[:] = [shellify(c) if isinstance(c, str | Text) else c for c in cells]
    if width is None or len(cols) < 4:
        return table
    natural: list[int] = []
    for col in cols:
        cells = getattr(col, "_cells", None) or []
        w = max([_cell_width(col.header), *(_cell_width(c) for c in cells)])
        natural.append(min(w, col.max_width) if col.max_width else w)
    gap = 2 * (len(cols) - 1)  # box=None tables: one cell of padding on each side
    room = width - sum(natural[:-1]) - gap
    if room < NOTE_MIN:
        for i, col in enumerate(cols[:-1]):
            if col.max_width and natural[i] > 8:  # capped columns are made to be cut
                give = min(natural[i] - 8, NOTE_MIN - room)
                col.max_width = natural[i] - give
                room += give
    last = cols[-1]
    header = str(last.header)
    if _cell_width(header) > room and " / " in header:
        last.header = header.split(" / ")[0]  # "progress / note" -> "progress"
    last.no_wrap = True
    last.overflow = "ellipsis"
    last.max_width = max(8, room)
    return table


class Collector(Console):
    """A rich Console whose print() hands renderables to the shell instead of a terminal,
    so the CLI's `render.print_*` functions and `Out` helpers draw into the transcript."""

    def __init__(self, sink: Callable[[list[RenderableType]], None], width: int = 100) -> None:
        super().__init__(file=io.StringIO(), width=width, highlight=False, color_system=None)
        self._sink = sink

    def print(self, *objects: Any, **kwargs: Any) -> None:
        if not objects:
            self._sink([Text("")])
            return
        if all(isinstance(o, str) for o in objects):
            self._sink([shellify(" ".join(objects))])
            return
        self._sink([shellify(o, self.width) for o in objects])


class Host(Protocol):
    """What the app offers a running command (implemented by GpuShell)."""

    paths: Paths
    cwd: Path

    def connect(self, note: Callable[[str], None]) -> GpuClient: ...
    def open_logs(self, job: JobView, attempt: int | None) -> None: ...
    def open_watch(self, job: JobView, metric: str | None) -> None: ...
    def run_suspended(self, argv: list[str]) -> int | None: ...
    def clear_transcript(self) -> None: ...
    def quit_shell(self) -> None: ...
    def poke(self) -> None: ...
    def snapshot(self) -> Snapshot | None: ...
    def output_width(self) -> int: ...


@dataclass
class Ctx:
    host: Host
    sink: Callable[[list[RenderableType]], None]
    _client: GpuClient | None = None

    def emit(self, *items: RenderableType) -> None:
        width = self.host.output_width()
        self.sink([shellify(i, width) for i in items])

    def note(self, message: str) -> None:
        self.emit(Text(message, style="dim"))

    @property
    def console(self) -> Collector:
        return Collector(self.sink, width=self.host.output_width())

    def out(self) -> Any:
        """The CLI's `Out` with both consoles pointed at this block."""
        from gpu_router.cli.app import Out

        out = Out(False)
        out.console = self.console
        out.err = self.console
        out.client = self._client
        return out

    def client(self) -> GpuClient:
        if self._client is None:
            self._client = self.host.connect(self.note)
        return self._client

    def close(self) -> None:
        if self._client is not None:
            self._client.close()


class _Parser(argparse.ArgumentParser):
    """argparse that raises UsageError instead of printing and exiting."""

    def error(self, message: str) -> Any:
        raise UsageError(message)

    def exit(self, status: int = 0, message: str | None = None) -> Any:
        raise UsageError(message or "bad arguments")


def parser(name: str) -> _Parser:
    return _Parser(prog=f"/{name}", add_help=False, allow_abbrev=False)


def _ref(args: Sequence[str], cmd: str) -> str:
    if not args:
        raise UsageError(f"/{cmd} needs a job id, e.g. /{cmd} a7f2 (tab lists them)")
    return args[0]


# --------------------------------------------------------------------------- run / route


def _spec_parser(name: str, *, with_run_flags: bool) -> _Parser:
    p = parser(name)
    p.add_argument("script", nargs="?")
    p.add_argument("--vram", type=float)
    p.add_argument("--hours", type=float)
    p.add_argument("--provider", "-p")
    p.add_argument("--gpu")
    p.add_argument("--project", "-C")
    p.add_argument("--smoke", action="store_true")  # quick smoke test: prefer this Mac
    p.add_argument("--json", action="store_true")  # accepted and ignored (pasted CLI lines)
    if with_run_flags:
        p.add_argument("--name")
        p.add_argument("--env", "-e", action="append")
        p.add_argument("--data", action="append")  # [NAME=]PATH | hf://datasets/...
        p.add_argument("--include", action="append")  # ship even if git ignores it (D60)
        p.add_argument("--wait", "-w", action="store_true")
        p.add_argument("--detach", "-d", action="store_true")
        p.add_argument("--dry-run", action="store_true")
    return p


def _spec(ns: argparse.Namespace, script_args: list[str]) -> Any:
    from gpu_router.cli.app import _build

    spec = _build(
        ns.script,
        script_args,
        vram=ns.vram,
        hours=ns.hours,
        provider=ns.provider,
        gpu=ns.gpu,
        name=getattr(ns, "name", None),
        env=getattr(ns, "env", None),
        project=Path(ns.project).expanduser() if ns.project else None,
        smoke=bool(ns.smoke),
        data=getattr(ns, "data", None),
        include=getattr(ns, "include", None),
    )
    return spec.model_copy(update={"source": Source.SHELL})


def cmd_run(ctx: Ctx, args: list[str]) -> None:
    from gpu_router.cli.app import _check_provider, _print_submitted, split_run_argv

    click_args, script_args, clashes = split_run_argv(args)
    ns = _spec_parser("run", with_run_flags=True).parse_args(click_args)
    for flag in dict.fromkeys(clashes):
        ctx.note(f"`{flag}` after the script goes to the script; gpu options go before it")
    spec = _spec(ns, script_args)
    client = ctx.client()
    _check_provider(client, spec)
    if ns.dry_run:
        from gpu_router.cli.bundling import preview

        bundle = preview(spec)
        render.print_route(ctx.console, spec, client.route(spec))
        if bundle is not None:
            render.print_bundle(ctx.console, bundle)
        ctx.note("dry run: nothing submitted")
        return
    ctx.note("packaging the project and submitting…")
    job = client.submit(spec)
    out = ctx.out()
    _print_submitted(out, job)
    ctx.host.poke()
    sid = job.short_id
    if ns.detach:
        ctx.note(f"  /logs {sid} · /watch {sid} · /cancel {sid}")
        return
    ctx.host.open_logs(job, None)


def cmd_route(ctx: Ctx, args: list[str]) -> None:
    from gpu_router.cli.app import _check_provider

    ns = _spec_parser("route", with_run_flags=False).parse_args(args)
    spec = _spec(ns, [])
    client = ctx.client()
    _check_provider(client, spec)
    render.print_route(ctx.console, spec, client.route(spec))


# --------------------------------------------------------------------------- jobs / status


def _list(ctx: Ctx, args: list[str], *, finished: bool) -> None:
    """/jobs and /history: the CLI's own listing (`cli.app.list_jobs`: the page, its
    empty state and the "… more" hint), so `/jobs --all --before <ts>` pages the same way
    `gpu jobs` does (D44)."""
    from gpu_router.cli.app import list_jobs

    name = "history" if finished else "jobs"
    p = parser(name)
    if finished:
        p.add_argument("--failed", action="store_true", dest="wide")
    else:
        p.add_argument("--all", "-a", action="store_true", dest="wide")
    p.add_argument("--here", action="store_true")
    p.add_argument("--limit", "-n", type=int, default=20 if finished else 50)
    p.add_argument("--before")
    p.add_argument("--json", action="store_true")  # accepted and ignored (pasted CLI lines)
    ns = p.parse_args(args)
    client = ctx.client()
    list_jobs(
        ctx.out(),
        client,
        finished=finished,
        wide=bool(ns.wide),
        here=bool(ns.here),
        limit=max(1, min(500, ns.limit)),
        before=ns.before,
    )


def cmd_jobs(ctx: Ctx, args: list[str]) -> None:
    _list(ctx, args, finished=False)


def cmd_history(ctx: Ctx, args: list[str]) -> None:
    _list(ctx, args, finished=True)


def cmd_status(ctx: Ctx, args: list[str]) -> None:
    from gpu_router.cli.app import _with_quota

    client = ctx.client()
    now = _clock.now()
    if not args:
        view = client.status()
        provs = list(view.providers)
        if not view.active and not view.recent:
            provs = _with_quota(client, provs)
        render.print_status(ctx.console, view.active, view.recent, provs, now)
        return
    render.print_detail(ctx.console, client.job(args[0]), now)


# --------------------------------------------------------------------------- live views


def cmd_logs(ctx: Ctx, args: list[str]) -> None:
    p = parser("logs")
    p.add_argument("ref", nargs="?")
    p.add_argument("--attempt", type=int)
    p.add_argument("--follow", "-f", action="store_true")  # always follows; accepted
    ns = p.parse_args(args)
    job = ctx.client().job(_ref([ns.ref] if ns.ref else [], "logs")).job
    ctx.host.open_logs(job, ns.attempt)


def cmd_watch(ctx: Ctx, args: list[str]) -> None:
    job = ctx.client().job(_ref(args, "watch")).job
    ctx.host.open_watch(job, args[1] if len(args) > 1 else None)


# --------------------------------------------------------------------------- actions


def cmd_cancel(ctx: Ctx, args: list[str]) -> None:
    from gpu_router.cli.app import _action_result

    client = ctx.client()
    before = client.job(_ref(args, "cancel")).job
    job = client.cancel(before.id)
    ctx.host.poke()
    if is_terminal(before.state):
        ctx.note(f"job {job.short_id} already finished ({job.state}); nothing to cancel")
        return
    verb = "cancel requested" if job.state is JobState.CANCELLING else "cancelled"
    _action_result(ctx.out(), job, verb)


def _decision(ctx: Ctx, args: list[str], verb: Literal["approve", "deny"]) -> None:
    from gpu_router.cli.app import _action_result

    p = parser(verb)
    p.add_argument("ref", nargs="?")
    p.add_argument("--reason")
    p.add_argument("words", nargs="*")
    ns = p.parse_args(args)
    ref = _ref([ns.ref] if ns.ref else [], verb)
    reason = ns.reason or (" ".join(ns.words) if ns.words else None)
    client = ctx.client()
    job = (client.approve if verb == "approve" else client.deny)(ref, reason=reason)
    ctx.host.poke()
    _action_result(ctx.out(), job, "approved" if verb == "approve" else "denied")


def cmd_approve(ctx: Ctx, args: list[str]) -> None:
    _decision(ctx, args, "approve")


def cmd_deny(ctx: Ctx, args: list[str]) -> None:
    _decision(ctx, args, "deny")


def cmd_fetch(ctx: Ctx, args: list[str]) -> None:
    from gpu_router.cli.app import _copy_outputs, _wait_fetch

    p = parser("fetch")
    p.add_argument("ref", nargs="?")
    p.add_argument("--dest")
    p.add_argument("--timeout", type=float, default=600.0)
    ns = p.parse_args(args)
    client = ctx.client()
    detail = client.job(_ref([ns.ref] if ns.ref else [], "fetch"))
    job = detail.job
    after = max((e.seq for e in detail.events), default=0)
    client.fetch(job.id)
    ctx.note(f"fetching outputs of job {job.short_id}…")
    result = _wait_fetch(client, job.id, after, ns.timeout)
    job = client.job(job.id).job
    if result is None or not result["ok"]:
        msg = result["message"] if result else f"no answer within {ns.timeout:g}s"
        ctx.emit(_failed(msg))
        return
    where = job.outputs_dir
    if ns.dest and job.outputs_dir:
        copied, err = _copy_outputs(Path(job.outputs_dir), Path(ns.dest))
        if err:
            ctx.emit(_failed(err))
        where = copied or where
    line = seg(f"{render.ICON_DONE} ", "green")
    line.append(f"{int(result.get('files', 0))} files → {render.rel_path(where)}")
    ctx.emit(line)


# --------------------------------------------------------------------------- providers


def cmd_quota(ctx: Ctx, args: list[str]) -> None:
    """`/quota [--refresh]`: the ledger (live readings where a provider has them, else
    estimated from job history; each row says which). --refresh re-reads live ones now."""
    p = parser("quota")
    p.add_argument("--refresh", "-r", action="store_true")
    p.add_argument("--json", action="store_true")  # accepted and ignored (pasted CLI lines)
    ns = p.parse_args(args)
    client = ctx.client()
    if ns.refresh:
        ctx.note("asking every live provider for its quota…")
    quotas = client.quota(refresh=ns.refresh)
    providers = client.providers()
    from gpu_router.cli.lanes import quota_extras  # phase 7b: the inference lane

    _, extra = quota_extras(client)
    if not providers and not quotas:
        ctx.note("no providers connected yet.")
    else:
        ctx.emit(render.quota_table(quotas, providers, _clock.now()))
    if extra:
        ctx.emit(*extra)


def cmd_providers(ctx: Ctx, args: list[str]) -> None:
    from gpu_router.cli.lanes import provider_extras  # phase 7b: not routed + inference

    client = ctx.client()
    found = client.providers()
    _, extra = provider_extras(client, ctx.host.paths)
    if not found:
        ctx.note("no providers connected yet.")
    else:
        ctx.emit(render.providers_table(found, _clock.now()))
    if extra:
        ctx.emit(*extra)
    if any(p.health is ProviderHealth.AUTH_REQUIRED for p in found if p.enabled):
        ctx.note("/login <provider> shows how to log in")


LOGIN_STEPS: dict[str, tuple[str, ...]] = {
    "kaggle": (
        "create an API token: kaggle.com → Settings → API → Create New Token",
        "then, in a terminal (checks the token and keeps it in the Keychain):",
        "  gpu login kaggle",
    ),
    "colab": (
        "log in once with Application Default Credentials (opens a browser):",
        "  gcloud auth application-default login --scopes=openid,"
        "https://www.googleapis.com/auth/cloud-platform,"
        "https://www.googleapis.com/auth/userinfo.email,"
        "https://www.googleapis.com/auth/colaboratory",
        "the colaboratory scope is required; without it every session gets a 403",
    ),
    "lightning": (
        "lightning.ai → Settings → Keys: copy the user id and API key, then run",
        "  gpu login lightning        (prompts; nothing is shown or put on the command line)",
        "or, after `lightning login`:  gpu login lightning --import",
    ),
    "local": ("nothing to log into: jobs run on this Mac",),
}


def _notes_path(kind: str) -> str:
    return f"src/gpu_router/providers/{kind}/NOTES.md"


def cmd_login(ctx: Ctx, args: list[str]) -> None:
    client = ctx.client()
    providers = {p.name: p for p in client.providers()}
    if not args:
        table = Table(box=None, show_header=False, pad_edge=False)
        for _ in range(3):
            table.add_column(no_wrap=True)
        for p in providers.values():
            if p.enabled:
                table.add_row(
                    p.name, render.health_text(p.health), Text(p.health_reason or "", "dim")
                )
        ctx.emit(table)
        ctx.note("/login <provider> re-checks one and shows how to log in")
        return
    name = args[0].strip().lower()
    from gpu_router.shell.infer import inference_login_steps  # phase 7b

    steps_7b = inference_login_steps(name) if name not in providers else None
    if steps_7b is not None:
        # raw sink: `gpu login <p>` must stay a terminal command, not become /login
        ctx.sink([Text(f"  {s}") for s in steps_7b])
        return
    if name not in providers:
        close = difflib.get_close_matches(name, list(providers), n=1, cutoff=0.5)
        raise InvalidRequest(
            f"there is no provider named {name!r}",
            hint=(f"did you mean {close[0]}? " if close else "")
            + f"known: {', '.join(providers) or 'none'}",
        )
    ctx.note(f"checking {name}…")
    view = client.healthcheck(name)
    line = Text(f"{view.name}  ")
    line.append_text(render.health_text(view.health))
    if view.health_reason:
        line.append(f"  {view.health_reason}", style="dim")
    ctx.emit(line)
    if view.health is ProviderHealth.OK:
        ctx.note(f"{name} is logged in and answering; nothing to do")
        return
    steps = LOGIN_STEPS.get(view.kind) or LOGIN_STEPS.get(name) or ()
    # raw sink: `gpu login kaggle|lightning` prompt in a terminal; shellified they would read
    # `/login ...`, which only re-checks (like the phase-7b inference steps above)
    ctx.sink([Text(f"  {s}") for s in steps])
    ctx.note(f"details: {_notes_path(view.kind)} · then /login {name} again")


def cmd_doctor(ctx: Ctx, args: list[str]) -> None:
    """`/doctor`: the same checks and rendering as `gpu doctor` (phase 8a, doctor/cli.py),
    over the shell's daemon connection."""
    from gpu_router.doctor.cli import shell_doctor

    shell_doctor(ctx, args)


def cmd_policy(ctx: Ctx, args: list[str]) -> None:
    """`/policy`, `/policy set KEY VALUE`, `/policy reset`: the CLI's `gpu policy` (phase-5
    rules over GET/PUT /v1/policy), then the jobs waiting for an answer right now."""
    from gpu_router.cli.app import _print_policy

    client = ctx.client()
    out = ctx.out()
    sub = args[0] if args else "show"
    if sub == "set":
        from gpu_router.policy import apply_policy_setting

        if len(args) != 3:
            raise UsageError(
                "usage: /policy set KEY VALUE, e.g. /policy set agent.auto_max_hours 2"
            )
        view = client.policy()
        if not view.editable or view.policy is None:
            raise InvalidRequest(
                f"the daemon's approval policy ({view.name}) has no editable rules",
                hint="restart the daemon (gpu daemon stop; gpu daemon start)",
            )
        view = client.set_policy(apply_policy_setting(view.policy, args[1], args[2]))
        ctx.emit(_ok(f"{args[1]} = {args[2]}"))
    elif sub == "reset":
        from gpu_router.policy import PolicyConfig

        view = client.set_policy(PolicyConfig())
        ctx.emit(_ok("approval rules reset"))
    elif sub == "show":
        view = client.policy()
    else:
        raise UsageError("usage: /policy [show | set KEY VALUE | reset]")
    _print_policy(out, view)
    snap = ctx.host.snapshot()
    waiting = [j for j in (snap.active if snap else ()) if j.state is JobState.AWAITING_APPROVAL]
    ctx.emit(Text(""))
    if not waiting:
        ctx.note("no jobs waiting for approval")
    for j in waiting:
        line = seg(f"{render.ICON_WAITING} ", "yellow")
        line.append(f"job {j.short_id} {j.name}  ")
        line.append(f"/approve {j.short_id} · /deny {j.short_id}", style="dim")
        ctx.emit(line)


def policy_keys() -> list[str]:
    """`/policy set` keys for tab completion: bare rule names (both audiences), then the
    same per audience (`agent.auto_max_hours`, ...)."""
    from gpu_router.policy import PolicyRules

    rules = list(PolicyRules.model_fields)
    return [*rules, *(f"{who}.{rule}" for who in ("agent", "user") for rule in rules)]


def cmd_config(ctx: Ctx, args: list[str]) -> None:
    import yaml

    from gpu_router.config import load_config

    paths = ctx.host.paths
    target = paths.config
    if args[:1] == ["path"]:
        ctx.emit(Text(str(target)))
        return
    if args[:1] == ["edit"]:
        editor = os.environ.get("VISUAL") or os.environ.get("EDITOR") or "vi"
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("version: 1\n")
        code = ctx.host.run_suspended([*shlex.split(editor), str(target)])
        if code is None:
            ctx.emit(_failed(f"could not open {editor!r} here"))
            ctx.note(f"edit {target} in another terminal; /config shows the result")
            return
        try:
            load_config(paths)
        except Exception as exc:
            ctx.emit(_failed(f"config.yaml is invalid now: {exc}"))
            ctx.note("the daemon keeps its current settings; /config edit to fix it")
            return
        ctx.emit(_ok("config.yaml is valid"))
        ctx.note(
            "the daemon reads it at start: gpu daemon stop, then any command restarts it"
            + (f" (editor exited {code})" if code else "")
        )
        return
    if args:
        raise UsageError("usage: /config [edit|path]")
    try:
        cfg = load_config(paths)
    except Exception as exc:
        ctx.emit(_failed(str(exc)))
        return
    data = cfg.model_dump(mode="json", exclude={"test_mode"})
    text = yaml.safe_dump(data, sort_keys=False, default_flow_style=False).rstrip()
    ctx.emit(Text.assemble(("effective settings  ", "dim"), (str(target), "dim")))
    ctx.emit(Text(text))
    if cfg.test_mode:
        ctx.note("test mode is on (GPU_ROUTER_TEST_MODE): only the fake providers are real")
    ctx.note(
        "/config edit opens it in $EDITOR · env overrides: GPU_ROUTER_PORT, GPU_ROUTER_LOG_LEVEL"
    )


def help_renderable(topic: str | None = None) -> RenderableType:
    if topic:
        cmd = lookup(topic)
        if cmd is None:
            return Text(f"no command {topic!r}; /help lists them", style="dim")
        return Group(
            Text.assemble((f"/{cmd.name} {cmd.usage}".rstrip(), "bold")),
            Text(f"  {cmd.summary}", style="dim"),
        )
    keys = Text(
        "/ opens the command list · tab completes commands, job ids and scripts · ↑↓ history · "
        "esc stops a live view · pgup/pgdn scroll (the newest /logs output first) · "
        "ctrl+d quits (jobs keep running)",
        style="dim",
    )
    return Group(HelpList(), Text(""), keys)


HELP_TWO_COLUMNS = 64  # narrower than this, each usage line gets its summary under it
HELP_USAGE_SHARE = 0.45  # of the width, at most, for the usage column (long ones wrap)


class HelpList:
    """/help's command list, laid out for the width it is drawn at, so a resize redraws it
    right: usage | summary when there is room (the usage column capped, long usages wrap
    inside it), else usage lines with the summary indented under each. Cells are Text, not
    markup: `[id]`, `[reason]`, `[edit|path]` would otherwise be eaten as style tags."""

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        width = options.max_width
        rows = [(Text(f"/{c.name} {c.usage}".rstrip()), Text(c.summary)) for c in COMMANDS]
        if width < HELP_TWO_COLUMNS:
            for usage, summary in rows:
                yield usage
                summary.stylize("dim")
                yield Padding(summary, (0, 0, 0, 2))  # wrapped lines keep the indent
            return
        longest = max(usage.cell_len for usage, _ in rows)
        table = Table(box=None, show_header=False, pad_edge=False, padding=(0, 2, 0, 0))
        table.add_column(width=min(longest, int(width * HELP_USAGE_SHARE)), overflow="fold")
        table.add_column(style="dim", overflow="fold")
        for usage, summary in rows:
            table.add_row(usage, summary)
        yield table


def cmd_help(ctx: Ctx, args: list[str]) -> None:
    ctx.emit(help_renderable(args[0] if args else None))


def cmd_clear(ctx: Ctx, args: list[str]) -> None:
    ctx.host.clear_transcript()


def cmd_exit(ctx: Ctx, args: list[str]) -> None:
    ctx.host.quit_shell()


HANDLERS: dict[str, Callable[[Ctx, list[str]], None]] = {
    "run": cmd_run,
    "route": cmd_route,
    "jobs": cmd_jobs,
    "status": cmd_status,
    "logs": cmd_logs,
    "watch": cmd_watch,
    "cancel": cmd_cancel,
    "fetch": cmd_fetch,
    "approve": cmd_approve,
    "deny": cmd_deny,
    "quota": cmd_quota,
    "history": cmd_history,
    "providers": cmd_providers,
    "infer": cmd_infer,
    "login": cmd_login,
    "doctor": cmd_doctor,
    "policy": cmd_policy,
    "config": cmd_config,
    "help": cmd_help,
    "clear": cmd_clear,
    "exit": cmd_exit,
}
assert set(HANDLERS) == set(BY_NAME)


def _ok(message: str) -> Text:
    """`✓ message`: green on the icon only."""
    line = seg(f"{render.ICON_DONE} ", "green")
    line.append(message)
    return line


def _failed(message: str) -> Text:
    """`✗ message`: red on the icon only, like every state icon (spec UX 6)."""
    line = seg(f"{render.ICON_FAILED} ", "red")
    line.append(message)
    return line


def error_renderables(exc: BaseException) -> list[RenderableType]:
    """`✗ message` + dim hint, the CLI's error shape, for the transcript."""
    if isinstance(exc, UsageError):
        return [_failed(str(exc))]
    if isinstance(exc, GpuRouterError):
        out: list[RenderableType] = [_failed(exc.message)]
        if exc.hint:
            out.append(shellify(Text(f"  {exc.hint}", style="dim")))
        matches = exc.detail.get("matches") if isinstance(exc.detail, dict) else None
        if matches:
            out.append(Text("  " + "  ".join(str(m) for m in list(matches)[:5]), style="dim"))
        return out
    return [
        _failed(f"internal error: {type(exc).__name__}: {exc}"),
        Text("  this is a gpu-router bug; the daemon log may say more", style="dim"),
    ]
