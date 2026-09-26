"""The wizard: step order, `--only`, resume, the closing summary and exit codes (phase 8b).

Exit codes: 0 finished (declined items are fine), 1 an item failed, 2 questions were left
unanswered because there was no terminal and no `--yes` (wins over 1; usage errors are 2
too), 130 Ctrl-C (the run stays resumable: `gpu setup` continues it).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from gpu_router.errors import InvalidRequest
from gpu_router.setup import check, integrations, logins, service, tools
from gpu_router.setup.base import Ctx, ItemResult, Options, Outcome
from gpu_router.setup.state import SetupState
from gpu_router.setup.ui import Mark

if TYPE_CHECKING:
    from gpu_router.setup.context import SetupEnv
    from gpu_router.setup.ui import Ui

__all__ = ["ALL_ITEMS", "STEPS", "Step", "WizardResult", "expand_only", "first_run", "run_wizard"]

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_UNANSWERED = 2
EXIT_INTERRUPTED = 130  # returned by the CLI front end


@dataclass(frozen=True)
class Step:
    name: str
    title: str
    sub: str
    fn: Callable[[Ctx], None]
    items: tuple[str, ...]


STEPS: tuple[Step, ...] = (
    Step("tools", "tools", "the CLIs gpu-router drives, installed with uv", tools.run, tools.ITEMS),
    Step(
        "logins",
        "logins",
        "one account per provider; tokens go to the Keychain",
        logins.run,
        logins.ITEMS,
    ),
    Step("launchd", "launchd", "start the daemon at login", service.run, (service.ITEM,)),
    Step(
        "integration",
        "Claude Code and Codex",
        "each shows exactly what changes and asks first",
        integrations.run,
        integrations.ITEMS,
    ),
    Step(
        "check",
        "check",
        "gpu doctor, a GPU smoke test per provider, free hours",
        check.run,
        check.ITEMS,
    ),
)
ALL_ITEMS: tuple[str, ...] = tuple(i for s in STEPS for i in s.items)
_ALIASES = {"login": "logins", "tool": "tools", "integrations": "integration", "service": "launchd"}


def expand_only(values: Iterable[str]) -> frozenset[str]:
    """--only values (steps, item ids, or an item's last part when unique; comma lists ok)
    -> item ids. Raises InvalidRequest naming what is valid."""
    out: set[str] = set()
    for raw in values:
        for part in raw.split(","):
            name = part.strip().lower().replace("-", "_")
            if not name:
                continue
            name = _ALIASES.get(name, name)
            step = next((s for s in STEPS if s.name == name), None)
            if step is not None:
                out.update(step.items)
                continue
            if name in ALL_ITEMS:
                out.add(name)
                continue
            matches = [i for i in ALL_ITEMS if i.split(".", 1)[-1] == name]
            if len(matches) == 1:
                out.add(matches[0])
                continue
            if len(matches) > 1:
                raise InvalidRequest(
                    f"--only {part.strip()} is ambiguous: {', '.join(matches)}",
                    hint="name the full item, e.g. --only " + matches[0],
                )
            raise InvalidRequest(
                f"--only {part.strip()}: no such setup step or item",
                hint="steps: "
                + ", ".join(s.name for s in STEPS)
                + "; items: "
                + ", ".join(ALL_ITEMS),
            )
    return frozenset(out)


@dataclass
class WizardResult:
    code: int
    results: list[ItemResult]
    unanswered: list[str]
    dismissed: bool = False

    def to_json(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for r in self.results:
            counts[str(r.outcome)] = counts.get(str(r.outcome), 0) + 1
        summary = next((r.summary for r in self.results if r.id == "check.summary"), None)
        return {
            "ok": self.code == EXIT_OK,
            "exit_code": self.code,
            "counts": counts,
            "summary": summary,
            "items": [r.to_json() for r in self.results],
            "unanswered": self.unanswered,
            "dismissed": self.dismissed,
        }


def _when(ts: Any) -> str:
    from datetime import datetime

    try:
        return datetime.fromtimestamp(float(ts)).strftime("%a %H:%M")  # noqa: DTZ006 - local time for display
    except (TypeError, ValueError, OverflowError, OSError):
        return "earlier"


def run_wizard(
    env: SetupEnv,
    ui: Ui,
    opts: Options,
    *,
    first_run: str | None = None,
) -> WizardResult:
    from gpu_router.doctor.probe import home_label

    state = SetupState.load(env.paths.home)
    now = env.clock.now()
    if first_run is not None:
        question = (
            "the last setup run stopped part way; continue where it stopped?"
            if first_run == "resume"
            else "gpu-router is not set up yet; set it up now? (a few minutes)"
        )
        if not ui.ask(question, True):
            env.paths.ensure()
            state.dismiss(now)
            ui.say("ok: run `gpu setup` any time", "dim")
            return WizardResult(EXIT_OK, [], [], dismissed=True)
    ctx = Ctx(env=env, ui=ui, state=state, opts=opts)
    steps = [s for s in STEPS if any(ctx.selected(i) for i in s.items)]
    n_steps = f"{len(steps)} step{'s' if len(steps) != 1 else ''}"
    sub = f"data dir {home_label(env.paths.home, env.user_home)} · {n_steps}"
    if opts.dry_run:
        sub += " · dry run: nothing changes"
    else:
        sub += " · ctrl+c stops; `gpu setup` picks up where it stopped"
    ui.title("gpu-router setup", sub)
    if not opts.dry_run:
        env.paths.ensure()  # the data dir, 0700, before anything lands in it
        if opts.only:
            state.begin_subset()
        else:
            state.begin(now)
            if state.resumed:
                ui.say(
                    f"continuing the setup run from {_when(state.run.get('started_at'))}; "
                    "your earlier answers stand",
                    "dim",
                )
    for n, step in enumerate(steps, 1):
        ui.step(n, len(steps), step.title, step.sub)
        step.fn(ctx)
    if not opts.dry_run:
        state.finish(env.clock.now(), complete=not opts.only)
    return _close(ctx)


def _close(ctx: Ctx) -> WizardResult:
    ui = ctx.ui
    failed = [r for r in ctx.results if r.outcome is Outcome.FAILED]
    manual = [r for r in ctx.results if r.outcome is Outcome.MANUAL]
    declined = [r for r in ctx.results if r.outcome is Outcome.DECLINED]
    ui.say("")
    summary = next((r for r in ctx.results if r.id == "check.summary"), None)
    if summary is not None:
        mark = Mark.DONE if summary.outcome is Outcome.DONE else Mark.FAIL
        ui.item(mark, summary.summary)
    todo = failed + manual
    if todo:
        ui.say("still to do (each line is the command):", "dim")
        for r in todo:
            ui.item(Mark.FAIL if r.outcome is Outcome.FAILED else Mark.WAIT, r.summary, r.fix)
    if declined:
        ui.say(
            f"{len(declined)} skipped by choice; `gpu setup` offers them again, "
            "`gpu setup --only <item>` offers one",
            "dim",
        )
    if ctx.unanswered:
        if ctx.agent() is not None:
            # no "--yes" advice for an agent: those answers are the user's to give
            ui.say(
                f"{len(ctx.unanswered)} question(s) need a terminal: ask the user to run "
                "`gpu setup` in one",
                "yellow",
            )
        else:
            ui.say(
                f"{len(ctx.unanswered)} question(s) need a terminal: run `gpu setup` in one, "
                "or add --yes to answer yes to all",
                "yellow",
            )
    if ctx.held_for_human:
        ui.say(
            f"{len(ctx.held_for_human)} change(s) to your own setup were left for you (an agent "
            "ran `gpu setup`); run `gpu setup` yourself to review them",
            "yellow",
        )
    if not todo and not ctx.unanswered and not ctx.opts.dry_run:
        if ctx.opts.only:
            ui.say("done; `gpu setup` goes through every step", "dim")
        else:
            ui.say("setup is complete; `gpu doctor` re-checks everything any time", "dim")
    code = EXIT_OK
    if ctx.unanswered:  # first thing to do: answer them (later failures often follow)
        code = EXIT_UNANSWERED
    elif failed:
        code = EXIT_FAILED
    return WizardResult(code, ctx.results, ctx.unanswered)


def first_run(mode: str) -> bool:
    """Bare `gpu` on a terminal with no finished setup (entry.py): offer the wizard.
    Returns True when the caller should open the shell afterwards."""
    from gpu_router.clock import SystemClock
    from gpu_router.paths import Paths
    from gpu_router.setup.context import SetupEnv
    from gpu_router.setup.ui import TerminalUi

    env = SetupEnv(paths=Paths.from_env(), clock=SystemClock())
    ui = TerminalUi()
    try:
        result = run_wizard(env, ui, Options(), first_run=mode)
    except KeyboardInterrupt:
        ui.say("")
        ui.say("stopped; `gpu setup` continues where it stopped", "dim")
        return False
    if result.dismissed:
        return True
    try:
        return ui.ask("open the gpu shell now?", True)
    except KeyboardInterrupt:
        return False
