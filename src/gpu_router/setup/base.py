"""Shared wizard types (phase 8b): item outcomes, options, and the per-run context whose
`confirm()` is the only place the wizard asks a yes/no question.

`confirm()` applies the rules in one place:

- `--dry-run`: never asks, never changes anything ("dry run: would ask ...");
- a resumed run keeps its earlier "no" (asked again only when `--only` names the item);
- `--yes`: yes, except for things a human must do at the keyboard (a browser sign-in, a
  pasted token) when no terminal is attached;
- run by an agent (`agent.agent_marker()`: CLAUDECODE=1, CODEX_SANDBOX, GPU_ROUTER_AGENT=1):
  `--yes` never answers the items that change the user's own setup (AGENT_GUARDED: the
  Claude Code status line and plugin, the Codex config, the colab skill, the launchd
  agent). They are asked when a terminal is attached, else left `manual` with the command
  (review fix: an agent told "run `gpu setup`" could add --yes and rewrite them unseen);
- no terminal and no `--yes`: not asked; the item ends as `manual` with the command to run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from gpu_router.setup.ui import Mark

if TYPE_CHECKING:
    from gpu_router.setup.context import SetupEnv
    from gpu_router.setup.state import SetupState
    from gpu_router.setup.ui import Ui

__all__ = ["AGENT_GUARDED", "Ctx", "ItemResult", "Options", "Outcome"]

#: item ids (or `prefix.`) an agent's `--yes` never answers
AGENT_GUARDED = ("integration.", "launchd")


def agent_guarded(item: str) -> bool:
    return any(item == g or (g.endswith(".") and item.startswith(g)) for g in AGENT_GUARDED)


class Outcome(StrEnum):
    DONE = "done"  # this run did it
    ALREADY = "already"  # it was already done (idempotent re-runs land here)
    SKIPPED = "skipped"  # does not apply here, or a dry run
    DECLINED = "declined"  # the user said no
    MANUAL = "manual"  # needs the user: `fix` is the command to run
    FAILED = "failed"


MARK = {
    Outcome.DONE: Mark.DONE,
    Outcome.ALREADY: Mark.DONE,
    Outcome.SKIPPED: Mark.SKIP,
    Outcome.DECLINED: Mark.SKIP,
    Outcome.MANUAL: Mark.WAIT,
    Outcome.FAILED: Mark.FAIL,
}


@dataclass
class ItemResult:
    id: str  # "tools.kaggle", "login.colab", "integration.plugin", ...
    outcome: Outcome
    summary: str
    fix: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "id": self.id,
            "outcome": str(self.outcome),
            "summary": self.summary,
        }
        if self.fix:
            body["fix"] = self.fix
        if self.detail:
            body["detail"] = self.detail
        return body


@dataclass(frozen=True)
class Options:
    yes: bool = False
    dry_run: bool = False
    only: frozenset[str] = frozenset()  # expanded item ids; empty = every item
    check: bool = True  # verify credentials with the provider before storing them
    smoke: bool = True  # False = --no-smoke
    again: bool = False  # re-run smoke tests that already passed


@dataclass
class Ctx:
    env: SetupEnv
    ui: Ui
    state: SetupState
    opts: Options
    results: list[ItemResult] = field(default_factory=list)
    unanswered: list[str] = field(default_factory=list)  # needed a terminal
    held_for_human: list[str] = field(default_factory=list)  # an agent's --yes skipped them

    def agent(self) -> str | None:
        """The agent marker when an agent runs this setup (`NAME=value`), else None."""
        from gpu_router.agent import agent_marker

        return agent_marker(self.env.environ)

    # ------------------------------------------------------------------ selection

    def selected(self, item: str) -> bool:
        return not self.opts.only or item in self.opts.only

    def explicit(self, item: str) -> bool:
        """Named by --only (so an earlier "no" does not stand)."""
        return item in self.opts.only

    # ------------------------------------------------------------------ asking

    def confirm(
        self, item: str, question: str, *, default: bool, needs_human: bool = False
    ) -> bool | None:
        """True/False, or None when nobody can answer (no terminal, no --yes)."""
        if self.opts.dry_run:
            self.ui.say(f"dry run: would ask “{question}”", "dim")
            return False
        previous = self.state.answer(item)
        if self.state.resumed and previous == "no" and not self.explicit(item):
            self.ui.say(
                f"you said no earlier in this setup run; `gpu setup --only {item}` asks again",
                "dim",
            )
            return False
        held = self.opts.yes and agent_guarded(item) and self.agent() is not None
        if self.opts.yes and not held and (self.ui.interactive or not needs_human):
            self.state.record_answer(item, True)
            return True
        if held and not self.ui.interactive:
            self.held_for_human.append(item)
            return None
        if not self.ui.interactive:
            if not self.opts.yes:  # with --yes it is a human-only step: `manual`, not a question
                self.unanswered.append(item)
            return None
        answer = self.ui.ask(question, default)
        self.state.record_answer(item, answer)
        return answer

    def can_type(self) -> bool:
        """A human is at the keyboard (secrets are only ever typed, never passed in argv)."""
        return self.ui.interactive and not self.opts.dry_run

    # ------------------------------------------------------------------ results

    def done(
        self,
        item: str,
        outcome: Outcome,
        summary: str,
        fix: str | None = None,
        *,
        quiet: bool = False,
        **detail: Any,
    ) -> ItemResult:
        """Print the item's line (unless `quiet`: the closing summary prints it), remember
        it in setup.json, return it."""
        res = ItemResult(
            item, outcome, summary, fix, {k: v for k, v in detail.items() if v is not None}
        )
        self.results.append(res)
        if not quiet:
            self.ui.item(
                MARK[outcome],
                summary,
                fix if outcome in (Outcome.MANUAL, Outcome.FAILED) else None,
            )
        if not self.opts.dry_run:
            self.state.record_item(item, str(outcome), summary, self.env.clock.now())
        return res

    def not_asked(self, item: str, what: str, fix: str) -> ItemResult:
        """No terminal and no --yes: say what would happen and how to do it later."""
        if item in self.held_for_human:
            why = f"left for you: an agent ran setup ({self.agent() or 'agent'})"
            return self.done(item, Outcome.MANUAL, f"{what} ({why})", fix)
        return self.done(item, Outcome.MANUAL, f"{what} (not asked: no terminal)", fix)

    def needs_keyboard(self, item: str, what: str, fix: str) -> ItemResult:
        """A secret must be typed (or a sign-in chosen) and no terminal is attached. Same
        rule as confirm(needs_human=True): `manual`, and without --yes the question counts
        as unanswered (exit 2) (review fix: these logins exited 0)."""
        if not self.opts.yes:
            self.unanswered.append(item)
        return self.not_asked(item, what, fix)

    def declined(self, item: str, what: str, fix: str | None = None) -> ItemResult:
        later = f"; later: {fix}" if fix else ""
        return self.done(item, Outcome.DECLINED, f"skipped: {what}{later}")

    def dry(self, item: str, what: str) -> ItemResult:
        return self.done(item, Outcome.SKIPPED, f"dry run: would {what}")
