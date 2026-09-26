"""Completion for the prompt (phase 4): the `/` popup and tab.

`complete(text, known)` looks at the word under the cursor (always the end of the input)
and returns the candidates that fit it: command names after `/`, job ids after commands
that take one (only jobs waiting for approval after /approve and /deny, when there are
any), scripts and directories relative to the shell's cwd after /run and /route, provider
names after /login and `-p`. Pure: the caller supplies what the shell knows.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from rich.cells import cell_len
from rich.text import Text

from gpu_router.api import JobView, ProviderView
from gpu_router.cli import render
from gpu_router.models import ProviderHealth
from gpu_router.shell.commands import COMMANDS, Command, lookup
from gpu_router.shell.panel import pad
from gpu_router.statemachine import JobState

MAX_CANDIDATES = 50
MATCH_LIMIT = 1000  # matching entries collected before sorting (huge dirs)
SCRIPT_SUFFIXES = (".py", ".sh", ".ipynb")
SKIP_DIRS = {"__pycache__", "node_modules", ".git", ".venv", "venv", "runs", ".mypy_cache"}
_VALUE_FLAGS = {
    "--vram",
    "--hours",
    "--gpu",
    "--name",
    "--env",
    "-e",
    "--project",
    "-C",
    "--dest",
    "--reason",
    "--attempt",
    "--limit",
    "-n",
    "--timeout",
}
_PROVIDER_FLAGS = {"--provider", "-p"}
_PATH_FLAGS = {"--data"}  # [NAME=]PATH: any file or directory
POLICY_SUBCOMMANDS = (
    ("show", "the rules in force"),
    ("set", "KEY VALUE  change one rule"),
    ("reset", "back to the defaults"),
)


@dataclass(frozen=True)
class Candidate:
    value: str  # replaces the word being completed
    label: Text  # what the popup shows
    final: bool = True  # False for a directory: keep completing, no trailing space
    more: bool = False  # the line still needs an argument after it: Enter keeps editing


@dataclass(frozen=True)
class Completion:
    start: int  # index in the input where the replaced word starts
    word: str
    candidates: tuple[Candidate, ...]

    def common_prefix(self) -> str:
        values = [c.value for c in self.candidates]
        return os.path.commonprefix(values) if values else ""


@dataclass(frozen=True)
class Known:
    """What the shell knows right now (from the feed's snapshot)."""

    jobs: Sequence[JobView] = ()
    providers: Sequence[ProviderView] = ()
    cwd: Path = field(default_factory=Path.cwd)


def command_label(name: str, usage: str, summary: str, width: int = 26) -> Text:
    head = f"/{name} {usage}".rstrip()
    label = Text(pad(head, width - 2) + "  ")
    label.append(summary, style="dim")
    return label


def _command_candidates(cmds: Sequence[Command], *, slash: bool = True) -> list[Candidate]:
    width = min(34, max((cell_len(f"/{c.name} {c.hint}".rstrip()) for c in cmds), default=10) + 2)
    return [
        Candidate(
            f"/{c.name}" if slash else c.name, command_label(c.name, c.hint, c.summary, width)
        )
        for c in cmds
    ]


def _commands(word: str) -> list[Candidate]:
    typed = word.lstrip("/").lower()
    first = [c for c in COMMANDS if c.name.startswith(typed)]
    rest = [c for c in COMMANDS if typed and typed in c.name and c not in first]
    return _command_candidates([*first, *rest])


def job_label(job: JobView, name_w: int = 24) -> Text:
    icon, style = render.state_style(job.state)
    label = Text(pad(job.short_id, 6) + "  ")
    label.append(pad(job.name, name_w) + "  ")
    label.append(f"{icon} ", style=style)
    word = "needs approval" if job.state is JobState.AWAITING_APPROVAL else str(job.state)
    label.append(word.replace("_", " "), style=style)
    if job.provider:
        label.append(f" · {job.provider}", style="dim")
    return label


def _jobs(word: str, jobs: Sequence[JobView], *, approvals: bool) -> list[Candidate]:
    typed = word.lower()
    pool = list(jobs)
    if approvals:
        waiting = [j for j in pool if j.state is JobState.AWAITING_APPROVAL]
        pool = waiting or pool
    seen: set[str] = set()
    picked: list[JobView] = []
    for j in pool:
        if j.id in seen or not (j.short_id.startswith(typed) or j.id.startswith(typed)):
            continue
        seen.add(j.id)
        picked.append(j)
    name_w = min(24, max((cell_len(j.name) for j in picked), default=8))
    return [Candidate(j.short_id, job_label(j, name_w)) for j in picked]


def _providers(word: str, providers: Sequence[ProviderView]) -> list[Candidate]:
    out = []
    for p in providers:
        if p.enabled and p.name.startswith(word.lower()):
            label = Text(f"{p.name:<12}")
            label.append_text(
                render.health_text(p.health if p.enabled else ProviderHealth.DISABLED)
            )
            out.append(Candidate(p.name, label))
    return out


def _scripts(word: str, cwd: Path, *, any_file: bool = False) -> list[Candidate]:
    """Directories and scripts under the typed path (`any_file`: every file, for --data;
    a `NAME=` prefix is kept)."""
    if any_file and "=" in word and not word.startswith(("/", ".", "~")):
        name, _, rest = word.partition("=")
        return [
            Candidate(f"{name}={c.value}", c.label, c.final)
            for c in _scripts(rest, cwd, any_file=True)
        ]
    expanded = os.path.expanduser(word)
    head, _, prefix = expanded.rpartition("/")
    base = Path(head) if head else Path(".")
    root = base if base.is_absolute() else cwd / base
    shown_head = word[: len(word) - len(prefix)]
    # This runs on the UI thread at every keystroke: names are filtered by the prefix
    # before anything is stat'ed (scandir's d_type), and collecting stops at MATCH_LIMIT,
    # so a data folder with 50k images costs one directory read, not 50k stats (D44).
    found: list[tuple[bool, str]] = []  # (is_dir, name)
    try:
        with os.scandir(root) as it:
            for entry in it:
                name = entry.name
                if not name.startswith(prefix):
                    continue
                if name.startswith(".") and not prefix.startswith("."):
                    continue
                try:
                    is_dir = entry.is_dir()
                except OSError:
                    continue
                if is_dir:
                    if name in SKIP_DIRS or name.endswith(".egg-info"):
                        continue
                elif not (any_file or name.endswith(SCRIPT_SUFFIXES)):
                    continue
                found.append((is_dir, name))
                if len(found) >= MATCH_LIMIT:
                    break
    except OSError:
        return []
    found.sort(key=lambda e: (not e[0], e[1].lower()))
    out: list[Candidate] = []
    for is_dir, name in found[:MAX_CANDIDATES]:
        if is_dir:
            out.append(
                Candidate(f"{shown_head}{name}/", Text(f"{name}/", style="dim"), final=False)
            )
        else:
            out.append(Candidate(f"{shown_head}{name}", Text(name)))
    return out


def _policy(words: Sequence[str], word: str) -> list[Candidate]:
    """`/policy <show|set|reset>`, then `/policy set <key>` (phase-5 rules)."""
    from gpu_router.shell.commands import policy_keys

    if len(words) == 2:
        return [
            Candidate(name, command_label(name, "", summary, 10), more=name == "set")
            for name, summary in POLICY_SUBCOMMANDS
            if name.startswith(word.lower())
        ]
    if len(words) == 3 and words[1] == "set":
        return [
            Candidate(k, Text(k), more=True) for k in policy_keys() if k.startswith(word.lower())
        ]
    return []


def complete(text: str, known: Known) -> Completion | None:
    """Candidates for the last word of `text`, or None when nothing applies."""
    stripped = text.lstrip()
    if stripped.startswith("gpu "):
        stripped = stripped[4:].lstrip()
    lead = len(text) - len(stripped)
    if " " not in stripped:
        if not stripped.startswith("/"):
            return None
        cands = _commands(stripped)
        return Completion(lead, stripped, tuple(cands)) if cands else None
    words = stripped.split(" ")
    cmd = lookup(words[0])
    if cmd is None:
        return None
    word = words[-1]
    start = len(text) - len(word)
    prev = words[-2] if len(words) >= 2 else ""
    if prev in _PROVIDER_FLAGS:
        cands = _providers(word, known.providers)
    elif prev in _PATH_FLAGS:
        cands = _scripts(word, known.cwd, any_file=True)
    elif prev in _VALUE_FLAGS or word.startswith("-"):
        return None
    elif cmd.name == "policy":
        cands = _policy(words, word)
    elif cmd.arg == "script":
        cands = _scripts(word, known.cwd)
    elif cmd.arg in ("job", "approval"):
        positional = [w for w in words[1:-1] if w and not w.startswith("-")]
        if positional:
            return None  # the id is already there
        cands = _jobs(word, known.jobs, approvals=cmd.arg == "approval")
    elif cmd.arg == "provider":
        if len(words) > 2:
            return None
        cands = _providers(word, known.providers)
    elif cmd.arg == "command":
        if len(words) > 2:
            return None
        cands = _command_candidates(
            [c for c in COMMANDS if c.name.startswith(word.lower())], slash=False
        )
    else:
        return None
    return Completion(start, word, tuple(cands)) if cands else None
