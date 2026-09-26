"""How the wizard talks to the user (phase 8b): one small interface, a terminal version and
the answers tests script.

Output uses the CLI's visual language (spec UX 6): ✓ green done, ⏸ yellow waiting on you,
✗ red failed, `·` dim skipped. Commands print unwrapped so they copy whole. Prompts go to
the same stream as the output (stderr with `gpu setup --json`, so stdout stays one JSON
document).
"""

from __future__ import annotations

import sys
from typing import Protocol, TextIO

from rich.console import Console
from rich.text import Text

__all__ = ["Mark", "TerminalUi", "Ui"]


class Mark:
    DONE = ("✓", "green")
    WAIT = ("⏸", "yellow")
    FAIL = ("✗", "red")
    SKIP = ("·", "dim")
    WARN = ("!", "yellow")
    INFO = (" ", "")


class Ui(Protocol):
    interactive: bool  # a human can answer (stdin and the output stream are terminals)

    def title(self, text: str, sub: str = "") -> None: ...

    def step(self, n: int, total: int, title: str, sub: str = "") -> None: ...

    def item(self, mark: tuple[str, str], text: str, fix: str | None = None) -> None: ...

    def say(self, text: str, style: str = "") -> None: ...

    def command(self, argv_text: str) -> None: ...

    def block(self, text: str) -> None: ...

    def ask(self, question: str, default: bool) -> bool: ...

    def secret(self, prompt: str) -> str: ...

    def choose(self, question: str, options: list[tuple[str, str]], default: str) -> str: ...

    def write(self, text: str) -> None: ...


class TerminalUi:
    """rich output on `stream`; `input()` / getpass for answers."""

    def __init__(self, stream: TextIO | None = None, *, interactive: bool | None = None) -> None:
        self.stream = stream or sys.stdout
        self.console = Console(file=self.stream, highlight=False, soft_wrap=True)
        if interactive is None:
            try:
                interactive = sys.stdin.isatty() and self.stream.isatty()
            except (AttributeError, ValueError):
                interactive = False
        self.interactive = interactive

    def title(self, text: str, sub: str = "") -> None:
        line = Text()
        line.append(text, style="bold")
        if sub:
            line.append(f"  {sub}", style="dim")
        self.console.print(line)

    def step(self, n: int, total: int, title: str, sub: str = "") -> None:
        self.console.print(Text(""))
        line = Text()
        line.append(f"{n}/{total} ", style="dim")
        line.append(title, style="bold")
        if sub:
            line.append(f"  {sub}", style="dim")
        self.console.print(line)

    def item(self, mark: tuple[str, str], text: str, fix: str | None = None) -> None:
        icon, style = mark  # colour only on the icon (spec UX 6)
        line = Text()
        line.append(f"  {icon} ", style=style)
        line.append(text, style="dim" if mark == Mark.SKIP else "")
        self.console.print(line)
        if fix:
            self.command(fix)

    def say(self, text: str, style: str = "") -> None:
        self.console.print(Text(f"    {text}", style=style))

    def command(self, argv_text: str) -> None:
        line = Text()
        line.append("    $ ", style="dim")
        line.append(argv_text)
        line.no_wrap = False
        self.console.print(line, soft_wrap=True)

    def block(self, text: str) -> None:
        for raw in text.rstrip("\n").splitlines():
            style = ""
            if raw.startswith("+") and not raw.startswith("+++"):
                style = "green"
            elif raw.startswith("-") and not raw.startswith("---"):
                style = "red"
            elif raw.startswith(("@@", "---", "+++")):
                style = "dim"
            self.console.print(Text(f"    {raw}", style=style), soft_wrap=True)

    def _input(self, prompt: str) -> str:
        self.stream.write(prompt)
        self.stream.flush()
        try:
            return input()
        except EOFError:
            return ""

    def ask(self, question: str, default: bool) -> bool:
        hint = "[Y/n]" if default else "[y/N]"
        answer = self._input(f"  {question} {hint} ").strip().lower()
        if not answer:
            return default
        return answer in ("y", "yes")

    def secret(self, prompt: str) -> str:
        import getpass

        return getpass.getpass(f"  {prompt}", stream=self.stream)

    def choose(self, question: str, options: list[tuple[str, str]], default: str) -> str:
        keys = "/".join(k.upper() if k == default else k for k, _ in options)
        for key, label in options:
            self.console.print(Text(f"    {key}  {label}", style="dim"))
        while True:
            answer = self._input(f"  {question} [{keys}] ").strip().lower()
            if not answer:
                return default
            for key, _ in options:
                if answer == key:
                    return key

    def write(self, text: str) -> None:
        """Raw text from a reused flow (the status-line installer's diff and messages)."""
        for raw in text.splitlines():
            if raw.startswith(("---", "+++", "@@", "+", "-")):
                self.block(raw)
            elif raw.strip():
                self.console.print(Text(f"    {raw}", style="dim"), soft_wrap=True)
