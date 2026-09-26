"""`gpu setup`, `gpu login kaggle`, `gpu login colab` and bare `gpu login` (phase 8b).

    gpu setup [--yes] [--only STEP|ITEM ...] [--dry-run] [--no-check] [--no-smoke]
              [--again] [--json]
    gpu login                       every login gpu-router uses and how to add the missing ones
    gpu login kaggle [--file PATH | --stdin] [--no-check] [--json]
    gpu login colab [--run] [--json]

`gpu setup --json` keeps stdout for one JSON document (the items and their outcomes); the
wizard's own lines and questions go to stderr.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from gpu_router.setup.context import SetupEnv

__all__ = ["login_status", "register"]

JsonOpt = Annotated[bool, typer.Option("--json", help="Machine-readable JSON on stdout.")]


def _env() -> SetupEnv:
    from gpu_router.paths import Paths
    from gpu_router.setup.context import SetupEnv

    return SetupEnv.real(Paths.from_env())


# =========================================================================== gpu setup


def setup(
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Answer yes to every question (non-interactive)."),
    ] = False,
    only: Annotated[
        list[str] | None,
        typer.Option(
            "--only",
            help="Only these steps or items: tools, logins, launchd, integration, check, or "
            "an item such as login.kaggle, plugin, codex, smoke. Repeatable.",
        ),
    ] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Show what each step would do; change nothing.")
    ] = False,
    no_check: Annotated[
        bool,
        typer.Option(
            "--no-check", help="Store credentials without checking them with the provider."
        ),
    ] = False,
    no_smoke: Annotated[
        bool, typer.Option("--no-smoke", help="Do not run the GPU smoke tests.")
    ] = False,
    again: Annotated[
        bool, typer.Option("--again", help="Re-run smoke tests that passed before.")
    ] = False,
    as_json: JsonOpt = False,
) -> None:
    """Set gpu-router up: install the CLIs, log in to each provider, start the daemon at
    login, connect Claude Code and Codex, then check everything with a GPU smoke test.
    Re-running skips what is done; an interrupted run continues where it stopped."""
    from gpu_router.cli import exitcodes
    from gpu_router.cli.app import Out, guarded
    from gpu_router.setup.base import Options
    from gpu_router.setup.ui import TerminalUi
    from gpu_router.setup.wizard import EXIT_INTERRUPTED, expand_only, run_wizard

    out = Out(as_json)
    with guarded(out):
        opts = Options(
            yes=yes,
            dry_run=dry_run,
            only=expand_only(only or []),
            check=not no_check,
            smoke=not no_smoke,
            again=again,
        )
        ui = TerminalUi(sys.stderr if as_json else sys.stdout)
        try:
            result = run_wizard(_env(), ui, opts)
        except KeyboardInterrupt:
            ui.say("")
            ui.say("stopped; `gpu setup` continues where it stopped", "dim")
            raise typer.Exit(EXIT_INTERRUPTED) from None
        if as_json:
            out.emit(result.to_json())
        if result.code != exitcodes.OK:
            raise typer.Exit(result.code)


# =========================================================================== gpu login


_STATUS_STYLE = {"ok": ("✓", "green"), "partial": ("!", "yellow"), "missing": ("·", "dim")}


def login_status(as_json: bool) -> None:
    """Bare `gpu login`: every login and its state (local facts only: Keychain names, file
    stats; nothing is sent anywhere)."""
    from gpu_router.cli.app import Out, guarded
    from gpu_router.setup.logins import login_rows

    out = Out(as_json)
    with guarded(out):
        rows = login_rows(_env())
        if as_json:
            out.emit({"logins": [r.to_json() for r in rows]})
            return
        table = Table.grid(padding=(0, 2))
        for _ in range(4):
            table.add_column(no_wrap=False)
        for r in rows:
            mark, style = _STATUS_STYLE.get(r.status, ("?", ""))
            table.add_row(
                Text(mark, style=style),
                Text(r.name),
                Text(r.summary, style="dim" if r.status == "ok" else ""),
                Text(r.command if r.status != "ok" else "", style="yellow"),
            )
        out.console.print(table)
        out.console.print(
            Text("gpu login <provider> adds or replaces one; gpu setup walks through all", "dim")
        )


def login_kaggle(
    file: Annotated[
        Path | None,
        typer.Option("--file", help="Import this kaggle.json (or access_token file)."),
    ] = None,
    stdin: Annotated[
        bool,
        typer.Option("--stdin", help="Read kaggle.json or an access token from stdin."),
    ] = False,
    no_check: Annotated[
        bool, typer.Option("--no-check", help="Do not check them with kaggle first.")
    ] = False,
    as_json: JsonOpt = False,
) -> None:
    """Store your Kaggle credentials in the Keychain: kaggle.json (from ~/.kaggle or
    ~/Downloads, where kaggle.com > Settings > API > Create New Token saves it) or an access
    token. The file stays where it is; gpu-router then uses the Keychain copy."""
    from gpu_router.cli import render
    from gpu_router.cli.app import Out, guarded
    from gpu_router.doctor.probe import home_label
    from gpu_router.errors import InvalidRequest
    from gpu_router.setup import logins

    out = Out(as_json)
    with guarded(out):
        env = _env()
        tty = sys.stdin.isatty() and not as_json
        source = "prompt"
        if file is not None:
            text, source = logins._read_file(file.expanduser()), str(file)
        elif stdin or not sys.stdin.isatty():
            text, source = sys.stdin.read(), "stdin"
        else:
            found = logins.kaggle_candidates(env)
            text = ""
            if found:
                label = home_label(found[0], env.user_home)
                reply = input(f"found {label}; import it into the Keychain? [Y/n] ").strip().lower()
                if reply in ("", "y", "yes"):
                    text, source = logins._read_file(found[0]), label
            if not text and tty:
                import getpass

                out.note(f"create a token at {logins.KAGGLE_TOKEN_URL}")
                text = getpass.getpass("kaggle.json contents or access token (not shown): ")
        if not text.strip():
            raise InvalidRequest(
                "no kaggle credentials given; nothing stored",
                hint=f"create a token at {logins.KAGGLE_TOKEN_URL}, then `gpu login kaggle`",
            )
        creds = logins.parse_kaggle(text)
        name, notes = logins.import_kaggle(env, creds, check=not no_check)
        if as_json:
            out.emit(
                {
                    "secret": name,
                    "stored": True,
                    "user": creds.username,
                    "verified": not no_check and not notes,
                    "source": source,
                    "notes": notes,
                }
            )
            return
        who = f" for {creds.username}" if creds.username else ""
        out.console.print(
            Text(f"{render.ICON_DONE} stored {name}{who} in the Keychain", style="green")
        )
        for note in notes:
            out.console.print(Text(f"  {note}", style="yellow"))
        out.console.print(
            Text("  gpu-router now uses the Keychain copy; check: gpu providers", "dim")
        )


def login_colab(
    run: Annotated[
        bool,
        typer.Option(
            "--run",
            help="Run the gcloud sign-in now if it is needed (it opens your browser).",
        ),
    ] = False,
    as_json: JsonOpt = False,
) -> None:
    """Check Colab's login: Google application-default credentials with the colaboratory
    scope. When they are missing, prints the exact gcloud command (and runs it with --run or
    when you say yes). The credentials stay in gcloud's own file; gpu-router never copies
    them."""
    from gpu_router.cli import render
    from gpu_router.cli.app import Out, guarded
    from gpu_router.doctor.model import Status
    from gpu_router.errors import InvalidRequest
    from gpu_router.setup import logins

    out = Out(as_json)
    with guarded(out):
        env = _env()
        row = logins.colab_check(env)
        if row is None:
            raise InvalidRequest("colab is not in providers.yaml")
        ran = False
        if row.status is not Status.OK and row.fix == logins.ADC_LOGIN:
            gcloud = logins.gcloud_path(env)
            tty = sys.stdin.isatty() and not as_json
            wants = run
            if not wants and tty and gcloud is not None:
                out.console.print(Text(f"colab: {row.summary}", style="yellow"))
                out.console.print(Text(f"  {logins.ADC_LOGIN}", style=""))
                reply = input("run it now? it opens your browser [Y/n] ").strip().lower()
                wants = reply in ("", "y", "yes")
            if wants:
                if gcloud is None:
                    raise InvalidRequest(
                        "gcloud is not installed",
                        hint=f"{logins.GCLOUD_INSTALL}, then `gpu login colab --run`",
                    )
                logins.run_adc_login(env, gcloud)
                ran = True
                row = logins.colab_check(env) or row
        ok = row.status in (Status.OK, Status.SKIP)
        if as_json:
            out.emit(
                {
                    "provider": "colab",
                    "ok": row.status is Status.OK,
                    "status": str(row.status),
                    "summary": row.summary,
                    "command": row.fix,
                    "ran_gcloud": ran,
                }
            )
        elif row.status is Status.OK:
            out.console.print(Text(f"{render.ICON_DONE} colab: {row.summary}", style="green"))
        else:
            style = "dim" if row.status is Status.SKIP else "red"
            out.console.print(Text(f"colab: {row.summary}", style=style))
            if row.fix:
                out.console.print(Text("  run:", style="dim"))
                out.console.print(Text(f"  {row.fix}"), soft_wrap=True)
        if not ok:
            raise typer.Exit(1)


def register(app: typer.Typer) -> None:
    """`gpu setup` on the main app; kaggle/colab on `gpu login`."""
    from gpu_router.cli.login import login_app

    app.command("setup")(setup)
    login_app.command("kaggle")(login_kaggle)
    login_app.command("colab")(login_colab)
