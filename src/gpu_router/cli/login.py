"""`gpu login hf`: store a Hugging Face token for checkpoint storage (phase 5).

The token goes to the Keychain through gpu_router.secrets (invariant 12): read from a
no-echo prompt, stdin (`--stdin`), or copied from $HF_TOKEN / the hf CLI's token file
(`--import`), never from argv. `--remote` stores the least-privilege token remote
runtimes get (HF_TOKEN_REMOTE) instead of the daemon's own (HF_TOKEN).

The token is checked with one whoami call (skip with `--no-check`); the owner's name is
cached (non-secret) in <data dir>/storage/hf.json so the daemon need not ask again.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.text import Text

from gpu_router.errors import InvalidRequest

__all__ = ["login_app"]

login_app = typer.Typer(
    name="login",
    help="Store credentials gpu-router needs (Keychain; values never shown). Bare "
    "`gpu login` lists every login and how to add the missing ones.",
    no_args_is_help=False,
    invoke_without_command=True,
    add_completion=False,
    rich_markup_mode=None,
    pretty_exceptions_enable=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)

JsonOpt = Annotated[bool, typer.Option("--json", help="Machine-readable JSON on stdout.")]


@login_app.callback()
def login_overview(ctx: typer.Context, as_json: JsonOpt = False) -> None:
    """Bare `gpu login` (phase 8b): every login and its state, from local facts only."""
    if ctx.invoked_subcommand is not None:
        return
    from gpu_router.setup.cli import login_status

    login_status(as_json)


#: whoami answers by token fingerprint, kept briefly so a name check and a role check of
#: the same token make one request
_WHOAMI_CACHE: dict[str, tuple[float, Any, str | None]] = {}
_WHOAMI_TTL_S = 30.0


def _whoami_doc(token: str) -> tuple[Any, str | None]:
    """(whoami document or None, problem). A rejected token is a problem; an unreachable HF
    is not."""
    import time

    fp = hashlib.sha256(token.encode()).hexdigest()[:16]
    now = time.monotonic()
    hit = _WHOAMI_CACHE.get(fp)
    if hit is not None and now - hit[0] < _WHOAMI_TTL_S:
        return hit[1], hit[2]
    doc: Any = None
    problem: str | None = None
    try:
        from huggingface_hub import HfApi

        doc = HfApi(token=token).whoami()
    except Exception as exc:
        code = getattr(getattr(exc, "response", None), "status_code", None)
        if code in (401, 403):
            problem = "hugging face rejected this token"
    if len(_WHOAMI_CACHE) > 16:
        _WHOAMI_CACHE.clear()
    _WHOAMI_CACHE[fp] = (now, doc, problem)
    return doc, problem


def _whoami(token: str) -> tuple[str | None, str | None]:
    """(user name, problem). A rejected token is a problem; an unreachable HF is not
    (the token is stored anyway, with a note)."""
    who, problem = _whoami_doc(token)
    if problem is not None:
        return None, problem
    name = who.get("name") if isinstance(who, dict) else None
    return (str(name) if name else None), None


def token_role(doc: Any) -> str | None:
    """ "write", "read" or None (unknown) from a whoami document's auth.accessToken: a
    fine-grained token counts as write when any of its permissions writes."""
    auth = doc.get("auth") if isinstance(doc, dict) else None
    access = auth.get("accessToken") if isinstance(auth, dict) else None
    if not isinstance(access, dict):
        return None
    role = access.get("role")
    if role in ("write", "read"):
        return str(role)
    if role == "fineGrained":
        raw = access.get("fineGrained")
        fine: dict[str, Any] = raw if isinstance(raw, dict) else {}
        perms: list[str] = [str(p) for p in fine.get("global") or []]
        for scope in fine.get("scoped") or []:
            if isinstance(scope, dict):
                perms += [str(p) for p in scope.get("permissions") or []]
        return "write" if any(p.endswith("write") for p in perms) else "read"
    return None


def _role(token: str) -> str | None:
    """The token's role per Hugging Face ("write" / "read"), None when unknown."""
    doc, problem = _whoami_doc(token)
    return None if problem is not None else token_role(doc)


def _cache_namespace(token: str, name: str, home: Path | None = None) -> None:
    from gpu_router.paths import Paths

    home = home if home is not None else Paths.from_env().home
    path = home / "storage" / "hf.json"
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fp = hashlib.sha256(token.encode()).hexdigest()[:16]
    tmp = path.with_name(f".hf.json.{os.getpid()}")
    tmp.write_text(json.dumps({"namespace": name, "token_fp": fp}), encoding="utf-8")
    os.replace(tmp, path)


@login_app.command("hf")
def login_hf(
    stdin: Annotated[
        bool, typer.Option("--stdin", help="Read the token from stdin instead of a prompt.")
    ] = False,
    remote: Annotated[
        bool,
        typer.Option(
            "--remote",
            help="Store the token remote runtimes get (HF_TOKEN_REMOTE): a fine-grained token "
            "with write access to your buckets only.",
        ),
    ] = False,
    import_: Annotated[
        bool,
        typer.Option("--import", help="Copy the token from $HF_TOKEN or the hf CLI's token file."),
    ] = False,
    no_check: Annotated[
        bool, typer.Option("--no-check", help="Do not verify the token with Hugging Face.")
    ] = False,
    as_json: JsonOpt = False,
) -> None:
    """Store a Hugging Face token so checkpoints and datasets can move through a private
    HF Storage Bucket (<you>/gpu-router)."""
    from gpu_router import secrets as secret_store
    from gpu_router.checkpoint import tokens
    from gpu_router.cli import render
    from gpu_router.cli.app import Out, guarded

    out = Out(as_json)
    with guarded(out):
        name_key = tokens.REMOTE_SECRET if remote else tokens.ADMIN_SECRET
        source = "prompt"
        if import_:
            found = tokens.external_token()
            if found is None:
                raise InvalidRequest(
                    "no token found in $HF_TOKEN or the hf CLI's token file",
                    hint="run `gpu login hf` and paste a token from huggingface.co/settings/tokens",
                )
            source, value = found
        elif stdin or not sys.stdin.isatty():
            value = sys.stdin.read().strip()
            source = "stdin"
        else:
            import getpass

            value = getpass.getpass("Hugging Face token (not shown): ").strip()
        problem = tokens.token_shape_problem(value)
        if problem is not None:
            raise InvalidRequest(f"{problem}; nothing stored")
        secret_store.register_for_redaction(value)
        user: str | None = None
        checked = False
        if not no_check:
            user, rejected = _whoami(value)
            if rejected:
                raise InvalidRequest(
                    f"{rejected}; nothing stored",
                    hint="create a token at huggingface.co/settings/tokens and try again",
                )
            checked = user is not None
        secret_store.set_secret(name_key, value)
        if user and not remote:
            _cache_namespace(value, user)
        if as_json:
            out.emit(
                {
                    "secret": name_key,
                    "stored": True,
                    "user": user,
                    "verified": checked,
                    "source": source,
                }
            )
            return
        who = f" for {user}" if user else ""
        out.console.print(
            Text(f"{render.ICON_DONE} stored {name_key}{who} in the Keychain", style="green")
        )
        if not no_check and not checked:
            out.console.print(
                Text("  could not reach hugging face to verify it; stored anyway", style="yellow")
            )
        if remote:
            out.console.print(Text("  remote runtimes (kaggle, colab) now get this token", "dim"))
        else:
            where = f"hf://buckets/{user}/gpu-router" if user else "your gpu-router bucket"
            out.console.print(Text(f"  checkpoints and datasets go to {where} (private)", "dim"))
            out.console.print(
                Text(
                    "  remote runs (kaggle, colab) also need gpu login hf --remote with a "
                    "fine-grained token that can only write your buckets",
                    "dim",
                )
            )


@login_app.command("lightning")
def login_lightning(
    stdin: Annotated[
        bool,
        typer.Option(
            "--stdin",
            help="Read the user id and API key from stdin (two lines, or the JSON document "
            "`lightning login` writes).",
        ),
    ] = False,
    import_: Annotated[
        bool,
        typer.Option("--import", help="Copy the credentials from ~/.lightning/credentials.json."),
    ] = False,
    browser: Annotated[
        bool,
        typer.Option(
            "--browser",
            help="Sign in on lightning.ai in the browser (Google/GitHub/email); lightning.ai "
            "hands the keys back to a one-shot server on 127.0.0.1. Nothing to paste.",
        ),
    ] = False,
    no_open: Annotated[
        bool,
        typer.Option("--no-open", help="With --browser: print the sign-in URL, do not open it."),
    ] = False,
    no_check: Annotated[
        bool, typer.Option("--no-check", help="Do not verify them with lightning.ai.")
    ] = False,
    as_json: JsonOpt = False,
) -> None:
    """Store your Lightning AI user id and API key (lightning.ai > Settings > Keys) so
    jobs can run on Lightning's free monthly credits (phase 7a)."""
    from gpu_router.cli import render
    from gpu_router.cli.app import Out, guarded
    from gpu_router.providers.lightning.login import BrowserSignIn, run_login

    out = Out(as_json)
    with guarded(out):
        tty = sys.stdin.isatty() and not as_json

        def sign_in() -> tuple[str, str] | None:
            import webbrowser

            with BrowserSignIn() as flow:
                out.note(f"sign in to lightning.ai to connect gpu-router: {flow.url}")
                if not no_open and not webbrowser.open(flow.url):
                    out.note("could not open a browser; open the URL above yourself")
                out.note("waiting for lightning.ai (up to 10 min; Ctrl-C cancels)")
                return flow.wait()

        if browser:
            source = "browser"
        elif import_:
            source = "import"
        elif stdin or not sys.stdin.isatty():
            source = "stdin"
        else:
            source = "prompt"

        def ask(question: str) -> bool:
            reply = input(f"{question} [Y/n] ").strip().lower()
            return reply in ("", "y", "yes")

        def ask_no(question: str) -> bool:
            reply = input(f"{question} [y/N] ").strip().lower()
            return reply in ("y", "yes")

        def secret(prompt: str) -> str:
            import getpass

            return getpass.getpass(prompt)

        if source == "prompt" and not no_check:
            out.note("checking with lightning.ai (the first check can fetch the lightning SDK)")
        outcome = run_login(
            source=source,
            stdin_text=sys.stdin.read,
            prompt_secret=secret,
            confirm=ask if tty else None,
            check=not no_check,
            browser=sign_in,
            confirm_account=ask_no if tty else None,
        )
        if as_json:
            out.emit(outcome.to_json())
            return
        who = f" for {outcome.user}" if outcome.user else ""
        out.console.print(
            Text(
                f"{render.ICON_DONE} stored LIGHTNING_USER_ID and LIGHTNING_API_KEY{who} "
                "in the Keychain",
                style="green",
            )
        )
        if outcome.teamspace:
            out.console.print(Text(f"  jobs run in teamspace {outcome.teamspace}", "dim"))
        for note in outcome.notes:
            out.console.print(Text(f"  {note}", style="yellow"))
        out.console.print(Text("  check it with: gpu providers", "dim"))
