"""`gpu infer` and `gpu login groq|gemini|cloudflare` (phase 7b: the inference lane).

    gpu infer --model gpt-oss-20b "What is 2+2?"          one prompt; the reply on stdout,
                                                            provider + quota on stderr
    gpu infer -m gpt-oss-20b -p groq --json "..."          pin a provider; InferResult JSON
    gpu infer -m llama-3.1-8b --file evals.jsonl -o out.jsonl   a JSONL eval batch
    gpu infer -m gemini-3.5-flash --dry-run "..."          where it would go, and why
    gpu infer --list                                        providers, keys, models, quota

The daemon routes and calls (invariant 2); this module only builds requests and prints.
Keys go to the Keychain through gpu_router.secrets (invariant 12): read with getpass (no
echo) or from stdin, never from argv; `gpu login hf` (cli/login.py) already stores the
Hugging Face token the `hf` entry uses.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer
from rich.table import Table
from rich.text import Text

from gpu_router.errors import InvalidRequest
from gpu_router.inference.models import InferRequest, InferResult, InferRoute

if TYPE_CHECKING:
    import httpx

    from gpu_router.cli.app import Out
    from gpu_router.inference.catalog import InferenceEntry

__all__ = ["register"]

JsonOpt = Annotated[bool, typer.Option("--json", help="Machine-readable JSON on stdout.")]
SINGLE_WAIT_S = 10.0
BATCH_WAIT_S = 30.0
PREVIEW_CHARS = 72

#: tests point `gpu login`'s key check at a mock transport (never a real provider)
verify_transport: httpx.BaseTransport | None = None


# --------------------------------------------------------------------------- printing


def print_route(out: Out, decision: InferRoute) -> None:
    style = {"place": "green", "wait": "yellow", "no_fit": "red"}[decision.outcome]
    icon = {"place": "→", "wait": "⏸", "no_fit": "✗"}[decision.outcome]
    line = Text(f"{icon} ", style=style)
    line.append(decision.reason)
    out.console.print(line)
    if decision.candidates:
        t = Table(box=None, pad_edge=False, show_edge=False, header_style="dim")
        for col in ("candidate", "model id", "score", "why"):
            t.add_column(col, no_wrap=col != "why", overflow="fold")
        for c in decision.candidates:
            t.add_row(c.provider, c.model_id, f"{c.score:g}", Text(c.reason, style="dim"))
        out.console.print(t)
    for r in decision.rejected:
        out.console.print(Text(f"  ruled out  {r.reason}", style="dim"))


def print_result(out: Out, result: InferResult) -> None:
    """The reply on stdout (untouched, pipeable); where it ran and what is left on stderr."""
    sys.stdout.write(result.text)
    if not result.text.endswith("\n"):
        sys.stdout.write("\n")
    sys.stdout.flush()
    u = result.usage
    tokens = ""
    if u.input_tokens is not None or u.output_tokens is not None:
        tokens = f" · {u.input_tokens or 0}+{u.output_tokens or 0} tokens"
    model = (
        result.model if result.model == result.model_id else f"{result.model} ({result.model_id})"
    )
    meta = f"{result.provider} · {model}{tokens} · {result.latency_s:.1f}s"
    if result.quota:
        meta += f" · {result.quota.split(': ', 1)[-1]}"
    out.err.print(Text(meta, style="dim"), soft_wrap=True)
    if result.finish_reason == "length":
        why = "an empty reply: " if not result.text.strip() else ""
        out.err.print(
            Text(
                f"  {why}cut at the token limit (thinking models spend it on reasoning first);"
                " raise --max-tokens",
                style="yellow",
            ),
            soft_wrap=True,
        )
    for f in result.fallbacks:
        out.err.print(Text(f"  tried {f.provider} first: {f.message}", style="dim"), soft_wrap=True)


def _preview(text: str | None) -> str:
    flat = " ".join((text or "").split())
    return flat if len(flat) <= PREVIEW_CHARS else flat[: PREVIEW_CHARS - 1] + "…"


# --------------------------------------------------------------------------- gpu infer


def _read_prompt(words: list[str] | None) -> str | None:
    if not words:
        return None
    if words == ["-"]:
        return sys.stdin.read()
    return " ".join(words)


def infer_command(
    prompt: Annotated[
        list[str] | None,
        typer.Argument(help="The prompt (words are joined); - reads it from stdin."),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option("--model", "-m", help="Model alias (gpt-oss-20b, llama-3.1-8b, ...) or id."),
    ] = None,
    provider: Annotated[
        str | None,
        typer.Option("--provider", "-p", help="Pin one: groq, cloudflare, gemini, hf."),
    ] = None,
    system: Annotated[str | None, typer.Option("--system", "-s", help="System prompt.")] = None,
    max_tokens: Annotated[
        int | None, typer.Option("--max-tokens", min=1, max=65_536, help="Output cap.")
    ] = None,
    temperature: Annotated[
        float | None, typer.Option("--temperature", "-t", min=0.0, max=2.0)
    ] = None,
    file: Annotated[
        Path | None,
        typer.Option("--file", "-f", help="JSONL eval file: one {prompt|messages, ...} per line."),
    ] = None,
    out_path: Annotated[
        Path | None, typer.Option("--out", "-o", help="Write the batch results here (JSONL).")
    ] = None,
    limit: Annotated[
        int | None, typer.Option("--limit", "-n", min=1, help="Only the first N lines.")
    ] = None,
    wait: Annotated[
        float | None,
        typer.Option(
            "--wait",
            min=0.0,
            max=60.0,
            help="Seconds to wait for a provider in a per-minute cooldown (default 10, "
            "batches 30).",
        ),
    ] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Show where it would go and why; spend nothing.")
    ] = False,
    list_: Annotated[
        bool, typer.Option("--list", help="Inference providers, keys, models and quota left.")
    ] = False,
    as_json: JsonOpt = False,
) -> None:
    """LLM calls and evals on free inference APIs (Groq, Cloudflare Workers AI, Gemini, HF),
    routed by model and daily quota left. Not a GPU job."""
    from gpu_router.cli.app import Out, connect, guarded

    out = Out(as_json)
    with guarded(out):
        if list_:
            _list(out, connect(out))
            return
        if model is None and file is None:
            raise InvalidRequest(
                "--model is required",
                hint='e.g. gpu infer -m gpt-oss-20b "hello"; gpu infer --list shows the models',
            )
        if file is not None:
            if prompt:
                raise InvalidRequest("give a prompt or --file, not both")
            _batch(
                out,
                file=file,
                out_path=out_path,
                limit=limit,
                model=model,
                provider=provider,
                system=system,
                max_tokens=max_tokens,
                temperature=temperature,
                wait=BATCH_WAIT_S if wait is None else wait,
            )
            return
        text = _read_prompt(prompt)
        if text is None or not text.strip():
            raise InvalidRequest(
                "no prompt", hint="pass it as an argument, or - to read stdin, or use --file"
            )
        req = _request(
            model=model or "",
            prompt=text,
            provider=provider,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            wait_s=SINGLE_WAIT_S if wait is None else wait,
        )
        from gpu_router.inference import remote

        client = connect(out)
        if dry_run:
            from gpu_router.cli import exitcodes

            decision = remote.route(client, req)
            if as_json:
                out.emit(decision.model_dump(mode="json"))
            else:
                print_route(out, decision)
            if decision.outcome == "no_fit":  # like `gpu route`: nothing can take it now
                raise typer.Exit(exitcodes.NO_FIT)
            return
        result = remote.infer(client, req)
        if as_json:
            out.emit(result.model_dump(mode="json"))
            return
        print_result(out, result)


def _request(**fields: Any) -> InferRequest:
    from pydantic import ValidationError

    try:
        return InferRequest.model_validate({k: v for k, v in fields.items() if v is not None})
    except ValidationError as exc:
        first = exc.errors()[0]
        raise InvalidRequest(f"bad request: {first['msg']}") from None


def _list(out: Out, client: Any) -> None:
    from gpu_router.cli import lanes
    from gpu_router.inference import remote

    views = remote.providers(client)
    quotas = remote.quota(client)
    if out.json:
        out.emit(
            {
                "providers": [v.model_dump(mode="json") for v in views],
                "quota": [q.model_dump(mode="json") for q in quotas],
            }
        )
        return
    out.console.print(lanes.inference_providers_table(views))
    out.console.print()
    out.console.print(lanes.inference_quota_table(quotas))
    notes = lanes.inference_notes(views)
    if notes:
        out.console.print()
        for line in notes:
            out.console.print(line)
    missing = [v.login for v in views if v.routable and not v.logged_in]
    if missing:
        out.console.print(
            Text(
                "add a key: " + ", ".join(f"gpu login {n}" for n in dict.fromkeys(missing)),
                style="dim",
            )
        )


def _batch(
    out: Out,
    *,
    file: Path,
    out_path: Path | None,
    limit: int | None,
    model: str | None,
    provider: str | None,
    system: str | None,
    max_tokens: int | None,
    temperature: float | None,
    wait: float,
) -> None:
    from gpu_router.cli.app import connect
    from gpu_router.inference import remote
    from gpu_router.inference.batch import parse_lines, run_batch

    try:
        lines = file.expanduser().read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise InvalidRequest(
            f"cannot read {file}: {exc.strerror or exc}", hint="give a JSONL file"
        ) from None
    items = list(
        parse_lines(
            lines,
            model=model,
            provider=provider,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    )
    if limit is not None:
        items = items[:limit]
    if not items:
        raise InvalidRequest(f"{file} has no prompts", hint="one JSON object per line")
    items = [
        i if i.request is None else _with_wait(i, wait)  # batches ride out RPM cooldowns
        for i in items
    ]
    client = connect(out)
    sink = None
    if out_path is not None:
        out_path = out_path.expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sink = out_path.open("w", encoding="utf-8")

    def emit(rec: dict[str, Any]) -> None:
        line = json.dumps(rec, ensure_ascii=False)
        if sink is not None:
            sink.write(line + "\n")
            sink.flush()
        if out.json:
            if sink is None:
                sys.stdout.write(line + "\n")
                sys.stdout.flush()
            return
        n = f"{rec['line']}"
        if rec["ok"]:
            row = Text("✓ ", style="green")
            row.append(f"{rec['id']:<8} {rec['provider']:<10} {rec['latency_s']:.1f}s  ")
            row.append(_preview(rec["output"]), style="dim")
        else:
            row = Text("✗ ", style="red")
            row.append(f"{rec['id']:<8} line {n}  ")
            row.append(str(rec["error"]["message"]), style="dim")
        out.console.print(row, soft_wrap=True)

    try:
        summary = run_batch(items, lambda r: remote.infer(client, r), emit)
    finally:
        if sink is not None:
            sink.close()
    doc = {"summary": {**summary.as_dict(), "results": str(out_path) if out_path else None}}
    if out.json:
        out.emit(doc)
        return
    msg = summary.line() + (f"; results in {out_path}" if out_path else "")
    out.console.print(Text(msg, style="dim" if not summary.failed else "yellow"))
    if summary.stopped:
        out.console.print(Text(f"stopped early: {summary.stopped}", style="yellow"))


def _with_wait(item: Any, wait: float) -> Any:
    from dataclasses import replace

    return replace(item, request=item.request.model_copy(update={"wait_s": wait}))


# --------------------------------------------------------------------------- gpu login


def _inference_entry(name: str) -> InferenceEntry:
    from gpu_router.cli.lanes import load_listing_catalog
    from gpu_router.inference.catalog import load_inference_catalog

    catalog = load_listing_catalog()
    entry = load_inference_catalog(catalog).get(name) if catalog is not None else None
    if entry is None:
        raise InvalidRequest(f"no inference provider named {name!r} in providers.yaml")
    return entry


def _key_problem(value: str, what: str) -> str | None:
    if not value:
        return f"the {what} is empty"
    if any(ch.isspace() for ch in value):
        return f"the {what} contains whitespace"
    if len(value) < 16:
        return f"that is too short for a {what}"
    return None


def login_inference(
    name: str,
    *,
    label: str,
    stdin: bool,
    no_check: bool,
    account_id: str | None,
    as_json: bool,
) -> None:
    """Store `name`'s key(s) in the Keychain after one free check call."""
    from pydantic import SecretStr

    from gpu_router import secrets as secret_store
    from gpu_router.cli import render
    from gpu_router.cli.app import Out, guarded
    from gpu_router.inference.clients import ChatClient
    from gpu_router.inference.errors import InferAuthRequired, InferenceError

    out = Out(as_json)
    with guarded(out):
        entry = _inference_entry(name)
        values: dict[str, str] = {}
        for role in entry.secrets:
            if role == "account_id":
                value = (account_id or "").strip()
                if not value:
                    if stdin or not sys.stdin.isatty():
                        raise InvalidRequest(
                            f"{label} also needs --account-id", hint=entry.link or None
                        )
                    value = input(f"{label} account id: ").strip()
                if not value or any(ch.isspace() for ch in value):
                    raise InvalidRequest(f"that is not a {label} account id; nothing stored")
                values[role] = value
                continue
            if stdin or not sys.stdin.isatty():
                value = sys.stdin.readline().strip()
            else:
                import getpass

                value = getpass.getpass(f"{label} API key (not shown): ").strip()
            problem = _key_problem(value, "key")
            if problem is not None:
                raise InvalidRequest(f"{problem}; nothing stored", hint=entry.link or None)
            secret_store.register_for_redaction(value)
            values[role] = value
        verified = False
        note: str | None = None
        if not no_check and entry.verify_url:
            client = ChatClient(entry, transport=verify_transport, timeout_s=20.0)
            try:
                client.verify({r: SecretStr(v) for r, v in values.items()})
                verified = True
            except InferAuthRequired as exc:
                raise InvalidRequest(f"{exc.message}; nothing stored", hint=exc.hint) from None
            except InferenceError as exc:
                note = f"could not check it ({exc.message}); stored anyway"
            finally:
                client.close()
        stored = []
        for role, value in values.items():
            secret_store.set_secret(entry.secrets[role], value)
            stored.append(entry.secrets[role])
        if as_json:
            out.emit({"provider": entry.name, "stored": stored, "verified": verified, "note": note})
            return
        out.console.print(
            Text(f"{render.ICON_DONE} stored the {label} key in the Keychain", style="green")
        )
        if note:
            out.console.print(Text(f"  {note}", style="yellow"))
        elif verified:
            out.console.print(Text(f"  {label} accepted it", style="dim"))
        if entry.note:
            out.console.print(Text(f"  {entry.note}", style="dim"))
        out.console.print(
            Text(f'  try: gpu infer -m {next(iter(entry.models), "MODEL")} "hello"', style="dim")
        )


StdinOpt = Annotated[
    bool, typer.Option("--stdin", help="Read the key from stdin instead of a prompt.")
]
NoCheckOpt = Annotated[
    bool, typer.Option("--no-check", help="Store without checking the key with the provider.")
]


def register(app: typer.Typer) -> None:
    """Add `gpu infer` to the app and the inference logins to `gpu login`."""
    from gpu_router.cli.login import login_app

    app.command("infer")(infer_command)

    @login_app.command("groq")
    def login_groq(
        stdin: StdinOpt = False, no_check: NoCheckOpt = False, as_json: JsonOpt = False
    ) -> None:
        """Store a Groq API key (console.groq.com/keys) for `gpu infer`."""
        login_inference(
            "groq", label="Groq", stdin=stdin, no_check=no_check, account_id=None, as_json=as_json
        )

    @login_app.command("gemini")
    def login_gemini(
        stdin: StdinOpt = False, no_check: NoCheckOpt = False, as_json: JsonOpt = False
    ) -> None:
        """Store a Google AI Studio (Gemini API) key (aistudio.google.com/apikey)."""
        login_inference(
            "gemini",
            label="Gemini",
            stdin=stdin,
            no_check=no_check,
            account_id=None,
            as_json=as_json,
        )

    @login_app.command("cloudflare")
    def login_cloudflare(
        account_id: Annotated[
            str | None,
            typer.Option("--account-id", help="Cloudflare account id (dashboard, Workers AI)."),
        ] = None,
        stdin: StdinOpt = False,
        no_check: NoCheckOpt = False,
        as_json: JsonOpt = False,
    ) -> None:
        """Store a Cloudflare API token with Workers AI access, and the account id."""
        login_inference(
            "cloudflare",
            label="Cloudflare",
            stdin=stdin,
            no_check=no_check,
            account_id=account_id,
            as_json=as_json,
        )
