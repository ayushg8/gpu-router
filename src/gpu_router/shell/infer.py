"""The shell's /infer (phase 7b): LLM calls and JSONL evals on the free inference lane,
the same requests as `gpu infer` (cli/infer.py) through the daemon (`/v1/infer`).

    /infer -m gpt-oss-20b what is 2+2?        one prompt; the reply, then where it ran
    /infer -m llama-3.1-8b --file evals.jsonl [-o out.jsonl] [-n 20]
    /infer -m gemini-3.5-flash --dry-run hi   the route only
    /infer --list                             providers, keys, models, quota

Keys cannot be typed into the TUI (no echo-free prompt there): /login groq says to run
`gpu login groq` in a terminal. Model output is untrusted: control characters are
stripped before it reaches the transcript.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich.table import Table
from rich.text import Text

from gpu_router.errors import InvalidRequest

if TYPE_CHECKING:
    from gpu_router.shell.commands import Ctx

__all__ = ["INFERENCE_LOGINS", "cmd_infer", "inference_login_steps"]

#: `/login <name>` for inference providers points at the terminal command
INFERENCE_LOGINS = {
    "groq": "gpu login groq",
    "gemini": "gpu login gemini",
    "cloudflare": "gpu login cloudflare --account-id <id>",
}
PREVIEW = 64
SINGLE_WAIT_S = 10.0
BATCH_WAIT_S = 30.0


def inference_login_steps(name: str) -> list[str] | None:
    command = INFERENCE_LOGINS.get(name)
    if command is None:
        return None
    return [
        f"{name} is an inference provider (/infer); its key is read without echo, so store it",
        f"from a terminal:  {command}",
        "then /infer --list shows it with a key",
    ]


def _clean(text: str) -> str:
    from gpu_router.models import strip_controls

    return "\n".join(strip_controls(line) for line in text.splitlines())


def _parser() -> Any:
    from gpu_router.shell.commands import parser

    p = parser("infer")
    p.add_argument("prompt", nargs="*")
    p.add_argument("--model", "-m")
    p.add_argument("--provider", "-p")
    p.add_argument("--system", "-s")
    p.add_argument("--max-tokens", type=int)
    p.add_argument("--temperature", "-t", type=float)
    p.add_argument("--file", "-f")
    p.add_argument("--out", "-o")
    p.add_argument("--limit", "-n", type=int)
    p.add_argument("--wait", type=float)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--list", action="store_true")
    p.add_argument("--json", action="store_true")  # accepted and ignored (pasted CLI lines)
    return p


def cmd_infer(ctx: Ctx, args: list[str]) -> None:
    from pydantic import ValidationError

    from gpu_router.cli import lanes
    from gpu_router.inference import remote
    from gpu_router.inference.models import InferRequest
    from gpu_router.shell.commands import UsageError

    ns = _parser().parse_args(args)
    client = ctx.client()
    if ns.list:
        ctx.emit(lanes.inference_providers_table(remote.providers(client)))
        ctx.emit(lanes.inference_quota_table(remote.quota(client)))
        return
    if ns.file:
        _batch(ctx, ns, client)
        return
    if not ns.model:
        raise UsageError("/infer needs --model, e.g. /infer -m gpt-oss-20b hello (--list: models)")
    text = " ".join(ns.prompt).strip()
    if not text:
        raise UsageError("/infer needs a prompt after the flags, or --file evals.jsonl")
    body = {
        "model": ns.model,
        "prompt": text,
        "provider": ns.provider,
        "system": ns.system,
        "max_tokens": ns.max_tokens,
        "temperature": ns.temperature,
        "wait_s": SINGLE_WAIT_S if ns.wait is None else ns.wait,
    }
    try:
        req = InferRequest.model_validate({k: v for k, v in body.items() if v is not None})
    except ValidationError as exc:
        raise UsageError(f"bad /infer arguments: {exc.errors()[0]['msg']}") from None
    if ns.dry_run:
        decision = remote.route(client, req)
        ctx.emit(Text(decision.reason))
        if decision.candidates:
            t = Table(box=None, show_header=False, pad_edge=False)
            t.add_column(no_wrap=True)
            t.add_column(overflow="fold")
            for c in decision.candidates:
                t.add_row(c.provider, Text(c.reason, style="dim"))
            ctx.emit(t)
        for r in decision.rejected:
            ctx.note(f"ruled out  {r.reason}")
        return
    ctx.note(f"asking {ns.provider or 'the best free provider'} for {ns.model}…")
    result = remote.infer(client, req)
    ctx.emit(Text(_clean(result.text) or "(empty reply)"))
    u = result.usage
    meta = f"{result.provider} · {result.model_id}"
    if u.input_tokens is not None or u.output_tokens is not None:
        meta += f" · {u.input_tokens or 0}+{u.output_tokens or 0} tokens"
    meta += f" · {result.latency_s:.1f}s"
    if result.quota:
        meta += f" · {result.quota.split(': ', 1)[-1]}"
    ctx.note(meta)
    for f in result.fallbacks:
        ctx.note(f"tried {f.provider} first: {f.message}")


def _batch(ctx: Ctx, ns: Any, client: Any) -> None:
    from gpu_router.inference import remote
    from gpu_router.inference.batch import parse_lines, run_batch

    path = Path(ns.file).expanduser()
    if not path.is_absolute():
        path = ctx.host.cwd / path
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise InvalidRequest(f"cannot read {path}: {exc.strerror or exc}") from None
    items = list(
        parse_lines(
            lines,
            model=ns.model,
            provider=ns.provider,
            system=ns.system,
            max_tokens=ns.max_tokens,
            temperature=ns.temperature,
        )
    )
    if ns.limit:
        items = items[: ns.limit]
    if not items:
        raise InvalidRequest(f"{path} has no prompts", hint="one JSON object per line")
    wait = BATCH_WAIT_S if ns.wait is None else ns.wait
    from dataclasses import replace

    items = [
        i
        if i.request is None
        else replace(i, request=i.request.model_copy(update={"wait_s": wait}))
        for i in items
    ]
    out_path: Path | None = None
    sink = None
    if ns.out:
        out_path = Path(ns.out).expanduser()
        if not out_path.is_absolute():
            out_path = ctx.host.cwd / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sink = out_path.open("w", encoding="utf-8")
    ctx.note(f"sending {len(items)} prompts from {path.name}…")

    def emit(rec: dict[str, Any]) -> None:
        if sink is not None:
            sink.write(json.dumps(rec, ensure_ascii=False) + "\n")
            sink.flush()
        if rec["ok"]:
            row = Text("✓ ", style="green")
            row.append(f"{rec['id']} {rec['provider']} {rec['latency_s']:.1f}s  ")
            flat = " ".join(_clean(rec["output"] or "").split())
            row.append(flat[:PREVIEW] + ("…" if len(flat) > PREVIEW else ""), style="dim")
        else:
            row = Text("✗ ", style="red")
            row.append(f"{rec['id']}  ")
            row.append(str(rec["error"]["message"]), style="dim")
        ctx.emit(row)

    try:
        summary = run_batch(items, lambda r: remote.infer(client, r), emit)
    finally:
        if sink is not None:
            sink.close()
    ctx.note(summary.line() + (f"; results in {out_path}" if out_path else ""))
    if summary.stopped:
        ctx.emit(Text(f"stopped early: {summary.stopped}", style="yellow"))
