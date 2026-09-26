"""Step 4, integration: Claude Code and Codex (phase 8b). Every item shows exactly what
changes and asks first; the default answer is no (these are other programs' settings).

- integration.statusline: done when doctor's row is ok; else `gpu statusline install`'s
  own flow (settings.json diff, [y/N], backup).
- integration.plugin: done when doctor's row is ok; else `claude plugin marketplace add
  <repo>/plugin` + `claude plugin install gpu-router@gpu-router-local` (or enable/update).
- integration.codex: done when doctor's row is ok (skipped without Codex); else appends
  [mcp_servers.gpu-router] to config.toml (diff, [y/N], backup).
- integration.colab_skill: done when <claude dir>/skills/colab is absent; else moves it
  to skills-disabled-colab.

All paths come from `SetupEnv` (CLAUDE_CONFIG_DIR, CODEX_HOME, the user home), so tests
run against sandboxes; the `claude` CLI runs through `SetupEnv.run` (tests fake it).
"""

from __future__ import annotations

import json
import os
import tomllib
from pathlib import Path
from typing import Any

from gpu_router.doctor.probe import home_label
from gpu_router.setup.base import Ctx, Outcome
from gpu_router.setup.ui import Mark

__all__ = ["CODEX_BLOCK", "ITEMS", "codex_plan", "plugin_commands", "run"]

ITEMS = (
    "integration.statusline",
    "integration.plugin",
    "integration.codex",
    "integration.colab_skill",
)
MARKETPLACE = "gpu-router-local"
CLAUDE_TIMEOUT_S = 180.0
CODEX_BLOCK = (
    "# gpu-router MCP server (added by `gpu setup`; see docs/codex/AGENTS-section.md)\n"
    "[mcp_servers.gpu-router]\n"
    'command = "gpu"\n'
    'args = ["mcp"]\n'
    "startup_timeout_sec = 30\n"
    "tool_timeout_sec = 330\n"
)
CODEX_CLI_FIX = "codex mcp add gpu-router -- gpu mcp"


class _UiWriter:
    """`out` for the reused status-line installer: forwards to the wizard UI as it goes
    (the diff must be on screen before the question) and keeps a copy."""

    def __init__(self, ctx: Ctx) -> None:
        self.ctx = ctx
        self.text: list[str] = []

    def write(self, text: str) -> int:
        self.text.append(text)
        self.ctx.ui.write(text)
        return len(text)

    def flush(self) -> None:
        pass

    def last_error(self) -> str:
        lines = [ln for ln in "".join(self.text).splitlines() if ln.startswith("gpu statusline:")]
        return lines[-1].removeprefix("gpu statusline:").strip() if lines else ""


# =========================================================================== status line


def _statusline(ctx: Ctx) -> None:
    from gpu_router.statusline import install

    item = "integration.statusline"
    env = ctx.env
    settings = env.claude_settings
    label = home_label(settings, env.user_home)
    info = install.status(settings, env.paths.home, user_home=env.user_home)
    if info.get("error"):
        ctx.done(item, Outcome.FAILED, f"status line: cannot read {label}: {info['error']}")
        return
    if info.get("installed") and info.get("wrapper_exists") and not info.get("other_home"):
        ctx.done(item, Outcome.ALREADY, "status line: gpu rows are installed under your own lines")
        return
    ctx.ui.item(
        Mark.SKIP,
        "status line: add gpu rows under your Claude Code status line while GPU work is "
        "active (your own lines stay as they are)",
    )
    asked: dict[str, bool | None] = {}

    def ask(prompt: str) -> str:
        answer = ctx.confirm(item, prompt.strip().removesuffix("[y/N]").strip(), default=False)
        asked["answer"] = answer
        return "y" if answer else "n"

    out = _UiWriter(ctx)
    rc = install.run_install(
        settings,
        env.paths.home,
        yes=False,
        dry_run=ctx.opts.dry_run,
        interactive=True,
        ask=ask,
        out=out,
        gpu_bin=env.gpu_executable(),
        user_home=env.user_home,
    )
    if ctx.opts.dry_run:
        ctx.dry(item, f"change statusLine in {label} (diff above)")
        return
    if "answer" in asked and asked["answer"] is None:
        ctx.not_asked(item, "status line: not installed", "gpu statusline install")
        return
    if "answer" in asked and asked["answer"] is False:
        ctx.declined(item, "status line unchanged", "gpu statusline install")
        return
    after = install.status(settings, env.paths.home, user_home=env.user_home)
    if rc == 0 and after.get("installed"):
        outcome = Outcome.DONE if "answer" in asked else Outcome.ALREADY
        ctx.done(
            item,
            outcome,
            "status line: gpu rows appear under your lines while GPU work is active "
            "(undo: gpu statusline uninstall)",
        )
        return
    ctx.done(
        item,
        Outcome.FAILED,
        f"status line: {out.last_error() or f'install exited {rc}'}",
        "gpu statusline install",
    )


# =========================================================================== plugin


def _known_marketplace(ctx: Ctx) -> bool:
    f = ctx.env.claude_dir / "plugins" / "known_marketplaces.json"
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(data, dict) and MARKETPLACE in data


def plugin_commands(ctx: Ctx, fix: str | None) -> list[list[str]] | None:
    """`claude` argv tails for the state doctor reported (None: cannot tell from here)."""
    from gpu_router.doctor.checks import PLUGIN_KEY

    repo = ctx.env.gpu_router_repo()
    if fix and fix.startswith("claude plugin enable"):
        return [["plugin", "enable", PLUGIN_KEY]]
    if fix and "plugin update" in fix:
        return [["plugin", "marketplace", "update", MARKETPLACE], ["plugin", "update", PLUGIN_KEY]]
    if repo is None:
        return None
    cmds: list[list[str]] = []
    if not _known_marketplace(ctx):
        cmds.append(["plugin", "marketplace", "add", str(repo / "plugin")])
    cmds.append(["plugin", "install", PLUGIN_KEY])
    return cmds


def _shell(argv: list[str]) -> str:
    import shlex

    return " ".join(shlex.quote(a) for a in argv)


def _plugin(ctx: Ctx) -> None:
    from gpu_router.doctor.checks import check_plugin
    from gpu_router.doctor.model import Status

    item = "integration.plugin"
    env = ctx.env
    row = check_plugin(env.probe())
    if row.status is Status.OK:
        ctx.done(item, Outcome.ALREADY, f"Claude Code plugin: {row.summary}")
        return
    cmds = plugin_commands(ctx, row.fix)
    if cmds is None:
        ctx.done(item, Outcome.MANUAL, f"Claude Code plugin: {row.summary}", row.fix)
        return
    lines = [_shell(["claude", *c]) for c in cmds]
    claude = env.find("claude", env.user_home / ".claude" / "local" / "claude")
    ctx.ui.item(Mark.SKIP, f"Claude Code plugin: {row.summary}")
    if claude is None:
        ctx.done(
            item,
            Outcome.MANUAL,
            "Claude Code plugin: `claude` is not on PATH; run these once it is",
            " && ".join(lines),
        )
        return
    plugins = home_label(env.claude_dir / "plugins", env.user_home)
    ctx.ui.say("runs:", "dim")
    for line in lines:
        ctx.ui.command(line)
    ctx.ui.say(
        f"Claude Code then records it in {plugins}/known_marketplaces.json and "
        f"installed_plugins.json and turns it on in enabledPlugins of "
        f"{home_label(env.claude_settings, env.user_home)}",
        "dim",
    )
    ctx.ui.say(
        "adds: the gpu-router MCP server (gpu mcp: gpu_submit ... gpu_infer), the gpu-router "
        "skill, /gpu-run /gpu-status /gpu-approve /gpu-statusline",
        "dim",
    )
    ctx.ui.say("undo: claude plugin uninstall gpu-router@gpu-router-local", "dim")
    if ctx.opts.dry_run:
        ctx.dry(item, "install the Claude Code plugin")
        return
    answer = ctx.confirm(item, "install the gpu-router plugin into Claude Code?", default=False)
    if answer is None:
        ctx.not_asked(item, "Claude Code plugin: not installed", " && ".join(lines))
        return
    if not answer:
        ctx.declined(item, "Claude Code plugin not installed", " && ".join(lines))
        return
    for cmd, line in zip(cmds, lines, strict=True):
        res = env.run([claude, *cmd], CLAUDE_TIMEOUT_S, env=env.environ)
        if not res.ok:
            tail = next((ln for ln in reversed(res.text.splitlines()) if ln.strip()), "")
            ctx.done(
                item,
                Outcome.FAILED,
                f"Claude Code plugin: `{line}` failed: {res.error or tail[:200] or res.returncode}",
                " && ".join(lines),
            )
            return
    again = check_plugin(env.probe())
    if again.status is Status.OK:
        ctx.done(
            item,
            Outcome.DONE,
            f"Claude Code plugin: {again.summary}; restart Claude Code sessions to load it",
        )
    else:
        ctx.done(item, Outcome.FAILED, f"Claude Code plugin: {again.summary}", again.fix)


# =========================================================================== codex


def codex_plan(config: Path) -> tuple[bytes | None, str, str | None]:
    """(current bytes or None, new text, problem). The block is appended as text, so every
    existing byte stays as it is; the result must parse and name the server."""
    try:
        raw: bytes | None = config.read_bytes()
    except FileNotFoundError:
        raw = None
    except OSError as exc:
        return None, "", f"cannot read {config}: {exc.strerror or exc}"
    before = raw.decode("utf-8", errors="replace") if raw is not None else ""
    if not before or before.endswith("\n\n"):
        sep = ""
    elif before.endswith("\n"):
        sep = "\n"
    else:
        sep = "\n\n"
    after = before + sep + CODEX_BLOCK
    try:
        doc = tomllib.loads(after)
    except tomllib.TOMLDecodeError as exc:
        return raw, after, f"config.toml would not parse with the entry appended ({exc})"
    server = (doc.get("mcp_servers") or {}).get("gpu-router")
    if not isinstance(server, dict) or server.get("command") != "gpu":
        return raw, after, "config.toml would not end up with the gpu-router server"
    return raw, after, None


def _codex(ctx: Ctx) -> None:
    from gpu_router.doctor.checks import check_codex
    from gpu_router.doctor.model import Status
    from gpu_router.statusline.install import (
        _atomic_write,
        _mode_of,
        _write_backup,
        unified_diff,
    )

    item = "integration.codex"
    env = ctx.env
    row = check_codex(env.probe())
    if row.status is Status.SKIP:
        ctx.done(item, Outcome.SKIPPED, f"Codex: {row.summary}")
        return
    if row.status is Status.OK:
        ctx.done(item, Outcome.ALREADY, f"Codex: {row.summary}")
        return
    config = env.codex_dir / "config.toml"
    label = home_label(config, env.user_home)
    raw, after, problem = codex_plan(config)
    if problem is not None:
        ctx.done(item, Outcome.MANUAL, f"Codex: {problem}", CODEX_CLI_FIX)
        return
    ctx.ui.item(Mark.SKIP, "Codex: no gpu-router MCP server")
    before = raw.decode("utf-8", errors="replace") if raw is not None else ""
    ctx.ui.block(unified_diff(config, before, after))
    ctx.ui.say(
        "tool_timeout_sec = 330: Codex stops a tool after 60 s by default, and a big "
        "gpu_submit or gpu_fetch can take longer",
        "dim",
    )
    ctx.ui.say(
        "the AGENTS.md section that tells Codex when to use it is in "
        "docs/codex/AGENTS-section.md (paste it yourself)",
        "dim",
    )
    if ctx.opts.dry_run:
        ctx.dry(item, f"append the gpu-router entry to {label}")
        return
    answer = ctx.confirm(item, f"add the gpu-router MCP server to {label}?", default=False)
    if answer is None:
        ctx.not_asked(item, "Codex: no gpu-router MCP server", CODEX_CLI_FIX)
        return
    if not answer:
        ctx.declined(item, f"{label} unchanged", CODEX_CLI_FIX)
        return
    try:
        current: bytes | None = config.read_bytes()
    except FileNotFoundError:
        current = None
    except OSError as exc:
        ctx.done(item, Outcome.FAILED, f"Codex: cannot read {label}: {exc.strerror or exc}")
        return
    if current != raw:
        ctx.done(
            item,
            Outcome.FAILED,
            f"Codex: {label} changed while you were deciding; nothing written",
            "gpu setup --only integration.codex",
        )
        return
    target = Path(os.path.realpath(config))
    mode = _mode_of(target, 0o600)
    backup: Path | None = None
    try:
        if raw is not None:
            backup = _write_backup(target, raw, mode=mode)
        _atomic_write(target, after.encode("utf-8"), mode=mode)
    except OSError as exc:
        ctx.done(
            item,
            Outcome.FAILED,
            f"Codex: cannot write {label}: {exc.strerror or exc}; nothing changed",
            CODEX_CLI_FIX,
        )
        return
    if backup is not None:
        ctx.ui.say(f"backup: {home_label(backup, env.user_home)}", "dim")
    again = check_codex(env.probe())
    if again.status is Status.OK:
        ctx.done(item, Outcome.DONE, f"Codex: added the gpu-router MCP server to {label}")
    else:
        ctx.done(item, Outcome.FAILED, f"Codex: {again.summary}", again.fix)


# =========================================================================== colab skill


def _colab_skill(ctx: Ctx) -> None:
    item = "integration.colab_skill"
    env = ctx.env
    skill = env.claude_dir / "skills" / "colab"
    target = env.claude_dir / "skills-disabled-colab"
    if not skill.exists():
        ctx.done(item, Outcome.ALREADY, "colab skill: none competing for “run this on a GPU”")
        return
    src, dst = home_label(skill, env.user_home), home_label(target, env.user_home)
    ctx.ui.item(
        Mark.SKIP,
        f"colab skill: {src} also claims “run this on a GPU” and drives colab directly, past "
        "the quota ledger",
    )
    if target.exists():
        ctx.done(
            item,
            Outcome.MANUAL,
            f"colab skill: {dst} already exists; move {src} somewhere else yourself",
        )
        return
    ctx.ui.say(f"moves {src} to {dst} (kept, not deleted; move it back to undo)", "dim")
    if ctx.opts.dry_run:
        ctx.dry(item, f"move {src} to {dst}")
        return
    answer = ctx.confirm(item, "disable the colab skill for job running?", default=False)
    fix = f"mv {src} {dst}"
    if answer is None:
        ctx.not_asked(item, "colab skill: still active", fix)
        return
    if not answer:
        ctx.declined(item, f"{src} stays active", fix)
        return
    try:
        os.rename(skill, target)
    except OSError as exc:
        ctx.done(item, Outcome.FAILED, f"colab skill: cannot move it: {exc.strerror or exc}", fix)
        return
    ctx.done(item, Outcome.DONE, f"colab skill: moved to {dst}")


# =========================================================================== step


def run(ctx: Ctx) -> None:
    steps: tuple[tuple[str, Any], ...] = (
        ("integration.statusline", _statusline),
        ("integration.plugin", _plugin),
        ("integration.codex", _codex),
        ("integration.colab_skill", _colab_skill),
    )
    for item, fn in steps:
        if ctx.selected(item):
            fn(ctx)
