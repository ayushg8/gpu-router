"""Step 1, tools: find the CLIs gpu-router drives and install the missing ones with
`uv tool install` after one confirmation for the batch (phase 8b).

- tools.uv: PATH, ~/.local/bin, Homebrew; never installed by us (the curl command).
- tools.gpu: `gpu` on PATH; `uv tool install --editable <this repo>`.
- tools.kaggle: `kaggle` (PATH or ~/.local/bin); `uv tool install kaggle`.
- tools.colab: `colab`; `uv tool install google-colab-cli`.
- tools.lightning: uv's lightning-sdk tool env (or providers.lightning.python);
  `uv tool install lightning-sdk==<the pinned version>`.
- tools.hf: `hf`; `uv tool install 'huggingface_hub[cli]'`.

Provider tools are offered only for enabled providers. Lightning and hf are optional:
without the lightning-sdk env the adapter runs the pinned SDK through `uv run`, and
gpu-router carries its own huggingface_hub; the prompt says so.
"""

from __future__ import annotations

import os
import shlex
from dataclasses import dataclass
from pathlib import Path

from gpu_router.doctor.probe import home_label, tool_info
from gpu_router.setup.base import Ctx, Outcome
from gpu_router.setup.providers import config_or_none, enabled_kinds
from gpu_router.setup.ui import Mark

__all__ = ["ITEMS", "UV_INSTALL", "run"]

UV_INSTALL = "curl -LsSf https://astral.sh/uv/install.sh | sh"
INSTALL_TIMEOUT_S = 600.0


@dataclass(frozen=True)
class Tool:
    item: str
    label: str
    exe: str | None  # executable that proves it is installed (None: lightning's tool env)
    dist: str
    args: tuple[str, ...]  # after `uv tool install`
    kind: str | None  # provider kind that needs it (None: always)
    optional: str | None = None  # why it may be skipped
    version_args: tuple[str, ...] = ("--version",)


def _lightning_pin() -> str | None:
    try:
        from gpu_router.providers.lightning.sdk import DEFAULT_SDK_VERSION
    except ImportError:
        return None
    return DEFAULT_SDK_VERSION


def _tools(ctx: Ctx) -> list[Tool]:
    repo = ctx.env.gpu_router_repo()
    gpu_args = ("--editable", str(repo)) if repo is not None else ("gpu-router",)
    pin = _lightning_pin()
    return [
        Tool("tools.gpu", "gpu (gpu-router itself)", "gpu", "gpu-router", gpu_args, None),
        Tool("tools.kaggle", "kaggle CLI", "kaggle", "kaggle", ("kaggle",), "kaggle"),
        Tool(
            "tools.colab",
            "colab CLI",
            "colab",
            "google-colab-cli",
            ("google-colab-cli",),
            "colab",
            version_args=("version",),
        ),
        Tool(
            "tools.lightning",
            "lightning-sdk",
            None,
            "lightning-sdk",
            (f"lightning-sdk=={pin}" if pin else "lightning-sdk",),
            "lightning",
            optional="optional: without it the SDK runs through `uv run`, slower at first",
        ),
        Tool(
            "tools.hf",
            "hf CLI",
            "hf",
            "huggingface_hub",
            ("huggingface_hub[cli]",),
            None,
            optional="optional: gpu-router has its own huggingface_hub; the CLI is for you",
        ),
    ]


ITEMS = ("tools.uv", "tools.gpu", "tools.kaggle", "tools.colab", "tools.lightning", "tools.hf")


def _install_line(tool: Tool) -> str:
    return " ".join(shlex.quote(a) for a in ["uv", "tool", "install", *tool.args])


def _lightning_env(ctx: Ctx) -> tuple[bool, str]:
    """(present, where) for the lightning-sdk interpreter the adapter would use."""
    from gpu_router.doctor.probe import _dist_version

    config = config_or_none(ctx.env)
    settings = config.providers.get("lightning") if config is not None else None
    python = (settings.model_extra or {}).get("python") if settings is not None else None
    if python:
        return os.path.exists(str(python)), f"providers.lightning.python = {python}"
    base = ctx.env.environ.get("UV_TOOL_DIR")
    root = Path(base).expanduser() if base else ctx.env.user_home / ".local/share/uv/tools"
    tool_python = root / "lightning-sdk" / "bin" / "python"
    if tool_python.is_file():
        ver = _dist_version(str(tool_python), "lightning-sdk")
        return True, f"lightning-sdk {ver or '(version unknown)'} · uv tool env"
    return False, "no lightning-sdk tool env"


def _detect(ctx: Ctx, tool: Tool) -> tuple[bool, str]:
    """(installed, one-line description)."""
    env = ctx.env
    if tool.exe is None:
        return _lightning_env(ctx)
    info = tool_info(
        tool.exe,
        tool.dist,
        which=env.which,
        run=env.run,
        user_home=env.user_home,
        version_args=tool.version_args,
    )
    if info.path is None:
        return False, f"{tool.label} is not installed"
    ver = f" {info.version}" if info.version else ""
    return True, f"{tool.dist}{ver} · {home_label(info.path, env.user_home)}"


def _path_note(ctx: Ctx, tool: Tool) -> str | None:
    """`gpu` must be on PATH (Claude Code, Codex and the status line call it by name)."""
    if tool.exe != "gpu" or ctx.env.which("gpu"):
        return None
    return "uv tool update-shell"


def run(ctx: Ctx) -> None:
    env = ctx.env
    kinds = enabled_kinds(env)
    tools = [t for t in _tools(ctx) if ctx.selected(t.item) and (t.kind is None or t.kind in kinds)]
    uv = env.uv()
    if ctx.selected("tools.uv"):
        if uv is None:
            ctx.done(
                "tools.uv",
                Outcome.FAILED,
                "uv is missing: gpu-router installs its tools and runs local jobs with it",
                UV_INSTALL,
            )
        else:
            ctx.done("tools.uv", Outcome.ALREADY, f"uv · {home_label(uv, env.user_home)}")
    missing: list[Tool] = []
    for tool in tools:
        present, what = _detect(ctx, tool)
        if present:
            fix = _path_note(ctx, tool)
            if fix:
                ctx.done(
                    tool.item,
                    Outcome.MANUAL,
                    f"{what}, but ~/.local/bin is not on PATH (Claude Code and Codex run "
                    "`gpu` by name)",
                    fix,
                )
            else:
                ctx.done(tool.item, Outcome.ALREADY, what)
        else:
            missing.append(tool)
    if not missing:
        return
    if uv is None:
        for tool in missing:
            ctx.done(
                tool.item,
                Outcome.MANUAL,
                f"{tool.label} is not installed (needs uv)",
                _install_line(tool),
            )
        return
    for tool in missing:
        note = f"; {tool.optional}" if tool.optional else ""
        ctx.ui.item(Mark.SKIP, f"{tool.label} is not installed{note}")
    ctx.ui.say("to install:", "dim")
    for tool in missing:
        ctx.ui.command(_install_line(tool))
    if ctx.opts.dry_run:
        for tool in missing:
            ctx.dry(tool.item, f"install {tool.label}")
        return
    names = ", ".join(t.dist for t in missing)
    answer = ctx.confirm("tools.install", f"install {names} with uv?", default=True)
    if answer is None:
        for tool in missing:
            ctx.not_asked(tool.item, f"{tool.label} is not installed", _install_line(tool))
        return
    if not answer:
        for tool in missing:
            ctx.declined(tool.item, f"{tool.label} not installed", _install_line(tool))
        return
    for tool in missing:
        _install(ctx, uv, tool)


def _install(ctx: Ctx, uv: str, tool: Tool) -> None:
    env = ctx.env
    ctx.ui.say(f"installing {tool.dist} ...", "dim")
    child = {**env.environ, "NO_COLOR": "1"}
    res = env.run([uv, "tool", "install", *tool.args], INSTALL_TIMEOUT_S, env=child)
    present, what = _detect(ctx, tool)
    if res.ok and present:
        fix = _path_note(ctx, tool)
        if fix:
            ctx.done(
                tool.item, Outcome.MANUAL, f"installed {what}; now put ~/.local/bin on PATH", fix
            )
        else:
            ctx.done(tool.item, Outcome.DONE, f"installed {what}")
        return
    lines = [ln for ln in res.text.splitlines() if ln.strip()]
    why = res.error or (lines[-1][:200] if lines else f"exit {res.returncode}")
    if res.ok:
        why = f"uv finished but {tool.exe or 'the tool env'} is still not found"
    ctx.done(
        tool.item,
        Outcome.FAILED,
        f"could not install {tool.dist}: {why}",
        _install_line(tool),
    )
