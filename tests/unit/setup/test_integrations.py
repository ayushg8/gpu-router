"""Step 4: Claude Code status line, plugin, Codex MCP entry, colab skill. All against the
sandboxed ~/.claude and ~/.codex; each asks first and defaults to no."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any

from gpu_router.setup import integrations
from gpu_router.setup.base import Outcome
from tests.unit.setup.conftest import Sandbox, ScriptedUi, fail, ok

USER_LINE = {"type": "command", "command": 'bash "$HOME/.claude/statusline.sh"', "padding": 0}


def _settings(sandbox: Sandbox, data: dict[str, Any]) -> Path:
    return sandbox.write(".claude/settings.json", json.dumps(data, indent=2) + "\n", 0o644)


# =========================================================================== status line


def test_statusline_shows_the_diff_then_installs(sandbox: Sandbox) -> None:
    f = _settings(sandbox, {"model": "opus", "statusLine": USER_LINE})
    ui = ScriptedUi([True])
    ctx = sandbox.ctx(ui, only=["integration.statusline"])
    integrations.run(ctx)
    assert ctx.results[-1].outcome is Outcome.DONE, ui.text
    data = json.loads(f.read_text())
    assert "gpu-statusline.sh" in data["statusLine"]["command"]
    assert data["model"] == "opus"
    assert data["statusLine"]["padding"] == 0
    assert list(f.parent.glob("settings.json.gpu-router-*.bak"))  # backup kept
    diff_at = next(i for i, ln in enumerate(ui.lines) if ln.startswith("+") and "command" in ln)
    asked_at = next(i for i, ln in enumerate(ui.lines) if ln.startswith("? apply this change"))
    assert diff_at < asked_at  # the diff is on screen before the question


def test_statusline_declined_changes_nothing(sandbox: Sandbox) -> None:
    f = _settings(sandbox, {"statusLine": USER_LINE})
    before = f.read_bytes()
    ctx = sandbox.ctx(ScriptedUi([False]), only=["integration.statusline"])
    integrations.run(ctx)
    assert ctx.results[-1].outcome is Outcome.DECLINED
    assert f.read_bytes() == before
    assert not (sandbox.paths.home / "statusline").exists() or not any(
        (sandbox.paths.home / "statusline").iterdir()
    )


def test_statusline_twice_is_already_done(sandbox: Sandbox) -> None:
    _settings(sandbox, {"statusLine": USER_LINE})
    integrations.run(sandbox.ctx(ScriptedUi([True]), only=["integration.statusline"]))
    ctx = sandbox.ctx(ScriptedUi([]), only=["integration.statusline"])
    integrations.run(ctx)
    assert ctx.results[-1].outcome is Outcome.ALREADY


def test_statusline_no_terminal(sandbox: Sandbox) -> None:
    f = _settings(sandbox, {"statusLine": USER_LINE})
    before = f.read_bytes()
    ctx = sandbox.ctx(ScriptedUi([], interactive=False), only=["integration.statusline"])
    integrations.run(ctx)
    assert ctx.results[-1].outcome is Outcome.MANUAL
    assert ctx.results[-1].fix == "gpu statusline install"
    assert f.read_bytes() == before


def test_statusline_dry_run(sandbox: Sandbox) -> None:
    f = _settings(sandbox, {"statusLine": USER_LINE})
    before = f.read_bytes()
    ui = ScriptedUi([])
    ctx = sandbox.ctx(ui, only=["integration.statusline"], dry_run=True)
    integrations.run(ctx)
    assert ctx.results[-1].outcome is Outcome.SKIPPED
    assert f.read_bytes() == before
    assert any(ln.startswith("+") for ln in ui.lines)  # the diff is still shown


# =========================================================================== plugin


def _claude_installs(sandbox: Sandbox, version: str = "0.1.0") -> None:
    plugins = sandbox.user_home / ".claude" / "plugins"

    def add(_argv: list[str], _env: Any) -> Any:
        plugins.mkdir(parents=True, exist_ok=True)
        (plugins / "known_marketplaces.json").write_text(json.dumps({"gpu-router-local": {}}))
        return ok("added marketplace")

    def install(_argv: list[str], _env: Any) -> Any:
        doc = {"plugins": {"gpu-router@gpu-router-local": [{"version": version}]}}
        (plugins / "installed_plugins.json").write_text(json.dumps(doc))
        return ok("installed")

    sandbox.run.on(r"claude plugin marketplace add ", add)
    sandbox.run.on(r"claude plugin install gpu-router@gpu-router-local$", install)


def test_plugin_runs_the_two_claude_commands_after_yes(sandbox: Sandbox) -> None:
    sandbox.tools["claude"] = "/fake/bin/claude"
    _claude_installs(sandbox)
    ui = ScriptedUi([True])
    ctx = sandbox.ctx(ui, only=["integration.plugin"])
    integrations.run(ctx)
    assert ctx.results[-1].outcome is Outcome.DONE, ui.text
    ran = [c[1:] for c in sandbox.run.ran(r"^/fake/bin/claude ")]
    repo = sandbox.env.gpu_router_repo()
    assert ran == [
        ["plugin", "marketplace", "add", str(repo / "plugin")],  # type: ignore[operator]
        ["plugin", "install", "gpu-router@gpu-router-local"],
    ]
    assert "installed_plugins.json" in ui.text
    assert "enabledPlugins" in ui.text


def test_plugin_declined_by_default(sandbox: Sandbox) -> None:
    sandbox.tools["claude"] = "/fake/bin/claude"
    ui = ScriptedUi([None])  # enter
    ctx = sandbox.ctx(ui, only=["integration.plugin"])
    integrations.run(ctx)
    assert ctx.results[-1].outcome is Outcome.DECLINED
    assert not sandbox.run.ran(r"claude plugin")


def test_plugin_disabled_is_enabled(sandbox: Sandbox) -> None:
    sandbox.tools["claude"] = "/fake/bin/claude"
    sandbox.write(
        ".claude/plugins/installed_plugins.json",
        json.dumps({"plugins": {"gpu-router@gpu-router-local": [{"version": "0.1.0"}]}}),
    )
    sandbox.write(
        ".claude/settings.json",
        json.dumps({"enabledPlugins": {"gpu-router@gpu-router-local": False}}),
    )

    def enable(_argv: list[str], _env: Any) -> Any:
        sandbox.write(
            ".claude/settings.json",
            json.dumps({"enabledPlugins": {"gpu-router@gpu-router-local": True}}),
        )
        return ok()

    sandbox.run.on(r"claude plugin enable gpu-router@gpu-router-local$", enable)
    ctx = sandbox.ctx(ScriptedUi([True]), only=["integration.plugin"])
    integrations.run(ctx)
    assert ctx.results[-1].outcome is Outcome.DONE
    assert [c[1:] for c in sandbox.run.ran("claude")] == [
        ["plugin", "enable", "gpu-router@gpu-router-local"]
    ]


def test_plugin_without_claude_is_manual(sandbox: Sandbox) -> None:
    ctx = sandbox.ctx(ScriptedUi([]), only=["integration.plugin"])
    integrations.run(ctx)
    res = ctx.results[-1]
    assert res.outcome is Outcome.MANUAL
    assert res.fix is not None
    assert "claude plugin install gpu-router@gpu-router-local" in res.fix


def test_plugin_command_failure(sandbox: Sandbox) -> None:
    sandbox.tools["claude"] = "/fake/bin/claude"
    sandbox.run.on(r"claude plugin marketplace add ", fail(1, err="Error: marketplace invalid"))
    ctx = sandbox.ctx(ScriptedUi([True]), only=["integration.plugin"])
    integrations.run(ctx)
    res = ctx.results[-1]
    assert res.outcome is Outcome.FAILED
    assert "marketplace invalid" in res.summary


def test_plugin_honours_claude_config_dir(sandbox: Sandbox, tmp_path: Path) -> None:
    alt = tmp_path / "claude-alt"
    sandbox.environ["CLAUDE_CONFIG_DIR"] = str(alt)
    sandbox.tools["claude"] = "/fake/bin/claude"
    ctx = sandbox.ctx(ScriptedUi([False]), only=["integration.plugin"])
    integrations.run(ctx)
    assert str(alt / "plugins") in ctx.ui.text  # type: ignore[attr-defined]


# =========================================================================== codex


def test_codex_appends_the_entry_keeping_every_byte(sandbox: Sandbox) -> None:
    sandbox.tools["codex"] = "/fake/bin/codex"
    original = 'model = "gpt-5"\n\n[mcp_servers.other]\ncommand = "x"\n'
    cfg = sandbox.write(".codex/config.toml", original, 0o600)
    ui = ScriptedUi([True])
    ctx = sandbox.ctx(ui, only=["integration.codex"])
    integrations.run(ctx)
    assert ctx.results[-1].outcome is Outcome.DONE, ui.text
    text = cfg.read_text()
    assert text.startswith(original)
    doc = tomllib.loads(text)
    assert doc["mcp_servers"]["gpu-router"] == {
        "command": "gpu",
        "args": ["mcp"],
        "startup_timeout_sec": 30,
        "tool_timeout_sec": 330,
    }
    assert doc["mcp_servers"]["other"] == {"command": "x"}
    assert cfg.stat().st_mode & 0o777 == 0o600
    (bak,) = cfg.parent.glob("config.toml.gpu-router-*.bak")
    assert bak.read_text() == original
    assert any(ln.startswith("+[mcp_servers.gpu-router]") for ln in ui.lines)


def test_codex_new_file(sandbox: Sandbox) -> None:
    sandbox.tools["codex"] = "/fake/bin/codex"
    ctx = sandbox.ctx(ScriptedUi([True]), only=["integration.codex"])
    integrations.run(ctx)
    cfg = sandbox.user_home / ".codex" / "config.toml"
    assert ctx.results[-1].outcome is Outcome.DONE
    assert tomllib.loads(cfg.read_text())["mcp_servers"]["gpu-router"]["command"] == "gpu"


def test_codex_skipped_without_codex_and_already_done(sandbox: Sandbox) -> None:
    ctx = sandbox.ctx(ScriptedUi([]), only=["integration.codex"])
    integrations.run(ctx)
    assert ctx.results[-1].outcome is Outcome.SKIPPED
    sandbox.tools["codex"] = "/fake/bin/codex"
    sandbox.write(".codex/config.toml", '[mcp_servers.gpu-router]\ncommand = "gpu"\n')
    ctx = sandbox.ctx(ScriptedUi([]), only=["integration.codex"])
    integrations.run(ctx)
    assert ctx.results[-1].outcome is Outcome.ALREADY


def test_codex_inline_table_falls_back_to_the_cli(sandbox: Sandbox) -> None:
    sandbox.tools["codex"] = "/fake/bin/codex"
    original = 'mcp_servers = { other = { command = "x" } }\n'
    cfg = sandbox.write(".codex/config.toml", original)
    ctx = sandbox.ctx(ScriptedUi([]), only=["integration.codex"])
    integrations.run(ctx)
    res = ctx.results[-1]
    assert res.outcome is Outcome.MANUAL
    assert res.fix == integrations.CODEX_CLI_FIX
    assert cfg.read_text() == original


def test_codex_changed_while_asking_writes_nothing(sandbox: Sandbox) -> None:
    sandbox.tools["codex"] = "/fake/bin/codex"
    cfg = sandbox.write(".codex/config.toml", 'model = "a"\n')

    class Meddling(ScriptedUi):
        def ask(self, question: str, default: bool) -> bool:
            cfg.write_text('model = "b"\n')
            return True

    ctx = sandbox.ctx(Meddling(), only=["integration.codex"])
    integrations.run(ctx)
    assert ctx.results[-1].outcome is Outcome.FAILED
    assert cfg.read_text() == 'model = "b"\n'


# =========================================================================== colab skill


def test_colab_skill_is_moved_not_deleted(sandbox: Sandbox) -> None:
    skill = sandbox.write(".claude/skills/colab/SKILL.md", "---\nname: colab\n---\n")
    ctx = sandbox.ctx(ScriptedUi([True]), only=["integration.colab_skill"])
    integrations.run(ctx)
    assert ctx.results[-1].outcome is Outcome.DONE
    assert not skill.exists()
    assert (sandbox.user_home / ".claude/skills-disabled-colab/SKILL.md").exists()
    ctx = sandbox.ctx(ScriptedUi([]), only=["integration.colab_skill"])
    integrations.run(ctx)
    assert ctx.results[-1].outcome is Outcome.ALREADY


def test_colab_skill_declined(sandbox: Sandbox) -> None:
    skill = sandbox.write(".claude/skills/colab/SKILL.md", "x")
    ctx = sandbox.ctx(ScriptedUi([False]), only=["integration.colab_skill"])
    integrations.run(ctx)
    assert ctx.results[-1].outcome is Outcome.DECLINED
    assert skill.exists()
