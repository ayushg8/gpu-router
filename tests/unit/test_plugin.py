"""plugin/: the Claude Code plugin is one install (spec "Agent integration"): MCP server,
the skill, the slash commands and the status-line wrapper, all consistent with the code.

Static checks only; `claude plugin validate plugin` is the manual check (CLAUDE.md).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import tomllib
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO = Path(__file__).parents[2]
PLUGIN = REPO / "plugin"
SPEC_TOOLS = {
    "gpu_submit",
    "gpu_status",
    "gpu_logs",
    "gpu_fetch",
    "gpu_cancel",
    "gpu_quota",
    "gpu_route",
}
#: phase 7b: gpu_infer (the inference lane) is additive to the spec's seven tools
TOOLS = SPEC_TOOLS | {"gpu_infer"}
COMMANDS = {"gpu-run", "gpu-status", "gpu-approve", "gpu-statusline"}
HUMAN_ONLY = {"gpu-approve", "gpu-statusline"}  # Claude can never invoke these itself


def _json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def _frontmatter(path: Path) -> tuple[dict[str, Any], str]:
    text = path.read_text(encoding="utf-8")
    match = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.S)
    assert match, f"{path.name}: no YAML frontmatter"
    meta = yaml.safe_load(match.group(1))
    assert isinstance(meta, dict)
    return meta, match.group(2)


def test_manifests_agree_on_name_and_version() -> None:
    plugin = _json(PLUGIN / ".claude-plugin" / "plugin.json")
    market = _json(PLUGIN / ".claude-plugin" / "marketplace.json")
    version = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]["version"]
    assert plugin["name"] == "gpu-router"
    assert plugin["version"] == version
    assert market["name"] == "gpu-router-local"
    [entry] = market["plugins"]
    assert entry["name"] == plugin["name"]
    assert entry["version"] == version
    assert entry["source"] == "./"  # the marketplace and the plugin are the same folder


def test_repo_root_marketplace_installs_the_same_plugin_from_github() -> None:
    """`claude plugin marketplace add ayushg8/gpu-router` clones the repo and reads
    .claude-plugin/marketplace.json at its root: it must point at plugin/ and agree with
    the checkout-local marketplace, so both installs give the same plugin."""
    root = _json(REPO / ".claude-plugin" / "marketplace.json")
    local = _json(PLUGIN / ".claude-plugin" / "marketplace.json")
    plugin = _json(PLUGIN / ".claude-plugin" / "plugin.json")
    assert root["name"] == "gpu-router"  # install id: gpu-router@gpu-router
    assert root["name"] != local["name"]  # both can be added side by side
    [entry] = root["plugins"]
    assert entry["source"] == "./plugin"
    assert entry["name"] == plugin["name"]
    assert entry["version"] == plugin["version"] == root["metadata"]["version"]
    assert {k: v for k, v in entry.items() if k != "source"} == {
        k: v for k, v in local["plugins"][0].items() if k != "source"
    }
    assert (REPO / entry["source"] / ".claude-plugin" / "plugin.json").is_file()
    assert root["owner"] == local["owner"]


def test_mcp_server_is_gpu_mcp() -> None:
    servers = _json(PLUGIN / ".mcp.json")["mcpServers"]
    assert servers == {"gpu-router": {"command": "gpu", "args": ["mcp"]}}


def test_server_registers_exactly_the_spec_tools() -> None:
    from gpu_router.mcp.server import build_server

    async def names() -> set[str]:
        return {tool.name for tool in await build_server().list_tools()}

    assert asyncio.run(names()) == TOOLS  # no approve/deny tool, ever
    description = _json(PLUGIN / ".claude-plugin" / "plugin.json")["description"]
    for tool in TOOLS:
        assert tool in description


def test_commands_are_the_spec_set_plus_statusline() -> None:
    found = {p.stem for p in (PLUGIN / "commands").glob("*.md")}
    assert found == COMMANDS
    description = _json(PLUGIN / ".claude-plugin" / "plugin.json")["description"]
    for name in COMMANDS:
        assert f"/{name}" in description


@pytest.mark.parametrize("name", sorted(COMMANDS))
def test_command_frontmatter(name: str) -> None:
    meta, body = _frontmatter(PLUGIN / "commands" / f"{name}.md")
    assert isinstance(meta.get("description"), str)
    assert meta["description"]
    assert body.strip()
    human_only = meta.get("disable-model-invocation") is True
    assert human_only == (name in HUMAN_ONLY)


def test_human_only_commands_cannot_change_state_beyond_their_job() -> None:
    approve, _ = _frontmatter(PLUGIN / "commands" / "gpu-approve.md")
    assert approve["allowed-tools"] == "Bash(gpu approve:*), Bash(gpu status:*)"
    statusline, body = _frontmatter(PLUGIN / "commands" / "gpu-statusline.md")
    allowed = statusline["allowed-tools"]
    assert allowed == "Bash(gpu statusline status:*), Bash(gpu statusline preview:*)"
    assert "install" not in allowed
    assert "settings" not in allowed
    assert "Never run `gpu statusline install`" in body


def test_statusline_command_previews_real_sample_states() -> None:
    from gpu_router.statusline import samples

    _, body = _frontmatter(PLUGIN / "commands" / "gpu-statusline.md")
    states = re.findall(r"--state ([a-z-]+)", body)
    assert states
    known = {s.key for s in samples.SAMPLES}
    assert set(states) <= known


def test_skill_frontmatter() -> None:
    meta, body = _frontmatter(PLUGIN / "skills" / "gpu-router" / "SKILL.md")
    assert meta["name"] == "gpu-router"
    assert meta["description"]
    plain = body.replace("`", "").lower()
    assert "never call kaggle, colab or lightning" in plain
    assert "gpu_infer" in plain  # phase 7b: LLM calls go through the inference lane


def test_wrapper_ships_in_the_plugin_and_matches_the_package() -> None:
    from gpu_router.statusline import install

    wrapper = PLUGIN / "statusline" / install.WRAPPER_NAME
    assert os.access(wrapper, os.X_OK)
    assert wrapper.read_text(encoding="utf-8") == install.wrapper_source()
    assert wrapper.read_text(encoding="utf-8").startswith("#!/bin/bash\n")
