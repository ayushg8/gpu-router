"""MCP test fixtures (phase 6): a real daemon IN this process on the fake providers (the
shell suite's `InProcDaemon`: uvicorn on its own thread, port 0, tmp home), a project whose
gpu.yaml carries fake directives, and a FastMCP in-memory client over `build_server()`.

Auto-start is off for every MCP test (GPU_ROUTER_NO_AUTOSTART=1): a tool that finds no
daemon must say so, never spawn one.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import pytest

from tests.shell.conftest import InProcDaemon


@pytest.fixture
def daemon(gpu_home: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[InProcDaemon]:
    monkeypatch.setenv("GPU_ROUTER_NO_AUTOSTART", "1")
    d = InProcDaemon(home=gpu_home)
    d.start()
    try:
        yield d
    finally:
        d.stop()


@pytest.fixture
def no_daemon(gpu_home: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("GPU_ROUTER_NO_AUTOSTART", "1")
    gpu_home.mkdir(parents=True, exist_ok=True)
    return gpu_home


def write_gpu_yaml(project: Path, **fake: Any) -> None:
    body = ", ".join(f"{k}: {json.dumps(v)}" for k, v in fake.items())
    (project / "gpu.yaml").write_text(
        f"version: 1\nscript: train.py\nprovider_options:\n  fake: {{{body}}}\n"
    )


@pytest.fixture
def project(tmp_path: Path) -> Path:
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "train.py").write_text("print('hello from train.py')\n")
    write_gpu_yaml(proj, duration=1.0, steps=40)
    return proj


@pytest.fixture
async def mcp() -> AsyncIterator[Any]:
    from fastmcp import Client

    from gpu_router.mcp.server import build_server

    async with Client(build_server()) as client:
        yield client


async def call(client: Any, tool: str, **args: Any) -> dict[str, Any]:
    """Call a tool that must succeed; return its structured JSON."""
    result = await client.call_tool(tool, args, raise_on_error=False)
    text = result.content[0].text if result.content else ""
    assert not result.is_error, f"{tool} failed: {text}"
    data = result.structured_content
    assert isinstance(data, dict)
    assert json.loads(text) == data  # the text an agent reads is the same JSON
    return data


async def call_error(client: Any, tool: str, **args: Any) -> dict[str, Any]:
    """Call a tool that must fail; return the error envelope's `error` object."""
    result = await client.call_tool(tool, args, raise_on_error=False)
    assert result.is_error, f"{tool} should have failed: {result.structured_content}"
    body = json.loads(result.content[0].text)
    assert set(body) == {"error"}
    return dict(body["error"])
