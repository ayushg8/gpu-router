"""CLI integration fixtures: one real daemon subprocess (fake providers, fast polling) per
test module, driven through typer's CliRunner in this process.

`cli` (per test) points GPU_ROUTER_HOME at the module daemon, disables auto-start (so a
dead daemon is an error, not a surprise spawn), gives the test its own project dir as the
cwd, and returns a `Cli` helper.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from click.testing import Result
from typer.testing import CliRunner

from gpu_router.cli.app import app
from gpu_router.paths import ENV_HOME
from tests.crash.harness import DaemonProc

FAST = {"duration": 0.3, "steps": 3}


@pytest.fixture(scope="module")
def cli_daemon(tmp_path_factory: pytest.TempPathFactory) -> Iterator[DaemonProc]:
    base = tmp_path_factory.mktemp("cli")
    d = DaemonProc(home=base / "home", project=base / "unused")
    d.prepare()
    client = d.start()
    client.close()
    try:
        yield d
    finally:
        d.kill()


@dataclass
class Cli:
    runner: CliRunner
    project: Path
    home: Path

    def __call__(self, *args: str) -> Result:
        return self.runner.invoke(app, list(args), catch_exceptions=False)

    def json(self, *args: str) -> tuple[Any, Result]:
        res = self(args[0], "--json", *args[1:])  # before any `--`
        lines = [ln for ln in res.stdout.splitlines() if ln.strip()]
        assert lines, f"no stdout (exit {res.exit_code}); stderr: {res.stderr}"
        return json.loads(lines[-1]), res

    def write_yaml(self, text: str) -> None:
        (self.project / "gpu.yaml").write_text(text)

    def fake(self, **directives: Any) -> None:
        """gpu.yaml running train.py on the fake with these directives (FAST defaults)."""
        opts = {**FAST, **directives}
        body = ", ".join(f"{k}: {json.dumps(v)}" for k, v in opts.items())
        self.write_yaml(f"version: 1\nscript: train.py\nprovider_options:\n  fake: {{{body}}}\n")


@pytest.fixture
def cli(
    cli_daemon: DaemonProc,
    gpu_home: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Cli:
    project = tmp_path / "project"
    project.mkdir()
    (project / "train.py").write_text("print('hello from train.py')\n")
    monkeypatch.setenv(ENV_HOME, str(cli_daemon.home))
    monkeypatch.setenv("GPU_ROUTER_NO_AUTOSTART", "1")
    monkeypatch.chdir(project)
    monkeypatch.setattr("gpu_router.cli.app.POLL_S", 0.1)
    c = Cli(runner=CliRunner(), project=project, home=cli_daemon.home)
    c.fake()
    return c
