"""Step 3: the launchd agent, with launchctl mocked (SetupEnv.run) and the plist in the
sandboxed ~/Library/LaunchAgents."""

from __future__ import annotations

import os
import plistlib
from typing import Any

from gpu_router.setup import service
from gpu_router.setup.base import Outcome
from tests.unit.setup.conftest import Sandbox, ScriptedUi, fail, ok


def _loaded_after_bootstrap(sandbox: Sandbox) -> None:
    """launchctl print answers "running" once bootstrap ran."""
    state: dict[str, bool] = {"loaded": False}

    def bootstrap(_argv: list[str], _env: Any) -> Any:
        state["loaded"] = True
        return ok()

    def show(_argv: list[str], _env: Any) -> Any:
        if state["loaded"]:
            return ok("\tstate = running\n\tpid = 777\n")
        return fail(113, err="Could not find service")

    sandbox.run.on(r"^/bin/launchctl bootstrap ", bootstrap)
    sandbox.run.on(r"^/bin/launchctl bootout ", fail(3, err="No such process"))
    sandbox.run.on(r"^/bin/launchctl print ", show)


def test_installs_the_agent_after_yes(sandbox: Sandbox) -> None:
    _loaded_after_bootstrap(sandbox)
    ui = ScriptedUi([True])
    ctx = sandbox.ctx(ui)
    service.run(ctx)
    res = ctx.results[-1]
    assert res.outcome is Outcome.DONE, ui.text
    plist = sandbox.user_home / "Library/LaunchAgents/dev.gpu-router.daemon.plist"
    agent = plistlib.loads(plist.read_bytes())
    assert agent["ProgramArguments"][1:] == ["daemon", "run", "--launchd"]
    assert agent["ProgramArguments"][0] == str(sandbox.gpu_bin.resolve())  # type: ignore[union-attr]
    assert "EnvironmentVariables" not in agent  # default data dir: nothing to pass
    boots = sandbox.run.ran(r"^/bin/launchctl bootstrap ")
    assert boots == [["/bin/launchctl", "bootstrap", f"gui/{os.getuid()}", str(plist)]]
    assert "undo: gpu daemon uninstall" in ui.text
    assert "start the gpu-router daemon at login?" in ui.questions


def test_declined_writes_nothing(sandbox: Sandbox) -> None:
    ctx = sandbox.ctx(ScriptedUi([False]))
    service.run(ctx)
    assert ctx.results[-1].outcome is Outcome.DECLINED
    assert not (sandbox.user_home / "Library/LaunchAgents").exists()
    assert not sandbox.run.ran(r"launchctl bootstrap")


def test_already_loaded_agent_is_left_alone(sandbox: Sandbox) -> None:
    _loaded_after_bootstrap(sandbox)
    service.run(sandbox.ctx(ScriptedUi([True])))
    calls = len(sandbox.run.ran(r"launchctl bootstrap"))
    ctx = sandbox.ctx(ScriptedUi([]))
    service.run(ctx)
    assert ctx.results[-1].outcome is Outcome.ALREADY
    assert len(sandbox.run.ran(r"launchctl bootstrap")) == calls


def test_bootstrap_failure_is_reported(sandbox: Sandbox) -> None:
    sandbox.run.on(r"^/bin/launchctl bootstrap ", fail(5, err="Bootstrap failed: 5: I/O error"))
    ctx = sandbox.ctx(ScriptedUi([True]))
    service.run(ctx)
    res = ctx.results[-1]
    assert res.outcome is Outcome.FAILED
    assert "I/O error" in res.summary
    assert res.fix == "gpu daemon install-launchd"


def test_custom_data_dir_does_not_take_the_agent(sandbox: Sandbox, monkeypatch: Any) -> None:
    monkeypatch.setattr("gpu_router.paths.DEFAULT_HOME", sandbox.user_home / "elsewhere")
    sandbox.environ["GPU_ROUTER_HOME"] = str(sandbox.paths.home)
    ctx = sandbox.ctx(ScriptedUi([]))
    service.run(ctx)
    assert ctx.results[-1].outcome is Outcome.SKIPPED
    assert not sandbox.run.ran(r"launchctl bootstrap")


def test_no_terminal_no_yes(sandbox: Sandbox) -> None:
    ctx = sandbox.ctx(ScriptedUi([], interactive=False))
    service.run(ctx)
    assert ctx.results[-1].outcome is Outcome.MANUAL
    assert ctx.results[-1].fix == "gpu daemon install-launchd"


def test_an_existing_plist_is_diffed_and_backed_up_before_it_is_replaced(
    sandbox: Sandbox,
) -> None:
    """Review fix: the step replaced the plist without showing or keeping it (a hand-added
    PATH was lost), and printed `launchctl bootstrap ... '~/Library/...'` (a quoted ~ does
    not expand)."""
    _loaded_after_bootstrap(sandbox)
    plist = sandbox.user_home / "Library/LaunchAgents/dev.gpu-router.daemon.plist"
    plist.parent.mkdir(parents=True)
    old = plistlib.dumps(
        {
            "Label": "dev.gpu-router.daemon",
            "ProgramArguments": ["/old/gpu", "daemon", "run", "--launchd"],
            "EnvironmentVariables": {"PATH": "/opt/homebrew/bin:/usr/bin:/bin"},
        }
    )
    plist.write_bytes(old)
    ui = ScriptedUi([True])
    ctx = sandbox.ctx(ui)
    service.run(ctx)
    assert ctx.results[-1].outcome is Outcome.DONE, ui.text
    assert "it replaces the existing ~/Library/LaunchAgents/dev.gpu-router.daemon.plist" in ui.text
    assert "-\t\t<string>/opt/homebrew/bin:/usr/bin:/bin</string>" in ui.text
    backups = list((sandbox.paths.home / "backups").glob("dev.gpu-router.daemon.plist.*.bak"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == old
    assert oct(backups[0].stat().st_mode & 0o777) == "0o600"
    assert "backup: " in ui.text
    assert f"launchctl bootstrap gui/{os.getuid()} {plist}" in ui.text
    assert "'~/" not in ui.text
    assert not list(plist.parent.glob("*.bak"))  # never next to the agent (launchd reads it)
