"""Step 1: tools are detected, missing ones installed with `uv tool install` after one
confirmation; never without it."""

from __future__ import annotations

from typing import Any

from gpu_router.setup import tools
from gpu_router.setup.base import Outcome
from tests.unit.setup.conftest import Sandbox, ScriptedUi, fail, ok


def _outcomes(ctx: Any) -> dict[str, Outcome]:
    return {r.id: r.outcome for r in ctx.results}


def _uv_installs(sandbox: Sandbox, created: dict[str, str]) -> None:
    """`uv tool install X` leaves X's executable in ~/.local/bin (or the lightning env)."""

    def install(argv: list[str], _env: Any) -> Any:
        spec = argv[-1]
        exe = created.get(spec.split("==")[0])
        if exe == "lightning-env":
            py = sandbox.user_home / ".local/share/uv/tools/lightning-sdk/bin/python"
            py.parent.mkdir(parents=True, exist_ok=True)
            py.write_text("")
        elif exe:
            sandbox.install_tool(exe)
        return ok("Installed 1 executable")

    sandbox.run.on(r"uv tool install ", install)


def test_all_present_is_already_done_and_asks_nothing(sandbox: Sandbox) -> None:
    sandbox.install_tool("hf")
    py = sandbox.user_home / ".local/share/uv/tools/lightning-sdk/bin/python"
    py.parent.mkdir(parents=True)
    py.write_text("")
    ctx = sandbox.ctx(ScriptedUi([]))
    tools.run(ctx)
    got = _outcomes(ctx)
    assert set(got.values()) == {Outcome.ALREADY}
    assert set(got) == set(tools.ITEMS)
    assert not sandbox.run.ran(r"tool install")


def test_missing_tools_install_after_one_yes(sandbox: Sandbox) -> None:
    del sandbox.tools["kaggle"]
    del sandbox.tools["colab"]
    _uv_installs(
        sandbox,
        {
            "kaggle": "kaggle",
            "google-colab-cli": "colab",
            "lightning-sdk": "lightning-env",
            "huggingface_hub[cli]": "hf",
        },
    )
    ui = ScriptedUi([True])
    ctx = sandbox.ctx(ui)
    tools.run(ctx)
    assert len(ui.questions) == 1
    assert "install kaggle, google-colab-cli" in ui.questions[0]
    installs = [c[3:] for c in sandbox.run.ran(r"tool install")]
    pin = tools._lightning_pin()
    assert installs == [
        ["kaggle"],
        ["google-colab-cli"],
        [f"lightning-sdk=={pin}"],
        ["huggingface_hub[cli]"],
    ]
    got = _outcomes(ctx)
    for item in ("tools.kaggle", "tools.colab", "tools.lightning", "tools.hf"):
        assert got[item] is Outcome.DONE, ui.text
    assert "$ uv tool install kaggle" in ui.text  # the exact commands were shown first


def test_no_means_nothing_is_installed(sandbox: Sandbox) -> None:
    del sandbox.tools["kaggle"]
    ctx = sandbox.ctx(ScriptedUi([False]), only=["tools.kaggle"])
    tools.run(ctx)
    assert not sandbox.run.ran(r"tool install")
    res = ctx.results[-1]
    assert res.outcome is Outcome.DECLINED
    assert "uv tool install kaggle" in res.summary


def test_no_terminal_and_no_yes_leaves_the_command(sandbox: Sandbox) -> None:
    del sandbox.tools["kaggle"]
    ctx = sandbox.ctx(ScriptedUi([], interactive=False), only=["tools.kaggle"])
    tools.run(ctx)
    assert not sandbox.run.ran(r"tool install")
    assert ctx.results[-1].outcome is Outcome.MANUAL
    assert ctx.results[-1].fix == "uv tool install kaggle"
    assert ctx.unanswered == ["tools.install"]


def test_yes_installs_without_a_terminal(sandbox: Sandbox) -> None:
    del sandbox.tools["kaggle"]
    _uv_installs(sandbox, {"kaggle": "kaggle"})
    ctx = sandbox.ctx(ScriptedUi([], interactive=False), only=["tools.kaggle"], yes=True)
    tools.run(ctx)
    assert ctx.results[-1].outcome is Outcome.DONE


def test_a_failed_install_says_why_and_how(sandbox: Sandbox) -> None:
    del sandbox.tools["kaggle"]
    sandbox.run.on(r"uv tool install kaggle", fail(2, err="error: no network\n"))
    ctx = sandbox.ctx(ScriptedUi([True]), only=["tools.kaggle"])
    tools.run(ctx)
    res = ctx.results[-1]
    assert res.outcome is Outcome.FAILED
    assert "no network" in res.summary
    assert res.fix == "uv tool install kaggle"


def test_without_uv_nothing_can_be_installed(sandbox: Sandbox) -> None:
    del sandbox.tools["uv"]
    del sandbox.tools["kaggle"]
    ctx = sandbox.ctx(ScriptedUi([]), only=["tools"])
    tools.run(ctx)
    got = {r.id: r for r in ctx.results}
    assert got["tools.uv"].outcome is Outcome.FAILED
    assert got["tools.uv"].fix == tools.UV_INSTALL
    assert got["tools.kaggle"].outcome is Outcome.MANUAL


def test_gpu_only_in_local_bin_needs_path(sandbox: Sandbox) -> None:
    del sandbox.tools["gpu"]  # the file stays in ~/.local/bin, PATH does not have it
    ctx = sandbox.ctx(ScriptedUi([]), only=["tools.gpu"])
    tools.run(ctx)
    res = ctx.results[-1]
    assert res.outcome is Outcome.MANUAL
    assert res.fix == "uv tool update-shell"


def test_disabled_provider_tools_are_not_offered(sandbox: Sandbox) -> None:
    (sandbox.paths.config).write_text("version: 1\nproviders:\n  kaggle: {enabled: false}\n")
    del sandbox.tools["kaggle"]
    ctx = sandbox.ctx(ScriptedUi([]), only=["tools.kaggle"])
    tools.run(ctx)
    assert ctx.results == []


def test_dry_run_installs_nothing_and_asks_nothing(sandbox: Sandbox) -> None:
    del sandbox.tools["kaggle"]
    ui = ScriptedUi([])
    ctx = sandbox.ctx(ui, only=["tools.kaggle"], dry_run=True)
    tools.run(ctx)
    assert not sandbox.run.ran(r"tool install")
    assert ctx.results[-1].outcome is Outcome.SKIPPED
    assert not (sandbox.paths.home / "setup.json").exists()
