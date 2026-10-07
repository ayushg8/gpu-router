"""The whole wizard in a sandbox: a first run from nothing, an idempotent re-run, resume
after Ctrl-C, --yes without a terminal, no terminal without --yes, --only, --dry-run."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from gpu_router import __version__, secrets
from gpu_router.errors import InvalidRequest
from gpu_router.setup import check
from gpu_router.setup.base import Options, Outcome
from gpu_router.setup.state import SetupState
from gpu_router.setup.wizard import ALL_ITEMS, expand_only, run_wizard
from tests.shell.conftest import InProcDaemon
from tests.unit.setup.conftest import (
    HF_TOKEN,
    KAGGLE_KEY,
    Sandbox,
    ScriptedUi,
    fail,
    ok,
)

KAGGLE_DOC = json.dumps({"username": "demo-user", "key": KAGGLE_KEY})
USER_LINE = {"type": "command", "command": 'bash "$HOME/.claude/statusline.sh"'}


def _world(sandbox: Sandbox) -> None:
    """A Mac with uv and the kaggle file, nothing else set up."""
    del sandbox.tools["kaggle"]
    del sandbox.tools["colab"]
    sandbox.tools["claude"] = "/fake/bin/claude"
    sandbox.tools["codex"] = "/fake/bin/codex"
    sandbox.tools["gcloud"] = "/fake/bin/gcloud"
    sandbox.write("Downloads/kaggle.json", KAGGLE_DOC)
    sandbox.write(".claude/settings.json", json.dumps({"statusLine": USER_LINE}) + "\n", 0o644)
    sandbox.write(".claude/skills/colab/SKILL.md", "x")
    sandbox.write(".codex/config.toml", 'model = "gpt-5"\n')
    installed = {"kaggle": "kaggle", "google-colab-cli": "colab", "huggingface_hub[cli]": "hf"}

    def uv_install(argv: list[str], _env: Any) -> Any:
        spec = argv[-1].split("==")[0]
        if spec == "lightning-sdk":
            py = sandbox.user_home / ".local/share/uv/tools/lightning-sdk/bin/python"
            py.parent.mkdir(parents=True, exist_ok=True)
            py.write_text("")
        elif spec in installed:
            sandbox.install_tool(installed[spec])
        return ok()

    sandbox.run.on(r"uv tool install ", uv_install)
    loaded = {"yes": False}

    def bootstrap(_argv: list[str], _env: Any) -> Any:
        loaded["yes"] = True
        return ok()

    sandbox.run.on(r"^/bin/launchctl bootstrap ", bootstrap)
    sandbox.run.on(r"^/bin/launchctl bootout ", fail(3))
    sandbox.run.on(
        r"^/bin/launchctl print ",
        lambda _a, _e: ok("\tstate = running\n\tpid = 99\n") if loaded["yes"] else fail(113),
    )
    plugins = sandbox.user_home / ".claude" / "plugins"

    def add(_a: list[str], _e: Any) -> Any:
        plugins.mkdir(parents=True, exist_ok=True)
        (plugins / "known_marketplaces.json").write_text(json.dumps({"gpu-router-local": {}}))
        return ok()

    def install(_a: list[str], _e: Any) -> Any:
        doc = {"plugins": {"gpu-router@gpu-router-local": [{"version": __version__}]}}
        (plugins / "installed_plugins.json").write_text(json.dumps(doc))
        return ok()

    sandbox.run.on(r"claude plugin marketplace add ", add)
    sandbox.run.on(r"claude plugin install ", install)
    sandbox.on_attached = lambda _argv: (
        sandbox.write(".config/gcloud/application_default_credentials.json", "{}") and 0
    )


def _files(root: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and "setup.json" not in p.name:
            out[str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


@pytest.fixture
def daemon(sandbox: Sandbox, gpu_home: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    d = InProcDaemon(home=gpu_home)
    d.start()
    sandbox.connect = d.client
    real = check.smoke_spec

    def quick(project: Path, view: Any) -> Any:
        s = real(project, view)
        return s.model_copy(update={"provider_options": {view.name: {"duration": 0.2}}})

    monkeypatch.setattr(check, "smoke_spec", quick)
    with d.client() as c:
        for name in ("fake", "fake-b"):
            c.healthcheck(name)
    try:
        yield d
    finally:
        d.stop()


# answers for a full first run, in the order the wizard asks
FIRST_RUN = [
    True,  # install kaggle, google-colab-cli, lightning-sdk, huggingface_hub[cli]?
    True,  # import ~/Downloads/kaggle.json?
    True,  # run the gcloud sign-in now?
    "b",  # lightning: browser sign-in
    True,  # lightning.ai signed you in as ...; store it? (review fix: confirm the account)
    True,  # paste a Hugging Face token now?
    HF_TOKEN,
    False,  # remote HF token? no
    True,  # launchd
    True,  # status line
    True,  # plugin
    True,  # codex
    True,  # colab skill
    True,  # smoke test
]


def test_first_run_from_nothing_then_an_idempotent_rerun(sandbox: Sandbox, daemon: Any) -> None:
    _world(sandbox)
    ui = ScriptedUi(list(FIRST_RUN))
    result = run_wizard(sandbox.env, ui, Options())
    got = {r.id: r.outcome for r in result.results}
    assert not ui.answers, "every scripted answer was used"
    done = {i for i, o in got.items() if o is Outcome.DONE}
    assert {
        "tools.kaggle",
        "tools.colab",
        "tools.lightning",
        "tools.hf",
        "login.kaggle",
        "login.colab",
        "login.lightning",
        "login.hf",
        "launchd",
        "integration.statusline",
        "integration.plugin",
        "integration.codex",
        "integration.colab_skill",
        "check.smoke",
        "check.summary",
    } <= done, ui.text
    assert got["login.hf_remote"] is Outcome.DECLINED
    assert result.code in (0, 1)  # doctor may flag the sandbox (fake daemon), never a crash
    summary = next(r for r in result.results if r.id == "check.summary")
    assert summary.summary.startswith("2 providers ready")  # the in-process daemon's fakes
    assert secrets.get_secret("kaggle")
    assert secrets.get_secret("HF_TOKEN") == HF_TOKEN
    state = SetupState.load(sandbox.paths.home)
    assert state.completed_at is not None
    assert not state.interrupted
    assert HF_TOKEN not in (sandbox.paths.home / "setup.json").read_text()
    assert HF_TOKEN not in ui.text
    assert KAGGLE_KEY not in ui.text

    before = _files(sandbox.user_home)
    calls = len(sandbox.run.calls)
    again_ui = ScriptedUi([False])  # a re-run only offers what was declined (the remote token)
    again = run_wizard(sandbox.env, again_ui, Options())
    assert again_ui.questions == ["paste the fine-grained remote token now?"]
    outcomes = {r.id: r.outcome for r in again.results}
    changed = {
        i: o
        for i, o in outcomes.items()
        if o not in (Outcome.ALREADY, Outcome.SKIPPED, Outcome.DECLINED)
        and i not in ("check.doctor", "check.summary")
    }
    assert changed == {}, again_ui.text
    assert _files(sandbox.user_home) == before
    new_calls = sandbox.run.calls[calls:]
    assert not [c for c in new_calls if "install" in c or "bootstrap" in c]


def test_ctrl_c_then_resume_keeps_earlier_answers(sandbox: Sandbox) -> None:
    _world(sandbox)

    class StopAtLaunchd(ScriptedUi):
        def ask(self, question: str, default: bool) -> bool:
            if "daemon at login" in question:
                raise KeyboardInterrupt
            return super().ask(question, default)

    first = StopAtLaunchd([True, False, False, "s", False])  # tools yes; kaggle/colab/.. no
    with pytest.raises(KeyboardInterrupt):
        run_wizard(sandbox.env, first, Options())
    state = SetupState.load(sandbox.paths.home)
    assert state.interrupted
    assert state.answer("login.kaggle") == "no"

    ui = ScriptedUi([False, False, False, False, False, False])  # launchd .. smoke: no
    result = run_wizard(sandbox.env, ui, Options())
    assert "continuing the setup run" in ui.text
    assert not any("kaggle" in q for q in ui.questions)  # the earlier no stands
    assert "start the gpu-router daemon at login?" in ui.questions[0]
    got = {r.id: r.outcome for r in result.results}
    assert got["tools.kaggle"] is Outcome.ALREADY  # installed in the first, interrupted run
    assert got["login.kaggle"] is Outcome.DECLINED
    assert SetupState.load(sandbox.paths.home).completed_at is not None


def test_no_terminal_without_yes_changes_nothing_and_exits_2(sandbox: Sandbox) -> None:
    _world(sandbox)
    before = _files(sandbox.user_home)
    ui = ScriptedUi([], interactive=False)
    result = run_wizard(sandbox.env, ui, Options())
    assert result.code == 2
    assert "tools.install" in result.unanswered
    assert "launchd" in result.unanswered
    assert _files(sandbox.user_home) == before
    assert not [c for c in sandbox.run.calls if "install" in c or "bootstrap" in c]
    assert secrets.secret_names() == []
    assert "need a terminal" in ui.text


def test_yes_without_terminal_does_everything_a_machine_can(sandbox: Sandbox) -> None:
    _world(sandbox)
    result = run_wizard(sandbox.env, ScriptedUi([], interactive=False), Options(yes=True))
    got = {r.id: r for r in result.results}
    for item in ("tools.kaggle", "login.kaggle", "launchd", "integration.codex"):
        assert got[item].outcome is Outcome.DONE, got[item]
    for item in ("login.colab", "login.lightning", "login.hf"):  # need a human
        assert got[item].outcome is Outcome.MANUAL, got[item]
    assert sandbox.attached == []  # no browser sign-in without a terminal
    assert result.unanswered == []


def test_an_agents_yes_never_changes_the_users_own_setup(sandbox: Sandbox) -> None:
    """Review fix: an agent told "run `gpu setup`" got "add --yes" and rewrote the status
    line, installed the plugin, edited ~/.codex/config.toml, moved the colab skill and
    registered launchd with nobody seeing a diff."""
    _world(sandbox)
    sandbox.environ["CLAUDECODE"] = "1"
    before = {
        name: (sandbox.user_home / name).read_bytes()
        for name in (".claude/settings.json", ".codex/config.toml")
    }
    ui = ScriptedUi([], interactive=False)
    result = run_wizard(sandbox.env, ui, Options(yes=True))
    got = {r.id: r for r in result.results}
    for item in ("integration.statusline", "integration.codex", "integration.colab_skill"):
        assert got[item].outcome is Outcome.MANUAL, got[item]
        assert "an agent ran setup (CLAUDECODE=1)" in got[item].summary
        assert got[item].fix
    assert got["launchd"].outcome is Outcome.MANUAL
    assert got["tools.kaggle"].outcome is Outcome.DONE  # machine steps still run
    for name, data in before.items():
        assert (sandbox.user_home / name).read_bytes() == data
    assert (sandbox.user_home / ".claude/skills/colab/SKILL.md").exists()
    assert not sandbox.run.ran(r"launchctl bootstrap")
    assert not sandbox.run.ran(r"claude plugin install")
    assert "left for you" in ui.text


def test_an_agent_is_not_told_to_add_yes(sandbox: Sandbox) -> None:
    _world(sandbox)
    sandbox.environ["CLAUDECODE"] = "1"
    ui = ScriptedUi([], interactive=False)
    result = run_wizard(sandbox.env, ui, Options())
    assert result.code == 2
    assert "--yes" not in ui.text
    assert "ask the user to run `gpu setup`" in ui.text


def test_dry_run_changes_nothing(sandbox: Sandbox) -> None:
    _world(sandbox)
    before = _files(sandbox.user_home)
    ui = ScriptedUi([])
    result = run_wizard(sandbox.env, ui, Options(dry_run=True))
    assert _files(sandbox.user_home) == before
    assert not (sandbox.paths.home / "setup.json").exists()
    assert secrets.secret_names() == []
    assert not [c for c in sandbox.run.calls if "install" in c or "bootstrap" in c]
    assert "dry run: nothing changes" in ui.text
    assert any(r.summary.startswith("dry run: would") for r in result.results)


def test_only_runs_just_that_item(sandbox: Sandbox) -> None:
    _world(sandbox)
    ui = ScriptedUi([True])
    result = run_wizard(sandbox.env, ui, Options(only=expand_only(["codex"])))
    assert [r.id for r in result.results] == ["integration.codex"]
    assert result.results[0].outcome is Outcome.DONE
    assert SetupState.load(sandbox.paths.home).completed_at is None  # a subset never completes


def test_expand_only() -> None:
    assert expand_only(["plugin"]) == {"integration.plugin"}
    assert expand_only(["login"]) == {i for i in ALL_ITEMS if i.startswith("login.")}
    assert expand_only(["tools,launchd"]) >= {"tools.kaggle", "launchd"}
    assert expand_only(["colab-skill"]) == {"integration.colab_skill"}
    with pytest.raises(InvalidRequest, match="ambiguous"):
        expand_only(["kaggle"])
    with pytest.raises(InvalidRequest, match="no such setup step"):
        expand_only(["bogus"])
