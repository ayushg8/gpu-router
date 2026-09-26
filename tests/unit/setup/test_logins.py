"""Step 2: logins. Credentials go to the (in-memory) Keychain only, are checked first when
possible, never printed; colab's ADC is only checked, with the exact gcloud command."""

from __future__ import annotations

import json
from typing import Any

import pytest

from gpu_router import secrets
from gpu_router.errors import InvalidRequest
from gpu_router.providers.lightning.sdk import CallResult
from gpu_router.setup import logins
from gpu_router.setup.base import Outcome
from tests.unit.setup.conftest import (
    HF_TOKEN,
    KAGGLE_KEY,
    LIGHTNING_KEY,
    LIGHTNING_USER,
    WHOAMI_NO_SCOPE,
    WHOAMI_OK,
    Sandbox,
    ScriptedUi,
    fail,
    ok,
)

KAGGLE_DOC = json.dumps({"username": "demo-user", "key": KAGGLE_KEY})


def _last(ctx: Any) -> Any:
    return ctx.results[-1]


def _no_secret_in(text: str, *values: str) -> None:
    for v in values:
        assert v not in text


# =========================================================================== kaggle


def test_parse_kaggle_json_and_tokens() -> None:
    c = logins.parse_kaggle(KAGGLE_DOC)
    assert (c.kind, c.username, c.value) == ("json", "demo-user", KAGGLE_KEY)
    assert KAGGLE_KEY not in repr(c)
    t = logins.parse_kaggle("KGAT_" + "x" * 30 + "\n")
    assert t.kind == "token"
    assert t.username is None
    for bad in ("", "{broken", '{"username": "a"}', "short", "has space in it and more"):
        with pytest.raises(InvalidRequest) as info:
            logins.parse_kaggle(bad)
        assert "nothing stored" in info.value.message or "empty" in info.value.message


def test_kaggle_json_is_checked_then_imported(sandbox: Sandbox) -> None:
    sandbox.write(".kaggle/kaggle.json", KAGGLE_DOC)
    ui = ScriptedUi([True])
    ctx = sandbox.ctx(ui, only=["login.kaggle"])
    logins.run(ctx)
    assert _last(ctx).outcome is Outcome.DONE
    assert json.loads(secrets.get_secret("kaggle") or "{}") == {
        "username": "demo-user",
        "key": KAGGLE_KEY,
    }
    (call,) = sandbox.run.ran(r"kaggle -W quota")
    env = sandbox.run.envs[sandbox.run.calls.index(call)] or {}
    assert env["KAGGLE_USERNAME"] == "demo-user"
    assert env["KAGGLE_KEY"] == KAGGLE_KEY
    assert env["KAGGLE_CONFIG_DIR"] != str(sandbox.user_home / ".kaggle")  # isolated
    assert (sandbox.user_home / ".kaggle/kaggle.json").exists()  # the file stays
    _no_secret_in(ui.text, KAGGLE_KEY)


def test_kaggle_rejected_credentials_are_not_stored(sandbox: Sandbox) -> None:
    sandbox.write("Downloads/kaggle.json", KAGGLE_DOC)
    sandbox.run.on(r"kaggle -W quota", fail(1, err="401 Client Error: Unauthorized"))
    ctx = sandbox.ctx(ScriptedUi([True]), only=["login.kaggle"])
    logins.run(ctx)
    assert _last(ctx).outcome is Outcome.FAILED
    assert "rejected" in _last(ctx).summary
    assert secrets.get_secret("kaggle") is None


def test_kaggle_unreachable_stores_with_a_note(sandbox: Sandbox) -> None:
    sandbox.write(".kaggle/kaggle.json", KAGGLE_DOC)
    sandbox.run.on(r"kaggle -W quota", fail(1, err="ConnectionError: name resolution"))
    ui = ScriptedUi([True])
    ctx = sandbox.ctx(ui, only=["login.kaggle"])
    logins.run(ctx)
    assert _last(ctx).outcome is Outcome.DONE
    assert "stored anyway" in ui.text
    assert secrets.get_secret("kaggle") is not None


def test_kaggle_already_in_keychain_is_skipped(sandbox: Sandbox) -> None:
    secrets.set_secret("kaggle", KAGGLE_DOC)
    sandbox.write(".kaggle/kaggle.json", KAGGLE_DOC)
    ctx = sandbox.ctx(ScriptedUi([]), only=["login.kaggle"])
    logins.run(ctx)
    assert _last(ctx).outcome is Outcome.ALREADY
    assert not sandbox.run.ran("kaggle -W")


def test_kaggle_declined_import_keeps_the_cli_file(sandbox: Sandbox) -> None:
    sandbox.write(".kaggle/kaggle.json", KAGGLE_DOC)
    ctx = sandbox.ctx(ScriptedUi([False]), only=["login.kaggle"])
    logins.run(ctx)
    res = _last(ctx)
    assert res.outcome is Outcome.DECLINED
    assert "keeps using it" in res.summary
    assert secrets.get_secret("kaggle") is None


def test_kaggle_paste_when_no_file(sandbox: Sandbox) -> None:
    token = "KGAT_" + "z" * 32
    ui = ScriptedUi([True, token])
    ctx = sandbox.ctx(ui, only=["login.kaggle"])
    logins.run(ctx)
    assert _last(ctx).outcome is Outcome.DONE
    assert secrets.get_secret("KAGGLE_API_TOKEN") == token
    _no_secret_in(ui.text, token)


def test_kaggle_without_terminal_names_the_command(sandbox: Sandbox) -> None:
    ctx = sandbox.ctx(ScriptedUi([], interactive=False), only=["login.kaggle"], yes=True)
    logins.run(ctx)
    assert _last(ctx).outcome is Outcome.MANUAL
    assert _last(ctx).fix == "gpu login kaggle"


@pytest.mark.parametrize(
    ("item", "fix"),
    [
        ("login.kaggle", "gpu login kaggle"),
        ("login.lightning", "gpu login lightning --browser"),
        ("login.hf", "gpu login hf"),
    ],
)
def test_paste_only_logins_without_a_terminal_are_unanswered(
    sandbox: Sandbox, item: str, fix: str
) -> None:
    """Review fix: these logins ended `manual` without adding to ctx.unanswered, so a
    provisioning script's `gpu setup --only logins` with no tty exited 0."""
    ctx = sandbox.ctx(ScriptedUi([], interactive=False), only=[item])
    logins.run(ctx)
    assert _last(ctx).outcome is Outcome.MANUAL
    assert _last(ctx).fix == fix
    assert ctx.unanswered == [item]
    yes = sandbox.ctx(ScriptedUi([], interactive=False), only=[item], yes=True)
    logins.run(yes)
    assert yes.unanswered == []  # --yes: a human-only step is `manual`, not a question


def test_the_optional_remote_token_without_a_terminal_is_unanswered(sandbox: Sandbox) -> None:
    ctx = sandbox.ctx(ScriptedUi([], interactive=False), only=["login.hf_remote"])
    logins.run(ctx)
    assert _last(ctx).outcome is Outcome.MANUAL
    assert ctx.unanswered == ["login.hf_remote"]


# =========================================================================== colab


def _adc(sandbox: Sandbox) -> None:
    sandbox.write(".config/gcloud/application_default_credentials.json", "{}")


def test_colab_ok_when_adc_has_the_scope(sandbox: Sandbox) -> None:
    _adc(sandbox)
    ctx = sandbox.ctx(ScriptedUi([]), only=["login.colab"])
    logins.run(ctx)
    assert _last(ctx).outcome is Outcome.ALREADY
    assert "colaboratory scope" in _last(ctx).summary


def test_colab_missing_adc_runs_the_exact_gcloud_command(sandbox: Sandbox) -> None:
    sandbox.tools["gcloud"] = "/fake/bin/gcloud"

    def sign_in(argv: list[str]) -> int:
        _adc(sandbox)
        return 0

    sandbox.on_attached = sign_in
    ui = ScriptedUi([True])
    ctx = sandbox.ctx(ui, only=["login.colab"])
    logins.run(ctx)
    assert _last(ctx).outcome is Outcome.DONE
    assert f"$ {logins.ADC_LOGIN}" in ui.text
    assert sandbox.attached == [["/fake/bin/gcloud", *logins.ADC_LOGIN.split()[1:]]]
    assert "--scopes=openid," in " ".join(sandbox.attached[0])


def test_colab_without_gcloud_is_manual(sandbox: Sandbox) -> None:
    ctx = sandbox.ctx(ScriptedUi([]), only=["login.colab"])
    logins.run(ctx)
    res = _last(ctx)
    assert res.outcome is Outcome.MANUAL
    assert res.fix is not None
    assert logins.ADC_LOGIN in res.fix
    assert "brew" in res.fix


def test_colab_missing_scope_offers_the_login_again(sandbox: Sandbox) -> None:
    _adc(sandbox)
    sandbox.tools["gcloud"] = "/fake/bin/gcloud"
    sandbox.run.on(r"colab --auth=adc --config \S+ whoami$", ok(WHOAMI_NO_SCOPE))
    ctx = sandbox.ctx(ScriptedUi([False]), only=["login.colab"])
    logins.run(ctx)
    res = _last(ctx)
    assert res.outcome is Outcome.DECLINED
    assert logins.ADC_LOGIN in res.summary
    assert sandbox.attached == []


def test_colab_gcloud_sign_in_never_runs_without_a_human(sandbox: Sandbox) -> None:
    sandbox.tools["gcloud"] = "/fake/bin/gcloud"
    ctx = sandbox.ctx(ScriptedUi([], interactive=False), only=["login.colab"], yes=True)
    logins.run(ctx)
    assert sandbox.attached == []
    assert _last(ctx).outcome is Outcome.MANUAL


def test_colab_still_failing_after_gcloud_says_so(sandbox: Sandbox) -> None:
    sandbox.tools["gcloud"] = "/fake/bin/gcloud"
    sandbox.on_attached = lambda _argv: 1
    ctx = sandbox.ctx(ScriptedUi([True]), only=["login.colab"])
    logins.run(ctx)
    assert _last(ctx).outcome is Outcome.FAILED
    assert WHOAMI_OK  # (keeps the import used for the scope case above)


# =========================================================================== lightning


def test_lightning_file_is_imported_into_the_keychain(sandbox: Sandbox) -> None:
    sandbox.write(
        ".lightning/credentials.json",
        json.dumps({"user_id": LIGHTNING_USER, "api_key": LIGHTNING_KEY}),
    )
    ui = ScriptedUi([True])
    ctx = sandbox.ctx(ui, only=["login.lightning"])
    logins.run(ctx)
    assert _last(ctx).outcome is Outcome.DONE, ui.text
    assert secrets.get_secret("LIGHTNING_USER_ID") == LIGHTNING_USER
    assert secrets.get_secret("LIGHTNING_API_KEY") == LIGHTNING_KEY
    assert "teamspace demo/default" in _last(ctx).summary
    _no_secret_in(ui.text, LIGHTNING_USER, LIGHTNING_KEY)


def test_lightning_keys_only_in_the_environment_are_offered_for_the_keychain(
    sandbox: Sandbox,
) -> None:
    """Review fix: env-only keys counted as ALREADY, then the launchd daemon (no such
    environment) had no Lightning credentials after the next login."""
    sandbox.environ.update(
        {"LIGHTNING_USER_ID": LIGHTNING_USER, "LIGHTNING_API_KEY": LIGHTNING_KEY}
    )
    ui = ScriptedUi([True])
    ctx = sandbox.ctx(ui, only=["login.lightning"])
    logins.run(ctx)
    assert _last(ctx).outcome is Outcome.DONE, ui.text
    assert "only in this shell's environment" in ui.text
    assert secrets.get_secret("LIGHTNING_USER_ID") == LIGHTNING_USER
    assert secrets.get_secret("LIGHTNING_API_KEY") == LIGHTNING_KEY
    _no_secret_in(ui.text, LIGHTNING_USER, LIGHTNING_KEY)


def test_lightning_already_in_keychain(sandbox: Sandbox) -> None:
    secrets.set_secret("LIGHTNING_USER_ID", LIGHTNING_USER)
    secrets.set_secret("LIGHTNING_API_KEY", LIGHTNING_KEY)
    ctx = sandbox.ctx(ScriptedUi([]), only=["login.lightning"])
    logins.run(ctx)
    assert _last(ctx).outcome is Outcome.ALREADY


def test_lightning_file_mode_is_left_to_the_user(sandbox: Sandbox) -> None:
    sandbox.paths.config.write_text("version: 1\nproviders:\n  lightning: {login_source: file}\n")
    ctx = sandbox.ctx(ScriptedUi([]), only=["login.lightning"])
    logins.run(ctx)
    res = _last(ctx)
    assert res.outcome is Outcome.MANUAL
    assert res.fix == "lightning login"
    assert secrets.get_secret("LIGHTNING_API_KEY") is None


def test_lightning_browser_sign_in(sandbox: Sandbox) -> None:
    ui = ScriptedUi(["b", True])
    ctx = sandbox.ctx(ui, only=["login.lightning"])
    logins.run(ctx)
    assert _last(ctx).outcome is Outcome.DONE
    assert secrets.get_secret("LIGHTNING_API_KEY") == LIGHTNING_KEY
    # review fix: the verified account is shown and confirmed before it is stored
    assert "lightning.ai signed you in as demo, teamspace demo/default; store it?" in (ui.questions)


def test_lightning_browser_sign_in_for_someone_else_is_not_stored(sandbox: Sandbox) -> None:
    ui = ScriptedUi(["b", False])
    ctx = sandbox.ctx(ui, only=["login.lightning"])
    logins.run(ctx)
    assert _last(ctx).outcome is not Outcome.DONE
    assert secrets.get_secret("LIGHTNING_API_KEY") is None


def test_lightning_rejected_keys_are_not_stored(sandbox: Sandbox) -> None:
    sandbox.lightning = CallResult(ok=False, result=None, kind="auth", error="401 unauthorized")
    ctx = sandbox.ctx(ScriptedUi(["p", LIGHTNING_USER, LIGHTNING_KEY]), only=["login.lightning"])
    logins.run(ctx)
    assert _last(ctx).outcome is Outcome.FAILED
    assert secrets.get_secret("LIGHTNING_API_KEY") is None


def test_lightning_skip_is_remembered_for_the_resumed_run(sandbox: Sandbox) -> None:
    ctx = sandbox.ctx(ScriptedUi(["s"]))
    ctx.state.begin(1.0)
    logins._login_lightning(ctx)
    assert _last(ctx).outcome is Outcome.DECLINED
    resumed = sandbox.ctx(ScriptedUi([]))
    resumed.state.begin(2.0)
    assert resumed.state.resumed
    logins._login_lightning(resumed)  # no question: the earlier skip stands
    assert _last(resumed).outcome is Outcome.DECLINED


# =========================================================================== hugging face


def test_hf_token_from_the_cli_file(sandbox: Sandbox) -> None:
    sandbox.write(".cache/huggingface/token", HF_TOKEN)
    ui = ScriptedUi([True])
    ctx = sandbox.ctx(ui, only=["login.hf"])
    logins.run(ctx)
    assert _last(ctx).outcome is Outcome.DONE
    assert "for demo" in _last(ctx).summary
    assert secrets.get_secret("HF_TOKEN") == HF_TOKEN
    cached = json.loads((sandbox.paths.home / "storage" / "hf.json").read_text())
    assert cached["namespace"] == "demo"
    _no_secret_in(ui.text, HF_TOKEN)


def test_a_read_only_hf_token_is_stored_with_a_note(sandbox: Sandbox) -> None:
    """Review fix: store_hf stored a read-only token without a word, although checkpoint
    storage needs write access."""
    sandbox.hf_role = "read"
    user, notes = logins.store_hf(sandbox.env, HF_TOKEN, remote=False, check=True)
    assert user == "demo"
    assert any("read-only" in n for n in notes)
    assert secrets.get_secret("HF_TOKEN") == HF_TOKEN


def test_hf_rejected_token_is_not_stored(sandbox: Sandbox) -> None:
    sandbox.whoami = (None, "hugging face rejected this token")
    ctx = sandbox.ctx(ScriptedUi([True, HF_TOKEN]), only=["login.hf"])
    logins.run(ctx)
    assert _last(ctx).outcome is Outcome.FAILED
    assert secrets.get_secret("HF_TOKEN") is None


def test_hf_backend_off_needs_no_token(sandbox: Sandbox) -> None:
    sandbox.paths.config.write_text("version: 1\ncheckpoint: {backend: local}\n")
    ctx = sandbox.ctx(ScriptedUi([]), only=["logins"])
    logins._login_hf(ctx)
    assert _last(ctx).outcome is Outcome.SKIPPED


def test_hf_remote_is_optional_and_default_no(sandbox: Sandbox) -> None:
    secrets.set_secret("HF_TOKEN", HF_TOKEN)
    ui = ScriptedUi([None])  # enter = the default (no)
    ctx = sandbox.ctx(ui, only=["login.hf", "login.hf_remote"])
    logins.run(ctx)
    got = {r.id: r.outcome for r in ctx.results}
    assert got == {"login.hf": Outcome.ALREADY, "login.hf_remote": Outcome.DECLINED}
    assert "[y/N]" not in ui.text  # the ScriptedUi gets the default, the terminal shows it


# =========================================================================== gpu login


def test_login_rows_use_local_facts_only(sandbox: Sandbox) -> None:
    sandbox.write(".kaggle/kaggle.json", KAGGLE_DOC)
    secrets.set_secret("HF_TOKEN", HF_TOKEN)
    rows = {r.name: r for r in logins.login_rows(sandbox.env)}
    assert rows["kaggle"].status == "ok"
    assert "gpu login kaggle" in rows["kaggle"].summary
    assert rows["colab"].status == "missing"
    assert rows["lightning"].status == "missing"
    assert rows["lightning"].command == "gpu login lightning --browser"
    assert rows["hf"].status == "ok"
    assert {"groq", "gemini", "cloudflare"} <= set(rows)
    assert sandbox.run.calls == []  # no subprocess, no network


def test_hf_token_from_the_environment_is_named_not_shown(sandbox: Sandbox) -> None:
    sandbox.environ["HF_TOKEN"] = HF_TOKEN
    ui = ScriptedUi([True])
    ctx = sandbox.ctx(ui, only=["login.hf"])
    logins.run(ctx)
    assert "$HF_TOKEN" in ui.questions[0]
    assert _last(ctx).outcome is Outcome.DONE
    _no_secret_in(ui.text, HF_TOKEN)


def test_hf_token_dry_run_reads_no_file(sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch) -> None:
    sandbox.write(".cache/huggingface/token", HF_TOKEN)

    def no_read(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("a dry run read the token")

    monkeypatch.setattr("gpu_router.checkpoint.tokens.external_token", no_read)
    ctx = sandbox.ctx(ScriptedUi([]), only=["login.hf"], dry_run=True)
    logins.run(ctx)
    assert _last(ctx).outcome is Outcome.SKIPPED
