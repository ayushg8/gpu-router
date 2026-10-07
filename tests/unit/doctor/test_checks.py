"""Each `gpu doctor` check: ok / warn / fail / skip and the exact fix (phase 8a). Every probe
runs against tmp dirs, a scripted runner and a fake daemon client."""

from __future__ import annotations

import json
import os
import plistlib
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from gpu_router import secrets
from gpu_router.doctor import checks
from gpu_router.doctor.model import Status
from gpu_router.doctor.probe import CmdResult, ProbeEnv, tool_info
from gpu_router.errors import NotReady
from gpu_router.paths import Paths
from gpu_router.providers.catalog import load_catalog
from tests.unit.doctor.conftest import (
    FakeClient,
    FakeRun,
    daemon_up,
    default_run,
    ok,
    provider,
    write,
)

MakeEnv = Callable[..., ProbeEnv]
OK, WARN, FAIL, SKIP = Status.OK, Status.WARN, Status.FAIL, Status.SKIP


def entry(name: str) -> Any:
    return load_catalog(None).providers[name]


# =========================================================================== daemon


def test_daemon_down_and_idle_is_a_warning(make_env: MakeEnv) -> None:
    r = checks.check_daemon_running(make_env())
    assert r.status is WARN
    assert r.fix == "gpu daemon start"
    assert "nothing is active" in r.summary


def test_daemon_down_with_active_jobs_fails(make_env: MakeEnv, paths: Paths) -> None:
    write(paths.state, json.dumps({"written_at": 1, "active": [{"id": "a"}, {"id": "b"}]}))
    r = checks.check_daemon_running(make_env())
    assert r.status is FAIL
    assert "2 job(s) were active" in r.summary


def test_daemon_up(make_env: MakeEnv) -> None:
    env = make_env(daemon=daemon_up(FakeClient(), pid=77, test_mode=True))
    r = checks.check_daemon_running(env)
    assert r.status is OK
    assert "pid 77" in r.summary
    assert "port 47291" in r.summary
    assert "test mode" in r.summary
    not_ready = make_env(daemon=daemon_up(FakeClient(), ready=False))
    assert checks.check_daemon_running(not_ready).status is WARN


def test_version_mismatch_says_restart(make_env: MakeEnv) -> None:
    env = make_env(daemon=daemon_up(FakeClient(), version="0.0.9"))
    r = checks.check_daemon_version(env)
    assert r.status is WARN
    assert r.fix == "gpu daemon stop && gpu daemon start"
    assert checks.check_daemon_version(make_env(daemon=daemon_up(FakeClient()))).status is OK
    assert checks.check_daemon_version(make_env()).status is SKIP


def _plist(
    env: ProbeEnv, *, program: str, home: Path | None, process_type: str | None = None
) -> None:
    agent: dict[str, Any] = {"Label": "dev.gpu-router.daemon", "ProgramArguments": [program]}
    if process_type is not None:
        agent["ProcessType"] = process_type
    if home is not None:
        agent["EnvironmentVariables"] = {"GPU_ROUTER_HOME": str(home)}
    assert env.launchd_plist is not None
    env.launchd_plist.parent.mkdir(parents=True, exist_ok=True)
    env.launchd_plist.write_bytes(plistlib.dumps(agent))


def test_launchd_not_installed(make_env: MakeEnv) -> None:
    r = checks.check_launchd(make_env())
    assert r.status is WARN
    assert r.fix == "gpu daemon install-launchd"
    custom = make_env(environ={"GPU_ROUTER_HOME": "/somewhere/else"})
    assert checks.check_launchd(custom).status is SKIP


def test_launchd_agent_for_this_home(make_env: MakeEnv, paths: Paths, tmp_path: Path) -> None:
    program = write(tmp_path / "bin" / "gpu", "#!/bin/sh\n", 0o755)
    printed = ok("dev.gpu-router.daemon = {\n\tstate = running\n\tpid = 4242\n}\n")
    run = default_run().on(r"launchctl print gui/\d+/dev.gpu-router.daemon", printed)
    env = make_env(run=run, daemon=daemon_up(FakeClient(), pid=4242))
    _plist(env, program=str(program), home=paths.home)
    r = checks.check_launchd(env)
    assert r.status is OK
    assert "pid 4242" in r.summary


def test_launchd_agent_at_background_priority_is_a_warning(
    make_env: MakeEnv, paths: Paths, tmp_path: Path
) -> None:
    """2026-10-06: an agent with ProcessType Background got 1 s of CPU in 14 minutes while
    the Mac was busy; agents written before the fix are flagged with the reinstall."""
    program = write(tmp_path / "bin" / "gpu", "#!/bin/sh\n", 0o755)
    printed = ok("dev.gpu-router.daemon = {\n\tstate = running\n\tpid = 4242\n}\n")
    run = default_run().on(r"launchctl print gui/\d+/dev.gpu-router.daemon", printed)
    env = make_env(run=run, daemon=daemon_up(FakeClient(), pid=4242))
    _plist(env, program=str(program), home=paths.home, process_type="Background")
    r = checks.check_launchd(env)
    assert r.status is WARN
    assert "background priority" in r.summary
    assert r.fix == "gpu daemon install-launchd"
    _plist(env, program=str(program), home=paths.home, process_type="Standard")
    assert checks.check_launchd(env).status is OK


def test_launchd_installed_but_not_loaded(make_env: MakeEnv, paths: Paths, tmp_path: Path) -> None:
    program = write(tmp_path / "bin" / "gpu", "#!/bin/sh\n", 0o755)
    run = default_run().on(r"launchctl print", CmdResult(113, "", "Could not find service"))
    env = make_env(run=run)
    _plist(env, program=str(program), home=paths.home)
    r = checks.check_launchd(env)
    assert r.status is WARN
    assert r.fix is not None
    assert r.fix.startswith("launchctl bootstrap gui/")


def test_launchd_agent_whose_program_is_gone(make_env: MakeEnv, paths: Paths) -> None:
    env = make_env()
    _plist(env, program="/nowhere/gpu", home=paths.home)
    r = checks.check_launchd(env)
    assert r.status is FAIL
    assert r.fix == "gpu daemon install-launchd"


def test_launchd_agent_for_another_home(make_env: MakeEnv, tmp_path: Path) -> None:
    program = write(tmp_path / "bin" / "gpu", "#!/bin/sh\n", 0o755)
    env = make_env()
    _plist(env, program=str(program), home=tmp_path / "other-home")
    assert checks.check_launchd(env).status is SKIP


def test_a_custom_home_never_fails_about_the_default_homes_agent(
    make_env: MakeEnv, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Review fix: with GPU_ROUTER_HOME set, a stale default-home agent FAILed (exit 1) with
    the fix `gpu daemon install-launchd`, which would hand the global agent to the tmp dir."""
    monkeypatch.setattr("gpu_router.paths.DEFAULT_HOME", tmp_path / "default-home")
    env = make_env(environ={"GPU_ROUTER_HOME": "/tmp/x"})
    _plist(env, program="/nonexistent/gpu", home=None)  # serves the default data dir
    r = checks.check_launchd(env)
    assert r.status is SKIP
    assert "not this data dir" in r.summary
    assert env.launchd_plist is not None
    env.launchd_plist.write_bytes(b"not a plist")
    r = checks.check_launchd(env)
    assert r.status is SKIP  # unreadable, but not this (custom) dir's business
    assert r.fix is None


def test_install_launchd_refuses_a_custom_home(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The label is global: `gpu daemon install-launchd` with GPU_ROUTER_HOME set (as every
    test has) must not take the agent away from the default data dir."""
    from gpu_router.daemon import __main__ as daemon_cli
    from gpu_router.daemon import launchd

    def boom(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("launchd.install must not run")

    monkeypatch.setattr(launchd, "install", boom)
    code = daemon_cli.main(["install-launchd", "--json"])
    assert code != 0
    doc = json.loads(capsys.readouterr().out)
    assert "GPU_ROUTER_HOME is set" in doc["error"]["message"]
    assert "--this-home" in doc["error"]["hint"]


def test_a_cleanly_stopped_daemon_under_launchd_is_a_warning(
    make_env: MakeEnv, paths: Paths, tmp_path: Path
) -> None:
    """Review fix: `gpu daemon stop` exits 0 and KeepAlive {SuccessfulExit: false} does not
    restart it, yet doctor FAILed 'although launchd should keep it running'."""
    program = write(tmp_path / "bin" / "gpu", "#!/bin/sh\n", 0o755)
    clean = ok("dev.gpu-router.daemon = {\n\tstate = not running\n\tlast exit code = 0\n}\n")
    env = make_env(run=default_run().on(r"launchctl print gui/", clean))
    _plist(env, program=str(program), home=paths.home)
    r = checks.check_daemon_running(env)
    assert r.status is WARN
    assert "launchd starts it again at login" in r.summary
    crashed = ok("dev.gpu-router.daemon = {\n\tstate = not running\n\tlast exit code = 1\n}\n")
    env = make_env(run=default_run().on(r"launchctl print gui/", crashed))
    r = checks.check_daemon_running(env)
    assert r.status is FAIL
    assert "exited with code 1" in r.summary


def test_state_json(make_env: MakeEnv, paths: Paths) -> None:
    up = daemon_up(FakeClient(), pid=4242, started_at=0.0)
    assert checks.check_state_file(make_env(daemon=up)).status is WARN  # missing while up
    assert checks.check_state_file(make_env()).status is SKIP  # missing, daemon down
    env = make_env(daemon=up)
    now = env.clock.now()
    write(paths.state, json.dumps({"written_at": now - 5, "daemon_pid": 4242, "active": []}))
    r = checks.check_state_file(env)
    assert r.status is OK
    assert "written 5s ago" in r.summary
    stale = {"written_at": now - 900, "daemon_pid": 4242, "active": [{}], "heartbeat_s": 60}
    write(paths.state, json.dumps(stale))
    r = checks.check_state_file(env)
    assert r.status is WARN
    assert "not rewritten for 15m" in r.summary
    write(paths.state, json.dumps({"written_at": now, "daemon_pid": 1, "active": []}))
    assert "pid 1" in checks.check_state_file(env).summary
    write(paths.state, "{not json")
    assert checks.check_state_file(env).status is WARN


# =========================================================================== providers


def test_tool_version_comes_from_the_uv_tool_env_without_running_it(
    user_home: Path, tmp_path: Path
) -> None:
    venv = tmp_path / "tools" / "kaggle"
    write(venv / "pyvenv.cfg", "home = /x\n")
    exe = write(venv / "bin" / "kaggle", "#!/bin/sh\n", 0o755)
    (venv / "lib" / "python3.12" / "site-packages" / "kaggle-2.2.4.dist-info").mkdir(parents=True)
    run = FakeRun()
    info = tool_info(
        "kaggle", "kaggle", which={"kaggle": str(exe)}.get, run=run, user_home=user_home
    )
    assert info.version == "2.2.4"
    assert info.how == "dist-info"
    assert run.calls == []


def test_provider_cli(make_env: MakeEnv) -> None:
    r = checks.check_provider_cli(make_env(), entry("kaggle"))
    assert r.status is OK
    assert "kaggle 2.2.4" in r.summary
    missing = make_env(which={})
    r = checks.check_provider_cli(missing, entry("colab"))
    assert r.status is FAIL
    assert r.fix == "uv tool install google-colab-cli"
    old = default_run().on(r"kaggle --version$", ok("Kaggle API 1.6.17"))
    r = checks.check_provider_cli(make_env(run=old), entry("kaggle"))
    assert r.status is WARN
    assert r.fix == "uv tool upgrade kaggle"


def test_disabled_providers_are_not_probed(make_env: MakeEnv, paths: Paths) -> None:
    write(paths.config, "providers:\n  colab:\n    enabled: false\n")
    run = default_run()
    env = make_env(run=run)
    for fn in (checks.check_provider_cli, checks.check_colab_login, checks.check_colab_gpus):
        r = fn(env, entry("colab"))
        assert r.status is SKIP
        assert "not enabled" in r.summary
    assert run.calls == []


def test_kaggle_login_without_credentials(make_env: MakeEnv) -> None:
    r = checks.check_kaggle_login(make_env(), entry("kaggle"))
    assert r.status is FAIL
    assert r.fix == checks.KAGGLE_FILE_FIX


def test_kaggle_json_is_judged_by_its_mode_never_read(make_env: MakeEnv, user_home: Path) -> None:
    token = write(user_home / ".kaggle" / "kaggle.json", '{"username": "u", "key": "k"}', 0o000)
    r = checks.check_kaggle_login(make_env(), entry("kaggle"))
    assert r.status is OK
    assert "~/.kaggle/kaggle.json (0000)" in r.summary
    os.chmod(token, 0o644)
    r = checks.check_kaggle_login(make_env(), entry("kaggle"))
    assert r.status is WARN
    assert r.fix == f"chmod 600 {token}"
    assert '"key"' not in json.dumps(r.model_dump())


def test_kaggle_login_from_the_keychain_and_env(make_env: MakeEnv, paths: Paths) -> None:
    secrets.set_secret("kaggle", '{"username": "u", "key": "k"}')
    r = checks.check_kaggle_login(make_env(), entry("kaggle"))
    assert r.status is OK
    assert "Keychain (kaggle)" in r.summary
    secrets.delete_secret("kaggle")
    env = make_env(environ={"KAGGLE_USERNAME": "u", "KAGGLE_KEY": "k"})
    r = checks.check_kaggle_login(env, entry("kaggle"))
    assert "environment variables" in r.summary
    # review fix: only this shell has them; the launchd daemon would not
    assert r.status is WARN
    assert "launchd does not" in r.summary
    assert r.fix == "gpu login kaggle"


def test_colab_login(make_env: MakeEnv, user_home: Path) -> None:
    r = checks.check_colab_login(make_env(), entry("colab"))
    assert r.status is FAIL
    assert r.fix == checks.ADC_LOGIN
    write(user_home / ".config" / "gcloud" / "application_default_credentials.json")
    run = default_run()
    r = checks.check_colab_login(make_env(run=run), entry("colab"))
    assert r.status is OK
    assert "me@example.com" in r.summary
    # the CLI ran with a private HOME (nothing lands in ~/.config/colab-cli) and the real
    # gcloud dir for the credentials; the throwaway HOME is gone afterwards
    i = next(i for i, c in enumerate(run.calls) if c[-1] == "whoami")
    child = run.envs[i]
    assert child is not None
    assert child["HOME"] != str(user_home)
    assert not Path(child["HOME"]).exists()
    assert child["CLOUDSDK_CONFIG"] == str(user_home / ".config" / "gcloud")


@pytest.mark.parametrize(
    ("result", "status", "fix"),
    [
        (ok("Email: me@example.com\nScopes:\n  - openid\n"), FAIL, checks.ADC_LOGIN),
        (
            CmdResult(1, "", "[colab] whoami: failed to refresh credentials: invalid_grant"),
            FAIL,
            checks.ADC_LOGIN,
        ),
        (CmdResult(1, "", "boom"), WARN, None),
        (CmdResult(None, "", "", error="timed out after 25s"), WARN, None),
    ],
)
def test_colab_login_problems(
    make_env: MakeEnv, user_home: Path, result: CmdResult, status: Status, fix: str | None
) -> None:
    write(user_home / ".config" / "gcloud" / "application_default_credentials.json")
    run = default_run().on(r"whoami$", result)
    r = checks.check_colab_login(make_env(run=run), entry("colab"))
    assert r.status is status
    assert r.fix == fix


def test_lightning_sdk(make_env: MakeEnv, user_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tool = user_home / ".local" / "share" / "uv" / "tools" / "lightning-sdk"
    write(tool / "pyvenv.cfg", "home = /x\n")
    write(tool / "bin" / "python", "", 0o755)
    (
        tool / "lib" / "python3.12" / "site-packages" / "lightning_sdk-2026.9.18.post1.dist-info"
    ).mkdir(parents=True)
    r = checks.check_lightning_sdk(make_env(), entry("lightning"))
    assert r.status is OK
    assert "lightning-sdk 2026.9.18.post1" in r.summary
    (tool / "bin" / "python").unlink()
    r = checks.check_lightning_sdk(make_env(), entry("lightning"))
    assert r.status is OK
    assert "uv run" in r.summary
    monkeypatch.setattr(checks, "_uv_path", lambda _env: None)
    r = checks.check_lightning_sdk(make_env(), entry("lightning"))
    assert r.status is FAIL
    assert r.fix == "uv tool install lightning-sdk"


def test_lightning_login(make_env: MakeEnv, user_home: Path) -> None:
    r = checks.check_lightning_login(make_env(), entry("lightning"))
    assert r.status is FAIL
    assert r.fix in ("gpu login lightning", checks._lightning_login_fix())
    secrets.set_secret("LIGHTNING_USER_ID", "u123")
    r = checks.check_lightning_login(make_env(), entry("lightning"))
    assert r.status is WARN
    assert r.fix == "gpu secrets set LIGHTNING_API_KEY"
    secrets.set_secret("LIGHTNING_API_KEY", "k" * 20)
    assert checks.check_lightning_login(make_env(), entry("lightning")).status is OK
    for n in ("LIGHTNING_USER_ID", "LIGHTNING_API_KEY"):
        secrets.delete_secret(n)
    cred = write(user_home / ".lightning" / "credentials.json", "{}", 0o644)
    r = checks.check_lightning_login(make_env(), entry("lightning"))
    assert r.status is WARN
    assert r.fix == f"chmod 600 {cred}"


def test_lightning_login_follows_login_source(
    make_env: MakeEnv, user_home: Path, paths: Paths
) -> None:
    lightning = entry("lightning")
    # LIGHTNING_AUTH_TOKEN is not a source: the adapter clears it from the SDK's env
    r = checks.check_lightning_login(make_env(environ={"LIGHTNING_AUTH_TOKEN": "t"}), lightning)
    assert r.status is FAIL
    env_vars = {"LIGHTNING_USER_ID": "u1", "LIGHTNING_API_KEY": "k" * 20}
    r = checks.check_lightning_login(make_env(environ=env_vars), lightning)
    # review fix: doctor's shell is not the daemon's environment (launchd has none)
    assert r.status is WARN
    assert "launchd does not" in r.summary
    assert r.fix == checks._lightning_login_fix()
    assert r.detail["uses"] == "env"
    dumped = r.model_dump_json()
    assert "u1" not in dumped
    assert "k" * 20 not in dumped
    # keychain mode: env vars and the file do not count
    write(paths.config, "providers:\n  lightning:\n    login_source: keychain\n")
    write(user_home / ".lightning" / "credentials.json", "{}", 0o600)
    r = checks.check_lightning_login(make_env(environ=env_vars), lightning)
    assert r.status is FAIL
    assert "login_source is `keychain`" in r.summary
    assert "found:" in r.summary
    assert r.fix == checks._lightning_login_fix()
    secrets.set_secret("LIGHTNING_USER_ID", "u123")
    secrets.set_secret("LIGHTNING_API_KEY", "k" * 20)
    r = checks.check_lightning_login(make_env(environ=env_vars), lightning)
    assert r.status is OK
    assert r.detail["uses"] == "keychain"
    assert "uses the keychain copy" in r.summary
    # file mode: the Keychain does not count
    write(paths.config, "providers:\n  lightning:\n    login_source: file\n")
    (user_home / ".lightning" / "credentials.json").unlink()
    r = checks.check_lightning_login(make_env(), lightning)
    assert r.status is FAIL
    assert r.fix == "lightning login"
    # env mode without the vars in this shell: the daemon's env is what matters
    write(paths.config, "providers:\n  lightning:\n    login_source: env\n")
    r = checks.check_lightning_login(make_env(), lightning)
    assert r.status is WARN
    assert "daemon needs LIGHTNING_USER_ID" in r.summary
    write(paths.config, "providers:\n  lightning:\n    login_source: sso\n")
    r = checks.check_lightning_login(make_env(), lightning)
    assert r.status is FAIL
    assert r.fix == f"${{EDITOR:-nano}} {paths.config}"


def _live_env(make_env: MakeEnv, **kw: Any) -> ProbeEnv:
    client = FakeClient(
        providers=[provider("kaggle"), provider("colab", enabled=False)],
        quota=[
            {
                "provider": "kaggle",
                "used": 3.25,
                "limit": 30,
                "unit": "gpu_hours",
                "resets_at": None,
                "source": "live",
                "detail": {"basis": "live"},
                "observed_at": 0,
            }
        ],
        **kw,
    )
    return make_env(daemon=daemon_up(client))


def test_live_healthcheck(make_env: MakeEnv) -> None:
    assert checks.check_provider_live(make_env(), entry("kaggle")).status is SKIP
    env = _live_env(make_env)
    r = checks.check_provider_live(env, entry("kaggle"))
    assert r.status is OK
    assert "3.25 of 30h used" in r.summary
    assert checks.check_provider_live(env, entry("colab")).status is SKIP  # not enabled


@pytest.mark.parametrize(
    ("health", "status", "fix"),
    [
        (
            {"health": "auth_required", "health_reason": "kaggle is not logged in"},
            FAIL,
            "gpu login kaggle",
        ),
        ({"health": "degraded", "health_reason": "2 other sessions"}, WARN, None),
        (
            {"health": "unavailable", "health_reason": "the kaggle CLI is not installed"},
            FAIL,
            "uv tool install kaggle",
        ),
        ({"health": "unavailable", "health_reason": "429 too many requests"}, WARN, None),
        ({"health": "disabled", "health_reason": "off in test mode"}, SKIP, None),
    ],
)
def test_live_problems(
    make_env: MakeEnv, health: dict[str, str], status: Status, fix: str | None
) -> None:
    env = _live_env(make_env, health={"kaggle": health})
    r = checks.check_provider_live(env, entry("kaggle"))
    assert r.status is status
    assert r.fix == fix


def test_live_while_recovering(make_env: MakeEnv) -> None:
    env = _live_env(make_env, errors={"/providers/kaggle/healthcheck": NotReady("recovering")})
    assert checks.check_provider_live(env, entry("kaggle")).status is SKIP


def test_modal_is_listed_as_excluded_with_its_own_words(make_env: MakeEnv) -> None:
    env = make_env()
    excluded = checks._excluded(env)
    assert "payment method on file" in excluded["modal"]
    r = checks.check_excluded(env, "modal", excluded["modal"])
    assert r.status is OK
    # data-driven: lightning's L4 (24GB) is the largest free GPU in the packaged catalog
    assert "no free provider fits jobs over 16GB VRAM (largest: colab)" in r.summary


def test_leftover_modal_settings_and_secrets(make_env: MakeEnv, paths: Paths) -> None:
    why = checks.EXCLUDED["modal"]
    secrets.set_secret("MODAL_TOKEN_ID", "ak-xxxxxxxxxxxxxxxxxxxxxxxx")
    r = checks.check_excluded(make_env(), "modal", why)
    assert r.status is WARN
    assert r.fix == "gpu secrets rm MODAL_TOKEN_ID"
    write(paths.config, "providers:\n  modal:\n    enabled: true\n")
    r = checks.check_excluded(make_env(), "modal", why)
    assert r.status is WARN
    assert r.fix is not None
    assert str(paths.config) in r.fix


def test_verify_lane_and_manual_entries_are_never_probed(make_env: MakeEnv) -> None:
    ids = [t.id for t in checks.default_tasks(make_env())]
    assert "provider.verify_at_signup" in ids
    assert not any(
        i.startswith(("provider.paperspace", "provider.saturn", "provider.sagemaker")) for i in ids
    )
    assert "provider.modal.excluded" in ids
    assert not any(i.startswith("provider.fake") for i in ids)  # no test-mode daemon


# =========================================================================== storage


def test_hf_token(make_env: MakeEnv, paths: Paths) -> None:
    r = checks.check_hf_token(make_env())
    assert r.status is WARN
    assert r.fix == "gpu login hf"
    secrets.set_secret("HF_TOKEN", "hf_" + "a" * 34)
    r = checks.check_hf_token(make_env(whoami=("demo", None)))
    assert r.status is OK
    assert "user demo" in r.summary
    r = checks.check_hf_token(make_env(whoami=(None, "hugging face rejected this token")))
    assert r.status is FAIL
    assert r.fix == "gpu login hf"
    assert checks.check_hf_token(make_env(whoami=(None, None))).status is WARN
    write(paths.config, "checkpoint:\n  backend: local\n")
    assert checks.check_hf_token(make_env()).status is SKIP


def test_hf_remote_token(make_env: MakeEnv) -> None:
    r = checks.check_hf_remote(make_env())
    assert r.status is WARN
    assert r.fix == "gpu login hf --remote"
    secrets.set_secret("HF_TOKEN_REMOTE", "hf_" + "b" * 34)
    assert checks.check_hf_remote(make_env()).status is OK


def test_hf_rows_check_the_token_not_just_its_name(make_env: MakeEnv) -> None:
    """Review fix: HF_TOKEN 'works' for a read-only token, and HF_TOKEN_REMOTE was ok because
    the Keychain name existed (a revoked token passed)."""
    secrets.set_secret("HF_TOKEN", "hf_" + "a" * 34)
    r = checks.check_hf_token(make_env(role="read"))
    assert r.status is WARN
    assert "read-only" in r.summary
    assert r.fix == "gpu login hf"
    assert checks.check_hf_token(make_env(role=None)).status is OK  # unknown role: no claim
    secrets.set_secret("HF_TOKEN_REMOTE", "hf_" + "b" * 34)
    r = checks.check_hf_remote(make_env(whoami=(None, "hugging face rejected this token")))
    assert r.status is FAIL
    assert r.fix == "gpu login hf --remote"
    r = checks.check_hf_remote(make_env(whoami=(None, None)))
    assert r.status is WARN
    assert "not verified" in r.summary
    r = checks.check_hf_remote(make_env(role="read"))
    assert r.status is WARN
    assert "read-only" in r.summary
    r = checks.check_hf_remote(make_env())
    assert r.status is OK
    assert "works (user me)" in r.summary


def test_token_role_reads_whoamis_access_token() -> None:
    from gpu_router.cli.login import token_role

    def doc(access: dict[str, Any]) -> dict[str, Any]:
        return {"name": "me", "auth": {"accessToken": access}}

    assert token_role(doc({"role": "write"})) == "write"
    assert token_role(doc({"role": "read"})) == "read"
    fine_w = {"role": "fineGrained", "fineGrained": {"scoped": [{"permissions": ["repo.write"]}]}}
    fine_r = {"role": "fineGrained", "fineGrained": {"scoped": [{"permissions": ["repo.read"]}]}}
    assert token_role(doc(fine_w)) == "write"
    assert token_role(doc(fine_r)) == "read"
    assert token_role({"name": "me"}) is None
    assert token_role(None) is None


def test_hf_hub_version(make_env: MakeEnv) -> None:
    assert checks.check_hf_hub(make_env()).status is OK


# =========================================================================== inference


def test_inference_keys_are_optional(make_env: MakeEnv) -> None:
    r = checks.check_inference(make_env())
    assert r.status is SKIP
    assert r.summary.startswith("optional: no inference keys yet (groq")
    assert r.fix == "gpu login groq"


def test_inference_keys_by_name_only(make_env: MakeEnv, monkeypatch: pytest.MonkeyPatch) -> None:
    secrets.set_secret("INFER_GROQ_API_KEY", "gsk_" + "a" * 40)
    secrets.set_secret("HF_TOKEN", "hf_" + "a" * 34)

    def boom(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("doctor must not read secret values for the inference row")

    monkeypatch.setattr(secrets, "get_secret", boom)
    r = checks.check_inference(make_env())
    assert r.status is OK
    assert r.summary.startswith("keys for groq, hf")
    assert "gemini (gpu login gemini)" in r.summary
    assert r.detail["ready"] == ["groq", "hf"]
    assert "gsk_" not in r.model_dump_json()


def test_inference_half_set_up_and_rejected(make_env: MakeEnv) -> None:
    secrets.set_secret("INFER_CLOUDFLARE_API_TOKEN", "c" * 40)
    r = checks.check_inference(make_env())
    assert r.status is WARN
    assert "INFER_CLOUDFLARE_ACCOUNT_ID" in r.summary
    assert r.fix == "gpu login cloudflare"
    secrets.set_secret("INFER_CLOUDFLARE_ACCOUNT_ID", "a" * 32)
    client = FakeClient(
        infer_quota=[
            {"provider": "groq", "blocked": {}},
            {"provider": "cloudflare", "blocked": {"*": "key rejected until 14:05"}},
        ]
    )
    r = checks.check_inference(make_env(daemon=daemon_up(client)))
    assert r.status is WARN
    assert r.summary == (
        "cloudflare: key rejected until 14:05 (the provider refused the stored key)"
    )
    assert r.fix == "gpu login cloudflare"
    assert ("GET", "/infer/quota") in client.calls
    # an older daemon without the endpoint is not an error
    old = FakeClient(errors={"/infer/quota": NotReady("no such endpoint")})
    assert checks.check_inference(make_env(daemon=daemon_up(old))).status is OK


def test_inference_broken_entry(make_env: MakeEnv, paths: Paths) -> None:
    write(paths.user_providers, "inference:\n  groq:\n    base_url: 12\n    priority: x\n")
    r = checks.check_inference(make_env())
    assert r.status is WARN
    assert r.summary.startswith("inference.groq in providers.yaml is skipped")
    assert r.fix == f"${{EDITOR:-nano}} {paths.user_providers}"


# =========================================================================== limits


SAT = 1_790_380_800.0  # a Saturday 00:00 UTC (2026-09-26)


def _quota_env(make_env: MakeEnv, **q: Any) -> ProbeEnv:
    snap = {
        "provider": "kaggle",
        "used": 1,
        "limit": 30,
        "unit": "gpu_hours",
        "resets_at": SAT,
        "source": "live",
        "detail": {"basis": "live"},
        "observed_at": SAT - 3 * 86400,
    }
    snap.update(q)
    client = FakeClient(providers=[provider("kaggle"), provider("colab")], quota=[snap])
    return make_env(daemon=daemon_up(client))


def test_live_anchor() -> None:
    assert checks.live_anchor("weekly", SAT) == "sat 00:00 UTC"
    assert checks.live_anchor("monthly", SAT + 40) == "day 26 00:01 UTC"
    assert checks.live_anchor("unknown", SAT) is None


def test_kaggle_limits_match(make_env: MakeEnv) -> None:
    r = checks.check_limits(_quota_env(make_env), entry("kaggle"))
    assert r.status is OK
    assert "resets sat 00:00 UTC" in r.summary


def test_kaggle_limit_and_reset_drift(make_env: MakeEnv) -> None:
    r = checks.check_limits(_quota_env(make_env, limit=29, resets_at=SAT - 86400), entry("kaggle"))
    assert r.status is WARN
    assert r.fix == "gpu doctor --update-catalog"
    drift = {d["key"]: d for d in r.detail["drift"]}
    assert drift["quota.limit"]["live"] == 29
    assert drift["quota.limit"]["catalog"] == 30
    assert drift["quota.reset_anchor"]["live"] == "fri 00:00 UTC"


def test_limits_skip_without_a_live_reading(make_env: MakeEnv) -> None:
    assert checks.check_limits(make_env(), entry("kaggle")).status is SKIP
    est = _quota_env(make_env, source="estimate", detail={"basis": "history"})
    assert checks.check_limits(est, entry("kaggle")).status is SKIP
    r = checks.check_limits(_quota_env(make_env), entry("colab"))
    assert r.status is SKIP
    assert "does not publish" in r.summary


def test_colab_gpus_from_wrapped_help() -> None:
    from tests.unit.doctor.conftest import COLAB_NEW_HELP

    assert checks.colab_gpus_from_help(COLAB_NEW_HELP) == ["T4", "L4", "G4", "H100", "A100"]
    assert checks.colab_gpus_from_help("nothing here") is None


def test_colab_gpus(make_env: MakeEnv, paths: Paths) -> None:
    r = checks.check_colab_gpus(make_env(), entry("colab"))
    assert r.status is OK
    assert "T4" in r.summary
    assert "paid tiers" in r.summary
    write(paths.user_providers, "providers:\n  colab:\n    gpus: [{name: K80, vram_gb: 12}]\n")
    env = make_env()
    r = checks.check_colab_gpus(env, env.catalog().providers["colab"])
    assert r.status is WARN
    assert "K80" in r.summary


# =========================================================================== local


def test_perms(make_env: MakeEnv, paths: Paths) -> None:
    write(paths.token, "t")
    write(paths.state, "{}", 0o644)  # 0644 by design (inside the 0700 dir)
    r = checks.check_perms(make_env())
    assert r.status is OK, r.summary
    os.chmod(paths.token, 0o644)
    r = checks.check_perms(make_env())
    assert r.status is FAIL
    assert r.fix == f"chmod 600 {paths.token}"
    os.chmod(paths.token, 0o600)
    os.chmod(paths.home, 0o755)
    r = checks.check_perms(make_env())
    assert r.status is FAIL
    assert r.fix == f"chmod 700 {paths.home}"
    os.chmod(paths.home, 0o700)
    os.chmod(paths.logs_dir, 0o755)
    assert checks.check_perms(make_env()).status is WARN
    os.chmod(paths.logs_dir, 0o700)
    write(paths.home / "notes.txt", "x", 0o666)
    assert checks.check_perms(make_env()).status is FAIL  # writable by others


def test_perms_before_the_first_start(make_env: MakeEnv, tmp_path: Path) -> None:
    env = make_env()
    env.paths = Paths(tmp_path / "not-yet")
    assert checks.check_perms(env).status is SKIP


@pytest.mark.parametrize(("free_gb", "status"), [(0.5, FAIL), (3, WARN), (100, OK)])
def test_disk(
    make_env: MakeEnv, monkeypatch: pytest.MonkeyPatch, free_gb: float, status: Status
) -> None:
    import shutil

    monkeypatch.setattr(
        shutil, "disk_usage", lambda _p: shutil._ntuple_diskusage(1, 1, int(free_gb * 1e9))
    )  # type: ignore[attr-defined]
    assert checks.check_disk(make_env()).status is status


def test_config(make_env: MakeEnv, paths: Paths) -> None:
    assert checks.check_config(make_env()).status is OK
    write(paths.config, "notifications:\n  backend: growl\n")
    r = checks.check_config(make_env())
    assert r.status is FAIL
    assert r.fix == f"${{EDITOR:-nano}} {paths.config}"
    write(paths.config, "daemon: [\n")
    assert checks.check_config(make_env()).status is FAIL
    write(paths.config, "")
    write(paths.user_providers, "providers:\n  kaggle:\n    quota: {limit: -}\n  x: 1\n")
    r = checks.check_config(make_env())
    assert r.status is FAIL
    assert str(paths.user_providers) in (r.fix or "")


def test_git_and_uv(make_env: MakeEnv) -> None:
    assert checks.check_git(make_env()).status is OK
    r = checks.check_git(make_env(which={}))
    assert r.status is FAIL
    assert r.fix == "xcode-select --install"
    assert "uv 0.11.29" in checks.check_uv(make_env()).summary
    assert checks.check_uv(make_env(which={})).status is WARN


# =========================================================================== integration


def test_gpu_on_path(make_env: MakeEnv) -> None:
    assert checks.check_gpu_on_path(make_env()).status is OK
    r = checks.check_gpu_on_path(make_env(which={}))
    assert r.status is WARN
    assert r.fix is not None
    assert r.fix.startswith("uv tool install")


def test_gpu_installed_but_off_path_says_update_shell(make_env: MakeEnv, user_home: Path) -> None:
    """Review fix: `uv tool install` answers "already installed" when ~/.local/bin just is
    not on PATH, so doctor's install fix changed nothing and the warning repeated forever."""
    write(user_home / ".local" / "bin" / "gpu", "#!/bin/sh\n", 0o755)
    r = checks.check_gpu_on_path(make_env(which={}))
    assert r.status is WARN
    assert r.fix == "uv tool update-shell"
    assert "~/.local/bin/gpu" in r.summary
    custom = user_home / "bin-tools"
    write(custom / "gpu", "#!/bin/sh\n", 0o755)
    r = checks.check_gpu_on_path(make_env(which={}, environ={"UV_TOOL_BIN_DIR": str(custom)}))
    assert r.fix == "uv tool update-shell"


def test_statusline(make_env: MakeEnv, user_home: Path) -> None:
    r = checks.check_statusline(make_env())
    assert r.status is WARN
    assert r.fix == "gpu statusline install"
    settings = user_home / ".claude" / "settings.json"
    write(settings, json.dumps({"statusLine": {"type": "command", "command": "bash x.sh"}}), 0o644)
    assert checks.check_statusline(make_env()).fix == "gpu statusline install"
    gone = {"statusLine": {"type": "command", "command": 'bash "/nowhere/gpu-statusline.sh"'}}
    write(settings, json.dumps(gone), 0o644)
    r = checks.check_statusline(make_env())
    assert r.status is WARN


def test_plugin(make_env: MakeEnv, user_home: Path) -> None:
    r = checks.check_plugin(make_env())
    assert r.status is WARN
    assert "claude plugin install gpu-router@gpu-router-local" in (r.fix or "")
    installed = user_home / ".claude" / "plugins" / "installed_plugins.json"
    version = checks._packaged_plugin_version(make_env())
    body = {"version": 2, "plugins": {checks.PLUGIN_KEY: [{"version": version}]}}
    write(installed, json.dumps(body), 0o644)
    assert checks.check_plugin(make_env()).status is OK
    write(
        user_home / ".claude" / "settings.json",
        json.dumps({"enabledPlugins": {checks.PLUGIN_KEY: False}}),
    )
    r = checks.check_plugin(make_env())
    assert r.status is WARN
    assert r.fix == f"claude plugin enable {checks.PLUGIN_KEY}"
    (user_home / ".claude" / "settings.json").unlink()
    body["plugins"][checks.PLUGIN_KEY][0]["version"] = "0.0.1"  # type: ignore[index]
    write(installed, json.dumps(body), 0o644)
    assert "claude plugin update" in (checks.check_plugin(make_env()).fix or "")


def test_plugin_installed_from_github(
    make_env: MakeEnv, user_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The README's one-line install (`claude plugin marketplace add ayushg8/gpu-router`)
    records gpu-router@gpu-router: doctor sees it and its fixes name that marketplace."""
    installed = user_home / ".claude" / "plugins" / "installed_plugins.json"
    version = checks._packaged_plugin_version(make_env())
    body = {"version": 2, "plugins": {checks.GITHUB_PLUGIN_KEY: [{"version": version}]}}
    write(installed, json.dumps(body), 0o644)
    r = checks.check_plugin(make_env())
    assert r.status is OK
    assert r.summary.startswith("gpu-router@gpu-router installed")
    body["plugins"][checks.GITHUB_PLUGIN_KEY][0]["version"] = "0.0.1"  # type: ignore[index]
    write(installed, json.dumps(body), 0o644)
    assert checks.check_plugin(make_env()).fix == (
        "claude plugin marketplace update gpu-router && claude plugin update gpu-router@gpu-router"
    )
    installed.unlink()
    # not a checkout (e.g. `uv tool install git+https://...`): the fix installs from GitHub
    monkeypatch.setattr(ProbeEnv, "gpu_router_repo", lambda self: None)
    assert checks.check_plugin(make_env()).fix == (
        "claude plugin marketplace add ayushg8/gpu-router && "
        "claude plugin install gpu-router@gpu-router"
    )


def test_colab_skill_is_offered_a_confirmed_move(make_env: MakeEnv, user_home: Path) -> None:
    assert checks.check_colab_skill(make_env()).status is OK
    (user_home / ".claude" / "skills" / "colab").mkdir(parents=True)
    r = checks.check_colab_skill(make_env())
    assert r.status is WARN
    # review fix: `read -r -p` was bash-only (zsh: "-p: no coprocess"); the wizard's step
    # shows the move and asks in any shell
    assert r.fix == "gpu setup --only integration.colab_skill"
    assert (user_home / ".claude" / "skills" / "colab").is_dir()  # doctor never moves it


def test_codex(make_env: MakeEnv, user_home: Path) -> None:
    assert checks.check_codex(make_env()).status is SKIP
    env = make_env(which={"codex": "/fake/bin/codex"})
    r = checks.check_codex(env)
    assert r.status is WARN
    assert r.fix == "codex mcp add gpu-router -- gpu mcp"
    write(user_home / ".codex" / "config.toml", '[mcp_servers.gpu-router]\ncommand = "gpu"\n')
    assert checks.check_codex(env).status is OK


def test_notifications(make_env: MakeEnv, paths: Paths) -> None:
    r = checks.check_notifications(make_env())
    assert r.status is OK
    assert "finished, failed, approval, migrated" in r.summary
    assert checks.check_notifications(make_env(environ={"PYTEST_CURRENT_TEST": "x"})).status is SKIP
    write(paths.config, "notifications:\n  backend: terminal-notifier\n")
    r = checks.check_notifications(make_env())
    assert r.status is WARN
    assert r.fix == "brew install terminal-notifier"
    write(paths.config, "notifications:\n  enabled: false\n")
    assert checks.check_notifications(make_env()).status is SKIP


def test_notifications_report_the_running_daemons_backend(make_env: MakeEnv, paths: Paths) -> None:
    """Review fix: doctor recomputed the backend with the shell's PATH ("on via
    terminal-notifier") while the launchd daemon had silently picked none."""
    from tests.unit.doctor.conftest import FakeClient, daemon_up

    write(paths.config, "notifications:\n  backend: terminal-notifier\n")
    info = daemon_up(FakeClient())
    assert info.health is not None
    info.health = info.health.model_copy(
        update={"notifications": "off: terminal-notifier is not installed (brew install x)"}
    )
    shell_has_it = {"terminal-notifier": "/opt/homebrew/bin/terminal-notifier"}
    r = checks.check_notifications(make_env(daemon=info, which=shell_has_it))
    assert r.status is WARN
    assert r.summary.startswith("the daemon: terminal-notifier is not installed")
    info.health = info.health.model_copy(update={"notifications": "terminal-notifier"})
    r = checks.check_notifications(make_env(daemon=info))
    assert r.status is OK
    assert "(the daemon's)" in r.summary


def test_the_colab_probe_ends_before_the_deadline_and_leaves_no_home(
    make_env: MakeEnv, user_home: Path
) -> None:
    """Review fix: the colab CLI got the whole remaining deadline, so at the deadline the
    runner returned first and the worker's rmtree of gpu-doctor-colab-* never ran."""
    write(user_home / ".config" / "gcloud" / "application_default_credentials.json")
    budgets: list[float] = []
    homes: list[str] = []

    def slow_whoami(argv: list[str]) -> CmdResult:
        homes.append(argv[argv.index("--config") + 1])
        return CmdResult(124, "", "timed out")

    run = default_run().on(r"colab --auth=adc --config \S+ whoami$", slow_whoami)
    real_call = run.__call__

    def spy(argv: list[str], timeout: float, env: dict[str, str] | None = None) -> CmdResult:
        if "whoami" in argv:
            budgets.append(timeout)
        return real_call(argv, timeout, env=env)

    env = make_env(run=spy, deadline_s=3.0)  # type: ignore[arg-type]
    checks.check_colab_login(env, entry("colab"))
    assert budgets
    assert budgets[0] <= 3.0 - checks.COLAB_CLEANUP_S + 0.01
    assert homes
    assert not Path(homes[0]).parent.exists()  # the throwaway dir is gone
    assert not checks._COLAB_TMP
    left = Path(tempfile.mkdtemp(prefix="gpu-doctor-colab-"))
    checks._COLAB_TMP.add(left)
    checks._sweep_colab_tmp()  # the atexit sweep for a probe still running at exit
    assert not left.exists()
