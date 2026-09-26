"""notifications: config and the macOS backends (phase 8a). No real notification is sent:
every runner here is a fake."""

from __future__ import annotations

import subprocess
from typing import Any

import pytest

from gpu_router.config import Config
from gpu_router.errors import ConfigError
from gpu_router.notify.backends import (
    ENV_REAL,
    NotifyError,
    NullBackend,
    OsaScriptBackend,
    TerminalNotifierBackend,
    choose_backend,
)
from gpu_router.notify.format import Notification
from gpu_router.notify.settings import EVENT_KINDS, NotifySettings, notify_settings


def note(body: str = "3h12m on kaggle", subtitle: str = "✓ train finished") -> Notification:
    return Notification(
        kind="finished",
        title="gpu-router",
        subtitle=subtitle,
        body=body,
        job_id="a7f2",
        sound="Glass",
        group="gpu-router.a7f2",
    )


class Recorder:
    def __init__(self, rc: int = 0, stderr: str = "", exc: BaseException | None = None) -> None:
        self.calls: list[tuple[list[str], float]] = []
        self.rc = rc
        self.stderr = stderr
        self.exc = exc

    def __call__(self, argv: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
        self.calls.append((argv, timeout))
        if self.exc is not None:
            raise self.exc
        return subprocess.CompletedProcess(argv, self.rc, "", self.stderr)


# --------------------------------------------------------------------------- settings


def test_defaults_notify_every_event_type() -> None:
    s = notify_settings(None)
    assert s.enabled
    assert s.backend == "auto"
    assert s.sound
    assert s.enabled_kinds() == list(EVENT_KINDS)


def test_per_event_switches() -> None:
    s = notify_settings({"events": {"finished": False, "migrated": False}})
    assert s.enabled_kinds() == ["failed", "approval"]
    assert not s.wants("finished")
    assert not notify_settings({"enabled": False}).wants("failed")
    assert not notify_settings({"backend": "off"}).wants("failed")


@pytest.mark.parametrize(
    "raw",
    [
        {"events": {"finshed": True}},
        {"backend": "growl"},
        {"max_per_minute": 0},
        {"dedupe_s": -1},
        {"unknown": 1},
    ],
)
def test_bad_sections_are_config_errors(raw: dict[str, Any]) -> None:
    with pytest.raises(ConfigError) as exc:
        notify_settings(raw, source="config.yaml")
    assert "notifications" in exc.value.message


def test_config_accepts_the_section() -> None:
    config = Config.model_validate({"notifications": {"events": {"failed": False}}})
    assert notify_settings(config.notifications).enabled_kinds() == [
        "finished",
        "approval",
        "migrated",
    ]


# --------------------------------------------------------------------------- osascript


def test_osascript_passes_text_as_argv_never_as_script() -> None:
    run = Recorder()
    evil = 'x" & (do shell script "touch /tmp/pwned") & "'
    OsaScriptBackend(run=run).send(note(body=evil, subtitle=evil))
    argv, timeout = run.calls[0]
    assert argv[0] == "/usr/bin/osascript"
    script = " ".join(a for a in argv[: argv.index("--")] if a != "-e")
    assert "touch" not in script  # the job-controlled text is not AppleScript source
    assert argv[argv.index("--") + 1 :] == [evil, "gpu-router", evil, "Glass"]
    assert timeout == 10


def test_osascript_silent_notification_passes_an_empty_sound() -> None:
    run = Recorder()
    n = note()
    OsaScriptBackend(run=run).send(
        Notification(kind=n.kind, title=n.title, subtitle=n.subtitle, body=n.body)
    )
    assert run.calls[0][0][-1] == ""


@pytest.mark.parametrize(
    ("run", "match"),
    [
        (Recorder(rc=1, stderr="execution error: Not authorized"), "exited 1: execution error"),
        (Recorder(exc=subprocess.TimeoutExpired("osascript", 10)), "did not finish"),
        (Recorder(exc=FileNotFoundError(2, "No such file")), "cannot run osascript"),
    ],
)
def test_osascript_failures_are_notify_errors(run: Recorder, match: str) -> None:
    with pytest.raises(NotifyError, match=match):
        OsaScriptBackend(run=run).send(note())


# --------------------------------------------------------------------------- terminal-notifier


def test_terminal_notifier_groups_per_job_and_keeps_leading_dashes_literal() -> None:
    run = Recorder()
    TerminalNotifierBackend(path="/opt/homebrew/bin/terminal-notifier", run=run).send(
        note(body="-exec rm -rf ~")
    )
    argv = run.calls[0][0]
    assert argv[argv.index("-group") + 1] == "gpu-router.a7f2"
    assert argv[argv.index("-sound") + 1] == "Glass"
    assert argv[argv.index("-message") + 1] == " -exec rm -rf ~"
    assert "-execute" not in argv
    assert "-open" not in argv


# --------------------------------------------------------------------------- choosing


def which_none(_name: str) -> str | None:
    return None


def which_tn(name: str) -> str | None:
    return "/opt/homebrew/bin/terminal-notifier" if name == "terminal-notifier" else None


def test_never_real_under_pytest() -> None:
    s = NotifySettings(backend="osascript")
    b = choose_backend(s, test_mode=False, environ={"PYTEST_CURRENT_TEST": "x"})
    assert isinstance(b, NullBackend)
    assert "pytest" in b.why
    real = choose_backend(
        s, test_mode=False, environ={"PYTEST_CURRENT_TEST": "x", ENV_REAL: "1"}, which=which_none
    )
    assert isinstance(real, OsaScriptBackend)


def test_auto_is_off_in_test_mode_but_an_explicit_backend_is_honoured() -> None:
    auto = choose_backend(NotifySettings(), test_mode=True, environ={}, which=which_none)
    assert isinstance(auto, NullBackend)
    assert "test mode" in auto.why
    forced = choose_backend(
        NotifySettings(backend="osascript"), test_mode=True, environ={}, which=which_none
    )
    assert isinstance(forced, OsaScriptBackend)


def test_auto_prefers_terminal_notifier() -> None:
    b = choose_backend(NotifySettings(), test_mode=False, environ={}, which=which_tn)
    assert isinstance(b, TerminalNotifierBackend)
    b2 = choose_backend(NotifySettings(), test_mode=False, environ={}, which=which_none)
    assert isinstance(b2, OsaScriptBackend)


def test_explicit_terminal_notifier_missing_says_how_to_install() -> None:
    b = choose_backend(
        NotifySettings(backend="terminal-notifier"), test_mode=False, environ={}, which=which_none
    )
    assert isinstance(b, NullBackend)
    assert "brew install terminal-notifier" in b.why


def test_off() -> None:
    for s in (NotifySettings(enabled=False), NotifySettings(backend="off")):
        assert isinstance(choose_backend(s, test_mode=False, environ={}), NullBackend)


def test_timeout_setting_reaches_the_backend() -> None:
    b = choose_backend(NotifySettings(timeout_s=3), test_mode=False, environ={}, which=which_none)
    assert isinstance(b, OsaScriptBackend)
    assert b.timeout_s == 3


def test_a_launchd_path_still_finds_homebrews_terminal_notifier() -> None:
    """Review fix: a launchd daemon's PATH (/usr/bin:/bin:/usr/sbin:/sbin) never has
    Homebrew, so `backend: terminal-notifier` silently became a NullBackend."""
    installed = "/opt/homebrew/bin/terminal-notifier"

    def launchd_which(name: str) -> str | None:  # PATH lookups fail; the file exists
        return name if name == installed else None

    b = choose_backend(
        NotifySettings(backend="terminal-notifier"),
        test_mode=False,
        environ={"PATH": "/usr/bin:/bin:/usr/sbin:/sbin"},
        which=launchd_which,
    )
    assert isinstance(b, TerminalNotifierBackend)
    assert b.path == installed
    intel = choose_backend(
        NotifySettings(),
        test_mode=False,
        environ={},
        which=lambda n: n if n == "/usr/local/bin/terminal-notifier" else None,
    )
    assert isinstance(intel, TerminalNotifierBackend)


def test_describe_names_the_backend_or_why_it_is_off() -> None:
    from gpu_router.notify.backends import describe

    assert describe(TerminalNotifierBackend(path="/x")) == "terminal-notifier"
    assert describe(NullBackend(why="running under pytest")) == "off: running under pytest"
