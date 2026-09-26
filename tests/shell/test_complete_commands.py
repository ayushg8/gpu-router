"""Completion, command parsing and the bare-`gpu` dispatch (no daemon needed)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from rich.text import Text

from gpu_router.errors import InvalidRequest
from gpu_router.models import JobState, ProviderHealth
from gpu_router.shell import commands as cmds
from gpu_router.shell.complete import Known, complete
from tests.shell.helpers import job, provider


@pytest.fixture
def known(tmp_path: Path) -> Known:
    (tmp_path / "train.py").write_text("")
    (tmp_path / "train_yolo.py").write_text("")
    (tmp_path / "notes.txt").write_text("")
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "resnet.py").write_text("")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / ".hidden.py").write_text("")
    jobs = [
        job("a7f2", "train_yolo.py"),
        job("c19e", "eval.py", JobState.AWAITING_APPROVAL),
        job("c2d4", "sweep.py", JobState.QUEUED),
    ]
    providers = [provider("kaggle"), provider("colab", health=ProviderHealth.AUTH_REQUIRED)]
    return Known(jobs=jobs, providers=providers, cwd=tmp_path)


def values(text: str, known: Known) -> list[str]:
    comp = complete(text, known)
    return [c.value for c in comp.candidates] if comp else []


def test_slash_lists_every_command(known: Known) -> None:
    got = values("/", known)
    assert got[:3] == ["/run", "/route", "/jobs"]
    assert set(got) == {f"/{c.name}" for c in cmds.COMMANDS}


def test_command_prefix_and_substring(known: Known) -> None:
    assert values("/ap", known) == ["/approve"]
    assert values("/lo", known) == ["/logs", "/login"]
    assert "/providers" in values("/vid", known)  # substring match after prefixes
    assert values("jobs", known) == []  # no popup for a plain word without a space


def test_job_ids_after_commands_that_take_one(known: Known) -> None:
    assert values("/logs ", known) == ["a7f2", "c19e", "c2d4"]
    assert values("/logs c", known) == ["c19e", "c2d4"]
    assert values("/cancel a7", known) == ["a7f2"]
    assert values("/logs a7f2 ", known) == []  # the id is already there
    assert values("gpu logs c1", known) == ["c19e"]  # pasted CLI lines work too


def test_approve_offers_only_waiting_jobs(known: Known) -> None:
    assert values("/approve ", known) == ["c19e"]
    assert values("/deny ", known) == ["c19e"]
    label = complete("/approve ", known).candidates[0].label.plain  # type: ignore[union-attr]
    assert "c19e" in label
    assert "eval.py" in label
    assert "needs approval" in label


def test_scripts_and_directories(known: Known) -> None:
    assert values("/run ", known) == ["models/", "train.py", "train_yolo.py"]
    assert values("/run tr", known) == ["train.py", "train_yolo.py"]
    assert values("/run models/", known) == ["models/resnet.py"]
    assert values("/route mo", known) == ["models/"]
    comp = complete("/run tr", known)
    assert comp is not None
    assert comp.common_prefix() == "train"
    dirs = complete("/run mo", known)
    assert dirs is not None
    assert not dirs.candidates[0].final  # keep completing inside it


def test_providers_after_login_and_dash_p(known: Known) -> None:
    assert values("/login ", known) == ["kaggle", "colab"]
    assert values("/login co", known) == ["colab"]
    assert values("/run -p k", known) == ["kaggle"]
    assert values("/run --vram ", known) == []  # a value, not a script


def test_phase5_completion_data_paths_and_policy_keys(known: Known) -> None:
    assert values("/run --data ", known) == ["models/", "notes.txt", "train.py", "train_yolo.py"]
    assert values("/run --data ds=no", known) == ["ds=notes.txt"]
    assert values("/run --smoke ", known) == ["models/", "train.py", "train_yolo.py"]
    assert values("/policy ", known) == ["show", "set", "reset"]
    assert values("/policy set agent.a", known) == [
        "agent.auto_max_hours",
        "agent.ask_providers",
        "agent.ask_secrets",  # D48
    ]
    assert "agent.enforce_hours" in values("/policy set agent.e", known)
    assert "max_quota_share" in values("/policy set ", known)
    assert values("/policy set user.auto_max_hours ", known) == []
    assert values("/quota --", known) == []


def test_split_line_and_aliases() -> None:
    cmd, _word, args = cmds.split_line("/run train.py --lr 3e-4")
    assert cmd is not None
    assert cmd.name == "run"
    assert args == ["train.py", "--lr", "3e-4"]
    assert cmds.split_line("gpu jobs --all")[0].name == "jobs"  # type: ignore[union-attr]
    assert cmds.split_line("quit")[0].name == "exit"  # type: ignore[union-attr]
    assert cmds.split_line("/nope")[0] is None
    assert cmds.suggest("/aprove") == "approve"
    with pytest.raises(cmds.UsageError):
        cmds.split_line('/run "unbalanced')


def test_shellify_rewrites_cli_hints() -> None:
    styled = Text("  gpu approve a7f2  or  gpu deny a7f2", style="dim")
    out = cmds.shellify(styled)
    assert out.plain == "  /approve a7f2  or  /deny a7f2"
    assert str(out.style) == "dim"
    assert cmds.shellify("gpu daemon start").plain == "gpu daemon start"  # not a shell command
    assert cmds.shellify("gpu logs a7f2 --follow").plain == "/logs a7f2 --follow"


def test_error_renderables() -> None:
    lines = cmds.error_renderables(InvalidRequest("'zz' is not a job id", hint="gpu jobs lists"))
    assert [t.plain for t in lines] == ["✗ 'zz' is not a job id", "  /jobs lists"]  # type: ignore[union-attr]
    usage = cmds.error_renderables(cmds.UsageError("argument --vram: invalid float value"))
    assert usage[0].plain.startswith("✗ argument --vram")  # type: ignore[union-attr]
    boom = cmds.error_renderables(RuntimeError("x"))
    assert "internal error: RuntimeError: x" in boom[0].plain  # type: ignore[union-attr]


def test_parser_raises_usage_errors() -> None:
    p = cmds.parser("jobs")
    p.add_argument("--limit", type=int)
    with pytest.raises(cmds.UsageError):
        p.parse_args(["--limit", "many"])
    with pytest.raises(cmds.UsageError):
        p.parse_args(["--bogus"])


def test_every_command_has_a_handler_and_help() -> None:
    assert set(cmds.HANDLERS) == {c.name for c in cmds.COMMANDS}
    spec_commands = {
        "run",
        "route",
        "jobs",
        "logs",
        "watch",
        "cancel",
        "fetch",
        "approve",
        "deny",
        "quota",
        "history",
        "providers",
        "login",
        "doctor",
        "policy",
        "config",
        "help",
    }
    assert spec_commands <= set(cmds.HANDLERS)
    table = cmds.help_renderable()
    from rich.console import Console

    console = Console(width=120, record=True, color_system=None)
    console.print(table)
    text = console.export_text()
    for name in spec_commands:
        assert f"/{name}" in text


# --------------------------------------------------------------------------- entry dispatch


def test_bare_gpu_without_a_terminal_prints_help(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from gpu_router import entry

    monkeypatch.setattr(sys, "argv", ["gpu"])
    monkeypatch.setattr(entry, "_is_tty", lambda: False)
    opened: list[bool] = []
    monkeypatch.setattr("gpu_router.shell.app.run", lambda: opened.append(True) or 0)
    with pytest.raises(SystemExit):
        entry.main()
    assert not opened
    assert "Run scripts on free cloud GPUs" in capsys.readouterr().out


def test_bare_gpu_in_a_terminal_opens_the_shell(monkeypatch: pytest.MonkeyPatch) -> None:
    from gpu_router import entry

    monkeypatch.setattr(sys, "argv", ["gpu"])
    monkeypatch.setattr(entry, "_is_tty", lambda: True)
    monkeypatch.setattr("gpu_router.shell.app.run", lambda: 0)
    with pytest.raises(SystemExit) as exc:
        entry.main()
    assert exc.value.code == 0


def test_is_tty_false_for_pipes(monkeypatch: pytest.MonkeyPatch) -> None:
    import io

    from gpu_router import entry

    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    assert entry._is_tty() is False


def test_bare_gpu_subprocess_without_tty_prints_help(tmp_path: Path) -> None:
    import os
    import subprocess

    env = {**os.environ, "GPU_ROUTER_HOME": str(tmp_path / "home")}
    res = subprocess.run(
        [sys.executable, "-m", "gpu_router"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    assert "Run scripts on free cloud GPUs" in res.stdout
    assert not (tmp_path / "home" / "daemon.json").exists()  # no daemon was started


def test_list_tables_keep_one_row_per_job_in_a_narrow_transcript() -> None:
    from rich.console import Console

    from gpu_router.cli import render
    from tests.shell.helpers import NOW

    jobs = [
        job(
            "e4d3",
            "train_yolo_with_a_long_name.py",
            progress={"step": 27, "total": 9000},
            last_metrics={"loss": 0.87},
        ),
        job(
            "428c",
            "eval.py",
            JobState.AWAITING_APPROVAL,
            message="waiting for your approval; run `gpu approve 428c` or `gpu deny 428c`",
        ),
    ]
    for width in (74, 120):
        table = cmds.fit_table(render.jobs_table(jobs, NOW), width)
        console = Console(width=width, record=True, color_system=None)
        console.print(table)
        lines = [ln for ln in console.export_text().splitlines() if ln.strip()]
        assert len(lines) == 3  # header + one row per job, nothing folded
        assert "needs approval" in lines[2]  # the state column is never cut
        assert all(len(ln) <= width for ln in lines)
    wide = Console(width=160, record=True, color_system=None)
    wide.print(cmds.fit_table(render.jobs_table(jobs, NOW), 160))
    assert "/approve 428c" in wide.export_text()  # CLI hints inside cells are shellified
