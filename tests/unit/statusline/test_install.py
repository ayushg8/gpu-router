"""`gpu statusline install | uninstall | status` against tmp settings files only.

The real ~/.claude/settings.json is never written: every test passes its own path, and
`test_real_settings_untouched` checks the real file's bytes did not change."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from gpu_router.statusline import install

REAL_SETTINGS = Path.home() / ".claude" / "settings.json"
ORIGINAL_LINE = {
    "type": "command",
    "command": 'bash "$HOME/.claude/statusline.sh"',
    "refreshInterval": 2,
}
_REAL_BEFORE = REAL_SETTINGS.read_bytes() if REAL_SETTINGS.is_file() else None


@pytest.fixture
def env(tmp_path: Path) -> dict[str, Path]:
    user_home = tmp_path / "user"
    settings = user_home / ".claude" / "settings.json"
    settings.parent.mkdir(parents=True)
    doc = {
        "effortLevel": "medium",
        "statusLine": ORIGINAL_LINE,
        "theme": "dark",
        "hooks": {"Stop": [{"hooks": [{"type": "command", "command": "echo done"}]}]},
    }
    settings.write_text(install.dumps(doc))
    gpu_bin = tmp_path / "bin" / "gpu"
    gpu_bin.parent.mkdir()
    gpu_bin.write_text("#!/bin/sh\nexit 0\n")
    gpu_bin.chmod(0o755)
    return {
        "settings": settings,
        "home": user_home / "Library" / "Application Support" / "gpu-router",
        "user_home": user_home,
        "gpu": gpu_bin,
    }


def _install(e: dict[str, Path], **kw: Any) -> tuple[int, str]:
    out = io.StringIO()
    kw.setdefault("interactive", True)
    code = install.run_install(
        e["settings"], e["home"], out=out, gpu_bin=e["gpu"], user_home=e["user_home"], **kw
    )
    return code, out.getvalue()


def _uninstall(e: dict[str, Path], **kw: Any) -> tuple[int, str]:
    out = io.StringIO()
    kw.setdefault("interactive", True)
    code = install.run_uninstall(e["settings"], e["home"], out=out, user_home=e["user_home"], **kw)
    return code, out.getvalue()


def _no_ask(prompt: str) -> str:
    raise AssertionError(f"asked unexpectedly: {prompt}")


def _backups(e: dict[str, Path]) -> list[Path]:
    return sorted(e["settings"].parent.glob("settings.json.gpu-router-*.bak"))


def wrapped(e: dict[str, Path], original: str | None = ORIGINAL_LINE["command"]) -> str:
    """The installed command: this settings file's record, degrading to `original`."""
    wrapper = install.record_dir(e["home"], e["settings"]) / install.WRAPPER_NAME
    return install.wrapper_command(wrapper, user_home=e["user_home"], original=original)


def folder_of(e: dict[str, Path]) -> Path:
    return install.record_dir(e["home"], e["settings"])


def test_install_shows_the_exact_diff_and_asks(env: dict[str, Path]) -> None:
    before = env["settings"].read_bytes()
    prompts: list[str] = []

    def ask(prompt: str) -> str:
        prompts.append(prompt)
        return "y"

    code, out = _install(env, ask=ask)
    assert code == 0, out
    assert prompts == [f"apply this change to {env['settings']}? [y/N] "]
    changed = [
        line
        for line in out.splitlines()
        if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
    ]
    assert changed == [
        '-    "command": "bash \\"$HOME/.claude/statusline.sh\\"",',
        f'+    "command": {json.dumps(wrapped(env))},',
    ]
    after = json.loads(env["settings"].read_text())
    assert after["statusLine"] == {**ORIGINAL_LINE, "command": wrapped(env)}
    assert wrapped(env).startswith('f="$HOME/Library/Application Support/gpu-router/statusline/')
    assert {k: v for k, v in after.items() if k != "statusLine"} == {
        k: v for k, v in json.loads(before).items() if k != "statusLine"
    }
    # only the command line differs: formatting kept
    old_lines, new_lines = before.decode().splitlines(), env["settings"].read_text().splitlines()
    assert len(old_lines) == len(new_lines)
    assert sum(a != b for a, b in zip(old_lines, new_lines, strict=True)) == 1
    (backup,) = _backups(env)
    assert backup.read_bytes() == before
    folder = folder_of(env)
    assert folder.parent == env["home"] / "statusline"
    wrapper = folder / "gpu-statusline.sh"
    assert wrapper.read_text() == install.wrapper_source()
    assert os.access(wrapper, os.X_OK)
    assert (folder / "original-command").read_text() == ORIGINAL_LINE["command"]
    record = json.loads((folder / "original.json").read_text())
    assert record["statusLine"] == ORIGINAL_LINE
    assert record["settings"] == os.path.realpath(env["settings"])
    assert (folder / "gpu-bin").read_text() == str(env["gpu"])
    assert (folder / "gpu-home").read_text() == str(env["home"])
    assert "undo: gpu statusline uninstall" in out


def test_install_without_a_terminal_refuses(env: dict[str, Path]) -> None:
    before = env["settings"].read_bytes()
    code, out = _install(env, interactive=False, ask=_no_ask)
    assert code == install.EXIT_REFUSED
    assert "--yes" in out
    assert env["settings"].read_bytes() == before
    assert _backups(env) == []
    assert not (env["home"] / "statusline").exists()


@pytest.mark.parametrize("answer", ["", "n", "no", "N", "maybe"])
def test_declining_writes_nothing(env: dict[str, Path], answer: str) -> None:
    before = env["settings"].read_bytes()
    code, out = _install(env, ask=lambda _p: answer)
    assert code == install.EXIT_ERROR
    assert "nothing written" in out
    assert env["settings"].read_bytes() == before
    assert not (env["home"] / "statusline").exists()


def test_eof_at_the_prompt_declines(env: dict[str, Path]) -> None:
    def eof(_p: str) -> str:
        raise EOFError

    assert _install(env, ask=eof)[0] == install.EXIT_ERROR


def test_yes_applies_without_asking(env: dict[str, Path]) -> None:
    code, _ = _install(env, yes=True, interactive=False, ask=_no_ask)
    assert code == 0
    assert json.loads(env["settings"].read_text())["statusLine"]["command"] == wrapped(env)


def test_dry_run_writes_nothing(env: dict[str, Path]) -> None:
    before = env["settings"].read_bytes()
    code, out = _install(env, dry_run=True, ask=_no_ask)
    assert code == 0
    assert "dry run: nothing written" in out
    assert f'+    "command": {json.dumps(wrapped(env))},' in out
    assert env["settings"].read_bytes() == before
    assert not (env["home"] / "statusline").exists()


def test_settings_changed_while_asking_aborts(env: dict[str, Path]) -> None:
    def ask(_p: str) -> str:
        doc = json.loads(env["settings"].read_text())
        doc["theme"] = "light"  # e.g. /config in another Claude Code window
        env["settings"].write_text(install.dumps(doc))
        return "y"

    code, out = _install(env, ask=ask)
    assert code == install.EXIT_ERROR
    assert "changed while you were deciding" in out
    doc = json.loads(env["settings"].read_text())
    assert doc["theme"] == "light"
    assert doc["statusLine"] == ORIGINAL_LINE
    assert not (env["home"] / "statusline").exists()


def test_install_twice_is_a_no_op_and_repairs_the_wrapper(env: dict[str, Path]) -> None:
    assert _install(env, yes=True)[0] == 0
    installed = env["settings"].read_bytes()
    wrapper = folder_of(env) / "gpu-statusline.sh"
    wrapper.write_text("#!/bin/bash\n# stale\n")
    (folder_of(env) / "original-command").unlink()
    code, out = _install(env, ask=_no_ask)
    assert code == 0
    assert "already installed" in out
    assert env["settings"].read_bytes() == installed
    assert len(_backups(env)) == 1
    assert wrapper.read_text() == install.wrapper_source()
    assert (folder_of(env) / "original-command").read_text() == ORIGINAL_LINE["command"]


def test_install_again_repoints_a_gpu_bin_that_is_gone(env: dict[str, Path]) -> None:
    assert _install(env, yes=True)[0] == 0
    (folder_of(env) / "gpu-bin").write_text("/gone/.venv/bin/gpu")
    code, out = _install(env, ask=_no_ask)
    assert code == 0
    assert "restored gpu-bin" in out
    assert (folder_of(env) / "gpu-bin").read_text() == str(env["gpu"])


def test_uninstall_restores_the_original_exactly(env: dict[str, Path]) -> None:
    original = env["settings"].read_bytes()
    assert _install(env, yes=True)[0] == 0
    prompts: list[str] = []
    code, out = _uninstall(env, ask=lambda p: prompts.append(p) or "yes")
    assert code == 0, out
    assert len(prompts) == 1
    assert "restore your original status line" in prompts[0]
    assert f'-    "command": {json.dumps(wrapped(env))},' in out
    assert env["settings"].read_bytes() == original
    assert not (env["home"] / "statusline").exists()
    assert len(_backups(env)) >= 1
    code, out = _uninstall(env, ask=_no_ask)
    assert code == 0
    assert "nothing to undo" in out


def test_uninstall_needs_confirmation_too(env: dict[str, Path]) -> None:
    assert _install(env, yes=True)[0] == 0
    installed = env["settings"].read_bytes()
    assert _uninstall(env, interactive=False, ask=_no_ask)[0] == install.EXIT_REFUSED
    assert _uninstall(env, ask=lambda _p: "n")[0] == install.EXIT_ERROR
    assert env["settings"].read_bytes() == installed
    assert _uninstall(env, dry_run=True, ask=_no_ask)[0] == 0
    assert env["settings"].read_bytes() == installed


def test_no_status_line_before(env: dict[str, Path]) -> None:
    doc = json.loads(env["settings"].read_text())
    del doc["statusLine"]
    env["settings"].write_text(install.dumps(doc))
    original = env["settings"].read_bytes()
    code, out = _install(env, yes=True)
    assert code == 0
    assert "no status line was set" in out
    line = json.loads(env["settings"].read_text())["statusLine"]
    assert line == {"type": "command", "command": wrapped(env, None), "refreshInterval": 2}
    assert (folder_of(env) / "original-command").read_text() == ""
    assert _uninstall(env, yes=True)[0] == 0
    assert env["settings"].read_bytes() == original


def test_missing_settings_file(env: dict[str, Path]) -> None:
    env["settings"].unlink()
    code, _ = _install(env, yes=True)
    assert code == 0
    assert _backups(env) == []
    assert json.loads(env["settings"].read_text())["statusLine"]["command"] == wrapped(env, None)
    assert _uninstall(env, yes=True)[0] == 0
    assert json.loads(env["settings"].read_text()) == {}


def test_symlinked_settings_stay_a_symlink(env: dict[str, Path], tmp_path: Path) -> None:
    real = tmp_path / "dotfiles" / "claude-settings.json"
    real.parent.mkdir()
    real.write_bytes(env["settings"].read_bytes())
    env["settings"].unlink()
    env["settings"].symlink_to(real)
    assert _install(env, yes=True)[0] == 0
    assert env["settings"].is_symlink()
    assert json.loads(real.read_text())["statusLine"]["command"] == wrapped(env)


def test_mode_is_kept(env: dict[str, Path]) -> None:
    env["settings"].chmod(0o600)
    assert _install(env, yes=True)[0] == 0
    assert env["settings"].stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("text", ["{not json", "[1, 2]", '{"statusLine": "bash x"}'])
def test_bad_settings_are_refused(env: dict[str, Path], text: str) -> None:
    env["settings"].write_text(text)
    code, out = _install(env, yes=True)
    assert code == install.EXIT_ERROR
    assert out.startswith("gpu statusline: ")
    assert env["settings"].read_text() == text


def test_non_command_status_line_is_refused(env: dict[str, Path]) -> None:
    env["settings"].write_text(install.dumps({"statusLine": {"type": "static", "text": "x"}}))
    code, out = _install(env, yes=True)
    assert code == install.EXIT_ERROR
    assert "not a command" in out


def test_status_reports_what_is_wrapped(env: dict[str, Path]) -> None:
    assert install.status(env["settings"], env["home"])["installed"] is False
    _install(env, yes=True)
    info = install.status(env["settings"], env["home"], user_home=env["user_home"])
    assert info["installed"] is True
    assert info["original_command"] == ORIGINAL_LINE["command"]
    assert info["wrapper_exists"] is True


def test_wrapper_command_quoting(tmp_path: Path) -> None:
    home = tmp_path / "u"
    assert install.wrapper_command(home / "a b" / "w.sh", user_home=home) == (
        'bash "$HOME/a b/w.sh"'
    )
    assert install.wrapper_command(Path("/opt/it's/w.sh"), user_home=home) == (
        "bash '/opt/it'\"'\"'s/w.sh'"
    )
    assert install.wrapper_command(home / 'q"uote' / "w.sh", user_home=home).startswith("bash '")


def test_default_settings_path_honours_claude_config_dir(tmp_path: Path) -> None:
    assert install.default_settings_path({"CLAUDE_CONFIG_DIR": str(tmp_path)}) == (
        tmp_path / "settings.json"
    )
    assert install.default_settings_path({}) == Path.home() / ".claude" / "settings.json"


def test_packaged_wrapper_is_the_plugin_copy() -> None:
    plugin = Path(__file__).parents[3] / "plugin" / "statusline" / "gpu-statusline.sh"
    assert install.wrapper_source() == plugin.read_text()
    assert os.access(plugin, os.X_OK)


def _gpu(*args: str, home: Path) -> subprocess.CompletedProcess[str]:
    gpu = Path(sys.executable).with_name("gpu")
    return subprocess.run(
        [str(gpu), *args],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        stdin=subprocess.DEVNULL,
        env={**os.environ, "GPU_ROUTER_HOME": str(home)},
    )


def test_cli_through_the_real_executable(env: dict[str, Path]) -> None:
    before = env["settings"].read_bytes()
    home = env["home"]
    proc = _gpu("statusline", "install", "--settings", str(env["settings"]), home=home)
    assert proc.returncode == install.EXIT_REFUSED, proc.stdout + proc.stderr  # no tty
    assert env["settings"].read_bytes() == before
    proc = _gpu("statusline", "install", "--dry-run", "--settings", str(env["settings"]), home=home)
    assert proc.returncode == 0
    assert "gpu-statusline.sh" in proc.stdout
    assert env["settings"].read_bytes() == before
    proc = _gpu("statusline", "status", "--json", "--settings", str(env["settings"]), home=home)
    assert json.loads(proc.stdout)["installed"] is False
    proc = _gpu("statusline", "preview", "--plain", home=home)
    assert proc.returncode == 0
    for key in ("running:", "approval:", "finished:", "migrated:", "failed:", "idle:"):
        assert f"-- {key}" in proc.stdout
    assert "\x1b[" not in proc.stdout
    proc = _gpu("statusline", "preview", "--state", "bogus", home=home)
    assert proc.returncode == 2
    proc = _gpu("--help", home=home)
    assert "statusline" in proc.stdout


def test_real_settings_untouched() -> None:
    if _REAL_BEFORE is None:
        pytest.skip("no ~/.claude/settings.json on this machine")
    assert REAL_SETTINGS.read_bytes() == _REAL_BEFORE
