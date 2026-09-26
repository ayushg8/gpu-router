"""`gpu statusline install | uninstall | status`: wrap the user's Claude Code status line.

Claude Code runs `statusLine.command` from settings.json on every refresh. `install`
points it at the wrapper (`gpu-statusline.sh`), which runs the ORIGINAL command with the
same stdin JSON, prints its output unchanged, and then appends `gpu status --line`.

One record per settings file (D48): `<home>/statusline/<key>/` with key = the first 12 hex
digits of sha1(realpath(settings.json)) holds that file's wrapper copy and data files
(original-command, original.json = the whole original statusLine object plus the settings
path it belongs to, gpu-bin, gpu-home). Two settings files (CLAUDE_CONFIG_DIR, --settings)
under one gpu-router home never share or overwrite each other's original.

The installed command degrades to the original on its own (D48):

    f="$HOME/.../statusline/<key>/gpu-statusline.sh"; if [ -f "$f" ]; then bash "$f"; \
else eval '<original command>'; fi

so deleting the gpu-router data dir leaves the user's own lines working, and settings.json
itself keeps the original command (uninstall falls back to it when the record is gone).

install:
  1. reads settings.json (missing = {}); anything that is not a JSON object is refused; a
     statusLine that already runs a gpu-router wrapper from another data dir or record is
     refused (uninstall that first), never wrapped twice;
  2. plans the change: every other statusLine key (refreshInterval, padding) is kept, and a
     missing statusLine gets `refreshInterval: 2` so running jobs tick; prints the diff;
  3. asks `apply this change? [y/N]` on a terminal; `--yes` answers for the setup wizard;
     no terminal and no `--yes` = refused (exit 2); `--dry-run` stops after the diff;
  4. checks the settings directory is writable, then writes the record, a backup
     `settings.json.gpu-router-<stamp>[-N].bak` next to settings.json (never overwriting
     an older backup), then settings.json atomically (tmp + fsync + rename onto the symlink
     target, same mode), after checking it did not change since the diff was shown. A
     failed write removes the record it just wrote and says nothing changed.
uninstall: only when the statusLine runs THIS home's wrapper for THIS settings file (else
  it names the data dir the command points at); restores that record's statusLine object
  exactly (or removes the key when there was none), with the same diff / confirm / backup
  / atomic rules, then removes only that record.

Never run from a build session or a test against the real ~/.claude (the tests pass
`--settings` and a tmp GPU_ROUTER_HOME).
"""

from __future__ import annotations

import contextlib
import difflib
import hashlib
import json
import os
import re
import shlex
import shutil
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

WRAPPER_NAME = "gpu-statusline.sh"
MARKER = WRAPPER_NAME  # a statusLine command containing this is ours
DEFAULT_REFRESH_S = 2  # for a brand-new statusLine (the user's own value is kept)
DATA_FILES = ("original-command", "gpu-bin", "gpu-home", "original.json")

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_REFUSED = 2


class InstallError(Exception):
    """A problem the user must fix; the message says what and what to do."""


def default_settings_path(environ: dict[str, str] | None = None) -> Path:
    """`$CLAUDE_CONFIG_DIR/settings.json` when set, else ~/.claude/settings.json."""
    env = os.environ if environ is None else environ
    base = env.get("CLAUDE_CONFIG_DIR")
    root = Path(base).expanduser() if base else Path.home() / ".claude"
    return root / "settings.json"


def statusline_dir(home: Path) -> Path:
    return home / "statusline"


def settings_key(settings_path: Path) -> str:
    """The record name of one settings file: sha1 of its real path, 12 hex digits."""
    real = os.path.realpath(os.path.expanduser(str(settings_path)))
    return hashlib.sha1(real.encode("utf-8"), usedforsecurity=False).hexdigest()[:12]


def record_dir(home: Path, settings_path: Path) -> Path:
    """`<home>/statusline/<key>/`: the wrapper and saved original for one settings file."""
    return statusline_dir(home) / settings_key(settings_path)


def wrapper_source() -> str:
    """The packaged wrapper (same bytes as plugin/statusline/gpu-statusline.sh)."""
    return (resources.files("gpu_router.statusline") / WRAPPER_NAME).read_text(encoding="utf-8")


def find_gpu_bin() -> Path | None:
    """The `gpu` console script of this installation (next to the interpreter), else PATH."""
    beside = Path(sys.executable).with_name("gpu")
    if beside.is_file() and os.access(beside, os.X_OK):
        return beside
    found = shutil.which("gpu")
    return Path(found) if found else None


def is_ours(statusline: Any) -> bool:
    return isinstance(statusline, dict) and MARKER in str(statusline.get("command", ""))


_SAFE_IN_DQ = re.compile(r'^[^"$`\\]*$')


def _path_word(path: Path, user_home: Path | None) -> str:
    """`"$HOME/..."` like the user's own command when the path allows it, else a
    shell-quoted absolute path."""
    home = user_home if user_home is not None else Path.home()
    text = str(path)
    prefix = str(home).rstrip("/") + "/"
    if text.startswith(prefix) and _SAFE_IN_DQ.match(text):
        return f'"$HOME/{text[len(prefix) :]}"'
    return shlex.quote(text)


def wrapper_command(
    wrapper: Path, *, user_home: Path | None = None, original: str | bool | None = False
) -> str:
    """The statusLine command. `original=False` (the default) is the bare `bash <wrapper>`
    form; a string (or None = there was no status line) gives the self-degrading form that
    runs the original when the wrapper file is gone (D48)."""
    word = _path_word(wrapper, user_home)
    if original is False:
        return f"bash {word}"
    fallback = "true" if not original else f"eval {shlex.quote(str(original))}"
    return f'f={word}; if [ -f "$f" ]; then bash "$f"; else {fallback}; fi'


_GUARDED = re.compile(
    r'^f=(?P<f>"[^"]*"|\'[^\']*\'|\S+); if \[ -f "\$f" \]; then bash "\$f"; '
    r"else (?P<orig>.*); fi$",
    re.S,
)


def _expand(text: str, user_home: Path | None) -> Path:
    """An unquoted path word: a leading `$HOME/` is the user's home."""
    if text.startswith("$HOME/"):
        home = user_home if user_home is not None else Path.home()
        text = str(home).rstrip("/") + text[len("$HOME") :]
    return Path(text)


def wrapper_path_of(command: Any, *, user_home: Path | None = None) -> Path | None:
    """The wrapper file a gpu-router statusLine command runs, or None."""
    if not isinstance(command, str):
        return None
    m = _GUARDED.match(command.strip())
    try:
        parts = shlex.split(m.group("f") if m is not None else command)
    except ValueError:
        return None
    if m is not None:
        return _expand(parts[0], user_home) if len(parts) == 1 else None
    if len(parts) == 2 and parts[0] == "bash" and parts[1].endswith(WRAPPER_NAME):
        return _expand(parts[1], user_home)  # the pre-D48 `bash <wrapper>` form
    return None


_NO_ORIGINAL = object()


def embedded_original(command: Any) -> Any:
    """The original command kept inside a self-degrading statusLine command: a string,
    None (there was no status line), or _NO_ORIGINAL when the command has none."""
    if not isinstance(command, str):
        return _NO_ORIGINAL
    m = _GUARDED.match(command.strip())
    if m is None:
        return _NO_ORIGINAL
    orig = m.group("orig").strip()
    if orig == "true":
        return None
    try:
        parts = shlex.split(orig)
    except ValueError:
        return _NO_ORIGINAL
    if len(parts) == 2 and parts[0] == "eval":
        return parts[1]
    return _NO_ORIGINAL


def _same(a: Path, b: Path) -> bool:
    return os.path.realpath(a) == os.path.realpath(b)


def _home_of(wrapper: Path) -> Path:
    """The gpu-router home a wrapper path belongs to (record or legacy flat layout)."""
    if wrapper.parent.parent.name == "statusline":
        return wrapper.parents[2]
    return wrapper.parents[1]


# --------------------------------------------------------------------------- settings io


@dataclass(slots=True)
class Settings:
    path: Path  # as given (may be a symlink)
    raw: bytes | None  # None = the file does not exist
    data: dict[str, Any]

    @property
    def target(self) -> Path:
        """Where writes go: the symlink's target, so a dotfiles link stays a link."""
        return self.path.resolve() if self.path.is_symlink() else self.path


def read_settings(path: Path) -> Settings:
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return Settings(path, None, {})
    except OSError as exc:
        raise InstallError(f"cannot read {path}: {exc.strerror or exc}") from None
    try:
        data = json.loads(raw.decode("utf-8")) if raw.strip() else {}
    except (UnicodeDecodeError, ValueError) as exc:
        raise InstallError(
            f"{path} is not valid JSON ({exc}); fix it (or restore a backup), then run again"
        ) from None
    if not isinstance(data, dict):
        raise InstallError(f"{path} does not hold a JSON object; nothing changed")
    return Settings(path, raw, data)


def dumps(data: dict[str, Any]) -> str:
    """Claude Code's own format: 2-space indent, UTF-8, trailing newline."""
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def unified_diff(path: Path, before: str, after: str) -> str:
    lines = difflib.unified_diff(
        before.splitlines(keepends=True),
        after.splitlines(keepends=True),
        fromfile=f"{path} (now)",
        tofile=f"{path} (after)",
        n=2,
    )
    return "".join(lines)


def _write_fd(fd: int, data: bytes) -> None:
    try:
        view = memoryview(data)
        while view:
            view = view[os.write(fd, view) :]
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_write(path: Path, data: bytes, *, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.gpu-router-tmp{os.getpid()}")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
    try:
        _write_fd(fd, data)
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def _mode_of(path: Path, default: int = 0o644) -> int:
    try:
        return path.stat().st_mode & 0o777
    except OSError:
        return default


def _stamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S", time.localtime())


def _write_backup(target: Path, data: bytes, *, mode: int) -> Path:
    """A new backup file next to `target`; never replaces an existing one (two runs in the
    same second get `-2`, `-3`, ...; D48)."""
    base = f"{target.name}.gpu-router-{_stamp()}"
    for n in range(1, 1000):
        path = target.with_name(f"{base}.bak" if n == 1 else f"{base}-{n}.bak")
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
        except FileExistsError:
            continue
        try:
            _write_fd(fd, data)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(path)
            raise
        return path
    raise InstallError(f"too many backups named {base}*.bak next to {target}")


# --------------------------------------------------------------------------- plans


@dataclass(slots=True)
class Plan:
    action: str  # "install" | "uninstall"
    settings: Settings
    new_data: dict[str, Any] | None  # None = settings.json unchanged
    diff: str
    notes: list[str]
    original: Any = None  # install: the statusLine object being wrapped
    folder: Path | None = None  # the record this plan writes or removes
    legacy: bool = False  # uninstall: the pre-D48 flat layout <home>/statusline/
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def changes_settings(self) -> bool:
        return self.new_data is not None


def _refuse_foreign(settings_path: Path, pointed: Path | None, rec: Path, action: str) -> None:
    if pointed is None:
        raise InstallError(
            f"statusLine in {settings_path} runs a gpu-router wrapper this version cannot "
            "read; restore it from a settings.json.gpu-router-*.bak backup by hand"
        )
    other = _home_of(pointed)
    if _same(statusline_dir(other), rec.parent):
        what = f"record {pointed.parent} of another settings file"
        how = "it was copied from another Claude config dir; restore statusLine by hand"
    else:
        what = f"data dir {other}"
        how = f"run `GPU_ROUTER_HOME='{other}' gpu statusline {action}`"
        if action == "install":
            how = f"run `GPU_ROUTER_HOME='{other}' gpu statusline uninstall` first"
    raise InstallError(
        f"statusLine in {settings_path} already runs gpu-router's wrapper from {what}, not "
        f"this one ({rec}); {how}"
    )


def plan_install(settings_path: Path, home: Path, *, user_home: Path | None = None) -> Plan:
    settings = read_settings(settings_path)
    current = settings.data.get("statusLine")
    if current is not None and not isinstance(current, dict):
        raise InstallError(f"statusLine in {settings_path} is not an object; nothing changed")
    rec = record_dir(home, settings_path)
    wrapper = rec / WRAPPER_NAME
    if isinstance(current, dict) and is_ours(current):
        pointed = wrapper_path_of(current.get("command"), user_home=user_home)
        if pointed is not None and _same(pointed, wrapper):
            return Plan("install", settings, None, "", ["already installed"], current, rec)
        legacy = statusline_dir(home) / WRAPPER_NAME
        if pointed is not None and _same(pointed, legacy):
            raise InstallError(
                f"{settings_path} runs the wrapper of an older gpu-router ({legacy}); run "
                "`gpu statusline uninstall`, then install again"
            )
        _refuse_foreign(settings_path, pointed, rec, "install")
    original = current
    if isinstance(original, dict) and original.get("type") not in (None, "command"):
        raise InstallError(
            f"statusLine type {original.get('type')!r} is not a command; nothing to wrap"
        )
    orig_cmd = original.get("command") if isinstance(original, dict) else None
    command = wrapper_command(
        wrapper, user_home=user_home, original=str(orig_cmd) if orig_cmd else None
    )
    notes: list[str] = []
    new_line: dict[str, Any] = dict(current) if isinstance(current, dict) else {}
    new_line["type"] = "command"
    new_line["command"] = command
    if current is None:
        new_line.setdefault("refreshInterval", DEFAULT_REFRESH_S)
        notes.append("no status line was set: the gpu rows will be the whole status line")
    new_data = dict(settings.data)
    new_data["statusLine"] = new_line
    before = settings.raw.decode("utf-8") if settings.raw is not None else ""
    return Plan(
        "install",
        settings,
        new_data,
        unified_diff(settings_path, before, dumps(new_data)),
        notes,
        original,
        rec,
    )


def _saved_original(folder: Path, settings_path: Path | None = None) -> Any:
    path = folder / "original.json"
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise InstallError(
            f"no saved original status line at {path}; restore statusLine in settings.json "
            "from a settings.json.gpu-router-*.bak backup by hand"
        ) from None
    except (OSError, ValueError) as exc:
        raise InstallError(f"cannot read {path}: {exc}") from None
    if not isinstance(doc, dict) or "statusLine" not in doc:
        raise InstallError(f"{path} is not a gpu-router status-line record")
    owner = doc.get("settings")
    foreign = isinstance(owner, str) and not _same(Path(owner), settings_path or Path(owner))
    if foreign:
        raise InstallError(
            f"{path} belongs to {owner}, not {settings_path}; nothing changed (restore "
            "statusLine by hand from a settings.json.gpu-router-*.bak backup)"
        )
    return doc["statusLine"]


def plan_uninstall(settings_path: Path, home: Path, *, user_home: Path | None = None) -> Plan:
    settings = read_settings(settings_path)
    current = settings.data.get("statusLine")
    if not is_ours(current):
        return Plan("uninstall", settings, None, "", ["not installed: nothing to undo"])
    assert isinstance(current, dict)
    command = current.get("command")
    pointed = wrapper_path_of(command, user_home=user_home)
    rec = record_dir(home, settings_path)
    legacy_wrapper = statusline_dir(home) / WRAPPER_NAME
    legacy = False
    if pointed is not None and _same(pointed, rec / WRAPPER_NAME):
        folder = rec
    elif pointed is not None and _same(pointed, legacy_wrapper):
        folder, legacy = statusline_dir(home), True
    else:
        _refuse_foreign(settings_path, pointed, rec, "uninstall")
        raise AssertionError  # unreachable: _refuse_foreign raises
    notes: list[str] = []
    if (folder / "original.json").exists():
        original = _saved_original(folder, settings_path)  # refuses another file's record
    else:
        # the record is gone (data dir deleted): settings.json kept the original command
        kept = embedded_original(command)
        if kept is _NO_ORIGINAL:
            _saved_original(folder, settings_path)  # raises: no saved original
        original = None if kept is None else {**current, "command": kept}
        notes.append(
            f"the saved record in {folder} is gone; restoring the original command kept in "
            "settings.json"
        )
    new_data = dict(settings.data)
    if original is None:
        new_data.pop("statusLine", None)
    else:
        new_data["statusLine"] = original
    before = settings.raw.decode("utf-8") if settings.raw is not None else ""
    return Plan(
        "uninstall",
        settings,
        new_data,
        unified_diff(settings_path, before, dumps(new_data)),
        notes,
        original,
        folder,
        legacy,
    )


# --------------------------------------------------------------------------- apply


def _private_dirs(folder: Path, home: Path) -> None:
    """Create `folder` and every parent below the data dir 0700 (doctor's data-dir check
    flags any subdirectory other users can open; the wrapper runs as you)."""
    folder.mkdir(parents=True, exist_ok=True)
    for d in (folder, *folder.parents):
        if d == home or home not in d.parents:
            break
        try:
            if d.stat().st_mode & 0o077:
                os.chmod(d, 0o700)
        except OSError:
            pass


def _write_record(folder: Path, original: Any, gpu_bin: Path, home: Path, settings: Path) -> Path:
    _private_dirs(folder, home)
    wrapper = folder / WRAPPER_NAME
    _atomic_write(wrapper, wrapper_source().encode("utf-8"), mode=0o755)
    command = original.get("command", "") if isinstance(original, dict) else ""
    _atomic_write(folder / "original-command", str(command).encode("utf-8"), mode=0o644)
    record = {
        "statusLine": original,
        "settings": os.path.realpath(os.path.expanduser(str(settings))),
        "saved_at": _stamp(),
        "note": "gpu statusline install",
    }
    _atomic_write(folder / "original.json", dumps(record).encode("utf-8"), mode=0o644)
    _atomic_write(folder / "gpu-bin", str(gpu_bin).encode("utf-8"), mode=0o644)
    _atomic_write(folder / "gpu-home", str(home).encode("utf-8"), mode=0o644)
    return wrapper


def _check_unchanged(plan: Plan) -> None:
    fresh = read_settings(plan.settings.path)
    if fresh.raw != plan.settings.raw:
        raise InstallError(
            f"{plan.settings.path} changed while you were deciding; nothing written, "
            "run the command again to see the new diff"
        )


def _check_writable(plan: Plan) -> None:
    """Before any file is written: settings.json's directory must take a new file (the
    atomic rename and the backup both need it)."""
    parent = plan.settings.target.parent
    probe = parent
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    if not os.access(probe, os.W_OK | os.X_OK):
        raise InstallError(
            f"cannot write to {parent} (where {plan.settings.target.name} lives, e.g. a "
            "read-only dotfiles or Nix store); nothing changed"
        )


def _write_settings(plan: Plan) -> Path | None:
    """Backup, then the atomic write. Returns the backup path (None: there was no file)."""
    assert plan.new_data is not None
    _check_unchanged(plan)
    target = plan.settings.target
    mode = _mode_of(target)
    backup: Path | None = None
    if plan.settings.raw is not None:
        backup = _write_backup(target, plan.settings.raw, mode=mode)
    _atomic_write(target, dumps(plan.new_data).encode("utf-8"), mode=mode)
    return backup


def _repair(home: Path, settings: Path, gpu_bin: Path | None, out: Any) -> None:
    """Already installed: bring the wrapper up to date and rewrite its data files from
    original.json when they went missing (settings.json is not touched)."""
    folder = record_dir(home, settings)
    wrapper = folder / WRAPPER_NAME
    current = wrapper.read_text(encoding="utf-8") if wrapper.exists() else None
    if current != wrapper_source():
        _private_dirs(folder, home)
        _atomic_write(wrapper, wrapper_source().encode("utf-8"), mode=0o755)
        out.write(f"updated {wrapper}\n")
    missing = [n for n in ("original-command", "gpu-bin", "gpu-home") if not (folder / n).exists()]
    recorded = folder / "gpu-bin"
    if recorded.exists() and not os.access(recorded.read_text(encoding="utf-8").strip(), os.X_OK):
        missing.append("gpu-bin")  # the venv it pointed at is gone (repo moved, reinstalled)
    if missing:
        try:
            original = _saved_original(folder, settings)
        except InstallError as exc:
            out.write(f"warning: {exc}\n")
            return
        binary = gpu_bin or find_gpu_bin()
        if binary is None:
            out.write("warning: cannot find the `gpu` executable; gpu rows stay off\n")
            return
        _write_record(folder, original, binary, home, settings)
        out.write(f"restored {', '.join(missing)} in {folder}\n")


def _remove_record(folder: Path, *, legacy: bool = False) -> None:
    """Remove one record (and the statusline dir once no record is left)."""
    for name in (WRAPPER_NAME, *DATA_FILES):
        with contextlib.suppress(FileNotFoundError):
            (folder / name).unlink()
    with contextlib.suppress(OSError):
        folder.rmdir()
    if not legacy:
        with contextlib.suppress(OSError):
            folder.parent.rmdir()  # only when empty: other settings files keep theirs


Ask = Callable[[str], str]


def _confirm(prompt: str, *, yes: bool, interactive: bool, ask: Ask, out: Any) -> int | None:
    """None = go ahead; else the exit code to return."""
    if yes:
        return None
    if not interactive:
        out.write(
            "not a terminal, so nothing was written; run it in a terminal to answer, "
            "or add --yes to apply without asking\n"
        )
        return EXIT_REFUSED
    try:
        answer = ask(prompt)
    except (EOFError, KeyboardInterrupt):
        answer = ""
    if answer.strip().lower() not in ("y", "yes"):
        out.write("nothing written\n")
        return EXIT_ERROR
    return None


def _os_error(exc: OSError) -> str:
    where = f" {exc.filename}" if exc.filename else ""
    return f"cannot write{where}: {exc.strerror or exc}; nothing changed in settings.json"


def run_install(
    settings_path: Path,
    home: Path,
    *,
    yes: bool = False,
    dry_run: bool = False,
    interactive: bool | None = None,
    ask: Ask = input,
    out: Any = None,
    gpu_bin: Path | None = None,
    user_home: Path | None = None,
) -> int:
    out = out if out is not None else sys.stdout
    interactive = sys.stdin.isatty() if interactive is None else interactive
    try:
        plan = plan_install(settings_path, home, user_home=user_home)
        if not plan.changes_settings:
            if not dry_run:
                _repair(home, settings_path, gpu_bin, out)
            out.write(f"already installed: {settings_path} runs {plan.original['command']}\n")
            return EXIT_OK
        binary = gpu_bin or find_gpu_bin()
        if binary is None:
            raise InstallError(
                "cannot find the `gpu` executable; install gpu-router (uv tool install) first"
            )
        assert plan.folder is not None
        for note in plan.notes:
            out.write(f"note: {note}\n")
        out.write(plan.diff)
        out.write(
            f"\nthe wrapper runs your original command unchanged, then `gpu status --line` "
            f"({binary}); files go to {plan.folder}; if they are ever deleted your own "
            "command runs alone\n"
        )
        if dry_run:
            out.write("dry run: nothing written\n")
            return EXIT_OK
        _check_writable(plan)  # before asking: a yes that cannot be applied is no use
        stop = _confirm(
            f"apply this change to {settings_path}? [y/N] ",
            yes=yes,
            interactive=interactive,
            ask=ask,
            out=out,
        )
        if stop is not None:
            return stop
        _check_unchanged(plan)  # before any file is written
        existed = plan.folder.exists()
        try:
            wrapper = _write_record(plan.folder, plan.original, binary, home, settings_path)
            backup = _write_settings(plan)
        except BaseException:
            if not existed:
                _remove_record(plan.folder)  # settings.json does not point at it
            raise
        out.write(f"installed: Claude Code now runs {wrapper}\n")
        if backup is not None:
            out.write(f"backup: {backup}\n")
        out.write("undo: gpu statusline uninstall\n")
        return EXIT_OK
    except InstallError as exc:
        out.write(f"gpu statusline: {exc}\n")
        return EXIT_ERROR
    except OSError as exc:
        out.write(f"gpu statusline: {_os_error(exc)}\n")
        return EXIT_ERROR


def run_uninstall(
    settings_path: Path,
    home: Path,
    *,
    yes: bool = False,
    dry_run: bool = False,
    interactive: bool | None = None,
    ask: Ask = input,
    out: Any = None,
    user_home: Path | None = None,
) -> int:
    out = out if out is not None else sys.stdout
    interactive = sys.stdin.isatty() if interactive is None else interactive
    try:
        plan = plan_uninstall(settings_path, home, user_home=user_home)
        if not plan.changes_settings:
            out.write(f"{plan.notes[0]} ({settings_path})\n")
            return EXIT_OK
        for note in plan.notes:
            out.write(f"note: {note}\n")
        out.write(plan.diff)
        if dry_run:
            out.write("dry run: nothing written\n")
            return EXIT_OK
        _check_writable(plan)
        stop = _confirm(
            f"restore your original status line in {settings_path}? [y/N] ",
            yes=yes,
            interactive=interactive,
            ask=ask,
            out=out,
        )
        if stop is not None:
            return stop
        backup = _write_settings(plan)
        if plan.folder is not None:
            with contextlib.suppress(OSError):  # settings.json is restored: that is what counts
                _remove_record(plan.folder, legacy=plan.legacy)
        out.write("uninstalled: your original status line is back\n")
        if backup is not None:
            out.write(f"backup: {backup}\n")
        return EXIT_OK
    except InstallError as exc:
        out.write(f"gpu statusline: {exc}\n")
        return EXIT_ERROR
    except OSError as exc:
        out.write(f"gpu statusline: {_os_error(exc)}\n")
        return EXIT_ERROR


def status(settings_path: Path, home: Path, *, user_home: Path | None = None) -> dict[str, Any]:
    """Whether the wrapper is installed, and what it wraps (for `status` and the wizard)."""
    info: dict[str, Any] = {"settings": str(settings_path), "installed": False}
    try:
        settings = read_settings(settings_path)
    except InstallError as exc:
        info["error"] = str(exc)
        return info
    current = settings.data.get("statusLine")
    info["installed"] = is_ours(current)
    info["command"] = current.get("command") if isinstance(current, dict) else None
    folder = record_dir(home, settings_path)
    pointed = wrapper_path_of(info["command"], user_home=user_home) if info["installed"] else None
    wrapper = pointed if pointed is not None else folder / WRAPPER_NAME
    info["wrapper"] = str(wrapper)
    info["wrapper_exists"] = wrapper.exists()
    if pointed is not None and not _same(pointed, folder / WRAPPER_NAME):
        info["other_home"] = str(_home_of(pointed))
    if info["installed"]:
        with contextlib.suppress(InstallError):
            original = _saved_original(wrapper.parent, settings_path)
            info["original_command"] = (
                original.get("command") if isinstance(original, dict) else None
            )
        if "original_command" not in info:
            kept = embedded_original(info["command"])
            if kept is not _NO_ORIGINAL:
                info["original_command"] = kept
    return info
