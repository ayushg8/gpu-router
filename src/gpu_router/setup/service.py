"""Step 3, launchd: register the daemon's launchd agent so it starts at login (phase 8b).

Done = doctor's `daemon.launchd` row is ok (plist present, its program exists, it serves
this data dir, `launchctl print` knows it). Otherwise the wizard shows the plist path, what
it runs, where it logs and the launchctl command, asks, then calls
`daemon.launchd.install` (atomic write, boot out any old agent, bootstrap), with launchctl
going through `SetupEnv.run` so tests mock it.

The agent label is global (one per user): with a custom GPU_ROUTER_HOME the wizard does
not install it (that would take the agent away from the default data dir); it says so and
names `gpu daemon install-launchd` for whoever really wants it.

An existing plist that differs from the new one is shown as a unified diff before the
question and backed up under `<data dir>/backups/` before it is replaced (review fix: hand
edits such as an EnvironmentVariables PATH were lost silently). The printed launchctl
command uses the absolute plist path (a quoted `~` does not expand).
"""

from __future__ import annotations

import difflib
import os
import shlex
from pathlib import Path

from gpu_router.doctor.probe import home_label
from gpu_router.setup.base import Ctx, Outcome
from gpu_router.setup.ui import Mark

__all__ = ["ITEM", "run"]

ITEM = "launchd"


def run(ctx: Ctx) -> None:
    from gpu_router.daemon import launchd
    from gpu_router.doctor.checks import check_launchd
    from gpu_router.doctor.model import Status
    from gpu_router.paths import DEFAULT_HOME, ENV_HOME

    if not ctx.selected(ITEM):
        return
    env = ctx.env
    probe = env.probe()
    row = check_launchd(probe)
    if row.status is Status.OK:
        ctx.done(ITEM, Outcome.ALREADY, f"launchd: {row.summary}")
        return
    plist = env.launchd_plist
    label = home_label(plist, env.user_home)
    custom = bool(env.environ.get(ENV_HOME)) and env.paths.home.resolve() != DEFAULT_HOME.resolve()
    if row.status is Status.SKIP:  # the agent serves another data dir, or a custom home
        ctx.done(ITEM, Outcome.SKIPPED, f"launchd: {row.summary}")
        return
    if custom:
        ctx.done(
            ITEM,
            Outcome.SKIPPED,
            "launchd: GPU_ROUTER_HOME is set; the one launchd agent belongs to the default data "
            "dir, so this one starts on demand",
        )
        return
    exe = env.gpu_executable()
    if exe is None:
        ctx.done(
            ITEM,
            Outcome.FAILED,
            "launchd: cannot find the `gpu` executable for the agent to run",
            "gpu setup --only tools.gpu",
        )
        return
    uid = os.getuid()
    ctx.ui.item(Mark.SKIP, f"launchd: {row.summary}")
    ctx.ui.say(f"writes {label} and loads it:", "dim")
    ctx.ui.say(f"  runs  {exe} daemon run --launchd  at login; restarts it after a crash", "dim")
    ctx.ui.say(f"  logs  {home_label(env.paths.launchd_log, env.user_home)}", "dim")
    ctx.ui.command(f"launchctl bootstrap gui/{uid} {shlex.quote(str(plist))}")
    old = _read(plist)
    new = launchd.render_plist(
        executable=exe.resolve(), paths=env.paths, env_home=env.environ.get(ENV_HOME)
    )
    if old is not None and old != new:
        ctx.ui.say(f"it replaces the existing {label}:", "dim")
        for line in plist_diff(old, new, label):
            ctx.ui.say(f"  {line}", "dim")
    ctx.ui.say("undo: gpu daemon uninstall", "dim")
    if ctx.opts.dry_run:
        ctx.dry(ITEM, f"write {label} and load it")
        return
    answer = ctx.confirm(ITEM, "start the gpu-router daemon at login?", default=True)
    if answer is None:
        ctx.not_asked(ITEM, "launchd: agent not installed", "gpu daemon install-launchd")
        return
    if not answer:
        ctx.declined(
            ITEM, "launchd: the daemon starts on demand only", "gpu daemon install-launchd"
        )
        return
    if old is not None and old != new:
        from gpu_router.statusline.install import InstallError

        try:
            backup = backup_plist(env.paths.home, plist.name, old)
        except (OSError, InstallError) as exc:
            ctx.done(
                ITEM,
                Outcome.FAILED,
                f"launchd: could not back up the existing {label} ({exc}); nothing changed",
                "gpu daemon install-launchd",
            )
            return
        ctx.ui.say(f"backup: {home_label(backup, env.user_home)}", "dim")
    try:
        path = launchd.install(
            env.paths,
            executable=exe,
            home=env.user_home,
            launchctl=env.launchctl,
            environ=env.environ,
        )
    except (OSError, RuntimeError) as exc:
        ctx.done(
            ITEM,
            Outcome.FAILED,
            f"launchd: could not install the agent: {exc}",
            "gpu daemon install-launchd",
        )
        return
    again = check_launchd(env.probe())
    summary = f"launchd: installed {home_label(path, env.user_home)}; the daemon starts at login"
    if again.status is not Status.OK:
        summary += f" ({again.summary})"
    ctx.done(ITEM, Outcome.DONE, summary)


def _read(path: Path) -> bytes | None:
    try:
        return path.read_bytes()
    except OSError:
        return None


def plist_diff(old: bytes, new: bytes, label: str) -> list[str]:
    """Unified diff of the plist on disk and the one the wizard would write."""
    a = old.decode("utf-8", "replace").splitlines()
    b = new.decode("utf-8", "replace").splitlines()
    return list(
        difflib.unified_diff(a, b, fromfile=f"{label} (now)", tofile=f"{label} (new)", lineterm="")
    )


def backup_plist(data_home: Path, name: str, data: bytes) -> Path:
    """A new backup of the replaced plist in `<data dir>/backups/` (not LaunchAgents, where
    launchd would read it); never replaces an earlier backup."""
    from gpu_router.statusline.install import _write_backup

    folder = data_home / "backups"
    folder.mkdir(mode=0o700, parents=True, exist_ok=True)
    return _write_backup(folder / name, data, mode=0o600)
