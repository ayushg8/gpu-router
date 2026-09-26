"""`gpu daemon ...` subcommands (phase 1; owner: group C). argparse only: this runs before
typer is imported (entry.py dispatch).

    gpu daemon run [--foreground] [--port N]   run in this process until SIGTERM/SIGINT
    gpu daemon start [--json]                  start in the background, wait until ready
    gpu daemon status [--json]                 is it running? pid, port, version, ready
    gpu daemon stop [--json]                   POST /v1/daemon/shutdown, wait for exit
    gpu daemon install [--print] [--this-home] [--json]   write + load the launchd agent
                                               (launchd.py); refused with a custom
                                               GPU_ROUTER_HOME unless --this-home
    gpu daemon install-launchd [...]           same as install
    gpu daemon uninstall [--json]              unload + remove the launchd agent

Every subcommand except `run` takes --json: one JSON document on stdout, errors as
{"error": {code, message, hint, detail}} (same envelope as the rest of the CLI).

Exit codes: 0 ok, 1 error (message on stderr), 2 usage, 3 already running (run), not
running (status/stop), or no ready daemon after `start`. `run --launchd` (the launchd
agent) exits 0 instead when another daemon holds the lock or the port is taken, so
KeepAlive does not respawn it in a loop.

`python -m gpu_router.daemon ...` runs the same subcommands.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gpu_router.paths import Paths

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2
EXIT_STATE = 3


class _Usage(Exception):
    def __init__(self, parser: argparse.ArgumentParser, message: str) -> None:
        super().__init__(message)
        self.parser = parser
        self.message = message


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:  # type: ignore[override]
        raise _Usage(self, message)


def _parser() -> argparse.ArgumentParser:
    p = _Parser(prog="gpu daemon", description="Run and manage the gpu-router daemon.")
    sub = p.add_subparsers(dest="cmd", required=True, metavar="COMMAND")
    run = sub.add_parser("run", help="run the daemon in this process")
    run.add_argument(
        "--foreground", action="store_true", help="also log to stderr in a human format"
    )
    run.add_argument(
        "--launchd",
        action="store_true",
        help=argparse.SUPPRESS,  # set by the launchd agent: "already running" is not a crash
    )
    run.add_argument(
        "--port",
        type=int,
        default=None,
        help="listen port (0 = pick a free one; default from config)",
    )
    start = sub.add_parser("start", help="start the daemon in the background")
    start.add_argument("--json", action="store_true", help="machine-readable output")
    start.add_argument(
        "--wait", type=float, default=20.0, metavar="S", help="seconds to wait for it (20)"
    )
    st = sub.add_parser("status", help="show whether the daemon is running")
    st.add_argument("--json", action="store_true", help="machine-readable output")
    stop = sub.add_parser("stop", help="stop the running daemon (remote runs keep going)")
    stop.add_argument("--json", action="store_true", help="machine-readable output")
    for name in ("install", "install-launchd"):
        inst = sub.add_parser(name, help="install and load the launchd agent (start at login)")
        inst.add_argument(
            "--print",
            dest="print_only",
            action="store_true",
            help="print the launchd plist instead of installing it",
        )
        inst.add_argument(
            "--this-home",
            dest="this_home",
            action="store_true",
            help="with GPU_ROUTER_HOME set: hand the one launchd agent to that data dir",
        )
        inst.add_argument("--json", action="store_true", help="machine-readable output")
    uninst = sub.add_parser("uninstall", help="unload and remove the launchd agent")
    uninst.add_argument("--json", action="store_true", help="machine-readable output")
    return p


def _fail(
    message: str,
    hint: str | None = None,
    code: int = EXIT_ERROR,
    *,
    as_json: bool = False,
    error_code: str = "internal",
) -> int:
    if as_json:
        body: dict[str, object] = {
            "code": error_code,
            "message": message,
            "hint": hint,
            "detail": {},
        }
        _emit({"error": body})
        return code
    sys.stderr.write(f"gpu: {message}\n")
    if hint:
        sys.stderr.write(f"  {hint}\n")
    return code


def _emit(data: object) -> None:
    sys.stdout.write(json.dumps(data) + "\n")
    sys.stdout.flush()


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _cmd_run(args: argparse.Namespace) -> int:
    from gpu_router.config import load_config
    from gpu_router.daemon.server import run_daemon
    from gpu_router.errors import DaemonAlreadyRunning
    from gpu_router.paths import Paths

    paths = Paths.from_env()
    config = load_config(paths)
    # Everything the daemon creates (gpu.db-wal/-shm, backups, logs, state.json) is private.
    old_umask = os.umask(0o077)
    try:
        return run_daemon(paths, config, foreground=args.foreground, port=args.port)
    except DaemonAlreadyRunning as exc:
        # Under launchd (KeepAlive SuccessfulExit=false) a non-zero exit means "crashed,
        # restart me": another daemon already serving is success, not an endless respawn.
        code = EXIT_OK if args.launchd else EXIT_STATE
        return _fail(exc.message, exc.hint or "stop it with `gpu daemon stop`", code)
    except OSError as exc:
        if not args.launchd:
            raise
        # port taken: respawning every throttle interval cannot fix it; `gpu` kickstarts
        # the agent again when it needs the daemon.
        return _fail(str(exc), "free the port or change daemon.port in config.toml", EXIT_OK)
    finally:
        os.umask(old_umask)


def _cmd_status(args: argparse.Namespace) -> int:
    from gpu_router.client import GpuClient
    from gpu_router.errors import DaemonUnavailable
    from gpu_router.paths import Paths

    paths = Paths.from_env()
    try:
        with GpuClient.from_env(paths, client_name="cli", timeout_s=3.0) as client:
            health = client.health()
            port = int(client.base_url.rsplit(":", 1)[1])
    except DaemonUnavailable as exc:
        if args.json:
            sys.stdout.write(json.dumps({"running": False, "message": exc.message}) + "\n")
            return EXIT_STATE
        return _fail(exc.message, exc.hint, EXIT_STATE)
    info = {
        "running": True,
        "pid": health.pid,
        "port": port,
        "version": health.version,
        "ready": health.ready,
        "test_mode": health.test_mode,
        "started_at": health.model_dump(mode="json")["started_at"],
    }
    if args.json:
        sys.stdout.write(json.dumps(info) + "\n")
    else:
        ready = "ready" if health.ready else "recovering jobs"
        mode = " (test mode)" if health.test_mode else ""
        sys.stdout.write(
            f"daemon running{mode}: pid {health.pid}, port {port}, v{health.version}, {ready}\n"
        )
    return EXIT_OK


def _cmd_stop(args: argparse.Namespace) -> int:
    from gpu_router.client import GpuClient
    from gpu_router.config import load_config
    from gpu_router.errors import DaemonUnavailable
    from gpu_router.paths import Paths

    as_json = bool(getattr(args, "json", False))
    paths = Paths.from_env()
    try:
        with GpuClient.from_env(paths, client_name="cli", timeout_s=5.0) as client:
            pid = client.health().pid
            client.shutdown()
    except DaemonUnavailable as exc:
        if as_json:
            _emit({"running": False, "stopped": False, "message": exc.message})
            return EXIT_STATE
        return _fail(exc.message, exc.hint, EXIT_STATE)
    grace = load_config(paths).daemon.shutdown_grace_s
    deadline = time.monotonic() + grace + 10
    while time.monotonic() < deadline:
        if not _pid_alive(pid) or _exited_cleanly(paths, pid):
            if as_json:
                _emit({"stopped": True, "pid": pid})
            else:
                sys.stdout.write(f"daemon stopped (pid {pid}); remote runs keep going\n")
            return EXIT_OK
        time.sleep(0.1)
    return _fail(
        f"daemon (pid {pid}) did not exit within {grace + 10:g}s",
        f"check `ps -p {pid}`; `kill {pid}` stops it without touching remote runs",
        as_json=as_json,
    )


def _exited_cleanly(paths: Paths, pid: int) -> bool:
    """daemon.json is removed as the very last step of a clean exit; a pid that still
    answers kill(0) afterwards is a zombie waiting for its parent to reap it."""
    from gpu_router.client import read_runtime_info

    info = read_runtime_info(paths)
    return info is None or info.pid != pid


def _executable() -> Path:
    found = shutil.which("gpu")
    if found:
        return Path(found)
    return Path(sys.argv[0]).resolve()


def _cmd_start(args: argparse.Namespace) -> int:
    from gpu_router.daemon.spawn import connect
    from gpu_router.errors import DaemonUnavailable, NotReady
    from gpu_router.paths import Paths

    paths = Paths.from_env()
    if not args.json:
        sys.stderr.write("gpu: starting the gpu-router daemon...\n")
    try:
        conn = connect(paths, wait_s=args.wait)
    except (DaemonUnavailable, NotReady) as exc:
        if args.json:
            sys.stdout.write(json.dumps({"error": exc.to_body()}) + "\n")
            return EXIT_STATE
        return _fail(exc.message, exc.hint, EXIT_STATE)
    conn.client.close()
    info = {"running": True, "started": conn.started, "pid": conn.pid, "port": conn.port}
    if args.json:
        sys.stdout.write(json.dumps(info) + "\n")
    elif conn.started:
        sys.stdout.write(f"daemon started: pid {conn.pid}, port {conn.port}\n")
    else:
        sys.stdout.write(f"daemon already running: pid {conn.pid}, port {conn.port}\n")
    return EXIT_OK


def _cmd_install(args: argparse.Namespace) -> int:
    from gpu_router.daemon import launchd
    from gpu_router.paths import ENV_HOME, Paths

    as_json = bool(getattr(args, "json", False))
    if getattr(args, "print_only", False):
        data = launchd.render_plist(
            executable=_executable().resolve(),
            paths=Paths.from_env(),
            env_home=os.environ.get(ENV_HOME),
        )
        if as_json:
            _emit({"installed": False, "plist": data.decode("utf-8")})
        else:
            sys.stdout.write(data.decode("utf-8"))
        return EXIT_OK
    from gpu_router.paths import DEFAULT_HOME

    paths = Paths.from_env()
    custom = bool(os.environ.get(ENV_HOME)) and paths.home.resolve() != DEFAULT_HOME.resolve()
    if custom and not getattr(args, "this_home", False):
        # the label is global (one agent per user): installing here would take it away
        # from the default data dir (review fix)
        return _fail(
            f"GPU_ROUTER_HOME is set to {paths.home}; the launchd agent is global and would "
            "serve that dir instead of the default one, so nothing was installed",
            "unset GPU_ROUTER_HOME, or add --this-home to hand the agent to this dir",
            as_json=as_json,
        )
    try:
        path = launchd.install(paths, executable=_executable())
    except (OSError, RuntimeError) as exc:
        return _fail(f"could not install the launchd agent: {exc}", as_json=as_json)
    if as_json:
        _emit({"installed": True, "path": str(path)})
    else:
        sys.stdout.write(f"installed {path}; the daemon now starts at login\n")
    return EXIT_OK


def _cmd_uninstall(args: argparse.Namespace) -> int:
    from gpu_router.daemon import launchd

    as_json = bool(getattr(args, "json", False))
    try:
        removed = launchd.uninstall()
    except OSError as exc:
        return _fail(f"could not remove the launchd agent: {exc}", as_json=as_json)
    if as_json:
        _emit({"removed": bool(removed)})
    else:
        sys.stdout.write(
            "launchd agent removed\n" if removed else "launchd agent was not installed\n"
        )
    return EXIT_OK


_COMMANDS = {
    "run": _cmd_run,
    "start": _cmd_start,
    "status": _cmd_status,
    "stop": _cmd_stop,
    "install": _cmd_install,
    "install-launchd": _cmd_install,
    "uninstall": _cmd_uninstall,
}


def main(argv: list[str]) -> int:
    """Parse argv (everything after `daemon`) and dispatch. Never raises; errors print
    `gpu: <message>` (+ hint) to stderr (or the JSON envelope on stdout with --json) and
    return a non-zero code."""
    as_json = "--json" in argv
    try:
        args = _parser().parse_args(argv)
    except _Usage as exc:
        if as_json:
            return _fail(
                f"gpu daemon: {exc.message}",
                "see `gpu daemon --help`",
                EXIT_USAGE,
                as_json=True,
                error_code="invalid_request",
            )
        exc.parser.print_usage(sys.stderr)
        sys.stderr.write(f"gpu daemon: {exc.message}\n")
        return EXIT_USAGE
    except SystemExit as exc:  # --help
        return int(exc.code) if isinstance(exc.code, int) else EXIT_USAGE
    try:
        return _COMMANDS[args.cmd](args)
    except KeyboardInterrupt:
        return EXIT_OK
    except Exception as exc:
        from gpu_router.errors import GpuRouterError

        if isinstance(exc, GpuRouterError):
            return _fail(exc.message, exc.hint, as_json=as_json, error_code=str(exc.code))
        return _fail(f"{type(exc).__name__}: {exc}", as_json=as_json)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
