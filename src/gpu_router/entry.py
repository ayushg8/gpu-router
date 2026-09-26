"""Console-script entry point for `gpu` (phase 1; owner: group C).

STDLIB ONLY AT IMPORT TIME (invariant 14). This module decides which front end handles
argv *before* anything heavy (typer, pydantic, fastapi, httpx, textual) is imported:

    gpu status --line ...   -> gpu_router.statusline.fast.render   (phase 6; stdlib only, <50ms)
    gpu daemon ...          -> gpu_router.daemon.__main__.main     (phase 1; argparse)
    gpu mcp                 -> gpu_router.mcp.server.main          (phase 6; stdio MCP server)
    gpu statusline ...      -> gpu_router.statusline.cli.main      (phase 6b; argparse)
    gpu        (a terminal) -> gpu_router.shell.app.run            (phase 4; Textual shell),
                               after the setup wizard on the first run (phase 8b)
    gpu     (no terminal)   -> `gpu --help` (a pipe, CI or an agent never gets a TUI)
    gpu <anything else>     -> gpu_router.cli.app.main             (phase 2; Typer)

Until a later phase lands, its branch degrades gracefully: the status line prints nothing
and exits 0 (a status line must never break the user's prompt), the shell falls back to
the CLI, and the CLI prints which phase it is waiting on and exits 2.
"""

import sys

EXIT_USAGE = 2


def _is_status_line(argv: list[str]) -> bool:
    """True for `gpu status --line [...]` (flag may appear anywhere after `status`)."""
    return len(argv) >= 2 and argv[0] == "status" and "--line" in argv[1:]


def _status_line(argv: list[str]) -> int:
    """Render the Claude Code status-line rows. Never raises, always exits 0."""
    try:
        from gpu_router.statusline.fast import render
    except ImportError:
        return 0
    try:
        out = render(argv[1:])
    except Exception:
        return 0
    if out:
        sys.stdout.write(out if out.endswith("\n") else out + "\n")
    return 0


def _is_tty() -> bool:
    """Both stdin and stdout are terminals: only then does bare `gpu` open the shell."""
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except (AttributeError, ValueError):  # closed or replaced streams
        return False


def _first_run() -> bool:
    """Phase 8b: before the first shell, offer `gpu setup` (setup/firstrun.py decides; it
    is stdlib only). Returns False when the user chose not to open the shell. A broken
    wizard never keeps the shell from opening."""
    try:
        from gpu_router.setup.firstrun import first_run_mode

        mode = first_run_mode()
    except Exception:
        return True
    if mode is None:
        return True
    try:
        from gpu_router.setup.wizard import first_run
    except ImportError:
        return True
    try:
        return first_run(mode)
    except KeyboardInterrupt:
        return False
    except Exception as exc:
        sys.stderr.write(f"gpu: the setup wizard failed ({exc}); run `gpu setup` to retry\n")
        return True


def main() -> None:
    """Dispatch argv to the right front end and exit with its return code."""
    argv = sys.argv[1:]
    if _is_status_line(argv):
        sys.exit(_status_line(argv))

    if argv[:1] == ["daemon"]:
        from gpu_router.daemon.__main__ import main as daemon_main

        sys.exit(daemon_main(argv[1:]))

    if argv[:1] == ["mcp"]:  # stdout is the MCP transport: no typer/rich output first
        from gpu_router.mcp.server import main as mcp_main

        sys.exit(mcp_main(argv[1:]))

    if argv[:1] == ["statusline"]:  # phase 6b: install/uninstall/preview/status (argparse)
        from gpu_router.statusline.cli import main as statusline_main

        sys.exit(statusline_main(argv[1:]))

    if not argv:
        if not _is_tty():
            argv = ["--help"]
        else:
            if not _first_run():
                sys.exit(0)
            try:
                from gpu_router.shell.app import run as shell_run
            except ImportError:
                argv = ["--help"]
            else:
                sys.exit(shell_run())

    try:
        from gpu_router.cli.app import main as cli_main
    except ImportError:
        sys.stderr.write(
            "gpu: the command-line interface lands in phase 2; "
            "for now run the daemon with `gpu daemon run`.\n"
        )
        sys.exit(EXIT_USAGE)
    sys.exit(cli_main(argv))
