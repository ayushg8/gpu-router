"""`gpu statusline install | uninstall | preview | status` (phase 6b; argparse, no typer).

entry.py dispatches here before Typer is imported (like `gpu daemon`); the Typer app only
forwards (so `gpu --help` lists it).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gpu statusline",
        description=(
            "Add gpu rows under your Claude Code status line while GPU work is active. "
            "Your own lines never change: a wrapper runs your command first."
        ),
    )
    sub = parser.add_subparsers(dest="cmd", required=True, metavar="COMMAND")

    def settings_arg(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--settings",
            metavar="PATH",
            help="settings.json to change (default: $CLAUDE_CONFIG_DIR or ~/.claude)",
        )

    p = sub.add_parser(
        "install", help="wrap your status line (shows the settings.json diff, asks first)"
    )
    p.add_argument("-y", "--yes", action="store_true", help="apply without asking (setup)")
    p.add_argument("--dry-run", action="store_true", help="show the change, write nothing")
    settings_arg(p)
    p = sub.add_parser("uninstall", help="restore your original status line (asks first)")
    p.add_argument("-y", "--yes", action="store_true", help="apply without asking")
    p.add_argument("--dry-run", action="store_true", help="show the change, write nothing")
    settings_arg(p)
    p = sub.add_parser("preview", help="sample gpu rows for every state, on your line's grid")
    p.add_argument("--plain", action="store_true", help="no colours")
    p.add_argument("--color", action="store_true", help="colours even when not a terminal")
    p.add_argument("--state", action="append", metavar="NAME", help="only this sample")
    p.add_argument("--no-line", action="store_true", help="omit the sample of your row 2")
    p = sub.add_parser("status", help="is the wrapper installed, and what does it wrap")
    p.add_argument("--json", action="store_true", help="one JSON document on stdout")
    settings_arg(p)
    return parser


def main(argv: list[str]) -> int:
    args = _parser().parse_args(argv)
    if args.cmd == "preview":
        from gpu_router.statusline import samples

        known = [s.key for s in samples.SAMPLES]
        unknown = [k for k in args.state or [] if k not in known]
        if unknown:
            sys.stderr.write(
                f"gpu statusline: unknown state {', '.join(unknown)}; one of {', '.join(known)}\n"
            )
            return 2
        color = args.color or (not args.plain and sys.stdout.isatty())
        sys.stdout.write(
            samples.preview(color=color, keys=args.state, with_line=not args.no_line) + "\n"
        )
        return 0

    from gpu_router.paths import home_from_env
    from gpu_router.statusline import install

    settings = Path(args.settings).expanduser() if args.settings else None
    settings = settings or install.default_settings_path()
    home = Path(home_from_env()).expanduser()
    if args.cmd == "install":
        return install.run_install(settings, home, yes=args.yes, dry_run=args.dry_run)
    if args.cmd == "uninstall":
        return install.run_uninstall(settings, home, yes=args.yes, dry_run=args.dry_run)
    info = install.status(settings, home)
    if args.json:
        sys.stdout.write(json.dumps(info, indent=2) + "\n")
    elif info.get("error"):
        sys.stdout.write(f"gpu statusline: {info['error']}\n")
    elif info["installed"]:
        sys.stdout.write(
            f"installed: {info['settings']} runs {info['command']}\n"
            f"wrapping: {info.get('original_command') or '(no status line of your own)'}\n"
        )
        if info.get("other_home"):
            sys.stdout.write(
                f"note: the wrapper belongs to another gpu-router data dir "
                f"({info['other_home']}); manage it with GPU_ROUTER_HOME set to that\n"
            )
    else:
        sys.stdout.write(
            f"not installed ({info['settings']}); preview with `gpu statusline preview`, "
            "add with `gpu statusline install`\n"
        )
    return 1 if info.get("error") else 0
