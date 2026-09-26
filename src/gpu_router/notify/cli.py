"""`gpu notify [status|test] [--json]` (phase 8a).

`status` (default): the `notifications:` settings in force and the backend this Mac would
use. `test`: sends one harmless "gpu-router test notification" from this process through
that backend (no daemon needed; no job changes). In test mode `backend: auto` means none,
so `test` says so instead of pretending.
"""

from __future__ import annotations

from typing import Annotated, Any

import typer
from rich.text import Text

__all__ = ["notify_app", "status_info"]

notify_app = typer.Typer(
    name="notify",
    help="macOS notifications for finished, failed, approval and migrated jobs.",
    invoke_without_command=True,
    add_completion=False,
    rich_markup_mode=None,
    pretty_exceptions_enable=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)

JsonOpt = Annotated[bool, typer.Option("--json", help="Machine-readable JSON on stdout.")]


def status_info() -> dict[str, Any]:
    """Settings + chosen backend (also used by `gpu doctor`). Raises ConfigError."""
    from gpu_router.config import load_config
    from gpu_router.notify.backends import NullBackend, choose_backend
    from gpu_router.notify.settings import EVENT_KINDS, notify_settings
    from gpu_router.paths import Paths

    paths = Paths.from_env()
    config = load_config(paths)
    settings = notify_settings(config.notifications, source=str(paths.config))
    backend = choose_backend(settings, test_mode=config.test_mode)
    info: dict[str, Any] = {
        "enabled": settings.enabled,
        "configured_backend": settings.backend,
        "backend": backend.name,
        "events": {k: settings.wants(k) for k in EVENT_KINDS},
        "sound": settings.sound,
        "dedupe_s": settings.dedupe_s,
        "max_per_minute": settings.max_per_minute,
        "config_path": str(paths.config),
    }
    if isinstance(backend, NullBackend):
        info["why_none"] = backend.why
    path = getattr(backend, "path", None)
    if path:
        info["backend_path"] = path
    return info


def _backend() -> Any:
    from gpu_router.config import load_config
    from gpu_router.notify.backends import choose_backend
    from gpu_router.notify.settings import notify_settings
    from gpu_router.paths import Paths

    paths = Paths.from_env()
    config = load_config(paths)
    settings = notify_settings(config.notifications, source=str(paths.config))
    return choose_backend(settings, test_mode=config.test_mode)


@notify_app.callback()
def notify_main(ctx: typer.Context, as_json: JsonOpt = False) -> None:
    """Show the notification settings (same as `gpu notify status`)."""
    if ctx.invoked_subcommand is None:
        _status(as_json)


@notify_app.command("status")
def notify_status(as_json: JsonOpt = False) -> None:
    """Notification settings and the backend this Mac uses."""
    _status(as_json)


def _status(as_json: bool) -> None:
    from gpu_router.cli.app import Out, guarded

    out = Out(as_json)
    with guarded(out):
        info = status_info()
        if as_json:
            out.emit({"notifications": info})
            return
        on = [k for k, v in info["events"].items() if v]
        off = [k for k, v in info["events"].items() if not v]
        head = Text("notifications ")
        if on and "why_none" not in info:
            head.append("on", style="green")
            head.append(f"  via {info['backend']}", style="dim")
        else:
            head.append("off", style="yellow")
            if "why_none" in info:
                head.append(f"  {info['why_none']}", style="dim")
        out.console.print(head)
        out.console.print(Text(f"  events: {', '.join(on) or 'none'}", style="dim"))
        if off:
            out.console.print(Text(f"  off: {', '.join(off)}", style="dim"))
        out.console.print(
            Text(
                f"  change them under `notifications:` in {info['config_path']}; "
                "gpu notify test shows one",
                style="dim",
            )
        )


@notify_app.command("test")
def notify_test(as_json: JsonOpt = False) -> None:
    """Send one harmless test notification through the configured backend."""
    from gpu_router.cli import exitcodes
    from gpu_router.cli.app import Out, guarded
    from gpu_router.notify.backends import NotifyError, NullBackend
    from gpu_router.notify.format import harmless_notification

    out = Out(as_json)
    with guarded(out):
        backend = _backend()
        note = harmless_notification()
        if isinstance(backend, NullBackend):
            if as_json:
                out.emit({"sent": False, "backend": "none", "why": backend.why})
            else:
                out.console.print(Text(f"nothing sent: {backend.why}", style="yellow"))
            raise typer.Exit(exitcodes.ERROR)
        try:
            backend.send(note)
        except NotifyError as exc:
            if as_json:
                out.emit({"sent": False, "backend": backend.name, "why": str(exc)})
            else:
                out.console.print(Text(f"could not show it: {exc}", style="red"))
            raise typer.Exit(exitcodes.ERROR) from None
        if as_json:
            out.emit({"sent": True, "backend": backend.name, "notification": note.to_json()})
            return
        out.console.print(Text(f"✓ sent through {backend.name}", style="green"))
        out.console.print(
            Text(
                "  nothing on screen? allow notifications for "
                + ("terminal-notifier" if backend.name == "terminal-notifier" else "Script Editor")
                + " in System Settings > Notifications",
                style="dim",
            )
        )
