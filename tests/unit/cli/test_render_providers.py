"""`gpu providers` in an 80-column pipe (what agents and CI see), 2026-10-04 field test:
notes were a 7th column squeezed to 4 characters a line, and a provider cooling down after
failed submits read plain "up"."""

from __future__ import annotations

from rich.console import Console

from gpu_router.adapters.base import Capabilities
from gpu_router.api import ProviderView
from gpu_router.cli import render
from gpu_router.models import ProviderHealth, ProviderState

NOW = 1_800_000_000.0


def _view(name: str, **kw: object) -> ProviderView:
    base: dict[str, object] = {
        "name": name,
        "display_name": name,
        "kind": name,
        "enabled": True,
        "health": ProviderHealth.OK,
        "capabilities": Capabilities(),
        "gpus": ["P100", "2xT4"],
        "session_hours": 12,
    }
    base.update(kw)
    return ProviderView.model_validate(base)


def _plain(renderable: object) -> str:
    console = Console(width=80, color_system=None, force_terminal=False)
    with console.capture() as cap:
        console.print(renderable)
    return cap.get()


def test_notes_wrap_on_words_under_the_table() -> None:
    reason = (
        "lightning rejected the credentials: Authentication failed. Please run "
        "`lightning login`.; re-checking in 3m"
    )
    out = _plain(
        render.providers_table(
            [
                _view("kaggle"),
                _view("lightning", health=ProviderHealth.AUTH_REQUIRED, health_reason=reason),
            ],
            NOW,
        )
    )
    lines = out.splitlines()
    assert lines[0].split() == ["provider", "status", "gpus", "session", "running", "quota"]
    assert any(line.startswith("kaggle") and "P100, 2xT4" in line for line in lines)
    assert "lightning: lightning rejected the credentials: Authentication failed." in out
    assert max(len(line) for line in lines) <= 80


def test_a_cooldown_is_named() -> None:
    state = ProviderState(
        provider="kaggle", cooldown_until=NOW + 27 * 60, consecutive_failures=3, updated_at=NOW
    )
    out = _plain(render.providers_table([_view("kaggle", state=state)], NOW))
    assert "kaggle: cooling down for 27m00s after 3 failed calls in a row" in out
    expired = state.model_copy(update={"cooldown_until": NOW - 1})
    assert "cooling" not in _plain(render.providers_table([_view("kaggle", state=expired)], NOW))
