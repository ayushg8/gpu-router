"""Phase 7-8 integration: `/login <provider>` prints the terminal login command as is.

`gpu login kaggle` (phase 8b) and `gpu login lightning` prompt for secrets in a terminal;
shellified they read `/login kaggle`, which in the shell only re-checks the provider."""

from __future__ import annotations

from typing import Any

import pytest
from rich.console import Console

from gpu_router.api import ProviderView
from gpu_router.models import ProviderHealth
from gpu_router.shell import commands
from tests.shell.helpers import provider


class _Client:
    def __init__(self, view: ProviderView) -> None:
        self.view = view

    def providers(self) -> list[ProviderView]:
        return [self.view]

    def healthcheck(self, name: str) -> ProviderView:
        return self.view


class _Host:
    def output_width(self) -> int:
        return 100


def _view(name: str) -> ProviderView:
    return provider(name, health=ProviderHealth.AUTH_REQUIRED, reason=f"{name} is not logged in")


def _render(items: list[Any]) -> str:
    console = Console(width=100, record=True)
    for item in items:
        console.print(item)
    return console.export_text()


@pytest.mark.parametrize("name", ["kaggle", "lightning"])
def test_login_steps_keep_the_terminal_command(name: str) -> None:
    out: list[Any] = []
    ctx = commands.Ctx(host=_Host(), sink=out.extend, _client=_Client(_view(name)))  # type: ignore[arg-type]
    commands.cmd_login(ctx, [name])
    text = _render(out)
    assert f"gpu login {name}" in text
    assert f"  /login {name}" not in text  # the steps are not shellified
    assert f"then /login {name} again" in text  # the shell's own hint still is
