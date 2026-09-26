"""Builders for the Kaggle adapter under test: the real CLI (live, opt-in) and helpers
shared by the live tests."""

from __future__ import annotations

from gpu_router.adapters.base import AdapterDeps
from gpu_router.clock import Clock, SystemClock
from gpu_router.config import ProviderSettings
from gpu_router.paths import Paths
from gpu_router.providers.catalog import load_catalog
from gpu_router.providers.kaggle.adapter import KaggleAdapter
from gpu_router.providers.kaggle.cli import Runner

#: The executable an adapter over a simulated runner (SimKaggle, a patched SubprocessRunner)
#: is configured with. It is never executed; it only keeps those tests independent of
#: whether the kaggle CLI happens to be installed on the machine running them.
SIM_CLI = "/nonexistent/kaggle-sim"


def kaggle_deps(
    paths: Paths, clock: Clock | None = None, *, real: bool = False, **settings: object
) -> AdapterDeps:
    """Deps for a KaggleAdapter under test. `real=True` looks the kaggle CLI up like the
    daemon does (live, opt-in tests); otherwise `cli_path` defaults to SIM_CLI."""
    if not real:
        settings.setdefault("cli_path", SIM_CLI)
    entry = load_catalog().get("kaggle")
    return AdapterDeps(
        name="kaggle",
        entry=entry,
        settings=ProviderSettings(**settings),  # type: ignore[arg-type]
        paths=paths,
        clock=clock or SystemClock(),
    )


def real_adapter(paths: Paths, runner: Runner | None = None) -> KaggleAdapter:
    """KaggleAdapter over the real kaggle CLI and the user's existing credentials (the
    tests' in-memory keyring is empty, so "auto" falls back to the CLI's own files)."""
    return KaggleAdapter(kaggle_deps(paths, real=True), runner=runner)
