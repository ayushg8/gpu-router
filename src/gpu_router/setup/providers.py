"""Which providers the wizard sets up (phase 8b): the enabled GPU-lane entries of
providers.yaml + config.yaml, decided like the daemon's registry (never Modal or the other
excluded services; verify-at-signup and manual entries are never set up)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gpu_router.config import Config
    from gpu_router.providers.catalog import Catalog, ProviderEntry
    from gpu_router.setup.context import SetupEnv

__all__ = ["catalog_or_none", "config_or_none", "enabled_entries", "enabled_kinds"]


def config_or_none(env: SetupEnv) -> Config | None:
    from gpu_router.config import load_config
    from gpu_router.errors import ConfigError

    try:
        return load_config(env.paths, env.environ)
    except (ConfigError, OSError):
        return None


def catalog_or_none(env: SetupEnv) -> Catalog | None:
    from gpu_router.errors import ConfigError
    from gpu_router.providers.catalog import load_catalog

    try:
        return load_catalog(env.paths.user_providers)
    except (ConfigError, OSError):
        return None


def enabled_entries(env: SetupEnv) -> list[ProviderEntry]:
    from gpu_router.adapters.registry import is_enabled

    catalog = catalog_or_none(env)
    if catalog is None:
        return []
    config = config_or_none(env)
    out: list[ProviderEntry] = []
    for entry in catalog.ordered():
        if entry.name in catalog.excluded:
            continue
        settings = config.providers.get(entry.name) if config is not None else None
        test_mode = bool(config.test_mode) if config is not None else False
        if is_enabled(entry, settings, test_mode=test_mode, environ=env.environ):
            out.append(entry)
    return out


def enabled_kinds(env: SetupEnv) -> set[str]:
    return {e.kind for e in enabled_entries(env)}
