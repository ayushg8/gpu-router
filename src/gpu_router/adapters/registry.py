"""Adapter registry: provider name -> Adapter instance (phase 1; real code; owner: group B).

Built once by the daemon from the catalog + config. A provider is registered when:
- its catalog entry is in the `gpu` lane: not `manual_only` and `status: active` (phase 7b:
  `verify_at_signup` entries are listed only, whatever config.yaml says; excluded services
  such as Modal are not providers at all),
- it is not `test_only`, unless config.test_mode,
- config.providers.<name>.enabled is True, or is None and entry.enabled_by_default,
- in test mode, a real (non-`test_only`) provider only when config.providers.<name>.enabled
  is explicitly True or GPU_ROUTER_REAL_PROVIDERS lists it (invariant 20: a test-mode daemon
  never builds, healthchecks or routes to a real provider by default; phase-3 integration),
- its `kind` has an adapter class in ADAPTER_KINDS (kinds whose phase has not landed yet
  are skipped with a DEBUG log, never an error).

Adapter classes are imported lazily by dotted path so the registry never pays for adapters
it does not build and a later phase can add a kind with one line.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING

from gpu_router.adapters.base import Adapter, AdapterDeps
from gpu_router.config import ProviderSettings
from gpu_router.errors import ProviderNotFound

if TYPE_CHECKING:
    from gpu_router.clock import Clock
    from gpu_router.config import Config
    from gpu_router.paths import Paths
    from gpu_router.providers.catalog import Catalog, ProviderEntry

#: catalog kind -> "module:Class". Add a line when a phase lands an adapter.
ADAPTER_KINDS: dict[str, str] = {
    "fake": "gpu_router.adapters.fake:FakeAdapter",
    "local": "gpu_router.adapters.local:LocalAdapter",  # phase 3
    "kaggle": "gpu_router.adapters.kaggle:KaggleAdapter",  # phase 3
    "colab": "gpu_router.adapters.colab:ColabAdapter",  # phase 3
    "lightning": "gpu_router.adapters.lightning:LightningAdapter",  # phase 7
    # no modal: dropped 2026-09-24, it needs a card (providers.yaml `excluded:`)
}


def adapter_class(kind: str) -> type[Adapter] | None:
    """Import and return the class for `kind`, or None if unknown or not yet implemented."""
    target = ADAPTER_KINDS.get(kind)
    if target is None:
        return None
    module_name, _, cls_name = target.partition(":")
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name:
            return None
        raise
    cls = getattr(module, cls_name)
    if not (isinstance(cls, type) and issubclass(cls, Adapter)):
        raise TypeError(f"{target} is not an Adapter subclass")
    return cls


#: comma-separated provider names whose real adapters may run in test mode (tests/conftest.py
#: uses the same variable to un-skip `@pytest.mark.real_provider` tests).
ENV_REAL_PROVIDERS = "GPU_ROUTER_REAL_PROVIDERS"


def real_providers_opted_in(environ: Mapping[str, str] | None = None) -> frozenset[str]:
    """Names listed in GPU_ROUTER_REAL_PROVIDERS (comma-separated, blanks ignored)."""
    raw = (os.environ if environ is None else environ).get(ENV_REAL_PROVIDERS, "")
    return frozenset(p.strip() for p in raw.split(",") if p.strip())


def is_enabled(
    entry: ProviderEntry,
    settings: ProviderSettings | None,
    *,
    test_mode: bool,
    environ: Mapping[str, str] | None = None,
) -> bool:
    if entry.lane != "gpu":  # manual_only / verify_at_signup: never registered
        return False
    if entry.test_only and not test_mode:
        return False
    explicit = settings.enabled if settings is not None else None
    if explicit is not None:
        return explicit
    if test_mode and not entry.test_only:
        # Invariant 20: real providers stay out of test-mode daemons unless opted in.
        return entry.name in real_providers_opted_in(environ)
    return entry.enabled_by_default


class AdapterRegistry:
    """Immutable after construction. Iteration order = catalog order (priority, name)."""

    def __init__(self, adapters: Mapping[str, Adapter], catalog: Catalog) -> None:
        self._adapters = dict(adapters)
        self.catalog = catalog

    @classmethod
    def build(
        cls, *, config: Config, catalog: Catalog, paths: Paths, clock: Clock
    ) -> AdapterRegistry:
        adapters: dict[str, Adapter] = {}
        for entry in catalog.ordered():
            settings = config.providers.get(entry.name)
            if not is_enabled(entry, settings, test_mode=config.test_mode):
                continue
            klass = adapter_class(entry.kind)
            if klass is None:
                continue
            adapters[entry.name] = klass(
                AdapterDeps(
                    name=entry.name,
                    entry=entry,
                    settings=settings or ProviderSettings(),
                    paths=paths,
                    clock=clock,
                    test_mode=config.test_mode,
                )
            )
        return cls(adapters, catalog)

    @classmethod
    def of(cls, adapters: Mapping[str, Adapter], catalog: Catalog) -> AdapterRegistry:
        """Tests: wrap hand-built adapters."""
        return cls(adapters, catalog)

    def get(self, name: str) -> Adapter:
        """Raises ProviderNotFound (with the registered names as hint)."""
        try:
            return self._adapters[name]
        except KeyError:
            raise ProviderNotFound(
                f"provider {name!r} is not enabled",
                hint=f"enabled providers: {', '.join(self._adapters) or 'none'}",
            ) from None

    def __contains__(self, name: object) -> bool:
        return name in self._adapters

    def __iter__(self) -> Iterator[Adapter]:
        return iter(self._adapters.values())

    def __len__(self) -> int:
        return len(self._adapters)

    def names(self) -> list[str]:
        return list(self._adapters)

    def entry(self, name: str) -> ProviderEntry:
        return self.catalog.get(name)

    def poll_interval_s(self, name: str, settings: ProviderSettings | None = None) -> float:
        """config override, else catalog, else the adapter's recommendation."""
        if settings is not None and settings.poll_interval_s is not None:
            return settings.poll_interval_s
        entry = self.catalog.providers.get(name)
        if entry is not None:
            return entry.poll_interval_s
        return self.get(name).capabilities.poll_interval_s

    def close(self) -> None:
        for adapter in self._adapters.values():
            adapter.close()
