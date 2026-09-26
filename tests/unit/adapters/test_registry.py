"""AdapterRegistry: which providers get built, lookups, poll intervals. Owner: group B."""

from __future__ import annotations

from typing import Any

import pytest

from gpu_router.adapters.base import Adapter
from gpu_router.adapters.fake import FakeAdapter
from gpu_router.adapters.registry import (
    ENV_REAL_PROVIDERS,
    AdapterRegistry,
    adapter_class,
    is_enabled,
    real_providers_opted_in,
)
from gpu_router.clock import FakeClock
from gpu_router.config import Config, ProviderSettings
from gpu_router.errors import ProviderNotFound
from gpu_router.paths import Paths
from gpu_router.providers.catalog import Catalog, ProviderEntry, load_catalog


@pytest.fixture
def cat() -> Catalog:
    return load_catalog(None)


@pytest.fixture(autouse=True)
def _no_real_provider_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV_REAL_PROVIDERS, raising=False)


def _build(cat: Catalog, paths: Paths, clock: FakeClock, **config: Any) -> AdapterRegistry:
    return AdapterRegistry.build(
        config=Config(**config),
        catalog=cat,
        paths=paths,
        clock=clock,
    )


def test_test_mode_registers_both_fakes_in_priority_order(
    cat: Catalog, paths: Paths, clock: FakeClock
) -> None:
    reg = _build(cat, paths, clock, test_mode=True)
    # Phase 1: only the fake kind has an adapter class; later kinds are skipped quietly.
    assert reg.names() == ["fake", "fake-b"]
    assert len(reg) == 2
    assert "fake" in reg
    assert all(isinstance(a, FakeAdapter) for a in reg)
    fake_b = reg.get("fake-b")
    assert fake_b.name == "fake-b"
    assert fake_b.entry.max_vram_gb == 40
    assert fake_b.capabilities.max_session_hours == 24


def test_fakes_are_hidden_outside_test_mode(cat: Catalog, paths: Paths, clock: FakeClock) -> None:
    reg = _build(cat, paths, clock, test_mode=False)
    assert "fake" not in reg
    with pytest.raises(ProviderNotFound):
        reg.get("fake")


def test_settings_can_disable_a_provider(cat: Catalog, paths: Paths, clock: FakeClock) -> None:
    reg = _build(
        cat, paths, clock, test_mode=True, providers={"fake-b": ProviderSettings(enabled=False)}
    )
    assert reg.names() == ["fake"]
    with pytest.raises(ProviderNotFound) as info:
        reg.get("fake-b")
    assert info.value.hint == "enabled providers: fake"


def test_is_enabled_rules(cat: Catalog) -> None:
    fake = cat.get("fake")
    lightning = cat.get("lightning")
    assert not is_enabled(fake, None, test_mode=False)
    assert is_enabled(fake, None, test_mode=True)
    assert not is_enabled(fake, ProviderSettings(enabled=False), test_mode=True)
    assert is_enabled(lightning, None, test_mode=False)  # on by default since phase 7a
    assert not is_enabled(lightning, ProviderSettings(enabled=False), test_mode=False)
    assert not is_enabled(lightning, None, test_mode=True)  # D29: opt-in in test mode
    manual = ProviderEntry(
        name="studiolab", kind="manual", display_name="Studio Lab", manual_only=True
    )
    assert not is_enabled(manual, ProviderSettings(enabled=True), test_mode=True)


def test_test_mode_keeps_real_providers_out_unless_opted_in(cat: Catalog) -> None:
    """Invariant 20 (phase-3 integration): local/kaggle/colab are on by default, but a
    test-mode daemon builds them only when config enables them explicitly or
    GPU_ROUTER_REAL_PROVIDERS lists them."""
    for name in ("local", "kaggle", "colab"):
        entry = cat.get(name)
        assert entry.enabled_by_default
        assert is_enabled(entry, None, test_mode=False)
        assert not is_enabled(entry, None, test_mode=True, environ={})
        assert not is_enabled(entry, ProviderSettings(), test_mode=True, environ={})
        assert is_enabled(entry, ProviderSettings(enabled=True), test_mode=True, environ={})
        opted = {ENV_REAL_PROVIDERS: f" other ,{name}"}
        assert is_enabled(entry, None, test_mode=True, environ=opted)
        # an explicit disable still wins over the opt-in
        assert not is_enabled(entry, ProviderSettings(enabled=False), test_mode=True, environ=opted)
    assert real_providers_opted_in({ENV_REAL_PROVIDERS: "local,, kaggle "}) == {"local", "kaggle"}
    assert real_providers_opted_in({}) == frozenset()


def test_test_mode_registry_builds_opted_in_real_provider(
    cat: Catalog, paths: Paths, clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ENV_REAL_PROVIDERS, "local")
    reg = _build(cat, paths, clock, test_mode=True)
    assert "local" in reg
    assert "kaggle" not in reg
    assert "colab" not in reg
    reg.close()


def test_adapter_class_lookup() -> None:
    assert adapter_class("fake") is FakeAdapter
    assert adapter_class("no-such-kind") is None
    # Kinds whose phase has not landed resolve to None instead of raising.
    for kind in ("kaggle", "colab", "local", "lightning", "modal"):
        cls = adapter_class(kind)
        assert cls is None or issubclass(cls, Adapter)


def test_poll_interval_precedence(cat: Catalog, paths: Paths, clock: FakeClock) -> None:
    reg = _build(cat, paths, clock, test_mode=True)
    assert reg.poll_interval_s("fake") == 1
    assert reg.poll_interval_s("fake", ProviderSettings(poll_interval_s=0.2)) == 0.2
    assert reg.poll_interval_s("kaggle") == 60  # catalog value even when not registered


def test_of_wraps_hand_built_adapters(cat: Catalog, paths: Paths, clock: FakeClock) -> None:
    built = _build(cat, paths, clock, test_mode=True)
    reg = AdapterRegistry.of({"fake": built.get("fake")}, cat)
    assert reg.names() == ["fake"]
    assert reg.entry("kaggle").name == "kaggle"
    reg.close()
