"""Phase 7b catalog cleanup: Modal excluded (with the quote that decided it), verify-at-signup
and manual-only entries listed but never registered, and nothing shows Modal as available."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from gpu_router.adapters.registry import ADAPTER_KINDS, AdapterRegistry, is_enabled
from gpu_router.clock import FakeClock
from gpu_router.config import Config, ProviderSettings
from gpu_router.errors import ConfigError
from gpu_router.paths import Paths
from gpu_router.providers.catalog import load_catalog
from gpu_router.router.settings import RoutingSettings

MODAL_QUOTE = "Note that you must have a payment method on file in order to use Modal."


def test_modal_is_excluded_with_its_quote_source_and_date() -> None:
    cat = load_catalog()
    assert "modal" not in cat.providers
    modal = cat.excluded["modal"]
    assert modal.reason == "needs a card"
    assert modal.quote == MODAL_QUOTE
    assert modal.source == "https://modal.com/docs/guide/billing"
    assert modal.decided == date(2026, 9, 24)
    assert "16GB" in (modal.note or "")  # lightning's free tier refuses L4 (D56)
    # the spec's other exclusions are data too
    assert {"runpod", "lambda", "vast", "beam", "together"} <= set(cat.excluded)
    assert "modal" not in ADAPTER_KINDS


def test_a_user_file_cannot_bring_modal_back(tmp_path: Path) -> None:
    user = tmp_path / "providers.yaml"
    user.write_text(
        "providers:\n  modal:\n    kind: modal\n    display_name: Modal\n"
        "    gpus: [{name: H100, vram_gb: 80}]\n"
    )
    with pytest.raises(ConfigError, match="modal is excluded") as info:
        load_catalog(user)
    assert "needs a card" in info.value.message
    assert info.value.hint == "remove providers.modal from your providers.yaml"


def test_verify_later_entries_are_listed_with_what_to_check() -> None:
    cat = load_catalog()
    verify = {e.name: e for e in cat.listed("verify")}
    assert set(verify) == {"paperspace", "saturn"}
    for e in verify.values():
        assert e.status == "verify_at_signup"
        assert not e.enabled_by_default
        assert e.lane == "verify"
        assert e.link
        assert any("no card" in check for check in e.verify_at_signup)
        assert any("headless" in check for check in e.verify_at_signup)
    assert verify["paperspace"].session_hours == 6


def test_sagemaker_studio_lab_is_manual_only() -> None:
    cat = load_catalog()
    [manual] = cat.listed("manual")
    assert manual.name == "sagemaker_studio_lab"
    assert manual.manual_only
    assert manual.lane == "manual"
    assert [g.label for g in manual.gpus] == ["T4"]
    assert (manual.quota.limit, manual.quota.reset, manual.session_hours) == (4, "daily", 4)
    assert manual.link == "https://studiolab.sagemaker.aws/"


@pytest.mark.parametrize("name", ["paperspace", "saturn", "sagemaker_studio_lab"])
def test_listed_entries_never_register_even_when_config_enables_them(name: str) -> None:
    entry = load_catalog().get(name)
    for test_mode in (False, True):
        assert not is_enabled(entry, ProviderSettings(enabled=True), test_mode=test_mode)
        assert not is_enabled(entry, None, test_mode=test_mode)


def test_registry_and_provider_views_leave_them_out(tmp_path: Path) -> None:
    cat = load_catalog()
    config = Config(
        test_mode=False,
        providers={
            "paperspace": ProviderSettings(enabled=True),
            "saturn": ProviderSettings(enabled=True),
            "sagemaker_studio_lab": ProviderSettings(enabled=True),
            # keep real adapters from being constructed in a unit test
            "local": ProviderSettings(enabled=False),
            "kaggle": ProviderSettings(enabled=False),
            "colab": ProviderSettings(enabled=False),
            "lightning": ProviderSettings(enabled=False),
        },
    )
    reg = AdapterRegistry.build(
        config=config, catalog=cat, paths=Paths(tmp_path), clock=FakeClock()
    )
    try:
        assert not {"paperspace", "saturn", "sagemaker_studio_lab", "modal"} & set(reg.names())
    finally:
        reg.close()


def test_no_big_vram_provider_by_default() -> None:
    assert RoutingSettings().big_vram_providers == {}


def test_nothing_in_the_packaged_gpu_lane_needs_a_card() -> None:
    for entry in load_catalog().listed("gpu"):
        assert entry.card_required is False
        assert entry.name not in load_catalog().excluded
