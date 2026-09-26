"""Provider catalog loading and merging (providers/catalog.py). Owner: group B."""

from __future__ import annotations

from pathlib import Path

import pytest

from gpu_router.errors import ConfigError, ProviderNotFound
from gpu_router.models import QuotaUnit
from gpu_router.providers.catalog import CATALOG_VERSION, load_catalog, merge_raw


def test_packaged_catalog_loads_with_spec_providers() -> None:
    cat = load_catalog(None)
    assert cat.catalog_version == CATALOG_VERSION
    assert {"local", "kaggle", "colab", "lightning", "fake", "fake-b"} <= set(cat.providers)
    assert "modal" not in cat.providers  # phase 7b: dropped, it needs a card
    for name, entry in cat.providers.items():
        assert entry.name == name
        assert entry.card_required is False
    kaggle = cat.get("kaggle")
    assert kaggle.session_hours == 12
    assert kaggle.quota.limit == 30
    assert kaggle.quota.reset == "weekly"
    assert kaggle.gpus[1].label == "2xT4"
    assert cat.get("lightning").quota.unit is QuotaUnit.CREDITS
    assert cat.excluded["modal"].reason == "needs a card"
    assert cat.get("fake").test_only
    assert cat.get("fake-b").test_only
    assert cat.get("fake-b").max_vram_gb == 40


def test_ordered_is_by_priority_then_name() -> None:
    order = [e.name for e in load_catalog(None).ordered()]
    assert order[:2] == ["fake", "fake-b"]
    assert order.index("colab") < order.index("kaggle") < order.index("local")


def test_get_unknown_provider() -> None:
    with pytest.raises(ProviderNotFound) as info:
        load_catalog(None).get("runpod")
    assert info.value.hint is not None
    assert "kaggle" in info.value.hint


def test_missing_user_file_is_ignored(tmp_path: Path) -> None:
    assert load_catalog(tmp_path / "nope.yaml") == load_catalog(None)


def test_user_file_deep_merges(tmp_path: Path) -> None:
    user = tmp_path / "providers.yaml"
    user.write_text(
        "providers:\n"
        "  kaggle:\n"
        "    quota: {limit: 25}\n"
        "    gpus: [{name: T4, vram_gb: 15, count: 2}]\n"
        "  mybox:\n"
        "    kind: local\n"
        "    display_name: My box\n"
    )
    cat = load_catalog(user)
    kaggle = cat.get("kaggle")
    assert kaggle.quota.limit == 25
    assert kaggle.quota.reset == "weekly"  # untouched sibling key kept
    assert [g.label for g in kaggle.gpus] == ["2xT4"]  # lists replace
    assert cat.get("mybox").kind == "local"
    assert "colab" in cat.providers


def test_empty_user_file_is_fine(tmp_path: Path) -> None:
    user = tmp_path / "providers.yaml"
    user.write_text("# nothing yet\n")
    assert load_catalog(user) == load_catalog(None)


@pytest.mark.parametrize(
    ("body", "needle"),
    [
        ("providers: [1, 2", "not valid YAML"),
        ("- a\n- b\n", "mapping at the top level"),
        ("catalog_version: 99\n", "catalog_version 99"),
        ("providers:\n  kaggle:\n    card_required: true\n", "card"),
        ("providers:\n  kaggle:\n    max_concurrency: 0\n", "providers.kaggle.max_concurrency"),
        ("providers:\n  kaggle: 3\n", "must be a mapping"),
        ("providers:\n  x:\n    display_name: X\n", "providers.x.kind"),
    ],
)
def test_invalid_user_file_is_config_error(tmp_path: Path, body: str, needle: str) -> None:
    user = tmp_path / "providers.yaml"
    user.write_text(body)
    with pytest.raises(ConfigError) as info:
        load_catalog(user)
    assert needle in info.value.message
    assert str(user) in info.value.message


def test_merge_raw() -> None:
    base = {"a": {"b": 1, "c": [1, 2]}, "d": 1}
    out = merge_raw(base, {"a": {"c": [3], "e": 2}, "f": {"g": 1}})
    assert out == {"a": {"b": 1, "c": [3], "e": 2}, "d": 1, "f": {"g": 1}}
    assert base == {"a": {"b": 1, "c": [1, 2]}, "d": 1}  # inputs untouched
    assert merge_raw({"a": {"b": 1}}, {"a": None}) == {"a": None}
