"""Fixtures for Kaggle adapter unit tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.unit.packaging.helpers import isolate_git


@pytest.fixture(autouse=True)
def _no_blob_sweep(monkeypatch: pytest.MonkeyPatch) -> None:
    """The background sweep of stale blob datasets is off except in its own test."""
    from gpu_router.providers.kaggle import adapter

    monkeypatch.setattr(adapter, "BLOB_SWEEP", False)


@pytest.fixture(autouse=True)
def _no_user_git_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    isolate_git(monkeypatch, tmp_path)
