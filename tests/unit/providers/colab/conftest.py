"""Fixtures for the colab adapter tests: a simulated colab CLI + VM per test."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from gpu_router.clock import SystemClock
from gpu_router.paths import Paths
from gpu_router.providers.colab import adapter as colab_mod
from gpu_router.providers.colab.adapter import ColabAdapter
from tests.unit.providers.colab.helpers import ColabSim, make_adapter


@pytest.fixture(autouse=True)
def _no_janitor(monkeypatch: pytest.MonkeyPatch) -> None:
    """The janitor thread (D36) outlives a test; only the tests about it switch it on."""
    monkeypatch.setattr(colab_mod, "JANITOR_ENABLED", False)


@pytest.fixture
def sim(tmp_path: Path) -> Iterator[ColabSim]:
    s = ColabSim(tmp_path / "sim")
    yield s
    s.kill_all()


@pytest.fixture
def colab(paths: Paths, sim: ColabSim) -> Iterator[ColabAdapter]:
    adapter = make_adapter(paths, SystemClock(), sim)
    yield adapter
    adapter.close()
