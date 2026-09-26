from __future__ import annotations

from pathlib import Path

import pytest

from tests.unit.packaging.helpers import isolate_git


@pytest.fixture(autouse=True)
def _no_user_git_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    isolate_git(monkeypatch, tmp_path)
