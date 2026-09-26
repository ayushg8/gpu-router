from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.crash.harness import DaemonProc


@pytest.fixture
def daemon(tmp_path: Path) -> Iterator[DaemonProc]:
    d = DaemonProc(home=tmp_path / "home", project=tmp_path / "project")
    d.prepare()
    try:
        yield d
    finally:
        d.kill()
