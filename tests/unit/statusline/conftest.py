"""Status-line tests: reset times render in local time, so pin the zone the goldens use."""

from __future__ import annotations

import time
from collections.abc import Iterator

import pytest


@pytest.fixture(autouse=True)
def _pacific_time(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("TZ", "America/Los_Angeles")
    time.tzset()
    yield
    monkeypatch.undo()
    time.tzset()
