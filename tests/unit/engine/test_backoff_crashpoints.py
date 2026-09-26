from __future__ import annotations

import subprocess
import sys

import pytest

from gpu_router.engine import crashpoints
from gpu_router.engine.backoff import backoff_s, cooldown_until


@pytest.mark.parametrize(
    ("k", "expected"),
    [(0, 0), (-3, 0), (1, 30), (2, 60), (3, 120), (6, 960), (7, 1800), (50, 1800)],
)
def test_backoff_schedule(k: int, expected: float) -> None:
    assert backoff_s(k, base_s=30, cap_s=1800) == expected


def test_backoff_huge_k_does_not_overflow() -> None:
    assert backoff_s(10**9, base_s=30, cap_s=1800) == 1800


def test_cooldown_uses_max_of_retry_after_and_backoff() -> None:
    assert cooldown_until(100, k=1, base_s=30, cap_s=1800) == 130
    assert cooldown_until(100, k=1, base_s=30, cap_s=1800, retry_after=90) == 190
    assert cooldown_until(100, k=3, base_s=30, cap_s=1800, retry_after=5) == 220


def test_crashpoints_only_armed_in_test_mode() -> None:
    try:
        crashpoints.configure(test_mode=False, environ={crashpoints.ENV_CRASH_AT: "after_fetch"})
        assert crashpoints.armed() == frozenset()
        crashpoints.crashpoint("after_fetch")  # no-op
        crashpoints.configure(
            test_mode=True, environ={crashpoints.ENV_CRASH_AT: "after_fetch, before_fetch"}
        )
        assert crashpoints.armed() == {"after_fetch", "before_fetch"}
    finally:
        crashpoints.configure(test_mode=False, environ={})


def test_unknown_crashpoint_is_loud() -> None:
    with pytest.raises(ValueError, match="nope"):
        crashpoints.configure(test_mode=True, environ={crashpoints.ENV_CRASH_AT: "nope"})


def test_crashpoint_exits_137() -> None:
    code = (
        "from gpu_router.engine import crashpoints as c\n"
        "c.configure(test_mode=True, environ={'GPU_ROUTER_CRASH_AT': 'during_cancel'})\n"
        "c.crashpoint('after_fetch')\n"
        "c.crashpoint('during_cancel')\n"
        "print('unreachable')\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert proc.returncode == 137
    assert "unreachable" not in proc.stdout
