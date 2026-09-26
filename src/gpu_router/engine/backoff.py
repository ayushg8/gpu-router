"""Backoff schedule (phase 1; owner: group C).

Deterministic (no jitter) so FakeClock tests can assert exact times; the per-job attempt
number already de-synchronises jobs in practice.
"""

from __future__ import annotations

#: 2**62 already dwarfs any sane cap; clamping the exponent keeps the float finite.
_MAX_EXPONENT = 62


def backoff_s(k: int, *, base_s: float, cap_s: float) -> float:
    """Delay before retry number k (k >= 1): min(base_s * 2**(k-1), cap_s).
    k <= 0 returns 0. Must not overflow for large k."""
    if k <= 0:
        return 0.0
    exponent = min(k - 1, _MAX_EXPONENT)
    return float(min(base_s * (2.0**exponent), cap_s))


def cooldown_until(
    now: float, *, k: int, base_s: float, cap_s: float, retry_after: float | None = None
) -> float:
    """Provider cooldown end after its k-th consecutive failure:
    now + max(retry_after or 0, backoff_s(k))  (errors.py table: RateLimited / Unavailable)."""
    return now + max(retry_after or 0.0, backoff_s(k, base_s=base_s, cap_s=cap_s))
