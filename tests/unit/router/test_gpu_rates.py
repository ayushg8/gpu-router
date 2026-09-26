"""Per-GPU quota rates (phase 7-8 integration): Lightning's L4 burns credits ~2.5x faster
than its T4, so the router, the 50% approval rule and the quota handoff must convert credits
to hours with the rate of the offer the job would run on (`quota_per_gpu_hour_by_gpu`)."""

from __future__ import annotations

import pytest

from gpu_router.models import QuotaSnapshot, QuotaUnit
from gpu_router.providers.catalog import GpuOffer
from gpu_router.quota.ledger import gpu_name, to_gpu_hours
from gpu_router.router.base import RejectCode, RouteOutcome
from tests.unit.router.test_scoring import CAT, EXTRA, NOW, P, job, rejected, route


@pytest.fixture(autouse=True)
def _lightning_with_l4(monkeypatch: pytest.MonkeyPatch) -> None:
    """D56: the packaged catalog offers Lightning T4 only (the free tier refuses L4); these
    cases are a user catalog that re-adds the L4 (a plan that runs it), priced at the
    packaged L4 rate."""
    entry = CAT.get("lightning")
    l4 = GpuOffer(name="L4", vram_gb=24, count=1)
    monkeypatch.setitem(EXTRA, "lightning", entry.model_copy(update={"gpus": [*entry.gpus, l4]}))


#: what the live balance said on the phase-7a account: 4.95 credits left this month
LEFT = 4.95


def credits(left: float = LEFT) -> QuotaSnapshot:
    return QuotaSnapshot(
        provider="lightning",
        used=5.0 - left,
        limit=5.0,
        unit=QuotaUnit.CREDITS,
        resets_at=NOW + 6 * 86400,
        source="live",
        observed_at=NOW,
    )


def lightning() -> P:
    return P("lightning", quota=credits())


def test_the_catalog_prices_lightning_l4_on_its_own() -> None:
    entry = CAT.get("lightning")
    assert to_gpu_hours(entry, 1.68, "L4") == pytest.approx(1.0)
    assert to_gpu_hours(entry, 0.68, "T4") == pytest.approx(1.0)
    assert to_gpu_hours(entry, 0.68) == pytest.approx(1.0)  # no offer named = the default
    assert to_gpu_hours(entry, 0.68, "A100") == pytest.approx(1.0)  # unpriced offer
    kaggle = CAT.get("kaggle")
    assert to_gpu_hours(kaggle, 3.0, "2xT4") == 3.0  # gpu_hours providers never convert


def test_gpu_name_strips_the_count() -> None:
    assert gpu_name("2xT4") == "T4"
    assert gpu_name("L4") == "L4"
    assert gpu_name("A100-40GB") == "A100-40GB"
    assert gpu_name("xT4") == "xT4"
    assert gpu_name(None) is None


def test_a_24gb_job_on_l4_shows_its_real_share_of_the_credits() -> None:
    d = route(job(vram_gb=24, hours=2), ("kaggle", "colab", lightning()))
    assert d.outcome is RouteOutcome.PLACE
    assert d.chosen is not None
    assert (d.chosen.provider, d.chosen.gpu) == ("lightning", "L4")
    # 2h of L4 = 3.36 credits of 4.95 (68%); at the T4 rate it looked like 27%
    assert d.chosen.quota_share == pytest.approx(2 / (LEFT / 1.68), abs=1e-3)
    assert "uses 68% of the 2.9h left" in d.chosen.reason


def test_an_l4_job_that_cannot_checkpoint_must_fit_the_l4_hours() -> None:
    # 3h without checkpoints: fits 7.3h of T4 credits, not 2.9h of L4 credits
    big = route(job(vram_gb=24, hours=3, checkpoint_interval_min=0), ("kaggle", lightning()))
    assert big.outcome is RouteOutcome.WAIT
    assert rejected(big)["lightning"] is RejectCode.QUOTA
    small = route(job(vram_gb=16, hours=3, checkpoint_interval_min=0), ("colab", lightning()))
    assert "lightning" not in rejected(small)


def test_a_pinned_l4_job_uses_the_l4_rate() -> None:
    d = route(job(provider="lightning", gpu="L4", hours=1), ("kaggle", lightning()))
    assert d.chosen is not None
    assert d.chosen.gpu == "L4"
    assert d.chosen.quota_share == pytest.approx(1 / (LEFT / 1.68), abs=1e-3)
    t4 = route(job(provider="lightning", hours=1), ("kaggle", lightning()))
    assert t4.chosen is not None
    assert t4.chosen.gpu == "T4"
    assert t4.chosen.quota_share == pytest.approx(1 / (LEFT / 0.68), abs=1e-3)
