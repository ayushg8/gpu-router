"""The contract suite over LightningAdapter (tests/contract/lightning/test_contract.py
re-runs every shared contract test with this `target`).

- "lightning-sim" (always): the real LightningAdapter over SimLightning (a simulated SDK
  driver) with a FakeClock; honours fake-style directives, so every contract test runs.
- "lightning" (opt-in, real_provider): the real SDK and account. Only when
  GPU_ROUTER_REAL_PROVIDERS lists lightning, credentials are in the environment
  (LIGHTNING_USER_ID + LIGHTNING_API_KEY) or ~/.lightning/credentials.json (the tests'
  keyring is in memory), and the healthcheck passes. Each submitting test starts a T4 job
  (~15 of them, each a few minutes and a few cents of credits), so run it deliberately.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from gpu_router.adapters.base import Adapter
from gpu_router.clock import FakeClock, SystemClock
from gpu_router.paths import Paths
from tests.conftest import ENV_REAL_PROVIDERS
from tests.contract.harness import ContractTarget
from tests.contract.lightning.sim import SimLightning
from tests.contract.lightning.targets import (
    RealContractAdapter,
    SimLightningAdapter,
    store_sim_credentials,
)
from tests.unit.packaging.helpers import isolate_git


def sim_target(paths: Paths) -> ContractTarget:
    clock = FakeClock()
    sim = SimLightning(clock)
    store_sim_credentials()
    return ContractTarget(
        name="lightning",
        build=lambda: SimLightningAdapter(sim, paths),
        clock=clock,
        supports_directives=True,
        step_s=0.5,
        default_timeout_s=60,
        options={"lightning": {"duration": 5, "steps": 10}},
    )


def real_target(paths: Paths, work: Path) -> ContractTarget:
    return ContractTarget(
        name="lightning",
        build=lambda: RealContractAdapter(paths, work),
        clock=SystemClock(),
        real=True,
        step_s=15,
        default_timeout_s=1500,
        options={"lightning": {"timeout_s": 900}},
    )


def _params() -> list[object]:
    params: list[object] = [pytest.param("lightning-sim", id="lightning-sim")]
    enabled = {p.strip() for p in os.environ.get(ENV_REAL_PROVIDERS, "").split(",") if p.strip()}
    if "lightning" in enabled:
        params.append(
            pytest.param(
                "lightning",
                id="lightning",
                marks=[pytest.mark.real_provider("lightning"), pytest.mark.slow],
            )
        )
    return params


@pytest.fixture(params=_params())
def target(
    request: pytest.FixtureRequest,
    paths: Paths,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> ContractTarget:
    if request.param == "lightning-sim":
        return sim_target(paths)
    isolate_git(monkeypatch, tmp_path)
    return real_target(paths, tmp_path)


@pytest.fixture
def sim(target: ContractTarget, adapter: Adapter) -> SimLightning:
    if not isinstance(adapter, SimLightningAdapter):
        pytest.skip("needs the simulated lightning")
    return adapter.sim
