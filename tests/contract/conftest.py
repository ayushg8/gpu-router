"""Contract-suite fixtures (owner: group B).

`target` is parametrized over every adapter the suite knows: always "fake"; real providers
(phase 3+: "local", "kaggle", "colab"; phase 7: "lightning", "modal") only when listed in
GPU_ROUTER_REAL_PROVIDERS, and then skipped unless their healthcheck is OK. Adding a
provider = add a builder to REAL_BUILDERS, or (phase 3 pattern) a subdirectory suite that
re-collects the shared modules with its own `target` fixture, listed in OWN_SUITES.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator

import pytest

from gpu_router.adapters.base import Adapter, AdapterDeps
from gpu_router.clock import FakeClock, SystemClock
from gpu_router.config import ProviderSettings
from gpu_router.paths import Paths
from gpu_router.providers.catalog import GpuOffer, ProviderEntry, QuotaSpec
from tests.conftest import ENV_REAL_PROVIDERS
from tests.contract.harness import ContractTarget

FAKE_ENTRY = ProviderEntry(
    name="fake",
    kind="fake",
    display_name="Fake A",
    priority=1,
    test_only=True,
    gpus=(GpuOffer(name="T4", vram_gb=16),),
    session_hours=12,
    max_concurrency=4,
    poll_interval_s=1,
    quota=QuotaSpec(limit=30, reset="weekly"),
)


def _fake_target(paths: Paths) -> ContractTarget:
    clock = FakeClock()

    def build() -> Adapter:
        from gpu_router.adapters.fake import FakeAdapter

        return FakeAdapter(
            AdapterDeps(
                name="fake",
                entry=FAKE_ENTRY,
                settings=ProviderSettings(),
                paths=paths,
                clock=clock,
                test_mode=True,
            )
        )

    return ContractTarget(
        name="fake",
        build=build,
        clock=clock,
        supports_directives=True,
        step_s=0.5,
        default_timeout_s=60,
        options={"fake": {"duration": 5, "steps": 10}},
    )


#: provider name -> builder(paths) for real adapters (phase 3+ adds entries).
REAL_BUILDERS: dict[str, Callable[[Paths], ContractTarget]] = {}

ALL_REAL = ("local", "kaggle", "colab", "lightning", "modal")

#: providers whose contract targets (simulated + real) live in their own suite, so this
#: conftest does not parametrize them (phase-3 integration).
OWN_SUITES: dict[str, str] = {
    "local": "tests/contract/local/",
    "kaggle": "tests/contract/kaggle/",
    "colab": "tests/contract/test_colab_contract.py",
    "lightning": "tests/contract/lightning/",
}


def _params() -> list[object]:
    enabled = {p.strip() for p in os.environ.get(ENV_REAL_PROVIDERS, "").split(",") if p.strip()}
    params: list[object] = [pytest.param("fake", id="fake")]
    for name in ALL_REAL:
        if name in enabled and name not in OWN_SUITES:
            params.append(
                pytest.param(
                    name, id=name, marks=[pytest.mark.real_provider(name), pytest.mark.slow]
                )
            )
    return params


@pytest.fixture(params=_params())
def target(request: pytest.FixtureRequest, paths: Paths) -> ContractTarget:
    name: str = request.param
    if name == "fake":
        return _fake_target(paths)
    builder = REAL_BUILDERS.get(name)
    if builder is None:
        pytest.skip(f"no contract builder for {name!r} yet")
    t = builder(paths)
    assert isinstance(t.clock, SystemClock)
    return t


@pytest.fixture
def adapter(target: ContractTarget) -> Iterator[Adapter]:
    a = target.build()
    if target.real:
        health = a.healthcheck()
        if not health.ok:
            pytest.skip(f"{target.name} not healthy: {health.reason}")
    yield a
    a.close()
