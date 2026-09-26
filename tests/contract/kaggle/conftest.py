"""The contract suite over KaggleAdapter (tests/contract/kaggle/test_contract.py re-runs
every shared contract test with this `target`).

- "kaggle-sim" (always): the real KaggleAdapter over SimKaggle (a simulated kaggle CLI)
  with a FakeClock; honours fake-style directives, so every contract test runs.
- "kaggle" (opt-in, real_provider): the real CLI and account. Only when
  GPU_ROUTER_REAL_PROVIDERS lists kaggle and the healthcheck passes. Each submitting test
  starts a private kernel (about a dozen, ~1 min each), so run it deliberately.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from gpu_router.adapters.base import Adapter, AttemptContext, RemoteRef
from gpu_router.clock import FakeClock, SystemClock
from gpu_router.models import Job
from gpu_router.paths import Paths
from gpu_router.providers.kaggle import remote
from gpu_router.providers.kaggle.adapter import KaggleAdapter
from tests.conftest import ENV_REAL_PROVIDERS
from tests.contract.harness import ContractTarget
from tests.contract.kaggle.sim import SimKaggle
from tests.contract.kaggle.targets import kaggle_deps
from tests.unit.packaging.helpers import isolate_git


class SimKaggleAdapter(KaggleAdapter):
    """KaggleAdapter whose submit() hands the job's fake directives to SimKaggle (the real
    adapter would reject them as unknown provider_options) and supplies a stand-in bundle
    when the contract context has none."""

    def __init__(self, sim: SimKaggle, paths: Paths) -> None:
        super().__init__(kaggle_deps(paths, sim.clock), runner=sim)
        self.sim = sim

    def submit(self, job: Job, ctx: AttemptContext) -> RemoteRef:
        options = dict(job.spec.provider_options)
        directives: dict[str, Any] = dict(options.pop(self.name, None) or {})
        per_attempt = directives.pop("attempts", None) or {}
        directives.update(per_attempt.get(str(ctx.n), {}))
        slug = remote.slug_for_key(ctx.attempt_key)
        if slug is not None and slug not in self.sim.kernels:
            self.sim.register(slug, directives)
        spec = job.spec.model_copy(update={"provider_options": options})
        job = job.model_copy(update={"spec": spec})
        if ctx.bundle_archive is None:
            stand_in = self.scratch_dir / "sim-bundle.tar.gz"
            if not stand_in.exists():
                stand_in.write_bytes(b"sim bundle, never executed\n")
            ctx = ctx.model_copy(update={"bundle_archive": stand_in})
        return super().submit(job, ctx)


class RealContractAdapter(KaggleAdapter):
    """The real adapter, plus a real tiny bundle (TRAIN_OK) when the context has none."""

    def __init__(self, paths: Paths, work: Path) -> None:
        super().__init__(kaggle_deps(paths, real=True))
        self._work = work
        self._archive: Path | None = None
        self._paths = paths

    def submit(self, job: Job, ctx: AttemptContext) -> RemoteRef:
        if ctx.bundle_archive is None:
            if self._archive is None:
                from tests.unit.providers.kaggle.helpers import make_bundle

                self._archive = make_bundle(self._work / "proj", self._paths)
            ctx = ctx.model_copy(update={"bundle_archive": self._archive})
        return super().submit(job, ctx)


def sim_target(paths: Paths) -> ContractTarget:
    clock = FakeClock()
    sim = SimKaggle(clock)
    return ContractTarget(
        name="kaggle",
        build=lambda: SimKaggleAdapter(sim, paths),
        clock=clock,
        supports_directives=True,
        step_s=0.5,
        default_timeout_s=60,
        options={"kaggle": {"duration": 5, "steps": 10}},
    )


def real_target(paths: Paths, work: Path) -> ContractTarget:
    return ContractTarget(
        name="kaggle",
        build=lambda: RealContractAdapter(paths, work),
        clock=SystemClock(),
        real=True,
        step_s=15,
        default_timeout_s=1500,
        options={"kaggle": {"timeout_s": 900}},
    )


def _params() -> list[object]:
    params: list[object] = [pytest.param("kaggle-sim", id="kaggle-sim")]
    enabled = {p.strip() for p in os.environ.get(ENV_REAL_PROVIDERS, "").split(",") if p.strip()}
    if "kaggle" in enabled:
        params.append(
            pytest.param(
                "kaggle", id="kaggle", marks=[pytest.mark.real_provider("kaggle"), pytest.mark.slow]
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
    if request.param == "kaggle-sim":
        return sim_target(paths)
    isolate_git(monkeypatch, tmp_path)
    return real_target(paths, tmp_path)


@pytest.fixture
def sim(target: ContractTarget, adapter: Adapter) -> SimKaggle:
    if not isinstance(adapter, SimKaggleAdapter):
        pytest.skip("needs the simulated kaggle")
    return adapter.sim
