"""The shared contract suite, run against ColabAdapter (phase 3).

The tests are the shared ones from this package, re-collected here with this module's own
`target` / `adapter` fixtures, so tests/contract/conftest.py stays untouched:

- `colab-sim` (always): the real adapter against the simulated colab CLI
  (tests/unit/providers/colab/fake_colab.py) and the real bootstrap runner.
- `colab` (only with GPU_ROUTER_REAL_PROVIDERS=colab, marker real_provider("colab")): the
  real account. Every test starts its own T4 session (~1-3 min each, about 15 sessions in
  all), so run it deliberately; teardown cancels (= stops) every session a test created.
  The two-sessions-at-once test is skipped there: the free tier gives one GPU session.

The harness hands submit() a context without a bundle (phase-1 contract), so the target's
adapter gives every attempt a small demo bundle (tests/unit/providers/colab/helpers.py).
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from gpu_router.adapters.base import Adapter, AdapterDeps, AttemptContext, RemoteRef
from gpu_router.clock import SystemClock
from gpu_router.config import ProviderSettings
from gpu_router.models import Job
from gpu_router.paths import Paths
from gpu_router.providers.catalog import load_catalog
from gpu_router.providers.colab.adapter import ColabAdapter
from tests.conftest import ENV_REAL_PROVIDERS
from tests.contract import test_submit as _submit_suite
from tests.contract.harness import ContractTarget
from tests.contract.test_errors import *  # noqa: F403 - re-collect the shared contract tests
from tests.contract.test_fetch_cancel import *  # noqa: F403
from tests.contract.test_logs import *  # noqa: F403
from tests.contract.test_quota_health import *  # noqa: F403
from tests.contract.test_status import *  # noqa: F403
from tests.contract.test_submit import *  # noqa: F403
from tests.unit.providers.colab.helpers import COLAB_ENTRY, ColabSim, make_bundle


class _ContractColab(ColabAdapter):
    """Fills in the demo bundle and remembers every run so teardown can stop it."""

    demo_archive: Path
    created: list[RemoteRef]
    sim: ColabSim | None

    def submit(self, job: Job, ctx: AttemptContext) -> RemoteRef:
        if ctx.bundle_archive is None:
            ctx = ctx.model_copy(update={"bundle_archive": self.demo_archive})
        ref = super().submit(job, ctx)
        self.created.append(ref)
        return ref


def _params() -> list[object]:
    enabled = {p.strip() for p in os.environ.get(ENV_REAL_PROVIDERS, "").split(",")}
    params: list[object] = [pytest.param("colab-sim", id="colab-sim")]
    if "colab" in enabled:
        params.append(
            pytest.param(
                "colab", id="colab", marks=[pytest.mark.real_provider("colab"), pytest.mark.slow]
            )
        )
    return params


@pytest.fixture(params=_params())
def target(request: pytest.FixtureRequest, paths: Paths, tmp_path: Path) -> ContractTarget:
    real = request.param == "colab"
    archive = make_bundle(paths, tmp_path / "projects", seconds=20 if real else 2.5, steps=4)
    sim = None if real else ColabSim(tmp_path / "colab-sim")
    entry = load_catalog(None).get("colab") if real else COLAB_ENTRY
    settings = ProviderSettings() if real else sim.settings()  # type: ignore[union-attr]
    clock = SystemClock()

    def build() -> Adapter:
        adapter = _ContractColab(
            AdapterDeps(
                name="colab",
                entry=entry,
                settings=settings,
                paths=paths,
                clock=clock,
                test_mode=True,
            )
        )
        adapter.demo_archive = archive
        adapter.created = []
        adapter.sim = sim
        if sim is not None:
            sim.bind(adapter)
        return adapter

    return ContractTarget(
        name="colab",
        build=build,
        clock=clock,
        real=real,
        step_s=5.0 if real else 0.3,
        default_timeout_s=900 if real else 60,
    )


@pytest.fixture
def adapter(target: ContractTarget) -> Iterator[Adapter]:
    a = target.build()
    assert isinstance(a, _ContractColab)
    if target.real:
        health = a.healthcheck()
        if not health.ok:
            pytest.skip(f"colab not healthy: {health.reason}")
    try:
        yield a
    finally:
        for ref in a.created:
            try:
                a.cancel(ref)  # stops the session if the test left it running
            except Exception as exc:  # teardown must reach every session
                print(f"colab teardown: could not stop {ref.remote_id}: {exc}")
        a.close()  # ends the janitor thread (D36) with the test
        if a.sim is not None:
            a.sim.kill_all()


def test_new_attempt_key_gets_new_run(target: ContractTarget, adapter: Adapter) -> None:
    if target.real:
        pytest.skip("the free tier allows one GPU session at a time")
    _submit_suite.test_new_attempt_key_gets_new_run(target, adapter)
