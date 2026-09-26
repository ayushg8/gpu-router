"""The contract suite over LocalAdapter (tests/contract/local/test_contract.py re-runs every
shared contract test with this `target`).

- "local" (always): the real LocalAdapter running real processes on this Mac under the
  per-test tmp GPU_ROUTER_HOME, with `env: system` on the test interpreter (no venv, no
  installs, no network), so it is cheap enough for every `uv run pytest`.
- "local-venv" (opt-in, real_provider): the default settings, i.e. a uv-managed venv per
  deps key, exactly what the daemon uses. Only when GPU_ROUTER_REAL_PROVIDERS lists local
  and the healthcheck passes (it needs an Apple Silicon Mac).

Neither honours fake directives (the local provider never rate-limits, runs out of quota
or rejects a job for account reasons), so directive-only tests skip. The contract context
carries no bundle, so the adapter below supplies a real tiny one, like the engine would.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from gpu_router.adapters.base import AdapterDeps, AttemptContext, RemoteRef
from gpu_router.clock import SystemClock
from gpu_router.config import ProviderSettings
from gpu_router.models import Job
from gpu_router.paths import Paths
from gpu_router.providers.local.adapter import LocalAdapter
from tests.conftest import ENV_REAL_PROVIDERS
from tests.contract.harness import ContractTarget
from tests.unit.providers.local.helpers import LOCAL_ENTRY, build_project, make_bundle

CONTRACT_TRAIN = """
import time, gpu
gpu.total_steps(3)
for i in range(1, 4):
    time.sleep(0.2)
    gpu.log(step=i, loss=1.0 / i)
    print(f"step {i}/3", flush=True)
(gpu.output_dir() / "result.txt").write_text("ok\\n")
with gpu.atomic_checkpoint("last.txt") as tmp:
    tmp.write_text("3")
"""


class ContractLocalAdapter(LocalAdapter):
    """LocalAdapter plus a real tiny bundle when the contract context has none."""

    def __init__(self, deps: AdapterDeps, work: Path) -> None:
        super().__init__(deps)
        self._work = work
        self._archive: Path | None = None

    def submit(self, job: Job, ctx: AttemptContext) -> RemoteRef:
        if ctx.bundle_archive is None and ctx.bundle_dir is None:
            if self._archive is None:
                project = build_project(self._work, CONTRACT_TRAIN, name="contract-proj")
                self._archive = make_bundle(self.paths, project).archive
            ctx = ctx.model_copy(update={"bundle_archive": self._archive})
        return super().submit(job, ctx)


def local_target(
    paths: Paths, work: Path, settings: ProviderSettings, *, real: bool
) -> ContractTarget:
    clock = SystemClock()

    def build() -> LocalAdapter:
        deps = AdapterDeps(
            name="local", entry=LOCAL_ENTRY, settings=settings, paths=paths, clock=clock
        )
        return ContractLocalAdapter(deps, work)

    return ContractTarget(
        name="local",
        build=build,
        clock=clock,
        supports_directives=False,
        real=real,
        step_s=0.1,
        default_timeout_s=120,
    )


def _params() -> list[object]:
    params: list[object] = [pytest.param("local", id="local")]
    enabled = {p.strip() for p in os.environ.get(ENV_REAL_PROVIDERS, "").split(",") if p.strip()}
    if "local" in enabled:
        params.append(
            pytest.param(
                "local-venv",
                id="local-venv",
                marks=[pytest.mark.real_provider("local"), pytest.mark.slow],
            )
        )
    return params


@pytest.fixture(params=_params())
def target(request: pytest.FixtureRequest, paths: Paths, tmp_path: Path) -> ContractTarget:
    if request.param == "local":
        settings = ProviderSettings(env="system", python=sys.executable)
        return local_target(paths, tmp_path, settings, real=False)
    return local_target(paths, tmp_path, ProviderSettings(), real=True)
