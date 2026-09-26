"""Builders for the Lightning adapter under test: the simulated SDK (always) and the real
account (live, opt-in), plus helpers shared by the unit and live tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from gpu_router.adapters.base import AdapterDeps, AttemptContext, RemoteRef
from gpu_router.clock import Clock, SystemClock
from gpu_router.config import ProviderSettings
from gpu_router.models import Job
from gpu_router.paths import Paths
from gpu_router.providers.catalog import load_catalog
from gpu_router.providers.lightning.adapter import LightningAdapter, name_for_key
from tests.contract.lightning.sim import SimLightning

#: fake credentials the simulated adapters log in with (in-memory keyring only)
SIM_USER_ID = "sim-user-0123456789"
SIM_API_KEY = "sim-key-abcdef0123456789"


def lightning_deps(
    paths: Paths, clock: Clock | None = None, *, test_mode: bool = False, **settings: Any
) -> AdapterDeps:
    entry = load_catalog().get("lightning")
    return AdapterDeps(
        name="lightning",
        entry=entry,
        settings=ProviderSettings(**settings),
        paths=paths,
        clock=clock or SystemClock(),
        test_mode=test_mode,
    )


def store_sim_credentials() -> None:
    from gpu_router import secrets

    secrets.set_secret("LIGHTNING_USER_ID", SIM_USER_ID)
    secrets.set_secret("LIGHTNING_API_KEY", SIM_API_KEY)


class SimLightningAdapter(LightningAdapter):
    """LightningAdapter over SimLightning. submit() hands the job's fake directives to the
    simulator (the real adapter would reject them as unknown provider_options) and
    supplies a stand-in bundle when the contract context has none."""

    def __init__(
        self, sim: SimLightning, paths: Paths, *, missing_file: Path | None = None, **settings: Any
    ) -> None:
        super().__init__(
            lightning_deps(paths, sim.clock, **settings),
            runner=sim,
            interpreter=["sim-python"],
            credential_file=missing_file or paths.home / "no-such-credentials.json",
        )
        self.sim = sim

    def submit(self, job: Job, ctx: AttemptContext) -> RemoteRef:
        options = dict(job.spec.provider_options)
        directives: dict[str, Any] = dict(options.pop(self.name, None) or {})
        per_attempt = directives.pop("attempts", None) or {}
        directives.update(per_attempt.get(str(ctx.n), {}))
        real_opts = {k: directives.pop(k) for k in ("machine", "timeout_s") if k in directives}
        if real_opts:
            options[self.name] = real_opts
        name = name_for_key(ctx.attempt_key)
        if name is not None and name not in self.sim.jobs:
            self.sim.register(name, directives)
        spec = job.spec.model_copy(update={"provider_options": options})
        job = job.model_copy(update={"spec": spec})
        if ctx.bundle_archive is None:
            stand_in = self.scratch_dir / "sim-bundle.tar.gz"
            if not stand_in.exists():
                stand_in.write_bytes(b"sim bundle, never executed\n")
            ctx = ctx.model_copy(update={"bundle_archive": stand_in})
        return super().submit(job, ctx)


class RealContractAdapter(LightningAdapter):
    """The real adapter and account, plus a real tiny bundle when the context has none."""

    def __init__(self, paths: Paths, work: Path) -> None:
        super().__init__(lightning_deps(paths))
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
