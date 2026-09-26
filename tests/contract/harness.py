"""Contract-suite harness (owner: group B).

A `ContractTarget` knows how to build one adapter under test and how to let time pass for
it. The fake uses a FakeClock (instant, deterministic); a real provider (phase 3+) uses the
SystemClock and real waiting, and is only collected when GPU_ROUTER_REAL_PROVIDERS lists it
and its healthcheck passes.

Every contract test takes the `target` + `adapter` fixtures (tests/contract/conftest.py),
builds jobs with `make_job()` and contexts with `make_ctx()`, and waits with
`target.wait_for_phase()`. Tests that rely on fake directives (forcing errors) check
`target.supports_directives` and skip otherwise.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Collection
from dataclasses import dataclass, field
from typing import Any

from gpu_router.adapters.base import Adapter, AttemptContext, RemotePhase, RemoteRef, RemoteStatus
from gpu_router.clock import FakeClock, SystemClock
from gpu_router.ids import JOB_ID_LEN
from gpu_router.models import Job, JobSpec, JobState, Source


@dataclass
class ContractTarget:
    name: str  # provider name under test
    build: Callable[[], Adapter]  # fresh adapter for one test
    clock: FakeClock | SystemClock
    supports_directives: bool = False  # FakeDirectives honoured
    real: bool = False
    step_s: float = 0.5  # time advanced per wait iteration
    default_timeout_s: float = 120.0
    options: dict[str, Any] = field(default_factory=dict)

    def advance(self, seconds: float) -> None:
        if isinstance(self.clock, FakeClock):
            self.clock.advance(seconds)
        else:
            time.sleep(seconds)

    def wait_for_phase(
        self,
        adapter: Adapter,
        ref: RemoteRef,
        phases: Collection[RemotePhase],
        *,
        timeout_s: float | None = None,
    ) -> RemoteStatus:
        """Poll status() until its phase is in `phases`; AssertionError on timeout."""
        budget = self.default_timeout_s if timeout_s is None else timeout_s
        waited = 0.0
        while True:
            st = adapter.status(ref)
            if st.phase in phases:
                return st
            if waited >= budget:
                raise AssertionError(
                    f"{self.name}: still {st.phase} after {budget}s, wanted {sorted(phases)}"
                )
            self.advance(self.step_s)
            waited += self.step_s


_counter = 0


def make_job(
    target: ContractTarget,
    *,
    directives: dict[str, Any] | None = None,
    project_dir: str = "/tmp/gpu-router-contract",
    script: str = "train.py",
    **spec_fields: Any,
) -> Job:
    """A Job in state `provisioning` built directly from models (no store needed)."""
    global _counter
    _counter += 1
    job_id = hashlib.sha256(f"{target.name}-{_counter}-{time.monotonic_ns()}".encode()).hexdigest()[
        :JOB_ID_LEN
    ]
    options = dict(target.options)
    if directives is not None:
        options = {target.name: directives}
    spec = JobSpec(
        project_dir=project_dir,
        script=script,
        provider_options=options,
        source=Source.API,
        **spec_fields,
    )
    now = target.clock.now()
    return Job(
        id=job_id,
        short_id=job_id[:4],
        name=spec.display_name(),
        state=JobState.PROVISIONING,
        source=spec.source,
        project_dir=spec.project_dir,
        spec=spec,
        spec_hash=hashlib.sha256(spec.model_dump_json().encode()).hexdigest(),
        provider=target.name,
        created_at=now,
        updated_at=now,
    )


def make_ctx(job: Job, n: int = 1, **fields: Any) -> AttemptContext:
    return AttemptContext(
        attempt_id=f"{job.id}.{n}",
        attempt_key=f"gpu-{job.id}-{n}",
        n=n,
        **fields,
    )
