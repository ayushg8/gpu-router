"""EngineDeps: collaborators shared by the supervisor and every driver (phase 1; real code;
owner: group C)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gpu_router.adapters.registry import AdapterRegistry
    from gpu_router.checkpoint.hub import CheckpointHub
    from gpu_router.clock import Clock
    from gpu_router.config import Config
    from gpu_router.engine.calls import AdapterCaller
    from gpu_router.packaging import BundleBuilder
    from gpu_router.paths import Paths
    from gpu_router.policy import ApprovalPolicy
    from gpu_router.router.base import Router
    from gpu_router.store import Store


@dataclass(frozen=True, slots=True)
class EngineDeps:
    store: Store  # event-loop thread only (invariant 9)
    registry: AdapterRegistry
    router: Router
    policy: ApprovalPolicy
    caller: AdapterCaller
    clock: Clock
    config: Config
    paths: Paths
    #: Phase 2: builds the job bundle at submit (Supervisor.submit) and materializes it into
    #: jobs/<id>/. None (engine unit tests) = no bundle; adapters get bundle_dir=None.
    bundler: BundleBuilder | None = None
    #: Phase 5: checkpoint storage, planned handoff and datasets (gpu_router/checkpoint).
    #: None (engine unit tests) = phase-3 behaviour: adapters keep checkpoints themselves.
    checkpoints: CheckpointHub | None = None
