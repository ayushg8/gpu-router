"""DaemonRuntime: builds, owns and tears down every long-lived daemon object (phase 1;
owner: group C). The API test suite constructs one against a tmp home; server.py runs one.

`DaemonRuntime.create(paths, config, clock)` (sync) and `await runtime.start()` do, in order:
 1. paths.ensure(); InstanceLock.acquire(paths)             (invariant 1; DaemonAlreadyRunning)
 2. setup_daemon_logging(...) unless `configure_logging=False` (tests)
 3. config.persist_migrated_config(paths) if needed
 4. catalog = providers.catalog.load_catalog(paths.user_providers)
 5. store = Store.open(paths.db, lock=lock, clock=clock, listener=bus)
 6. registry = AdapterRegistry.build(...); caller = AdapterCaller(registry, config.engine)
 7. router = ScoringRouter(routing settings); policy = policy.default_policy(config)
    (phase 5; `routing:` / `policy:` are validated here, a bad section is a ConfigError);
    quota = QuotaService (ledger views + background live refresh, started in start())
 8. supervisor = Supervisor(EngineDeps(...)); statefile writer subscribed to the bus;
    notifier (phase 8a, notify/service.py) subscribed to the bus
 9. token = auth.ensure_token(paths); crashpoints.configure(test_mode=config.test_mode)
10. meta: instance_id (once), last_start_at, last_pid
11. await supervisor.start()  (recovery; runtime.ready follows supervisor.ready)
`await runtime.stop()`: supervisor.stop(), statefile flush, meta last_clean_shutdown_at,
store.close(), lock.release(). Idempotent.

`request_shutdown()` asks the hosting server to exit (POST /v1/daemon/shutdown); server.py
installs the hook, in-process tests may leave it unset.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from gpu_router.engine._obs import emit

if TYPE_CHECKING:
    from gpu_router.adapters.registry import AdapterRegistry
    from gpu_router.clock import Clock
    from gpu_router.config import Config
    from gpu_router.daemon.events import EventBus
    from gpu_router.engine.supervisor import Supervisor
    from gpu_router.inference.service import InferenceService
    from gpu_router.lock import InstanceLock
    from gpu_router.notify.service import Notifier
    from gpu_router.paths import Paths
    from gpu_router.providers.catalog import Catalog
    from gpu_router.quota.service import QuotaService
    from gpu_router.statefile import ProviderSummary, StateFileWriter, StateSnapshot
    from gpu_router.store import Store


@dataclass(slots=True)
class DaemonRuntime:
    paths: Paths
    config: Config
    clock: Clock
    lock: InstanceLock
    store: Store
    registry: AdapterRegistry
    supervisor: Supervisor
    bus: EventBus
    statefile: StateFileWriter
    token: str
    started_at: float
    port: int = 0  # set by server.py once uvicorn is bound
    quota: QuotaService | None = field(default=None, repr=False)  # phase 5 quota ledger
    inference: InferenceService | None = field(default=None, repr=False)  # phase 7b lane
    notifier: Notifier | None = field(default=None, repr=False)  # phase 8a notifications
    shutdown_hook: Callable[[], None] | None = field(default=None, repr=False)
    _writer_task: asyncio.Task[None] | None = field(default=None, repr=False)
    _started: bool = field(default=False, repr=False)
    _stopped: bool = field(default=False, repr=False)

    @classmethod
    def create(
        cls,
        paths: Paths,
        config: Config,
        clock: Clock,
        *,
        configure_logging: bool = True,
        foreground: bool = False,
    ) -> DaemonRuntime:
        """Steps 1-10 of the module docstring (no event loop needed)."""
        from gpu_router.adapters.registry import AdapterRegistry
        from gpu_router.checkpoint import set_active_hub
        from gpu_router.checkpoint.hub import CheckpointHub
        from gpu_router.config import persist_migrated_config
        from gpu_router.daemon import auth
        from gpu_router.daemon.events import EventBus
        from gpu_router.engine import crashpoints
        from gpu_router.engine.calls import AdapterCaller
        from gpu_router.engine.deps import EngineDeps
        from gpu_router.engine.supervisor import Supervisor
        from gpu_router.lock import InstanceLock
        from gpu_router.log import setup_daemon_logging
        from gpu_router.notify.service import attach_notifier
        from gpu_router.packaging import BundleBuilder
        from gpu_router.policy import default_policy
        from gpu_router.providers.catalog import load_catalog
        from gpu_router.quota.service import QuotaService
        from gpu_router.router.scoring import ScoringRouter
        from gpu_router.router.settings import routing_settings
        from gpu_router.statefile import StateFileWriter
        from gpu_router.store import Store

        paths.ensure()
        lock = InstanceLock.acquire(paths)
        store: Store | None = None
        try:
            if configure_logging:
                setup_daemon_logging(
                    paths,
                    level=config.logging.level,
                    max_bytes=config.logging.max_bytes,
                    backups=config.logging.backups,
                    foreground=foreground,
                )
            persist_migrated_config(paths)
            routing = routing_settings(config.routing)
            policy = default_policy(config)
            catalog = load_catalog(paths.user_providers)
            bus = EventBus()
            store = Store.open(paths.db, lock=lock, clock=clock, listener=bus)
            registry = AdapterRegistry.build(
                config=config, catalog=catalog, paths=paths, clock=clock
            )
            caller = AdapterCaller(registry, config.engine)
            deps = EngineDeps(
                store=store,
                registry=registry,
                router=ScoringRouter(routing),
                policy=policy,
                caller=caller,
                clock=clock,
                config=config,
                paths=paths,
                bundler=BundleBuilder(paths),
                checkpoints=CheckpointHub.from_config(config, paths, clock),
            )
            set_active_hub(deps.checkpoints)  # adapters read the log side channel through it
            supervisor = Supervisor(deps)
            token = auth.ensure_token(paths)
            crashpoints.configure(test_mode=config.test_mode)
            now = clock.now()
            if store.get_meta("instance_id") is None:
                store.set_meta("instance_id", uuid.uuid4().hex)
            store.set_meta("last_start_at", repr(now))
            store.set_meta("last_pid", str(os.getpid()))

            holder: dict[str, DaemonRuntime] = {}
            writer = StateFileWriter(
                paths.state, lambda: holder["rt"].build_state_snapshot(), wall=clock.now
            )
            bus.subscribe(lambda _job_id, _events: writer.mark_dirty())
            # phase 8a: finished / failed / approval / migrated -> macOS notifications,
            # off the engine's path (a bad `notifications:` section is a ConfigError here)
            notifier = attach_notifier(bus, store, config, clock)
            runtime = cls(
                paths=paths,
                config=config,
                clock=clock,
                lock=lock,
                store=store,
                registry=registry,
                supervisor=supervisor,
                bus=bus,
                statefile=writer,
                token=token,
                started_at=now,
                quota=QuotaService(
                    store=store,
                    registry=registry,
                    caller=caller,
                    clock=clock,
                    settings=routing.quota,
                ),
                inference=_inference_service(catalog, paths, clock, config.test_mode),
                notifier=notifier,
            )
            holder["rt"] = runtime
            return runtime
        except BaseException:
            if store is not None:
                with contextlib.suppress(Exception):
                    store.close()
            lock.release()
            raise

    async def start(self) -> None:
        """Step 11, plus start the statefile writer task."""
        if self._started:
            return
        self._started = True
        await self.supervisor.start()
        self._writer_task = asyncio.get_running_loop().create_task(
            self.statefile.run(), name="statefile-writer"
        )
        if self.quota is not None:
            self.quota.start()
        with contextlib.suppress(Exception):
            self.statefile.flush()
        emit(
            "daemon.start",
            f"daemon ready (pid {os.getpid()}, "
            f"{len(self.registry)} provider(s): {', '.join(self.registry.names()) or 'none'})",
            pid=os.getpid(),
            providers=self.registry.names(),
            test_mode=self.config.test_mode,
        )

    async def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        try:
            if self.quota is not None:
                with contextlib.suppress(Exception):
                    await self.quota.close()
            if self.inference is not None:  # phase 7b
                with contextlib.suppress(Exception):
                    self.inference.close()
            if self.notifier is not None:
                with contextlib.suppress(Exception):
                    self.notifier.close()
            await self.supervisor.stop()
        finally:
            if self._writer_task is not None:
                self._writer_task.cancel()
                with contextlib.suppress(BaseException):
                    await self._writer_task
                self._writer_task = None
            with contextlib.suppress(Exception):
                self.statefile.flush()
            with contextlib.suppress(Exception):
                self.store.set_meta("last_clean_shutdown_at", repr(self.clock.now()))
            emit("daemon.stop", "daemon stopped; remote runs keep going", pid=os.getpid())
            hub = self.supervisor.deps.checkpoints
            if hub is not None:
                from gpu_router.checkpoint import active_hub, set_active_hub

                if active_hub() is hub:
                    set_active_hub(None)
                with contextlib.suppress(Exception):
                    hub.close()
            with contextlib.suppress(Exception):
                self.store.close()
            self.lock.release()

    @property
    def ready(self) -> bool:
        return self.supervisor.ready

    def request_shutdown(self) -> bool:
        """Ask the hosting server to exit gracefully. False when no server is attached."""
        if self.shutdown_hook is None:
            return False
        self.shutdown_hook()
        return True

    # ------------------------------------------------------------------ state.json

    def provider_summaries(self) -> list[ProviderSummary]:
        from gpu_router.quota.ledger import remaining
        from gpu_router.statefile import ProviderSummary

        states = self.store.all_provider_states()
        quotas = (
            self.quota.views() if self.quota is not None else self.store.latest_quota_snapshots()
        )
        out: list[ProviderSummary] = []
        for name in self.registry.names():
            state = states.get(name)
            quota = quotas.get(name)
            out.append(
                ProviderSummary(
                    name=name,
                    health=str(state.health) if state else "unknown",
                    used=quota.used if quota else None,
                    limit=quota.limit if quota else None,
                    unit=str(quota.unit) if quota else None,
                    resets_at=quota.resets_at if quota else None,
                    source=quota.source if quota else None,
                    # phase 6b (status line): ledger facts the quota bar needs
                    unlimited=bool(quota.detail.get("unlimited")) if quota else False,
                    remaining=remaining(quota),
                )
            )
        return out

    def build_state_snapshot(self) -> StateSnapshot:
        from gpu_router.statefile import build_snapshot

        caps: dict[str, float | None] = {
            name: self.registry.entry(name).session_cap_s for name in self.registry.names()
        }
        return build_snapshot(
            store=self.store,
            provider_summaries=self.provider_summaries(),
            session_caps=caps,
            now=self.clock.now(),
            daemon_pid=os.getpid(),
            recent_window_s=self.config.statusline.finished_visible_s,
            migrated_window_s=self.config.statusline.migrated_visible_s,  # phase 6b
            metrics_path=self.paths.job_metrics,  # phase 6b: metric trend
        )


def _inference_service(
    catalog: Catalog, paths: Paths, clock: Clock, test_mode: bool
) -> InferenceService | None:
    """The phase-7b inference lane; None (its endpoints answer not_ready) if it cannot be
    built, so a broken inference catalog never stops GPU routing."""
    from gpu_router.inference.catalog import load_inference_catalog
    from gpu_router.inference.service import InferenceService, ledger_path

    try:
        return InferenceService(
            load_inference_catalog(catalog),
            clock,
            ledger_file=ledger_path(paths.home),
            test_mode=test_mode,
        )
    except Exception:
        emit("inference.unavailable", "inference lane off: it could not be built", exc_info=True)
        return None
