"""Router input assembly shared by drivers and dry runs (phase 1; owner: group C).

Phase 5: provider quota is the quota ledger's view (quota/ledger.py: latest live reading when
fresh, else an estimate from job history; computed from the store, never a provider call),
and the job's bundle estimate (`jobs/<id>/bundle/manifest.json["estimate"]`, D17) rides
along as `RoutingContext.estimate` for the scoring router.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from gpu_router.router.base import JobEstimate, ProviderSnapshot, RoutingContext

if TYPE_CHECKING:
    from gpu_router.engine.deps import EngineDeps
    from gpu_router.models import Job, JobSpec, QuotaSnapshot

_logger = logging.getLogger("gpu_router.engine.context")


def build_routing_context(
    deps: EngineDeps,
    job: Job,
    *,
    persisted: bool = True,
    estimate: JobEstimate | None = None,
) -> RoutingContext:
    """Snapshot of everything the router may look at for `job` right now.

    `persisted=False` is for dry runs of a job that is not in the store: exclusions,
    previous provider and resume state are then empty, and the estimate is the one passed
    in (the dry-run caller computes it from the project), not read from a bundle.
    """
    store = deps.store
    live = store.live_attempts_by_provider()
    quotas = quota_views(deps)
    providers: list[ProviderSnapshot] = []
    for name in deps.registry.names():
        adapter = deps.registry.get(name)
        providers.append(
            ProviderSnapshot(
                name=name,
                entry=deps.registry.entry(name),
                capabilities=adapter.capabilities,
                state=store.get_provider_state(name),
                live_attempts=live.get(name, 0),
                quota=quotas.get(name),
            )
        )
    excluded: frozenset[str] = frozenset()
    previous: str | None = None
    resuming = False
    resume_step: int | None = None
    if persisted:
        excluded = frozenset(store.excluded_providers(job.id))
        attempts = store.attempts_for(job.id)
        previous = attempts[-1].provider if attempts else None
        latest = store.latest_checkpoint(job.id)
        resuming = latest is not None
        resume_step = latest.step if latest is not None else None
        if estimate is None:
            estimate = bundle_estimate(deps, job)
    return RoutingContext(
        job=job,
        now=deps.clock.now(),
        providers=providers,
        excluded=excluded,
        data_unreachable=data_unreachable(deps, job, providers),
        previous_provider=previous,
        resuming=resuming,
        estimate=estimate,
        resume_step=resume_step,
    )


def data_unreachable(
    deps: EngineDeps, job: Job, providers: Sequence[ProviderSnapshot]
) -> dict[str, str]:
    """Providers this job's local `data:` paths cannot reach, with why: no HF storage for
    remote runs (cached hub state, no network) and no adapter.stage_data. Runs on this Mac
    link the path; URIs (hf://) need nothing. Empty when unknown (HF not tried yet)."""
    hub = deps.checkpoints
    if hub is None or not any(d.path for d in job.spec.data) or hub.remote_data_possible():
        return {}
    can = [p.name for p in providers if p.capabilities.stage_data]
    alt = f" ({', '.join(can)} can: it keeps datasets itself)" if can else ""
    out: dict[str, str] = {}
    for p in providers:
        if hub.is_local_kind(p.entry.kind) or p.capabilities.stage_data:
            continue
        out[p.name] = f"cannot receive data= without Hugging Face storage{alt}"
    return out


def quota_views(deps: EngineDeps) -> dict[str, QuotaSnapshot]:
    """The quota ledger's view of every registered provider. Falls back to the raw latest
    snapshots if the ledger cannot be computed (a bad `routing:` config must not stop
    routing; the daemon validates that section at start)."""
    from gpu_router.quota.ledger import ledger_views
    from gpu_router.router.settings import routing_settings

    try:
        settings = routing_settings(deps.config.routing).quota
        return ledger_views(deps.store, deps.registry, deps.clock.now(), settings)
    except Exception:
        _logger.warning("quota ledger unavailable; using raw quota snapshots", exc_info=True)
        return deps.store.latest_quota_snapshots()


def quota_left_hours(
    deps: EngineDeps, provider: str, gpu: str | None = None
) -> tuple[float | None, float | None]:
    """(GPU hours left on `provider`, when its quota resets) by the quota ledger, the view
    routing uses (D44): other attempts since the reading count, and a reading from before
    its own reset does not. A fresh live reading is taken as of now, minus our GPU time
    since it was observed. (None, None) when unknown or unlimited."""
    from gpu_router.errors import ProviderNotFound
    from gpu_router.quota.ledger import remaining, to_gpu_hours

    view = quota_views(deps).get(provider)
    if view is None:
        return None, None
    left = remaining(view)
    if left is None:
        return None, view.resets_at
    try:
        left_h = to_gpu_hours(deps.registry.entry(provider), left, gpu)
    except ProviderNotFound:
        return None, view.resets_at
    if left_h is not None and view.detail.get("basis") == "live":
        observed = view.detail.get("live_observed_at", view.observed_at)
        if isinstance(observed, int | float):
            now = deps.clock.now()
            left_h -= deps.store.usage_seconds(provider, float(observed), now) / 3600
    return (None if left_h is None else max(0.0, left_h)), view.resets_at


def bundle_estimate(deps: EngineDeps, job: Job) -> JobEstimate | None:
    """The estimate recorded in the job's materialized bundle manifest, if any."""
    path = deps.paths.job_bundle_dir(job.id) / "manifest.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return estimate_from_manifest(raw)


def estimate_from_manifest(manifest: Any) -> JobEstimate | None:
    est = manifest.get("estimate") if isinstance(manifest, dict) else None
    if not isinstance(est, dict):
        return None
    try:
        return JobEstimate.model_validate(
            {
                "vram_gb": est.get("vram_gb"),
                "hours": est.get("hours"),
                "vram_source": est.get("vram_source"),
                "hours_source": est.get("hours_source"),
                "mode": est.get("mode"),
                "reasons": tuple(str(r) for r in est.get("reasons") or ()),
            }
        )
    except Exception:
        return None


def project_estimate(spec: JobSpec) -> JobEstimate | None:
    """VRAM/runtime estimate straight from the project (dry runs, POST /v1/route), the same
    heuristics the bundle records. Blocking (git ls-files + source scan): call it in a
    worker thread. None when the project cannot be read."""
    try:
        from gpu_router.packaging.estimate import estimate
        from gpu_router.packaging.files import select_files

        est = estimate(spec, select_files(spec.project_dir).files)
        return estimate_from_manifest({"estimate": est.to_manifest()})
    except Exception:
        return None
