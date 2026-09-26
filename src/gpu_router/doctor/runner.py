"""Run the doctor's checks in parallel under one deadline (phase 8a).

Every task gets its own daemon thread (about 30 tasks, mostly waiting on subprocesses or
the daemon), results come back through a queue, and whatever has not answered when the
deadline passes becomes a warn row ("did not finish in 20s"). Daemon threads mean a stuck
probe never keeps `gpu doctor` from exiting; every subprocess is bounded anyway.
"""

from __future__ import annotations

import queue
import threading
import time
from collections import Counter

from gpu_router.doctor.checks import Task, default_tasks
from gpu_router.doctor.model import GROUPS, CheckResult, DriftItem, Report, Status
from gpu_router.doctor.probe import ProbeEnv, detail_safe

__all__ = ["run_doctor"]


def _run_task(
    i: int, task: Task, env: ProbeEnv, out: queue.Queue[tuple[int, list[CheckResult]]]
) -> None:
    t0 = time.monotonic()
    try:
        got = task.fn(env)
        results = got if isinstance(got, list) else [got]
    except Exception as exc:  # a doctor bug must not hide every other finding
        results = [
            CheckResult(
                id=task.id,
                group=task.group,
                title=task.title,
                status=Status.WARN,
                summary=f"this check crashed ({type(exc).__name__}: {exc}); a gpu-router bug",
                detail={"unknown": "crashed"},
            )
        ]
    ms = int((time.monotonic() - t0) * 1000)
    out.put((i, [r.model_copy(update={"elapsed_ms": ms}) for r in results]))


def run_doctor(
    env: ProbeEnv, *, tasks: list[Task] | None = None, only: set[str] | None = None
) -> Report:
    """Run `tasks` (default: every check) and assemble the report. `only` keeps the groups
    (or check-id prefixes) named."""
    t0 = time.monotonic()
    specs = tasks if tasks is not None else default_tasks(env)
    if only:
        wanted = [s for s in specs if s.group in only or any(s.id.startswith(o) for o in only)]
        if not wanted:
            from gpu_router.errors import InvalidRequest

            raise InvalidRequest(
                f"no doctor check matches --only {' '.join(sorted(only))}",
                hint="use a group (" + ", ".join(GROUPS) + ") or a check id prefix such as "
                "provider.kaggle (gpu doctor --json lists the ids)",
            )
        specs = wanted
    results: queue.Queue[tuple[int, list[CheckResult]]] = queue.Queue()
    for i, spec in enumerate(specs):
        threading.Thread(
            target=_run_task, args=(i, spec, env, results), name=f"doctor-{spec.id}", daemon=True
        ).start()
    got: dict[int, list[CheckResult]] = {}
    while len(got) < len(specs):
        left = env.deadline_s - (time.monotonic() - env.started_mono)
        if left <= 0:
            break
        try:
            i, res = results.get(timeout=left)
        except queue.Empty:
            break
        got[i] = res
    checks: list[CheckResult] = []
    for i, spec in enumerate(specs):
        if i in got:
            checks += got[i]
        else:
            checks.append(
                CheckResult(
                    id=spec.id,
                    group=spec.group,
                    title=spec.title,
                    status=Status.WARN,
                    summary=f"did not finish in {env.deadline_s:g}s",
                    fix=f"gpu doctor --timeout {int(env.deadline_s * 3)}",
                    detail={"unknown": "timeout"},
                    elapsed_ms=int(env.deadline_s * 1000),
                )
            )
    drift: list[DriftItem] = []
    clean: list[CheckResult] = []
    for c in checks:
        detail = dict(c.detail)
        for raw in detail.pop("drift", None) or []:
            drift.append(DriftItem.model_validate(raw))
        clean.append(c.model_copy(update={"detail": detail_safe(detail)}))
    counts = Counter(str(c.status) for c in clean)
    unknown = sum(1 for c in clean if c.detail.get("unknown") in ("crashed", "timeout"))
    return Report(
        version=env.version,
        home=str(env.paths.home),
        checked_at=env.clock.now(),
        elapsed_ms=int((time.monotonic() - t0) * 1000),
        ok=counts.get(str(Status.FAIL), 0) == 0,
        counts={s.value: counts.get(s.value, 0) for s in Status},
        checks=clean,
        drift=drift,
        unknown=unknown,
    )
