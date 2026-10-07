"""Engine + checkpoint storage (phase 5) on a FakeClock: planned handoff before the session
cap and before the quota runs out (incl. no answer and a daemon restart in between),
storage reconciliation before a migration, datasets uploaded once and reused by hash, and
the degrade path without a Hugging Face token."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from gpu_router import secrets
from gpu_router.adapters.base import AttemptContext
from gpu_router.adapters.fake import FakeAdapter
from gpu_router.adapters.registry import AdapterRegistry
from gpu_router.checkpoint.hub import CheckpointHub
from gpu_router.clock import FakeClock, settle
from gpu_router.config import CheckpointConfig, Config
from gpu_router.engine.calls import AdapterCaller
from gpu_router.engine.deps import EngineDeps
from gpu_router.engine.supervisor import Supervisor
from gpu_router.models import (
    AttemptState,
    DataRef,
    Job,
    JobSpec,
    JobState,
    QuotaSnapshot,
    QuotaUnit,
    Reason,
    Source,
)
from gpu_router.paths import Paths
from gpu_router.policy import default_policy
from gpu_router.providers.catalog import Catalog
from gpu_router.router.simple import SimpleRouter
from gpu_router.runner import storage as rs
from gpu_router.statemachine import is_terminal
from gpu_router.store import Store
from tests.unit.checkpoint.fake_hfapi import FakeHfApi
from tests.unit.engine.conftest import InlineExecutor, engine_config

HOUR = 3600.0
TOKEN = "hf_" + "k" * 34
REMOTE_TOKEN = "hf_" + "r" * 34


@dataclass
class CEngine:
    store: Store
    clock: FakeClock
    paths: Paths
    config: Config
    catalog: Catalog
    hub: CheckpointHub
    supervisor: Supervisor
    project: str
    contexts: list[tuple[str, AttemptContext]]

    def job(self, job_id: str) -> Job:
        return self.store.get_job(job_id)

    def spec(self, fake: dict[str, Any] | None = None, **fields: Any) -> JobSpec:
        fields.setdefault("script", "train.py")
        return JobSpec(
            project_dir=self.project,
            provider_options={"fake": {"duration": 5, "steps": 5, **(fake or {})}},
            source=Source.API,
            **fields,
        )

    async def submit(self, spec: JobSpec) -> Job:
        job, created = await self.supervisor.submit(spec, actor="api")
        assert created
        return job

    async def run_until(
        self, pred: Callable[[], bool], *, max_s: float = 3600, step: float = 0.5
    ) -> None:
        waited = 0.0
        while True:
            await settle(30)
            if pred():
                return
            if waited >= max_s:
                raise AssertionError(f"condition not met after {max_s}s of fake time")
            self.clock.advance(step)
            waited += step

    def notes(self, job_id: str, reason: Reason) -> list[Any]:
        return [e for e in self.store.events_for(job_id, limit=10_000) if e.reason == reason]

    def transitions(self, job_id: str) -> list[tuple[str | None, str | None, str]]:
        return [
            (str(e.from_state) if e.from_state else None, str(e.to_state), e.reason)
            for e in self.store.events_for(job_id, limit=10_000)
            if e.kind == "transition"
        ]

    def local_root(self) -> Path:
        store = self.hub.local()
        assert store is not None
        return Path(store.raw.root)

    async def restart(self) -> None:
        await self.supervisor.stop()
        self.supervisor = build(self)
        await self.supervisor.start()


def build(ce: CEngine) -> Supervisor:
    registry = AdapterRegistry.build(
        config=ce.config, catalog=ce.catalog, paths=ce.paths, clock=ce.clock
    )
    for name in ("fake", "fake-b"):
        adapter = registry.get(name)
        assert isinstance(adapter, FakeAdapter)
        real = adapter.submit

        def spy(job: Any, ctx: AttemptContext, _real: Any = real, _name: str = name) -> Any:
            ce.contexts.append((_name, ctx))
            return _real(job, ctx)

        adapter.submit = spy  # type: ignore[method-assign]
    caller = AdapterCaller(registry, ce.config.engine, executor=InlineExecutor())
    deps = EngineDeps(
        store=ce.store,
        registry=registry,
        router=SimpleRouter(),
        policy=default_policy(),
        caller=caller,
        clock=ce.clock,
        config=ce.config,
        paths=ce.paths,
        checkpoints=ce.hub,
    )
    return Supervisor(deps)


def make_hub(paths: Paths, clock: FakeClock, cfg: CheckpointConfig, **kw: Any) -> CheckpointHub:
    return CheckpointHub(
        cfg,
        paths,
        clock,
        test_mode=True,
        executor=InlineExecutor(),
        bulk_executor=InlineExecutor(),
        **kw,
    )


@pytest.fixture
def hub_kw() -> dict[str, Any]:
    """Override per test: extra CheckpointHub keyword arguments."""
    return {"local_kinds": frozenset({"local", "fake"})}


@pytest.fixture
async def ceng(
    store: Store,
    clock: FakeClock,
    paths: Paths,
    catalog: Catalog,
    tmp_path: Path,
    hub_kw: dict[str, Any],
) -> AsyncIterator[CEngine]:
    project = tmp_path / "project"
    project.mkdir()
    config = engine_config()
    config.checkpoint = CheckpointConfig(handoff_margin_min=30, handoff_wait_min=5)
    hub = make_hub(paths, clock, config.checkpoint, **hub_kw)
    ce = CEngine(
        store=store,
        clock=clock,
        paths=paths,
        config=config,
        catalog=catalog,
        hub=hub,
        supervisor=None,  # type: ignore[arg-type]
        project=str(project),
        contexts=[],
    )
    ce.supervisor = build(ce)
    await ce.supervisor.start()
    await settle(30)
    try:
        yield ce
    finally:
        await ce.supervisor.stop()


def _control(ce: CEngine, job_id: str, n: int) -> dict[str, Any] | None:
    return rs.read_json(rs.LocalStore(ce.local_root()), rs.control_key(job_id, n))


def _ack_with_checkpoint(ce: CEngine, job_id: str, n: int, seq: int, step: int) -> str:
    """Act as the runner: publish checkpoint `seq` to local storage, then acknowledge."""
    store = rs.LocalStore(ce.local_root())
    req = _control(ce, job_id, n)
    assert req is not None
    src = ce.local_root().parent / f"stage-{seq}"
    src.mkdir(parents=True, exist_ok=True)
    (src / "state.json").write_text(json.dumps({"step": step}))
    latest = rs.publish_checkpoint(
        store, job_id, seq, src, [("state.json", 12, "x")], attempt=n, step=step
    )
    ack = dict(latest, id=req["id"], new=True)
    store.write_bytes(rs.ack_key(job_id, n), rs.dumps(ack))
    return str(latest["uri"])


def _running_attempt(ce: CEngine, job_id: str, n: int) -> bool:
    atts = ce.store.attempts_for(job_id)
    return len(atts) >= n and atts[n - 1].state is AttemptState.RUNNING


# --------------------------------------------------------------------------- handoff


async def test_planned_handoff_before_the_session_cap(ceng: CEngine) -> None:
    spec = ceng.spec(
        fake={
            "duration": 13 * HOUR,
            "checkpoint_every": HOUR,
            "attempts": {"2": {"duration": 60, "checkpoint_every": 0}},
        },
        provider="fake",
    )
    job = await ceng.submit(spec)
    await ceng.run_until(lambda: _control(ceng, job.id, 1) is not None, max_s=13 * HOUR, step=60)
    attempt = ceng.store.attempts_for(job.id)[0]
    assert attempt.session_deadline is not None
    left = attempt.session_deadline - ceng.clock.now()
    assert 29 * 60 <= left <= 31 * 60  # asked 30 min before fake's 12 h cap
    req = _control(ceng, job.id, 1)
    assert req is not None
    assert req["action"] == "handoff"
    assert req["wait_s"] == 300
    (asked,) = ceng.notes(job.id, Reason.HANDOFF_REQUESTED)
    assert "session limit" in asked.message
    assert asked.attempt_id == attempt.id
    assert ceng.job(job.id).state in (JobState.RUNNING, JobState.CHECKPOINTING)

    # the fake emitted checkpoints 1..11 (fake:// URIs); the runner saves #12 for the handoff
    uri = _ack_with_checkpoint(ceng, job.id, 1, seq=12, step=99)
    await ceng.run_until(lambda: _running_attempt(ceng, job.id, 2), step=1)
    trans = ceng.transitions(job.id)
    assert any(t[1] == "migrating" and t[2] == Reason.HANDOFF for t in trans)
    first, second = ceng.store.attempts_for(job.id)
    assert first.state is AttemptState.CANCELLED
    assert first.lost_reason == "stopped for a migration"
    assert second.resume_checkpoint_id == f"{job.id}.c12"
    latest = ceng.store.latest_checkpoint(job.id)
    assert latest is not None
    assert latest.seq == 12
    assert latest.uri == uri
    assert latest.step == 99
    (found,) = ceng.notes(job.id, Reason.CHECKPOINT_FOUND)
    assert "saved for the handoff" in found.message
    _name, ctx = ceng.contexts[-1]
    assert ctx.resume_from is not None
    assert ctx.resume_from.seq == 12
    assert ctx.env["GPU_RESUME_URI"] == uri  # the next runner restores from storage
    assert ctx.env["GPU_STORAGE"] == ceng.local_root().as_uri()
    owner = json.loads((ceng.local_root() / "jobs" / job.id / "owner.json").read_text())
    assert owner["attempt"] == 2  # an old runner still alive would stop publishing
    await ceng.run_until(lambda: is_terminal(ceng.job(job.id).state), step=5)
    assert ceng.job(job.id).state is JobState.DONE


async def test_handoff_before_the_quota_runs_out_marks_the_provider(ceng: CEngine) -> None:
    now = ceng.clock.now()
    ceng.store.record_quota_snapshot(
        QuotaSnapshot(
            provider="fake",
            used=29.7,
            limit=30,
            unit=QuotaUnit.GPU_HOURS,
            resets_at=now + 5 * 24 * HOUR,
            source="live",
            observed_at=now,
        )
    )
    job = await ceng.submit(ceng.spec(fake={"duration": 2 * HOUR}))
    await ceng.run_until(lambda: _control(ceng, job.id, 1) is not None, step=1)
    (asked,) = ceng.notes(job.id, Reason.HANDOFF_REQUESTED)
    assert "free GPU quota runs out" in asked.message
    store = rs.LocalStore(ceng.local_root())
    req = _control(ceng, job.id, 1)
    assert req is not None
    store.write_bytes(rs.ack_key(job.id, 1), rs.dumps({"id": req["id"], "new": False}))
    await ceng.run_until(lambda: len(ceng.store.attempts_for(job.id)) == 2, step=1)
    handoff = [e for e in ceng.store.events_for(job.id) if e.reason == Reason.HANDOFF]
    assert "no checkpoint yet" in handoff[0].message
    assert ceng.store.get_provider_state("fake").exhausted_until == now + 5 * 24 * HOUR
    assert ceng.store.attempts_for(job.id)[1].provider == "fake-b"


async def test_no_answer_means_the_job_runs_on(ceng: CEngine) -> None:
    job = await ceng.submit(ceng.spec(fake={"duration": 12.5 * HOUR}, provider="fake"))
    await ceng.run_until(lambda: _control(ceng, job.id, 1) is not None, max_s=13 * HOUR, step=60)
    await ceng.run_until(
        lambda: bool(ceng.notes(job.id, Reason.HANDOFF_SKIPPED)), max_s=HOUR, step=30
    )
    (skipped,) = ceng.notes(job.id, Reason.HANDOFF_SKIPPED)
    assert "did not answer the checkpoint request" in skipped.message
    assert ceng.job(job.id).state is JobState.RUNNING
    await ceng.run_until(lambda: is_terminal(ceng.job(job.id).state), max_s=2 * HOUR, step=60)
    assert ceng.job(job.id).state is JobState.DONE
    assert len(ceng.store.attempts_for(job.id)) == 1


async def test_a_pending_handoff_survives_a_daemon_restart(ceng: CEngine) -> None:
    job = await ceng.submit(
        ceng.spec(
            fake={"duration": 13 * HOUR, "attempts": {"2": {"duration": 30}}}, provider="fake"
        )
    )
    await ceng.run_until(lambda: _control(ceng, job.id, 1) is not None, max_s=13 * HOUR, step=60)
    await ceng.restart()
    _ack_with_checkpoint(ceng, job.id, 1, seq=1, step=7)
    await ceng.run_until(lambda: is_terminal(ceng.job(job.id).state), step=2)
    assert ceng.job(job.id).state is JobState.DONE
    assert len(ceng.notes(job.id, Reason.HANDOFF_REQUESTED)) == 1  # not asked twice
    assert ceng.store.attempts_for(job.id)[1].resume_checkpoint_id == f"{job.id}.c1"


async def test_migration_resumes_from_a_checkpoint_only_storage_knew(ceng: CEngine) -> None:
    """A session that dies before its last ckpt_end line reaches the daemon (Kaggle logs
    arrive only after the run): the migrating step reads latest.json and resumes from it."""
    job = await ceng.submit(
        ceng.spec(
            fake={"duration": 100, "die_after": 30, "attempts": {"2": {"duration": 10}}},
            provider="fake",
        )
    )
    await ceng.run_until(lambda: _running_attempt(ceng, job.id, 1), step=1)
    store = rs.LocalStore(ceng.local_root())
    src = ceng.local_root().parent / "stage"
    src.mkdir()
    (src / "w.pt").write_text("w")
    rs.publish_checkpoint(store, job.id, 3, src, [("w.pt", 1, "x")], attempt=1, step=30)
    await ceng.run_until(lambda: is_terminal(ceng.job(job.id).state), step=1)
    assert ceng.job(job.id).state is JobState.DONE
    second = ceng.store.attempts_for(job.id)[1]
    assert second.resume_checkpoint_id == f"{job.id}.c3"
    (found,) = ceng.notes(job.id, Reason.CHECKPOINT_FOUND)
    assert "log never reported it" in found.message


async def test_attempts_that_made_progress_do_not_count_against_max_attempts(
    ceng: CEngine,
) -> None:
    """A long job moved at every session end keeps going past max_attempts as long as
    each attempt saves new checkpoints."""
    per_attempt = {str(n): {"duration": 20, "die_after": 10, "checkpoint_every": 4} for n in (1, 2)}
    per_attempt["3"] = {"duration": 5}
    job = await ceng.submit(
        ceng.spec(fake={"attempts": per_attempt}, provider="fake", max_attempts=1)
    )
    await ceng.run_until(lambda: is_terminal(ceng.job(job.id).state), step=1)
    assert ceng.job(job.id).state is JobState.DONE
    assert len(ceng.store.attempts_for(job.id)) == 3


# --------------------------------------------------------------------------- datasets


@pytest.fixture
def hf_api() -> FakeHfApi:
    return FakeHfApi()


def _data_dir(ceng: CEngine, name: str, text: str) -> Path:
    d = Path(ceng.project) / name
    d.mkdir(exist_ok=True)
    (d / "rows.csv").write_text(text)
    return d


async def test_dataset_is_uploaded_once_and_reused(
    ceng: CEngine, hf_api: FakeHfApi, paths: Paths, clock: FakeClock
) -> None:
    secrets.set_secret("HF_TOKEN", TOKEN)
    secrets.set_secret("HF_TOKEN_REMOTE", REMOTE_TOKEN)  # remote runs get only this (D44)
    ceng.hub = make_hub(paths, clock, ceng.config.checkpoint, hf_api=hf_api.factory)
    await ceng.restart()
    data = _data_dir(ceng, "ds", "a,b\n1,2\n")
    spec = ceng.spec(data=[DataRef(mount="ds", path=str(data))], provider="fake")

    first = await ceng.submit(spec)
    await ceng.run_until(lambda: is_terminal(ceng.job(first.id).state), step=1)
    assert ceng.job(first.id).state is JobState.DONE
    (uploaded,) = ceng.notes(first.id, Reason.DATA_UPLOADED)
    sha = uploaded.detail["sha256"]
    bucket = hf_api.files("tester/gpu-router")
    assert bucket[f"datasets/{sha}/rows.csv"] == b"a,b\n1,2\n"
    assert f"datasets/{sha}/.gpu-data.json" in bucket
    cached = ceng.store.get_data_cache(sha)
    assert cached is not None
    assert cached.uri == f"hf://buckets/tester/gpu-router/datasets/{sha}"
    _name, ctx = ceng.contexts[-1]
    assert json.loads(ctx.env["GPU_DATA"]) == [{"mount": "ds", "uri": cached.uri, "sha256": sha}]
    assert ctx.env["GPU_STORAGE"] == "hf://buckets/tester/gpu-router"
    assert ctx.secrets["GPU_STORAGE_TOKEN"].get_secret_value() == REMOTE_TOKEN
    adds = [a for a in hf_api.ops("batch_bucket_files") if any("datasets/" in k for k in a["add"])]

    clock.advance(60)
    second = await ceng.submit(spec.model_copy(update={"name": "again"}))
    await ceng.run_until(lambda: is_terminal(ceng.job(second.id).state), step=1)
    assert ceng.job(second.id).state is JobState.DONE
    assert ceng.notes(second.id, Reason.DATA_UPLOADED) == []
    (reused,) = ceng.notes(second.id, Reason.DATA_REUSED)
    assert reused.detail["sha256"] == sha
    adds2 = [a for a in hf_api.ops("batch_bucket_files") if any("datasets/" in k for k in a["add"])]
    assert adds2 == adds  # nothing uploaded again
    again = ceng.store.get_data_cache(sha)
    assert again is not None
    assert again.last_used_at > cached.last_used_at

    (data / "rows.csv").write_text("a,b\n1,3\n")  # new content -> new upload
    third = await ceng.submit(spec.model_copy(update={"name": "changed"}))
    await ceng.run_until(lambda: is_terminal(ceng.job(third.id).state), step=1)
    (up3,) = ceng.notes(third.id, Reason.DATA_UPLOADED)
    assert up3.detail["sha256"] != sha


async def test_local_runs_link_datasets_instead_of_uploading(ceng: CEngine) -> None:
    data = _data_dir(ceng, "ds", "x\n")
    job = await ceng.submit(ceng.spec(data=[DataRef(mount="ds", path=str(data))], provider="fake"))
    await ceng.run_until(lambda: is_terminal(ceng.job(job.id).state), step=1)
    _name, ctx = ceng.contexts[-1]
    assert json.loads(ctx.env["GPU_DATA"]) == [{"mount": "ds", "local": str(data.resolve())}]
    assert ceng.notes(job.id, Reason.DATA_UPLOADED) == []


async def test_without_a_token_remote_runs_degrade_with_a_message(
    ceng: CEngine, hf_api: FakeHfApi, paths: Paths, clock: FakeClock
) -> None:
    ceng.hub = make_hub(paths, clock, ceng.config.checkpoint, hf_api=hf_api.factory)
    await ceng.restart()
    job = await ceng.submit(ceng.spec(provider="fake"))
    await ceng.run_until(lambda: is_terminal(ceng.job(job.id).state), step=1)
    assert ceng.job(job.id).state is JobState.DONE  # the job still runs
    (note,) = ceng.notes(job.id, Reason.STORAGE_UNAVAILABLE)
    assert "stay on that machine" in note.message
    assert "gpu login hf" in note.message
    _name, ctx = ceng.contexts[-1]
    assert "GPU_STORAGE" not in ctx.env
    assert "GPU_STORAGE_TOKEN" not in ctx.secrets

    data = _data_dir(ceng, "ds", "y\n")
    job2 = await ceng.submit(ceng.spec(data=[DataRef(mount="ds", path=str(data))]))
    await ceng.run_until(lambda: is_terminal(ceng.job(job2.id).state), step=1)
    failed = ceng.job(job2.id)
    assert failed.state is JobState.FAILED  # both fakes are remote here: nowhere to run
    # refused at routing (2026-10-04 field test), not one burned attempt per provider
    assert ceng.store.attempts_for(job2.id) == []
    (gave_up,) = ceng.notes(job2.id, Reason.NO_PROVIDER_FITS)
    assert "fake: cannot receive data= without Hugging Face storage" in gave_up.message
