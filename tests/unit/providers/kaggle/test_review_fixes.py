"""Phase-3 review regressions for the Kaggle adapter (D34-D36 in CLAUDE.md).

- a job id containing "429" is not a rate limit (the kernel ref is in every CLI error)
- an interrupted `kernels push` that may still be uploading is never pushed twice
- an empty log for an ERROR kernel is not judged until the log is published (bounded)
- the cached final log is redacted before it hits disk
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from gpu_router import secrets
from gpu_router.adapters.base import AttemptContext, RemotePhase
from gpu_router.clock import FakeClock
from gpu_router.errors import DEFINITIVE_SUBMIT_ERRORS, NotFound, RateLimited, Unavailable
from gpu_router.models import Job, JobSpec, JobState, Source
from gpu_router.paths import Paths
from gpu_router.providers.kaggle import adapter as kaggle_mod
from gpu_router.providers.kaggle import remote
from gpu_router.providers.kaggle.adapter import KaggleAdapter
from gpu_router.providers.kaggle.cli import CliResult, classify
from tests.contract.kaggle.sim import CANNOT_ACCESS, SimKaggle
from tests.contract.kaggle.targets import kaggle_deps

JOB_429 = "a4290c12de34"


def _job(job_id: str) -> Job:
    spec = JobSpec(project_dir="/tmp/proj", script="train.py", source=Source.API)
    return Job(
        id=job_id,
        short_id=job_id[:4],
        name="train",
        state=JobState.PROVISIONING,
        source=spec.source,
        project_dir=spec.project_dir,
        spec=spec,
        spec_hash="0" * 64,
        provider="kaggle",
        created_at=0,
        updated_at=0,
    )


def _ctx(job_id: str, archive: Path, n: int = 1, **fields: Any) -> AttemptContext:
    base: dict[str, Any] = {
        "attempt_id": f"{job_id}.{n}",
        "attempt_key": f"gpu-{job_id}-{n}",
        "n": n,
        "bundle_archive": archive,
        "env": {"GPU_ROUTER_JOB_ID": job_id, "GPU_ROUTER_ATTEMPT": str(n)},
    }
    base.update(fields)
    return AttemptContext(**base)


@pytest.fixture
def archive(tmp_path: Path) -> Path:
    p = tmp_path / "bundle.tar.gz"
    p.write_bytes(b"\x1f\x8bnot-really-a-bundle" * 10)
    return p


@pytest.fixture
def sim(clock: FakeClock) -> SimKaggle:
    return SimKaggle(clock)


@pytest.fixture
def adapter(paths: Paths, clock: FakeClock, sim: SimKaggle) -> KaggleAdapter:
    return KaggleAdapter(kaggle_deps(paths, clock), runner=sim)


# --------------------------------------------------------------------------- 429 in names


@pytest.mark.parametrize(
    "text",
    [
        CANNOT_ACCESS.format(ref="simuser/gpu-router-a4290c12de34-1"),
        CANNOT_ACCESS.format(ref="user429/gpu-router-0123456789ab-1"),
        CANNOT_ACCESS.format(ref="simuser/gpu-router-0123456789ab-429"),
        "404 Client Error: Not Found for url: https://www.kaggle.com/api/v1/kernels/pull"
        "?userName=user429&kernelSlug=gpu-router-a4290c12de34-1",
    ],
)
def test_a_missing_kernel_whose_name_contains_429_is_not_found(text: str) -> None:
    err = classify("kaggle", "kernels status", CliResult(("kaggle",), 1, "", text))
    assert type(err) is NotFound


@pytest.mark.parametrize(
    "text",
    [
        "429 Client Error: Too Many Requests for url: https://www.kaggle.com/api/v1/x",
        "requests.exceptions.HTTPError: HTTP Error 429",
        "rate limit exceeded, try again later",
    ],
)
def test_real_throttling_is_still_rate_limited(text: str) -> None:
    err = classify("kaggle", "kernels push", CliResult(("kaggle",), 1, "", text))
    assert isinstance(err, RateLimited)


def test_unknown_failure_naming_a_429_kernel_is_unavailable_not_rate_limited() -> None:
    text = "Traceback ...\nKeyError: 'simuser/gpu-router-a4290c12de34-1'"
    err = classify("kaggle", "kernels status", CliResult(("kaggle",), 1, "", text))
    assert type(err) is Unavailable


def test_a_job_id_with_429_submits_looks_up_and_cancels(
    adapter: KaggleAdapter, sim: SimKaggle, archive: Path, clock: FakeClock
) -> None:
    job = _job(JOB_429)
    ctx = _ctx(JOB_429, archive)
    assert adapter.lookup_by_key(ctx.attempt_key) is None  # NotFound, not RateLimited
    sim.register(remote.slug_for_key(ctx.attempt_key) or "", {"duration": 600})
    ref = adapter.submit(job, ctx)
    assert sim.pushes == [f"gpu-router-{JOB_429}-1"]
    clock.advance(30)
    assert adapter.status(ref).phase is RemotePhase.RUNNING
    adapter.cancel(ref)
    # the kernel is deleted: status must reach the tombstone branch, not RateLimited
    assert adapter.status(ref).phase is RemotePhase.CANCELLED


# --------------------------------------------------------------------------- interrupted push


def _intent(adapter: KaggleAdapter, key: str, started_at: float) -> Path:
    path = adapter.scratch_dir / "submits" / f"{key}.pushing"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"started_at": started_at}))
    return path


def test_a_push_that_may_still_be_uploading_is_never_pushed_again(
    paths: Paths, clock: FakeClock, sim: SimKaggle, archive: Path
) -> None:
    job, ctx = _job("0123456789ab"), _ctx("0123456789ab", archive)
    first = KaggleAdapter(kaggle_deps(paths, clock), runner=sim)
    intent = _intent(first, ctx.attempt_key, clock.now())  # the daemon died mid-push
    fresh = KaggleAdapter(kaggle_deps(paths, clock), runner=sim)  # restarted daemon
    with pytest.raises(Unavailable, match="may still be uploading"):
        fresh.lookup_by_key(ctx.attempt_key)
    with pytest.raises(Unavailable, match="may still be uploading") as info:
        fresh.submit(job, ctx)
    assert not isinstance(info.value, DEFINITIVE_SUBMIT_ERRORS)
    assert sim.pushes == []
    clock.advance(kaggle_mod.T_PUSH + kaggle_mod.PUSH_ORPHAN_GRACE_S + 1)
    assert fresh.lookup_by_key(ctx.attempt_key) is None  # the orphan never created it
    assert not intent.exists()
    fresh.submit(job, ctx)
    assert sim.pushes == ["gpu-router-0123456789ab-1"]


def test_an_interrupted_push_that_completed_is_found_not_repeated(
    paths: Paths, clock: FakeClock, sim: SimKaggle, archive: Path
) -> None:
    job, ctx = _job("0123456789ab"), _ctx("0123456789ab", archive)
    first = KaggleAdapter(kaggle_deps(paths, clock), runner=sim)
    first.submit(job, ctx)  # the orphaned CLI child finished its push ...
    (first.scratch_dir / "submits" / f"{ctx.attempt_key}.json").unlink()  # ... unrecorded
    _intent(first, ctx.attempt_key, clock.now())
    fresh = KaggleAdapter(kaggle_deps(paths, clock), runner=sim)
    ref = fresh.lookup_by_key(ctx.attempt_key)
    assert ref is not None
    assert fresh.submit(job, ctx).remote_id == ref.remote_id
    assert len(sim.pushes) == 1


def test_the_intent_file_lives_only_while_push_runs(
    adapter: KaggleAdapter, sim: SimKaggle, archive: Path
) -> None:
    seen: list[bool] = []
    real = sim.__call__

    def spy(argv: Any, **kw: Any) -> CliResult:
        if "push" in argv:
            seen.append((adapter.scratch_dir / "submits" / "gpu-0123456789ab-1.pushing").exists())
        return real(argv, **kw)

    adapter.cli._runner = spy  # type: ignore[assignment]
    adapter.submit(_job("0123456789ab"), _ctx("0123456789ab", archive))
    assert seen == [True]
    assert not (adapter.scratch_dir / "submits" / "gpu-0123456789ab-1.pushing").exists()
    sim.register("gpu-router-0123456789ab-2", {"unavailable_n": 1})
    with pytest.raises(Unavailable):
        adapter.submit(_job("0123456789ab"), _ctx("0123456789ab", archive, n=2))
    assert not (adapter.scratch_dir / "submits" / "gpu-0123456789ab-2.pushing").exists()


# --------------------------------------------------------------------------- empty final log


def test_an_error_kernel_with_an_unpublished_log_is_not_judged_yet(
    adapter: KaggleAdapter, sim: SimKaggle, archive: Path, clock: FakeClock
) -> None:
    ctx = _ctx("b1b2c3d4e5f6", archive)
    sim.register(remote.slug_for_key(ctx.attempt_key) or "", {"duration": 5, "exit_code": 1})
    ref = adapter.submit(_job("b1b2c3d4e5f6"), ctx)
    clock.advance(30)
    sim.fail_next["kernels logs"] = CliResult(("kaggle",), 0, "\n", "")  # not published yet
    with pytest.raises(Unavailable, match="has not published its log"):
        adapter.status(ref)
    st = adapter.status(ref)  # the log is there now
    assert st.phase is RemotePhase.FAILED
    assert st.exit_code == 1
    assert any(chunk.lines for chunk in adapter.logs(ref))


def test_an_empty_final_log_is_judged_after_the_grace_period(
    adapter: KaggleAdapter, sim: SimKaggle, archive: Path, clock: FakeClock
) -> None:
    ctx = _ctx("b1b2c3d4e5f7", archive)
    sim.register(remote.slug_for_key(ctx.attempt_key) or "", {"duration": 5, "exit_code": 1})
    ref = adapter.submit(_job("b1b2c3d4e5f7"), ctx)
    clock.advance(30)
    empty = CliResult(("kaggle",), 0, "\n", "")
    sim.fail_next["kernels logs"] = empty
    with pytest.raises(Unavailable):
        adapter.status(ref)
    clock.advance(kaggle_mod.EMPTY_LOG_GRACE_S + 1)
    sim.fail_next["kernels logs"] = empty  # never published
    st = adapter.status(ref)
    assert st.phase is RemotePhase.LOST  # bounded: no exit line to go on
    assert adapter.status(ref) == st  # cached


# --------------------------------------------------------------------------- redaction


def test_the_cached_final_log_is_redacted_line_for_line(
    adapter: KaggleAdapter, sim: SimKaggle, archive: Path, clock: FakeClock
) -> None:
    ctx = _ctx("c1c2c3d4e5f6", archive)
    sim.register(remote.slug_for_key(ctx.attempt_key) or "", {"duration": 5})
    ref = adapter.submit(_job("c1c2c3d4e5f6"), ctx)
    clock.advance(30)
    token = "hf_" + "a" * 34
    secrets.register_for_redaction("s3cr3t-registered-value")
    lines = [
        "hello",
        f"leaked {token}",
        "value s3cr3t-registered-value",
        '::gpu:: {"t":"exit","code":0}',
    ]
    events = [{"stream_name": "stdout", "time": 1.0, "data": line + "\n"} for line in lines]
    sim.fail_next["kernels logs"] = CliResult(("kaggle",), 0, json.dumps(events) + "\n", "")
    assert adapter.status(ref).phase is RemotePhase.SUCCEEDED
    cached = adapter.scratch_dir / "final" / f"{remote.slug_for_key(ctx.attempt_key)}.json"
    text = cached.read_text()
    assert token not in text
    assert "s3cr3t-registered-value" not in text
    served = [line for chunk in adapter.logs(ref) for line in chunk.lines]
    assert len(served) == len(lines)
    assert served[0] == "hello"
