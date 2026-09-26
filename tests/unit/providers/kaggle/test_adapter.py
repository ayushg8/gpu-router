"""KaggleAdapter behaviour beyond the shared contract: payload, idempotency, error
mapping, final-status judgement, credentials, fetch/cancel edge cases.

Most tests drive the real adapter over SimKaggle (tests/contract/kaggle/sim.py); the
error-mapping tests use a scripted runner that returns canned CLI output per command.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from gpu_router import secrets
from gpu_router.adapters.base import AttemptContext, RemotePhase, RemoteRef
from gpu_router.clock import FakeClock
from gpu_router.errors import (
    AuthRequired,
    InvalidJob,
    NotFound,
    Permanent,
    QuotaExhausted,
    RateLimited,
    Unavailable,
)
from gpu_router.models import Checkpoint, Job, JobSpec, JobState, ProviderHealth, Source
from gpu_router.paths import Paths
from gpu_router.providers.kaggle import remote
from gpu_router.providers.kaggle.adapter import KaggleAdapter
from gpu_router.providers.kaggle.cli import CliResult
from tests.contract.kaggle.sim import CANNOT_ACCESS, SimKaggle
from tests.contract.kaggle.targets import kaggle_deps

JOB_ID = "0123456789ab"
QUOTA_OK = json.dumps(
    [
        {
            "resource": "GPU",
            "used": "1.00h",
            "remaining": "29.00h",
            "total": "30.00h",
            "refreshAt": "2026-09-26T00:00:00",
        }
    ]
)
QUOTA_GONE = QUOTA_OK.replace('"1.00h"', '"30.00h"').replace('"29.00h"', '"0.00h"')


def make_job(**spec_fields: Any) -> Job:
    spec = JobSpec(project_dir="/tmp/proj", script="train.py", source=Source.API, **spec_fields)
    return Job(
        id=JOB_ID,
        short_id=JOB_ID[:4],
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


@pytest.fixture
def archive(tmp_path: Path) -> Path:
    p = tmp_path / "bundle.tar.gz"
    p.write_bytes(b"\x1f\x8bnot-really-a-bundle" * 10)
    return p


def make_ctx(archive: Path, n: int = 1, **fields: Any) -> AttemptContext:
    base: dict[str, Any] = {
        "attempt_id": f"{JOB_ID}.{n}",
        "attempt_key": f"gpu-{JOB_ID}-{n}",
        "n": n,
        "bundle_archive": archive,
        "env": {"GPU_ROUTER_JOB_ID": JOB_ID, "GPU_ROUTER_ATTEMPT": str(n)},
    }
    base.update(fields)
    return AttemptContext(**base)


@pytest.fixture
def sim(clock: FakeClock) -> SimKaggle:
    return SimKaggle(clock)


MakeAdapter = Callable[..., KaggleAdapter]


@pytest.fixture
def make_adapter(paths: Paths, clock: FakeClock, sim: SimKaggle) -> MakeAdapter:
    def build(runner: Any = None, **settings: Any) -> KaggleAdapter:
        return KaggleAdapter(kaggle_deps(paths, clock, **settings), runner=runner or sim)

    return build


class Scripted:
    """Runner answering by command prefix; records every argv and env."""

    def __init__(self, answers: Mapping[str, CliResult | list[CliResult]]) -> None:
        self.answers = {k: (v if isinstance(v, list) else [v]) for k, v in answers.items()}
        self.calls: list[list[str]] = []
        self.envs: list[dict[str, str]] = []
        self.timeouts: list[float] = []

    def __call__(
        self,
        argv: Sequence[str],
        *,
        timeout: float,
        env: Mapping[str, str],
        cwd: Path | None = None,
    ) -> CliResult:
        args = list(argv[2:])  # drop exe and -W
        self.calls.append(args)
        self.envs.append(dict(env))
        self.timeouts.append(timeout)
        for prefix in sorted(self.answers, key=len, reverse=True):
            if " ".join(args).startswith(prefix):
                queue = self.answers[prefix]
                res = queue.pop(0) if len(queue) > 1 else queue[0]
                return CliResult(tuple(argv), res.returncode, res.stdout, res.stderr)
        raise AssertionError(f"unexpected kaggle call {args}")

    def ops(self) -> list[str]:
        return [" ".join(c[:2]) for c in self.calls]


def ok(stdout: str) -> CliResult:
    return CliResult((), 0, stdout, "")


def fail(stderr: str = "", stdout: str = "", rc: int = 1) -> CliResult:
    return CliResult((), rc, stdout, stderr)


CONFIG = ok("Configuration values from /x\n- username: me\n- auth_method: LEGACY_API_KEY\n")
MISSING = fail(CANNOT_ACCESS.format(ref="me/x"))
PUSHED = ok(
    "Kernel version 1 successfully pushed.  Please check progress at "
    f"https://www.kaggle.com/code/me/gpu-router-{JOB_ID}-1\n"
)


def status_text(state: str, failure: str | None = None) -> CliResult:
    text = f'me/gpu-router-{JOB_ID}-1 has status "KernelWorkerStatus.{state}"\n'
    if failure:
        text += f'Failure message: "{failure}"\n'
    return ok(text)


def log_text(*lines: str) -> CliResult:
    return ok(json.dumps([{"stream_name": "stdout", "time": 1, "data": x + "\n"} for x in lines]))


REF = RemoteRef(remote_id=f"me/gpu-router-{JOB_ID}-1", meta={"gpu": "2xT4"})


# --------------------------------------------------------------------------- submit


def test_submit_pushes_a_private_gpu_kernel(
    make_adapter: MakeAdapter, sim: SimKaggle, archive: Path
) -> None:
    adapter = make_adapter()
    ref = adapter.submit(make_job(), make_ctx(archive, gpu="P100"))
    assert ref.remote_id == f"simuser/gpu-router-{JOB_ID}-1"
    assert ref.meta["gpu"] == "P100"
    assert ref.meta["machine_shape"] == "NvidiaTeslaP100"
    assert ref.meta["version"] == "1"
    kernel = sim.kernels[f"gpu-router-{JOB_ID}-1"]
    assert kernel.metadata["is_private"] is True
    assert kernel.metadata["title"] == f"gpu-router {JOB_ID} 1"
    assert kernel.metadata["machine_shape"] == "NvidiaTeslaP100"
    assert kernel.metadata["enable_internet"] is True
    assert kernel.timeout_s == 12 * 3600 - 300
    assert f"ATTEMPT_KEY = 'gpu-{JOB_ID}-1'" in kernel.run_py
    # the push folder (it holds the whole bundle) does not linger in scratch
    assert not any((adapter.scratch_dir / "push").iterdir())


def test_submit_is_idempotent_even_without_the_local_marker(
    make_adapter: MakeAdapter, sim: SimKaggle, archive: Path
) -> None:
    adapter = make_adapter()
    ctx = make_ctx(archive)
    first = adapter.submit(make_job(), ctx)
    assert adapter.submit(make_job(), ctx).remote_id == first.remote_id
    for marker in (adapter.scratch_dir / "submits").iterdir():
        marker.unlink()  # a crash between push and marker write
    fresh = make_adapter()
    assert fresh.submit(make_job(), ctx).remote_id == first.remote_id
    assert fresh.lookup_by_key(ctx.attempt_key) is not None
    assert sim.pushes == [f"gpu-router-{JOB_ID}-1"]  # never a second version


def test_submit_options_and_cpu(make_adapter: MakeAdapter, sim: SimKaggle, archive: Path) -> None:
    adapter = make_adapter()
    job = make_job(provider_options={"kaggle": {"accelerator": "none", "timeout_s": 900}})
    ref = adapter.submit(job, make_ctx(archive))
    kernel = sim.kernels[f"gpu-router-{JOB_ID}-1"]
    assert kernel.metadata["enable_gpu"] is False
    assert "machine_shape" not in kernel.metadata
    assert kernel.timeout_s == 900
    assert ref.meta["gpu"] == "cpu"
    assert not any(c[0] == "quota" for c in sim.calls)  # no GPU, no quota pre-check
    job = make_job(provider_options={"kaggle": {"timeout_s": 10**9, "enable_internet": False}})
    adapter.submit(job, make_ctx(archive, n=2))
    k2 = sim.kernels[f"gpu-router-{JOB_ID}-2"]
    assert k2.timeout_s == 12 * 3600 - 300  # clamped to the session cap
    assert k2.metadata["enable_internet"] is False
    assert k2.metadata["machine_shape"] == "NvidiaTeslaT4"  # default GPU


@pytest.mark.parametrize(
    ("job_kw", "ctx_kw", "match"),
    [
        ({"provider_options": {"kaggle": {"bogus": 1}}}, {}, "unknown provider_options"),
        ({"provider_options": {"kaggle": {"timeout_s": "soon"}}}, {}, "timeout_s"),
        ({}, {"gpu": "A100-40GB"}, "no A100-40GB GPUs"),
        ({}, {"bundle_archive": None}, "no bundle"),
    ],
)
def test_submit_refuses_what_kaggle_cannot_run(
    make_adapter: MakeAdapter,
    sim: SimKaggle,
    archive: Path,
    job_kw: dict[str, Any],
    ctx_kw: dict[str, Any],
    match: str,
) -> None:
    with pytest.raises(InvalidJob, match=match):
        make_adapter().submit(make_job(**job_kw), make_ctx(archive, **ctx_kw))
    assert sim.pushes == []


def test_submit_refuses_bundles_over_the_embed_limit(
    make_adapter: MakeAdapter, sim: SimKaggle, tmp_path: Path
) -> None:
    big = tmp_path / "big.tar.gz"
    big.write_bytes(b"0" * 300_000)
    adapter = make_adapter(max_embed_mb=0.1)
    assert adapter.capabilities.max_bundle_mb == pytest.approx(0.1)
    with pytest.raises(InvalidJob, match=r"kaggle takes up to 0\.1 MB"):
        adapter.submit(make_job(), make_ctx(big))
    assert sim.pushes == []


def test_quota_precheck_is_definitive(make_adapter: MakeAdapter, archive: Path) -> None:
    runner = Scripted({"config view": CONFIG, "kernels status": MISSING, "quota": ok(QUOTA_GONE)})
    with pytest.raises(QuotaExhausted) as info:
        make_adapter(runner).submit(make_job(), make_ctx(archive))
    assert info.value.resets_at == pytest.approx(1790380800.0)
    assert "kernels push" not in runner.ops()


def test_quota_outage_does_not_block_the_push(make_adapter: MakeAdapter, archive: Path) -> None:
    runner = Scripted(
        {
            "config view": CONFIG,
            "kernels status": MISSING,
            "quota": fail("503 Server Error: Service Unavailable"),
            "kernels push": PUSHED,
        }
    )
    ref = make_adapter(runner).submit(make_job(), make_ctx(archive))
    assert ref.remote_id == f"me/gpu-router-{JOB_ID}-1"
    assert ref.url == f"https://www.kaggle.com/code/me/gpu-router-{JOB_ID}-1"


@pytest.mark.parametrize(
    ("message", "kind"),
    [
        ("You have exceeded your weekly GPU quota", QuotaExhausted),
        ("Phone verification is required to use GPUs", AuthRequired),
        ("Maximum number of concurrent GPU sessions reached", RateLimited),
        ("This notebook violates the terms of service", Permanent),
        ("Invalid machine shape", InvalidJob),
        ("Something new", InvalidJob),
    ],
)
def test_push_errors_are_definitive_when_nothing_exists(
    make_adapter: MakeAdapter, archive: Path, message: str, kind: type[Exception]
) -> None:
    runner = Scripted(
        {
            "config view": CONFIG,
            "kernels status": MISSING,
            "quota": ok(QUOTA_OK),
            "kernels push": ok(f"Kernel push error: {message}\n"),  # exit 0, like the CLI
        }
    )
    with pytest.raises(kind):
        make_adapter(runner).submit(make_job(), make_ctx(archive))


def test_push_error_but_kernel_exists_is_ambiguous(
    make_adapter: MakeAdapter, archive: Path
) -> None:
    runner = Scripted(
        {
            "config view": CONFIG,
            "kernels status": [MISSING, status_text("QUEUED")],
            "quota": ok(QUOTA_OK),
            "kernels push": ok("Kernel push error: Invalid machine shape\n"),
        }
    )
    with pytest.raises(Unavailable, match="exists"):
        make_adapter(runner).submit(make_job(), make_ctx(archive))


@pytest.mark.parametrize(
    ("result", "kind"),
    [
        (fail("Title must be at least five characters"), InvalidJob),
        (fail("403 Client Error: Forbidden for url: .../SaveKernel"), AuthRequired),
        (fail(stdout="Authentication required to call the Kaggle API.\n"), AuthRequired),
        (fail("429 Client Error: Too Many Requests"), RateLimited),
        (fail("requests.exceptions.ConnectionError: Max retries exceeded"), Unavailable),
        (fail("500 Server Error"), Unavailable),
        (ok("something unexpected\n"), Unavailable),
    ],
)
def test_push_failures_without_a_server_answer(
    make_adapter: MakeAdapter, archive: Path, result: CliResult, kind: type[Exception]
) -> None:
    runner = Scripted(
        {
            "config view": CONFIG,
            "kernels status": MISSING,
            "quota": ok(QUOTA_OK),
            "kernels push": result,
        }
    )
    with pytest.raises(kind):
        make_adapter(runner).submit(make_job(), make_ctx(archive))


def test_resume_from_a_local_checkpoint_is_embedded(
    make_adapter: MakeAdapter, sim: SimKaggle, archive: Path, tmp_path: Path
) -> None:
    ckpt_file = tmp_path / "ckpt-4.tar.gz"
    ckpt_file.write_bytes(b"checkpoint-bytes")
    ckpt = Checkpoint(
        id=f"{JOB_ID}.c4",
        job_id=JOB_ID,
        attempt_id=f"{JOB_ID}.1",
        seq=4,
        uri=ckpt_file.as_uri(),
        created_at=0,
        recorded_at=0,
    )
    make_adapter().submit(make_job(), make_ctx(archive, n=2, resume_from=ckpt))
    run_py = sim.kernels[f"gpu-router-{JOB_ID}-2"].run_py
    assert "CKPT_SEQ_START = 5" in run_py
    assert "RESUME_SHA256 = None" not in run_py
    remote_ckpt = ckpt.model_copy(update={"uri": "file:///kaggle/working/.gpu-router/ckpt-4"})
    make_adapter().submit(make_job(), make_ctx(archive, n=3, resume_from=remote_ckpt))
    run_py = sim.kernels[f"gpu-router-{JOB_ID}-3"].run_py
    assert "RESUME_SHA256 = None" in run_py
    assert "cannot reach kaggle yet; starting fresh" in run_py


# --------------------------------------------------------------------------- status


def _finished(make_adapter: MakeAdapter, status: CliResult, log: CliResult, quota: str = QUOTA_OK):
    runner = Scripted({"kernels status": status, "kernels logs": log, "quota": ok(quota)})
    adapter = make_adapter(runner)
    return adapter, runner, adapter.status(REF)


def test_status_install_failure_is_lost(make_adapter: MakeAdapter) -> None:
    _, _, st = _finished(
        make_adapter, status_text("ERROR"), log_text('::gpu:: {"t":"exit","code":90}')
    )
    assert st.phase is RemotePhase.LOST
    assert st.lost_reason == "dependency install failed on kaggle"


def test_status_time_limit_is_lost(make_adapter: MakeAdapter) -> None:
    _, _, st = _finished(
        make_adapter,
        status_text("ERROR", "Your notebook exceeded the maximum run time"),
        log_text("working..."),
    )
    assert st.phase is RemotePhase.LOST
    assert st.lost_reason == "kaggle session time limit"
    assert not st.quota_exhausted


def test_status_quota_death_is_lost_and_exhausted(make_adapter: MakeAdapter) -> None:
    _, _, st = _finished(make_adapter, status_text("ERROR"), log_text("working..."), QUOTA_GONE)
    assert st.phase is RemotePhase.LOST
    assert st.quota_exhausted


def test_status_complete_without_exit_line_is_lost(make_adapter: MakeAdapter) -> None:
    _, _, st = _finished(make_adapter, status_text("COMPLETE"), log_text("no marker"))
    assert st.phase is RemotePhase.LOST
    assert st.lost_reason is not None
    assert "never reported" in st.lost_reason


def test_status_complete_before_the_log_exists_is_transient(make_adapter: MakeAdapter) -> None:
    runner = Scripted({"kernels status": status_text("COMPLETE"), "kernels logs": ok("\n")})
    with pytest.raises(Unavailable, match="has not published its log"):
        make_adapter(runner).status(REF)


def test_final_status_is_cached(make_adapter: MakeAdapter) -> None:
    adapter, runner, st = _finished(
        make_adapter, status_text("COMPLETE"), log_text("hi", '::gpu:: {"t":"exit","code":0}')
    )
    assert st.phase is RemotePhase.SUCCEEDED
    n = len(runner.calls)
    assert adapter.status(REF) == st
    assert next(adapter.logs(REF)).lines[0] == "hi"
    assert len(runner.calls) == n  # nothing more asked of kaggle


def test_status_kernel_cancelled_on_kaggle(make_adapter: MakeAdapter) -> None:
    runner = Scripted(
        {"kernels status": status_text("CANCEL_ACKNOWLEDGED"), "kernels logs": fail("500")}
    )
    st = make_adapter(runner).status(REF)
    assert st.phase is RemotePhase.CANCELLED


def test_missing_kernel_with_broken_auth_is_not_not_found(make_adapter: MakeAdapter) -> None:
    """Kernel-scoped 401/403 look like a missing kernel; NotFound would mark the attempt
    lost and rerun the job, so auth is proven first."""
    runner = Scripted(
        {
            "kernels status": MISSING,
            "quota": fail(stdout="Authentication required to call the Kaggle API.\n"),
        }
    )
    with pytest.raises(AuthRequired):
        make_adapter(runner).status(REF)
    runner = Scripted({"kernels status": MISSING, "quota": ok(QUOTA_OK)})
    with pytest.raises(NotFound):
        make_adapter(runner).status(REF)


def test_status_garbage_is_unavailable(make_adapter: MakeAdapter) -> None:
    runner = Scripted({"kernels status": ok("hello\n")})
    with pytest.raises(Unavailable, match="unexpected"):
        make_adapter(runner).status(REF)


def test_malformed_refs_are_not_found_without_calls(make_adapter: MakeAdapter) -> None:
    runner = Scripted({})
    adapter = make_adapter(runner)
    for rid in ("does-not-exist-000", "me/someone-elses-kernel", "../x/gpu-router-0-1"):
        with pytest.raises(NotFound):
            adapter.status(RemoteRef(remote_id=rid))
    assert adapter.lookup_by_key("fk-whatever") is None
    assert runner.calls == []


# --------------------------------------------------------------------------- logs / fetch


def test_logs_while_running_keep_the_cursor(make_adapter: MakeAdapter) -> None:
    runner = Scripted({"kernels status": status_text("RUNNING")})
    chunks = list(make_adapter(runner).logs(REF, since="7"))
    assert len(chunks) == 1
    assert chunks[0].lines == []
    assert chunks[0].cursor == "7"
    assert not chunks[0].eof


def test_logs_chunking_and_bad_cursor(make_adapter: MakeAdapter) -> None:
    lines = [f"line {i}" for i in range(2500)] + ['::gpu:: {"t":"exit","code":0}']
    adapter, _, _ = _finished(make_adapter, status_text("COMPLETE"), log_text(*lines))
    chunks = list(adapter.logs(REF))
    assert [len(c.lines) for c in chunks] == [1000, 1000, 501]
    assert [c.eof for c in chunks] == [False, False, True]
    assert chunks[-1].cursor == "2501"
    assert next(adapter.logs(REF, since="garbage")).lines[0] == "line 0"
    past = list(adapter.logs(REF, since="9999"))
    assert past[0].lines == []
    assert past[0].eof


def test_fetch_copies_only_outputs_and_keeps_user_files(
    make_adapter: MakeAdapter, sim: SimKaggle, archive: Path, clock: FakeClock, tmp_path: Path
) -> None:
    adapter = make_adapter()
    ref = adapter.submit(make_job(), make_ctx(archive))
    with pytest.raises(NotFound, match="outputs appear when it finishes"):
        adapter.fetch(ref, tmp_path / "early")
    clock.advance(30)
    assert adapter.status(ref).phase is RemotePhase.SUCCEEDED
    dest = tmp_path / "runs" / "0123"
    dest.mkdir(parents=True)
    (dest / "mine.txt").write_text("keep")
    res = adapter.fetch(ref, dest)
    assert res.files == 2
    assert (dest / "result.json").is_file()
    assert (dest / "model" / "weights.txt").is_file()
    assert not (dest / ".gpu-router").exists()
    assert (dest / "mine.txt").read_text() == "keep"
    assert adapter.fetch(ref, dest).files == 2
    assert not any((adapter.scratch_dir / "fetch").iterdir())
    output_call = next(c for c in sim.calls if c[:2] == ("kernels", "output"))
    assert remote.OUTPUT_PATTERN in output_call


# --------------------------------------------------------------------------- cancel


def test_cancel_finished_run_keeps_the_kernel(
    make_adapter: MakeAdapter, sim: SimKaggle, archive: Path, clock: FakeClock
) -> None:
    adapter = make_adapter()
    ref = adapter.submit(make_job(), make_ctx(archive))
    clock.advance(30)
    adapter.cancel(ref)
    assert sim.kernels[f"gpu-router-{JOB_ID}-1"].deleted_at is None
    assert adapter.status(ref).phase is RemotePhase.SUCCEEDED


def test_cancel_running_run_deletes_and_reports_cancelled(
    make_adapter: MakeAdapter, sim: SimKaggle, archive: Path, clock: FakeClock, tmp_path: Path
) -> None:
    adapter = make_adapter()
    ctx = make_ctx(archive)
    ref = adapter.submit(make_job(), ctx)
    clock.advance(2)
    adapter.cancel(ref)
    assert sim.kernels[f"gpu-router-{JOB_ID}-1"].deleted_at is not None
    assert adapter.status(ref).phase is RemotePhase.CANCELLED
    chunks = list(adapter.logs(ref))
    assert chunks[-1].eof
    with pytest.raises(NotFound, match="deleted by cancel"):
        adapter.fetch(ref, tmp_path / "out")
    adapter.cancel(ref)  # idempotent, no second delete
    assert sum(1 for c in sim.calls if c[:2] == ("kernels", "delete")) == 1
    assert adapter.lookup_by_key(ctx.attempt_key) is not None  # the local marker remains


def test_cancel_delete_outage_raises_unavailable(make_adapter: MakeAdapter) -> None:
    runner = Scripted(
        {"kernels status": status_text("RUNNING"), "kernels delete": fail("502 Server Error")}
    )
    with pytest.raises(Unavailable):
        make_adapter(runner).cancel(REF)


# --------------------------------------------------------------------------- quota / health


def test_quota_is_live(make_adapter: MakeAdapter, clock: FakeClock) -> None:
    runner = Scripted({"quota": ok(QUOTA_OK)})
    q = make_adapter(runner).quota()
    assert q.provider == "kaggle"
    assert q.used == pytest.approx(1)
    assert q.limit == pytest.approx(30)
    assert q.source == "live"
    assert q.resets_at == pytest.approx(1790380800.0)
    assert q.detail["remaining_h"] == pytest.approx(29)
    assert q.observed_at == clock.now()


def test_quota_garbage_is_unavailable(make_adapter: MakeAdapter) -> None:
    with pytest.raises(Unavailable):
        make_adapter(Scripted({"quota": ok("No quota information available\n")})).quota()


def test_healthcheck_ok_and_detail(make_adapter: MakeAdapter) -> None:
    runner = Scripted(
        {"--version": ok("Kaggle CLI 2.2.4\n"), "config view": CONFIG, "quota": ok(QUOTA_OK)}
    )
    h = make_adapter(runner).healthcheck()
    assert h.ok
    assert h.detail["cli_version"] == "2.2.4"
    assert h.detail["username"] == "me"
    assert h.detail["credentials"] == "cli"
    assert h.detail["gpu_limit_h"] == pytest.approx(30)


def test_healthcheck_explains_problems(
    make_adapter: MakeAdapter, monkeypatch: pytest.MonkeyPatch
) -> None:
    auth = Scripted(
        {
            "--version": ok("Kaggle CLI 2.2.4\n"),
            "config view": fail(stdout="Authentication required to call the Kaggle API.\n"),
        }
    )
    h = make_adapter(auth).healthcheck()
    assert h.health is ProviderHealth.AUTH_REQUIRED
    assert h.reason
    assert h.hint
    down = Scripted(
        {
            "--version": ok("Kaggle CLI 2.2.4\n"),
            "config view": CONFIG,
            "quota": fail("503 Server Error"),
        }
    )
    assert make_adapter(down).healthcheck().health is ProviderHealth.UNAVAILABLE
    weird = Scripted({"--version": ok("zsh: command not found\n")})
    assert make_adapter(weird).healthcheck().health is ProviderHealth.UNAVAILABLE
    monkeypatch.setattr(
        "gpu_router.providers.kaggle.adapter.find_executable", lambda configured=None: None
    )
    missing = make_adapter(Scripted({})).healthcheck()
    assert missing.health is ProviderHealth.AUTH_REQUIRED
    assert "uv tool install kaggle" in (missing.hint or "")


# --------------------------------------------------------------------------- credentials


def test_keychain_credentials_go_to_the_env_only(
    make_adapter: MakeAdapter, archive: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key = "0123456789abcdef0123456789abcdef"
    secrets.set_secret("kaggle", json.dumps({"username": "kc-user", "key": key}))
    monkeypatch.setenv("KAGGLE_KEY", "stale-inherited")
    runner = Scripted({"kernels status": MISSING, "quota": ok(QUOTA_OK), "kernels push": PUSHED})
    adapter = make_adapter(runner)
    ref = adapter.submit(make_job(), make_ctx(archive))
    assert ref.remote_id == f"kc-user/gpu-router-{JOB_ID}-1"
    assert "config view" not in runner.ops()  # username came from the Keychain doc
    for argv, env in zip(runner.calls, runner.envs, strict=True):
        assert key not in " ".join(argv)
        assert env["KAGGLE_KEY"] == key
        assert env["KAGGLE_USERNAME"] == "kc-user"
        assert env["KAGGLE_CONFIG_DIR"].endswith("cli-config")
    assert key not in ref.model_dump_json()
    health = make_adapter(
        Scripted({"--version": ok("Kaggle CLI 2.2.4\n"), "quota": ok(QUOTA_OK)})
    ).healthcheck()
    assert health.detail["credentials"] == "keychain-json"


def test_keychain_token_and_modes(
    make_adapter: MakeAdapter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)
    monkeypatch.delenv("KAGGLE_CONFIG_DIR", raising=False)
    secrets.set_secret("KAGGLE_API_TOKEN", "tok-" + "x" * 30)
    runner = Scripted({"quota": ok(QUOTA_OK)})
    make_adapter(runner).quota()
    assert runner.envs[0]["KAGGLE_API_TOKEN"].startswith("tok-")
    runner = Scripted({"quota": ok(QUOTA_OK)})
    make_adapter(runner, credentials="cli").quota()
    assert "KAGGLE_API_TOKEN" not in runner.envs[0]  # "cli" never reads the Keychain
    assert "KAGGLE_CONFIG_DIR" not in runner.envs[0]


def test_keychain_mode_without_secret_and_bad_doc(make_adapter: MakeAdapter) -> None:
    with pytest.raises(AuthRequired, match="no kaggle credentials in the Keychain"):
        make_adapter(Scripted({"quota": ok(QUOTA_OK)}), credentials="keychain").quota()
    secrets.set_secret("kaggle", "not json")
    with pytest.raises(AuthRequired, match=r"not a kaggle\.json document"):
        make_adapter(Scripted({"quota": ok(QUOTA_OK)})).quota()
    with pytest.raises(AuthRequired, match="unknown kaggle credentials mode"):
        make_adapter(Scripted({"quota": ok(QUOTA_OK)}), credentials="magic").quota()


def test_configured_username_skips_config_view(make_adapter: MakeAdapter, archive: Path) -> None:
    runner = Scripted({"kernels status": MISSING, "quota": ok(QUOTA_OK), "kernels push": PUSHED})
    ref = make_adapter(runner, username="configured").submit(make_job(), make_ctx(archive))
    assert ref.meta["owner"] == "configured"
    assert "config view" not in runner.ops()


def test_capabilities_follow_the_catalog(make_adapter: MakeAdapter) -> None:
    caps = make_adapter().capabilities
    assert caps.lookup_by_key
    assert caps.live_quota
    assert not caps.live_logs
    assert caps.cancel_confirms  # delete + tombstone; verified live to stop the session
    assert caps.resume  # phase 5: the runner resumes from checkpoint storage
    assert caps.max_session_hours == 12
    assert caps.poll_interval_s == 60


def test_every_method_fits_its_engine_timeout(make_adapter: MakeAdapter, archive: Path) -> None:
    """Rule A2: the CLI timeouts a method can accumulate stay below the engine's budget."""
    from gpu_router.config import CallTimeouts

    budget = CallTimeouts()
    runner = Scripted(
        {
            "config view": CONFIG,
            "kernels status": [MISSING, MISSING],
            "quota": ok(QUOTA_OK),
            "kernels push": ok("Kernel push error: Invalid machine shape\n"),
        }
    )
    with pytest.raises(InvalidJob):
        make_adapter(runner).submit(make_job(), make_ctx(archive))
    assert sum(runner.timeouts) < budget.submit
    runner = Scripted(
        {
            "kernels status": status_text("ERROR"),
            "kernels logs": log_text("x"),
            "quota": ok(QUOTA_OK),
        }
    )
    make_adapter(runner).status(REF)
    assert runner.ops() == ["kernels status", "kernels logs", "quota --format"]
    assert sum(runner.timeouts) < budget.status
    runner = Scripted({"kernels status": status_text("RUNNING"), "kernels delete": ok("deleted\n")})
    make_adapter(runner).cancel(REF)
    assert sum(runner.timeouts) < budget.cancel
    runner = Scripted(
        {"--version": ok("Kaggle CLI 2.2.4\n"), "config view": CONFIG, "quota": ok(QUOTA_OK)}
    )
    make_adapter(runner).healthcheck()
    assert sum(runner.timeouts) < budget.healthcheck
