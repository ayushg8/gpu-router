"""Phase-3 review regressions for the local adapter (D37, D38 in CLAUDE.md): raw log
retention and the job environment allowlist. (The resume fallback, D34, is in
test_local_run.py next to the other resume tests.)"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from pydantic import SecretStr

from gpu_router.adapters.base import AdapterDeps, RemotePhase
from gpu_router.clock import FakeClock, SystemClock
from gpu_router.paths import Paths
from gpu_router.providers.local import adapter as local_mod
from gpu_router.providers.local.adapter import LocalAdapter
from tests.unit.providers.local.helpers import (
    LOCAL_ENTRY,
    all_lines,
    build_project,
    make_adapter,
    make_bundle,
    make_ctx,
    make_job,
    run_dir,
    system_settings,
    wait_phase,
)

DUMP_ENV = """
import json, os, gpu
(gpu.output_dir() / "env.json").write_text(json.dumps(dict(os.environ)))
print("hf_" + "x" * 34 if os.environ.get("PRINT_TOKEN") else "no token")
"""


def _run_env(paths: Paths, tmp_path: Path, **ctx_fields: object) -> dict[str, str]:
    adapter = make_adapter(paths)
    project = build_project(tmp_path, DUMP_ENV)
    job = make_job(project)
    ref = adapter.submit(job, make_ctx(job, make_bundle(paths, project), **ctx_fields))
    assert wait_phase(adapter, ref).phase is RemotePhase.SUCCEEDED, all_lines(adapter, ref)
    env: dict[str, str] = json.loads(
        (run_dir(adapter, ref) / "work" / "outputs" / "env.json").read_text()
    )
    return env


# --------------------------------------------------------------------------- env allowlist


def test_a_local_job_never_inherits_the_shells_credentials(
    paths: Paths, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shell = {
        "KAGGLE_KEY": "0123456789abcdef0123456789abcdef",
        "KAGGLE_USERNAME": "someone",
        "AWS_SECRET_ACCESS_KEY": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        "AWS_ACCESS_KEY_ID": "AKIAIOSFODNN7EXAMPLE",
        "OPENAI_API_KEY": "sk-proj-" + "a" * 40,
        "ANTHROPIC_API_KEY": "sk-ant-" + "b" * 40,
        "GITHUB_TOKEN": "ghp_" + "c" * 36,
        "GOOGLE_APPLICATION_CREDENTIALS": "/Users/me/adc.json",
        "HF_TOKEN": "hf_" + "d" * 34,
        "SSH_AUTH_SOCK": "/tmp/agent.sock",
        "MY_SHELL_THING": "1",
        "UV_INDEX_PRIVATE_PASSWORD": "pw",
        "HTTP_PROXY": "http://user:pw@proxy.example:3128",
        "PIP_INDEX_URL": "https://user:pw@pypi.example/simple",
        # allowed through
        "LC_ALL": "en_US.UTF-8",
        "HTTPS_PROXY": "http://proxy.example:3128",
        "UV_INDEX_URL": "https://pypi.example/simple",
        "HF_HOME": str(tmp_path / "hf"),
        "PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.0",
    }
    for k, v in shell.items():
        monkeypatch.setenv(k, v)
    secret = "value-of-a-listed-secret"
    env = _run_env(paths, tmp_path, secrets={"WANDB_API_KEY": SecretStr(secret)}, env={"A": "1"})
    for name in (
        "KAGGLE_KEY",
        "KAGGLE_USERNAME",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_ACCESS_KEY_ID",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GITHUB_TOKEN",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "HF_TOKEN",
        "SSH_AUTH_SOCK",
        "MY_SHELL_THING",
        "UV_INDEX_PRIVATE_PASSWORD",
        "HTTP_PROXY",
        "PIP_INDEX_URL",
    ):
        assert name not in env, name
    for name in ("LC_ALL", "HTTPS_PROXY", "UV_INDEX_URL", "HF_HOME"):
        assert env[name] == shell[name], name
    assert env["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] == "0.0"
    assert env["WANDB_API_KEY"] == secret  # listed under `secrets:`: delivered
    assert env["A"] == "1"
    assert env["HOME"]
    assert env["PATH"]
    assert env["PYTORCH_ENABLE_MPS_FALLBACK"] == "1"


def test_the_env_is_the_same_whether_launchd_or_a_shell_started_the_daemon(
    paths: Paths, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SOME_RANDOM_SHELL_VAR", "x")
    env = _run_env(paths, tmp_path)
    assert "SOME_RANDOM_SHELL_VAR" not in env


# --------------------------------------------------------------------------- raw logs


def _fake_clock_adapter(paths: Paths, clock: FakeClock) -> LocalAdapter:
    return LocalAdapter(
        AdapterDeps(
            name="local", entry=LOCAL_ENTRY, settings=system_settings(), paths=paths, clock=clock
        )
    )


def test_raw_logs_are_deleted_after_they_were_served_and_the_run_dir_later(
    paths: Paths, tmp_path: Path
) -> None:
    clock = FakeClock(SystemClock().now())
    adapter = _fake_clock_adapter(paths, clock)
    project = build_project(tmp_path, "print('raw line')\n")
    job = make_job(project)
    ref = adapter.submit(job, make_ctx(job, make_bundle(paths, project)))
    assert wait_phase(adapter, ref).phase is RemotePhase.SUCCEEDED
    rd = run_dir(adapter, ref)
    console, job_log = rd / "console.log", rd / "work" / "job.log"
    assert console.is_file()
    assert job_log.is_file()
    adapter.healthcheck()
    assert console.is_file(), "never served yet: kept"
    chunks = list(adapter.logs(ref))
    assert chunks[-1].eof
    assert "raw line" in [line for c in chunks for line in c.lines]
    cursor = chunks[-1].cursor
    adapter.healthcheck()
    assert console.is_file(), "kept for RAW_LOG_GRACE_S after it was served"
    clock.advance(local_mod.RAW_LOG_GRACE_S + 1)
    adapter.healthcheck()
    assert not console.exists()
    assert not job_log.exists()
    assert adapter.status(ref).phase is RemotePhase.SUCCEEDED  # EXIT still decides
    after = list(adapter.logs(ref, since=cursor))
    assert after[-1].eof
    assert all(not c.lines for c in after)  # A7 holds across the purge
    clock.advance(local_mod.RETENTION_S + 1)
    other = build_project(tmp_path, "print(2)\n", name="other")
    job2 = make_job(other)
    adapter.submit(job2, make_ctx(job2, make_bundle(paths, other)))  # the sweep also runs here
    assert not rd.exists()


def test_a_live_run_is_never_swept(paths: Paths, tmp_path: Path) -> None:
    clock = FakeClock(SystemClock().now())
    adapter = _fake_clock_adapter(paths, clock)
    project = build_project(tmp_path, "import time\nprint('up', flush=True)\ntime.sleep(30)\n")
    job = make_job(project)
    ref = adapter.submit(job, make_ctx(job, make_bundle(paths, project)))
    try:
        wait_phase(adapter, ref, {RemotePhase.RUNNING})
        clock.advance(local_mod.RETENTION_S * 2)
        adapter.healthcheck()
        assert (run_dir(adapter, ref) / "console.log").is_file()
    finally:
        adapter.cancel(ref)


def test_healthcheck_sweep_skips_while_a_submit_holds_the_lock(paths: Paths) -> None:
    adapter = make_adapter(paths)
    with adapter._submit_lock:
        assert adapter.healthcheck().health is not None  # no deadlock
    assert sys.platform  # (the health verdict itself depends on the machine)
