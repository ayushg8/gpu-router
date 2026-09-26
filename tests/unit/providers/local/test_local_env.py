"""LocalAdapter: python env choice (venv per deps key / system), installs, job environment."""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import pytest
from pydantic import SecretStr

from gpu_router.adapters.base import RemotePhase
from gpu_router.config import ProviderSettings
from gpu_router.errors import InvalidJob
from gpu_router.paths import Paths
from gpu_router.protocol import parse_line
from tests.unit.providers.local.helpers import (
    all_lines,
    build_project,
    fake_uv_calls,
    make_adapter,
    make_bundle,
    make_ctx,
    make_job,
    real_uv,
    run_dir,
    wait_phase,
    write_fake_uv,
)

DUMP_ENV = """
import json, os, sys, gpu
keys = ["GPU_ROUTER_CRASH_AT", "GPU_ROUTER_HOME", "GPU_EXIT_FILE", "VIRTUAL_ENV", "PYTHONPATH",
        "MY_TOKEN", "USER_KNOB", "PYTORCH_ENABLE_MPS_FALLBACK", "GPU_DATA_DIR",
        "GPU_OUTPUT_DIR", "GPU_ROUTER_JOB_ID", "GPU_ROUTER_ATTEMPT", "PATH"]
env = {k: os.environ.get(k) for k in keys}
env["executable"] = sys.executable
(gpu.output_dir() / "env.json").write_text(json.dumps(env))
print("token present", "MY_TOKEN" in os.environ)
"""


def _env_of(paths: Paths, tmp_path: Path, settings: ProviderSettings, **ctx_fields: object) -> dict:
    adapter = make_adapter(paths, settings)
    project = build_project(tmp_path, DUMP_ENV)
    job = make_job(project)
    ref = adapter.submit(job, make_ctx(job, make_bundle(paths, project), **ctx_fields))
    st = wait_phase(adapter, ref)
    assert st.phase is RemotePhase.SUCCEEDED, all_lines(adapter, ref)
    env: dict = json.loads((run_dir(adapter, ref) / "work" / "outputs" / "env.json").read_text())
    env["_run_dir"] = str(run_dir(adapter, ref))
    return env


def test_job_env_drops_daemon_internals_and_carries_job_env_and_secrets(
    paths: Paths, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GPU_ROUTER_CRASH_AT", "after_submit_return")
    monkeypatch.setenv("GPU_EXIT_FILE", "/tmp/elsewhere/EXIT")
    monkeypatch.setenv("VIRTUAL_ENV", "/nope/venv")
    monkeypatch.setenv("PYTHONPATH", "/nope/site")
    secret = "s3cr3t-value-for-local-test"
    env = _env_of(
        paths,
        tmp_path,
        ProviderSettings(env="system", python=sys.executable),
        env={"USER_KNOB": "7", "GPU_EXIT_FILE": "/tmp/also-not"},
        secrets={"MY_TOKEN": SecretStr(secret)},
    )
    assert env["GPU_ROUTER_CRASH_AT"] is None
    assert env["GPU_ROUTER_HOME"] is None
    assert env["GPU_EXIT_FILE"] is None
    assert env["VIRTUAL_ENV"] is None
    assert env["PYTHONPATH"] is not None
    assert "/nope/site" not in env["PYTHONPATH"]
    assert env["USER_KNOB"] == "7"
    assert env["MY_TOKEN"] == secret
    assert env["PYTORCH_ENABLE_MPS_FALLBACK"] == "1"
    assert env["GPU_DATA_DIR"] == str(paths.provider_dir("local") / "data")
    assert env["GPU_OUTPUT_DIR"] == str(Path(env["_run_dir"]) / "work" / "outputs")
    assert env["GPU_ROUTER_ATTEMPT"] == "1"
    # the secret never touches the run dir on disk (invariant 12)
    rd = Path(env["_run_dir"])
    for name in ("run.json", "launch.json", "console.log", "pid.json", "env.json"):
        assert secret not in (rd / name).read_text(), name


def test_job_can_override_mps_fallback(paths: Paths, tmp_path: Path) -> None:
    env = _env_of(
        paths,
        tmp_path,
        ProviderSettings(env="system", python=sys.executable),
        env={"PYTORCH_ENABLE_MPS_FALLBACK": "0"},
    )
    assert env["PYTORCH_ENABLE_MPS_FALLBACK"] == "0"


def test_venv_is_created_once_per_deps_key_and_reused(paths: Paths, tmp_path: Path) -> None:
    uv = write_fake_uv(tmp_path / "fake-uv")
    settings = ProviderSettings(uv=str(uv), base_packages=["numpy"])
    adapter = make_adapter(paths, settings)
    project = build_project(tmp_path, DUMP_ENV, requirements="requests==2.0\n")
    bundle = make_bundle(paths, project)
    refs = []
    for _ in range(2):
        job = make_job(project)
        ref = adapter.submit(job, make_ctx(job, bundle))
        assert wait_phase(adapter, ref).phase is RemotePhase.SUCCEEDED, all_lines(adapter, ref)
        refs.append(ref)
    calls = fake_uv_calls(uv)
    assert [c.split()[0] for c in calls] == ["venv", "pip"], calls
    assert "numpy" in calls[1]
    assert "-r" in calls[1]
    assert calls[1].rstrip().endswith("requirements.txt")
    envs = [json.loads((run_dir(adapter, r) / "work/outputs/env.json").read_text()) for r in refs]
    venv = envs[0]["VIRTUAL_ENV"]
    assert venv == envs[1]["VIRTUAL_ENV"]
    assert Path(venv).parent == paths.provider_dir("local") / "venvs"
    assert envs[0]["executable"].startswith(venv)
    assert envs[0]["PATH"].split(":")[0] == f"{venv}/bin"
    assert any("reusing python env" in line for line in all_lines(adapter, refs[1]))

    # different deps -> a different venv
    other = build_project(tmp_path, DUMP_ENV, requirements="requests==3.0\n", name="other")
    job = make_job(other)
    ref = adapter.submit(job, make_ctx(job, make_bundle(paths, other)))
    assert wait_phase(adapter, ref).phase is RemotePhase.SUCCEEDED
    env3 = json.loads((run_dir(adapter, ref) / "work/outputs/env.json").read_text())
    assert env3["VIRTUAL_ENV"] != venv


def test_install_failure_is_lost_with_exit_90(paths: Paths, tmp_path: Path) -> None:
    uv = write_fake_uv(tmp_path / "fake-uv", pip_exit=1)
    adapter = make_adapter(paths, ProviderSettings(uv=str(uv)))
    project = build_project(tmp_path, "print('never runs')\n", requirements="nope-pkg\n")
    job = make_job(project)
    ref = adapter.submit(job, make_ctx(job, make_bundle(paths, project)))
    st = wait_phase(adapter, ref)
    assert st.phase is RemotePhase.LOST
    assert st.exit_code == 90
    assert st.lost_reason is not None
    assert "install failed" in st.lost_reason
    lines = all_lines(adapter, ref)
    assert "never runs" not in lines
    assert any('"t":"install_failed","code":1' in line for line in lines)
    exits = [e for line in lines if (e := parse_line(line)) is not None and e.t == "exit"]
    assert [e.code for e in exits] == [90]
    # the half-made venv is not marked ready: the next run retries the install
    uv.write_text(uv.read_text().replace("sys.exit(1)", "sys.exit(0)"))
    job2 = make_job(project)
    ref2 = adapter.submit(job2, make_ctx(job2, make_bundle(paths, project)))
    assert wait_phase(adapter, ref2).phase is RemotePhase.SUCCEEDED
    assert any("incomplete; recreating" in line for line in all_lines(adapter, ref2))


def test_pip_fallback_without_uv(paths: Paths, tmp_path: Path) -> None:
    """uv: false -> stdlib venv (with pip); a bundle without deps installs nothing."""
    adapter = make_adapter(paths, ProviderSettings(uv=False))
    project = build_project(tmp_path, DUMP_ENV)
    job = make_job(project)
    ref = adapter.submit(job, make_ctx(job, make_bundle(paths, project)))
    assert wait_phase(adapter, ref, timeout_s=120).phase is RemotePhase.SUCCEEDED
    assert any("with venv from" in line for line in all_lines(adapter, ref))


@pytest.mark.skipif(real_uv() is None, reason="uv not installed")
def test_real_uv_venv_without_deps(paths: Paths, tmp_path: Path) -> None:
    env = _env_of(paths, tmp_path, ProviderSettings())
    assert env["VIRTUAL_ENV"] is not None
    assert Path(env["VIRTUAL_ENV"]).parent == paths.provider_dir("local") / "venvs"
    assert env["executable"].startswith(env["VIRTUAL_ENV"])


def test_system_env_ignores_deps_and_says_so(paths: Paths, tmp_path: Path) -> None:
    adapter = make_adapter(paths, ProviderSettings(env="system", python=sys.executable))
    project = build_project(tmp_path, "print('ran')\n", requirements="nope-pkg\n")
    job = make_job(project)
    ref = adapter.submit(job, make_ctx(job, make_bundle(paths, project)))
    assert wait_phase(adapter, ref).phase is RemotePhase.SUCCEEDED
    lines = all_lines(adapter, ref)
    assert "ran" in lines
    assert any("dependencies are not installed" in line for line in lines)


def test_per_job_provider_options_choose_the_env(paths: Paths, tmp_path: Path) -> None:
    adapter = make_adapter(paths, ProviderSettings(uv=False))
    project = build_project(tmp_path, DUMP_ENV)
    job = make_job(project, provider_options={"local": {"env": "system", "python": sys.executable}})
    ref = adapter.submit(job, make_ctx(job, make_bundle(paths, project)))
    assert wait_phase(adapter, ref).phase is RemotePhase.SUCCEEDED
    env = json.loads((run_dir(adapter, ref) / "work/outputs/env.json").read_text())
    assert env["VIRTUAL_ENV"] is None
    assert ref.meta["env"] == "system"


@pytest.mark.parametrize(
    "settings",
    [
        {"env": "conda"},
        {"env": "system"},
        {"env": "system", "python": "/no/such/python3"},
        {"python": "/no/such/python3"},
        {"uv": "/no/such/uv"},
        {"base_packages": 5},
    ],
)
def test_bad_env_settings_are_invalid_job_and_unhealthy(
    paths: Paths, tmp_path: Path, settings: dict
) -> None:
    adapter = make_adapter(paths, ProviderSettings(**settings))
    project = build_project(tmp_path, "print(1)\n")
    job = make_job(project)
    ctx = make_ctx(job, make_bundle(paths, project))
    with pytest.raises(InvalidJob) as info:
        adapter.submit(job, ctx)
    assert info.value.message
    assert adapter.lookup_by_key(ctx.attempt_key) is None
    health = adapter.healthcheck()
    if sys.platform == "darwin" and platform.machine() == "arm64":
        assert not health.ok
        assert health.reason
        assert health.hint or settings == {"base_packages": 5}
