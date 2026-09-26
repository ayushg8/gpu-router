"""The real lightning-sdk still has every name and parameter driver.py and the fake SDK rely
on. Runs the pinned SDK from uv's cache with `--offline` (no network, no account, no
credentials: nothing here talks to lightning.ai); skipped when uv or the cached SDK is
not available."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from gpu_router.providers.lightning.sdk import DEFAULT_SDK_VERSION, SDK_PYTHON, find_uv

PROBE = r"""
import inspect, json, os
import lightning_sdk as sdk
from lightning_sdk.utils import resolve
from lightning_sdk.lightning_cloud import rest_client, env
from lightning_sdk.lightning_cloud.openapi.rest import ApiException
from lightning_sdk.filesystem import Filesystem

def params(fn):
    return sorted(inspect.signature(fn).parameters)

out = {
    "version": sdk.__version__,
    "job_run": params(sdk.Job.run),
    "job_init": params(sdk.Job.__init__),
    "job_attrs": [a for a in ("status", "started_at", "stopped_at", "total_cost", "id", "link",
                              "logs", "name", "stop", "delete", "list_artifacts",
                              "download_artifacts") if hasattr(sdk.Job, a)],
    "prevent_refetch": "_prevent_refetch_latest" in inspect.getsource(sdk.Job),
    "ts_upload": params(sdk.Teamspace.upload_file),
    "ts_download": params(sdk.Teamspace.download_file),
    "ts_machines": params(sdk.Teamspace.list_machines),
    "ts_list_jobs": params(sdk.Teamspace.list_jobs),
    "studio_init": params(sdk.Studio.__init__),
    "studio_cloud_account": isinstance(
        inspect.getattr_static(sdk.Studio, "cloud_account"), property
    ),
    "studio_skip_setup": hasattr(sdk.Studio, "_skip_setup"),
    "machines": [m for m in ("T4", "L4", "T4_SMALL") if hasattr(sdk.Machine, m)],
    "statuses": sorted(s.value for s in sdk.Status),
    "resolve": [f for f in ("_get_authed_user", "skip_studio_setup") if hasattr(resolve, f)],
    "client_init": params(rest_client.LightningClient.__init__),
    "balance": hasattr(rest_client.LightningClient, "billing_service_get_user_balance"),
    "api_exception": sorted(params(ApiException.__init__)),
    "api_exception_body": "self.body" in inspect.getsource(ApiException),
    "fs_rm": params(Filesystem.rm),
    "credential_path": env.LIGHTNING_CREDENTIAL_PATH,
    "settings_path": env.LIGHTNING_SETTINGS_PATH,
    "home": os.path.expanduser("~"),
}
print("PROBE:" + json.dumps(out))
"""


@pytest.fixture(scope="module")
def shape(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    uv = find_uv()
    if uv is None:
        pytest.skip("uv not installed")
    home = tmp_path_factory.mktemp("sdk-home")
    env = dict(os.environ)
    for key in [k for k in env if k.startswith("LIGHTNING_")]:
        env.pop(key)
    env.update(
        {
            "LIGHTNING_DISABLE_VERSION_CHECK": "1",
            "LIGHTNING_CREDENTIAL_PATH": str(home / ".lightning" / "credentials.json"),
            "LIGHTNING_SETTINGS_PATH": str(home / ".lightning" / "settings.json"),
            "BROWSER": "true",
            "UV_OFFLINE": "1",
        }
    )
    try:
        res = subprocess.run(
            [
                uv,
                "run",
                "--offline",
                "--no-project",
                "--quiet",
                "--python",
                SDK_PYTHON,
                "--with",
                f"lightning-sdk=={DEFAULT_SDK_VERSION}",
                "python",
                "-c",
                PROBE,
            ],
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
            check=False,
        )
    except subprocess.TimeoutExpired:
        pytest.skip("uv took too long to prepare the SDK")
    line = next((x for x in res.stdout.splitlines() if x.startswith("PROBE:")), None)
    if line is None:
        pytest.skip(f"the pinned SDK is not in uv's cache: {res.stderr.strip()[-200:]}")
    data: dict[str, object] = json.loads(line[len("PROBE:") :])
    return data


@pytest.mark.slow
def test_job_api(shape: dict[str, object]) -> None:
    assert shape["version"] == DEFAULT_SDK_VERSION
    for p in (
        "name",
        "machine",
        "command",
        "studio",
        "teamspace",
        "env",
        "interruptible",
        "max_run_attempts",
    ):
        assert p in shape["job_run"], p  # type: ignore[operator]
    assert {"name", "teamspace"} <= set(shape["job_init"])  # type: ignore[arg-type]
    assert len(shape["job_attrs"]) == 12  # type: ignore[arg-type]
    assert shape["prevent_refetch"]


@pytest.mark.slow
def test_teamspace_studio_machine_status(shape: dict[str, object]) -> None:
    assert {"file_path", "remote_path", "progress_bar", "cloud_account"} <= set(shape["ts_upload"])  # type: ignore[arg-type]
    assert {"remote_path", "file_path"} <= set(shape["ts_download"])  # type: ignore[arg-type]
    assert "machine" in shape["ts_machines"]  # type: ignore[operator]
    assert {"name", "teamspace", "create_ok"} <= set(shape["studio_init"])  # type: ignore[arg-type]
    assert shape["studio_cloud_account"]
    assert shape["studio_skip_setup"]
    assert shape["machines"] == ["T4", "L4", "T4_SMALL"]
    assert shape["statuses"] == sorted(
        ["NotCreated", "Pending", "Running", "Stopping", "Stopped", "Completed", "Failed"]
    )


@pytest.mark.slow
def test_auth_billing_filesystem_and_paths(shape: dict[str, object]) -> None:
    assert shape["resolve"] == ["_get_authed_user", "skip_studio_setup"]
    assert "retry" in shape["client_init"]  # type: ignore[operator]
    assert shape["balance"]
    assert {"status", "reason", "http_resp"} <= set(shape["api_exception"])  # type: ignore[arg-type]
    assert shape["api_exception_body"]
    assert {"path", "recursive"} <= set(shape["fs_rm"])  # type: ignore[arg-type]
    # the SDK honours the path overrides the adapter sets, so ~/.lightning is never used
    assert "/.lightning/" in str(shape["credential_path"])
    assert not str(shape["credential_path"]).startswith(str(Path.home() / ".lightning"))
    assert not str(shape["settings_path"]).startswith(str(Path.home() / ".lightning"))
