"""driver.py against a fake lightning_sdk (fake_sdk/, same names and signatures as the real
2026.9.18.post1 SDK): the real driver runs in a real subprocess through SdkBridge +
SubprocessRunner, exactly as the adapter runs it, with the fake on PYTHONPATH."""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from gpu_router.providers.lightning import driver
from gpu_router.providers.lightning.sdk import CallResult, SdkBridge, SubprocessRunner

FAKE_SDK = Path(__file__).with_name("fake_sdk")
CREDS = {"LIGHTNING_USER_ID": "user-0123456789", "LIGHTNING_API_KEY": "key-abcdef0123456789"}


class Fake:
    def __init__(self, tmp: Path, state: dict[str, Any] | None = None) -> None:
        self.tmp = tmp
        self.state_path = tmp / "state.json"
        self.calls_path = tmp / "calls.jsonl"
        self.home = tmp / "sdk-home"
        self.write(
            {
                "user": "me",
                "teamspaces": ["me/default"],
                "jobs": {},
                "studios": [],
                "balance": 12.5,
                **(state or {}),
            }
        )
        self.creds: dict[str, str] = dict(CREDS)

    def write(self, state: dict[str, Any]) -> None:
        self.state_path.write_text(json.dumps(state))

    def state(self) -> dict[str, Any]:
        return json.loads(self.state_path.read_text())

    def calls(self, name: str | None = None) -> list[dict[str, Any]]:
        if not self.calls_path.exists():
            return []
        rows = [json.loads(line) for line in self.calls_path.read_text().splitlines()]
        return [r for r in rows if name is None or r["call"] == name]

    def call(self, op: str, timeout: float = 60, **params: Any) -> CallResult:
        bridge = SdkBridge(
            "lightning",
            runner=SubprocessRunner(),
            interpreter=lambda: [sys.executable],
            env_factory=lambda: self.creds,
            home=lambda: self.home,
            extra_env={
                "PYTHONPATH": str(FAKE_SDK),
                "FAKE_LIGHTNING_STATE": str(self.state_path),
                "FAKE_LIGHTNING_CALLS": str(self.calls_path),
            },
        )
        return bridge.call(op, params, timeout=timeout)


@pytest.fixture
def fake(tmp_path: Path) -> Fake:
    return Fake(tmp_path)


def _job(status: str, **fields: Any) -> dict[str, Any]:
    return {"status": status, "total_cost": 0.25, **fields}


# ---------------------------------------------------------------------- environment


def test_whoami_runs_with_a_private_home_and_no_version_check(fake: Fake) -> None:
    res = fake.call("whoami")
    assert res.ok, res.error
    assert res.result == {
        "user": "me",
        "teamspaces": ["me/default"],
        "teamspace": "me/default",
        "sdk_version": "fake-2026.9.18",
    }
    imp = fake.calls("import")[0]
    assert imp["home"] == str(fake.home)
    assert imp["credential_path"].startswith(str(fake.home))
    assert imp["settings_path"].startswith(str(fake.home))
    assert imp["version_check"] == "1"
    assert imp["browser"] == "true"
    assert imp["has_key"] is True


def test_rejected_credentials_are_auth_not_an_outage(fake: Fake) -> None:
    fake.write({**fake.state(), "raise": {"whoami": {"type": "AuthFailed"}}})
    res = fake.call("whoami")
    assert (res.kind, res.status) == ("auth", 401)


def test_without_credentials_the_sdk_is_never_imported(fake: Fake) -> None:
    fake.creds = {}
    res = fake.call("whoami")
    assert (res.ok, res.kind) == (False, "auth")
    assert fake.calls("import") == []


def test_several_teamspaces_without_a_choice_is_a_config_problem(fake: Fake) -> None:
    fake.write({**fake.state(), "teamspaces": ["me/default", "acme/research"]})
    who = fake.call("whoami")
    assert who.ok
    assert who.result["teamspace"] is None
    res = fake.call("status", name="gr-0123456789ab-1")
    assert res.kind == "config"
    assert "acme/research" in (res.error or "")
    assert fake.call("whoami", teamspace="acme/research").result["teamspace"] == "acme/research"


def test_unknown_op_and_garbage_are_refused(fake: Fake) -> None:
    assert fake.call("format_disk").kind == "invalid"


# ---------------------------------------------------------------------- submit


def _submit_params(tmp: Path, name: str = "gr-0123456789ab-1", **extra: Any) -> dict[str, Any]:
    launch = tmp / "launch.py"
    launch.write_text("print('hi')\n")
    return {
        "teamspace": "me/default",
        "studio": "gpu-router",
        "create_studio": True,
        "name": name,
        "machine": "T4",
        "interruptible": False,
        "command": "echo hi",
        "env": {"GPU_ROUTER_ATTEMPT_KEY": "gpu-0123456789ab-1"},
        "files": [{"local": str(launch), "remote": f"uploads/gpu-router/{name}/launch.py"}],
        "min_balance": 0.05,
        **extra,
    }


def test_submit_creates_the_studio_uploads_and_runs_one_job(fake: Fake, tmp_path: Path) -> None:
    res = fake.call("submit", **_submit_params(tmp_path))
    assert res.ok, res.error
    assert res.result["existed"] is False
    assert res.result["name"] == "gr-0123456789ab-1"
    assert res.result["teamspace"] == "me/default"
    assert "job_name=gr-0123456789ab-1" in res.result["link"]
    studio = fake.calls("Studio")[0]
    assert studio == {"call": "Studio", "name": "gpu-router", "create_ok": True, "skip_setup": True}
    assert fake.calls("Studio.create")
    up = fake.calls("Teamspace.upload_file")[0]
    assert up["remote"] == "uploads/gpu-router/gr-0123456789ab-1/launch.py"
    assert up["progress_bar"] is False
    assert up["cloud_account"] == "lightning-public-prod"
    run = fake.calls("Job.run")[0]
    assert run["machine"] == "T4"
    assert run["studio"] == "gpu-router"
    assert run["teamspace"] == "default"
    assert run["command"] == "echo hi"
    assert run["env"] == {"GPU_ROUTER_ATTEMPT_KEY": "gpu-0123456789ab-1"}
    assert (run["interruptible"], run["max_run_attempts"], run["image"]) == (False, 1, None)
    again = fake.call("submit", **_submit_params(tmp_path))
    assert again.ok
    assert again.result["existed"] is True
    assert len(fake.calls("Job.run")) == 1


def test_a_renamed_duplicate_is_stopped_and_the_original_returned(
    fake: Fake, tmp_path: Path
) -> None:
    fake.write({**fake.state(), "rename_to": "gr-0123456789ab-1-x7"})
    res = fake.call("submit", **_submit_params(tmp_path))
    assert res.ok, res.error
    assert res.result["name"] == "gr-0123456789ab-1"
    assert res.result["existed"] is True
    assert res.result["duplicate_stopped"] == "gr-0123456789ab-1-x7"
    assert [c["name"] for c in fake.calls("Job.stop")] == ["gr-0123456789ab-1-x7"]
    assert "gr-0123456789ab-1-x7" not in fake.state()["jobs"]


@pytest.mark.parametrize(
    ("raise_spec", "kind", "stage", "status"),
    [
        (
            {"type": "ApiException", "status": 429, "reason": "Too Many Requests"},
            "rate",
            "run",
            429,
        ),
        (
            {"type": "ApiException", "status": 400, "body": '{"message":"insufficient balance"}'},
            "quota",
            "run",
            400,
        ),
        ({"type": "ApiException", "status": 400, "body": "bad machine"}, "invalid", "run", 400),
        (
            {"type": "ApiException", "status": 403, "body": "verify your phone"},
            "verify",
            "run",
            403,
        ),
        ({"type": "ApiException", "status": 401}, "auth", "run", 401),
        ({"type": "ApiException", "status": 503}, "unavailable", "run", 503),
        ({"type": "ConnectionError", "msg": "reset by peer"}, "unavailable", "run", None),
        ({"type": "Exception", "msg": "weird"}, "sdk", "run", None),
        ({"type": "AuthFailed"}, "auth", "run", 401),
        ({"type": "RetriesExhausted", "status": 429}, "rate", "run", 429),
        ({"type": "RetriesExhausted", "status": 403}, "auth", "run", 403),
        ({"type": "RetriesExhausted", "status": 502}, "unavailable", "run", 502),
    ],
)
def test_create_call_failures_are_classified_with_their_stage(
    fake: Fake,
    tmp_path: Path,
    raise_spec: dict[str, Any],
    kind: str,
    stage: str,
    status: int | None,
) -> None:
    fake.write({**fake.state(), "raise": {"Job.run": raise_spec}})
    res = fake.call("submit", **_submit_params(tmp_path))
    assert (res.ok, res.kind, res.stage, res.status) == (False, kind, stage, status)


def test_an_error_after_the_create_returns_the_created_job(fake: Fake, tmp_path: Path) -> None:
    """Review fix (invariant 6): Job.run creates the job, then its job.link read answers
    4xx. The job exists, so the submit succeeds with it instead of a definitive 4xx."""
    fake.write(
        {
            **fake.state(),
            "raise": {"Job.run.after": {"type": "ApiException", "status": 404, "once": True}},
        }
    )
    res = fake.call("submit", **_submit_params(tmp_path))
    assert res.ok, res.error
    assert res.result["name"] == "gr-0123456789ab-1"
    assert "(404)" in res.result["run_error"]
    assert "gr-0123456789ab-1" in fake.state()["jobs"]
    assert len(fake.calls("Job.run")) == 1


def test_a_status_read_failing_after_the_create_is_not_a_failed_submit(
    fake: Fake, tmp_path: Path
) -> None:
    """Review fix: the created job's status refetch answers 429. Before, the driver tagged
    it stage "pre" and the adapter called it a definitive RateLimited."""
    fake.write(
        {
            **fake.state(),
            "raise": {"Job.status": {"type": "ApiException", "status": 429, "once": True}},
        }
    )
    res = fake.call("submit", **_submit_params(tmp_path))
    assert res.ok, res.error
    assert res.result["name"] == "gr-0123456789ab-1"
    assert "gr-0123456789ab-1" in fake.state()["jobs"]


def test_an_existing_job_whose_status_read_fails_is_still_returned(
    fake: Fake, tmp_path: Path
) -> None:
    fake.write(
        {
            **fake.state(),
            "jobs": {"gr-0123456789ab-1": _job("Running")},
            "raise": {"Job.status": {"type": "ApiException", "status": 429, "once": True}},
        }
    )
    res = fake.call("submit", **_submit_params(tmp_path))
    assert res.ok, res.error
    assert res.result["existed"] is True
    assert fake.calls("Job.run") == []


def test_a_create_error_is_ambiguous_when_the_lookup_after_it_fails(
    fake: Fake, tmp_path: Path
) -> None:
    fake.write(
        {
            **fake.state(),
            "raise": {
                "Job.run": {"type": "ApiException", "status": 400, "body": "bad machine"},
                # the first Job lookup (the idempotency check) passes, the one after fails
                "Job": {"type": "ApiException", "status": 429, "skip": 1},
            },
        }
    )
    res = fake.call("submit", **_submit_params(tmp_path))
    assert (res.ok, res.kind, res.stage, res.status) == (False, "invalid", "post", 400)


def test_an_unstaged_failure_after_run_was_called_is_post(
    fake: Fake, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The driver's default for a submit failure without a stage is "post" (may exist),
    never "pre"."""
    fake.write({**fake.state(), "rename_to": "gr-0123456789ab-1-x7"})
    fake.write(
        {
            **fake.state(),
            "raise": {"Job": {"type": "ApiException", "status": 503, "skip": 1}},
        }
    )
    res = fake.call("submit", **_submit_params(tmp_path))
    assert (res.ok, res.stage) == (False, "post")


def test_http_errors_never_carry_response_headers(fake: Fake, tmp_path: Path) -> None:
    fake.write(
        {
            **fake.state(),
            "raise": {"Job.run": {"type": "ApiException", "status": 400, "body": "bad"}},
        }
    )
    res = fake.call("submit", **_submit_params(tmp_path))
    assert res.error
    assert "(400)" in res.error
    assert "bad" in res.error
    assert "fake-cookie" not in res.error


def test_failures_before_the_create_call_are_stage_pre(fake: Fake, tmp_path: Path) -> None:
    res = fake.call("submit", **_submit_params(tmp_path, create_studio=False))
    assert (res.kind, res.stage) == ("config", "pre")
    assert "does not exist" in (res.error or "")
    fake.write({**fake.state(), "balance": 0.01})
    res = fake.call("submit", **_submit_params(tmp_path))
    assert (res.kind, res.stage) == ("quota", "pre")
    fake.write(
        {
            **fake.state(),
            "balance": 5,
            "raise": {"Teamspace.upload_file": {"type": "ApiException", "status": 502}},
        }
    )
    res = fake.call("submit", **_submit_params(tmp_path))
    assert (res.kind, res.stage) == ("unavailable", "pre")
    assert fake.calls("Job.run") == []
    res = fake.call("submit", **_submit_params(tmp_path, machine="H100_X_8"))
    assert res.kind == "invalid"


def test_an_unreadable_balance_never_blocks_a_submit(fake: Fake, tmp_path: Path) -> None:
    fake.write({**fake.state(), "raise": {"balance": {"type": "ApiException", "status": 404}}})
    assert fake.call("submit", **_submit_params(tmp_path)).ok


# ---------------------------------------------------------------------- status, logs


def test_status_reads_the_job_once_and_adds_the_log_when_terminal(fake: Fake) -> None:
    fake.write(
        {
            **fake.state(),
            "jobs": {
                "gr-0123456789ab-1": _job(
                    "Failed", started_at=100.0, stopped_at=160.0, message="exit 3", log="a\nb\n"
                ),
                "gr-0123456789ab-2": _job("Running", started_at=100.0, log="x\n"),
            },
        }
    )
    res = fake.call("status", teamspace="me/default", name="gr-0123456789ab-1", with_log=True)
    assert res.ok, res.error
    out = res.result
    assert (out["status"], out["started_at"], out["stopped_at"]) == ("Failed", 100.0, 160.0)
    assert out["message"] == "exit 3"
    assert out["total_cost"] == 0.25
    assert out["log"]["lines"] == ["a", "b"]
    refetches = [c for c in fake.calls("Job.refetch") if c["name"] == "gr-0123456789ab-1"]
    assert len(refetches) == 1  # one status read, the rest pinned
    running = fake.call("status", teamspace="me/default", name="gr-0123456789ab-2", with_log=True)
    assert "log" not in running.result
    missing = fake.call("status", teamspace="me/default", name="gr-0123456789ab-9")
    assert missing.kind == "not_found"


def test_logs_while_pending_and_slow_logs_are_bounded(fake: Fake) -> None:
    fake.write(
        {
            **fake.state(),
            "jobs": {
                "gr-0123456789ab-1": _job(
                    "Pending", log_error="Logs are not available while the job is Pending."
                ),
                "gr-0123456789ab-2": _job("Running", log="late\n", log_sleep=5),
            },
        }
    )
    res = fake.call("logs", teamspace="me/default", name="gr-0123456789ab-1")
    assert res.result["log"]["lines"] == []
    assert "Pending" in res.result["log"]["note"]
    slow = fake.call("logs", teamspace="me/default", name="gr-0123456789ab-2", log_wait_s=0.5)
    assert slow.ok
    assert slow.result["log"]["timeout"] is True


def test_a_slow_final_log_falls_back_to_its_tail_flagged_as_a_timeout(fake: Fake) -> None:
    """Review fix: a whole-log read past log_wait_s used to come back as an empty log,
    which the adapter judged as "no exit line". The tail read keeps the verdict lines."""
    log = "".join(f"line {i}\n" for i in range(50)) + "::gpu:: exit 1\n"
    fake.write(
        {
            **fake.state(),
            "jobs": {"gr-0123456789ab-1": _job("Failed", log=log, full_log_sleep=5)},
        }
    )
    res = fake.call(
        "status",
        teamspace="me/default",
        name="gr-0123456789ab-1",
        with_log=True,
        log_wait_s=0.5,
        log_tail=10,
    )
    assert res.ok, res.error
    out = res.result["log"]
    assert out["timeout"] is True
    assert out["tail"] is True
    assert out["lines"][-1] == "::gpu:: exit 1"
    assert len(out["lines"]) == 10
    tails = [c for c in fake.calls("Job.logs") if c["tail"] == 10]
    assert len(tails) == 1
    # without a tail request the timeout is reported as before (no lines, timeout flag)
    res = fake.call(
        "status", teamspace="me/default", name="gr-0123456789ab-1", with_log=True, log_wait_s=0.5
    )
    assert res.result["log"]["lines"] == []
    assert res.result["log"]["timeout"] is True


# ---------------------------------------------------------------------- stop, fetch, cleanup


def test_stop_is_bounded_idempotent_and_quiet_for_missing_jobs(fake: Fake) -> None:
    fake.write(
        {
            **fake.state(),
            "jobs": {
                "gr-0123456789ab-1": _job("Running", stop_sleep=10),
                "gr-0123456789ab-2": _job("Completed"),
            },
        }
    )
    res = fake.call("stop", teamspace="me/default", name="gr-0123456789ab-1", wait_s=0.5)
    assert res.ok
    assert res.result == {"status": "Stopping", "confirmed": False}
    done = fake.call("stop", teamspace="me/default", name="gr-0123456789ab-2")
    assert done.result == {"status": "Completed", "already": True}
    assert fake.call("stop", teamspace="me/default", name="gr-0123456789ab-9").result == {
        "missing": True
    }


def test_fetch_prefers_the_drive_copy_then_the_artifacts(fake: Fake, tmp_path: Path) -> None:
    drive_key = "uploads/gpu-router/gr-0123456789ab-1/out/outputs.tar.gz"
    art_key = "jobs/gr-0123456789ab-2/gpu-router/gr-0123456789ab-2/outputs.tar.gz"
    fake.write(
        {
            **fake.state(),
            "jobs": {
                "gr-0123456789ab-1": _job("Completed"),
                "gr-0123456789ab-2": _job("Completed"),
                "gr-0123456789ab-3": _job("Completed"),
            },
            "drive": {
                drive_key: base64.b64encode(b"drive-archive").decode(),
                art_key: base64.b64encode(b"artifact-archive").decode(),
            },
        }
    )
    common = {"teamspace": "me/default"}
    one = fake.call(
        "fetch",
        **common,
        name="gr-0123456789ab-1",
        drive_dir="uploads/gpu-router/gr-0123456789ab-1",
        dest=str(tmp_path / "d1"),
    ).result
    assert one["source"] == "drive"
    assert Path(one["archive"]).read_bytes() == b"drive-archive"
    two = fake.call(
        "fetch",
        **common,
        name="gr-0123456789ab-2",
        drive_dir="uploads/gpu-router/gr-0123456789ab-2",
        dest=str(tmp_path / "d2"),
    ).result
    assert two["source"] == "artifacts"
    assert Path(two["archive"]).read_bytes() == b"artifact-archive"
    three = fake.call(
        "fetch",
        **common,
        name="gr-0123456789ab-3",
        drive_dir="uploads/gpu-router/gr-0123456789ab-3",
        dest=str(tmp_path / "d3"),
    ).result
    assert three["source"] is None
    assert three["archive"] is None
    assert any("drive" in n for n in three["notes"])


def test_cleanup_removes_drive_paths_through_lit_uris(fake: Fake) -> None:
    key = "uploads/gpu-router/gr-0123456789ab-1/secrets.json"
    fake.write({**fake.state(), "drive": {key: base64.b64encode(b"{}").decode()}})
    res = fake.call("cleanup", teamspace="me/default", paths=[key, "uploads/gone.json"])
    assert res.result == {"removed": [key, "uploads/gone.json"], "failed": []}
    assert next(c["path"] for c in fake.calls("Filesystem.rm")) == f"lit://me/default/{key}"
    assert key not in fake.state()["drive"]


# ---------------------------------------------------------------------- quota


def test_quota_reads_the_balance_job_costs_and_rates(fake: Fake) -> None:
    fake.write(
        {
            **fake.state(),
            "jobs": {
                "old": _job("Completed", created_at=100.0, total_cost=5.0),
                "new": _job("Completed", created_at=2000.0, total_cost=0.75),
            },
            "machines": {"T4": {"cost": 0.68, "interruptible_cost": 0.2, "wait_time": 30}},
        }
    )
    res = fake.call("quota", teamspace="me/default", since=1000.0, machines=["L4", "T4"])
    out = res.result
    assert out["balance"] == 12.5
    assert out["total_spent"] == 1.0
    assert out["jobs_cost"] == 0.75
    assert out["jobs_counted"] == 1
    assert out["rates"] == {"T4": {"cost": 0.68, "interruptible_cost": 0.2, "wait_time": 30}}
    assert fake.calls("LightningClient")[0]["retry"] is False
    fake.write({**fake.state(), "raise": {"balance": {"type": "ApiException", "status": 404}}})
    no_balance = fake.call("quota", teamspace="me/default", since=1000.0, machines=[]).result
    assert "balance" not in no_balance
    assert "404" in no_balance["balance_error"]


# ---------------------------------------------------------------------- classification


class _Api(Exception):
    def __init__(self, status: int, body: str = "") -> None:
        super().__init__(f"({status})")
        self.status = status
        self.body = body


@pytest.mark.parametrize(
    ("exc", "kind"),
    [
        (_Api(401), "auth"),
        (_Api(403), "auth"),
        (_Api(403, "please verify your phone number"), "verify"),
        (PermissionError("no access to jobs"), "auth"),
        (_Api(404), "not_found"),
        (_Api(429), "rate"),
        (_Api(500), "unavailable"),
        (_Api(402, "insufficient credits"), "quota"),
        (_Api(422, "invalid name"), "invalid"),
        (TimeoutError("slow"), "unavailable"),
        (ValueError("Studio teamspace does not match"), "invalid"),
        (KeyError("x"), "sdk"),
    ],
)
def test_sdk_exceptions_map_to_kinds(exc: BaseException, kind: str) -> None:
    assert driver._classified(exc).kind == kind


def test_the_driver_imports_nothing_from_gpu_router() -> None:
    source = Path(driver.__file__).read_text()
    assert "gpu_router" not in "\n".join(
        line for line in source.splitlines() if line.startswith(("import", "from"))
    )


# ---------------------------------------------------------------------- timestamps


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        # the live API's shapes (2026-09-25): Job.started_at, V1Job.created_at
        ("2026-09-25T07:00:43Z", 1790319643.0),
        ("2026-09-25T07:00:03.934202Z", 1790319603.934202),
        ("2026-09-25T07:00:03.934202123Z", 1790319603.934202123),
        ("2026-09-25T09:00:43+02:00", 1790319643.0),
        ("2026-09-25T02:00:43-0500", 1790319643.0),
        ("2026-09-25 07:00:43", 1790319643.0),
    ],
)
def test_epoch_reads_the_live_iso_strings(value: str, expected: float) -> None:
    assert driver.epoch(value) == pytest.approx(expected, abs=1e-6)


def test_epoch_reads_datetimes_and_refuses_the_rest() -> None:
    from datetime import UTC, datetime

    aware = datetime(2026, 9, 25, 7, 0, 43, tzinfo=UTC)
    assert driver.epoch(aware) == 1790319643.0
    assert driver.epoch(aware.replace(tzinfo=None)) == 1790319643.0  # naive = UTC
    for junk in (None, "", "yesterday", "2026-13-45T99:00:00Z", 1790319643, object()):
        assert driver.epoch(junk) is None


def test_status_times_and_the_month_filter_work_on_iso_strings(fake: Fake) -> None:
    """The fake returns ISO strings like the live API; a datetime-only parser once left
    started_at empty and counted last month's jobs in the month's cost."""
    fake.write(
        {
            **fake.state(),
            "jobs": {
                "gr-0123456789ab-1": _job("Completed", started_at=1790319643.0, total_cost=0.04),
                "gr-0123456789ab-2": _job("Completed", created_at=1788000000.0, total_cost=9.0),
            },
        }
    )
    out = fake.call("status", teamspace="me/default", name="gr-0123456789ab-1").result
    assert out["started_at"] == 1790319643.0
    quota = fake.call("quota", teamspace="me/default", since=1788220800.0, machines=[]).result
    assert quota["jobs_counted"] == 1
    assert quota["jobs_cost"] == pytest.approx(0.04)
