"""Doctor runner (parallel, one deadline, crash-proof, drift aggregation) and
`--update-catalog` (plan, validate, atomic 0600 write). Phase 8a."""

from __future__ import annotations

import json
import os
import stat
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from gpu_router.doctor.catalog_fix import apply_plan, plan_update
from gpu_router.doctor.checks import Task, default_tasks
from gpu_router.doctor.model import CheckResult, DriftItem, Status
from gpu_router.doctor.probe import ProbeEnv
from gpu_router.doctor.render import render_report
from gpu_router.doctor.runner import run_doctor
from gpu_router.errors import ConfigError
from gpu_router.paths import Paths
from gpu_router.providers.catalog import load_catalog

MakeEnv = Callable[..., ProbeEnv]


def res(task_id: str, status: Status = Status.OK, **detail: object) -> CheckResult:
    return CheckResult(
        id=task_id, group="local", title=task_id, status=status, summary="s", detail=dict(detail)
    )


def sleeper(task_id: str, seconds: float) -> Task:
    def fn(_env: ProbeEnv) -> CheckResult:
        time.sleep(seconds)
        return res(task_id)

    return Task(task_id, "local", task_id, fn)


def test_checks_run_in_parallel(make_env: MakeEnv) -> None:
    tasks = [sleeper(f"t{i}", 0.3) for i in range(8)]
    t0 = time.monotonic()
    report = run_doctor(make_env(), tasks=tasks)
    assert time.monotonic() - t0 < 1.5
    assert [c.id for c in report.checks] == [f"t{i}" for i in range(8)]  # task order kept
    assert report.ok
    assert report.counts["ok"] == 8


def test_the_deadline_turns_a_slow_check_into_a_warning(make_env: MakeEnv) -> None:
    tasks = [sleeper("fast", 0.0), sleeper("stuck", 30)]
    t0 = time.monotonic()
    report = run_doctor(make_env(deadline_s=0.5), tasks=tasks)
    assert time.monotonic() - t0 < 2
    stuck = next(c for c in report.checks if c.id == "stuck")
    assert stuck.status is Status.WARN
    assert "did not finish in 0.5s" in stuck.summary
    assert stuck.fix is not None
    assert stuck.fix.startswith("gpu doctor --timeout")


def test_a_crashing_check_is_a_warning_not_a_crash(make_env: MakeEnv) -> None:
    def boom(_env: ProbeEnv) -> CheckResult:
        raise KeyError("oops")

    report = run_doctor(make_env(), tasks=[Task("x", "local", "x", boom), sleeper("y", 0)])
    x = report.checks[0]
    assert x.status is Status.WARN
    assert "crashed (KeyError" in x.summary
    assert report.checks[1].status is Status.OK


def test_crashed_and_unfinished_rows_are_never_everything_works(make_env: MakeEnv) -> None:
    """Review fix: both were plain warnings, so the report said 'everything works' and
    `gpu doctor` exited 0 although those checks verified nothing."""
    from gpu_router.doctor.render import summary_line

    def boom(_env: ProbeEnv) -> CheckResult:
        raise RuntimeError("bug")

    tasks = [Task("x", "local", "x", boom), sleeper("stuck", 30), sleeper("fine", 0)]
    report = run_doctor(make_env(deadline_s=0.5), tasks=tasks)
    assert report.ok  # nothing failed ...
    assert report.unknown == 2  # ... but two rows know nothing
    assert {c.detail.get("unknown") for c in report.checks} == {"crashed", "timeout", None}
    line = summary_line(report).plain
    assert "everything works" not in line
    assert "2 check(s) crashed or did not finish" in line
    assert json.loads(report.model_dump_json())["unknown"] == 2


def test_gpu_doctor_exits_1_when_a_check_verified_nothing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from typer.testing import CliRunner

    from gpu_router.cli.app import app
    from gpu_router.doctor import runner

    def boom(_env: ProbeEnv) -> CheckResult:
        raise RuntimeError("bug")

    real = runner.run_doctor
    monkeypatch.setattr(
        runner,
        "run_doctor",
        lambda env, only=None: real(env, tasks=[Task("x", "local", "x", boom)]),
    )
    res_ = CliRunner().invoke(app, ["doctor", "--json"], catch_exceptions=False)
    assert res_.exit_code == 1
    assert json.loads(res_.stdout)["unknown"] == 1


def test_fail_counts_and_drift_are_collected(make_env: MakeEnv) -> None:
    drift = DriftItem(provider="kaggle", key="quota.limit", catalog=30, live=29, note="n")
    tasks = [
        Task("a", "limits", "a", lambda _e: res("a", Status.WARN, drift=[drift.model_dump()])),
        Task("b", "local", "b", lambda _e: res("b", Status.FAIL)),
    ]
    report = run_doctor(make_env(), tasks=tasks)
    assert not report.ok
    assert report.counts == {"ok": 0, "warn": 1, "fail": 1, "skip": 0}
    assert report.drift == [drift]
    assert "drift" not in report.checks[0].detail
    json.dumps(report.model_dump(mode="json"))  # the --json document


def test_only_filters_by_group_or_id_prefix(make_env: MakeEnv) -> None:
    env = make_env()
    ids = [c.id for c in run_doctor(env, only={"daemon", "provider.kaggle"}).checks]
    assert ids
    assert all(i.startswith(("daemon.", "provider.kaggle.")) for i in ids)


def test_only_that_matches_nothing_is_a_usage_error(make_env: MakeEnv) -> None:
    from gpu_router.errors import InvalidRequest

    with pytest.raises(InvalidRequest, match="no doctor check matches --only nope"):
        run_doctor(make_env(), only={"nope"})


def test_the_full_default_run_is_fast_and_complete(make_env: MakeEnv) -> None:
    env = make_env()
    t0 = time.monotonic()
    report = run_doctor(env)
    assert time.monotonic() - t0 < 5
    groups = {c.group for c in report.checks}
    assert groups == {
        "daemon",
        "providers",
        "storage",
        "inference",
        "limits",
        "local",
        "integration",
    }
    assert len(report.checks) == len(default_tasks(env))
    for c in report.checks:
        if c.status in (Status.WARN, Status.FAIL) and not c.id.startswith("provider.lightning"):
            assert c.fix, c  # every problem says how to fix it
    text = "\n".join(str(getattr(r, "plain", "")) for r in render_report(report))
    assert "gpu doctor" in text


# --------------------------------------------------------------------------- catalog fix


DRIFT = [
    DriftItem(provider="kaggle", key="quota.limit", catalog=30, live=29, note="n"),
    DriftItem(
        provider="kaggle",
        key="quota.reset_anchor",
        catalog="sat 00:00 UTC",
        live="fri 00:00 UTC",
        note="n",
    ),
    DriftItem(provider="colab", key="gpus", catalog=["T4"], live=["L4"], note="not automatic"),
]


def test_plan_writes_only_measurable_keys_and_validates(paths: Paths) -> None:
    plan = plan_update(paths.user_providers, DRIFT, now=0)
    assert plan.changed
    assert [d.key for d in plan.items] == ["quota.limit", "quota.reset_anchor"]
    assert "+    quota:" in plan.diff() or "limit: 29" in plan.diff()
    apply_plan(plan)
    assert stat.S_IMODE(os.stat(paths.user_providers).st_mode) == 0o600
    kaggle = load_catalog(paths.user_providers).providers["kaggle"]
    assert kaggle.quota.limit == 29
    assert kaggle.quota.reset_anchor == "fri 00:00 UTC"
    assert kaggle.gpus  # everything else still comes from the packaged catalog


def test_plan_keeps_other_user_overrides(paths: Paths) -> None:
    paths.user_providers.write_text("providers:\n  colab:\n    priority: 5\n")
    apply_plan(plan_update(paths.user_providers, DRIFT[:1], now=0))
    cat = load_catalog(paths.user_providers)
    assert cat.providers["colab"].priority == 5
    assert cat.providers["kaggle"].quota.limit == 29


def test_nothing_measurable_means_no_change(paths: Paths) -> None:
    assert not plan_update(paths.user_providers, DRIFT[2:], now=0).changed
    assert not paths.user_providers.exists()


def test_a_broken_user_file_is_refused(paths: Paths) -> None:
    paths.user_providers.write_text("providers: [\n")
    with pytest.raises(ConfigError):
        plan_update(paths.user_providers, DRIFT, now=0)
    bad = [
        DriftItem(
            provider="kaggle", key="quota.unit", catalog="gpu_hours", live="parsecs", note="n"
        )
    ]
    paths.user_providers.write_text("")
    with pytest.raises(ConfigError):
        plan_update(paths.user_providers, bad, now=0)
    assert not list(Path(paths.user_providers).parent.glob(".providers.yaml.check*"))
