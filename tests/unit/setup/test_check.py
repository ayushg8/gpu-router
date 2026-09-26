"""Step 5: doctor, smoke tests through a real in-process daemon on the fake providers, and
the "N providers ready, ~X free hrs/month" line."""

from __future__ import annotations

import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from gpu_router.models import QuotaSnapshot, QuotaUnit
from gpu_router.providers.catalog import load_catalog
from gpu_router.setup import check
from gpu_router.setup.base import Outcome
from tests.shell.conftest import InProcDaemon
from tests.unit.setup.conftest import Sandbox, ScriptedUi


def _snap(provider: str, limit: float | None, unit: QuotaUnit, source: str = "live") -> Any:
    return QuotaSnapshot(
        provider=provider, used=0, limit=limit, unit=unit, source=source, observed_at=0.0
    )


# =========================================================================== free hours


def test_free_hours_from_the_catalog() -> None:
    cat = load_catalog(None)
    entries = [cat.get(n) for n in ("kaggle", "colab", "lightning", "local")]
    hours = {h.provider: h for h in check.free_hours(entries, {})}
    assert hours["kaggle"].hours == pytest.approx(30 * 30.44 / 7)  # ~130h
    assert hours["lightning"].hours == pytest.approx(15 / 0.68)  # ~22h
    assert hours["colab"].hours is None
    assert "dynamic" in hours["colab"].note
    assert hours["local"].hours is None
    assert "unlimited" in hours["local"].note
    line = check.summary_line(["kaggle", "colab", "lightning", "local"], list(hours.values()))
    assert line.startswith("4 providers ready, ~153 free GPU hrs/month")
    assert "colab: dynamic, not counted" in line
    assert "local: unlimited" in line


def test_free_hours_prefer_the_live_ledger() -> None:
    cat = load_catalog(None)
    live = {
        "lightning": _snap("lightning", 4.994, QuotaUnit.CREDITS),
        "kaggle": _snap("kaggle", 30, QuotaUnit.GPU_HOURS),
    }
    hours = {
        h.provider: h for h in check.free_hours([cat.get("lightning"), cat.get("kaggle")], live)
    }
    assert hours["lightning"].hours == pytest.approx(4.994 / 0.68)
    assert hours["lightning"].note == "4.994 credits/month, live"
    line = check.summary_line(["kaggle", "lightning"], list(hours.values()))
    assert line.startswith("2 providers ready, ~138 free GPU hrs/month")


def test_one_provider_is_singular() -> None:
    assert check.summary_line(["local"], []).startswith("1 provider ready, ~0 free")


# =========================================================================== smoke script


def test_smoke_script_runs_and_reports(tmp_path: Path) -> None:
    script = tmp_path / check.SMOKE_NAME
    script.write_text(check.SMOKE_SCRIPT)
    env = {"GPU_SMOKE_LOCAL": "1", "GPU_OUTPUT_DIR": str(tmp_path / "out"), "PATH": "/usr/bin:/bin"}
    res = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, env=env, timeout=120
    )
    assert res.returncode == 0, res.stderr
    line = next(ln for ln in res.stdout.splitlines() if ln.startswith(check.SMOKE_MARKER))
    assert '"gpu":' in line
    assert (tmp_path / "out" / "gpu.json").is_file()


def test_smoke_script_without_a_gpu_exits_3(tmp_path: Path) -> None:
    script = tmp_path / check.SMOKE_NAME
    script.write_text(check.SMOKE_SCRIPT)
    env = {"GPU_OUTPUT_DIR": str(tmp_path / "out"), "PATH": str(tmp_path)}  # no nvidia-smi
    res = subprocess.run(
        [sys.executable, "-S", str(script)], capture_output=True, text=True, env=env, timeout=120
    )
    assert res.returncode == 3


# =========================================================================== with a daemon


@pytest.fixture
def daemon(sandbox: Sandbox, gpu_home: Path) -> Iterator[InProcDaemon]:
    d = InProcDaemon(home=gpu_home)
    d.start()
    sandbox.connect = d.client
    try:
        yield d
    finally:
        d.stop()


@pytest.fixture
def quick_fake(monkeypatch: pytest.MonkeyPatch) -> None:
    real = check.smoke_spec

    def spec(project: Path, view: Any) -> Any:
        s = real(project, view)
        return s.model_copy(update={"provider_options": {view.name: {"duration": 0.3}}})

    monkeypatch.setattr(check, "smoke_spec", spec)


def _healthy(d: InProcDaemon) -> None:
    with d.client() as c:
        for name in ("fake", "fake-b"):
            c.healthcheck(name)


def test_smoke_runs_one_job_per_ready_provider(
    sandbox: Sandbox, daemon: InProcDaemon, quick_fake: None
) -> None:
    _healthy(daemon)
    ui = ScriptedUi([True])
    ctx = sandbox.ctx(ui, only=["check.smoke", "check.summary"])
    check.run(ctx)
    got = {r.id: r for r in ctx.results}
    assert got["check.smoke"].outcome is Outcome.DONE, ui.text
    assert "2 of 2 passed" in got["check.smoke"].summary
    assert got["check.summary"].summary.startswith("2 providers ready, ~")
    assert set(ctx.state.smoke) == {"fake", "fake-b"}
    assert all(r["ok"] for r in ctx.state.smoke.values())
    assert (sandbox.paths.home / "setup" / "smoke" / check.SMOKE_NAME).is_file()
    for d in (sandbox.paths.home / "setup", sandbox.paths.home / "setup" / "smoke"):
        assert d.stat().st_mode & 0o777 == 0o700
    with daemon.client() as c:
        jobs = c.jobs().jobs
    assert sorted(j.spec.provider or "" for j in jobs) == ["fake", "fake-b"]
    assert all(j.spec.labels == {"via": "gpu setup"} for j in jobs)
    assert "run the smoke test on fake, fake-b?" in ui.questions


def test_smoke_passed_before_is_not_rerun(
    sandbox: Sandbox, daemon: InProcDaemon, quick_fake: None
) -> None:
    _healthy(daemon)
    ctx = sandbox.ctx(ScriptedUi([True]), only=["check.smoke"])
    check.run(ctx)
    again = sandbox.ctx(ScriptedUi([]))  # a full run: nothing new to test, no question
    check._smoke(again, daemon.client())
    assert again.results[-1].outcome is Outcome.SKIPPED
    assert "passed before" in again.ui.text  # type: ignore[attr-defined]


def test_smoke_failure_is_reported_and_not_ready(
    sandbox: Sandbox, daemon: InProcDaemon, monkeypatch: pytest.MonkeyPatch
) -> None:
    _healthy(daemon)
    real = check.smoke_spec

    def spec(project: Path, view: Any) -> Any:
        opts = {"duration": 0.3, "exit_code": 3 if view.name == "fake-b" else 0}
        return real(project, view).model_copy(update={"provider_options": {view.name: opts}})

    monkeypatch.setattr(check, "smoke_spec", spec)
    ctx = sandbox.ctx(ScriptedUi([True]), only=["check.smoke", "check.summary"])
    check.run(ctx)
    got = {r.id: r for r in ctx.results}
    assert got["check.smoke"].outcome is Outcome.FAILED
    assert "failed: fake-b" in got["check.smoke"].summary
    assert "saw no GPU" in ctx.state.smoke["fake-b"]["summary"]
    assert got["check.summary"].summary.startswith("1 provider ready")


def test_smoke_declined_submits_nothing(sandbox: Sandbox, daemon: InProcDaemon) -> None:
    _healthy(daemon)
    ctx = sandbox.ctx(ScriptedUi([False]), only=["check.smoke"])
    check.run(ctx)
    assert ctx.results[-1].outcome is Outcome.DECLINED
    with daemon.client() as c:
        assert c.jobs().jobs == []


def test_no_smoke_flag(sandbox: Sandbox, daemon: InProcDaemon) -> None:
    ctx = sandbox.ctx(ScriptedUi([]), only=["check.smoke"], smoke=False)
    check.run(ctx)
    assert ctx.results[-1].outcome is Outcome.SKIPPED


def test_doctor_runs_against_the_daemon(sandbox: Sandbox, daemon: InProcDaemon) -> None:
    ctx = sandbox.ctx(ScriptedUi([]), only=["check.doctor"])
    check.run(ctx)
    res = ctx.results[-1]
    assert res.id == "check.doctor"
    assert res.summary.startswith("doctor: ")
    assert "checks," in res.summary


def test_without_a_daemon_smoke_is_skipped_and_summary_uses_logins(sandbox: Sandbox) -> None:
    sandbox.write(".config/gcloud/application_default_credentials.json", "{}")
    ctx = sandbox.ctx(ScriptedUi([]), only=["check"])
    check.run(ctx)
    got = {r.id: r for r in ctx.results}
    assert got["check.smoke"].outcome is Outcome.SKIPPED
    assert "colab" in got["check.summary"].summary  # colab's ADC login row is ok
    assert "local" in got["check.summary"].summary


def test_smoke_jobs_past_the_deadline_are_cancelled(
    sandbox: Sandbox, daemon: InProcDaemon, monkeypatch: pytest.MonkeyPatch
) -> None:
    _healthy(daemon)
    real = check.smoke_spec

    def slow(project: Path, view: Any) -> Any:
        s = real(project, view)
        return s.model_copy(update={"provider_options": {view.name: {"duration": 600}}})

    monkeypatch.setattr(check, "smoke_spec", slow)
    monkeypatch.setattr(check, "SMOKE_DEADLINE_S", 1.0)
    ctx = sandbox.ctx(ScriptedUi([True]), only=["check.smoke"])
    check.run(ctx)
    assert ctx.results[-1].outcome is Outcome.FAILED
    assert all("cancelled" in r["summary"] for r in ctx.state.smoke.values())
    deadline = time.monotonic() + 15
    states: set[str] = set()
    with daemon.client() as c:
        while time.monotonic() < deadline:
            states = {str(j.state) for j in c.jobs().jobs}
            if states <= {"cancelled"}:
                break
            time.sleep(0.1)
    assert states == {"cancelled"}  # nothing left spending quota


def _asks_approval(monkeypatch: pytest.MonkeyPatch, names: set[str]) -> None:
    """Smoke jobs on `names` wait for approval (an agent job without --hours asks, D48)."""
    from gpu_router.models import Source

    real = check.smoke_spec

    def spec(project: Path, view: Any) -> Any:
        s = real(project, view)
        update: dict[str, Any] = {"provider_options": {view.name: {"duration": 0.3}}}
        if view.name in names:
            update.update(source=Source.AGENT, hours=None)
        return s.model_copy(update=update)

    monkeypatch.setattr(check, "smoke_spec", spec)


def test_a_smoke_job_waiting_for_approval_is_not_a_failure(
    sandbox: Sandbox, daemon: InProcDaemon, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review fix: an approval wait was recorded ok=False, so check.smoke FAILED (exit 1),
    the provider dropped out of 'ready' and every re-run queued another smoke job."""
    _healthy(daemon)
    _asks_approval(monkeypatch, {"fake-b"})
    ctx = sandbox.ctx(ScriptedUi([True]), only=["check.smoke", "check.summary"])
    check.run(ctx)
    got = {r.id: r for r in ctx.results}
    smoke = got["check.smoke"]
    assert smoke.outcome is Outcome.MANUAL, smoke
    assert "1 of 1 passed" in smoke.summary
    assert "fake-b waiting for your approval" in smoke.summary
    assert smoke.fix is not None
    assert smoke.fix.startswith("gpu approve ")
    rec = ctx.state.smoke["fake-b"]
    assert (rec["ok"], rec["pending"]) == (None, True)
    assert got["check.summary"].summary.startswith("2 providers ready")  # not smoke_failed
    # a re-run does not queue a second smoke job while the first still waits
    again = sandbox.ctx(ScriptedUi([]))  # a full run: fake passed before, fake-b waits
    check._smoke(again, daemon.client())
    assert again.results[-1].outcome is Outcome.SKIPPED
    assert "still waits for your approval" in again.ui.text  # type: ignore[attr-defined]
    with daemon.client() as c:
        on_b = [j for j in c.jobs().jobs if j.spec.provider == "fake-b"]
    assert len(on_b) == 1


def test_a_smoke_job_in_flight_is_reattached_not_submitted_again(
    sandbox: Sandbox, daemon: InProcDaemon, quick_fake: None
) -> None:
    """Review fix: Ctrl-C during the poll left the jobs running and recorded nothing, so a
    resumed run submitted them again. Job ids are recorded right after the submit."""
    _healthy(daemon)
    ctx = sandbox.ctx(ScriptedUi([True]), only=["check.smoke"])
    project = check.write_smoke_project(ctx)
    views = {v.name: v for v in check._ready_views(daemon.client())}
    with daemon.client() as c:
        job = c.submit(check.smoke_spec(project, views["fake"]))
    check._record(ctx, "fake", ok=None, job=job.id, summary="submitted")  # as if cut short
    check.run(ctx)
    assert ctx.state.smoke["fake"]["ok"] is True
    assert ctx.state.smoke["fake"]["job"] == job.id
    assert "back to smoke job" in ctx.ui.text  # type: ignore[attr-defined]
    with daemon.client() as c:
        on_fake = [j for j in c.jobs().jobs if j.spec.provider == "fake"]
    assert len(on_fake) == 1
