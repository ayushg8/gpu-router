"""SVG screenshots of the shell's main states, rendered by Textual against a real in-process
daemon. They double as render smoke tests (each state must come up and draw).

Set GPU_SHELL_SHOTS=<dir> to keep them (docs/screenshots/shell-*.png are made from these
with resvg: see CLAUDE.md D42); otherwise they go to the test's tmp dir.

The daemon here runs the fake adapter under the mockup's provider names and the catalog's
shapes (kaggle 2xT4 on a weekly 30 h quota, colab T4, lightning T4 on a new account's 5
monthly credits; modal was dropped in phase 7b), so the pictures read like the spec's
mockup; every job is a
simulated fake run and no real provider is touched (invariant 20). Phase-5 data is real:
jobs are placed by the scoring router (reasons in the approval row and /route), the
approval comes from the default rules (an agent job over the 1h auto limit), the footer
and /quota are the quota ledger, and checkpoints go through the local storage backend
(`checkpoint.fake_storage`, D43).
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from gpu_router.models import Source
from gpu_router.shell.app import GpuShell
from gpu_router.shell.widgets import CommandBlock, LogsBlock, WatchBlock
from tests.crash.harness import FAST_CONFIG
from tests.shell.conftest import InProcDaemon, settle

SIZE = (100, 28)

POSED_PROVIDERS = """\
providers:
  kaggle:
    kind: fake
    test_only: true
    poll_interval_s: 0.05
    gpus: [{name: T4, vram_gb: 16, count: 2}]
    quota: {unit: gpu_hours, limit: 30, reset: weekly, reset_anchor: "sat 00:00 UTC"}
  colab:
    kind: fake
    test_only: true
    poll_interval_s: 0.05
    gpus: [{name: T4, vram_gb: 16}]
    quota: {unit: gpu_hours, limit: null, reset: unknown}
  lightning:
    kind: fake
    test_only: true
    enabled_by_default: true
    poll_interval_s: 0.05
    gpus: [{name: T4, vram_gb: 16}]
    quota: {unit: credits, limit: 5, reset: monthly, reset_anchor: "day 1 00:00 UTC"}
    options: {quota_per_gpu_hour: 0.68}
"""
POSED_CONFIG = (
    FAST_CONFIG
    + """\
providers:
  fake: {enabled: false}
  fake-b: {enabled: false}
checkpoint:
  fake_storage: true
"""
)

TRAIN = {"duration": 900, "steps": 9000, "checkpoint_every": 2}


@pytest.fixture
def shots(tmp_path: Path) -> Path:
    out = Path(os.environ.get("GPU_SHELL_SHOTS") or tmp_path / "shots")
    out.mkdir(parents=True, exist_ok=True)
    return out


@pytest.fixture
def posed(gpu_home: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[InProcDaemon]:
    monkeypatch.setenv("GPU_ROUTER_NO_AUTOSTART", "1")
    d = InProcDaemon(home=gpu_home, providers_yaml=POSED_PROVIDERS, config_yaml=POSED_CONFIG)
    d.start()
    try:
        yield d
    finally:
        d.stop()


@pytest.fixture
def ml_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    proj = tmp_path / "yolo-experiments"
    proj.mkdir()
    for name in ("train_yolo.py", "eval.py", "prep_data.py"):
        (proj / name).write_text("print('ok')\n")
    monkeypatch.chdir(proj)
    return proj


def shell() -> GpuShell:
    return GpuShell(poll_s=0.2, quota_s=1.0, retry_s=3.0)


def save(app: GpuShell, shots: Path, name: str) -> Path:
    path = Path(app.save_screenshot(f"shell-{name}.svg", str(shots)))
    assert path.stat().st_size > 1000
    return path


async def up(pilot: Any, app: GpuShell, pred: Any) -> None:
    await settle(
        pilot, lambda: app.snap is not None and app.snap.conn.status == "up", what="connect"
    )
    await settle(pilot, pred, what="state")


async def test_shot_empty(posed: InProcDaemon, ml_project: Path, shots: Path) -> None:
    app = shell()
    async with app.run_test(size=SIZE) as pilot:
        await up(
            pilot, app, lambda: "lightning" in app.panel.plain and "kaggle" in app.footer.plain
        )
        await pilot.pause(0.3)
        assert "no jobs running" in app.panel.plain
        assert "/run train_yolo.py" in app.panel.plain
        save(app, shots, "empty")


def submit_train(posed: InProcDaemon, project: Path, name: str = "train_yolo.py") -> Any:
    """A 6-hour training job: the scoring router sends it to kaggle (saved for long jobs)."""
    return posed.submit(project, TRAIN, name=name, hours=6)


def submit_eval(posed: InProcDaemon, project: Path) -> Any:
    """An agent's 2-hour eval: routed to colab, and over the agent 1h auto limit, so it
    waits for approval (the spec's default rules)."""
    return posed.submit(project, {"duration": 30}, name="eval.py", hours=2, source=Source.AGENT)


async def test_shot_running(posed: InProcDaemon, ml_project: Path, shots: Path) -> None:
    # pinned: on the day before kaggle's weekly reset the scoring router sends a short job
    # to kaggle ("use it or lose it"), which would leave kaggle busy for the training job
    posed.submit(
        ml_project, {"duration": 0.5, "steps": 5}, name="prep_data.py", hours=0.2, provider="colab"
    )
    job = submit_train(posed, ml_project)
    assert posed.job(job.id).provider == "kaggle"
    posed.submit(
        ml_project, {"duration": 900, "steps": 4000}, name="finetune_lora.py", provider="lightning"
    )
    app = shell()
    async with app.run_test(size=SIZE) as pilot:
        await up(pilot, app, lambda: app.snap is not None and app.snap.running == 2)
        await settle(
            pilot,
            lambda: "ckpt" in app.panel.plain and "done in" in app.panel.plain,
            what="metrics, checkpoint and the finished row",
        )
        await pilot.pause(2.5)  # let the sparklines fill
        save(app, shots, "running")


async def test_shot_approval(posed: InProcDaemon, ml_project: Path, shots: Path) -> None:
    submit_train(posed, ml_project)
    waiting = submit_eval(posed, ml_project)
    app = shell()
    async with app.run_test(size=SIZE) as pilot:
        await up(
            pilot,
            app,
            lambda: (
                "needs approval" in app.panel.plain
                and app.snap is not None
                and app.snap.running == 1
            ),
        )
        await pilot.pause(2.0)
        assert "route → colab T4 (fits 16GB" in app.panel.plain  # the scoring router's reason
        assert "over the 1h auto limit" in app.panel.plain  # the agent rule that asked
        await pilot.press("slash", "a", "p", "tab", "tab")
        assert app.prompt.value == f"/approve {waiting.short_id} "
        save(app, shots, "approval")


async def test_shot_popup(posed: InProcDaemon, ml_project: Path, shots: Path) -> None:
    submit_train(posed, ml_project)
    app = shell()
    async with app.run_test(size=SIZE) as pilot:
        await up(pilot, app, lambda: app.snap is not None and app.snap.running == 1)
        await pilot.pause(1.0)
        await pilot.press("slash")
        await settle(pilot, lambda: app.popup.display, what="popup")
        save(app, shots, "popup")


async def test_shot_logs(posed: InProcDaemon, ml_project: Path, shots: Path) -> None:
    job = submit_train(posed, ml_project)
    app = shell()
    async with app.run_test(size=SIZE) as pilot:
        await up(pilot, app, lambda: app.snap is not None and app.snap.running == 1)
        app.prompt.value = f"/logs {job.short_id}"
        await pilot.press("enter")
        await settle(pilot, lambda: bool(app.query(LogsBlock)), what="logs")
        await settle(pilot, lambda: app.query_one(LogsBlock).count > 30, what="lines")
        await pilot.pause(0.5)
        save(app, shots, "logs")


async def test_shot_watch(posed: InProcDaemon, ml_project: Path, shots: Path) -> None:
    job = submit_train(posed, ml_project)
    app = shell()
    async with app.run_test(size=SIZE) as pilot:
        await up(pilot, app, lambda: app.snap is not None and app.snap.running == 1)
        await pilot.pause(3.0)
        app.prompt.value = f"/watch {job.short_id}"
        await pilot.press("enter")
        await settle(pilot, lambda: bool(app.query(WatchBlock)), what="watch")
        watch = app.query_one(WatchBlock)
        await settle(pilot, lambda: len(watch.state.series.get("loss", [])) > 40, what="points")
        await pilot.pause(1.2)
        save(app, shots, "watch")


async def test_shot_daemon_down(no_daemon: Path, ml_project: Path, shots: Path) -> None:
    app = shell()
    async with app.run_test(size=SIZE) as pilot:
        await settle(
            pilot, lambda: app.snap is not None and app.snap.conn.status == "down", what="down"
        )
        app.prompt.value = "/jobs"
        await pilot.press("enter")
        await settle(
            pilot,
            lambda: "not running" in "".join(b.plain for b in app.query("CommandBlock")),
            what="error block",
        )  # type: ignore[attr-defined]
        await pilot.pause(0.3)
        save(app, shots, "daemon-down")


async def test_shot_narrow(posed: InProcDaemon, ml_project: Path, shots: Path) -> None:
    """80x24, five jobs: the panel folds to one row per job and /jobs keeps one row each."""
    submit_train(posed, ml_project, name="train_yolo_with_a_long_name.py")
    submit_eval(posed, ml_project)
    for i in range(3):
        posed.submit(
            ml_project,
            {"duration": 900, "steps": 900},
            name=f"sweep_lr_{i}.py",
            provider="lightning",
        )
    app = shell()
    async with app.run_test(size=(80, 24)) as pilot:
        await up(
            pilot,
            app,
            lambda: app.snap is not None and app.snap.running == 2 and len(app.snap.active) == 5,
        )
        await pilot.pause(1.0)
        panel = app.panel.plain.splitlines()
        assert len(panel) <= app.panel.max_rows
        assert any("needs approval" in ln for ln in panel)
        app.prompt.value = "/jobs"
        await pilot.press("enter")

        def jobs_block() -> str:
            done = [b for b in app.query(CommandBlock) if b.line == "/jobs" and not b.running]
            return done[0].plain if done else ""

        await settle(pilot, lambda: "needs approval" in jobs_block(), what="/jobs")
        rows = [ln for ln in jobs_block().splitlines()[1:] if ln.strip()]
        assert len(rows) == 6  # header + one row per job
        await pilot.pause(0.3)
        save(app, shots, "narrow")


async def test_shot_route(posed: InProcDaemon, ml_project: Path, shots: Path) -> None:
    """Phase 5: /route explains the scoring router's pick; /quota is the ledger."""
    submit_train(posed, ml_project)
    app = shell()
    async with app.run_test(size=(100, 34)) as pilot:
        await up(pilot, app, lambda: app.snap is not None and app.snap.running == 1)
        route = "/route --hours 0.3 eval.py"
        app.prompt.value = route
        await pilot.press("enter")

        def block(line: str) -> str:
            done = [b for b in app.query(CommandBlock) if b.line == line and not b.running]
            return done[0].plain if done else ""

        await settle(
            pilot, lambda: "candidates" in block("/route --hours 0.3 eval.py"), what="route"
        )
        assert "→ colab" in block("/route --hours 0.3 eval.py")
        app.prompt.value = "/quota"
        await pilot.press("enter")
        await settle(pilot, lambda: "kaggle" in block("/quota"), what="quota")
        await pilot.pause(0.3)
        save(app, shots, "route")
