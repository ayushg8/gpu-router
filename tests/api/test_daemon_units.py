"""Daemon building blocks without a server: auth guard, token file, EventBus, launchd plist,
`gpu daemon` argv handling."""

from __future__ import annotations

import asyncio
import plistlib
import stat
from pathlib import Path

import pytest

from gpu_router.daemon import auth, launchd
from gpu_router.daemon.__main__ import EXIT_STATE, EXIT_USAGE
from gpu_router.daemon.__main__ import main as daemon_main
from gpu_router.daemon.events import EventBus
from gpu_router.errors import Forbidden, Unauthorized
from gpu_router.models import JobEvent
from gpu_router.paths import Paths

# --------------------------------------------------------------------------- auth


def test_token_created_once_with_0600(paths: Paths) -> None:
    assert auth.read_token(paths) is None
    token = auth.ensure_token(paths)
    assert len(token) >= 40
    assert stat.S_IMODE(paths.token.stat().st_mode) == 0o600
    assert auth.ensure_token(paths) == token
    assert auth.read_token(paths) == token
    paths.token.chmod(0o644)
    auth.ensure_token(paths)
    assert stat.S_IMODE(paths.token.stat().st_mode) == 0o600


def _check(
    headers: dict[str, str], *, method: str = "GET", path: str = "/v1/status", port: int = 5000
) -> None:
    auth.check_request(method=method, path=path, headers=headers, token="tok", port=port)


def test_guard_rules() -> None:
    ok = {"host": "127.0.0.1:5000", "authorization": "Bearer tok"}
    _check(ok)
    _check({**ok, "host": "localhost:5000"})
    _check({"host": "127.0.0.1:5000"}, path="/v1/health")
    _check({"host": "127.0.0.1:1234", "authorization": "Bearer tok"}, port=0)
    for bad_host in ("evil.com:5000", "127.0.0.1:5001", "", "127.0.0.2:5000"):
        with pytest.raises(Forbidden):
            _check({**ok, "host": bad_host})
    with pytest.raises(Forbidden):
        _check({**ok, "origin": "null"})
    with pytest.raises(Forbidden):  # origin blocked even on the public route
        _check({"host": "127.0.0.1:5000", "origin": "http://x"}, path="/v1/health")
    for bad in ("", "Bearer", "Bearer tok2", "Token tok"):
        with pytest.raises(Unauthorized):
            _check({"host": "127.0.0.1:5000", "authorization": bad})
    with pytest.raises(Unauthorized):  # only GET /v1/health is public
        _check({"host": "127.0.0.1:5000"}, method="POST", path="/v1/health")


# --------------------------------------------------------------------------- events


def _ev(seq: int, job: str = "a" * 12) -> JobEvent:
    return JobEvent(
        seq=seq, job_id=job, kind="note", reason="recovered", message="m", actor="engine", ts=0
    )


async def test_event_bus_wakes_waiters_and_subscribers() -> None:
    bus = EventBus()
    seen: list[tuple[str, int]] = []
    unsubscribe = bus.subscribe(lambda job, evs: seen.append((job, len(evs))))

    waiter = asyncio.create_task(bus.wait_for_events(0, 5))
    job_waiter = asyncio.create_task(bus.wait_for_job_change("a" * 12, 5))
    await asyncio.sleep(0)
    bus.job_changed("a" * 12, [])  # field-only change: job waiters only
    assert await asyncio.wait_for(job_waiter, 1) is True
    await asyncio.sleep(0)
    assert not waiter.done()
    bus.job_changed("a" * 12, [_ev(3)])
    await asyncio.wait_for(waiter, 1)
    assert bus.max_seq == 3
    assert seen == [("a" * 12, 0), ("a" * 12, 1)]
    await bus.wait_for_events(2, 5)  # already satisfied: returns at once
    unsubscribe()
    bus.job_changed("b" * 12, [_ev(4, "b" * 12)])
    assert len(seen) == 2


async def test_event_bus_timeouts() -> None:
    bus = EventBus()
    await asyncio.wait_for(bus.wait_for_events(0, 0.01), 1)
    assert await bus.wait_for_job_change("x", 0.01) is False


async def test_event_bus_isolates_subscriber_bugs() -> None:
    bus = EventBus()

    def broken(_job: str, _evs: object) -> None:
        raise RuntimeError("bug")

    got: list[str] = []
    bus.subscribe(broken)
    bus.subscribe(lambda job, _evs: got.append(job))
    bus.job_changed("j", [])
    assert got == ["j"]


# --------------------------------------------------------------------------- launchd


def test_plist_render_and_install_without_loading(
    paths: Paths, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exe = tmp_path / "bin" / "gpu"
    exe.parent.mkdir()
    exe.write_text("#!/bin/sh\n")
    data = plistlib.loads(launchd.render_plist(executable=exe, paths=paths, env_home=None))
    assert data["Label"] == launchd.LABEL
    assert data["ProgramArguments"] == [str(exe), "daemon", "run", "--launchd"]
    assert data["KeepAlive"] == {"SuccessfulExit": False}
    assert data["RunAtLoad"] is True
    assert data["StandardOutPath"] == str(paths.launchd_log)
    assert "EnvironmentVariables" not in data
    with_home = plistlib.loads(launchd.render_plist(executable=exe, paths=paths, env_home="/x/y"))
    assert with_home["EnvironmentVariables"] == {"GPU_ROUTER_HOME": "/x/y"}

    fake_home = tmp_path / "userhome"
    target = launchd.install(paths, executable=exe, load=False, home=fake_home)
    assert target == launchd.plist_path(fake_home)
    assert stat.S_IMODE(target.stat().st_mode) == 0o644
    assert plistlib.loads(target.read_bytes())["Label"] == launchd.LABEL
    assert launchd.is_installed(fake_home)
    assert launchd.uninstall(unload=False, home=fake_home) is True
    assert launchd.uninstall(unload=False, home=fake_home) is False


# --------------------------------------------------------------------------- gpu daemon argv


def test_daemon_cli_usage_and_not_running(capsys: pytest.CaptureFixture[str]) -> None:
    assert daemon_main([]) == EXIT_USAGE
    assert daemon_main(["bogus"]) == EXIT_USAGE
    assert daemon_main(["status"]) == EXIT_STATE
    assert "not running" in capsys.readouterr().err
    assert daemon_main(["status", "--json"]) == EXIT_STATE
    assert '"running": false' in capsys.readouterr().out
    assert daemon_main(["stop"]) == EXIT_STATE
