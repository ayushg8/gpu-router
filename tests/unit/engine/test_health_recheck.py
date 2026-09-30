"""Provider health loop: backoff re-checks and recovery after the Mac sleeps.

Found live: after a night asleep on a flaky network, the first healthchecks after the wake
timed out (`kaggle --version` > 15 s, `lightning whoami` > 42 s) and those providers stayed
"unavailable" for the flat 15 min `health_recheck_s`. Now an unhealthy provider is re-checked
after 60 s, doubling up to 15 min, and a wake from sleep re-checks every provider once the
network had `wake_grace_s` to come back.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from itertools import pairwise
from typing import Any

import pytest

from gpu_router.adapters.base import Health
from gpu_router.adapters.registry import AdapterRegistry
from gpu_router.clock import FAKE_EPOCH, FakeClock, settle
from gpu_router.config import CallTimeouts, Config
from gpu_router.engine.calls import AdapterCaller
from gpu_router.engine.deps import EngineDeps
from gpu_router.engine.supervisor import Supervisor
from gpu_router.models import JobState, ProviderHealth
from gpu_router.paths import Paths
from gpu_router.policy import default_policy
from gpu_router.providers.catalog import Catalog
from gpu_router.router.simple import SimpleRouter
from gpu_router.store import Store
from tests.unit.engine.conftest import Engine, engine_config

BACKOFF = [60, 120, 240, 480, 900, 900]  # health_recheck_min_s doubling to health_recheck_s


def record_checks(eng: Engine, name: str) -> list[float]:
    """Wrap the fake's healthcheck; return the list it appends each check's time to."""
    adapter = eng.fake(name)
    original: Callable[[], Health] = adapter.healthcheck
    seen: list[float] = []

    def healthcheck() -> Health:
        seen.append(eng.clock.now())
        return original()

    adapter.healthcheck = healthcheck  # type: ignore[method-assign]
    return seen


async def advance(eng: Engine, seconds: float, step: float = 1.0) -> None:
    """Move fake time forward in small steps (never enough at once to look like a sleep)."""
    end = eng.clock.now() + seconds
    while eng.clock.now() < end:
        await settle(20)
        eng.clock.advance(min(step, end - eng.clock.now()))
    await settle(20)


def health(eng: Engine, name: str) -> ProviderHealth:
    return eng.store.get_provider_state(name).health


async def test_unhealthy_provider_is_rechecked_on_a_doubling_backoff(eng: Engine) -> None:
    seen = record_checks(eng, "fake-b")
    eng.fake("fake-b").set_health("unavailable", "network is unreachable")
    t0 = eng.clock.now()
    await eng.supervisor.healthcheck("fake-b")
    assert health(eng, "fake-b") is ProviderHealth.UNAVAILABLE
    await advance(eng, sum(BACKOFF) + 30)
    checks = [t - t0 for t in seen]
    assert checks[0] == 0  # the manual check
    assert [b - a for a, b in pairwise(checks)] == BACKOFF
    # a healthy provider is never polled by the loop
    assert health(eng, "fake") is ProviderHealth.OK


async def test_a_healthy_answer_resets_the_backoff(eng: Engine) -> None:
    seen = record_checks(eng, "fake-b")
    eng.fake("fake-b").set_health("unavailable", "down")
    t0 = eng.clock.now()
    await eng.supervisor.healthcheck("fake-b")
    await advance(eng, 60 + 120 + 240 + 5)  # re-checked at +60, +180, +420: 3 failures
    assert [t - t0 for t in seen] == [0, 60, 180, 420]

    eng.fake("fake-b").set_health(None)  # back; the next re-check (+900) sees it
    await advance(eng, 480)
    assert health(eng, "fake-b") is ProviderHealth.OK
    assert seen[-1] - t0 == 900
    view = eng.supervisor.provider_view("fake-b")
    assert view.next_healthcheck_at is None
    assert view.health_reason is None

    await advance(eng, 3600)  # healthy providers are not polled
    assert seen[-1] - t0 == 900

    eng.fake("fake-b").set_health("unavailable", "down again")
    t1 = eng.clock.now()
    await eng.supervisor.healthcheck("fake-b")
    await advance(eng, 65)
    assert seen[-1] - t1 == 60  # starts short again, not at 900


async def test_unhealthy_view_says_when_it_is_rechecked(eng: Engine) -> None:
    reason = "kaggle --version did not answer within 15s"
    eng.fake("fake-b").set_health("unavailable", reason)
    t0 = eng.clock.now()
    view = await eng.supervisor.healthcheck("fake-b")
    assert view.health is ProviderHealth.UNAVAILABLE
    assert view.health_reason == f"{reason}; re-checking in 1m"
    assert view.next_healthcheck_at == pytest.approx(t0 + 60)
    # the stored reason stays the provider's own words (router messages quote it)
    assert eng.store.get_provider_state("fake-b").health_reason == reason
    await advance(eng, 20)
    assert eng.supervisor.provider_view("fake-b").health_reason == f"{reason}; re-checking in 40s"
    await advance(eng, 45)  # re-checked at +60 and failed again: the next one is at +180
    view = eng.supervisor.provider_view("fake-b")
    assert view.next_healthcheck_at == pytest.approx(t0 + 180)
    assert view.health_reason == f"{reason}; re-checking in 1m"  # 115 s, in whole minutes
    doc = view.model_dump(mode="json")
    assert doc["next_healthcheck_at"].endswith("Z")  # ISO-8601 at the JSON edge


async def test_a_login_problem_found_by_a_driver_is_rechecked_soon(eng: Engine) -> None:
    """A driver that hits AuthRequired marks the provider without a healthcheck; the loop
    picks it up one short backoff later instead of 15 min."""
    seen = record_checks(eng, "fake-b")
    t0 = eng.clock.now()
    eng.store.upsert_provider_state(
        "fake-b",
        health=ProviderHealth.AUTH_REQUIRED,
        health_reason="fake-b needs login",
        last_healthcheck_at=t0,
    )
    assert eng.supervisor.provider_view("fake-b").health_reason == (
        "fake-b needs login; re-checking in 1m"
    )
    await advance(eng, 65)
    assert [t - t0 for t in seen] == [60]
    assert health(eng, "fake-b") is ProviderHealth.OK


async def test_waking_from_sleep_rechecks_every_provider_after_a_grace(eng: Engine) -> None:
    seen_a = record_checks(eng, "fake")
    seen_b = record_checks(eng, "fake-b")
    await advance(eng, 10)
    eng.clock.suspend(5 * 3600)  # the Mac sleeps; its loop clock stands still
    woke = eng.clock.now()
    await advance(eng, 45)
    assert seen_a == []  # the grace: no check on a network still waking
    assert seen_b == []
    await advance(eng, 30)
    # the loop noticed at its next tick (<= 30 s), then gave the network 30 s
    assert len(seen_a) == 1
    assert len(seen_b) == 1
    assert 30 <= seen_a[0] - woke <= 60
    assert seen_b[0] == seen_a[0]  # concurrently, not one after the other


async def test_a_long_stall_of_the_loop_clock_counts_as_a_wake(eng: Engine) -> None:
    """The other signal: wall time far past the planned sleep (> 2 x tick + 60 s)."""
    seen = record_checks(eng, "fake")
    await advance(eng, 5)
    eng.clock.advance(4 * 3600)  # one jump: the loop's sleep ends hours late
    await settle(20)
    assert seen == []
    await advance(eng, 31)
    assert len(seen) == 1


async def test_after_a_wake_a_failed_check_is_retried_in_a_minute_not_15(
    eng: Engine,
) -> None:
    """The live incident: the first check after the wake fails on a network that is not
    back yet. The backoff restarts from 60 s, whatever it was before the sleep."""
    seen = record_checks(eng, "fake-b")
    eng.fake("fake-b").set_health("unavailable", "down before the sleep")
    await eng.supervisor.healthcheck("fake-b")
    await advance(eng, sum(BACKOFF[:4]) + 5)  # 5 failures: the next re-check is 15 min out
    assert len(seen) == 5

    eng.fake("fake-b").set_health("unavailable", "lightning whoami did not answer within 42s")
    eng.clock.suspend(8 * 3600)
    await advance(eng, 65)  # wake noticed + grace: checked again, still failing
    assert len(seen) == 6
    failed_at = seen[-1]
    assert eng.supervisor.provider_view("fake-b").next_healthcheck_at == failed_at + 60

    eng.fake("fake-b").set_health(None)  # the network is back 20 s later
    await advance(eng, 65)
    assert seen[-1] - failed_at == 60
    assert health(eng, "fake-b") is ProviderHealth.OK


async def test_a_wake_lets_queued_jobs_route_again(eng: Engine) -> None:
    """A job waiting for a quota reset sleeps on the loop clock, which stood still while the
    Mac slept past the reset: the wake sends it routing again instead of waiting hours."""
    reset = eng.clock.now() + 2 * 3600
    for name in ("fake", "fake-b"):
        eng.store.upsert_provider_state(name, exhausted_until=reset)
    job = await eng.submit()
    await eng.run_until(lambda: eng.job(job.id).not_before is not None)
    assert eng.job(job.id).state is JobState.QUEUED
    assert eng.job(job.id).not_before == reset
    eng.clock.suspend(3 * 3600)  # asleep past the reset
    done = await eng.until_terminal(job.id, max_s=300)
    assert done.state is JobState.DONE


async def test_the_loop_survives_a_bookkeeping_error(
    eng: Engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = record_checks(eng, "fake-b")
    eng.fake("fake-b").set_health("unavailable", "down")
    await eng.supervisor.healthcheck("fake-b")
    real = eng.supervisor._health_sleep_s
    calls = {"n": 0}

    def flaky() -> float:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")
        return real()

    monkeypatch.setattr(eng.supervisor, "_health_sleep_s", flaky)
    await advance(eng, 200)
    assert calls["n"] > 1
    assert len(seen) >= 2  # still re-checking


# ---------------------------------------------------------------- a healthcheck that hangs


def _supervisor_with_threads(
    store: Store, clock: FakeClock, paths: Paths, config: Config, catalog: Catalog
) -> Supervisor:
    """Like conftest.build_supervisor, but adapter calls run in real worker threads so the
    caller's timeout can fire."""
    registry = AdapterRegistry.build(config=config, catalog=catalog, paths=paths, clock=clock)
    deps = EngineDeps(
        store=store,
        registry=registry,
        router=SimpleRouter(),
        policy=default_policy(),
        caller=AdapterCaller(registry, config.engine),
        clock=clock,
        config=config,
        paths=paths,
    )
    return Supervisor(deps)


async def test_a_healthcheck_timeout_marks_the_provider_unavailable_and_says_when_next(
    store: Store, clock: FakeClock, paths: Paths, catalog: Catalog
) -> None:
    config = engine_config(timeouts=CallTimeouts(healthcheck=0.2))
    sup = _supervisor_with_threads(store, clock, paths, config, catalog)
    await sup.start()
    release = threading.Event()
    adapter: Any = sup.deps.registry.get("fake-b")

    def hangs() -> Health:
        release.wait(10)
        raise AssertionError("never answers in time")

    adapter.healthcheck = hangs
    try:
        view = await sup.healthcheck("fake-b")
        assert view.health is ProviderHealth.UNAVAILABLE
        assert view.health_reason == ("fake-b healthcheck timed out after 0.2s; re-checking in 1m")
        assert view.next_healthcheck_at == pytest.approx(clock.now() + 60)
        assert clock.now() == FAKE_EPOCH  # fake time did not move: the bound is real time
    finally:
        release.set()
        await sup.stop()
