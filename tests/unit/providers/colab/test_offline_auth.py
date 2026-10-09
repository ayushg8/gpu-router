"""A network outage is not a login problem (2026-10-08). After every wake on a flaky network
the daemon marked colab "login needed" while kaggle said NameResolutionError: the colab CLI
reports a failed credential refresh as "No valid default credentials found" (exit 0 for
`sessions`) whether the sign-in expired or Google cannot be reached. An auth-looking failure
now counts as a login problem only while Google's token endpoint answers."""

from __future__ import annotations

import socket
import threading
import time
from pathlib import Path

import pytest

from gpu_router.clock import SystemClock
from gpu_router.errors import AuthRequired, Unavailable
from gpu_router.models import ProviderHealth
from gpu_router.paths import Paths
from gpu_router.providers.colab import cli as colab_cli
from gpu_router.providers.colab.adapter import ColabAdapter, session_name
from gpu_router.providers.colab.cli import CliResult, classify
from gpu_router.providers.colab.state import RunState
from tests.contract.harness import ContractTarget, make_ctx, make_job
from tests.unit.providers.colab.helpers import ColabSim, make_bundle, with_bundle

NO_ADC = "No valid default credentials found. To authenticate, run:\n"
#: Captured at import, before the conftest's autouse stub replaces it for each test.
REAL_REACHABLE = colab_cli.google_reachable


def _res(stderr: str, code: int = 1) -> CliResult:
    return CliResult(("sessions",), code, "", stderr)


def test_auth_text_while_google_is_unreachable_is_an_outage() -> None:
    err = classify(_res(NO_ADC), provider="colab", op="sessions", reachable=lambda: False)
    assert isinstance(err, Unavailable)
    assert "network looks down" in err.message
    err = classify(_res(NO_ADC), provider="colab", op="sessions", reachable=lambda: True)
    assert isinstance(err, AuthRequired)
    assert classify(_res(NO_ADC), provider="colab", op="sessions").__class__ is AuthRequired


def test_the_probe_runs_only_for_auth_looking_failures() -> None:
    calls: list[int] = []

    def reachable() -> bool:
        calls.append(1)
        return False

    scope = "your credentials are missing an OAuth scope required by Colab"
    assert isinstance(
        classify(_res(scope), provider="colab", op="new", reachable=reachable), AuthRequired
    )
    assert isinstance(
        classify(_res("Max retries exceeded"), provider="colab", op="new", reachable=reachable),
        Unavailable,
    )
    assert calls == []


def test_healthcheck_says_unavailable_when_google_cannot_be_reached(
    colab: ColabAdapter, sim: ColabSim, monkeypatch: pytest.MonkeyPatch
) -> None:
    sim.control(sessions_mode="auth")  # what the real CLI prints offline, exit 0
    assert colab.healthcheck().health is ProviderHealth.AUTH_REQUIRED
    monkeypatch.setattr(colab_cli, "google_reachable", lambda *a, **k: False)
    h = colab.healthcheck()
    assert h.health is ProviderHealth.UNAVAILABLE
    assert h.reason is not None
    assert "oauth2.googleapis.com is unreachable" in h.reason


def test_submit_offline_is_not_a_definitive_login_refusal(
    colab: ColabAdapter,
    sim: ColabSim,
    paths: Paths,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unavailable from `colab new` is ambiguous (invariant 6): the name is stopped, the
    record abandoned, lookup_by_key finds nothing (the engine rejects the attempt), and the
    next attempt submits normally once the network is back."""
    monkeypatch.setattr(colab_cli, "google_reachable", lambda *a, **k: False)
    sim.control(new="auth")
    job = make_job(ContractTarget(name="colab", build=lambda: colab, clock=SystemClock()))
    ctx = with_bundle(make_ctx(job), make_bundle(paths, tmp_path))
    with pytest.raises(Unavailable) as info:
        colab.submit(job, ctx)
    assert not isinstance(info.value, AuthRequired)
    assert info.value.hint is not None
    assert "gcloud auth application-default login" in info.value.hint
    assert "stop" in sim.commands()
    assert colab.lookup_by_key(ctx.attempt_key) is None
    rec = colab.store.load(session_name(ctx.attempt_key))
    assert rec is not None
    assert rec.state is RunState.ABANDONED

    monkeypatch.setattr(colab_cli, "google_reachable", lambda *a, **k: True)
    sim.control(new=None)
    retry = make_ctx(job, n=2)
    ref = colab.submit(job, with_bundle(retry, make_bundle(paths, tmp_path / "again")))
    assert ref.remote_id == session_name(retry.attempt_key)


def test_network_words_next_to_auth_words_skip_the_probe() -> None:
    def never() -> bool:
        raise AssertionError("not needed: the text already says the network failed")

    text = "RefreshError: ('Failed to retrieve token', ConnectionError('Max retries exceeded'))"
    err = classify(_res(text), provider="colab", op="new", reachable=never)
    assert isinstance(err, Unavailable)


def _local_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(colab_cli, "_behind_proxy", lambda _host: False)


def test_google_reachable_against_local_sockets(monkeypatch: pytest.MonkeyPatch) -> None:
    _local_only(monkeypatch)
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]
        monkeypatch.setattr(colab_cli, "GOOGLE_TOKEN_HOST", ("127.0.0.1", port))
        assert REAL_REACHABLE(timeout=2.0) is True
    assert REAL_REACHABLE(timeout=2.0) is False  # closed: refused


def test_a_hanging_lookup_counts_as_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    _local_only(monkeypatch)
    release = threading.Event()

    def hang(*_a: object, **_k: object) -> list[object]:
        release.wait(5)
        raise OSError("gave up")

    monkeypatch.setattr(colab_cli.socket, "getaddrinfo", hang)
    started = time.monotonic()
    try:
        assert REAL_REACHABLE(timeout=0.2) is False
    finally:
        release.set()
    assert time.monotonic() - started < 3


def test_ipv4_is_tried_first() -> None:
    v6 = (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("2001:db8::1", 443, 0, 0))
    v4 = (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("192.0.2.1", 443))
    assert colab_cli.connect_order([v6, v6, v4]) == [v4, v6, v6]
    assert len(colab_cli.connect_order([v6] * 9)) == colab_cli.REACH_MAX_ADDRS


def test_behind_a_proxy_the_probe_cannot_tell(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:8080")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)

    def boom(*_a: object, **_k: object) -> list[object]:
        raise AssertionError("no direct connection behind a proxy")

    monkeypatch.setattr(colab_cli.socket, "getaddrinfo", boom)
    assert REAL_REACHABLE() is True
    monkeypatch.setenv("NO_PROXY", "googleapis.com")  # bypassed: the probe means something
    assert colab_cli._behind_proxy("oauth2.googleapis.com") is False
