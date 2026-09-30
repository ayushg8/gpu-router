"""Shared fixtures for every test (owner: group A). Invariant 20: no test touches real
providers, the real Keychain or the user's data dir.

Autouse:
- `gpu_home`       tmp GPU_ROUTER_HOME (+ GPU_ROUTER_TEST_MODE=1), clears other GPU_ROUTER_* vars,
                   PYTHON_KEYRING_BACKEND=null for child processes
- `memory_keyring` in-memory keyring backend for this process

On request:
- `paths`       Paths for the tmp home (dirs created)
- `clock`       FakeClock at FAKE_EPOCH; drive with clock.advance(dt) then `await settle()`
- `test_config` Config tuned for fast tests (test_mode, port 0, short engine timers)
- `catalog`     packaged provider catalog (includes the test-only fakes)
- `store`       migrated in-memory Store on `clock`

Area-specific fixtures belong in tests/<area>/conftest.py, owned by that area's group.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import keyring
import keyring.backend
import pytest

from gpu_router.clock import FakeClock
from gpu_router.config import Config, DaemonConfig, EngineConfig
from gpu_router.paths import ENV_HOME, Paths

ENV_REAL_PROVIDERS = "GPU_ROUTER_REAL_PROVIDERS"


# --------------------------------------------------------------------------- keyring


class MemoryKeyring(keyring.backend.KeyringBackend):
    """Process-local keyring. Never touches the macOS Keychain."""

    priority = 1  # type: ignore[assignment]

    def __init__(self) -> None:
        super().__init__()
        self.store: dict[tuple[str, str], str] = {}

    def get_password(self, service: str, username: str) -> str | None:
        return self.store.get((service, username))

    def set_password(self, service: str, username: str, password: str) -> None:
        self.store[(service, username)] = password

    def delete_password(self, service: str, username: str) -> None:
        from keyring.errors import PasswordDeleteError

        try:
            del self.store[(service, username)]
        except KeyError:
            raise PasswordDeleteError(username) from None


@pytest.fixture(autouse=True)
def memory_keyring() -> Iterator[MemoryKeyring]:
    backend = MemoryKeyring()
    previous = keyring.get_keyring()
    keyring.set_keyring(backend)
    from gpu_router import secrets

    secrets.use_backend(backend)
    secrets._reset_redaction_for_tests()
    yield backend
    secrets._reset_redaction_for_tests()
    keyring.set_keyring(previous)


# --------------------------------------------------------------------------- home / env


@pytest.fixture(autouse=True)
def gpu_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "gpu-home"
    for key in list(os.environ):
        if key.startswith("GPU_ROUTER_") and key != ENV_REAL_PROVIDERS:
            monkeypatch.delenv(key)
    # a suite run from inside Claude Code / Codex must not turn every `gpu run` in the
    # tests into an agent job (agent.py markers, D48)
    for key in ("CLAUDECODE", "CODEX_SANDBOX", "CODEX_SANDBOX_NETWORK_DISABLED"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv(ENV_HOME, str(home))
    monkeypatch.setenv("GPU_ROUTER_TEST_MODE", "1")
    # invariant 20 for child processes too (daemon subprocesses, the real `gpu` executable,
    # `gpu mcp` over stdio): memory_keyring only covers this process, so children get
    # keyring's null backend and can never read, write or prompt for the macOS Keychain
    # (CI runners included)
    monkeypatch.setenv("PYTHON_KEYRING_BACKEND", "keyring.backends.null.Keyring")
    return home


@pytest.fixture
def paths(gpu_home: Path) -> Paths:
    p = Paths.from_env()
    p.ensure()
    return p


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def test_config() -> Config:
    return Config(
        test_mode=True,
        daemon=DaemonConfig(port=0, shutdown_grace_s=1),
        engine=EngineConfig(
            backoff_base_s=1,
            backoff_cap_s=8,
            default_poll_interval_s=1,
            max_workers=4,
        ),
    )


@pytest.fixture
def catalog() -> Any:
    from gpu_router.providers.catalog import load_catalog

    return load_catalog(None)


@pytest.fixture
def store(clock: FakeClock) -> Iterator[Any]:
    from gpu_router.store import Store

    s = Store.open_memory(clock)
    yield s
    s.close()


# --------------------------------------------------------------------------- markers


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip @pytest.mark.real_provider("<name>") unless GPU_ROUTER_REAL_PROVIDERS lists it
    (comma-separated). The contract suite additionally requires a passing healthcheck."""
    allowed = {p.strip() for p in os.environ.get(ENV_REAL_PROVIDERS, "").split(",") if p.strip()}
    for item in items:
        for mark in item.iter_markers("real_provider"):
            name = mark.args[0] if mark.args else None
            if name not in allowed:
                item.add_marker(
                    pytest.mark.skip(
                        reason=f"real provider {name!r} not enabled ({ENV_REAL_PROVIDERS})"
                    )
                )
