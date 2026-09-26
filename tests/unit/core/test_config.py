from __future__ import annotations

from typing import Any

import pytest
import yaml

from gpu_router import config as cfg
from gpu_router.config import (
    CONFIG_VERSION,
    DEFAULT_PORT,
    Config,
    load_config,
    migrate_config,
    persist_migrated_config,
    save_config,
)
from gpu_router.errors import ConfigError
from gpu_router.paths import Paths


def test_defaults_when_file_absent(paths: Paths) -> None:
    c = load_config(paths, environ={})
    assert c.daemon.port == DEFAULT_PORT
    assert c.version == CONFIG_VERSION
    assert c.test_mode is False
    assert not paths.config.exists()  # never writes


def test_file_values_and_env_precedence(paths: Paths) -> None:
    paths.config.write_text(
        "version: 1\ndaemon:\n  port: 5000\nlogging:\n  level: debug\n"
        "providers:\n  kaggle:\n    enabled: false\n    notebook_prefix: gr\n"
    )
    c = load_config(paths, environ={})
    assert c.daemon.port == 5000
    assert c.logging.level == "DEBUG"
    assert c.providers["kaggle"].enabled is False
    assert c.providers["kaggle"].model_extra == {"notebook_prefix": "gr"}
    env = {"GPU_ROUTER_PORT": "0", "GPU_ROUTER_LOG_LEVEL": "warning", "GPU_ROUTER_TEST_MODE": "1"}
    c2 = load_config(paths, environ=env)
    assert c2.daemon.port == 0
    assert c2.logging.level == "WARNING"
    assert c2.test_mode is True


def test_test_mode_never_read_from_file(paths: Paths) -> None:
    paths.config.write_text("test_mode: true\n")
    assert load_config(paths, environ={}).test_mode is False


def test_uses_process_env_by_default(paths: Paths) -> None:
    # conftest sets GPU_ROUTER_TEST_MODE=1
    assert load_config(paths).test_mode is True


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("daemon: [1, 2\n", "not valid YAML"),
        ("- a\n- b\n", "must be a mapping"),
        ("daemon:\n  port: 99999\n", "daemon.port"),
        ("unknown_key: 1\n", "unknown_key"),
        ("version: 99\n", "newer gpu-router"),
        ("version: zero\n", "positive integer"),
        ("providers:\n  kaggle:\n    api_key: abc\n", "looks like a secret"),
        ("providers:\n  hf:\n    token: abc\n", "looks like a secret"),
    ],
)
def test_invalid_configs(paths: Paths, text: str, match: str) -> None:
    paths.config.write_text(text)
    with pytest.raises(ConfigError, match=match) as info:
        load_config(paths, environ={})
    assert info.value.code == "config_invalid"


def test_bad_env_port(paths: Paths) -> None:
    with pytest.raises(ConfigError, match="GPU_ROUTER_PORT"):
        load_config(paths, environ={"GPU_ROUTER_PORT": "abc"})


def test_empty_file_is_defaults(paths: Paths) -> None:
    paths.config.write_text("")
    assert load_config(paths, environ={}) == Config()


def test_save_roundtrip_and_mode(paths: Paths) -> None:
    c = Config(test_mode=True)
    c.daemon.port = 1234
    save_config(paths, c)
    assert paths.config.stat().st_mode & 0o777 == 0o600
    raw = yaml.safe_load(paths.config.read_text())
    assert "test_mode" not in raw
    assert raw["version"] == CONFIG_VERSION
    loaded = load_config(paths, environ={})
    assert loaded.daemon.port == 1234
    save_config(paths, loaded, backup_suffix="bak-manual")
    assert (paths.home / "config.yaml.bak-manual").exists()
    assert not list(paths.home.glob(".config.yaml.tmp*"))


def test_migrate_config_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cfg, "CONFIG_VERSION", 3)

    def v1_to_v2(raw: dict[str, Any]) -> dict[str, Any]:
        raw = dict(raw)
        raw["renamed"] = raw.pop("old", None)
        return raw

    def v2_to_v3(raw: dict[str, Any]) -> dict[str, Any]:
        return {**raw, "added": True}

    monkeypatch.setattr(cfg, "CONFIG_MIGRATIONS", {1: v1_to_v2, 2: v2_to_v3})
    out, changed = migrate_config({"version": 1, "old": 5})
    assert changed
    assert out == {"version": 3, "renamed": 5, "added": True}
    out2, changed2 = migrate_config({"version": 3})
    assert not changed2
    assert out2 == {"version": 3}
    monkeypatch.setattr(cfg, "CONFIG_MIGRATIONS", {1: v1_to_v2})
    with pytest.raises(ConfigError, match="no config migration from version 2"):
        migrate_config({"version": 1})


def test_migrate_config_missing_version_is_v1() -> None:
    out, changed = migrate_config({})
    assert out == {"version": 1}
    assert changed  # version key added


def test_persist_migrated_config(paths: Paths, monkeypatch: pytest.MonkeyPatch) -> None:
    assert persist_migrated_config(paths) is False  # no file
    paths.config.write_text("version: 1\ndaemon:\n  port: 4000\n")
    assert persist_migrated_config(paths) is False  # already current
    monkeypatch.setattr(cfg, "CONFIG_VERSION", 2)
    monkeypatch.setattr(cfg, "CONFIG_MIGRATIONS", {1: lambda raw: dict(raw)})
    assert persist_migrated_config(paths) is True
    assert yaml.safe_load(paths.config.read_text())["version"] == 2
    backup = paths.home / "config.yaml.bak-v1"
    assert yaml.safe_load(backup.read_text())["version"] == 1
