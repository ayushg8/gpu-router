"""credentials.py: modes, precedence, the `lightning login` file, redaction."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gpu_router import secrets
from gpu_router.errors import AuthRequired
from gpu_router.providers.lightning.credentials import read_file, resolve

USER = "user-0123456789"
KEY = "key-abcdef0123456789"


def _file(tmp_path: Path, doc: object) -> Path:
    path = tmp_path / "credentials.json"
    path.write_text(json.dumps(doc) if not isinstance(doc, str) else doc)
    return path


def test_keychain_wins_over_env_and_file(tmp_path: Path) -> None:
    secrets.set_secret("LIGHTNING_USER_ID", USER)
    secrets.set_secret("LIGHTNING_API_KEY", KEY)
    env = {"LIGHTNING_USER_ID": "env-user-00", "LIGHTNING_API_KEY": "env-key-000"}
    found = resolve("auto", environ=env, file=_file(tmp_path, {"user_id": "f", "api_key": "g"}))
    assert found is not None
    assert found.source == "keychain"
    assert found.plain_env() == {"LIGHTNING_USER_ID": USER, "LIGHTNING_API_KEY": KEY}


def test_env_then_file_then_nothing(tmp_path: Path) -> None:
    env = {"LIGHTNING_USER_ID": USER, "LIGHTNING_API_KEY": KEY}
    assert resolve("auto", environ=env, file=tmp_path / "none").source == "env"  # type: ignore[union-attr]
    path = _file(tmp_path, {"user_id": USER, "api_key": KEY, "auth_token": ""})
    found = resolve("auto", environ={}, file=path)
    assert found is not None
    assert found.source == "file"
    assert resolve("auto", environ={}, file=tmp_path / "none") is None


def test_explicit_modes_do_not_fall_through(tmp_path: Path) -> None:
    env = {"LIGHTNING_USER_ID": USER, "LIGHTNING_API_KEY": KEY}
    assert resolve("keychain", environ=env) is None
    assert (
        resolve("env", environ={}, file=_file(tmp_path, {"user_id": USER, "api_key": KEY})) is None
    )
    assert resolve("file", environ=env, file=tmp_path / "none") is None
    with pytest.raises(AuthRequired, match="unknown lightning credentials mode"):
        resolve("magic")


def test_a_broken_file_is_explained_without_its_content(tmp_path: Path) -> None:
    path = _file(tmp_path, '{"user_id": "abc", "api_key": ')
    with pytest.raises(AuthRequired) as info:
        read_file(path)
    assert "abc" not in info.value.message
    assert read_file(_file(tmp_path, ["not", "a", "dict"])) is None
    assert read_file(_file(tmp_path, {"user_id": USER, "api_key": ""})) is None


def test_values_are_registered_for_redaction_and_never_shown(tmp_path: Path) -> None:
    found = resolve("file", environ={}, file=_file(tmp_path, {"user_id": USER, "api_key": KEY}))
    assert found is not None
    assert KEY not in repr(found)
    assert USER not in repr(found)
    assert secrets.redact(f"Basic {USER}:{KEY}") == "Basic ***:***"


def test_a_locked_keychain_only_fails_in_keychain_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    from gpu_router.errors import SecretsError

    def locked(name: str) -> str:
        raise SecretsError("locked", hint="unlock it")

    monkeypatch.setattr(secrets, "get_secret", locked)
    env = {"LIGHTNING_USER_ID": USER, "LIGHTNING_API_KEY": KEY}
    assert resolve("auto", environ=env).source == "env"  # type: ignore[union-attr]
    with pytest.raises(AuthRequired, match="Keychain"):
        resolve("keychain", environ=env)
