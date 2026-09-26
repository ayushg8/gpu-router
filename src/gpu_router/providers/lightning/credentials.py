"""Where the Lightning AI credentials come from (invariant 12).

Lightning authenticates API calls with a user id + API key (HTTP Basic `user_id:api_key`,
NOTES.md "Login"). Modes (config `providers.lightning.login_source`, default "auto"):

- "keychain": Keychain only, through `gpu_router.secrets` (service "gpu-router"), secrets
  `LIGHTNING_USER_ID` and `LIGHTNING_API_KEY` (written by `gpu login lightning`).
- "env": LIGHTNING_USER_ID + LIGHTNING_API_KEY in the daemon's own environment.
- "file": the `lightning login` file (~/.lightning/credentials.json, keys `user_id`,
  `api_key`), read by gpu-router; the SDK itself never sees that path (the adapter points
  LIGHTNING_CREDENTIAL_PATH inside the data dir so it cannot write ~/.lightning either).
- "auto": keychain, else env, else file.

Values reach the SDK subprocess only through its environment, never argv, logs or files.
Every value read here is registered for redaction.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import SecretStr

from gpu_router.errors import AuthRequired, SecretsError

__all__ = [
    "KEYCHAIN_KEY",
    "KEYCHAIN_USER",
    "Credentials",
    "default_file",
    "read_file",
    "resolve",
]

KEYCHAIN_USER = "LIGHTNING_USER_ID"
KEYCHAIN_KEY = "LIGHTNING_API_KEY"  # a secret name, not a value
ENV_USER = "LIGHTNING_USER_ID"
ENV_KEY = "LIGHTNING_API_KEY"
#: every env var through which the SDK could pick up another identity; always cleared from
#: the inherited environment of the SDK child before ours are set.
SDK_ENV_VARS = (
    "LIGHTNING_USER_ID",
    "LIGHTNING_API_KEY",
    "LIGHTNING_AUTH_TOKEN",
    "LIGHTNING_USERNAME",
    "LIGHTNING_TEAMSPACE",
    "LIGHTNING_ORG",
    "LIGHTNING_CLOUD_URL",
    "LIGHTNING_CREDENTIAL_PATH",
    "LIGHTNING_SETTINGS_PATH",
    "LIGHTNING_CLOUD_PROJECT_ID",
    "LIGHTNING_CLOUD_SPACE_ID",
    "LIGHTNING_CLUSTER_ID",
)
LOGIN_HINT = (
    "run `gpu login lightning` (lightning.ai > Settings > Keys has your user id and API key)"
)

Mode = Literal["auto", "keychain", "env", "file"]
Source = Literal["keychain", "env", "file"]


@dataclass(frozen=True)
class Credentials:
    source: Source
    user_id: SecretStr
    api_key: SecretStr

    def __repr__(self) -> str:  # never show values, even masked ones
        return f"Credentials(source={self.source!r})"

    def plain_env(self) -> dict[str, str]:
        """For the SDK subprocess environment only."""
        return {
            ENV_USER: self.user_id.get_secret_value(),
            ENV_KEY: self.api_key.get_secret_value(),
        }


def default_file() -> Path:
    return Path.home() / ".lightning" / "credentials.json"


def _clean(value: object) -> str:
    return str(value).strip() if value is not None else ""


def read_file(path: Path) -> tuple[str, str] | None:
    """(user_id, api_key) from a `lightning login` credentials file; None when the file
    is absent or has no API key. Values are registered for redaction, never returned in
    an error."""
    from gpu_router import secrets

    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        doc = json.loads(raw)
    except ValueError:
        raise AuthRequired(
            f"{path} is not a JSON credentials file", provider="lightning", hint=LOGIN_HINT
        ) from None
    if not isinstance(doc, dict):
        return None
    user_id, api_key = _clean(doc.get("user_id")), _clean(doc.get("api_key"))
    for value in (user_id, api_key):
        if value:
            secrets.register_for_redaction(value)
    if not user_id or not api_key:
        return None
    return user_id, api_key


def _from_keychain(provider: str, strict: bool) -> Credentials | None:
    from gpu_router import secrets

    try:
        user_id = secrets.get_secret(KEYCHAIN_USER)
        api_key = secrets.get_secret(KEYCHAIN_KEY)
    except SecretsError as exc:
        if not strict:
            return None
        raise AuthRequired(
            f"could not read lightning credentials from the Keychain: {exc.message}",
            provider=provider,
            hint=exc.hint,
        ) from None
    if user_id and api_key and user_id.strip() and api_key.strip():
        return Credentials(
            source="keychain",
            user_id=SecretStr(user_id.strip()),
            api_key=SecretStr(api_key.strip()),
        )
    return None


def _from_env(environ: Mapping[str, str]) -> Credentials | None:
    from gpu_router import secrets

    user_id, api_key = _clean(environ.get(ENV_USER)), _clean(environ.get(ENV_KEY))
    if not user_id or not api_key:
        return None
    secrets.register_for_redaction(user_id)
    secrets.register_for_redaction(api_key)
    return Credentials(source="env", user_id=SecretStr(user_id), api_key=SecretStr(api_key))


def _from_file(path: Path) -> Credentials | None:
    found = read_file(path)
    if found is None:
        return None
    return Credentials(source="file", user_id=SecretStr(found[0]), api_key=SecretStr(found[1]))


def resolve(
    mode: str,
    *,
    provider: str = "lightning",
    environ: Mapping[str, str] | None = None,
    file: Path | None = None,
) -> Credentials | None:
    """Credentials for the next SDK call, or None when the chosen mode has none. Raises
    AuthRequired for an unknown mode, a locked Keychain in "keychain" mode or a broken
    credentials file."""
    if mode not in ("auto", "keychain", "env", "file"):
        raise AuthRequired(
            f"unknown lightning credentials mode {mode!r}",
            provider=provider,
            hint="set providers.lightning.login_source to auto, keychain, env or file",
        )
    env = os.environ if environ is None else environ
    path = file or default_file()
    if mode in ("auto", "keychain"):
        found = _from_keychain(provider, strict=mode == "keychain")
        if found is not None or mode == "keychain":
            return found
    if mode in ("auto", "env"):
        found = _from_env(env)
        if found is not None or mode == "env":
            return found
    return _from_file(path)
