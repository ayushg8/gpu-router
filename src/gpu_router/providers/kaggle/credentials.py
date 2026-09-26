"""Where the kaggle CLI's credentials come from (invariant 12).

Modes (config `providers.kaggle.credentials`, default "auto"):

- "keychain": Keychain only, through `gpu_router.secrets` (service "gpu-router"):
  secret `kaggle` = the kaggle.json document {"username": ..., "key": ...}, or secret
  `KAGGLE_API_TOKEN` = an access token. Values reach the CLI subprocess only as
  environment variables (KAGGLE_USERNAME/KAGGLE_KEY or KAGGLE_API_TOKEN), with
  KAGGLE_CONFIG_DIR pointed at an empty private dir so a stale ~/.kaggle file cannot win.
- "cli": the CLI's own files (~/.kaggle/kaggle.json, access_token, OAuth cache). gpu-router
  never opens those files; the CLI reads them itself.
- "auto": Keychain when either secret exists, else "cli".

Migration path from ~/.kaggle/kaggle.json to the Keychain (NOTES.md "Login"):
`gpu login kaggle` (phase 8b; `gpu secrets set kaggle --stdin < ~/.kaggle/kaggle.json` still
works), check `gpu providers`, then the file may be deleted (or kept: "auto" prefers the
Keychain either way).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import SecretStr

from gpu_router.errors import AuthRequired, SecretsError

__all__ = ["KEYCHAIN_JSON", "KEYCHAIN_TOKEN", "Credentials", "resolve"]

KEYCHAIN_JSON = "kaggle"
KEYCHAIN_TOKEN = "KAGGLE_API_TOKEN"  # noqa: S105 - a secret name, not a value
#: env vars the CLI reads; cleared from the inherited env whenever Keychain creds are used.
CLI_ENV_VARS = ("KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_API_TOKEN", "KAGGLE_CONFIG_DIR")

Mode = Literal["auto", "keychain", "cli"]


@dataclass(frozen=True)
class Credentials:
    source: Literal["keychain-json", "keychain-token", "cli"]
    env: dict[str, SecretStr] = field(default_factory=dict)
    username: str | None = None  # known without calling the CLI (keychain-json only)

    def __repr__(self) -> str:  # never show values, even masked ones
        return f"Credentials(source={self.source!r}, username={self.username!r})"

    def plain_env(self) -> dict[str, str]:
        """For the subprocess environment only."""
        return {k: v.get_secret_value() for k, v in self.env.items()}


def _from_json(raw: str, provider: str) -> tuple[str, str]:
    try:
        doc = json.loads(raw)
        username = str(doc["username"]).strip()
        key = str(doc["key"]).strip()
    except (ValueError, KeyError, TypeError):
        raise AuthRequired(
            f"the Keychain secret {KEYCHAIN_JSON!r} is not a kaggle.json document",
            provider=provider,
            hint=f"`gpu secrets set {KEYCHAIN_JSON} --stdin < ~/.kaggle/kaggle.json`",
        ) from None
    if not username or not key:
        raise AuthRequired(
            f"the Keychain secret {KEYCHAIN_JSON!r} has an empty username or key",
            provider=provider,
        )
    return username, key


def resolve(mode: str, *, provider: str, config_dir: Path) -> Credentials:
    """Credentials for the next CLI call. Raises AuthRequired when the chosen mode has none
    (or the Keychain is unreadable in "keychain" mode)."""
    from gpu_router import secrets

    if mode not in ("auto", "keychain", "cli"):
        raise AuthRequired(
            f"unknown kaggle credentials mode {mode!r}",
            provider=provider,
            hint="set providers.kaggle.credentials to auto, keychain or cli",
        )
    if mode == "cli":
        return Credentials(source="cli")
    try:
        raw_json = secrets.get_secret(KEYCHAIN_JSON)
        token = None if raw_json is not None else secrets.get_secret(KEYCHAIN_TOKEN)
    except SecretsError as exc:
        if mode == "auto":
            return Credentials(source="cli")
        raise AuthRequired(
            f"could not read kaggle credentials from the Keychain: {exc.message}",
            provider=provider,
            hint=exc.hint,
        ) from None
    isolated = {"KAGGLE_CONFIG_DIR": SecretStr(str(config_dir))}
    if raw_json is not None:
        username, key = _from_json(raw_json, provider)
        return Credentials(
            source="keychain-json",
            env={"KAGGLE_USERNAME": SecretStr(username), "KAGGLE_KEY": SecretStr(key), **isolated},
            username=username,
        )
    if token is not None:
        return Credentials(
            source="keychain-token",
            env={"KAGGLE_API_TOKEN": SecretStr(token.strip()), **isolated},
        )
    if mode == "keychain":
        raise AuthRequired(
            "no kaggle credentials in the Keychain",
            provider=provider,
            hint=f"`gpu secrets set {KEYCHAIN_JSON} --stdin < ~/.kaggle/kaggle.json`",
        )
    return Credentials(source="cli")
