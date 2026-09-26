"""Hugging Face tokens for checkpoint storage (phase 5). Values only via gpu_router.secrets
(invariant 12): the Keychain, service `gpu-router`.

    HF_TOKEN          the daemon's own token (create the bucket, upload datasets, read
                      heartbeats/acks, write checkpoint requests). `gpu login hf`.
    HF_TOKEN_REMOTE   the least-privilege token handed to remote runtimes (Kaggle,
                      Colab) as the job secret GPU_STORAGE_TOKEN. `gpu login hf --remote`.
                      Required for remote storage (D44): the job, and anything pip
                      installs, runs as the same user as the runner and can read it (on
                      Kaggle the secrets file sits on a read-only input mount), so the
                      admin HF_TOKEN is never sent. Without it remote runs get no storage
                      and a note says how to add one. Create it as a fine-grained token
                      with write access to your namespace's buckets only (whether HF can
                      scope a token to one bucket is not documented; NOTES in
                      docs/notes/hf-hub.md).

`HF_TOKEN` in the environment and the hf CLI's `~/.cache/huggingface/token` are never
read implicitly (a test-mode daemon must not pick up a real token, and a remote must never
get a token the user did not hand to gpu-router); `gpu login hf --import` copies one of
them into the Keychain on request.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import SecretStr

from gpu_router import secrets
from gpu_router.errors import SecretsError

__all__ = [
    "ADMIN_SECRET",
    "REMOTE_SECRET",
    "admin_token",
    "external_token",
    "remote_token",
    "token_shape_problem",
]

ADMIN_SECRET = "HF_TOKEN"  # noqa: S105 - Keychain names, not values
REMOTE_SECRET = "HF_TOKEN_REMOTE"  # noqa: S105


def _get(name: str) -> SecretStr | None:
    value = secrets.get_secret(name)  # registers the value for redaction
    return SecretStr(value) if value else None


def admin_token() -> SecretStr | None:
    """The daemon's HF token, or None. Raises SecretsError when the Keychain is locked."""
    return _get(ADMIN_SECRET)


def remote_token() -> SecretStr | None:
    """The token remote runtimes get: HF_TOKEN_REMOTE, else None (never the admin
    HF_TOKEN: a remote job can read what its runner holds, D44)."""
    return _get(REMOTE_SECRET)


def external_token(environ: dict[str, str] | None = None) -> tuple[str, str] | None:
    """(where, value) of a token outside gpu-router: $HF_TOKEN, else the hf CLI's token
    file ($HF_TOKEN_PATH, else $HF_HOME/token, else ~/.cache/huggingface/token). Only for
    an explicit `gpu login hf --import`; the value is registered for redaction."""
    env = dict(os.environ) if environ is None else environ
    value = (env.get("HF_TOKEN") or "").strip()
    if value:
        secrets.register_for_redaction(value)
        return "$HF_TOKEN", value
    raw_path = env.get("HF_TOKEN_PATH")
    if raw_path:
        path = Path(raw_path).expanduser()
    else:
        hf_home = env.get("HF_HOME")
        base = Path(hf_home).expanduser() if hf_home else Path.home() / ".cache" / "huggingface"
        path = base / "token"
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not text:
        return None
    secrets.register_for_redaction(text)
    return str(path), text


def token_shape_problem(value: str) -> str | None:
    """Why `value` does not look like a Hugging Face user token (hf_...), or None."""
    if not value:
        return "the token is empty"
    if any(ch.isspace() for ch in value):
        return "the token contains whitespace"
    if not value.startswith("hf_") or len(value) < 20:
        return "Hugging Face tokens start with hf_ (create one at huggingface.co/settings/tokens)"
    return None


def safe_admin_token() -> tuple[SecretStr | None, str | None]:
    """(token, problem): never raises; a locked Keychain is a problem string."""
    try:
        return admin_token(), None
    except SecretsError as exc:
        return None, exc.message


def safe_remote_token() -> tuple[SecretStr | None, str | None]:
    try:
        return remote_token(), None
    except SecretsError as exc:
        return None, exc.message
