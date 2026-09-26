"""Inference keys (phase 7b): Keychain items per provider, only via gpu_router.secrets
(invariant 12). `gpu login groq|gemini|cloudflare` stores them; Hugging Face reuses the
`HF_TOKEN` that `gpu login hf` stores (checkpoint/tokens.py rules: Keychain only, never
$HF_TOKEN or the hf CLI's file implicitly).

Names are prefixed `INFER_` so they never collide with a job secret the user stored under
the provider's usual env name (`gpu secrets set GROQ_API_KEY` stays the user's), and they
are in `models.RESERVED_SECRET_NAMES`, so no job can ask for them (D48).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import SecretStr

from gpu_router import secrets
from gpu_router.errors import SecretsError
from gpu_router.inference.errors import InferAuthRequired

if TYPE_CHECKING:
    from gpu_router.inference.catalog import InferenceEntry

__all__ = ["INFERENCE_SECRET_NAMES", "key_status", "load_keys"]

#: every Keychain name the packaged catalog uses for inference (HF_TOKEN is reserved
#: already as the checkpoint admin token)
INFERENCE_SECRET_NAMES: frozenset[str] = frozenset(
    {
        "INFER_GROQ_API_KEY",
        "INFER_GEMINI_API_KEY",
        "INFER_CLOUDFLARE_API_TOKEN",
        "INFER_CLOUDFLARE_ACCOUNT_ID",
    }
)


def key_status(entry: InferenceEntry) -> tuple[bool, list[str], str | None]:
    """(all keys present, missing names, problem). Never raises: a locked Keychain is a
    problem string, not an exception."""
    missing: list[str] = []
    for name in entry.secrets.values():
        try:
            value = secrets.get_secret(name)
        except SecretsError as exc:
            return False, list(entry.secrets.values()), exc.message
        if not value:
            missing.append(name)
    return not missing, missing, None


def load_keys(entry: InferenceEntry) -> dict[str, SecretStr]:
    """role -> value for every secret the entry names. Raises InferAuthRequired naming
    the `gpu login` command when one is missing or the Keychain is locked."""
    out: dict[str, SecretStr] = {}
    for role, name in entry.secrets.items():
        try:
            value = secrets.get_secret(name)
        except SecretsError as exc:
            raise InferAuthRequired(
                f"{entry.name}: {exc.message}", provider=entry.name, hint=exc.hint
            ) from None
        if not value:
            raise InferAuthRequired(
                f"{entry.name} has no key in the Keychain",
                provider=entry.name,
                hint=f"run `gpu login {entry.login_name}`",
            )
        out[role] = SecretStr(value)
    return out
