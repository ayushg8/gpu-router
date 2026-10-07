"""Secret storage and redaction (phase 1 minimal; owner: group A).

The ONLY way gpu-router reads or writes credentials (invariant 12). Backed by `keyring`,
which on macOS is the login Keychain. Service name "gpu-router"; the keyring username is
the secret name, e.g. "kaggle", "hf_token", "HF_TOKEN" (names are case-sensitive).

Rules:
- Values never go into config, SQLite, logs, job events, bundles, state.json, exception
  messages or `repr`s. Pass them around as `pydantic.SecretStr`.
- Every value read through `get_secret` is registered for redaction in this process.
- Tests must never touch the real Keychain: tests/conftest.py installs an in-memory backend
  via `use_backend()` for every test (autouse fixture).
- Names only (never values) are indexed in `<home>/secrets.index` so `gpu secrets list`
  and `gpu doctor` can enumerate them (keyring cannot enumerate).

This file is imported by the stdlib `logging` filter in gpu_router.log, so importing it must
stay cheap: `keyring` is imported lazily inside the functions.
"""

from __future__ import annotations

import os
import re
import threading
from typing import Any

SERVICE = "gpu-router"
REDACTED = "***"

#: Patterns redacted from every daemon log record and captured job log line, in addition to
#: the exact values registered at runtime.
TOKEN_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"hf_[A-Za-z0-9]{30,}"),  # Hugging Face
    re.compile(r"gh[pousr]_[A-Za-z0-9]{36,}"),  # GitHub
    re.compile(r"github_pat_[A-Za-z0-9_]{40,}"),
    re.compile(r"sk-(?:ant-|proj-)?[A-Za-z0-9_-]{20,}"),  # Anthropic / OpenAI style
    re.compile(r"xox[abprs]-[A-Za-z0-9-]{10,}"),  # Slack
    re.compile(r"AKIA[0-9A-Z]{16}"),  # AWS access key id
    re.compile(r"ak-[A-Za-z0-9]{20,}|as-[A-Za-z0-9]{20,}"),  # Modal token id / secret
    re.compile(r"gsk_[A-Za-z0-9]{20,}"),  # Groq (phase 7b inference lane)
    re.compile(r"AIza[0-9A-Za-z_-]{35}"),  # Google API key (Gemini, phase 7b)
    # JWT / JWE (signed download URLs: Kaggle's kernels output links carry a JWE whose
    # second part is empty, seen in a fetch error on 2026-10-05)
    re.compile(r"\beyJ[A-Za-z0-9_-]{8,}(?:\.[A-Za-z0-9_-]*){1,4}"),
    re.compile(r"(?i)(authorization\s*[:=]\s*bearer\s+)[^\s\"']+"),
    re.compile(r"(?i)((?:kaggle_key|api_key|token|secret|password)\s*[=:]\s*)[^\s\"',}]+"),
    re.compile(r"(?i)(\"key\"\s*:\s*\")[0-9a-f]{32}(\")"),  # kaggle.json shape
)

INDEX_FILE_NAME = "secrets.index"
MIN_REDACT_LEN = 6

_lock = threading.Lock()
_backend: Any = None
_registered: set[str] = set()
_registered_re: re.Pattern[str] | None = None


def _keyring_get(name: str) -> str | None:
    from gpu_router.errors import SecretsError

    try:
        if _backend is not None:
            value = _backend.get_password(SERVICE, name)
        else:
            import keyring

            value = keyring.get_password(SERVICE, name)
    except Exception as exc:  # keyring raises many backend-specific types
        raise SecretsError(
            f"could not read secret {name!r} from the Keychain ({type(exc).__name__})",
            hint="unlock the login Keychain and try again",
        ) from None
    return None if value is None else str(value)


def _index_path() -> str:
    from gpu_router.paths import home_from_env

    return os.path.join(home_from_env(), INDEX_FILE_NAME)


def _read_index() -> set[str]:
    try:
        with open(_index_path(), encoding="utf-8") as fh:
            return {line.strip() for line in fh if line.strip()}
    except OSError:
        return set()


def _write_index(names: set[str]) -> None:
    path = _index_path()
    os.makedirs(os.path.dirname(path), mode=0o700, exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write("".join(f"{n}\n" for n in sorted(names)))
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def use_backend(backend: Any) -> None:
    """Install a keyring backend for this process (tests: in-memory; never the Keychain)."""
    global _backend
    _backend = backend


def get_secret(name: str) -> str | None:
    """Return the secret or None if absent. Registers the value for redaction.
    Raises SecretsError if the Keychain is locked/unavailable."""
    value = _keyring_get(name)
    if value is not None:
        register_for_redaction(value)
    return value


def set_secret(name: str, value: str) -> None:
    """Store/replace a secret and add its name to secrets.index. Raises SecretsError."""
    from gpu_router.errors import SecretsError

    try:
        if _backend is not None:
            _backend.set_password(SERVICE, name, value)
        else:
            import keyring

            keyring.set_password(SERVICE, name, value)
    except Exception as exc:
        raise SecretsError(
            f"could not store secret {name!r} in the Keychain ({type(exc).__name__})",
            hint="unlock the login Keychain and try again",
        ) from None
    register_for_redaction(value)
    with _lock:
        names = _read_index()
        names.add(name)
        _write_index(names)


def delete_secret(name: str) -> None:
    """Remove a secret (no-op if absent) and drop it from secrets.index."""
    from gpu_router.errors import SecretsError

    if _keyring_get(name) is not None:
        try:
            if _backend is not None:
                _backend.delete_password(SERVICE, name)
            else:
                import keyring

                keyring.delete_password(SERVICE, name)
        except Exception as exc:
            raise SecretsError(
                f"could not delete secret {name!r} from the Keychain ({type(exc).__name__})",
                hint="unlock the login Keychain and try again",
            ) from None
    with _lock:
        names = _read_index()
        if name in names:
            names.discard(name)
            _write_index(names)


def secret_names() -> list[str]:
    """Names from secrets.index, sorted."""
    with _lock:
        return sorted(_read_index())


def register_for_redaction(value: str) -> None:
    """Add an exact value to the in-process redaction set (ignored if shorter than 6 chars)."""
    global _registered_re
    if len(value) < MIN_REDACT_LEN:
        return
    with _lock:
        if value in _registered:
            return
        _registered.add(value)
        # Longest first so a value containing another is replaced whole.
        ordered = sorted(_registered, key=len, reverse=True)
        _registered_re = re.compile("|".join(re.escape(v) for v in ordered))


def redact(text: str) -> str:
    """Replace registered values and TOKEN_PATTERNS matches with REDACTED, keeping any
    captured prefix group (e.g. 'Authorization: Bearer ***'). Pure; safe on any string."""
    if not text:
        return text
    pattern = _registered_re
    if pattern is not None:
        text = pattern.sub(REDACTED, text)
    for token_re in TOKEN_PATTERNS:
        text = token_re.sub(_replace_match, text)
    return text


def _replace_match(match: re.Match[str]) -> str:
    """Keep captured prefix/suffix groups (e.g. 'Authorization: Bearer ') around REDACTED."""
    groups = match.groups()
    if not groups:
        return REDACTED
    prefix = groups[0] or ""
    suffix = groups[1] if len(groups) > 1 and groups[1] else ""
    return f"{prefix}{REDACTED}{suffix}"


def _reset_redaction_for_tests() -> None:
    """Forget registered values (tests only)."""
    global _registered_re
    with _lock:
        _registered.clear()
        _registered_re = None
