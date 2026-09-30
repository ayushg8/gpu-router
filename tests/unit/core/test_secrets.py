from __future__ import annotations

from typing import Any

import pytest

from gpu_router import secrets
from gpu_router.errors import SecretsError
from gpu_router.paths import Paths


def test_roundtrip_and_index(paths: Paths, memory_keyring: Any) -> None:
    assert secrets.get_secret("HF_TOKEN") is None
    secrets.set_secret("HF_TOKEN", "value-123456")
    secrets.set_secret("kaggle", "k-abcdefgh")
    assert memory_keyring.store[(secrets.SERVICE, "HF_TOKEN")] == "value-123456"
    assert secrets.get_secret("HF_TOKEN") == "value-123456"
    assert secrets.secret_names() == ["HF_TOKEN", "kaggle"]
    index = paths.home / secrets.INDEX_FILE_NAME
    assert index.stat().st_mode & 0o777 == 0o600
    assert "value-123456" not in index.read_text()
    secrets.delete_secret("HF_TOKEN")
    secrets.delete_secret("HF_TOKEN")  # no-op when absent
    assert secrets.get_secret("HF_TOKEN") is None
    assert secrets.secret_names() == ["kaggle"]


def test_read_values_are_redacted(memory_keyring: Any) -> None:
    memory_keyring.store[(secrets.SERVICE, "x")] = "supersecretvalue"
    assert secrets.redact("got supersecretvalue here") == "got supersecretvalue here"
    secrets.get_secret("x")
    assert secrets.redact("got supersecretvalue here") == "got *** here"


def test_short_values_not_registered() -> None:
    secrets.register_for_redaction("abc")
    assert secrets.redact("abc abc") == "abc abc"


def test_longest_registered_value_wins() -> None:
    secrets.register_for_redaction("abcdef")
    secrets.register_for_redaction("abcdefghij")
    assert secrets.redact("x abcdefghij y") == "x *** y"


@pytest.mark.parametrize(
    ("text", "want"),
    [
        ("token hf_" + "a" * 34 + " end", "token *** end"),
        ("ghp_" + "b" * 36, "***"),
        ("key sk-ant-" + "c" * 30, "key ***"),
        ("AKIA" + "A" * 16, "***"),
        ("Authorization: Bearer abc.def.ghi", "Authorization: Bearer ***"),
        ("api_key=abcd1234 rest", "api_key=*** rest"),
        ('{"username": "me", "key": "' + "0" * 32 + '"}', '{"username": "me", "key": "***"}'),
        ("password: hunter22", "password: ***"),
        ("nothing to see", "nothing to see"),
        ("", ""),
    ],
)
def test_token_patterns(text: str, want: str) -> None:
    assert secrets.redact(text) == want


def test_backend_errors_become_secrets_error() -> None:
    class Broken:
        def get_password(self, *_a: object) -> str:
            raise RuntimeError("locked with value hunter2hunter2")

        def set_password(self, *_a: object) -> None:
            raise RuntimeError("locked")

        def delete_password(self, *_a: object) -> None:
            raise RuntimeError("locked")

    secrets.use_backend(Broken())
    with pytest.raises(SecretsError) as info:
        secrets.get_secret("x")
    assert "hunter2" not in str(info.value)
    assert info.value.__cause__ is None
    with pytest.raises(SecretsError):
        secrets.set_secret("x", "value-123456")
    assert secrets.redact("value-123456") == "value-123456"  # not stored, not registered


def test_child_processes_never_reach_the_macos_keychain() -> None:
    """Invariant 20 across processes: the in-memory keyring covers only the test process;
    a daemon subprocess or the real `gpu` executable gets keyring's null backend, so it can
    neither read a real secret nor show a Keychain prompt (CI runners have no one to click)."""
    import subprocess
    import sys

    probe = "import keyring; print(type(keyring.get_keyring()).__module__)"
    out = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, timeout=60, check=True
    )
    assert out.stdout.strip() == "keyring.backends.null"
