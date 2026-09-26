"""Fake lightning_sdk.utils.resolve."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

from lightning_sdk import User, _state


def _get_authed_user() -> User:
    _state.call("resolve._get_authed_user")
    _state.maybe_raise("whoami")
    return User(_state.STATE.get("user", "me"))


@contextlib.contextmanager
def skip_studio_setup() -> Iterator[None]:
    from lightning_sdk import Studio

    prev = getattr(Studio._skip_setup, "value", False)
    Studio._skip_setup.value = True
    try:
        yield
    finally:
        Studio._skip_setup.value = prev
