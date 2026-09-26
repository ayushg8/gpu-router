"""Fake LightningClient: only the billing call the driver makes."""

from __future__ import annotations

from typing import Any

from lightning_sdk import _state


class LightningClient:
    def __init__(
        self, retry: bool = True, max_tries: int | None = 10, with_auth: bool = True
    ) -> None:
        _state.call("LightningClient", retry=retry)

    def billing_service_get_user_balance(self, **kwargs: Any) -> Any:
        _state.maybe_raise("balance")
        value = _state.STATE.get("balance")
        return type("R", (), {"balance": value, "total_spent": 1.0})()
