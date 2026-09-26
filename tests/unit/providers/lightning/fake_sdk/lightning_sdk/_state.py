"""Shared state for the fake SDK: loaded from FAKE_LIGHTNING_STATE, saved after changes."""

from __future__ import annotations

import base64
import json
import os
import time
from typing import Any

_PATH = os.environ.get("FAKE_LIGHTNING_STATE", "")
_CALLS = os.environ.get("FAKE_LIGHTNING_CALLS", "")


def load() -> dict[str, Any]:
    if not _PATH or not os.path.exists(_PATH):
        return {}
    with open(_PATH, encoding="utf-8") as fh:
        return json.load(fh)


STATE: dict[str, Any] = load()


def save() -> None:
    if _PATH:
        with open(_PATH, "w", encoding="utf-8") as fh:
            json.dump(STATE, fh)


def call(event: str, /, **fields: Any) -> None:
    if _CALLS:
        with open(_CALLS, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"call": event, **fields}, default=str) + "\n")


def maybe_raise(key: str) -> None:
    spec = (STATE.get("raise") or {}).get(key)
    if not spec:
        return
    if spec.get("skip"):  # let this many calls through first
        spec["skip"] = int(spec["skip"]) - 1
        save()
        return
    if spec.get("once"):
        STATE["raise"].pop(key)
        save()
    if spec.get("sleep"):
        time.sleep(float(spec["sleep"]))
    kind = spec.get("type", "ApiException")
    msg = spec.get("msg", "fake failure")
    if kind == "ApiException":
        from lightning_sdk.lightning_cloud.openapi.rest import ApiException

        exc = ApiException(status=spec.get("status"), reason=spec.get("reason"))
        body = spec.get("body")
        exc.body = body.encode() if isinstance(body, str) else body
        raise exc
    if kind == "ValueError":
        raise ValueError(msg)
    if kind == "PermissionError":
        raise PermissionError(msg)
    if kind == "RuntimeError":
        raise RuntimeError(msg)
    if kind == "AuthFailed":  # what the real SDK raises for a 401
        raise ConnectionError("Authentication failed. Please run `lightning login`.")
    if kind == "RetriesExhausted":  # the real retry wrapper's last word
        status = spec["status"]
        raise Exception(f"The create request failed to reach the server, response: {status}.")
    if kind == "ConnectionError":
        raise RequestsConnectionError(msg)
    raise Exception(msg)


#: mimics requests.exceptions.ConnectionError (not the builtin): same class name
RequestsConnectionError = type("ConnectionError", (Exception,), {})


def drive_put(path: str, data: bytes) -> None:
    STATE.setdefault("drive", {})[path.strip("/")] = base64.b64encode(data).decode()
    save()


def drive_get(path: str) -> bytes | None:
    raw = (STATE.get("drive") or {}).get(path.strip("/"))
    return None if raw is None else base64.b64decode(raw)
