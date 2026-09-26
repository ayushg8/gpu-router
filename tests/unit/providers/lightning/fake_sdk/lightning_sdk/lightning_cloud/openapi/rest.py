"""Fake ApiException with the real one's constructor (status, reason, http_resp) and
attributes (status, reason, body, headers); str() prints the headers like the real one."""

from __future__ import annotations

from typing import Any


class ApiException(Exception):
    def __init__(self, status: Any = None, reason: Any = None, http_resp: Any = None) -> None:
        self.status = status
        self.reason = reason
        self.body: Any = None
        self.headers: Any = {"Set-Cookie": "session=fake-cookie-must-not-leak"}

    def __str__(self) -> str:
        return f"({self.status})\nReason: {self.reason}\nHTTP response headers: {self.headers}\n"
