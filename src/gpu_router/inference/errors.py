"""Inference lane errors (phase 7b). They subclass `GpuRouterError`, so the daemon turns
them into the usual `{"error": {code, message, hint, detail}}` envelope; codes reuse
`errors.ErrorCode` values (API v1 is additive-only, invariant 17). Messages never carry
key values (invariant 12): provider error bodies are redacted and cut before they get here.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

from gpu_router.errors import ErrorCode, GpuRouterError

__all__ = [
    "InferAuthRequired",
    "InferBadRequest",
    "InferModelUnavailable",
    "InferQuotaExhausted",
    "InferRateLimited",
    "InferUnavailable",
    "InferenceError",
]


class InferenceError(GpuRouterError):
    """Base: something went wrong at an inference provider."""

    code: ClassVar[ErrorCode] = ErrorCode.PROVIDER_UNAVAILABLE
    http_status: ClassVar[int] = 502
    #: move on to the next candidate after this error (else the request itself is bad)
    reroute: ClassVar[bool] = True

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        hint: str | None = None,
        detail: Mapping[str, Any] | None = None,
    ) -> None:
        full = {**(detail or {}), **({"provider": provider} if provider else {})}
        super().__init__(message, hint=hint, detail=full)
        self.provider = provider
        #: quota facts the failed response still carried (ledger.LiveReading), recorded by
        #: the service; never part of the error envelope
        self.live: list[Any] = []


class InferUnavailable(InferenceError):
    """Outage, 5xx, timeout, network error, or a test-mode daemon refusing real calls.
    `status` is the HTTP status when the provider answered (5xx), None otherwise."""

    def __init__(
        self, message: str, *, status: int | None = None, sent: bool = False, **kw: Any
    ) -> None:
        super().__init__(message, **kw)
        self.status = status
        #: the request reached the provider before the read timed out (it may have run
        #: and billed the generation: the service records an estimate)
        self.sent = sent


class InferAuthRequired(InferenceError):
    """No key in the Keychain, or the provider rejected it."""

    code = ErrorCode.AUTH_REQUIRED


class InferRateLimited(InferenceError):
    """A per-minute limit: the provider asked us to come back after `retry_after` s."""

    code = ErrorCode.RATE_LIMITED
    http_status = 429

    def __init__(self, message: str, *, retry_after: float | None = None, **kw: Any) -> None:
        super().__init__(message, **kw)
        self.retry_after = retry_after
        if retry_after is not None:
            self.detail["retry_after_s"] = round(retry_after, 1)


class InferQuotaExhausted(InferenceError):
    """The free allowance of the window (day / month) is used up until `resets_at`.
    `scope` is the model id when the limit is per model, "" when provider-wide."""

    code = ErrorCode.QUOTA_EXHAUSTED
    http_status = 429

    def __init__(
        self, message: str, *, resets_at: float | None = None, scope: str = "", **kw: Any
    ) -> None:
        super().__init__(message, **kw)
        self.resets_at = resets_at
        self.scope = scope
        if resets_at is not None:
            self.detail["resets_at"] = resets_at


class InferModelUnavailable(InferenceError):
    """This provider does not serve the model (404 / unknown model): try another one."""

    code = ErrorCode.INVALID_REQUEST
    http_status = 400


class InferBadRequest(InferenceError):
    """The request itself is bad (too long, bad parameter): another provider won't help."""

    code = ErrorCode.INVALID_REQUEST
    http_status = 400
    reroute = False
