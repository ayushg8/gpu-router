"""Every exception gpu-router raises on purpose (phase 1; real code, owner: group A).

Three families:

1. `AdapterError` and its seven subclasses: the ONLY exceptions a provider adapter may raise
   (adapter contract rule A3). Two class-level flags drive the engine's reaction:

   | class          | retryable | reroute | engine reaction                                       |
   |----------------|-----------|---------|-------------------------------------------------------|
   | RateLimited    | yes       | yes     | cooldown provider max(retry_after, backoff); reroute  |
   | Unavailable    | yes       | yes     | cooldown provider backoff(k); reroute                 |
   | QuotaExhausted | no        | yes     | provider exhausted until resets_at; reroute           |
   | AuthRequired   | no        | yes     | provider health=auth_required; reroute; tell the user |
   | InvalidJob     | no        | yes     | exclude provider for THIS job; reroute                |
   | NotFound       | no        | no      | status() -> attempt lost; cancel() -> no-op           |
   | Permanent      | no        | no      | job failed (failure_kind=provider_error)              |

   `retryable` = the same provider may succeed later (after backoff).
   `reroute`   = another provider may succeed now.

2. Domain / API errors raised by the store, engine and daemon. Each has a stable `code`
   (the API error envelope's `error.code`) and an HTTP status.

3. Client-side errors (`DaemonUnavailable`, `SubmitUncertain`, `ApiError`) raised by
   gpu_router.client.

Message convention (UX principle 4): `message` says what happened; `hint` says what the
user can do. Engine events additionally say what the tool did next.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Any, ClassVar


class ErrorCode(StrEnum):
    """Stable machine codes used in the API error envelope and `--json` error output."""

    INTERNAL = "internal"
    INVALID_REQUEST = "invalid_request"
    INVALID_SPEC = "invalid_spec"
    UNAUTHORIZED = "unauthorized"
    FORBIDDEN = "forbidden"
    JOB_NOT_FOUND = "job_not_found"
    AMBIGUOUS_JOB_REF = "ambiguous_job_ref"
    PROVIDER_NOT_FOUND = "provider_not_found"
    INVALID_TRANSITION = "invalid_transition"
    CONFLICT = "conflict"
    NOT_READY = "not_ready"
    DAEMON_UNAVAILABLE = "daemon_unavailable"
    DAEMON_ALREADY_RUNNING = "daemon_already_running"
    CONFIG_INVALID = "config_invalid"
    SCHEMA_TOO_NEW = "schema_too_new"
    SECRETS_UNAVAILABLE = "secrets_unavailable"
    SUBMIT_UNCERTAIN = "submit_uncertain"  # client side: the submit may or may not have landed
    # adapter taxonomy (surface through the API only inside event details / provider views)
    RATE_LIMITED = "rate_limited"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    QUOTA_EXHAUSTED = "quota_exhausted"
    AUTH_REQUIRED = "auth_required"
    REMOTE_NOT_FOUND = "remote_not_found"
    INVALID_JOB = "invalid_job"
    PROVIDER_PERMANENT = "provider_permanent"


class GpuRouterError(Exception):
    """Base class. `message` = what happened; `hint` = what the user can do about it."""

    code: ClassVar[ErrorCode] = ErrorCode.INTERNAL
    http_status: ClassVar[int] = 500

    def __init__(
        self,
        message: str,
        *,
        hint: str | None = None,
        detail: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint
        self.detail: dict[str, Any] = dict(detail or {})

    def to_body(self) -> dict[str, Any]:
        """The `error` object of the API envelope: {code, message, hint, detail}."""
        return {
            "code": str(self.code),
            "message": self.message,
            "hint": self.hint,
            "detail": self.detail,
        }

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.message!r})"


# --------------------------------------------------------------------------- adapter taxonomy


class AdapterError(GpuRouterError):
    """Base of the adapter error taxonomy. Adapters raise only subclasses of this."""

    retryable: ClassVar[bool] = False
    reroute: ClassVar[bool] = False
    http_status: ClassVar[int] = 502

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        hint: str | None = None,
        detail: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message, hint=hint, detail=detail)
        self.provider = provider


class RateLimited(AdapterError):
    """The provider throttled us. `retry_after` (seconds) if the provider said so."""

    code = ErrorCode.RATE_LIMITED
    retryable = True
    reroute = True

    def __init__(
        self,
        message: str,
        *,
        retry_after: float | None = None,
        provider: str | None = None,
        hint: str | None = None,
        detail: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message, provider=provider, hint=hint, detail=detail)
        self.retry_after = retry_after


class Unavailable(AdapterError):
    """Outage, no GPU free right now, network error, CLI timeout. Transient."""

    code = ErrorCode.PROVIDER_UNAVAILABLE
    retryable = True
    reroute = True


class QuotaExhausted(AdapterError):
    """Free quota is used up. `resets_at` (epoch seconds) if known."""

    code = ErrorCode.QUOTA_EXHAUSTED
    retryable = False
    reroute = True

    def __init__(
        self,
        message: str,
        *,
        resets_at: float | None = None,
        provider: str | None = None,
        hint: str | None = None,
        detail: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message, provider=provider, hint=hint, detail=detail)
        self.resets_at = resets_at


class AuthRequired(AdapterError):
    """Not logged in, token expired or revoked, CLI not configured."""

    code = ErrorCode.AUTH_REQUIRED
    retryable = False
    reroute = True


class NotFound(AdapterError):
    """The provider has no run for this remote id / attempt key."""

    code = ErrorCode.REMOTE_NOT_FOUND
    retryable = False
    reroute = False


class InvalidJob(AdapterError):
    """This provider cannot run this job as specified (bundle too big, GPU type unsupported)."""

    code = ErrorCode.INVALID_JOB
    retryable = False
    reroute = True


class Permanent(AdapterError):
    """A failure no retry or reroute will fix. The job fails."""

    code = ErrorCode.PROVIDER_PERMANENT
    retryable = False
    reroute = False


ADAPTER_ERRORS: tuple[type[AdapterError], ...] = (
    RateLimited,
    Unavailable,
    QuotaExhausted,
    AuthRequired,
    NotFound,
    InvalidJob,
    Permanent,
)

#: Errors that prove the provider did NOT start anything for this submit. Only these let the
#: engine mark an attempt `rejected` directly. Any other outcome of submit() (Unavailable,
#: timeout, a non-taxonomy exception) is AMBIGUOUS: the request may have reached the
#: provider, so the engine resolves it by attempt_key before placing the job anywhere else
#: (invariant 6). Adapters raise these five from submit() ONLY when they know nothing was
#: created remotely; when in doubt they raise Unavailable.
DEFINITIVE_SUBMIT_ERRORS: tuple[type[AdapterError], ...] = (
    RateLimited,
    QuotaExhausted,
    AuthRequired,
    InvalidJob,
    Permanent,
)


class AdapterContractViolation(GpuRouterError):
    """Engine-side wrapper for an adapter that broke the contract: raised a non-taxonomy
    exception, returned the wrong type, or reported an impossible status transition. The
    engine treats it like Unavailable for routing (cooldown + reroute, ambiguous on submit)
    and logs `adapter.bug` with the traceback; it also counts toward the job's
    internal_error_limit."""

    code = ErrorCode.INTERNAL
    http_status = 500

    def __init__(self, provider: str, op: str, cause: BaseException | str) -> None:
        super().__init__(
            f"{provider} adapter broke the contract in {op}: {cause}",
            detail={
                "provider": provider,
                "op": op,
                "cause": type(cause).__name__ if isinstance(cause, BaseException) else str(cause),
            },
        )
        self.provider = provider
        self.op = op


# --------------------------------------------------------------------------- domain / API


class InvalidRequest(GpuRouterError):
    code = ErrorCode.INVALID_REQUEST
    http_status = 400


class InvalidSpec(GpuRouterError):
    """A JobSpec failed validation (bad path, both script and command, secret-looking env)."""

    code = ErrorCode.INVALID_SPEC
    http_status = 400


class Unauthorized(GpuRouterError):
    code = ErrorCode.UNAUTHORIZED
    http_status = 401


class Forbidden(GpuRouterError):
    """Non-loopback Host header or a browser Origin header (DNS-rebinding guard)."""

    code = ErrorCode.FORBIDDEN
    http_status = 403


class JobNotFound(GpuRouterError):
    code = ErrorCode.JOB_NOT_FOUND
    http_status = 404


class AmbiguousJobRef(GpuRouterError):
    """A job id prefix matched more than one job. `detail['matches']` lists full ids."""

    code = ErrorCode.AMBIGUOUS_JOB_REF
    http_status = 409


class ProviderNotFound(GpuRouterError):
    code = ErrorCode.PROVIDER_NOT_FOUND
    http_status = 404


class InvalidTransition(GpuRouterError):
    """A state change not in statemachine.TRANSITIONS (a bug, or a user action on a job
    whose state does not allow it, e.g. approving a running job)."""

    code = ErrorCode.INVALID_TRANSITION
    http_status = 409

    def __init__(self, from_state: str, to_state: str, *, hint: str | None = None) -> None:
        super().__init__(
            f"job cannot go from {from_state} to {to_state}",
            hint=hint,
            detail={"from": from_state, "to": to_state},
        )
        self.from_state = from_state
        self.to_state = to_state


class StaleState(GpuRouterError):
    """Compare-and-set failed: the job's state changed since the caller read it.
    The caller must reload the job and decide again; never blindly retry the write."""

    code = ErrorCode.CONFLICT
    http_status = 409

    def __init__(self, job_id: str, expected: str, actual: str) -> None:
        super().__init__(
            f"job {job_id} is {actual}, expected {expected}",
            detail={"job_id": job_id, "expected": expected, "actual": actual},
        )
        self.job_id = job_id
        self.expected = expected
        self.actual = actual


class Conflict(GpuRouterError):
    """A request conflicts with existing state (e.g. an Idempotency-Key reused for a
    different job spec). Generic 409; StaleState is the store-level CAS flavour."""

    code = ErrorCode.CONFLICT
    http_status = 409


class NotReady(GpuRouterError):
    """The daemon is starting up (recovery in progress)."""

    code = ErrorCode.NOT_READY
    http_status = 503


class ConfigError(GpuRouterError):
    code = ErrorCode.CONFIG_INVALID
    http_status = 500


class SchemaTooNew(GpuRouterError):
    """gpu.db was written by a newer gpu-router than this one."""

    code = ErrorCode.SCHEMA_TOO_NEW
    http_status = 500


class DaemonAlreadyRunning(GpuRouterError):
    code = ErrorCode.DAEMON_ALREADY_RUNNING
    http_status = 409


class SecretsError(GpuRouterError):
    """Keychain unavailable or locked."""

    code = ErrorCode.SECRETS_UNAVAILABLE
    http_status = 500


# --------------------------------------------------------------------------- client side


class DaemonUnavailable(GpuRouterError):
    """The client could not reach the daemon (not running, wrong port, still starting)."""

    code = ErrorCode.DAEMON_UNAVAILABLE
    http_status = 503


class SubmitUncertain(GpuRouterError):
    """A submit was sent but no answer came back (read timeout, dropped connection): the job
    may exist. `detail` carries the idempotency key; resubmitting with the same key is safe."""

    code = ErrorCode.SUBMIT_UNCERTAIN
    http_status = 504


class ApiError(GpuRouterError):
    """An error envelope whose code this client does not know (newer daemon)."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        status: int,
        hint: str | None = None,
        detail: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message, hint=hint, detail=detail)
        self.raw_code = code
        self.status = status


_CLIENT_ERRORS: tuple[type[GpuRouterError], ...] = (
    InvalidRequest,
    InvalidSpec,
    Unauthorized,
    Forbidden,
    JobNotFound,
    AmbiguousJobRef,
    ProviderNotFound,
    Conflict,
    NotReady,
    ConfigError,
    SchemaTooNew,
    DaemonAlreadyRunning,
    SecretsError,
    DaemonUnavailable,
    SubmitUncertain,
)

ERRORS_BY_CODE: dict[str, type[GpuRouterError]] = {str(c.code): c for c in _CLIENT_ERRORS}


def error_from_body(body: Mapping[str, Any], status: int) -> GpuRouterError:
    """Rebuild an exception from an API envelope's `error` object (used by the client).

    InvalidTransition has a structured constructor, so it is rebuilt as ApiError carrying
    the original code; callers match on `.raw_code` or `.code`. A `conflict` envelope
    (Conflict or StaleState on the daemon side) is rebuilt as Conflict.
    """
    code = str(body.get("code", ErrorCode.INTERNAL))
    message = str(body.get("message", "unknown error"))
    hint = body.get("hint")
    detail = body.get("detail") or {}
    cls = ERRORS_BY_CODE.get(code)
    if cls is not None:
        return cls(message, hint=hint, detail=detail)
    return ApiError(code, message, status=status, hint=hint, detail=detail)
