"""Synchronous HTTP client for the daemon (phase 1; owner: group C).

Used by the CLI (phase 2), shell (phase 4, from a worker thread) and MCP server (phase 6).
Imports httpx + gpu_router.api only; never the store, engine or adapters (invariant 2).

Discovery (`GpuClient.from_env()`): read <home>/daemon.json (api.RuntimeInfo) for the port
and <home>/daemon.token for the bearer token. Missing file, refused connection or a
daemon.json whose pid is dead -> errors.DaemonUnavailable(hint=START_HINT, which points at
`gpu daemon start`).

Errors: any non-2xx response with an error envelope is rebuilt with
errors.error_from_body(body["error"], status) and raised; a non-envelope error body raises
ApiError(code="internal"). Transport errors -> DaemonUnavailable ("cannot reach" for a
refused/failed connection, "did not answer within Ns" for a timeout, which does not claim
the daemon is down). Every request sends `X-Gpu-Router-Client: <client_name>` and the
bearer token.

Submit: the daemon packages the project inside POST /v1/jobs (D15), which can take tens of
seconds for a big project, so `submit` uses SUBMIT_TIMEOUT_S. Only a connection that never
reached the daemon is retried automatically (same idempotency key); a timeout or dropped
connection after the request was sent raises errors.SubmitUncertain naming the key, because
the job may exist.
"""

from __future__ import annotations

import json as jsonlib
import os
import subprocess
import sys
import time
import uuid
from collections.abc import Iterator
from types import TracebackType
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

import httpx
from pydantic import TypeAdapter

from gpu_router.api import (
    API_PREFIX,
    IDEMPOTENCY_HEADER,
    EventList,
    HealthView,
    JobDetail,
    JobList,
    JobView,
    LogRecord,
    PolicyView,
    ProviderView,
    RouteDecision,
    RuntimeInfo,
    StatusView,
    SubmitRequest,
)
from gpu_router.errors import (
    ApiError,
    DaemonUnavailable,
    GpuRouterError,
    InvalidRequest,
    SubmitUncertain,
    error_from_body,
)
from gpu_router.ids import normalize_ref
from gpu_router.models import _PROVIDER_NAME, JobSpec, JobState, QuotaSnapshot

if TYPE_CHECKING:
    from gpu_router.paths import Paths
    from gpu_router.policy import PolicyConfig

DEFAULT_TIMEOUT_S = 10.0
#: The daemon bundles the project inside POST /v1/jobs (up to the 200 MB cap, plus git).
SUBMIT_TIMEOUT_S = 300.0
CLIENT_HEADER = "X-Gpu-Router-Client"
START_HINT = (
    "start it with `gpu daemon start` (or `gpu daemon install-launchd` to start it at login)"
)
BUSY_HINT = "it may be busy or stuck; `gpu daemon status` shows whether it is up"

_PROVIDERS = TypeAdapter(list[ProviderView])
_QUOTAS = TypeAdapter(list[QuotaSnapshot])


def _ref(ref: str) -> str:
    """A job ref as one safe URL path segment. Refs can come from an LLM agent (MCP tools)
    that read untrusted job logs, so '../daemon/shutdown#' must never reach another route:
    validate against the job-id alphabet, then quote anyway."""
    return quote(normalize_ref(ref), safe="")


def _provider(name: str) -> str:
    norm = name.strip().lower()
    if not _PROVIDER_NAME.match(norm):
        raise InvalidRequest(
            f"{name.strip()!r} is not a provider name",
            hint="provider names look like kaggle or fake-b (see `gpu providers`)",
        )
    return quote(norm, safe="")


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def read_runtime_info(paths: Paths) -> RuntimeInfo | None:
    """daemon.json, or None if missing/corrupt."""
    try:
        return RuntimeInfo.model_validate_json(paths.runtime.read_bytes())
    except (OSError, ValueError):
        return None


class GpuClient:
    def __init__(
        self,
        base_url: str,
        token: str | None,
        *,
        client_name: str = "cli",
        timeout_s: float = DEFAULT_TIMEOUT_S,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        """`transport` lets tests inject httpx.MockTransport or a WSGI-style transport."""
        self.base_url = base_url
        self.client_name = client_name
        self.timeout_s = timeout_s
        self._token = token
        self._transport = transport
        self._http: httpx.Client | None = None

    @classmethod
    def from_env(
        cls,
        paths: Paths | None = None,
        *,
        client_name: str = "cli",
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> GpuClient:
        """Discover the running daemon (module docstring). Raises DaemonUnavailable."""
        from gpu_router.daemon.auth import read_token
        from gpu_router.paths import Paths

        p = paths or Paths.from_env()
        info = read_runtime_info(p)
        if info is None:
            raise DaemonUnavailable("the gpu-router daemon is not running", hint=START_HINT)
        if not _pid_alive(info.pid):
            raise DaemonUnavailable(
                f"the gpu-router daemon (pid {info.pid}) is not running", hint=START_HINT
            )
        token = read_token(p)
        if token is None:
            raise DaemonUnavailable(
                "cannot read the daemon token (daemon.token)",
                hint="restart the daemon: `gpu daemon stop` then `gpu daemon start`",
            )
        return cls(info.base_url, token, client_name=client_name, timeout_s=timeout_s)

    # ------------------------------------------------------------------ plumbing

    def _client(self) -> httpx.Client:
        if self._http is None:
            headers = {CLIENT_HEADER: self.client_name}
            if self._token:
                headers["Authorization"] = f"Bearer {self._token}"
            self._http = httpx.Client(
                base_url=self.base_url,
                headers=headers,
                timeout=self.timeout_s,
                transport=self._transport,
                trust_env=False,
            )
        return self._http

    @staticmethod
    def _raise_for(resp: httpx.Response) -> None:
        if resp.is_success:
            return
        try:
            body = resp.json()
        except ValueError:
            body = None
        if isinstance(body, dict) and isinstance(body.get("error"), dict):
            raise error_from_body(body["error"], resp.status_code)
        raise ApiError(
            "internal",
            f"daemon returned HTTP {resp.status_code}",
            status=resp.status_code,
            detail={"body": resp.text[:500]},
        )

    def request(
        self,
        method: str,
        path: str,
        *,
        json: Any = None,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout_s: float | None = None,
    ) -> Any:
        """Send one request to `/v1<path>`; return decoded JSON (None for 202/204).
        Raises the rebuilt GpuRouterError on error envelopes, DaemonUnavailable on
        transport errors."""
        clean = {k: v for k, v in (params or {}).items() if v is not None}
        try:
            resp = self._client().request(
                method,
                f"{API_PREFIX}{path}",
                json=json,
                params=clean,
                headers=headers,
                timeout=self.timeout_s if timeout_s is None else timeout_s,
            )
        except httpx.TransportError as exc:
            raise self._transport_error(exc, timeout_s) from exc
        self._raise_for(resp)
        if resp.status_code in (202, 204) or not resp.content:
            return None
        return resp.json()

    def _transport_error(self, exc: httpx.TransportError, timeout_s: float | None) -> Exception:
        """DaemonUnavailable worded for what happened: a timeout means the daemon answered
        the connection but not the request, so it is not "down"."""
        name = type(exc).__name__
        if isinstance(exc, httpx.TimeoutException) and not isinstance(exc, httpx.ConnectTimeout):
            waited = self.timeout_s if timeout_s is None else timeout_s
            return DaemonUnavailable(
                f"the gpu-router daemon at {self.base_url} did not answer within {waited:g}s "
                f"({name})",
                hint=BUSY_HINT,
                detail={"timeout": True},
            )
        return DaemonUnavailable(
            f"cannot reach the gpu-router daemon at {self.base_url} ({name})", hint=START_HINT
        )

    def close(self) -> None:
        if self._http is not None:
            self._http.close()
            self._http = None

    def __enter__(self) -> GpuClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # ------------------------------------------------------------------ endpoints

    def health(self) -> HealthView:
        return HealthView.model_validate(self.request("GET", "/health"))

    def status(self) -> StatusView:
        return StatusView.model_validate(self.request("GET", "/status"))

    def submit(
        self,
        spec: JobSpec,
        *,
        idempotency_key: str | None = None,
        timeout_s: float = SUBMIT_TIMEOUT_S,
    ) -> JobView:
        """POST /v1/jobs with a long timeout (the daemon bundles the project in-request).

        Generates a uuid4 idempotency key when none is given. A connection that never
        reached the daemon (refused, connect timeout) is retried once with the same key. If
        the request was sent but no answer came back (read timeout, dropped connection),
        the job may exist: raises SubmitUncertain with the key instead of retrying (a
        replay would rebuild the bundle and wait just as long)."""
        key = idempotency_key or str(uuid.uuid4())
        body = SubmitRequest(spec=spec).model_dump(mode="json")
        headers = {IDEMPOTENCY_HEADER: key}
        for attempt in (1, 2):
            try:
                data = self.request(
                    "POST", "/jobs", json=body, headers=headers, timeout_s=timeout_s
                )
                break
            except DaemonUnavailable as exc:
                cause = exc.__cause__
                never_sent = isinstance(cause, httpx.ConnectError | httpx.ConnectTimeout)
                if never_sent and attempt == 1:
                    continue
                if never_sent:
                    raise
                raise SubmitUncertain(
                    f"sent the job to the daemon but got no answer ({exc.message}); "
                    "it may have been submitted",
                    hint="check `gpu jobs` before running it again; the daemon keeps "
                    "running anything it accepted",
                    detail={"idempotency_key": key, "maybe_submitted": True},
                ) from exc
        return JobView.model_validate(data)

    def jobs(
        self,
        *,
        states: list[JobState] | None = None,
        project_dir: str | None = None,
        limit: int = 50,
        before: float | str | None = None,
    ) -> JobList:
        """`before`: epoch seconds or an ISO-8601 timestamp (a previous `next_before`)."""
        params: dict[str, Any] = {"project_dir": project_dir, "limit": limit, "before": before}
        if states:
            params["state"] = [str(s) for s in states]
        return JobList.model_validate(self.request("GET", "/jobs", params=params))

    def job(self, ref: str) -> JobDetail:
        return JobDetail.model_validate(self.request("GET", f"/jobs/{_ref(ref)}"))

    def events(self, ref: str, *, after: int = 0) -> EventList:
        return EventList.model_validate(
            self.request("GET", f"/jobs/{_ref(ref)}/events", params={"after": after})
        )

    def logs(
        self,
        ref: str,
        *,
        attempt: int | None = None,
        offset: int = 0,
        follow: bool = False,
        protocol: bool = False,
    ) -> Iterator[LogRecord]:
        """Stream NDJSON LogRecords (no read timeout while following). Heartbeats are
        filtered out; the final eof record is yielded. `protocol=True` also returns the
        `::gpu::` lines (the shell parses them for metric history, D42)."""
        params = {
            k: v
            for k, v in {
                "attempt": attempt,
                "offset": offset,
                "follow": "true" if follow else "false",
                "protocol": "true" if protocol else None,
            }.items()
            if v is not None
        }
        timeout = httpx.Timeout(self.timeout_s, read=None if follow else self.timeout_s)
        try:
            with self._client().stream(
                "GET", f"{API_PREFIX}/jobs/{_ref(ref)}/logs", params=params, timeout=timeout
            ) as resp:
                if not resp.is_success:
                    resp.read()
                    self._raise_for(resp)
                for raw in resp.iter_lines():
                    if not raw.strip():
                        continue
                    rec = LogRecord.model_validate(jsonlib.loads(raw))
                    if rec.heartbeat:
                        continue
                    yield rec
        except httpx.TransportError as exc:
            if isinstance(exc, httpx.ConnectError | httpx.ConnectTimeout):
                raise self._transport_error(exc, None) from exc
            raise DaemonUnavailable(
                f"lost the connection to the daemon at {self.base_url} ({type(exc).__name__})",
                hint=BUSY_HINT,
            ) from exc

    def cancel(self, ref: str) -> JobView:
        return JobView.model_validate(self.request("POST", f"/jobs/{_ref(ref)}/cancel"))

    def approve(self, ref: str, *, reason: str | None = None) -> JobView:
        return JobView.model_validate(
            self.request("POST", f"/jobs/{_ref(ref)}/approve", json={"reason": reason})
        )

    def deny(self, ref: str, *, reason: str | None = None) -> JobView:
        return JobView.model_validate(
            self.request("POST", f"/jobs/{_ref(ref)}/deny", json={"reason": reason})
        )

    def fetch(self, ref: str) -> JobView:
        return JobView.model_validate(self.request("POST", f"/jobs/{_ref(ref)}/fetch"))

    def route(self, spec: JobSpec) -> RouteDecision:
        body = SubmitRequest(spec=spec).model_dump(mode="json")
        return RouteDecision.model_validate(self.request("POST", "/route", json=body))

    def providers(self) -> list[ProviderView]:
        return _PROVIDERS.validate_python(self.request("GET", "/providers"))

    def provider(self, name: str) -> ProviderView:
        return ProviderView.model_validate(self.request("GET", f"/providers/{_provider(name)}"))

    def healthcheck(self, name: str) -> ProviderView:
        return ProviderView.model_validate(
            self.request("POST", f"/providers/{_provider(name)}/healthcheck", timeout_s=120.0)
        )

    def quota(self, *, refresh: bool = False) -> list[QuotaSnapshot]:
        """Quota ledger views; `refresh=True` re-reads every live provider first."""
        params = {"refresh": "true"} if refresh else None
        return _QUOTAS.validate_python(
            self.request("GET", "/quota", params=params, timeout_s=120.0)
        )

    def policy(self) -> PolicyView:
        """Approval rules in force (phase 5)."""
        return PolicyView.model_validate(self.request("GET", "/policy"))

    def set_policy(self, policy: PolicyConfig) -> PolicyView:
        """Replace the approval rules; the daemon persists them to config.yaml."""
        return PolicyView.model_validate(
            self.request("PUT", "/policy", json=policy.model_dump(mode="json"))
        )

    def wait_events(self, *, after: int, timeout_s: float = 30.0) -> EventList:
        """Long poll GET /v1/events (HTTP timeout = timeout_s + 5)."""
        return EventList.model_validate(
            self.request(
                "GET",
                "/events",
                params={"after": after, "timeout": timeout_s},
                timeout_s=timeout_s + 5,
            )
        )

    def shutdown(self) -> None:
        self.request("POST", "/daemon/shutdown")


def _spawn_daemon(paths: Paths) -> None:
    from gpu_router.daemon import launchd

    if launchd.is_installed() and launchd.kickstart():
        return
    paths.logs_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    with open(paths.launchd_log, "ab") as log:
        subprocess.Popen(
            [sys.executable, "-m", "gpu_router", "daemon", "run"],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            start_new_session=True,
            close_fds=True,
        )


def ensure_daemon(paths: Paths, *, start: bool = True, wait_s: float = 10.0) -> GpuClient:
    """Return a client for a healthy daemon. If none is running and `start`, spawn
    `gpu daemon run` detached (launchd agent if installed, else subprocess with
    start_new_session=True and output to logs/launchd.log), then poll /v1/health until
    ready or wait_s elapses (DaemonUnavailable)."""

    def try_connect() -> GpuClient | None:
        try:
            client = GpuClient.from_env(paths, timeout_s=2.0)
        except DaemonUnavailable:
            return None
        try:
            if client.health().ready:
                client.timeout_s = DEFAULT_TIMEOUT_S
                client.close()
                return client
        except GpuRouterError:
            pass
        client.close()
        return None

    client = try_connect()
    if client is not None:
        return client
    if not start:
        raise DaemonUnavailable("the gpu-router daemon is not running", hint=START_HINT)
    _spawn_daemon(paths)
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        time.sleep(0.1)
        client = try_connect()
        if client is not None:
            return client
    raise DaemonUnavailable(
        f"started the daemon but it was not ready within {wait_s:g}s",
        hint=f"see {paths.launchd_log} and {paths.daemon_log}",
    )
