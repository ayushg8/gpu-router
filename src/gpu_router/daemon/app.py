"""FastAPI app factory (phase 1; owner: group C).

`create_app(runtime)` returns a FastAPI app with:
- middleware applying auth.check_request (invariant 18) and adding the
  api.VERSION_HEADER to every response;
- exception handlers: GpuRouterError -> its http_status + api.ErrorEnvelope(err.to_body());
  RequestValidationError -> 400 invalid_request (invalid_spec when the failing location is
  under body.spec) with pydantic's error list in detail; any other exception -> 500
  internal, message "internal error; see the daemon log", logged as `api.bug`;
- NotReady (503) for mutating routes while runtime.ready is False;
- the routers from routes.py mounted under api.API_PREFIX;
- no docs/openapi routes (docs_url=None, redoc_url=None, openapi_url=None).
The runtime is stored on app.state.runtime; routes read it through a dependency.

The guard is a plain ASGI middleware (not BaseHTTPMiddleware) so NDJSON log streams pass
through untouched.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from gpu_router import __version__
from gpu_router.api import API_PREFIX, VERSION_HEADER
from gpu_router.engine._obs import emit
from gpu_router.errors import GpuRouterError, InvalidRequest, NotReady

if TYPE_CHECKING:
    from fastapi import FastAPI
    from starlette.types import ASGIApp, Message, Receive, Scope, Send

    from gpu_router.daemon.runtime import DaemonRuntime

_logger = logging.getLogger("gpu_router.daemon.app")

MUTATING_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
#: Mutations allowed while recovery runs.
READY_EXEMPT: frozenset[str] = frozenset({f"{API_PREFIX}/daemon/shutdown"})


def error_body(exc: GpuRouterError) -> bytes:
    return json.dumps({"error": exc.to_body()}, default=str).encode("utf-8")


async def _send_error(send: Send, exc: GpuRouterError) -> None:
    body = error_body(exc)
    await send(
        {
            "type": "http.response.start",
            "status": exc.http_status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
                (VERSION_HEADER.lower().encode(), __version__.encode()),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


class GuardMiddleware:
    """Loopback/auth guard (invariant 18), not-ready gate and version header."""

    def __init__(self, app: ASGIApp, runtime: DaemonRuntime) -> None:
        self.app = app
        self.runtime = runtime

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        from gpu_router.daemon.auth import check_request

        headers = {
            k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])
        }
        method = str(scope.get("method", "GET")).upper()
        path = str(scope.get("path", ""))
        try:
            check_request(
                method=method,
                path=path,
                headers=headers,
                token=self.runtime.token,
                port=self.runtime.port,
            )
            if method in MUTATING_METHODS and path not in READY_EXEMPT and not self.runtime.ready:
                raise NotReady("the daemon is still recovering jobs", hint="retry in a few seconds")
        except GpuRouterError as exc:
            await _send_error(send, exc)
            return

        version = (VERSION_HEADER.lower().encode(), __version__.encode())

        async def send_with_version(message: Message) -> None:
            if message["type"] == "http.response.start":
                message = dict(message)
                message["headers"] = [*message.get("headers", []), version]
            await send(message)

        await self.app(scope, receive, send_with_version)


def _jsonable_errors(errors: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for err in errors:
        item = {
            "loc": [str(p) if not isinstance(p, int) else p for p in err.get("loc", ())],
            "msg": str(err.get("msg", "")),
            "type": str(err.get("type", "")),
        }
        out.append(item)
    return out


def create_app(runtime: DaemonRuntime) -> FastAPI:
    from fastapi import FastAPI, Request
    from fastapi.exceptions import RequestValidationError
    from fastapi.responses import JSONResponse
    from starlette.exceptions import HTTPException as StarletteHTTPException

    from gpu_router.daemon.routes import build_router

    app = FastAPI(
        title="gpu-router", version=__version__, docs_url=None, redoc_url=None, openapi_url=None
    )
    app.state.runtime = runtime

    def envelope(exc: GpuRouterError) -> JSONResponse:
        return JSONResponse({"error": exc.to_body()}, status_code=exc.http_status)

    @app.exception_handler(GpuRouterError)
    async def _domain_error(_request: Request, exc: GpuRouterError) -> JSONResponse:
        return envelope(exc)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_request: Request, exc: RequestValidationError) -> JSONResponse:
        errors = _jsonable_errors(exc.errors())
        under_spec = any(
            e["loc"][:2] == ["body", "spec"] and not (len(e["loc"]) == 2 and e["type"] == "missing")
            for e in errors
        )
        first = (
            next(
                (e for e in errors if e["loc"][:2] == ["body", "spec"] and len(e["loc"]) > 2), None
            )
            if under_spec
            else None
        )
        first = first or (errors[0] if errors else {"loc": [], "msg": "invalid request"})
        where = ".".join(str(p) for p in first["loc"] if p != "body") or "request"
        from gpu_router.errors import InvalidSpec

        err: GpuRouterError
        if under_spec:
            where = ".".join(str(p) for p in first["loc"][2:]) or "spec"
            err = InvalidSpec(
                f"invalid job spec: {where}: {first['msg']}", detail={"errors": errors}
            )
        else:
            err = InvalidRequest(
                f"invalid request: {where}: {first['msg']}", detail={"errors": errors}
            )
        return envelope(err)

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_request: Request, exc: StarletteHTTPException) -> JSONResponse:
        err = InvalidRequest(
            str(exc.detail) if exc.detail else "request failed", detail={"status": exc.status_code}
        )
        return JSONResponse({"error": err.to_body()}, status_code=exc.status_code)

    @app.exception_handler(Exception)
    async def _bug(request: Request, exc: Exception) -> JSONResponse:
        emit(
            "api.bug",
            f"{request.method} {request.url.path}: {type(exc).__name__}: {exc}",
            level=logging.ERROR,
            exc_info=True,
            log=_logger,
        )
        err = GpuRouterError(
            "internal error; see the daemon log", detail={"error": type(exc).__name__}
        )
        return envelope(err)

    app.include_router(build_router(), prefix=API_PREFIX)
    from gpu_router.daemon.lanes import build_lanes_router  # phase 7b: the inference lane

    app.include_router(build_lanes_router(), prefix=API_PREFIX)
    app.add_middleware(GuardMiddleware, runtime=runtime)
    return app
