"""Phase-7b endpoints: the inference lane (mounted at /v1 by app.py next to routes.py).

    GET  /infer/providers  -> list[InferProviderView]  (keys present? limits, models)
    GET  /infer/quota      -> list[InferQuotaView]     (the daily-quota ledger, live or est)
    POST /infer/route      InferRequest -> InferRoute  (dry run: no call, nothing spent)
    POST /infer            InferRequest -> InferResult

All work runs in a worker thread (HTTP calls, Keychain reads, the ledger file): the event
loop never blocks and the Store is never touched (invariant 9). POST /infer runs on the
service's own executor (`InferenceService.executor`, INFER_WORKERS threads). Errors use the usual
envelope (inference/errors.py). Additive to API v1 (invariant 17).
"""

from __future__ import annotations

import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends, Request

from gpu_router.errors import NotReady
from gpu_router.inference.models import (
    InferProviderView,
    InferQuotaView,
    InferRequest,
    InferResult,
    InferRoute,
)
from gpu_router.inference.service import InferenceService


def _service(request: Request) -> InferenceService:
    service: InferenceService | None = getattr(request.app.state.runtime, "inference", None)
    if service is None:
        raise NotReady(
            "the inference lane is not available in this daemon",
            hint="restart the daemon (`gpu daemon stop`, then any gpu command)",
        )
    return service


Service = Annotated[InferenceService, Depends(_service)]


def build_lanes_router() -> APIRouter:
    router = APIRouter()

    @router.get("/infer/providers", response_model=list[InferProviderView])
    async def infer_providers(service: Service) -> list[InferProviderView]:
        return await asyncio.to_thread(service.providers)

    @router.get("/infer/quota", response_model=list[InferQuotaView])
    async def infer_quota(service: Service) -> list[InferQuotaView]:
        return await asyncio.to_thread(service.quota)

    @router.post("/infer/route", response_model=InferRoute)
    async def infer_route(service: Service, body: InferRequest) -> InferRoute:
        return await asyncio.to_thread(service.route, body)

    @router.post("/infer", response_model=InferResult)
    async def infer(service: Service, body: InferRequest) -> InferResult:
        # its own bounded executor: requests parked on a provider slot never hold the
        # default executor that submits, bundling and the other endpoints share
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(service.executor, service.infer, body)

    return router


__all__ = ["build_lanes_router"]
