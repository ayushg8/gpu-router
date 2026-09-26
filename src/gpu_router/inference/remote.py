"""Client side of the inference lane (phase 7b): thin calls over `GpuClient.request`, used
by the CLI (`gpu infer`), the shell (/infer) and the MCP tool (`gpu_infer`). The daemon
owns the ledger and the provider calls (invariant 2); nothing here talks to a provider.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gpu_router.inference.models import (
    InferProviderView,
    InferQuotaView,
    InferRequest,
    InferResult,
    InferRoute,
)

if TYPE_CHECKING:
    from gpu_router.client import GpuClient

__all__ = ["INFER_TIMEOUT_S", "infer", "providers", "quota", "route"]

#: a call may try several providers (120 s read timeout each) and wait up to 60 s
INFER_TIMEOUT_S = 420.0
#: the daemon gives up this much before the client does, so it never keeps sending the
#: prompt to providers after the client has stopped listening
DEADLINE_MARGIN_S = 20.0


def _body(req: InferRequest) -> dict[str, object]:
    return req.model_dump(mode="json", exclude_none=True)


def infer(client: GpuClient, req: InferRequest) -> InferResult:
    if req.deadline_s is None:
        req = req.model_copy(update={"deadline_s": INFER_TIMEOUT_S - DEADLINE_MARGIN_S})
    raw = client.request("POST", "/infer", json=_body(req), timeout_s=INFER_TIMEOUT_S)
    return InferResult.model_validate(raw)


def route(client: GpuClient, req: InferRequest) -> InferRoute:
    raw = client.request("POST", "/infer/route", json=_body(req), timeout_s=30.0)
    return InferRoute.model_validate(raw)


def quota(client: GpuClient) -> list[InferQuotaView]:
    raw = client.request("GET", "/infer/quota", timeout_s=30.0)
    return [InferQuotaView.model_validate(x) for x in raw or []]


def providers(client: GpuClient) -> list[InferProviderView]:
    raw = client.request("GET", "/infer/providers", timeout_s=30.0)
    return [InferProviderView.model_validate(x) for x in raw or []]
