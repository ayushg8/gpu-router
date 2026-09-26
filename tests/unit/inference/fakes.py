"""Mocked inference providers for phase-7b tests: one httpx.MockTransport that answers
like Groq, Cloudflare Workers AI, the Gemini API's OpenAI endpoint and HF's router, by
host. Nothing here reaches the network (invariant 20)."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import httpx

from gpu_router import secrets
from gpu_router.clock import FakeClock
from gpu_router.inference.catalog import InferenceCatalog, load_inference_catalog
from gpu_router.inference.service import InferenceService
from gpu_router.providers.catalog import load_catalog

HOSTS = {
    "api.groq.com": "groq",
    "api.cloudflare.com": "cloudflare",
    "generativelanguage.googleapis.com": "gemini",
    "router.huggingface.co": "hf",
}
KEYS = {
    "INFER_GROQ_API_KEY": "gsk_testkey0123456789abcdefghij",
    "INFER_GEMINI_API_KEY": "AIzaTESTKEY_0123456789abcdefghijklmnopq",
    "INFER_CLOUDFLARE_API_TOKEN": "cf-test-token-0123456789abcdefghijkl",
    "INFER_CLOUDFLARE_ACCOUNT_ID": "acct0123456789abcdef",
    "HF_TOKEN": "hf_testtoken0123456789abcdefghijklmnop",
}
PROVIDER_KEYS = {
    "groq": ["INFER_GROQ_API_KEY"],
    "gemini": ["INFER_GEMINI_API_KEY"],
    "cloudflare": ["INFER_CLOUDFLARE_API_TOKEN", "INFER_CLOUDFLARE_ACCOUNT_ID"],
    "hf": ["HF_TOKEN"],
}


def login(*providers: str) -> None:
    """Put test keys for `providers` in the (in-memory) keyring."""
    for p in providers:
        for name in PROVIDER_KEYS[p]:
            secrets.set_secret(name, KEYS[name])


def completion(text: str = "4", *, model: str = "m", prompt: int = 12, out: int = 3) -> dict:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": prompt, "completion_tokens": out, "total_tokens": prompt + out},
    }


def groq_headers(remaining: int, limit: int = 1000, reset: str = "1m26.4s") -> dict[str, str]:
    return {
        "x-ratelimit-limit-requests": str(limit),
        "x-ratelimit-remaining-requests": str(remaining),
        "x-ratelimit-reset-requests": reset,
        "x-ratelimit-limit-tokens": "8000",
        "x-ratelimit-remaining-tokens": "7900",
        "x-ratelimit-reset-tokens": "7.66s",
    }


@dataclass
class Seen:
    provider: str
    method: str
    url: str
    auth: str | None
    body: dict[str, Any] | None


@dataclass
class FakeProviders:
    """Scripted answers per provider; unscripted requests get a normal completion."""

    script: dict[str, list[Any]] = field(default_factory=lambda: defaultdict(list))
    seen: list[Seen] = field(default_factory=list)
    groq_left: int = 1000

    def queue(self, provider: str, *answers: Any) -> None:
        self.script[provider].extend(answers)

    def calls(self, provider: str | None = None) -> list[Seen]:
        return [s for s in self.seen if provider is None or s.provider == provider]

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self._handle)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        provider = HOSTS.get(request.url.host, "?")
        body = json.loads(request.content) if request.content else None
        self.seen.append(
            Seen(
                provider,
                request.method,
                str(request.url),
                request.headers.get("authorization"),
                body,
            )
        )
        queued = self.script.get(provider)
        if queued:
            answer = queued.pop(0)
            if isinstance(answer, BaseException):
                raise answer
            if callable(answer):
                return answer(request)
            return answer
        if request.method == "GET":
            return httpx.Response(200, json={"data": [], "success": True})
        model = (body or {}).get("model", "m")
        headers: dict[str, str] = {}
        if provider == "groq":
            self.groq_left -= 1
            headers = groq_headers(self.groq_left)
        return httpx.Response(200, json=completion(model=model), headers=headers)


def inference_catalog(overrides: dict[str, Any] | None = None) -> InferenceCatalog:
    cat = load_catalog()
    raw = dict(cat.inference)
    for name, patch in (overrides or {}).items():
        raw[name] = {**raw.get(name, {}), **patch}
    return load_inference_catalog(cat.model_copy(update={"inference": raw}))


def service(
    fake: FakeProviders,
    clock: FakeClock,
    *,
    catalog: InferenceCatalog | None = None,
    ledger_file: Any = None,
    test_mode: bool = True,
    sleeps: list[float] | None = None,
) -> InferenceService:
    def sleep(s: float) -> None:
        if sleeps is not None:
            sleeps.append(s)
        clock.set(clock.now() + s)

    return InferenceService(
        catalog or inference_catalog(),
        clock,
        ledger_file=ledger_file,
        test_mode=test_mode,
        transport=fake.transport(),
        sleep=sleep,
    )
