"""Inference lane API models (phase 7b), shared by the daemon (`/v1/infer*`), the CLI,
the shell and the MCP tool. Additive-only like the rest of API v1 (invariant 17).

    POST /v1/infer            InferRequest -> InferResult
    POST /v1/infer/route      InferRequest -> InferRoute (dry run: no call, no quota spent)
    GET  /v1/infer/quota      -> list[InferQuotaView]
    GET  /v1/infer/providers  -> list[InferProviderView]

Prompts and replies are never logged or stored by the daemon; only provider, model,
token counts and timings reach the ledger and the daemon log.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "MAX_PROMPT_CHARS",
    "InferAttempt",
    "InferCandidate",
    "InferCounter",
    "InferMessage",
    "InferProviderView",
    "InferQuotaView",
    "InferRejection",
    "InferRequest",
    "InferResult",
    "InferRoute",
    "InferUsage",
]

#: one request's prompt (all messages together); a free tier would refuse more anyway
MAX_PROMPT_CHARS = 400_000
MAX_WAIT_S = 60.0


class _Api(BaseModel):
    model_config = ConfigDict(extra="ignore")


class InferMessage(_Api):
    role: Literal["system", "user", "assistant"]
    content: str


class InferRequest(_Api):
    """One chat completion. Give `prompt` (+ optional `system`) or `messages`."""

    model: str = Field(min_length=1, max_length=200)
    prompt: str | None = None
    system: str | None = None
    messages: list[InferMessage] | None = Field(default=None, max_length=200)
    provider: str | None = Field(default=None, max_length=32)
    max_tokens: int | None = Field(default=None, ge=1, le=65_536)
    temperature: float | None = Field(default=None, ge=0, le=2)
    #: wait at most this long for a provider in a short cooldown (per-minute limits)
    wait_s: float = Field(default=0.0, ge=0, le=MAX_WAIT_S)
    #: the daemon stops trying providers after this many seconds (clients send one a
    #: little under their own HTTP timeout); None = the service default
    deadline_s: float | None = Field(default=None, gt=0, le=3600)

    @model_validator(mode="after")
    def _one_prompt(self) -> InferRequest:
        if (self.prompt is None) == (self.messages is None):
            raise ValueError("give either prompt or messages")
        if self.messages is not None and not self.messages:
            raise ValueError("messages is empty")
        if self.prompt is not None and not self.prompt.strip():
            raise ValueError("prompt is empty")
        if sum(len(m.content) for m in self.chat()) > MAX_PROMPT_CHARS:
            raise ValueError(f"the prompt is over {MAX_PROMPT_CHARS} characters")
        return self

    def chat(self) -> list[InferMessage]:
        """The messages actually sent (system first when given with `prompt`)."""
        if self.messages is not None:
            return list(self.messages)
        out: list[InferMessage] = []
        if self.system:
            out.append(InferMessage(role="system", content=self.system))
        out.append(InferMessage(role="user", content=self.prompt or ""))
        return out


class InferCandidate(_Api):
    provider: str
    model_id: str
    score: float
    reason: str  # one line: "groq: 870/1000 requests left today (live), resets in 6h"
    left: str | None = None  # the binding counter, e.g. "870/1000 requests (live)"
    unlisted: bool = False


class InferRejection(_Api):
    provider: str
    code: Literal[
        "no_model", "no_key", "disabled", "cooldown", "exhausted", "quota", "not_routable"
    ]
    reason: str
    retry_at: float | None = None


class InferRoute(_Api):
    outcome: Literal["place", "wait", "no_fit"]
    model: str  # as asked
    chosen: InferCandidate | None = None
    candidates: list[InferCandidate] = Field(default_factory=list)
    rejected: list[InferRejection] = Field(default_factory=list)
    reason: str
    retry_at: float | None = None


class InferUsage(_Api):
    input_tokens: int | None = None
    output_tokens: int | None = None
    neurons: float | None = None  # Cloudflare, computed from tokens (est)
    usd: float | None = None  # HF credits, estimated from tokens


class InferAttempt(_Api):
    """A candidate that failed before the one that answered (fallback trail)."""

    provider: str
    model_id: str
    code: str
    message: str


class InferCounter(_Api):
    """One quota counter: `scope` "" = the provider's shared allowance, else a model id."""

    scope: str = ""
    unit: str  # requests | tokens | neurons | usd | gpu_seconds
    used: float
    limit: float | None = None
    remaining: float | None = None
    source: Literal["live", "estimate"] = "estimate"
    resets_at: float | None = None


class InferQuotaView(_Api):
    provider: str
    display_name: str
    window: str  # daily_utc | daily_pacific | monthly | rolling_24h
    window_resets_at: float | None = None
    requests_today: int = 0  # requests in the current window (all models)
    counters: list[InferCounter] = Field(default_factory=list)
    blocked: dict[str, str] = Field(default_factory=dict)  # scope -> "exhausted until ..."
    blocked_until: float | None = None  # provider-wide block, if any
    summary: str  # one line: "groq: 870/1000 requests left today (live) · resets in 6h"


class InferProviderView(_Api):
    name: str
    display_name: str
    routable: bool  # has a chat client (False: listed only, e.g. ZeroGPU Spaces)
    logged_in: bool
    login: str  # `gpu login <login>`
    missing: list[str] = Field(default_factory=list)  # Keychain names not set
    reset: str
    limits: dict[str, float | None] = Field(default_factory=dict)
    models: dict[str, str] = Field(default_factory=dict)  # alias -> provider model id
    note: str | None = None
    link: str | None = None
    docs: list[str] = Field(default_factory=list)
    verified_at: str | None = None
    problem: str | None = None  # a broken catalog entry


class InferResult(_Api):
    provider: str
    model: str  # as asked
    model_id: str  # what the provider ran
    text: str  # model output: untrusted data
    finish_reason: str | None = None
    usage: InferUsage = Field(default_factory=InferUsage)
    latency_s: float
    route_reason: str
    fallbacks: list[InferAttempt] = Field(default_factory=list)
    quota: str | None = None  # one line after this call: "groq: 869/1000 requests left ..."
