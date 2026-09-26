"""OpenAI-compatible chat client for the inference lane (phase 7b).

Every routable provider in the catalog (Groq, Cloudflare Workers AI, the Gemini API, HF
Inference Providers) speaks `POST <base_url>/chat/completions` with a bearer key, so one
client covers them; what differs is data (base URL, headers) plus two small rule sets here:

- `live_readings`: Groq's `x-ratelimit-{limit,remaining,reset}-requests` headers are the
  model's requests per day (their docs: "Always refers to Requests Per Day").
- `classify`: turns an error response into the inference error taxonomy. 401/403 (and a
  400 that says the API key is not valid, Gemini's answer) = key rejected, 402 = credits
  used up (HF), 404 / unknown model = not served here, 429 = per minute (cooldown for
  Retry-After) unless the body or headers say the day's allowance is gone (Groq
  remaining-requests 0 or "per day", Gemini `...PerDay...` quota ids,
  Cloudflare's daily neuron allocation), 5xx / network = unavailable.

Keys arrive as SecretStr and only ever go into the Authorization header; URLs with an
account id are never put into messages; provider error text is redacted and cut to 300
characters before it reaches an exception (invariant 12).
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import httpx

from gpu_router import __version__, secrets
from gpu_router.inference.errors import (
    InferAuthRequired,
    InferBadRequest,
    InferenceError,
    InferModelUnavailable,
    InferQuotaExhausted,
    InferRateLimited,
    InferUnavailable,
)
from gpu_router.inference.ledger import LiveReading, window_bounds

if TYPE_CHECKING:
    from pydantic import SecretStr

    from gpu_router.inference.catalog import InferenceEntry
    from gpu_router.inference.models import InferMessage

__all__ = ["ChatClient", "ChatReply", "classify", "live_readings", "parse_duration"]

DEFAULT_TIMEOUT_S = 120.0
MAX_READ_TIMEOUT_S = 600.0
#: output tokens per second assumed when sizing a completion's read timeout (slow free
#: tiers; a non-streamed reply arrives only when the whole generation is done)
ASSUMED_TOKENS_PER_S = 100.0


def read_timeout_for(max_tokens: int | None, base: float = DEFAULT_TIMEOUT_S) -> float:
    """A completion's read timeout: `base`, longer for a large `max_tokens` (review fix: a
    fixed 120 s timed out long generations the provider still finished and billed)."""
    scaled = 30.0 + (max_tokens or 0) / ASSUMED_TOKENS_PER_S
    return min(MAX_READ_TIMEOUT_S, max(base, scaled))


DEFAULT_COOLDOWN_S = 30.0
MAX_ERROR_TEXT = 300
_DURATION = re.compile(r"(\d+(?:\.\d+)?)(ms|h|m|s)")
_DAILY = re.compile(
    r"per ?day|perday|daily|tokens per day|requests per day|\bRPD\b|\bTPD\b|neurons",
    re.IGNORECASE,
)
#: an invalid or revoked key that the provider answers with HTTP 400, not 401/403: the
#: Gemini API's OpenAI endpoints say `400 INVALID_ARGUMENT "Please pass a valid API key"`
#: (seen live 2026-09-25 on /v1beta/openai/models and /chat/completions), its native API
#: "API key not valid" with reason API_KEY_INVALID
_BAD_KEY = re.compile(
    r"API key not valid|pass a valid API key|API_KEY_INVALID|invalid API key", re.IGNORECASE
)
_MODEL_MISSING = re.compile(
    r"model.*(not found|does not exist|not supported|unknown|decommissioned|no longer)"
    r"|(unknown|invalid|unsupported) model|no such model",
    re.IGNORECASE,
)


def parse_duration(raw: str | None) -> float | None:
    """Seconds from Go-style durations ("2m59.56s", "7.66s", "250ms", "1h2m") or a plain
    number of seconds; None when unparseable."""
    if raw is None:
        return None
    text = raw.strip()
    if not text:
        return None
    try:
        return max(0.0, float(text))
    except ValueError:
        pass
    total = 0.0
    matched = False
    for num, unit in _DURATION.findall(text):
        matched = True
        value = float(num)
        total += {"h": 3600.0, "m": 60.0, "s": 1.0, "ms": 0.001}[unit] * value
    return total if matched else None


def _retry_after(headers: httpx.Headers, body: str) -> float | None:
    found = parse_duration(headers.get("retry-after"))
    if found is not None:
        return found
    m = re.search(r'"retryDelay"\s*:\s*"([0-9.]+)s"', body)  # Gemini RetryInfo
    return float(m.group(1)) if m else None


def live_readings(
    entry: InferenceEntry, model_id: str, headers: httpx.Headers, now: float
) -> list[LiveReading]:
    """Quota facts a response's headers carry (only Groq-style headers are understood)."""
    if entry.rate_limit_headers != "groq":
        return []
    remaining = headers.get("x-ratelimit-remaining-requests")
    if remaining is None:
        return []
    try:
        left = float(remaining)
    except ValueError:
        return []
    limit_raw = headers.get("x-ratelimit-limit-requests")
    try:
        limit = float(limit_raw) if limit_raw is not None else None
    except ValueError:
        limit = None
    reset = parse_duration(headers.get("x-ratelimit-reset-requests"))
    scope = model_id if entry.quota_scope == "model" else ""
    return [
        LiveReading(
            scope=scope,
            unit="requests",
            limit=limit,
            remaining=max(0.0, left),
            resets_at=now + reset if reset is not None else None,
            observed_at=now,
        )
    ]


def _error_text(body: str) -> str:
    """The provider's own message (redacted, cut), from OpenAI / Google / CF shapes."""
    message = body
    try:
        doc: Any = json.loads(body)
    except ValueError:
        doc = None
    if isinstance(doc, list) and doc:
        doc = doc[0]
    if isinstance(doc, dict):
        err = doc.get("error")
        if isinstance(err, dict) and err.get("message"):
            message = str(err["message"])
        elif isinstance(err, str):
            message = err
        elif isinstance(doc.get("errors"), list) and doc["errors"]:
            first = doc["errors"][0]
            message = str(first.get("message") if isinstance(first, dict) else first)
        elif doc.get("message"):
            message = str(doc["message"])
    message = secrets.redact(" ".join(message.split()))
    return message[:MAX_ERROR_TEXT] + ("…" if len(message) > MAX_ERROR_TEXT else "")


def rejects_key(status: int, body: str) -> bool:
    """The response says the key itself is bad (401/403, or Gemini's 400 for a bad key)."""
    return status in (401, 403) or (status == 400 and bool(_BAD_KEY.search(body)))


def classify(
    entry: InferenceEntry,
    model_id: str,
    status: int,
    headers: httpx.Headers,
    body: str,
    now: float,
) -> InferenceError:
    """The inference error for an HTTP error response from `entry`."""
    name = entry.name
    text = _error_text(body)
    said = f": {text}" if text else ""
    _, window_end = window_bounds(entry.reset, now)
    login = f"run `gpu login {entry.login_name}`"
    if rejects_key(status, body):
        return InferAuthRequired(
            f"{name} rejected the key (HTTP {status}){said}", provider=name, hint=login
        )
    if status == 402:
        return InferQuotaExhausted(
            f"{name}: the free credits are used up{said}",
            provider=name,
            resets_at=window_end,
            hint="it refuses until the credits renew; gpu-router never buys any",
        )
    if status == 404 or (status in (400, 422) and _MODEL_MISSING.search(text)):
        return InferModelUnavailable(
            f"{name} does not serve {model_id}{said}", provider=name, detail={"model": model_id}
        )
    if status == 429:
        retry = _retry_after(headers, body)
        daily = bool(_DAILY.search(body)) or bool(
            entry.rate_limit_headers == "groq"
            and headers.get("x-ratelimit-remaining-requests") in ("0", "0.0")
        )
        if daily:
            scope = model_id if entry.quota_scope == "model" else ""
            reset = parse_duration(headers.get("x-ratelimit-reset-requests"))
            until = now + reset if reset else (now + retry if retry and retry > 120 else None)
            what = f"{model_id} on {name}" if scope else name
            return InferQuotaExhausted(
                f"{what}: today's free allowance is used up{said}",
                provider=name,
                resets_at=until or window_end,
                scope=scope,
            )
        return InferRateLimited(
            f"{name} is rate limiting (HTTP 429){said}",
            provider=name,
            retry_after=retry if retry is not None else DEFAULT_COOLDOWN_S,
        )
    if status in (400, 413, 422):
        return InferBadRequest(f"{name} refused the request (HTTP {status}){said}", provider=name)
    return InferUnavailable(f"{name} answered HTTP {status}{said}", provider=name, status=status)


@dataclass(frozen=True)
class ChatReply:
    text: str
    finish_reason: str | None
    input_tokens: int | None
    output_tokens: int | None
    latency_s: float
    live: list[LiveReading] = field(default_factory=list)


def _int(v: Any) -> int | None:
    return int(v) if isinstance(v, int | float) and not isinstance(v, bool) else None


def _content(message: Any) -> str:
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # content parts
        return "".join(
            str(p.get("text", "")) for p in content if isinstance(p, dict) and p.get("text")
        )
    return ""


class ChatClient:
    """Blocking client for one provider (runs in the service's worker threads). The
    httpx.Client is shared across calls and threads; `close()` releases it."""

    def __init__(
        self,
        entry: InferenceEntry,
        *,
        transport: httpx.BaseTransport | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        self.entry = entry
        self._http = httpx.Client(
            transport=transport,
            timeout=httpx.Timeout(timeout_s, connect=10.0),
            headers={"User-Agent": f"gpu-router/{__version__}"},
            follow_redirects=False,
        )

    def close(self) -> None:
        self._http.close()

    def _url(self, template: str, keys: dict[str, SecretStr], path: str = "") -> str:
        fill = {role: v.get_secret_value() for role, v in keys.items() if role != "api_key"}
        try:
            base = template.format(**fill)
        except (KeyError, IndexError):
            raise InferAuthRequired(
                f"{self.entry.name} needs more than a key (its URL has a placeholder)",
                provider=self.entry.name,
                hint=f"run `gpu login {self.entry.login_name}`",
            ) from None
        return base.rstrip("/") + path

    def _send(
        self,
        method: str,
        url: str,
        keys: dict[str, SecretStr],
        body: dict[str, Any] | None,
        *,
        timeout_s: float | None = None,
    ) -> httpx.Response:
        headers = {"Authorization": f"Bearer {keys['api_key'].get_secret_value()}"}
        extra: dict[str, Any] = {}
        if timeout_s is not None:
            extra["timeout"] = httpx.Timeout(timeout_s, connect=min(10.0, timeout_s))
        try:
            return self._http.request(method, url, json=body, headers=headers, **extra)
        except httpx.ConnectTimeout:
            raise InferUnavailable(
                f"{self.entry.name} did not answer in time", provider=self.entry.name
            ) from None
        except httpx.TimeoutException:
            # the request went out: the provider may still generate (and bill) the reply
            raise InferUnavailable(
                f"{self.entry.name} did not answer in time", provider=self.entry.name, sent=True
            ) from None
        except httpx.HTTPError as exc:
            raise InferUnavailable(
                f"cannot reach {self.entry.name} ({type(exc).__name__})", provider=self.entry.name
            ) from None

    def chat(
        self,
        keys: dict[str, SecretStr],
        model_id: str,
        messages: list[InferMessage],
        *,
        max_tokens: int | None,
        temperature: float | None,
        now: float,
        timeout_s: float | None = None,
    ) -> ChatReply:
        """One chat completion. Raises only InferenceError subclasses. `timeout_s`: this
        call's read timeout (the service passes one sized to max_tokens and its deadline)."""
        if not self.entry.base_url:
            raise InferUnavailable(f"{self.entry.name} has no API URL", provider=self.entry.name)
        body: dict[str, Any] = {
            "model": model_id,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
        }
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if temperature is not None:
            body["temperature"] = temperature
        url = self._url(self.entry.base_url, keys, "/chat/completions")
        started = time.monotonic()
        response = self._send("POST", url, keys, body, timeout_s=timeout_s)
        latency = time.monotonic() - started
        live = live_readings(self.entry, model_id, response.headers, now)
        if response.status_code >= 400:
            err = classify(
                self.entry, model_id, response.status_code, response.headers, response.text, now
            )
            err.live = list(live)  # the service records these
            raise err
        try:
            doc = response.json()
        except ValueError:
            raise InferUnavailable(
                f"{self.entry.name} answered with something that is not JSON",
                provider=self.entry.name,
            ) from None
        choices = doc.get("choices") if isinstance(doc, dict) else None
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise InferUnavailable(
                f"{self.entry.name} answered without a completion", provider=self.entry.name
            )
        first = choices[0]
        usage = doc.get("usage") if isinstance(doc.get("usage"), dict) else {}
        return ChatReply(
            text=_content(first.get("message")),
            finish_reason=str(first["finish_reason"]) if first.get("finish_reason") else None,
            input_tokens=_int(usage.get("prompt_tokens")),
            output_tokens=_int(usage.get("completion_tokens")),
            latency_s=latency,
            live=live,
        )

    def verify(self, keys: dict[str, SecretStr]) -> None:
        """Check the key with the entry's free `verify_url` (a model list or a token check,
        never a completion). Raises InferAuthRequired when rejected, InferUnavailable when
        the provider cannot be reached."""
        if not self.entry.verify_url:
            return
        url = self._url(self.entry.verify_url, keys)
        response = self._send("GET", url, keys, None)
        if rejects_key(response.status_code, response.text):
            raise InferAuthRequired(
                f"{self.entry.name} rejected the key (HTTP {response.status_code})",
                provider=self.entry.name,
                hint=f"create a new key: {self.entry.link or 'see the provider settings'}",
            )
        if response.status_code == 404 and "{account_id}" in self.entry.verify_url:
            raise InferAuthRequired(
                f"{self.entry.name} does not know that account id (HTTP 404)",
                provider=self.entry.name,
                hint="copy the account id from the Cloudflare dashboard (Workers AI page)",
            )
        if response.status_code >= 400:
            raise InferUnavailable(
                f"{self.entry.name} answered HTTP {response.status_code} to the key check",
                provider=self.entry.name,
            )
