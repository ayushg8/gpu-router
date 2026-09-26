"""Inference catalog: the `inference:` section of providers.yaml, typed (phase 7b).

The GPU catalog (`providers/catalog.py`) keeps this section as raw mappings so a bad
inference entry never stops the daemon from routing GPU jobs; this module validates each
entry on its own and reports the broken ones as problems instead of failing.

Entry shape (see the packaged providers.yaml for the real ones):

    groq:
      display_name: Groq
      priority: 10                 # tie-break, lower first
      client: openai               # openai = chat completions at base_url; none = listed only
      base_url: https://api.groq.com/openai/v1      # may use {placeholders} filled from secrets
      secrets: {api_key: INFER_GROQ_API_KEY}        # role -> Keychain name (values never here)
      verify_url: https://api.groq.com/openai/v1/models   # `gpu login` checks the key with it
      login: groq                  # `gpu login <login>` (default: the entry's name)
      reset: daily_utc             # daily_utc | daily_pacific | monthly | rolling_24h
      limits: {neurons: 10000}     # provider-wide, per window; null = unknown
      rate_limit_headers: groq     # x-ratelimit-*-requests are requests per day (live)
      quota_scope: model           # a 429 "per day" blocks that model (else the provider)
      models:
        gpt-oss-20b: {id: openai/gpt-oss-20b, rpm: 30, limits: {requests: 1000}}
      passthrough: "^@cf/"         # provider-native ids accepted when not listed
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

if TYPE_CHECKING:
    from gpu_router.providers.catalog import Catalog

__all__ = [
    "UNITS",
    "InferenceCatalog",
    "InferenceEntry",
    "ModelEntry",
    "Reset",
    "Unit",
    "load_inference_catalog",
]

Unit = Literal["requests", "tokens", "neurons", "usd", "gpu_seconds"]
Reset = Literal["daily_utc", "daily_pacific", "monthly", "rolling_24h"]
UNITS: tuple[Unit, ...] = ("requests", "tokens", "neurons", "usd", "gpu_seconds")

_NAME = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")


class _M(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ModelEntry(_M):
    """One model a provider serves. `alias` is gpu-router's name (the same alias on several
    providers lets the router choose); `id` is what the provider's API expects."""

    alias: str
    id: str
    rpm: int | None = Field(default=None, ge=1)
    limits: dict[Unit, float | None] = Field(default_factory=dict)  # per window, this model
    neurons_per_mtok: tuple[float, float] | None = None  # [input, output] (Cloudflare)
    usd_per_mtok: float | None = Field(default=None, ge=0)
    unlisted: bool = False  # a passthrough id, not in the catalog


class InferenceEntry(_M):
    name: str
    display_name: str
    priority: int = 100
    client: Literal["openai", "none"] = "openai"
    base_url: str | None = None
    secrets: dict[str, str] = Field(default_factory=dict)
    verify_url: str | None = None
    login: str | None = None
    reset: Reset = "daily_utc"
    limits: dict[Unit, float | None] = Field(default_factory=dict)
    rate_limit_headers: Literal["groq", "none"] = "none"
    #: where "the day is used up" applies: per model (groq, gemini) or the whole provider
    quota_scope: Literal["model", "provider"] = "provider"
    models: dict[str, ModelEntry] = Field(default_factory=dict)
    passthrough: str | None = None
    default_neurons_per_mtok: tuple[float, float] | None = None
    usd_per_mtok: float | None = Field(default=None, ge=0)
    enabled: bool = True
    verified_at: date | None = None
    docs: tuple[str, ...] = ()
    link: str | None = None
    note: str | None = None

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        if not _NAME.match(v):
            raise ValueError("names are lowercase letters, digits, - or _ (max 32)")
        return v

    @field_validator("passthrough")
    @classmethod
    def _regex(cls, v: str | None) -> str | None:
        if v is not None:
            re.compile(v)
        return v

    @property
    def routable(self) -> bool:
        """An entry the router may send requests to (a chat client + a URL + enabled)."""
        return self.enabled and self.client == "openai" and bool(self.base_url)

    @property
    def login_name(self) -> str:
        return self.login or self.name

    def resolve(self, model: str) -> ModelEntry | None:
        """The model entry for `model`: an alias, then a listed provider id, then a
        passthrough id (unlisted, marked so); None when this provider does not serve it."""
        found = self.models.get(model)
        if found is not None:
            return found
        for entry in self.models.values():
            if entry.id == model:
                return entry
        if self.passthrough is not None and re.match(self.passthrough, model):
            return ModelEntry(alias=model, id=model, unlisted=True)
        return None

    def limit(self, model: ModelEntry, unit: Unit) -> tuple[str, float] | None:
        """(scope, limit) for `unit`: the model's own limit when it has one, else the
        provider-wide one; None when neither is known. Scope "" = provider-wide."""
        own = model.limits.get(unit)
        if own is not None:
            return model.id, own
        shared = self.limits.get(unit)
        if shared is not None:
            return "", shared
        return None

    def neurons(self, model: ModelEntry, input_tokens: int, output_tokens: int) -> float | None:
        rates = model.neurons_per_mtok or self.default_neurons_per_mtok
        if rates is None:
            return None
        return (input_tokens * rates[0] + output_tokens * rates[1]) / 1_000_000

    def usd(self, model: ModelEntry, input_tokens: int, output_tokens: int) -> float | None:
        rate = model.usd_per_mtok if model.usd_per_mtok is not None else self.usd_per_mtok
        if rate is None:
            return None
        return (input_tokens + output_tokens) * rate / 1_000_000


@dataclass(frozen=True)
class InferenceCatalog:
    entries: dict[str, InferenceEntry]
    problems: dict[str, str] = field(default_factory=dict)  # name -> why it was skipped

    def ordered(self) -> list[InferenceEntry]:
        return sorted(self.entries.values(), key=lambda e: (e.priority, e.name))

    def get(self, name: str) -> InferenceEntry | None:
        return self.entries.get(name)

    def aliases(self) -> list[str]:
        """Every model alias any routable provider lists, sorted."""
        return sorted({a for e in self.entries.values() if e.routable for a in e.models})


def _model_doc(alias: str, body: Any) -> dict[str, Any]:
    if isinstance(body, str):
        return {"alias": alias, "id": body}
    if not isinstance(body, dict):
        raise ValueError(f"models.{alias} must be a mapping or a model id")
    return {**body, "alias": alias}


def parse_entry(name: str, raw: Any) -> InferenceEntry:
    """Validate one raw entry. Raises ValueError with a one-line summary."""
    if not isinstance(raw, dict):
        raise ValueError("must be a mapping")
    models_raw = raw.get("models") or {}
    if not isinstance(models_raw, dict):
        raise ValueError("models must be a mapping of alias -> model")
    doc = {
        **raw,
        "name": name,
        "models": {str(a): _model_doc(str(a), b) for a, b in models_raw.items()},
    }
    try:
        entry = InferenceEntry.model_validate(doc)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()[:3]
        )
        raise ValueError(problems) from None
    if entry.client == "openai" and "api_key" not in entry.secrets:
        raise ValueError("secrets.api_key names the Keychain item holding the key")
    return entry


def load_inference_catalog(catalog: Catalog) -> InferenceCatalog:
    """Typed entries from `catalog.inference`; broken ones land in `problems`."""
    entries: dict[str, InferenceEntry] = {}
    problems: dict[str, str] = {}
    for name, raw in catalog.inference.items():
        try:
            entries[str(name)] = parse_entry(str(name), raw)
        except (ValueError, re.error) as exc:
            problems[str(name)] = str(exc)
    return InferenceCatalog(entries=entries, problems=problems)
