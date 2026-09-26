"""Provider catalog: providers.yaml as data (phase 1; models real, loader: group B).

Limits change often, so they live here as data, not prose (spec "Docs, in three layers").
The packaged file `gpu_router/providers/providers.yaml` is loaded with importlib.resources
and merged with the optional user file `<home>/providers.yaml`: user keys override packaged
keys per provider (deep merge of mappings, lists replaced). `/doctor` (phase 8) compares
real limits against these numbers and flags drift.

File shape:

    catalog_version: 1
    providers:
      kaggle:
        kind: kaggle              # which Adapter class (adapters.registry.ADAPTER_KINDS)
        display_name: Kaggle
        priority: 20              # phase-1 router: lower = tried first
        enabled_by_default: true
        test_only: false          # true = registered only when config.test_mode
        card_required: false      # must stay false (invariant 19)
        gpus: [{name: P100, vram_gb: 16, count: 1}, {name: T4, vram_gb: 16, count: 2}]
        session_hours: 12
        max_concurrency: 1
        poll_interval_s: 60
        quota: {unit: gpu_hours, limit: 30, reset: weekly, reset_anchor: "sat 00:00 UTC"}
        verified_at: null         # date a human last checked these numbers
        docs: [https://...]
        # phase 7b (additive):
        status: active            # active | verify_at_signup (listed, never registered)
        link: https://...         # where a human signs up / opens it
        note: one line shown next to the entry in `gpu providers`
        verify_at_signup: [what to check before an adapter is built]
    excluded:                     # never used; a name here may not be under providers:
      modal: {display_name: Modal, reason: needs a card, quote: "...", source: https://...,
              decided: 2026-09-24}
    inference: {...}              # the inference lane (gpu_router.inference.catalog)

Listing lanes (`ProviderEntry.lane`): `gpu` (registered when enabled and an adapter
exists), `manual` (manual_only: shown with a link, no adapter), `verify` (verify_at_signup:
shown with the checklist). Only the `gpu` lane ever reaches the registry.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from gpu_router.models import QuotaUnit

CATALOG_VERSION = 1
PACKAGED_RESOURCE = "providers.yaml"


class _M(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class GpuOffer(_M):
    name: str  # "T4", "P100", "L4", "A100-40GB", "MPS"
    vram_gb: float = Field(gt=0)
    count: int = Field(default=1, ge=1)

    @property
    def label(self) -> str:
        """'2xT4' or 'T4'."""
        return f"{self.count}x{self.name}" if self.count > 1 else self.name


class QuotaSpec(_M):
    unit: QuotaUnit = QuotaUnit.GPU_HOURS
    limit: float | None = None  # None = unknown / dynamic (Colab) / unlimited (local)
    reset: Literal["weekly", "monthly", "daily", "unknown", "none"] = "unknown"
    reset_anchor: str | None = None  # human-readable, parsed by the phase-5 ledger


ProviderStatus = Literal["active", "verify_at_signup"]


class ProviderEntry(_M):
    name: str
    kind: str
    display_name: str
    priority: int = 100
    enabled_by_default: bool = True
    test_only: bool = False
    card_required: bool = False
    manual_only: bool = False  # shown in UI, no adapter (SageMaker Studio Lab)
    gpus: tuple[GpuOffer, ...] = ()
    session_hours: float | None = None
    max_concurrency: int = Field(default=1, ge=1)
    poll_interval_s: float = Field(default=30, gt=0)
    quota: QuotaSpec = Field(default_factory=QuotaSpec)
    verified_at: date | None = None
    docs: tuple[str, ...] = ()
    options: dict[str, Any] = Field(default_factory=dict)  # adapter-specific, non-secret
    # phase 7b: listing facts for entries gpu-router shows but never routes to
    status: ProviderStatus = "active"
    link: str | None = None
    note: str | None = None
    verify_at_signup: tuple[str, ...] = ()

    @field_validator("card_required")
    @classmethod
    def _free_only(cls, v: bool) -> bool:
        if v:
            raise ValueError("providers that need a card are not allowed (invariant 19)")
        return v

    @property
    def lane(self) -> Literal["gpu", "manual", "verify"]:
        """How gpu-router treats the entry: only `gpu` entries can ever be registered."""
        if self.manual_only:
            return "manual"
        if self.status != "active":
            return "verify"
        return "gpu"

    @property
    def max_vram_gb(self) -> float:
        return max((g.vram_gb for g in self.gpus), default=0.0)

    @property
    def session_cap_s(self) -> float | None:
        return None if self.session_hours is None else self.session_hours * 3600


class ExcludedEntry(_M):
    """A service gpu-router must never use (spec "Excluded", phase-7 decisions)."""

    name: str
    display_name: str
    reason: str  # "needs a card", "paid only", ...
    quote: str | None = None  # the provider's own words, when a decision rests on them
    source: str  # URL (or docs/spec.md) the reason comes from
    decided: date
    note: str | None = None


class Catalog(_M):
    catalog_version: int = CATALOG_VERSION
    providers: dict[str, ProviderEntry]
    excluded: dict[str, ExcludedEntry] = Field(default_factory=dict)
    # the inference lane's raw entries, validated by gpu_router.inference.catalog so a bad
    # inference entry never stops the daemon from routing GPU jobs
    inference: dict[str, dict[str, Any]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _never_excluded(self) -> Catalog:
        clash = sorted(set(self.providers) & set(self.excluded))
        if clash:
            ex = self.excluded[clash[0]]
            raise ValueError(
                f"{clash[0]} is excluded ({ex.reason}; {ex.source}) and may not be listed "
                "under providers"
            )
        return self

    def get(self, name: str) -> ProviderEntry:
        """Raises ProviderNotFound."""
        try:
            return self.providers[name]
        except KeyError:
            from gpu_router.errors import ProviderNotFound

            raise ProviderNotFound(
                f"unknown provider {name!r}",
                hint=f"known providers: {', '.join(sorted(self.providers))}",
            ) from None

    def ordered(self) -> list[ProviderEntry]:
        """Entries by (priority, name)."""
        return sorted(self.providers.values(), key=lambda e: (e.priority, e.name))

    def listed(self, lane: str) -> list[ProviderEntry]:
        """Entries of one lane (`manual`, `verify`, `gpu`), test-only ones left out."""
        return [e for e in self.ordered() if e.lane == lane and not e.test_only]


def load_catalog(user_file: Path | None = None) -> Catalog:
    """Load the packaged catalog, deep-merge `user_file` (if it exists) on top, validate.

    Each provider mapping gets `name` filled from its key. Raises ConfigError naming the file
    and the pydantic error summary on invalid input, or if catalog_version is newer than
    CATALOG_VERSION.
    """
    from importlib import resources

    packaged = resources.files("gpu_router.providers").joinpath(PACKAGED_RESOURCE)
    raw = _parse_yaml(packaged.read_text(encoding="utf-8"), f"packaged {PACKAGED_RESOURCE}")
    source = f"packaged {PACKAGED_RESOURCE}"
    if user_file is not None and user_file.is_file():
        try:
            text = user_file.read_text(encoding="utf-8")
        except OSError as exc:
            from gpu_router.errors import ConfigError

            raise ConfigError(
                f"cannot read {user_file}: {exc.strerror or exc}",
                hint="fix the file permissions or remove it",
            ) from None
        override = _parse_yaml(text, str(user_file))
        raw = merge_raw(raw, override)
        source = f"{user_file} (merged over the packaged catalog)"
    return _validate(raw, source)


def merge_raw(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge: mappings merge recursively, everything else (lists included) replaces."""
    out: dict[str, Any] = dict(base)
    for key, value in override.items():
        current = out.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            out[key] = merge_raw(current, value)
        else:
            out[key] = value
    return out


def _parse_yaml(text: str, source: str) -> dict[str, Any]:
    import yaml

    from gpu_router.errors import ConfigError

    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(
            f"{source} is not valid YAML: {exc}", hint="fix the syntax or remove the file"
        ) from None
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(
            f"{source} must be a mapping at the top level", hint="start the file with `providers:`"
        )
    return data


def _validate(raw: dict[str, Any], source: str) -> Catalog:
    from pydantic import ValidationError

    from gpu_router.errors import ConfigError

    version = raw.get("catalog_version", CATALOG_VERSION)
    if not isinstance(version, int) or isinstance(version, bool):
        raise ConfigError(f"{source}: catalog_version must be an integer")
    if version > CATALOG_VERSION:
        raise ConfigError(
            f"{source} has catalog_version {version}; this gpu-router understands up to "
            f"{CATALOG_VERSION}",
            hint="upgrade gpu-router or lower catalog_version in your providers.yaml",
        )
    providers_raw = raw.get("providers") or {}
    if not isinstance(providers_raw, dict):
        raise ConfigError(f"{source}: `providers` must be a mapping of name -> settings")
    providers: dict[str, Any] = {}
    for name, body in providers_raw.items():
        if not isinstance(body, dict):
            raise ConfigError(f"{source}: provider {name!r} must be a mapping")
        providers[str(name)] = {**body, "name": str(name)}
    excluded_raw = raw.get("excluded") or {}
    if not isinstance(excluded_raw, dict):
        raise ConfigError(f"{source}: `excluded` must be a mapping of name -> reason")
    excluded: dict[str, Any] = {}
    for name, body in excluded_raw.items():
        if not isinstance(body, dict):
            raise ConfigError(f"{source}: excluded entry {name!r} must be a mapping")
        excluded[str(name)] = {**body, "name": str(name)}
    for name in sorted(set(providers) & set(excluded)):
        body = excluded[name]
        raise ConfigError(
            f"{source}: {name} is excluded ({body.get('reason', 'see the excluded list')}; "
            f"{body.get('source', 'providers.yaml')}), so it may not be listed under providers",
            hint=f"remove providers.{name} from your providers.yaml",
        )
    inference_raw = raw.get("inference") or {}
    if not isinstance(inference_raw, dict):
        raise ConfigError(f"{source}: `inference` must be a mapping of name -> settings")
    doc = {
        **raw,
        "catalog_version": version,
        "providers": providers,
        "excluded": excluded,
        "inference": inference_raw,
    }
    try:
        return Catalog.model_validate(doc)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()
        )
        raise ConfigError(
            f"{source} is invalid: {problems}", hint="fix the listed keys in providers.yaml"
        ) from None
