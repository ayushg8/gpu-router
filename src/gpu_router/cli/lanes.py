"""Phase-7b listings for `gpu providers` / `gpu quota` and the shell's /providers, /quota:
GPU entries gpu-router shows but never routes to (manual only, verify at signup), the
excluded services with their reason, and the inference lane's providers and quota.

The not-routed lanes come from the catalog (packaged providers.yaml + the user's override,
the same files the daemon reads; the shell's /doctor reads them the same way): they hold no
state. Inference views come from the daemon (`/v1/infer/*`), which owns the ledger.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from rich.table import Table
from rich.text import Text

from gpu_router.cli import render

if TYPE_CHECKING:
    from gpu_router.inference.models import InferProviderView, InferQuotaView
    from gpu_router.paths import Paths
    from gpu_router.providers.catalog import Catalog, ProviderEntry

__all__ = [
    "inference_notes",
    "inference_providers_table",
    "inference_quota_table",
    "load_listing_catalog",
    "not_routed",
    "not_routed_renderables",
]


def load_listing_catalog(paths: Paths | None = None) -> Catalog | None:
    """The catalog the daemon uses, or None when the user's providers.yaml is broken (the
    daemon reports that itself)."""
    from gpu_router.errors import GpuRouterError
    from gpu_router.paths import Paths as _Paths
    from gpu_router.providers.catalog import load_catalog

    p = paths or _Paths.from_env()
    try:
        return load_catalog(p.user_providers)
    except GpuRouterError:
        return None


def _entry_doc(e: ProviderEntry) -> dict[str, Any]:
    doc: dict[str, Any] = {
        "name": e.name,
        "display_name": e.display_name,
        "gpus": [g.label for g in e.gpus],
        "session_hours": e.session_hours,
        "quota": {"unit": str(e.quota.unit), "limit": e.quota.limit, "reset": e.quota.reset},
        "link": e.link,
        "note": e.note,
        "verified_at": e.verified_at.isoformat() if e.verified_at else None,
    }
    if e.verify_at_signup:
        doc["verify_at_signup"] = list(e.verify_at_signup)
    return doc


def not_routed(catalog: Catalog | None) -> dict[str, list[dict[str, Any]]]:
    """`gpu providers --json` key `not_routed`: manual, verify_at_signup and excluded."""
    if catalog is None:
        return {"manual": [], "verify_at_signup": [], "excluded": []}
    return {
        "manual": [_entry_doc(e) for e in catalog.listed("manual")],
        "verify_at_signup": [_entry_doc(e) for e in catalog.listed("verify")],
        "excluded": [
            {
                "name": x.name,
                "display_name": x.display_name,
                "reason": x.reason,
                "quote": x.quote,
                "source": x.source,
                "decided": x.decided.isoformat(),
                "note": x.note,
            }
            for x in catalog.excluded.values()
        ],
    }


def _limit_text(e: ProviderEntry) -> str:
    q = e.quota
    if q.limit is None:
        return "limit unknown"
    per = {"daily": "/day", "weekly": "/week", "monthly": "/month"}.get(q.reset, "")
    unit = "h" if str(q.unit) == "gpu_hours" else f" {q.unit}"
    return f"{render.fmt_num(q.limit)}{unit}{per}"


def not_routed_renderables(catalog: Catalog | None) -> list[Any]:
    """Rich renderables, one short block per entry (lines wrap at any width instead of a
    squeezed table): manual + verify-at-signup under "not routed", then "excluded"."""
    if catalog is None:
        return []
    out: list[Any] = []
    manual = catalog.listed("manual")
    verify = catalog.listed("verify")
    if manual or verify:
        out.extend([Text(""), Text("not routed (listed only)", style="dim")])
        for e, tag in [(m, "manual") for m in manual] + [(v, "verify at signup") for v in verify]:
            head = Text("  ")
            head.append(e.name)
            head.append(f"  {tag}", style="yellow")
            facts = [", ".join(g.label for g in e.gpus) or "GPU unknown", _limit_text(e)]
            if e.session_hours:
                facts.append(f"{render.fmt_num(e.session_hours)}h sessions")
            if e.link:
                facts.append(e.link)
            head.append("  " + " · ".join(facts), style="dim")
            out.append(head)
            if e.note:
                out.append(Text(f"    {e.note}", style="dim"))
            if e.verify_at_signup:
                out.append(
                    Text("    check at signup: " + "; ".join(e.verify_at_signup), style="dim")
                )
    if catalog.excluded:
        out.extend([Text(""), Text("excluded (never used)", style="dim")])
        quoted = [x for x in catalog.excluded.values() if x.quote]
        for ex in quoted:
            line = Text("  ")
            line.append(ex.name)
            line.append(f"  {ex.reason}: ", style="dim")
            line.append(f'"{ex.quote}"', style="dim italic")
            line.append(f" ({ex.source}, decided {ex.decided.isoformat()})", style="dim")
            out.append(line)
            if ex.note:
                out.append(Text(f"    {ex.note}", style="dim"))
        rest: dict[str, list[str]] = {}
        for ex in catalog.excluded.values():
            if not ex.quote:
                rest.setdefault(ex.reason, []).append(ex.name)
        if rest:
            also = "; ".join(f"{', '.join(names)} ({why})" for why, names in rest.items())
            out.append(Text(f"  also: {also}", style="dim"))
    return out


def _models_text(models: dict[str, str], shown: int = 3) -> str:
    names = list(models)
    if len(names) <= shown + 1:
        return ", ".join(names) or "-"
    return ", ".join(names[:shown]) + f", +{len(names) - shown} more"


def inference_providers_table(views: Sequence[InferProviderView]) -> Table:
    """One row per inference provider: key stored?, reset, a few model aliases."""
    t = Table(box=None, pad_edge=False, show_edge=False, header_style="dim")
    for col in ("inference", "key", "resets", "models"):
        t.add_column(col, no_wrap=col != "models", overflow="fold")
    for v in views:
        if v.problem and not v.routable and not v.models:
            key = Text(f"{render.ICON_FAILED} broken", style="red")
        elif not v.routable:
            key = Text("listed only", style="dim")
        elif v.logged_in:
            key = Text(f"{render.ICON_DONE} key", style="green")
        else:
            key = Text(f"gpu login {v.login}", style="yellow")
        reset = {
            "daily_utc": "00:00 UTC",
            "daily_pacific": "00:00 PT",
            "monthly": "monthly",
            "rolling_24h": "24h after use",
        }.get(v.reset, v.reset)
        t.add_row(
            v.name if v.routable else Text(v.name, style="dim"),
            key,
            reset,
            Text(_models_text(v.models), style="dim"),
        )
    return t


def inference_notes(views: Sequence[InferProviderView]) -> list[Any]:
    """Each provider's note (and a broken entry's problem) as a wrapped dim line."""
    out: list[Any] = []
    for v in views:
        text = v.problem or v.note
        if text:
            line = Text(f"  {v.name}: ", style="dim")
            line.append(text, style="dim")
            out.append(line)
    return out


def inference_quota_table(views: Sequence[InferQuotaView]) -> Table:
    t = Table(box=None, pad_edge=False, show_edge=False, header_style="dim")
    t.add_column("inference", no_wrap=True)
    t.add_column("today", no_wrap=True)
    t.add_column("left / resets", overflow="fold")
    for v in views:
        left = v.summary.split(": ", 1)[-1]
        t.add_row(v.provider, f"{v.requests_today} req", Text(left, style="dim"))
    return t


def _safe_inference(client: Any, what: str) -> list[Any] | None:
    """Inference views from the daemon, or None when it has no inference lane (an older
    daemon still running) or the call fails: the GPU listing must still print."""
    from gpu_router.errors import GpuRouterError
    from gpu_router.inference import remote

    try:
        return remote.providers(client) if what == "providers" else remote.quota(client)
    except (GpuRouterError, ValueError):
        return None


def provider_extras(client: Any, paths: Paths | None = None) -> tuple[dict[str, Any], list[Any]]:
    """(additive `gpu providers --json` keys, renderables) for the phase-7b lanes."""
    catalog = load_listing_catalog(paths)
    views = _safe_inference(client, "providers")
    doc: dict[str, Any] = {"not_routed": not_routed(catalog)}
    if views is not None:
        doc["inference"] = [v.model_dump(mode="json") for v in views]
    shown: list[Any] = list(not_routed_renderables(catalog))
    if views:
        shown.extend([Text(""), inference_providers_table(views)])
    return doc, shown


def quota_extras(client: Any) -> tuple[dict[str, Any], list[Any]]:
    """(additive `gpu quota --json` key `inference`, renderables)."""
    views = _safe_inference(client, "quota")
    if views is None:
        return {}, []
    shown = [Text(""), inference_quota_table(views)] if views else []
    return {"inference": [v.model_dump(mode="json") for v in views]}, shown
