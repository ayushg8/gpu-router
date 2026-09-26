"""`gpu doctor --update-catalog`: write live limits into <home>/providers.yaml (phase 8a).

Only the keys doctor can measure are written (`quota.limit`, `quota.reset_anchor`,
`quota.unit`), as overrides in the user file (the packaged catalog is never touched; the
loader deep-merges the user file over it). The new file is validated by loading it the
way the daemon does before anything is written; the write is atomic, mode 0600. YAML
comments in an existing user file are not kept (the diff shows everything that changes).
The daemon reads providers.yaml at start, so it needs a restart to use the new numbers.
"""

from __future__ import annotations

import contextlib
import difflib
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gpu_router.doctor.model import DriftItem
from gpu_router.errors import ConfigError

__all__ = ["AUTO_KEYS", "CatalogPlan", "apply_plan", "plan_update"]

AUTO_KEYS = frozenset({"quota.limit", "quota.reset_anchor", "quota.unit"})


class CatalogPlan:
    def __init__(self, path: Path, before: str, after: str, items: list[DriftItem]) -> None:
        self.path = path
        self.before = before
        self.after = after
        self.items = items

    @property
    def changed(self) -> bool:
        return self.before != self.after

    def diff(self) -> str:
        return "".join(
            difflib.unified_diff(
                self.before.splitlines(keepends=True),
                self.after.splitlines(keepends=True),
                fromfile=f"{self.path} (now)",
                tofile=f"{self.path} (with live limits)",
            )
        )


def plan_update(user_file: Path, drift: list[DriftItem], *, now: float) -> CatalogPlan:
    """The new user providers.yaml text with each auto-fixable drift item applied.
    Raises ConfigError when the current file is not valid YAML or the result would not
    load."""
    import yaml

    from gpu_router.providers.catalog import load_catalog

    try:
        before = user_file.read_text(encoding="utf-8")
    except FileNotFoundError:
        before = ""
    except OSError as exc:
        raise ConfigError(f"cannot read {user_file}: {exc.strerror or exc}") from None
    try:
        raw: Any = yaml.safe_load(before) if before.strip() else {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"{user_file} is not valid YAML: {exc}") from None
    data: dict[str, Any] = dict(raw) if isinstance(raw, dict) else {}
    items = [d for d in drift if d.key in AUTO_KEYS]
    providers = dict(data.get("providers") or {})
    for item in items:
        body = dict(providers.get(item.provider) or {})
        section, _, key = item.key.partition(".")
        sub = dict(body.get(section) or {})
        sub[key] = item.live
        body[section] = sub
        providers[item.provider] = body
    if items:
        data["providers"] = providers
        if "catalog_version" not in data:
            data = {"catalog_version": 1, **data}
    stamp = datetime.fromtimestamp(now, tz=UTC).strftime("%Y-%m-%d")
    header = (
        f"# gpu-router provider overrides (merged over the packaged catalog).\n"
        f"# live limits written by `gpu doctor --update-catalog` on {stamp}.\n"
    )
    body_text = yaml.safe_dump(data, sort_keys=False, default_flow_style=False) if data else ""
    after = header + body_text if items else before
    if items:
        tmp = user_file.with_name(f".{user_file.name}.check{os.getpid()}")
        try:
            tmp.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            tmp.write_text(after, encoding="utf-8")
            load_catalog(tmp)  # raises ConfigError when the result would not load
        finally:
            with contextlib.suppress(OSError):
                tmp.unlink()
    return CatalogPlan(user_file, before, after, items)


def apply_plan(plan: CatalogPlan) -> None:
    """Atomic write, mode 0600."""
    target = plan.path
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.tmp{os.getpid()}")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(plan.after)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise
