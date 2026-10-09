"""Bundle preview for `gpu run --dry-run` (phase 2).

The daemon builds the real bundle at submit (Supervisor.submit -> EngineDeps.bundler), so
`gpu run` never packages anything itself. A dry run calls the packaging layer's public
`build_bundle(project_dir, spec)` locally to show what would ship (files, size, detected
deps, VRAM/runtime estimate, warnings). The archive lands in the content-addressed bundle
cache, so the real submit that follows reuses it.

Packaging failures raise BundleError (an InvalidSpec: exit code 2) with a hint. If the
packaging layer is missing, `preview` returns None and the dry run shows only the route.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from gpu_router.models import JobSpec


def preview(spec: JobSpec) -> dict[str, Any] | None:
    """JSON-safe summary of the bundle `spec` would ship, or None without packaging.

    Shape (docs/cli.md): {sha256, file_count, size_bytes, code_bytes, cached, deps: {kind,
    file, packages, python_requires}, estimate: {vram_gb, hours, vram_source, hours_source,
    mode, reasons}, warnings: [str]}, plus (D60) `included` {files, bytes} when the spec has
    `include:`, and `left_out` [str] + `left_out_not_shown` + `hint` when ignored paths do
    not ship.
    """
    try:
        mod = importlib.import_module("gpu_router.packaging")
    except ImportError:
        return None
    build = getattr(mod, "build_bundle", None)
    if build is None:
        return None
    bundle = build(spec.project_dir, spec)
    deps = bundle.deps
    est = bundle.estimate
    out: dict[str, Any] = {
        "sha256": bundle.sha256,
        "file_count": bundle.file_count,
        "size_bytes": bundle.size_bytes,
        "code_bytes": bundle.code_bytes,
        "cached": bool(getattr(bundle, "cached", False)),
        "deps": {
            "kind": deps.kind,
            "file": deps.file,
            "packages": list(deps.packages),
            "python_requires": deps.python_requires,
        },
        "estimate": {
            "vram_gb": est.vram_gb,
            "hours": est.hours,
            "vram_source": est.vram_source,
            "hours_source": est.hours_source,
            "mode": est.mode,
            "reasons": list(est.reasons),
        },
        "warnings": list(bundle.warnings),  # already includes deps warnings
    }
    sel = getattr(bundle, "selection", None)
    if sel is not None:
        from gpu_router.packaging.bundle import arg_warnings, left_out_view

        # arguments naming paths the job will not find (not in the manifest's warnings)
        out["warnings"][:0] = arg_warnings(spec, sel, Path(spec.project_dir).resolve())
        if spec.include:
            out["included"] = {"files": len(sel.included), "bytes": sel.included_bytes}
        data_paths = [d.path for d in spec.data if d.path]
        out.update(left_out_view(spec.project_dir, sel, spec.include, data_paths))
    return out
