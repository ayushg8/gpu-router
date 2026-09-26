"""Registry entry point for the Local Mac adapter (catalog kind "local").

`adapters.registry.ADAPTER_KINDS["local"]` names `gpu_router.adapters.local:LocalAdapter`;
the implementation lives with its launcher and notes in `providers/local/`.
"""

from __future__ import annotations

from gpu_router.providers.local.adapter import LocalAdapter

__all__ = ["LocalAdapter"]
