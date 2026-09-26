"""Registry entry point for the Lightning AI adapter (catalog kind "lightning").

`adapters.registry.ADAPTER_KINDS["lightning"]` names
`gpu_router.adapters.lightning:LightningAdapter`; the implementation lives with its notes,
SDK bridge and in-job launcher in `providers/lightning/`.
"""

from __future__ import annotations

from gpu_router.providers.lightning.adapter import LightningAdapter

__all__ = ["LightningAdapter"]
