"""Registry entry point for the Kaggle adapter (catalog kind "kaggle").

`adapters.registry.ADAPTER_KINDS["kaggle"]` names `gpu_router.adapters.kaggle:KaggleAdapter`;
the implementation lives with its notes, CLI wrapper and remote runner in `providers/kaggle/`.
"""

from __future__ import annotations

from gpu_router.providers.kaggle.adapter import KaggleAdapter

__all__ = ["KaggleAdapter"]
