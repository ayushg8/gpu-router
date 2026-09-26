"""Registry entry point for the Colab adapter (catalog kind "colab").

`adapters.registry.ADAPTER_KINDS["colab"]` names `gpu_router.adapters.colab:ColabAdapter`;
the implementation lives with its notes and remote scripts in `providers/colab/`.
"""

from __future__ import annotations

from gpu_router.providers.colab.adapter import ColabAdapter

__all__ = ["ColabAdapter"]
