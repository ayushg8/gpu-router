"""Checkpoint handoff and data movement (phase 5).

- `storage`: typed facade over gpu_router/runner/storage.py (the layout, shared with the
  runner): HF Storage Buckets or a local directory.
- `hub`: CheckpointHub, the daemon service the engine uses (per-attempt storage env,
  checkpoint copies between backends, planned-handoff requests, datasets).
- `data`: dataset content hashing.
- `tokens`: Hugging Face tokens from the Keychain.
- `sidechannel`: near-live logs for providers without them (Kaggle) from log-tail.json.

The daemon runtime registers its hub with `set_active_hub` so adapters, which only get
AdapterDeps, can read the side channel (`active_hub()`); tests set or clear it.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gpu_router.checkpoint.hub import CheckpointHub

__all__ = ["active_hub", "set_active_hub"]

_lock = threading.Lock()
_active: CheckpointHub | None = None


def set_active_hub(hub: CheckpointHub | None) -> None:
    global _active
    with _lock:
        _active = hub


def active_hub() -> CheckpointHub | None:
    with _lock:
        return _active
