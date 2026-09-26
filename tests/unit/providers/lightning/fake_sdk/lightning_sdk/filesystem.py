"""Fake lightning_sdk.filesystem.Filesystem.rm."""

from __future__ import annotations

from lightning_sdk import _state


class Filesystem:
    def rm(self, path: str, recursive: bool = False) -> None:
        _state.call("Filesystem.rm", path=path, recursive=recursive)
        rel = path.split("/", 4)[-1]
        drive = _state.STATE.get("drive") or {}
        if rel not in drive:
            raise FileNotFoundError(path)
        drive.pop(rel)
        _state.save()
