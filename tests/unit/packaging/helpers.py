"""Shared helpers for packaging and runner tests: throwaway projects, with or without git."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


def isolate_git(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """No user/system git config (global excludes file, templates) leaks into a test."""
    empty = tmp_path / "gitconfig-empty"
    empty.write_text("")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(empty))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.delenv("GIT_DIR", raising=False)
    monkeypatch.delenv("GIT_WORK_TREE", raising=False)


def git(project: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(project), *args], check=True, capture_output=True)


def write_files(root: Path, files: dict[str, str | bytes]) -> None:
    for rel, content in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content)


def make_project(
    root: Path,
    files: dict[str, str | bytes],
    *,
    use_git: bool = True,
    track: bool = True,
) -> Path:
    """Create a project dir. With git, files are staged (`git add -A`) when `track`."""
    root.mkdir(parents=True, exist_ok=True)
    write_files(root, files)
    if use_git:
        git(root, "init", "-q")
        if track:
            git(root, "add", "-A")
    return root
