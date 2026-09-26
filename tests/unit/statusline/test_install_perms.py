"""Phase 7-8 integration: the status-line record dirs are private, so `gpu doctor`'s
data-dir check (any subdirectory other users can open is flagged) stays green after
`gpu statusline install` (found by the phase-8b wizard run: install, then doctor warned)."""

from __future__ import annotations

import os
from pathlib import Path

from gpu_router.statusline import install
from tests.unit.statusline.test_install import _install, _no_ask, env

__all__ = ["env"]  # the fixture, re-used


def test_install_creates_the_record_dirs_0700(env: dict[str, Path]) -> None:
    env["home"].mkdir(parents=True, mode=0o700)
    old = os.umask(0o022)
    try:
        code, _ = _install(env, yes=True, interactive=False, ask=_no_ask)
    finally:
        os.umask(old)
    assert code == 0
    record = install.record_dir(env["home"], env["settings"])
    for d in (record, install.statusline_dir(env["home"])):
        assert d.stat().st_mode & 0o777 == 0o700, d
    assert (record / install.WRAPPER_NAME).stat().st_mode & 0o777 == 0o755


def test_a_loose_statusline_dir_from_an_older_install_is_tightened(
    env: dict[str, Path],
) -> None:
    root = install.statusline_dir(env["home"])
    root.mkdir(parents=True)
    root.chmod(0o755)
    above = env["home"].parent  # "Application Support": never ours to change
    above.chmod(0o755)
    code, _ = _install(env, yes=True, interactive=False, ask=_no_ask)
    assert code == 0
    assert root.stat().st_mode & 0o777 == 0o700
    assert above.stat().st_mode & 0o777 == 0o755
