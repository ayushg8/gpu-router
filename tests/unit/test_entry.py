"""entry.py: stdlib-only dispatch (invariant 14)."""

from __future__ import annotations

import subprocess
import sys

import pytest

from gpu_router import entry


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "gpu_router", *args],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def test_import_is_stdlib_only() -> None:
    code = (
        "import sys, gpu_router.entry\n"
        "heavy = {'pydantic', 'fastapi', 'httpx', 'typer', 'textual', 'uvicorn', 'yaml'}\n"
        "print(sorted(m for m in heavy if m in sys.modules))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert out == "[]"


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["status", "--line"], True),
        (["status", "x", "--line"], True),
        (["status"], False),
        (["--line"], False),
        (["logs", "--line"], False),
    ],
)
def test_status_line_detection(argv: list[str], expected: bool) -> None:
    assert entry._is_status_line(argv) is expected


def test_status_line_never_fails() -> None:
    proc = _run("status", "--line")
    assert proc.returncode == 0
    assert proc.stderr == ""


def test_daemon_dispatch() -> None:
    proc = _run("daemon", "bogus")
    assert proc.returncode == 2
    assert "gpu daemon" in proc.stderr


def test_main_dispatches_daemon(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[list[str]] = []
    import gpu_router.daemon.__main__ as daemon_main

    monkeypatch.setattr(daemon_main, "main", lambda argv: seen.append(argv) or 0)
    monkeypatch.setattr(sys, "argv", ["gpu", "daemon", "status"])
    with pytest.raises(SystemExit) as info:
        entry.main()
    assert info.value.code == 0
    assert seen == [["status"]]
