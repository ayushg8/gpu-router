from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from gpu_router.paths import DEFAULT_HOME, Paths, home_from_env, state_file_fast


def test_home_override_and_default(tmp_path: Path) -> None:
    assert home_from_env({"GPU_ROUTER_HOME": str(tmp_path)}) == str(tmp_path)
    assert home_from_env({"GPU_ROUTER_HOME": "~/x"}).endswith("/x")
    assert home_from_env({}) == str(DEFAULT_HOME)
    assert state_file_fast({"GPU_ROUTER_HOME": str(tmp_path)}) == str(tmp_path / "state.json")


def test_layout(gpu_home: Path) -> None:
    p = Paths.from_env()
    assert p.home == gpu_home.resolve()
    assert p.db.name == "gpu.db"
    assert p.lock.name == "daemon.lock"
    assert p.job_log("abc", 2) == p.home / "jobs" / "abc" / "logs" / "attempt-2.log"
    assert p.job_bundle_archive("abc").name == "bundle.tar.gz"
    assert p.provider_dir("kaggle") == p.home / "providers" / "kaggle"
    assert p.fake_dir == p.home / "fake"


def test_ensure_creates_private_dirs(gpu_home: Path) -> None:
    p = Paths.from_env()
    p.ensure()
    p.ensure()  # idempotent
    assert p.home.stat().st_mode & 0o777 == 0o700
    for d in (p.logs_dir, p.jobs_dir, p.providers_dir):
        assert d.is_dir()


def test_stdlib_only_imports() -> None:
    """Invariant 14: paths and statefile import no third-party modules at runtime."""
    code = (
        "import sys; import gpu_router.paths, gpu_router.statefile; "
        "bad = [m for m in ('pydantic', 'yaml', 'keyring', 'fastapi', 'httpx') "
        "if m in sys.modules]; print(','.join(bad))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert out == ""
