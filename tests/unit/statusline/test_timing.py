"""`gpu status --line` wall time through the real console script (interpreter start
included): the spec's budget is 50 ms. p95 gets headroom for a loaded CI box / parallel
test runs; p50 must be inside the budget."""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import pytest

from gpu_router.statusline import samples

GPU = Path(sys.executable).with_name("gpu")
RUNS = 25


@pytest.mark.skipif(not GPU.is_file(), reason="no gpu console script in this venv")
@pytest.mark.parametrize("key", ["several", "idle"])
def test_under_50ms(tmp_path: Path, key: str) -> None:
    sample = next(s for s in samples.SAMPLES if s.key == key)
    (tmp_path / "state.json").write_text(json.dumps(dict(sample.snapshot, daemon_pid=1)))
    cmd = [str(GPU), "status", "--line", "--stdin", "--now", str(samples.NOW)]
    env = {"GPU_ROUTER_HOME": str(tmp_path), "PATH": "/usr/bin:/bin"}
    payload = b'{"workspace": {"current_dir": "/tmp"}}'
    first = subprocess.run(cmd, input=payload, capture_output=True, env=env, check=True)
    assert (first.stdout.count(b"\n") == 2) is (key == "several")
    wall = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        subprocess.run(cmd, input=payload, capture_output=True, env=env, check=True)
        wall.append((time.perf_counter() - t0) * 1000)
    wall.sort()
    p50 = statistics.median(wall)
    p95 = wall[int(len(wall) * 0.95) - 1]
    assert p50 < 50, f"p50 {p50:.1f} ms, p95 {p95:.1f} ms"
    assert p95 < 120, f"p50 {p50:.1f} ms, p95 {p95:.1f} ms"
