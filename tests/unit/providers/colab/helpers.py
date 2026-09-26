"""Shared helpers for the colab adapter tests and the colab contract targets.

`ColabSim` wraps `fake_colab.py` (a stand-in for the `colab` CLI): the adapter under test is
built with the provider settings `cli=[python, fake_colab.py]` and `remote_root=<tmp VM>`,
so every code path (session file, exec scripts, uploads, the real bootstrap runner, stop)
runs for real against the local filesystem.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from gpu_router.adapters.base import AdapterDeps, AttemptContext
from gpu_router.clock import Clock
from gpu_router.config import ProviderSettings
from gpu_router.models import DepsSpec, JobSpec
from gpu_router.packaging.bundle import build_bundle
from gpu_router.paths import Paths
from gpu_router.providers.catalog import GpuOffer, ProviderEntry, QuotaSpec
from gpu_router.providers.colab.adapter import ColabAdapter

SIM = Path(__file__).with_name("fake_colab.py")

COLAB_ENTRY = ProviderEntry(
    name="colab",
    kind="colab",
    display_name="Google Colab",
    priority=10,
    gpus=(GpuOffer(name="T4", vram_gb=16),),
    session_hours=12,
    max_concurrency=1,
    poll_interval_s=30,
    quota=QuotaSpec(limit=None, reset="unknown"),
)

DEMO_TRAIN = r"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import gpu

cfg = json.loads(Path(__file__).with_name("demo.json").read_text())
steps = int(cfg.get("steps", 5))
seconds = float(cfg.get("seconds", 1.0))
gpu.total_steps(steps)
if cfg.get("resume_check"):
    print("resumed=%s" % gpu.is_resumed())
    rd = gpu.resume_dir()
    if rd is not None:
        print("resume_files=%s" % sorted(p.name for p in rd.iterdir()))
for i in range(1, steps + 1):
    time.sleep(seconds / steps)
    gpu.log(step=i, loss=round(1.0 / i, 4))
for n in range(int(cfg.get("lines", 0))):
    print("line %05d %s" % (n, "x" * int(cfg.get("width", 40))))
time.sleep(float(cfg.get("sleep_after", 0)))
if cfg.get("long_line"):
    print("L" * int(cfg["long_line"]))
name = cfg.get("secret_name")
if name:
    got = hashlib.sha256(os.environ.get(name, "").encode()).hexdigest()
    print("secret_ok=%s" % (got == cfg.get("secret_sha256")))
if cfg.get("checkpoint"):
    with gpu.atomic_checkpoint("state.txt") as tmp:
        Path(tmp).write_text("step=%d" % steps)
    time.sleep(float(cfg.get("sleep_after_checkpoint", 0)))
out = gpu.output_dir()
(out / "result.json").write_text(json.dumps({"steps": steps}))
if cfg.get("extra_outputs"):
    (out / "sub").mkdir(exist_ok=True)
    (out / "sub" / "b.txt").write_text("b")
print("demo done")
if cfg.get("partial_tail"):
    sys.stdout.write("tail without newline")
    sys.stdout.flush()
sys.exit(int(cfg.get("exit", 0)))
"""


def secret_sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def make_bundle(paths: Paths, root: Path, **cfg: Any) -> Path:
    """A throwaway project running DEMO_TRAIN with `cfg`, bundled like the daemon does."""
    digest = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10]
    project = root / f"demo-{digest}"
    project.mkdir(parents=True, exist_ok=True)
    (project / "train.py").write_text(DEMO_TRAIN)
    (project / "demo.json").write_text(json.dumps(cfg))
    spec = JobSpec(project_dir=str(project), script="train.py", deps=DepsSpec(kind="none"))
    return build_bundle(project, spec, paths=paths).archive


class ColabSim:
    """The simulated colab CLI + VM for one test."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()  # /var -> /private/var on macOS; bootstrap resolves too
        self.vm = self.root / "vm"
        self.remote_root = self.vm / "content" / "gr"
        self.remote_root.mkdir(parents=True, exist_ok=True)
        self.config_dir: Path | None = None

    def settings(self, **extra: Any) -> ProviderSettings:
        return ProviderSettings(
            cli=[sys.executable, str(SIM)], remote_root=str(self.remote_root), **extra
        )

    def bind(self, adapter: ColabAdapter) -> ColabAdapter:
        self.config_dir = adapter.config_file.parent
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.control(remote_root=str(self.remote_root))
        return adapter

    # --- control + observation
    def _control_path(self) -> Path:
        assert self.config_dir is not None, "bind() the sim to an adapter first"
        return self.config_dir / "sim-control.json"

    def control(self, **updates: Any) -> dict[str, Any]:
        path = self._control_path()
        data: dict[str, Any] = json.loads(path.read_text()) if path.exists() else {}
        data.update(updates)
        path.write_text(json.dumps(data))
        return data

    def calls(self) -> list[list[str]]:
        assert self.config_dir is not None
        path = self.config_dir / "sim-calls.jsonl"
        if not path.exists():
            return []
        return [json.loads(line)["argv"] for line in path.read_text().splitlines() if line]

    def commands(self) -> list[str]:
        """The subcommand of every call, e.g. ['new', 'exec', 'upload', ...]."""
        out = []
        for argv in self.calls():
            i = argv.index("--config") + 2 if "--config" in argv else 0
            out.append(argv[i])
        return out

    def sessions(self) -> dict[str, Any]:
        assert self.config_dir is not None
        path = self.config_dir / "sessions.json.sim.json"
        if not path.exists():
            return {}
        sessions: dict[str, Any] = json.loads(path.read_text())["sessions"]
        return sessions

    def run_dir(self, session: str) -> Path:
        return self.remote_root / session

    def kill_all(self) -> None:
        """Test teardown: no runner may outlive its test."""
        for marker in self.remote_root.glob("*/launched.json"):
            try:
                pid = int(json.loads(marker.read_text())["pid"])
                os.killpg(pid, signal.SIGKILL)
            except (OSError, ValueError, KeyError):
                pass


def make_adapter(
    paths: Paths,
    clock: Clock,
    sim: ColabSim | None,
    *,
    test_mode: bool = True,
    settings: ProviderSettings | None = None,
    entry: ProviderEntry = COLAB_ENTRY,
    **extra: Any,
) -> ColabAdapter:
    if settings is None:
        settings = sim.settings(**extra) if sim is not None else ProviderSettings(**extra)
    adapter = ColabAdapter(
        AdapterDeps(
            name="colab",
            entry=entry,
            settings=settings,
            paths=paths,
            clock=clock,
            test_mode=test_mode,
        )
    )
    if sim is not None:
        sim.bind(adapter)
    return adapter


def with_bundle(ctx: AttemptContext, archive: Path, **fields: Any) -> AttemptContext:
    return ctx.model_copy(update={"bundle_archive": archive, **fields})


def all_lines(chunks: Iterable[Any]) -> list[str]:
    out: list[str] = []
    for c in chunks:
        out.extend(c.lines)
    return out
