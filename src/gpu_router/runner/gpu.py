"""`import gpu`: the job-side helper (spec decision 3). STDLIB ONLY, Python 3.8+.

Inside a gpu-router run (env GPU_ROUTER_PROTOCOL=1) calls print `::gpu::` protocol lines
(format owned by gpu_router/protocol.py; mirrored here byte for byte) that the daemon turns
into progress bars and metrics. Outside gpu-router the same calls print plain
`step=10 loss=0.41` text, use ./checkpoints and ./outputs, and never fail.

    import gpu
    gpu.total_steps(1000)
    for step in range(start, 1000):
        ...
        gpu.log(step=step, loss=loss, lr=lr)
    with gpu.atomic_checkpoint("last.pt") as tmp:   # renamed into place when complete
        torch.save(state, tmp)
    ckpt = gpu.latest_checkpoint()          # None on the first attempt
    if gpu.checkpoint_requested():          # gpu-router is about to move the job: save now
        save(gpu.checkpoint_dir())

Every public function swallows its own errors: a metrics call must never kill a training run.
The one exception is `atomic_checkpoint`, which re-raises: a checkpoint that was not saved
must not pass silently.
"""

import contextlib
import json
import math
import os
import sys
import threading
from pathlib import Path

__all__ = [
    "atomic_checkpoint",
    "checkpoint_dir",
    "checkpoint_requested",
    "data_dir",
    "enabled",
    "is_resumed",
    "latest_checkpoint",
    "log",
    "output_dir",
    "resume_dir",
    "total_steps",
]

__version__ = "0.1"

PREFIX = "::gpu:: "  # must equal gpu_router.protocol.PREFIX

_lock = threading.Lock()
_total = None


def enabled():
    """True inside a gpu-router run (the daemon is listening for protocol lines)."""
    return os.environ.get("GPU_ROUTER_PROTOCOL") == "1"


def _emit(body):
    """Print one protocol line (compact JSON, same shape as protocol.format_event)."""
    line = PREFIX + json.dumps(body, separators=(",", ":"))
    _write(line)


def _write(text):
    try:
        with _lock:
            stream = sys.stdout if sys.stdout is not None else sys.__stdout__
            if stream is None:
                return
            stream.write(text + "\n")
            stream.flush()
    except Exception:  # noqa: S110 - a closed/broken stdout must never kill the job
        pass


def _number(value):
    """float(value) if it is a finite real number (incl. numpy/torch scalars), else None."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        if hasattr(value, "item") and callable(value.item):
            value = value.item()  # 0-d tensors / numpy scalars
        f = float(value)
    except Exception:
        return None
    return f if math.isfinite(f) else None


def _int(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        if hasattr(value, "item") and callable(value.item):
            value = value.item()
        f = float(value)
    except Exception:
        return None
    if not math.isfinite(f):
        return None
    return int(f)


def total_steps(n):
    """Declare the run's total step count so progress is real (step / total)."""
    global _total
    try:
        steps = _int(n)
        if steps is None or steps <= 0:
            return
        _total = steps
        if enabled():
            _emit({"t": "total", "steps": steps})
        else:
            _write("total_steps=%d" % steps)
    except Exception:  # noqa: S110
        pass


def log(step=None, **metrics):
    """Report metrics for one step: gpu.log(step=i, loss=0.41, lr=3e-4).

    Non-numeric or non-finite values are dropped silently. Returns None.
    """
    try:
        clean = {}
        for name, value in metrics.items():
            num = _number(value)
            if num is not None:
                clean[str(name)] = num
        step_i = _int(step)
        if enabled():
            body = {"t": "metric"}
            if step_i is not None:
                body["step"] = step_i
            if _total is not None:
                body["total"] = _total
            body["metrics"] = clean
            _emit(body)
            return
        parts = []
        if step_i is not None:
            parts.append(
                "step=%d/%d" % (step_i, _total) if _total is not None else "step=%d" % step_i
            )
        parts.extend("%s=%s" % (k, _fmt(v)) for k, v in clean.items())
        if parts:
            _write(" ".join(parts))
    except Exception:  # noqa: S110
        pass


def _fmt(v):
    if v == int(v) and abs(v) < 1e15:
        return str(int(v))
    return "%.6g" % v


def _dir_from_env(var, default):
    raw = os.environ.get(var)
    path = Path(raw) if raw else Path.cwd() / default
    with contextlib.suppress(Exception):
        path.mkdir(parents=True, exist_ok=True)
    return path


def checkpoint_dir():
    """Directory to save checkpoints in (synced by the runner; resumed after migrations)."""
    return _dir_from_env("GPU_CHECKPOINT_DIR", "checkpoints")


@contextlib.contextmanager
def atomic_checkpoint(name):
    """Save a checkpoint file atomically:

        with gpu.atomic_checkpoint("last.pt") as tmp:
            torch.save(state, tmp)

    `tmp` is a temporary path on the same disk (under checkpoint_dir()/.gpu-tmp/, which the
    runner never uploads). It is renamed to checkpoint_dir()/name only when the block ends
    without an error, so a kill or OOM mid-save never leaves a half-written file where the
    last good checkpoint was. Errors inside the block propagate (the temp file is removed).
    """
    final = checkpoint_dir() / name
    tmp_dir = checkpoint_dir() / ".gpu-tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp = tmp_dir / ("%s.%d.%d" % (final.name, os.getpid(), threading.get_ident()))
    try:
        yield tmp
    except BaseException:
        with contextlib.suppress(Exception):
            if tmp.is_dir():
                import shutil

                shutil.rmtree(str(tmp))
            else:
                tmp.unlink()
        raise
    final.parent.mkdir(parents=True, exist_ok=True)
    if tmp.is_dir() and final.is_dir():  # a directory checkpoint replaced wholesale
        import shutil

        old = final.with_name(".gpu-old-%s.%d" % (final.name, os.getpid()))
        os.replace(str(final), str(old))
        os.replace(str(tmp), str(final))
        shutil.rmtree(str(old), ignore_errors=True)
    else:
        os.replace(str(tmp), str(final))


def checkpoint_requested():
    """True while gpu-router asks for a checkpoint now: the job is about to be moved to
    another provider (a session cap or the free quota is close). Save to checkpoint_dir()
    at the next safe point; the runner waits for the save to settle, syncs it and the next
    attempt resumes from it. Cheap (one stat), so it can be called every step."""
    raw = os.environ.get("GPU_CHECKPOINT_DIR")
    if not raw:
        return False
    try:
        return os.path.exists(os.path.join(raw, ".gpu-checkpoint-request"))
    except Exception:
        return False


def output_dir():
    """Directory for final outputs (fetched to <project>/runs/<id>/ when the job ends)."""
    return _dir_from_env("GPU_OUTPUT_DIR", "outputs")


def data_dir():
    """Where datasets are mounted (GPU_DATA_DIR, default ./data outside gpu-router)."""
    raw = os.environ.get("GPU_DATA_DIR")
    return Path(raw) if raw else Path.cwd() / "data"


def resume_dir():
    """Directory holding the restored latest checkpoint, or None on a fresh start."""
    raw = os.environ.get("GPU_RESUME_DIR")
    if not raw:
        return None
    path = Path(raw)
    try:
        if path.is_dir() and any(path.iterdir()):
            return path
    except Exception:
        return None
    return None


def is_resumed():
    """True when this attempt continues from a checkpoint of an earlier attempt."""
    return resume_dir() is not None


def latest_checkpoint():
    """Newest entry (file or directory) in resume_dir(), else in checkpoint_dir() when it
    already has content (a restart on the same machine); None if there is nothing."""
    try:
        for base in (resume_dir(), _existing(os.environ.get("GPU_CHECKPOINT_DIR"))):
            if base is None:
                continue
            entries = [p for p in base.iterdir() if not p.name.startswith(".")]
            if entries:
                return max(entries, key=lambda p: (_mtime(p), p.name))
    except Exception:
        return None
    return None


def _existing(raw):
    if not raw:
        return None
    path = Path(raw)
    return path if path.is_dir() else None


def _mtime(path):
    try:
        return path.stat().st_mtime
    except Exception:
        return 0.0
