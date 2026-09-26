"""Local, durable record of every Colab run this adapter started (phase 3).

Invariant 10 keeps adapters stateless toward gpu-router, but a Colab run has state that
lives nowhere else: the session name (derived from the attempt key), how far teardown got,
the exit code and the harvested log/outputs once the VM is gone. One JSON file per session
under `<home>/providers/colab/runs/<session>/record.json`, written atomically, is that
state. It holds no secrets (A8): session tokens stay in the CLI's own session file, which
this module never reads.

The daemon can be SIGKILLed at any point; every step writes the record BEFORE the remote
action it describes (e.g. `creating` before `colab new`), so a restarted daemon finds the
session again through `lookup_by_key` and can finish or clean it up.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError


class RunState(StrEnum):
    CREATING = "creating"  # record written, `colab new` in progress
    SETUP = "setup"  # session exists; uploading / launching
    LAUNCHED = "launched"  # runner started on the VM
    EXITED = "exited"  # runner finished (exit_code set); teardown may still be running
    CANCELLED = "cancelled"  # stopped by gpu-router before the runner finished
    LOST = "lost"  # the VM went away (reclaimed, idle, usage limit) or setup was interrupted
    REJECTED = "rejected"  # `colab new` refused: nothing was created
    ABANDONED = "abandoned"  # submit gave up after creating a session; it was stopped


#: States in which submit() never returned a ref: lookup_by_key reports "no run".
NO_RUN_STATES = frozenset({RunState.REJECTED, RunState.ABANDONED})
#: States where the attempt is still being created by some submit() call.
STARTING_STATES = frozenset({RunState.CREATING, RunState.SETUP})
#: Final states from the engine's point of view (teardown tracked by `stopped`).
FINAL_STATES = frozenset(
    {RunState.EXITED, RunState.CANCELLED, RunState.LOST, RunState.REJECTED, RunState.ABANDONED}
)


class RunRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")

    session: str
    attempt_key: str
    job_id: str
    attempt_n: int
    state: RunState
    owner: str  # adapter-instance token that created it (stale-submit detection)
    gpu_requested: str
    cli_pid: int | None = None  # pid of the `colab new` child while state is creating
    gpu_seen: str | None = None
    created_at: float
    launched_at: float | None = None  # Mac clock (injected), when the runner started
    ended_at: float | None = None
    exit_code: int | None = None
    lost_reason: str | None = None
    quota_exhausted: bool = False
    message: str | None = None
    # teardown / harvest progress
    log_cached: bool = False
    outputs_cached: bool = False
    outputs_pending: bool = False  # exit 0 but outputs too big to pull eagerly: fetch pulls
    outputs_files: int | None = None
    outputs_bytes: int | None = None
    settle_tries: int = 0  # harvest attempts so far (gives up after a few)
    stopped: bool = False
    stop_error: str | None = None
    log_served_at: float | None = None  # logs() served the final eof (raw copy purge clock)
    log_purged: bool = False  # the raw harvested job.log was deleted (RAW_LOG_GRACE_S)
    # checkpoints
    ckpt_mirrored: str | None = None  # newest archive name copied to runs/<session>/ckpt/
    resume: str | None = None  # "uploaded" | "unavailable" | "storage" (phase 5) | None

    @property
    def final(self) -> bool:
        return self.state in FINAL_STATES


class RunStore:
    """Thread-safe enough for one daemon: callers serialize per session with their own lock;
    writes are atomic renames, reads tolerate a missing or half-written file."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def run_dir(self, session: str) -> Path:
        return self.root / session

    def _path(self, session: str) -> Path:
        return self.run_dir(session) / "record.json"

    def load(self, session: str) -> RunRecord | None:
        path = self._path(session)
        try:
            raw = path.read_text(encoding="utf-8")
        except (FileNotFoundError, NotADirectoryError):
            return None
        except OSError:
            return None
        try:
            return RunRecord.model_validate(json.loads(raw))
        except (ValueError, ValidationError):
            return None

    def save(self, rec: RunRecord) -> RunRecord:
        d = self.run_dir(rec.session)
        d.mkdir(mode=0o700, parents=True, exist_ok=True)
        data = rec.model_dump_json(indent=1)
        fd, tmp = tempfile.mkstemp(prefix=".record-", suffix=".json", dir=d)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self._path(rec.session))
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise
        return rec

    def all(self) -> list[RunRecord]:
        if not self.root.is_dir():
            return []
        out: list[RunRecord] = []
        for child in sorted(self.root.iterdir()):
            if child.is_dir():
                rec = self.load(child.name)
                if rec is not None:
                    out.append(rec)
        return out
