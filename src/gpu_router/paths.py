"""Data-directory layout (phase 1; owner: group A). STDLIB ONLY (invariant 14).

Everything gpu-router persists on this Mac lives under one directory, `Paths.home`:

    ~/Library/Application Support/gpu-router/     (override: GPU_ROUTER_HOME)
      gpu.db, gpu.db-wal, gpu.db-shm              SQLite (WAL). Only the daemon opens it.
      daemon.lock                                 flock held by the running daemon (single instance)
      daemon.token                                bearer token, mode 0600, created by the daemon
      daemon.json                                 runtime info: pid, port, version, started_at
      state.json                                  status-line cache, rewritten atomically
      config.yaml                                 user config, has `version: N`
      providers.yaml                              optional user overrides of the packaged catalog
      logs/daemon.jsonl                           structured daemon log (rotated)
      logs/launchd.log                            stdout/stderr of the launchd-run daemon
      jobs/<job_id>/bundle/                       job bundle (manifest.json, code/, gpu_runner/)
      jobs/<job_id>/bundle.tar.gz                 same bundle, archived for upload
      jobs/<job_id>/logs/attempt-<n>.log          captured remote log, one file per attempt
      jobs/<job_id>/metrics.jsonl                 parsed metric points
      bundles/<sha256>.tar.gz                     content-addressed bundle cache (phase 2, D15)
      storage/                                    local checkpoint storage + hf.json namespace
                                                  cache + data-index.json (phase 5, D40)
      providers/<name>/                           adapter-private scratch (never secrets)
      providers/local/{venvs,data}/               local provider: venv per deps key, GPU_DATA_DIR
      providers/colab/{runs,colab-cli}/           colab: run records, private CLI session file
      providers/kaggle/                           kaggle: submit markers, final/, tombstones
      local/<attempt key>/                        the local provider's "remote" run dirs (D28)
      fake/                                       the fake provider's "remote" (outside the daemon)

This file is real code (it is data: the file layout). Nothing here touches the disk except
`Paths.ensure()`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

ENV_HOME = "GPU_ROUTER_HOME"
DEFAULT_HOME = Path.home() / "Library" / "Application Support" / "gpu-router"
STATE_FILE_NAME = "state.json"


def home_from_env(environ: Mapping[str, str] | None = None) -> str:
    """Return the data dir as a string using only `os` (for the status-line fast path)."""
    env = os.environ if environ is None else environ
    override = env.get(ENV_HOME)
    return os.path.expanduser(override) if override else str(DEFAULT_HOME)


def state_file_fast(environ: Mapping[str, str] | None = None) -> str:
    """Path of state.json without constructing `Paths` (fast path helper)."""
    return os.path.join(home_from_env(environ), STATE_FILE_NAME)


@dataclass(frozen=True, slots=True)
class Paths:
    """Resolved locations inside the data directory. Construct with `Paths.from_env()`."""

    home: Path

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Paths:
        return cls(Path(home_from_env(environ)).resolve())

    # ---- top-level files
    @property
    def db(self) -> Path:
        return self.home / "gpu.db"

    @property
    def lock(self) -> Path:
        return self.home / "daemon.lock"

    @property
    def token(self) -> Path:
        return self.home / "daemon.token"

    @property
    def runtime(self) -> Path:
        return self.home / "daemon.json"

    @property
    def state(self) -> Path:
        return self.home / STATE_FILE_NAME

    @property
    def config(self) -> Path:
        return self.home / "config.yaml"

    @property
    def user_providers(self) -> Path:
        return self.home / "providers.yaml"

    # ---- logs
    @property
    def logs_dir(self) -> Path:
        return self.home / "logs"

    @property
    def daemon_log(self) -> Path:
        return self.logs_dir / "daemon.jsonl"

    @property
    def launchd_log(self) -> Path:
        return self.logs_dir / "launchd.log"

    # ---- per job
    @property
    def jobs_dir(self) -> Path:
        return self.home / "jobs"

    def job_dir(self, job_id: str) -> Path:
        return self.jobs_dir / job_id

    def job_bundle_dir(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "bundle"

    def job_bundle_archive(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "bundle.tar.gz"

    def job_log(self, job_id: str, attempt_n: int) -> Path:
        return self.job_dir(job_id) / "logs" / f"attempt-{attempt_n}.log"

    def job_metrics(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "metrics.jsonl"

    # ---- providers
    @property
    def providers_dir(self) -> Path:
        return self.home / "providers"

    def provider_dir(self, name: str) -> Path:
        return self.providers_dir / name

    @property
    def fake_dir(self) -> Path:
        return self.home / "fake"

    def ensure(self) -> None:
        """Create home (0700), logs/, jobs/, providers/. Idempotent. mkdir(mode=) does not
        touch a directory that already exists (e.g. GPU_ROUTER_HOME created earlier with a
        plain mkdir, 0755), so an existing one owned by this user is tightened to 0700:
        the WAL, state.json and job logs inside must not be readable by other accounts."""
        self.home.mkdir(mode=0o700, parents=True, exist_ok=True)
        for d in (self.logs_dir, self.jobs_dir, self.providers_dir):
            d.mkdir(mode=0o700, exist_ok=True)
        for d in (self.home, self.logs_dir, self.jobs_dir, self.providers_dir):
            _tighten(d)


def _tighten(path: Path) -> None:
    """chmod 0700 if owned by the current user and group/other have any access."""
    try:
        st = path.stat()
        if st.st_uid == os.getuid() and st.st_mode & 0o077:
            os.chmod(path, 0o700)
    except OSError:
        pass
