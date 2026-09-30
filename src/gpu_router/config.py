"""User configuration: config.yaml with `version: N` and forward migrations (phase 1; owner: A).

Precedence (highest first): environment variables > config.yaml > model defaults.

    GPU_ROUTER_HOME        data dir (read by paths.py, not here)
    GPU_ROUTER_PORT        daemon.port
    GPU_ROUTER_LOG_LEVEL   logging.level
    GPU_ROUTER_TEST_MODE   "1" enables test-only hooks (crash points) and the fake provider

Migrations: `CONFIG_MIGRATIONS[n]` upgrades a raw dict from version n to n+1. `load_config`
applies them in order in memory; only the daemon persists the upgraded file (after writing
`config.yaml.bak-v<old>`). A file with a version newer than CONFIG_VERSION is a ConfigError
("written by a newer gpu-router").

Secrets never live here (invariant 12): `Config` has no field that can hold a token, and
load_config rejects any key under `providers.<name>` matching models.SECRET_ENV_NAME.

This module's models are real code; the load/save/migrate functions are phase-1 work.
"""

from __future__ import annotations

import contextlib
import os
import shutil
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from gpu_router.errors import ConfigError
from gpu_router.paths import Paths

CONFIG_VERSION = 1
DEFAULT_PORT = 47291  # unassigned by IANA; override with GPU_ROUTER_PORT or daemon.port

ENV_PORT = "GPU_ROUTER_PORT"
ENV_LOG_LEVEL = "GPU_ROUTER_LOG_LEVEL"
ENV_TEST_MODE = "GPU_ROUTER_TEST_MODE"


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DaemonConfig(_Section):
    # The daemon always binds 127.0.0.1 (invariant 18); the host is deliberately not
    # configurable. port=0 means "pick a free port" (tests, dev); clients read daemon.json.
    port: int = Field(default=DEFAULT_PORT, ge=0, le=65535)
    # Wait this long for drivers/threads on SIGTERM; remote runs are never touched by a
    # daemon shutdown.
    shutdown_grace_s: float = 10


class CallTimeouts(_Section):
    """Upper bound (seconds) the engine waits for each adapter call before treating it as
    Unavailable. Adapters must bound their own subprocess/network calls below these."""

    submit: float = 300
    status: float = 60
    logs: float = 60
    fetch: float = 1800
    cancel: float = 60
    quota: float = 60
    healthcheck: float = 60


class EngineConfig(_Section):
    max_attempts: int = Field(default=6, ge=1)  # accepted attempts (remote runs) per job
    max_placements: int = Field(default=30, ge=1)  # attempt rows incl. rejected submits
    max_queue_wait_s: float = 6 * 3600  # total time waiting for placement
    backoff_base_s: float = 30  # backoff(k) = min(base*2**(k-1), cap)
    backoff_cap_s: float = 1800
    approval_timeout_s: float = 24 * 3600  # then awaiting_approval -> denied
    provision_timeout_s: float = 1800  # pending this long -> cancel + reroute
    status_stale_after_s: float = 1800  # status unreachable this long -> note
    unreachable_lost_after_s: float = 3 * 3600  # ... this long -> attempt lost, migrate
    cancel_timeout_s: float = 300  # then cancelled (cancel_unconfirmed)
    checkpoint_stall_s: float = 900  # checkpointing without end -> running
    internal_error_limit: int = 5  # consecutive engine bugs -> failed
    unknown_quota_reset_s: float = 24 * 3600  # exhausted_until when resets_at unknown
    default_poll_interval_s: float = 15
    # Unhealthy providers are re-checked on a backoff: health_recheck_min_s after the first
    # failed healthcheck, doubling up to health_recheck_s; a healthy answer resets it.
    health_recheck_s: float = Field(default=900, gt=0)
    health_recheck_min_s: float = Field(default=60, gt=0)
    # The health loop wakes at least this often, which is how it notices the Mac woke up
    # (wall time jumped past the tick); then it waits wake_grace_s for the network before
    # re-checking every provider.
    health_tick_s: float = Field(default=30, gt=0)
    wake_grace_s: float = Field(default=30, ge=0)
    max_workers: int = 16  # adapter worker threads
    per_provider_concurrency: int = 4  # concurrent adapter calls per provider
    timeouts: CallTimeouts = Field(default_factory=CallTimeouts)


class ProviderSettings(BaseModel):
    """Per-provider user settings. Adapter-specific extra keys are allowed (non-secret)."""

    model_config = ConfigDict(extra="allow")

    enabled: bool | None = None  # None = catalog's enabled_by_default
    poll_interval_s: float | None = None  # None = catalog's poll_interval_s


class CheckpointConfig(_Section):
    """Checkpoint storage and handoff (phase 5, gpu_router/checkpoint/). Defaults work
    with no config: HF Storage Bucket `<you>/gpu-router` once `gpu login hf` stored a
    token, else a directory under the data dir for local runs only."""

    interval_min: int = 20
    hf_repo: str | None = None  # unused (phase 5 chose Storage Buckets over git repos, D40)
    # auto: hf when a token is in the Keychain, else local; hf / local force one; off = none
    backend: str = Field(default="auto", pattern=r"^(auto|hf|local|off)$")
    bucket: str = "gpu-router"  # HF bucket: "<name>" (your namespace) or "<ns>/<name>"
    local_dir: str | None = None  # local backend root (default <data dir>/storage)
    keep: int = Field(default=3, ge=1, le=100)  # checkpoints kept in storage per job
    handoff_margin_min: float = Field(default=30, ge=0)  # 0 = no planned handoff
    handoff_wait_min: float = Field(default=10, ge=0)  # how long the job may take to save
    status_push_s: float = Field(default=60, ge=0)  # runner heartbeat/log-tail push; 0 = off
    control_poll_s: float = Field(default=60, ge=0)  # runner checks for checkpoint requests
    # test mode only: the fake providers' simulated runner (its "remote" is <home>/fake/ on
    # this Mac) publishes checkpoints to the local backend and resumes from GPU_RESUME_URI,
    # so migrations exercise storage end to end without a real runner (D43)
    fake_storage: bool = False
    # phase-5 review (D44): how long a placement waits for checkpoint storage that cannot
    # be reached right now (it holds the checkpoint to resume from, or the run would get
    # none) before the attempt goes ahead without it (0 = never wait)
    storage_wait_s: float = Field(default=3600, ge=0)
    # delete a job's checkpoints, heartbeats and log tails from storage when it ends
    cleanup: bool = True
    # uploaded datasets no run used for this many days are deleted (0 = keep forever)
    dataset_keep_days: float = Field(default=30, ge=0)


class StatusLineConfig(_Section):
    # How long a finished or failed job keeps a row after it ends. 0 (the default, user
    # decision 2026-09-25) shows rows only while a job is in use: queued, waiting for
    # approval, running, migrating. The macOS notification still reports the ending.
    finished_visible_s: float = Field(default=0, ge=0)
    migrated_visible_s: float = 600


class LoggingConfig(_Section):
    level: str = "INFO"
    max_bytes: int = 10 * 1024 * 1024
    backups: int = 5

    @field_validator("level")
    @classmethod
    def _level(cls, v: str) -> str:
        v = v.upper()
        if v not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
            raise ValueError(f"unknown log level {v!r}")
        return v


class Config(_Section):
    version: int = CONFIG_VERSION
    daemon: DaemonConfig = Field(default_factory=DaemonConfig)
    engine: EngineConfig = Field(default_factory=EngineConfig)
    providers: dict[str, ProviderSettings] = Field(default_factory=dict)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    statusline: StatusLineConfig = Field(default_factory=StatusLineConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    # Typed in phase 5 (routing weights, approval rules). Until then: free-form, unused.
    routing: dict[str, Any] = Field(default_factory=dict)
    policy: dict[str, Any] = Field(default_factory=dict)
    # phase 8a: macOS notifications, typed and validated in notify/settings.py
    notifications: dict[str, Any] = Field(default_factory=dict)
    test_mode: bool = False  # set from GPU_ROUTER_TEST_MODE only; never written to disk


#: from_version -> function(raw dict at from_version) -> raw dict at from_version + 1.
CONFIG_MIGRATIONS: dict[int, Callable[[dict[str, Any]], dict[str, Any]]] = {}


def migrate_config(raw: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
    """Upgrade a raw config dict to CONFIG_VERSION. Returns (upgraded, changed).

    Missing `version` means 1. Raises ConfigError if version > CONFIG_VERSION or a
    migration step is missing.
    """
    data = dict(raw)
    version_raw = data.get("version", 1)
    if isinstance(version_raw, bool) or not isinstance(version_raw, int) or version_raw < 1:
        raise ConfigError(
            f"config version must be a positive integer, got {version_raw!r}",
            hint="set `version: 1` at the top of config.yaml",
        )
    version = version_raw
    if version > CONFIG_VERSION:
        raise ConfigError(
            f"config.yaml is version {version}, written by a newer gpu-router "
            f"(this one knows up to {CONFIG_VERSION})",
            hint="upgrade gpu-router (uv tool upgrade gpu-router)",
            detail={"version": version, "known_version": CONFIG_VERSION},
        )
    changed = "version" not in raw
    while version < CONFIG_VERSION:
        step = CONFIG_MIGRATIONS.get(version)
        if step is None:
            raise ConfigError(
                f"no config migration from version {version} to {version + 1}",
                hint="this is a gpu-router bug; report it",
            )
        data = step(data)
        version += 1
        data["version"] = version
        changed = True
    data["version"] = version
    return data, changed


def load_config(paths: Paths, environ: Mapping[str, str] | None = None) -> Config:
    """Read config.yaml (absent file = defaults), migrate, apply env overrides, validate.

    Raises ConfigError with the file path and the pydantic error summary on invalid input.
    Never writes to disk.
    """
    env = os.environ if environ is None else environ
    raw = _read_raw(paths.config)
    data, _ = migrate_config(raw)
    _reject_secret_keys(data, paths.config)
    data.pop("test_mode", None)  # env-only; never read from disk

    daemon = dict(data.get("daemon") or {})
    logging_section = dict(data.get("logging") or {})
    if env.get(ENV_PORT):
        try:
            daemon["port"] = int(env[ENV_PORT])
        except ValueError:
            raise ConfigError(
                f"{ENV_PORT}={env[ENV_PORT]!r} is not a port number",
                hint=f"unset {ENV_PORT} or set it to 0-65535",
            ) from None
        data["daemon"] = daemon
    if env.get(ENV_LOG_LEVEL):
        logging_section["level"] = env[ENV_LOG_LEVEL]
        data["logging"] = logging_section
    data["test_mode"] = env.get(ENV_TEST_MODE, "").strip().lower() in {"1", "true", "yes", "on"}

    try:
        return Config.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(
            f"{paths.config} is invalid: {_summarise(exc)}",
            hint="fix the listed keys, or move the file aside to use defaults",
            detail={
                "path": str(paths.config),
                "errors": exc.errors(include_url=False, include_input=False),
            },
        ) from None


def save_config(paths: Paths, config: Config, *, backup_suffix: str | None = None) -> None:
    """Atomically write config.yaml (tmp file + os.replace, mode 0600).

    Excludes `test_mode`. If `backup_suffix` is given, first copies the existing file to
    config.yaml.<backup_suffix>. YAML comments are not preserved (documented limitation).
    """
    import yaml

    paths.home.mkdir(mode=0o700, parents=True, exist_ok=True)
    target = paths.config
    if backup_suffix is not None and target.exists():
        shutil.copy2(target, target.with_name(f"{target.name}.{backup_suffix}"))
    data = config.model_dump(mode="json", exclude={"test_mode"})
    text = yaml.safe_dump(data, sort_keys=False, default_flow_style=False)
    _write_atomic(target, text)


def persist_migrated_config(paths: Paths) -> bool:
    """Daemon startup helper: if config.yaml is older than CONFIG_VERSION, back it up as
    config.yaml.bak-v<old>, write the migrated version and return True."""
    raw = _read_raw(paths.config)
    if not raw:
        return False
    old = raw.get("version", 1)
    data, changed = migrate_config(raw)
    if not changed or old == data["version"]:
        return False
    import yaml

    shutil.copy2(paths.config, paths.config.with_name(f"{paths.config.name}.bak-v{old}"))
    data.pop("test_mode", None)
    _write_atomic(paths.config, yaml.safe_dump(data, sort_keys=False))
    return True


def set_config_section(paths: Paths, section: str, value: Any) -> None:
    """Replace one top-level section of config.yaml (phase 5: `policy`, written by the
    daemon for `gpu policy set`), keeping every other key as the file has it. Migrates
    the file to CONFIG_VERSION first; atomic, mode 0600; comments are not preserved."""
    import yaml

    if section not in Config.model_fields or section in ("version", "test_mode"):
        raise ValueError(f"not a config section: {section!r}")
    raw = _read_raw(paths.config)
    data, _ = migrate_config(raw)
    data.pop("test_mode", None)
    data[section] = value
    paths.home.mkdir(mode=0o700, parents=True, exist_ok=True)
    _write_atomic(paths.config, yaml.safe_dump(data, sort_keys=False, default_flow_style=False))


# --------------------------------------------------------------------------- helpers


def _read_raw(path: Path) -> dict[str, Any]:
    """Parse config.yaml into a dict; absent or empty file -> {}."""
    import yaml

    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError as exc:
        raise ConfigError(f"cannot read {path}: {exc.strerror}") from None
    try:
        loaded = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(
            f"{path} is not valid YAML: {exc}",
            hint="fix the syntax, or move the file aside to use defaults",
            detail={"path": str(path)},
        ) from None
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ConfigError(
            f"{path} must be a mapping of settings, got {type(loaded).__name__}",
            detail={"path": str(path)},
        )
    return loaded


def _reject_secret_keys(data: Mapping[str, Any], path: Path) -> None:
    from gpu_router.models import SECRET_ENV_NAME

    providers = data.get("providers") or {}
    if not isinstance(providers, Mapping):
        return
    for pname, settings in providers.items():
        if not isinstance(settings, Mapping):
            continue
        for key in settings:
            if SECRET_ENV_NAME.search(str(key)):
                raise ConfigError(
                    f"{path}: providers.{pname}.{key} looks like a secret; secrets never "
                    "live in config.yaml",
                    hint=f"remove it and store it with `gpu secrets set {key}`",
                    detail={"path": str(path), "key": f"providers.{pname}.{key}"},
                )


def _summarise(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors()[:5]:
        loc = ".".join(str(p) for p in err["loc"]) or "(root)"
        parts.append(f"{loc}: {err['msg']}")
    more = len(exc.errors()) - len(parts)
    if more > 0:
        parts.append(f"and {more} more")
    return "; ".join(parts)


def _write_atomic(target: Path, text: str) -> None:
    tmp = target.with_name(f".{target.name}.tmp{os.getpid()}")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise
