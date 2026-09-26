"""Agent-submitted jobs (phase 6; D45, D48): how a job is recognised as an agent's, and the
extra intake rules an agent's spec must pass.

An agent can be steered by text it read (a job log, a cloned repo's gpu.yaml, a web page),
so its jobs get the agent approval rules (`policy.py`, source=agent) and never upload a
credential store, the home directory, or anything a data directory links to inside one.
Both the MCP server (`mcp/tools.py`) and `gpu run` from an agent's shell use this module:

- `gpu run` counts as an agent job with `--as-agent`, or when the environment says an agent
  runs it (`agent_marker`: Claude Code's `CLAUDECODE=1`, Codex's `CODEX_SANDBOX*`, or
  `GPU_ROUTER_AGENT=1` for any other tool). There is no opt-out: the markers are what an
  injected agent would have to strip, and a person who runs `! gpu run` inside Claude
  Code gets the agent rules too (they can approve with `gpu approve`).
- `check_agent_spec` refuses (400 invalid_request) a project root or dataset that is,
  sits in, or contains a credential store (`packaging.files.credential_path_problem`), a
  root or dataset that is the home directory or above it, and a dataset whose symlinks
  resolve into a credential store.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from gpu_router.errors import InvalidRequest
from gpu_router.models import JobSpec, Source

__all__ = [
    "AGENT_ENV_MARKERS",
    "MAX_DATA_WALK",
    "agent_marker",
    "check_agent_spec",
    "check_path",
    "mark_agent",
    "not_too_wide",
]

#: (variable, required value or None for "set to anything non-empty").
AGENT_ENV_MARKERS: tuple[tuple[str, str | None], ...] = (
    ("GPU_ROUTER_AGENT", "1"),
    ("CLAUDECODE", "1"),  # set by Claude Code for every Bash command it runs
    ("CODEX_SANDBOX", None),  # set by Codex's sandboxed shell (e.g. "seatbelt")
    ("CODEX_SANDBOX_NETWORK_DISABLED", None),
)

#: A dataset walk for symlinks stops after this many entries (checkpoint/data.scan still
#: never lists a file inside a credential store).
MAX_DATA_WALK = 20_000


def agent_marker(environ: Mapping[str, str] | None = None) -> str | None:
    """`NAME=value` of the first agent marker in the environment, or None."""
    env = os.environ if environ is None else environ
    for name, want in AGENT_ENV_MARKERS:
        value = env.get(name, "")
        if value and (want is None or value == want):
            return f"{name}={value}" if want is not None else name
    return None


def mark_agent(spec: JobSpec, via: str) -> JobSpec:
    """The spec as an agent's job: source=agent and the label `via`."""
    return JobSpec.model_validate(
        {**spec.model_dump(), "source": Source.AGENT, "labels": {**spec.labels, "via": via}}
    )


def not_too_wide(path: Path, what: str) -> None:
    """The home directory, `/` or anything above home would ship far more than a project
    (every personal file the credential deny list does not know about)."""
    home = Path.home().resolve()
    if path == home or path in home.parents:
        raise InvalidRequest(
            f"{what} {path} is the home directory or above it; gpu-router would ship all of it",
            hint="pass the project's own directory, e.g. ~/code/yolo",
            detail={what: str(path)},
        )


def _stores() -> tuple[str, ...]:
    """The gpu-router data dir in use is a credential store too (daemon.token)."""
    from gpu_router.paths import home_from_env

    return (str(home_from_env()),)


def check_path(path: Path, what: str, *, mount: str | None = None) -> None:
    """Refuse a path an agent wants uploaded when it is too wide or a credential store."""
    from gpu_router.packaging.files import credential_path_problem

    not_too_wide(path, what)
    problem = credential_path_problem(path, extra=_stores())
    if problem is not None:
        label = f"data {mount}" if mount else what
        detail: dict[str, str] = {"path": str(path)}
        if mount:
            detail["mount"] = mount
        raise InvalidRequest(
            f"{label}: {problem}; it is never uploaded",
            hint="give a dataset file or directory; secrets go in the Keychain "
            "(`gpu secrets set NAME`)"
            if mount
            else "pass the project's own directory, e.g. ~/code/yolo",
            detail=detail,
        )


def _check_links(root: Path, mount: str) -> None:
    """Symlinks under a dataset directory must not lead into a credential store (the
    dataset walk follows them, D44). Bounded by MAX_DATA_WALK entries."""
    from gpu_router.packaging.files import credential_path_problem

    seen = 0
    stores = _stores()
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        real_dir = os.path.realpath(dirpath)
        keep = []
        for name in sorted(dirnames):
            full = os.path.join(dirpath, name)
            real = os.path.realpath(full)
            if real_dir == real or real_dir.startswith(real.rstrip(os.sep) + os.sep):
                continue  # a loop back to an ancestor
            if os.path.islink(full):
                _refuse_link(full, real, mount, credential_path_problem(real, extra=stores))
            keep.append(name)
        dirnames[:] = keep
        for name in filenames:
            full = os.path.join(dirpath, name)
            if os.path.islink(full):
                real = os.path.realpath(full)
                _refuse_link(full, real, mount, credential_path_problem(real, extra=stores))
        seen += len(dirnames) + len(filenames)
        if seen > MAX_DATA_WALK:
            return


def _refuse_link(link: str, target: str, mount: str, problem: str | None) -> None:
    if problem is None:
        return
    raise InvalidRequest(
        f"data {mount}: {link} links to {target}, and {problem}; it is never uploaded",
        hint="remove the link or copy the files you need into the dataset",
        detail={"mount": mount, "path": link, "target": target},
    )


def check_agent_spec(spec: JobSpec, *, given_dir: Path | None = None) -> None:
    """The agent intake rules (module docstring). `given_dir`: the directory the agent
    named, when the project root (a git root) may sit above it."""
    if given_dir is not None:
        check_path(given_dir, "project_dir")
    check_path(Path(spec.project_dir), "project root")
    for ref in spec.data:
        if ref.path is None:
            continue
        raw = Path(ref.path).expanduser()
        path = (raw if raw.is_absolute() else Path(spec.project_dir) / raw).resolve()
        check_path(path, "data path", mount=ref.mount)
        if path.is_dir():
            _check_links(path, ref.mount)
