"""gpu.yaml + command-line flags -> JobSpec (phase 2).

Spec decision 2: a per-project `gpu.yaml` plus flags; flags override the file.

Project root: the nearest ancestor of the working directory (itself included) that holds a
`.git` entry, else the working directory. `gpu.yaml` is read from the project root only.

gpu.yaml schema, version 1 (every key optional; unknown keys are errors with a suggestion):

    version: 1                    # schema version; omit = 1; newer than this build = error
    name: train-yolo              # display name (default: script stem)
    script: train.py              # entrypoint relative to the project root ...
    command: [bash, run.sh]       # ... or an argv run from the project root (not both)
    args: [--epochs, "10"]        # extra argv for the script/command (string or list)
    vram: 16                      # GB of GPU memory the job needs (alias: vram_gb)
    hours: 2.5                    # expected runtime in hours
    provider: kaggle              # route only to this provider
    gpu: T4                       # GPU type constraint
    env: {WANDB_MODE: offline}    # NON-secret env vars (secret-looking names are rejected)
    secrets: [WANDB_API_KEY]      # Keychain secret names exposed to the job as env vars
                                  #   (never gpu-router's own: HF_TOKEN, kaggle, ...)
    deps: auto                    # auto | none | requirements.txt | pyproject.toml |
                                  #   {kind: requirements, file: reqs/gpu.txt}
    data:                         # datasets, mounted under $GPU_DATA_DIR/<mount>
      - {mount: coco, path: data/coco}
      - {mount: wiki, uri: "hf://datasets/me/wiki"}
    include: [third_party/, "data/crops/*.png"]   # ship these even if git ignores them
                                  #   (path or list; globs; `**` = any dirs; D60)
    checkpoint_interval_min: 20   # 0 disables checkpoint sync
    interactive: false
    requires_approval: false      # always ask before running
    smoke: false                  # quick smoke test: prefer the local Mac (MPS), phase 5
    max_attempts: 3               # remote runs before giving up
    labels: {team: vision}        # free-form metadata
    provider_options:             # opaque per-provider knobs
      fake: {duration: 2}

Merge rule: start from the file, then every flag the user actually passed replaces the
file's value; `--env K=V` entries are merged over the file's `env` (flag wins per key), and
script args given on the command line replace the file's `args` entirely. Passing a script
on the command line replaces both `script` and `command` from the file, and drops the file's
`name` unless it is the same entrypoint (or `--name` is given). gpu.yaml's `script` is checked
to exist whenever it is the entrypoint, with or without script args. `--include` entries are
added to the file's `include:` (both ship).

Errors are `GpuYamlError` / `InvalidSpec` (exit code 2 in the CLI) with messages that name
the file, the key (and its line), what is wrong and what would be right.
"""

from __future__ import annotations

import difflib
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gpu_router.errors import InvalidSpec
from gpu_router.models import (
    DataRef,
    DepsSpec,
    JobSpec,
    Source,
    secret_env_problem,
    secret_names_problem,
)

GPU_YAML = "gpu.yaml"
GPU_YAML_VERSION = 1

__all__ = [
    "GPU_YAML",
    "GPU_YAML_VERSION",
    "Flags",
    "GpuYaml",
    "GpuYamlError",
    "build_spec",
    "find_project_root",
    "load_gpu_yaml",
    "parse_env_pairs",
]

# key -> (canonical JobSpec field, human type description)
_KEYS: dict[str, tuple[str, str]] = {
    "version": ("", "an integer"),
    "name": ("name", "a string of at most 80 characters"),
    "script": ("script", "a path relative to the project root, e.g. train.py"),
    "command": ("command", "a list of strings, e.g. [bash, run.sh]"),
    "args": ("args", "a list of strings"),
    "vram": ("vram_gb", "a number of GB greater than 0, e.g. 16"),
    "vram_gb": ("vram_gb", "a number of GB greater than 0, e.g. 16"),
    "hours": ("hours", "a number of hours greater than 0, e.g. 2.5"),
    "provider": ("provider", "a provider name such as kaggle"),
    "gpu": ("gpu", "a GPU type such as T4"),
    "env": ("env", "a mapping of NAME: value"),
    "secrets": ("secrets", "a list of Keychain secret names"),
    "deps": ("deps", "auto, none, a requirements/pyproject file, or {kind:, file:}"),
    "data": ("data", "a list of {mount:, path:} or {mount:, uri:}"),
    "include": (
        "include",
        "a path or glob relative to the project root, or a list of them, e.g. "
        '[third_party/, "data/crops/*.png"]',
    ),
    "checkpoint_interval_min": ("checkpoint_interval_min", "whole minutes, 0 to disable"),
    "interactive": ("interactive", "true or false"),
    "requires_approval": ("requires_approval", "true or false"),
    "smoke": ("smoke", "true or false"),
    "max_attempts": ("max_attempts", "an integer from 1 to 50"),
    "labels": ("labels", "a mapping of string: string"),
    "provider_options": ("provider_options", "a mapping of provider: {option: value}"),
}


class GpuYamlError(InvalidSpec):
    """gpu.yaml could not be read or is invalid. `detail` carries file, key and line."""


@dataclass(frozen=True)
class GpuYaml:
    """A parsed, validated-shape gpu.yaml. `values` uses JobSpec field names."""

    path: Path | None
    values: dict[str, Any] = field(default_factory=dict)
    lines: dict[str, int] = field(default_factory=dict)  # canonical field -> 1-based line


@dataclass
class Flags:
    """What the user typed. `None` / empty means "not given" (the file's value stands)."""

    script: str | None = None  # the positional entrypoint as typed (.py -> script)
    args: list[str] | None = None  # argv after the entrypoint; None = not given
    vram_gb: float | None = None
    hours: float | None = None
    provider: str | None = None
    gpu: str | None = None
    name: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    smoke: bool | None = None  # --smoke (phase 5); None = gpu.yaml's value stands
    source: Source = Source.CLI
    provider_options: dict[str, dict[str, Any]] | None = None
    labels: dict[str, str] = field(default_factory=dict)
    include: list[str] = field(default_factory=list)  # --include (D60), added to gpu.yaml's


# --------------------------------------------------------------------------- project root


def find_project_root(start: Path | None = None) -> Path:
    """Nearest ancestor (inclusive) of `start` (default cwd) containing `.git`, else start.
    Pure filesystem walk: no `git` subprocess, so it is fast and works without git."""
    here = (start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        if (candidate / ".git").exists():
            return candidate
    return here


# --------------------------------------------------------------------------- gpu.yaml


def _key_lines(text: str) -> dict[str, int]:
    """Top-level key -> 1-based line number, via yaml.compose (best effort)."""
    import yaml

    try:
        node = yaml.compose(text, Loader=yaml.SafeLoader)
    except yaml.YAMLError:
        return {}
    if not isinstance(node, yaml.MappingNode):
        return {}
    out: dict[str, int] = {}
    for key_node, _ in node.value:
        if isinstance(key_node, yaml.ScalarNode):
            out[str(key_node.value)] = key_node.start_mark.line + 1
    return out


def _display(path: Path | None) -> str:
    """gpu.yaml path as the user would type it: relative to the cwd when inside it."""
    if path is None:
        return GPU_YAML
    try:
        rel = os.path.relpath(path)
    except ValueError:
        return str(path)
    return str(path) if rel.startswith("..") else rel


def _where(path: Path | None, key: str | None, lines: Mapping[str, int]) -> str:
    name = _display(path)
    if key is None:
        return name
    line = lines.get(key)
    return f"{name}:{line}: `{key}`" if line else f"{name}: `{key}`"


def load_gpu_yaml(project_root: Path) -> GpuYaml:
    """Read and shape-check `<project_root>/gpu.yaml`. Missing file -> empty GpuYaml."""
    import yaml

    path = project_root / GPU_YAML
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return GpuYaml(path=None)
    except OSError as exc:
        raise GpuYamlError(
            f"cannot read {path}: {exc.strerror or exc}",
            hint="check the file's permissions",
            detail={"file": str(path)},
        ) from exc
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        loc = f"{path}:{mark.line + 1}" if mark is not None else str(path)
        problem = getattr(exc, "problem", None) or "invalid YAML"
        raise GpuYamlError(
            f"{loc}: {problem}",
            hint="gpu.yaml must be a YAML mapping, e.g. `script: train.py`",
            detail={"file": str(path), "line": mark.line + 1 if mark is not None else None},
        ) from exc
    if raw is None:
        return GpuYaml(path=path)
    if not isinstance(raw, dict):
        raise GpuYamlError(
            f"{path}: expected a mapping of settings, got {type(raw).__name__}",
            hint="for example:\n    script: train.py\n    vram: 16",
            detail={"file": str(path)},
        )
    key_lines = _key_lines(text)
    return _shape(path, raw, key_lines)


def _fail(
    path: Path | None,
    key: str,
    lines: Mapping[str, int],
    message: str,
    *,
    type_hint: bool = True,
) -> GpuYamlError:
    expected = _KEYS.get(key, ("", ""))[1]
    hint = f"`{key}` must be {expected}" if expected and type_hint else None
    return GpuYamlError(
        f"{_where(path, key, lines)} {message}",
        hint=hint,
        detail={"file": str(path) if path else None, "key": key, "line": lines.get(key)},
    )


def _as_str_list(value: Any) -> list[str] | None:
    if isinstance(value, str):
        import shlex

        return shlex.split(value)
    if isinstance(value, list) and all(
        isinstance(v, str | int | float) and not isinstance(v, bool) for v in value
    ):
        return [str(v) for v in value]
    return None


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _shape(path: Path, raw: dict[Any, Any], lines: Mapping[str, int]) -> GpuYaml:
    values: dict[str, Any] = {}
    field_lines: dict[str, int] = {}
    version = raw.get("version", GPU_YAML_VERSION)
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise _fail(path, "version", lines, f"is {version!r}, not a schema version")
    if version > GPU_YAML_VERSION:
        raise GpuYamlError(
            f"{_where(path, 'version', lines)} is {version}, but this gpu-router reads "
            f"gpu.yaml version {GPU_YAML_VERSION}",
            hint="upgrade gpu-router, or set `version: 1`",
            detail={"file": str(path), "key": "version", "line": lines.get("version")},
        )
    for key_any, value in raw.items():
        key = str(key_any)
        if key not in _KEYS:
            close = difflib.get_close_matches(key, list(_KEYS), n=1, cutoff=0.6)
            hint = (
                f"did you mean `{close[0]}`?"
                if close
                else "known keys: " + ", ".join(k for k in _KEYS if k != "vram_gb")
            )
            raise GpuYamlError(
                f"{_where(path, key, lines)} is not a gpu.yaml setting",
                hint=hint,
                detail={"file": str(path), "key": key, "line": lines.get(key)},
            )
        if key == "version":
            continue
        target = _KEYS[key][0]
        if target in values:
            raise _fail(path, key, lines, f"duplicates another spelling of `{target}`")
        values[target] = _convert(path, key, value, lines)
        if key in lines:
            field_lines[target] = lines[key]
    if "script" in values and "command" in values:
        raise GpuYamlError(
            f"{path}: set either `script` or `command`, not both",
            hint="`script: train.py` for a Python file, `command: [bash, run.sh]` for anything",
            detail={"file": str(path), "key": "command", "line": lines.get("command")},
        )
    return GpuYaml(path=path, values=values, lines=field_lines)


def _convert(path: Path, key: str, value: Any, lines: Mapping[str, int]) -> Any:
    fail = _fail
    if key in ("name", "script", "provider", "gpu"):
        if not isinstance(value, str) or not value.strip():
            raise fail(path, key, lines, f"is {value!r}")
        # provider names are lowercase everywhere (the --provider flag is lowercased too)
        return value.strip().lower() if key == "provider" else value.strip()
    if key in ("command", "args", "secrets"):
        items = _as_str_list(value)
        if items is None or (key == "command" and not items):
            raise fail(path, key, lines, f"is {value!r}")
        return items
    if key in ("vram", "vram_gb", "hours"):
        num = _number(value)
        if num is None or num <= 0:
            raise fail(path, key, lines, f"is {value!r}")
        return num
    if key in ("checkpoint_interval_min", "max_attempts"):
        if isinstance(value, bool) or not isinstance(value, int):
            raise fail(path, key, lines, f"is {value!r}")
        return value
    if key in ("interactive", "requires_approval", "smoke"):
        if not isinstance(value, bool):
            raise fail(path, key, lines, f"is {value!r}")
        return value
    if key in ("env", "labels"):
        if not isinstance(value, dict):
            raise fail(path, key, lines, f"is {type(value).__name__}, not a mapping")
        out: dict[str, str] = {}
        for k, v in value.items():
            if isinstance(v, dict | list) or v is None:
                raise fail(path, key, lines, f"entry {k!r} must be a plain value, got {v!r}")
            out[str(k)] = str(v).lower() if isinstance(v, bool) else str(v)
        return out
    if key == "provider_options":
        if not isinstance(value, dict) or not all(isinstance(v, dict) for v in value.values()):
            raise fail(path, key, lines, "must map each provider name to a mapping")
        return {str(k): dict(v) for k, v in value.items()}
    if key == "deps":
        return _deps(path, value, lines)
    if key == "data":
        if not isinstance(value, list):
            raise fail(path, key, lines, f"is {type(value).__name__}, not a list")
        return value  # validated as DataRef in build_spec (messages carry the index)
    if key == "include":
        # one pattern or a list of them; never shell-split (a glob is one token)
        if isinstance(value, str):
            return [value]
        if isinstance(value, list) and all(isinstance(v, str) for v in value):
            return list(value)
        raise fail(path, key, lines, f"is {value!r}")
    raise AssertionError(key)  # pragma: no cover


def _deps(path: Path, value: Any, lines: Mapping[str, int]) -> dict[str, Any]:
    if isinstance(value, str):
        v = value.strip()
        if v in ("auto", "none"):
            return {"kind": v}
        if v.endswith(".toml"):
            return {"kind": "pyproject", "file": v}
        if v.endswith((".txt", ".in")):
            return {"kind": "requirements", "file": v}
        raise _fail(path, "deps", lines, f"is {value!r}")
    if isinstance(value, dict):
        return dict(value)
    raise _fail(path, "deps", lines, f"is {value!r}")


# --------------------------------------------------------------------------- flags


def parse_env_pairs(pairs: Sequence[str]) -> dict[str, str]:
    """["K=V", ...] -> {K: V}. Raises InvalidSpec naming the bad entry."""
    out: dict[str, str] = {}
    for pair in pairs:
        name, sep, value = pair.partition("=")
        if not sep or not name:
            raise InvalidSpec(
                f"--env {pair!r} is not NAME=VALUE",
                hint="for example: --env WANDB_MODE=offline",
                detail={"flag": "--env", "value": pair},
            )
        out[name] = value
    return out


def _entrypoint(project_root: Path, cwd: Path, token: str, args: list[str]) -> dict[str, Any]:
    """A positional entrypoint -> {"script"} for a .py file, else {"command"}."""
    if token.endswith(".py"):
        local = (cwd / token).resolve() if not os.path.isabs(token) else Path(token).resolve()
        try:
            rel = local.relative_to(project_root)
        except ValueError:
            raise InvalidSpec(
                f"{token} is outside the project ({project_root})",
                hint="run gpu from inside the project, or move the script into it",
                detail={"script": token, "project_dir": str(project_root)},
            ) from None
        if not local.is_file():
            raise InvalidSpec(
                f"script {rel.as_posix()} not found in {project_root}",
                hint="check the path; it is relative to the current directory",
                detail={"script": rel.as_posix(), "project_dir": str(project_root)},
            )
        return {"script": rel.as_posix(), "command": None}
    return {"script": None, "command": [token, *args]}


def build_spec(
    flags: Flags,
    *,
    cwd: Path | None = None,
    project_root: Path | None = None,
    gpu_yaml: GpuYaml | None = None,
) -> tuple[JobSpec, GpuYaml, Path]:
    """Merge gpu.yaml and flags into a validated JobSpec.

    Returns (spec, the gpu.yaml used, project root). Raises GpuYamlError / InvalidSpec.
    """
    from pydantic import ValidationError

    here = (cwd or Path.cwd()).resolve()
    root = (project_root or find_project_root(here)).resolve()
    doc = gpu_yaml if gpu_yaml is not None else load_gpu_yaml(root)
    values: dict[str, Any] = dict(doc.values)
    from_flags: set[str] = set()

    if flags.script is not None:
        args = list(flags.args or [])
        entry = _entrypoint(root, here, flags.script, args)
        if flags.name is None and not _same_entrypoint(doc.values, entry):
            # gpu.yaml's `name` describes gpu.yaml's entrypoint, not the one typed here
            values.pop("name", None)
        values.update(entry)
        if entry["script"] is not None:
            values["args"] = args
        else:
            values.pop("args", None)
        from_flags.update({"script", "command", "args"})
    else:
        if flags.args:
            values["args"] = list(flags.args)
            from_flags.add("args")
        # The entrypoint comes from gpu.yaml: check it exists up front, with or without
        # script args (the daemon only warns about a missing script, see packaging).
        if values.get("script") and not (root / values["script"]).is_file():
            raise GpuYamlError(
                f"{_where(doc.path, 'script', {'script': doc.lines.get('script', 0)})} "
                f"names {values['script']}, which does not exist in {root}",
                hint="fix `script:` in gpu.yaml or pass the script: gpu run <script>",
                detail={"file": str(doc.path), "key": "script"},
            )

    for attr in ("vram_gb", "hours", "provider", "gpu", "name", "smoke"):
        val = getattr(flags, attr)
        if val is not None:
            values[attr] = val
            from_flags.add(attr)
    if flags.env:
        values["env"] = {**values.get("env", {}), **flags.env}
        from_flags.add("env")
    if flags.labels:
        values["labels"] = {**values.get("labels", {}), **flags.labels}
    if flags.include:
        values["include"] = [*values.get("include", []), *flags.include]
        from_flags.add("include")
    if flags.provider_options is not None:
        values["provider_options"] = {
            **values.get("provider_options", {}),
            **flags.provider_options,
        }

    if values.get("script") is None and values.get("command") is None:
        raise InvalidSpec(
            "nothing to run: give a script, e.g. `gpu run train.py`",
            hint=f"or set `script: train.py` in {root / GPU_YAML}",
            detail={"project_dir": str(root)},
        )
    values = {k: v for k, v in values.items() if v is not None}

    data = values.pop("data", None)
    deps = values.pop("deps", None)
    try:
        spec_kwargs: dict[str, Any] = dict(values)
        if deps is not None:
            spec_kwargs["deps"] = _deps_spec(doc, deps)
        if data is not None:
            spec_kwargs["data"] = [_data_ref(doc, i, item) for i, item in enumerate(data)]
        spec = JobSpec(project_dir=str(root), source=flags.source, **spec_kwargs)
    except ValidationError as exc:
        raise _from_validation(exc, doc, flags, from_flags, deps_given=deps is not None) from None
    problem = secret_env_problem(dict(spec.env))
    if problem is not None:  # the daemon checks again (D39); here it gets a file:line
        key = next((k for k in spec.env if repr(k) in problem), None)
        if key is not None and key in flags.env:
            raise InvalidSpec(f"--env: {problem}", detail={"flag": "--env", "field": "env"})
        lines = {"env": doc.lines["env"]} if "env" in doc.lines else {}
        raise _fail(doc.path, "env", lines, f"is invalid: {problem}", type_hint=False)
    problem = secret_names_problem(list(spec.secrets))
    if problem is not None:  # the daemon checks again (D48)
        lines = {"secrets": doc.lines["secrets"]} if "secrets" in doc.lines else {}
        raise _fail(doc.path, "secrets", lines, f"is invalid: {problem}", type_hint=False)
    return spec, doc, root


def _same_entrypoint(file_values: Mapping[str, Any], entry: Mapping[str, Any]) -> bool:
    """Does the entrypoint typed on the command line match gpu.yaml's?"""
    if entry.get("script") is not None:
        return bool(file_values.get("script") == entry["script"])
    file_cmd = list(file_values.get("command") or [])
    cli_cmd = list(entry.get("command") or [])
    return bool(file_cmd) and cli_cmd[: len(file_cmd)] == file_cmd


def _deps_spec(doc: GpuYaml, deps: Any) -> DepsSpec:
    from pydantic import ValidationError

    try:
        return DepsSpec.model_validate(deps)
    except ValidationError as exc:
        err = exc.errors()[0]
        loc = ".".join(str(p) for p in err["loc"])
        lines = {"deps": doc.lines["deps"]} if "deps" in doc.lines else {}
        raise _fail(doc.path, "deps", lines, f"is invalid ({loc}): {_clean(err['msg'])}") from None


def _data_ref(doc: GpuYaml, index: int, item: Any) -> DataRef:
    from pydantic import ValidationError

    try:
        return DataRef.model_validate(item)
    except ValidationError as exc:
        err = exc.errors()[0]
        loc = ".".join(str(p) for p in err["loc"])
        where = _where(doc.path, "data", {"data": doc.lines.get("data", 0)})
        raise GpuYamlError(
            f"{where} entry {index + 1}{' ' + loc if loc else ''}: {_clean(err['msg'])}",
            hint="each entry is {mount: name, path: local/dir} or {mount: name, uri: hf://...}",
            detail={"file": str(doc.path), "key": "data", "index": index},
        ) from None


_FLAG_NAMES = {
    "vram_gb": "--vram",
    "hours": "--hours",
    "provider": "--provider",
    "gpu": "--gpu",
    "name": "--name",
    "smoke": "--smoke",
    "env": "--env",
    "include": "--include",
    "script": "the script argument",
    "command": "the command",
    "args": "the script arguments",
}

_FILE_KEYS = {v[0]: k for k, v in reversed(list(_KEYS.items())) if v[0]}


def _clean(msg: str) -> str:
    return msg.removeprefix("Value error, ")


def _from_validation(
    exc: Any, doc: GpuYaml, flags: Flags, from_flags: set[str], *, deps_given: bool
) -> InvalidSpec:
    err = exc.errors()[0]
    loc = [str(p) for p in err["loc"]]
    msg = _clean(str(err["msg"]))
    top = loc[0] if loc else ""
    # our own validators (value_error) already explain themselves; type/range errors
    # get a "must be ..." hint
    type_hint = err.get("type") != "value_error"
    # env and include are merged from both places; blame the one holding the entry the
    # message names (their validators quote the offending name / pattern with repr())
    given = {"env": list(flags.env), "include": [p.strip() for p in flags.include]}
    if (
        top in given
        and top in from_flags
        and top in doc.values
        and not any(repr(k) in msg for k in given[top])
    ):
        from_flags = from_flags - {top}
    if top in from_flags:
        flag = _FLAG_NAMES.get(top, top)
        expected = _KEYS.get(_FILE_KEYS.get(top, top), ("", "valid"))[1]
        return InvalidSpec(
            f"{flag}: {msg}",
            hint=f"{flag} must be {expected}" if type_hint else None,
            detail={"flag": _FLAG_NAMES.get(top, top), "field": top},
        )
    if top and (top in doc.values or (top == "deps" and deps_given)):
        key = _FILE_KEYS.get(top, top)
        lines = {key: doc.lines[top]} if top in doc.lines else {}
        sub = f" ({'.'.join(loc[1:])})" if len(loc) > 1 else ""
        return _fail(doc.path, key, lines, f"is invalid{sub}: {msg}", type_hint=type_hint)
    return InvalidSpec(f"invalid job spec: {msg}", detail={"loc": loc})
