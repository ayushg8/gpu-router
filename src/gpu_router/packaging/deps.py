"""Dependency detection (phase 2). Resolves `DepsSpec` at bundle time into what the remote
runner installs, so the py3.8 runner never has to parse TOML.

`auto`: requirements.txt if bundled, else pyproject.toml `[project].dependencies` (plus
`[tool.uv]`-style `dev` groups are ignored on purpose: remote runs need runtime deps only),
else none.

`[tool.uv.sources]` is honoured, because the remote side runs plain pip, which would
otherwise resolve every name against PyPI (an unrelated package with the same name =
dependency confusion): git and url sources become PEP 508 direct references
(`name @ git+https://...@rev`), a path source inside the project becomes `./<path>` (it
ships in the bundle; pip runs from code/), an index source keeps the name with a warning.
Path sources outside the project, workspace members and multi-source lists cannot be
reproduced by pip: DepsError, with `uv export` as the fix.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Collection
    from pathlib import Path

    from gpu_router.models import DepsSpec


class DepsError(ValueError):
    """An explicitly requested deps file is missing or unreadable."""


@dataclass(frozen=True, slots=True)
class DepsInfo:
    kind: Literal["requirements", "pyproject", "none"]
    file: str | None = None
    packages: list[str] = field(default_factory=list)
    python_requires: str | None = None
    warnings: list[str] = field(default_factory=list)

    def to_manifest(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "file": self.file,
            "packages": list(self.packages),
            "python_requires": self.python_requires,
        }


def _read_requirements(path: Path) -> list[str]:
    """Plain requirement lines (comments/blank dropped; -r/-e/options kept verbatim)."""
    out: list[str] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.split(" #", 1)[0].strip()
        if line and not line.startswith("#"):
            out.append(line)
    return out


_REQ = re.compile(
    r"^\s*(?P<name>[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?)\s*"
    r"(?P<extras>\[[^\]]*\])?\s*(?P<rest>[^;]*?)\s*(?:;\s*(?P<marker>.*))?$"
)
_EXPORT_HINT = (
    "export pinned requirements instead: `uv export --no-hashes --no-dev -o "
    "requirements.txt`, then `deps: requirements.txt` in gpu.yaml"
)


def _norm(name: str) -> str:
    """PEP 503 normalized project name."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _read_pyproject(path: Path) -> tuple[list[str], str | None, list[str]]:
    """(requirements for pip, requires-python, warnings)."""
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    project = data.get("project") or {}
    deps = project.get("dependencies") or []
    if not isinstance(deps, list):
        raise DepsError(f"{path.name}: [project].dependencies must be a list")
    requires = project.get("requires-python")
    tool = data.get("tool") or {}
    uv = tool.get("uv") if isinstance(tool, dict) else None
    sources = uv.get("sources") if isinstance(uv, dict) else None
    indexes = uv.get("index") if isinstance(uv, dict) else None
    reqs, warnings = _apply_uv_sources(
        path, [str(d) for d in deps], sources if isinstance(sources, dict) else {}, indexes
    )
    return reqs, str(requires) if requires else None, warnings


def _apply_uv_sources(
    path: Path, deps: list[str], sources: dict[str, Any], indexes: Any
) -> tuple[list[str], list[str]]:
    by_name = {_norm(str(k)): v for k, v in sources.items()}
    index_urls: dict[str, str] = {}
    if isinstance(indexes, list):
        for ix in indexes:
            if isinstance(ix, dict) and ix.get("name") and ix.get("url"):
                index_urls[str(ix["name"])] = str(ix["url"])
    out: list[str] = []
    warnings: list[str] = []
    for dep in deps:
        m = _REQ.match(dep)
        if m is None or _norm(m["name"]) not in by_name:
            out.append(dep)
            continue
        name, extras, marker = m["name"], m["extras"] or "", m["marker"]
        src = by_name[_norm(name)]
        where = f"{path.name}: [tool.uv.sources] {name}"
        if not isinstance(src, dict):
            raise DepsError(
                f"{where} lists several sources, which pip cannot reproduce; {_EXPORT_HINT}"
            )
        ref: str | None = None
        if "git" in src:
            ref = str(src["git"])
            ref = ref if ref.startswith("git+") else f"git+{ref}"
            for key in ("rev", "tag", "branch"):
                if src.get(key):
                    ref += f"@{src[key]}"
                    break
            if src.get("subdirectory"):
                ref += f"#subdirectory={src['subdirectory']}"
        elif "url" in src:
            ref = str(src["url"])
            if src.get("subdirectory"):
                ref += f"#subdirectory={src['subdirectory']}"
        elif "path" in src:
            rel = PurePosixPath(str(src["path"]))
            base = path.parent
            target = (base / str(rel)).resolve()
            project_root = base.resolve()
            if rel.is_absolute() or not target.is_relative_to(project_root):
                raise DepsError(
                    f"{where} is a path outside the project ({src['path']}), which is not "
                    f"in the bundle; {_EXPORT_HINT}"
                )
            out.append(f"./{target.relative_to(project_root).as_posix()}")
            continue
        elif src.get("workspace"):
            raise DepsError(
                f"{where} is a uv workspace member, which pip cannot install remotely; "
                f"{_EXPORT_HINT}"
            )
        elif "index" in src:
            url = index_urls.get(str(src["index"]), str(src["index"]))
            warnings.append(
                f"{name} comes from the uv index {src['index']!r} ({url}); the remote pip "
                "installs it from PyPI. use a requirements.txt with --extra-index-url to "
                "change that"
            )
            out.append(dep)
            continue
        else:
            raise DepsError(f"{where} has a source gpu-router does not understand; {_EXPORT_HINT}")
        out.append(f"{name}{extras} @ {ref}" + (f" ; {marker}" if marker else ""))
    return out, warnings


def detect_deps(project_dir: Path, spec: DepsSpec, bundled: Collection[str]) -> DepsInfo:
    """Resolve `spec` against the project. `bundled` = relative paths going into the bundle
    (a deps file that is not bundled cannot be installed remotely)."""
    if spec.kind == "none":
        return DepsInfo(kind="none")
    if spec.kind == "requirements":
        return _requirements(project_dir, spec.file or "requirements.txt", bundled)
    if spec.kind == "pyproject":
        return _pyproject(project_dir, spec.file or "pyproject.toml", bundled, explicit=True)
    # auto
    if spec.file is not None:
        if spec.file.endswith(".toml"):
            return _pyproject(project_dir, spec.file, bundled, explicit=True)
        return _requirements(project_dir, spec.file, bundled)
    if "requirements.txt" in bundled:
        return _requirements(project_dir, "requirements.txt", bundled)
    if "pyproject.toml" in bundled:
        return _pyproject(project_dir, "pyproject.toml", bundled, explicit=False)
    return DepsInfo(kind="none")


def _requirements(project_dir: Path, rel: str, bundled: Collection[str]) -> DepsInfo:
    path = project_dir / rel
    if not path.is_file():
        raise DepsError(f"deps file {rel} not found in {project_dir}")
    if rel not in bundled:
        raise DepsError(
            f"deps file {rel} is ignored by git, so it would not ship; "
            "commit it or remove it from .gitignore"
        )
    pkgs = _read_requirements(path)
    return DepsInfo(kind="requirements", file=rel, packages=pkgs)


def _pyproject(
    project_dir: Path, rel: str, bundled: Collection[str], *, explicit: bool
) -> DepsInfo:
    path = project_dir / rel
    if not path.is_file():
        raise DepsError(f"deps file {rel} not found in {project_dir}")
    try:
        pkgs, requires, warnings = _read_pyproject(path)
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        if explicit:
            raise DepsError(f"{rel} is not valid TOML: {exc}") from exc
        return DepsInfo(
            kind="none", warnings=[f"{rel} is not valid TOML ({exc}); installing no deps"]
        )
    if not pkgs:
        return DepsInfo(kind="none", python_requires=requires, warnings=warnings)
    return DepsInfo(
        kind="pyproject", file=rel, packages=pkgs, python_requires=requires, warnings=warnings
    )
