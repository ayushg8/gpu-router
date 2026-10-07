"""Which project files go into a bundle (phase 2).

Git projects: `git ls-files -co --exclude-standard` = tracked files plus untracked files that
.gitignore (and .git/info/exclude, the global excludes file) does not ignore. Untracked files
ship on purpose (a script an agent just wrote must run before anyone commits it), and every
untracked file that ships is named in a warning. Non-git projects (no `.git` in the project
dir or any parent): walk the directory with `DEFAULT_IGNORES` and warn. If the project IS
in a git repository but git fails (no Command Line Tools, "dubious ownership", timeout) the
bundle fails with git's message: falling back to the walk would ignore .gitignore and ship
files the user excluded.

In both modes files that look like credentials are never bundled (invariant 12) and each
exclusion is reported: `SECRET_PATTERNS` match the basename, `SECRET_DIRS` any directory on
the path, and symlinks are checked on their own name AND their target. Symlinks to files
inside the project are dereferenced; symlinks that resolve outside the project, directory
symlinks and submodules are skipped with a warning.

`include:` (gpu.yaml, `--include`, the MCP tools' `include`; D60) names paths or globs,
relative to the project root, that ship although git ignores them or never lists them (a
cloned `third_party/` repo, eval crops under an ignored `data/`). Matching files go
through every rule above; VCS dirs, virtualenvs, node_modules and caches inside included
trees are skipped; a pattern that goes through a symlink or matches nothing is a warning.
`left_out` lists, cheaply, the ignored paths that do not ship, for the bundle summary
agents get back.
"""

from __future__ import annotations

import fnmatch
import itertools
import os
import re
import subprocess
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Literal

GIT_TIMEOUT_S = 30.0

#: Used only when the project is not a git repository. Matched against every path component
#: (directories prune their whole subtree).
DEFAULT_IGNORES: tuple[str, ...] = (
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    "*.pyc",
    "*.pyo",
    ".venv",
    "venv",
    "env",
    ".tox",
    ".nox",
    "node_modules",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".ipynb_checkpoints",
    ".DS_Store",
    "*.egg-info",
    "dist",
    "build",
    "runs",
    "wandb",
    "mlruns",
    "lightning_logs",
    "checkpoints",
    "outputs",
    ".idea",
    ".vscode",
)

#: Never bundled, even if git tracks them (invariant 12: secrets never land in bundles).
#: Matched (case-insensitively) against the file name.
SECRET_PATTERNS: tuple[str, ...] = (
    # env files
    ".env",
    ".env.*",
    "*.env",
    ".envrc",
    # keys and certificates
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "*.jks",
    "*.keystore",
    "*.kdbx",
    "id_rsa*",
    "id_dsa*",
    "id_ed25519*",
    "id_ecdsa*",
    # tool credential files
    ".netrc",
    "_netrc",
    ".pypirc",
    ".npmrc",
    ".pgpass",
    ".git-credentials",
    ".htpasswd",
    ".boto",
    ".s3cfg",
    ".dockercfg",
    "kaggle.json",
    ".modal.toml",
    "credentials",
    "credentials.json",
    "credentials.toml",
    "credentials.yaml",
    "credentials.yml",
    "credentials.ini",
    "client_secret*.json",
    "service-account*.json",
    "application_default_credentials.json",
    "token.json",
    "token.pickle",
    "secrets.toml",
    "secrets.json",
    "secrets.yaml",
    "secrets.yml",
    # D48: Codex / Composer auth files, Claude Code's credential file, macOS keychains
    "auth.json",
    ".credentials.json",
    "*.keychain",
    "*.keychain-db",
)
#: Any file under one of these directories (anywhere on the path) is a credential store.
SECRET_DIRS: tuple[str, ...] = (
    ".aws",
    ".ssh",
    ".gnupg",
    ".kaggle",
    ".huggingface",
    ".azure",
    ".docker",
    ".kube",
    ".modal",
    ".codex",  # D48: Codex's auth.json
    ".lightning",
    ".password-store",
)
#: Multi-component credential dirs, matched as consecutive path parts.
SECRET_DIR_PAIRS: tuple[tuple[str, str], ...] = (
    (".config", "gcloud"),
    (".cache", "huggingface"),
    (".config", "gh"),
)
#: Credential stores in the home directory (D48). A dataset or project root that is one,
#: sits inside one, or CONTAINS one (`~/.config` holds gh's token, `~/Library` the
#: keychains) is never uploaded for an agent (mcp/tools, agent.py), and a dataset walk
#: never enters one (checkpoint/data.scan).
HOME_CREDENTIAL_STORES: tuple[str, ...] = (
    *SECRET_DIRS,
    ".config/gcloud",
    ".config/gh",
    ".config/hub",
    ".config/git/credentials",
    ".cache/huggingface",
    ".claude",
    ".claude.json",
    ".modal.toml",
    ".netrc",
    ".git-credentials",
    ".pypirc",
    ".npmrc",
    ".pgpass",
    ".local/share/keyrings",
    "Library/Keychains",
    "Library/Cookies",
    "Library/Application Support/gpu-router",
)
#: Templates of secret files are fine to ship.
SECRET_ALLOW: tuple[str, ...] = (
    ".env.example",
    ".env.sample",
    ".env.template",
    ".env.dist",
    "*.example",
    "*.sample",
    "*.template",
)


#: `include:` (D60): never shipped from an included tree: VCS metadata, virtualenvs (also
#: any dir holding a `pyvenv.cfg`), node_modules and tool caches. A cloned repo often has
#: its own .venv (the field test's third_party/ had one with 50k+ files).
VCS_DIRS = frozenset({".git", ".hg", ".svn"})
INCLUDE_SKIP_DIRS = VCS_DIRS | frozenset(
    {
        "__pycache__",
        ".venv",
        "venv",
        "node_modules",
        ".tox",
        ".nox",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".ipynb_checkpoints",
    }
)
INCLUDE_SKIP_FILES: tuple[str, ...] = ("*.pyc", "*.pyo", ".DS_Store")
#: An include walk visits at most this many files in all (a pattern over a dataset fails
#: fast with a hint to pass it as `data:` instead of walking millions of files).
MAX_INCLUDE_FILES = 50_000
MAX_INCLUDE_PATTERNS = 64
MAX_INCLUDE_CHARS = 300
_GLOB_CHARS = frozenset("*?[")

#: Ignored paths never named in the bundle summary: junk nobody wants on a GPU (venvs,
#: caches, build output, gpu-router's own runs/). Data-ish names (checkpoints, outputs,
#: data, weights) stay listed: those are what agents missed.
LEFT_OUT_JUNK: tuple[str, ...] = (
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    "*.pyc",
    "*.pyo",
    ".venv",
    "venv",
    ".tox",
    ".nox",
    "node_modules",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".ipynb_checkpoints",
    ".DS_Store",
    "*.egg-info",
    ".idea",
    ".vscode",
    "runs",
    "dist",
    "build",
    "wandb",
    "mlruns",
    "lightning_logs",
)
#: Entries shown in the summary; more are grouped by top-level dir, then counted.
MAX_LEFT_OUT = 8
#: Directory entries stat()ed per left-out path / in all to size it (else "N+ files").
LEFT_OUT_SIZE_BUDGET = 5_000
LEFT_OUT_TOTAL_BUDGET = 20_000


class GitError(RuntimeError):
    """The project is in a git repository but `git ls-files` failed. `hint` says how to
    fix it; packaging turns this into a BundleError (400 invalid_spec)."""

    def __init__(self, message: str, hint: str) -> None:
        super().__init__(message)
        self.hint = hint


class IncludeError(ValueError):
    """An `include:` pattern is invalid or covers too much; packaging turns it into a
    BundleError (400 invalid_spec) with `hint`."""

    def __init__(self, message: str, hint: str) -> None:
        super().__init__(message)
        self.hint = hint


@dataclass(frozen=True, slots=True)
class ProjectFile:
    rel: str  # POSIX path relative to the project dir
    path: Path  # absolute path on this machine (symlinks already resolved)
    size: int
    executable: bool


@dataclass(slots=True)
class FileSelection:
    files: list[ProjectFile]
    source: Literal["git", "walk"]
    warnings: list[str] = field(default_factory=list)
    excluded_secrets: list[str] = field(default_factory=list)
    untracked: list[str] = field(default_factory=list)  # git mode: shipped but not committed
    included: list[str] = field(default_factory=list)  # shipped only because of `include:`
    skipped: list[str] = field(default_factory=list)  # nested repos, dir symlinks, ...
    pruned: list[str] = field(default_factory=list)  # walk mode: dirs DEFAULT_IGNORES skipped

    @property
    def total_bytes(self) -> int:
        return sum(f.size for f in self.files)

    @property
    def included_bytes(self) -> int:
        shipped = set(self.included)
        return sum(f.size for f in self.files if f.rel in shipped)


@dataclass(frozen=True, slots=True)
class LeftOut:
    """A path that does not ship, for the bundle summary: `data/ (2.1 GB, ignored)`."""

    path: str  # relative; directories end with "/"
    why: str  # "ignored", "nested git repo", "3 paths inside left out", ...
    bytes: int | None = None  # None: not measured (more files than the size budget)
    files: int | None = None
    files_at_least: bool = False  # True: `files` is a lower bound (budget ran out)

    def text(self) -> str:
        if self.bytes is not None:
            return f"{self.path} ({human_bytes(self.bytes)}, {self.why})"
        if self.files is not None and self.files_at_least:
            return f"{self.path} ({self.why}, {self.files}+ files)"
        return f"{self.path} ({self.why})"


def human_bytes(n: int) -> str:
    """1536 -> '1.5 KB', 2_254_857_830 -> '2.1 GB'."""
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{int(size)} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"  # pragma: no cover


_SECRET_NAME_RE = re.compile("|".join(fnmatch.translate(p) for p in SECRET_PATTERNS))
_ALLOW_NAME_RE = re.compile("|".join(fnmatch.translate(p) for p in SECRET_ALLOW))


def secret_name(name: str) -> bool:
    """Does this file NAME (no directories) look like a credential file?"""
    low = name.lower()
    return _SECRET_NAME_RE.match(low) is not None and _ALLOW_NAME_RE.match(low) is None


def _secret_dirs(dirs: list[str]) -> bool:
    return any(d in SECRET_DIRS for d in dirs) or any(
        pair in SECRET_DIR_PAIRS for pair in itertools.pairwise(dirs)
    )


def looks_secret(rel: str) -> bool:
    """Does this (relative or absolute) path look like a credential file?"""
    p = PurePosixPath(rel.replace(os.sep, "/"))
    if _secret_dirs([part.lower() for part in p.parts[:-1]]):
        return True
    return secret_name(p.name)


def _inside(child: str, parent: str) -> bool:
    child, parent = child.lower(), parent.lower()
    return child == parent or child.startswith(parent.rstrip("/") + "/")


def _tilde(path: str, home: str) -> str:
    return "~" + path[len(home) :] if _inside(path, home) and len(path) > len(home) else path


def credential_path_problem(
    path: str | Path, *, home: str | Path | None = None, extra: tuple[str | Path, ...] = ()
) -> str | None:
    """Why an absolute path (file or directory, symlinks already resolved) must never be
    uploaded, or None: it is or sits inside a credential store (any SECRET_DIRS part,
    including its last one; a SECRET_DIR_PAIRS pair; a home store), it is named like a
    credential file, or it CONTAINS a home credential store (`~/.config`, `~/Library`).
    `extra` adds stores (the gpu-router data dir in use). Case-insensitive, like APFS."""
    p = PurePosixPath(str(path).replace(os.sep, "/"))
    text = p.as_posix()
    home_text = os.path.realpath(os.path.expanduser(str(home) if home else "~"))
    home_text = home_text.replace(os.sep, "/")
    shown = _tilde(text, home_text)
    parts = [part.lower() for part in p.parts]
    for part in parts:
        if part in SECRET_DIRS:
            return f"{shown} is inside a credential store ({part})"
    for a, b in itertools.pairwise(parts):
        if (a, b) in SECRET_DIR_PAIRS:
            return f"{shown} is inside a credential store ({a}/{b})"
    if parts and secret_name(parts[-1]):
        return f"{shown} looks like a credential file"
    stores = [f"{home_text}/{store}" for store in HOME_CREDENTIAL_STORES]
    stores += [os.path.realpath(str(x)).replace(os.sep, "/") for x in extra]
    for store in stores:
        if _inside(text, store):
            return f"{shown} is inside the credential store {_tilde(store, home_text)}"
        if _inside(store, text):
            return f"{shown} contains the credential store {_tilde(store, home_text)}"
    return None


def _ignored(name: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatchcase(name, pat) for pat in patterns)


def _in_git_repo(project: Path) -> bool:
    return any((d / ".git").exists() for d in (project, *project.parents))


def _git_ls_files(project: Path) -> tuple[list[str], set[str]] | None:
    """(relative paths, the untracked subset) from git, or None if `project` is not in a
    git repository. Raises GitError when it is one and git fails."""
    if not _in_git_repo(project):
        return None
    try:
        proc = subprocess.run(
            ["git", "-C", str(project), "ls-files", "-co", "--exclude-standard", "-t", "-z"],
            capture_output=True,
            timeout=GIT_TIMEOUT_S,
            check=False,
        )
    except subprocess.TimeoutExpired:
        raise GitError(
            f"`git ls-files` took longer than {GIT_TIMEOUT_S:g}s in {project}",
            hint="a huge untracked tree slows git down: add it to .gitignore",
        ) from None
    except OSError as exc:
        raise GitError(
            f"{project} is in a git repository but git could not run ({exc.strerror or exc})",
            hint="install git (on macOS: `xcode-select --install`)",
        ) from None
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", "replace").strip()
        lower = err.lower()
        if "not a git repository" in lower:
            return None  # a stray `.git` that is not a repository: treat as plain dir
        hint = "fix the git problem above, or run `git status` in the project to see it"
        if "dubious ownership" in lower:
            hint = f"trust the repository: git config --global --add safe.directory {project}"
        elif "xcrun" in lower or "developer tools" in lower or "active developer path" in lower:
            hint = "install the Command Line Tools: xcode-select --install"
        first = err.splitlines()[0] if err else f"exit {proc.returncode}"
        raise GitError(f"`git ls-files` failed in {project}: {first}", hint=hint)
    rels: set[str] = set()
    untracked: set[str] = set()
    for entry in proc.stdout.decode("utf-8", "surrogateescape").split("\0"):
        if len(entry) < 3:
            continue
        tag, rel = entry[0], entry[2:]
        rels.add(rel)
        if tag == "?":
            untracked.add(rel)
    return sorted(rels), untracked


def _walk(project: Path) -> tuple[list[str], list[str], list[str]]:
    """(files, directory symlinks that were not followed, dirs DEFAULT_IGNORES pruned)."""
    out: list[str] = []
    dir_links: list[str] = []
    pruned: list[str] = []
    for root, dirs, files in os.walk(project, followlinks=False):
        base = Path(root).relative_to(project)

        def rel(name: str, base: Path = base) -> str:
            return (base / name).as_posix() if str(base) != "." else name

        kept = []
        for d in sorted(dirs):
            if _ignored(d, DEFAULT_IGNORES):
                pruned.append(rel(d) + "/")
                continue
            if os.path.islink(os.path.join(root, d)):
                dir_links.append(rel(d))  # os.walk lists it but never descends
                continue
            kept.append(d)
        dirs[:] = kept
        for name in files:
            if _ignored(name, DEFAULT_IGNORES):
                continue
            out.append(rel(name))
    return sorted(out), sorted(dir_links), sorted(pruned)


def _listing(items: list[str], limit: int = 5) -> str:
    return ", ".join(items[:limit]) + (" ..." if len(items) > limit else "")


# --------------------------------------------------------------------------- include (D60)


def normalize_include(pattern: str) -> str:
    """An `include:` entry in canonical form ("./a//b/" -> "a/b/"; a trailing "/" means
    directories only). Raises ValueError (the message quotes the entry) when it is empty,
    absolute, starts with ~, climbs out with "..", or names the whole project."""
    if not isinstance(pattern, str):
        raise ValueError(f"include entries are strings, got {pattern!r}")
    text = pattern.strip()
    if not text:
        raise ValueError("an include entry is empty")
    if len(text) > MAX_INCLUDE_CHARS:
        raise ValueError(f"include entry {text[:40]!r}... is over {MAX_INCLUDE_CHARS} chars")
    if any(ch in text for ch in "\x00\n\r"):
        raise ValueError(f"include entry {text!r} contains a control character")
    if text.startswith(("/", "~")):
        raise ValueError(
            f"include entry {text!r} must be relative to the project root (no absolute or ~ paths)"
        )
    parts = [p for p in text.split("/") if p not in ("", ".")]
    if ".." in parts:
        raise ValueError(f"include entry {text!r} must stay inside the project (no '..')")
    if not parts or all(p == "**" for p in parts):
        raise ValueError(
            f"include entry {text!r} would ship every ignored file; name the paths you need"
        )
    return "/".join(parts) + ("/" if text.endswith("/") else "")


def _segment_regex(seg: str) -> str:
    """One path segment of a glob: `*` and `?` never cross "/"; `[...]` classes."""
    out: list[str] = []
    i, n = 0, len(seg)
    while i < n:
        c = seg[i]
        i += 1
        if c == "*":
            while i < n and seg[i] == "*":
                i += 1
            out.append("[^/]*")
        elif c == "?":
            out.append("[^/]")
        elif c == "[":
            j = i
            if j < n and seg[j] in "!^":
                j += 1
            if j < n and seg[j] == "]":
                j += 1
            while j < n and seg[j] != "]":
                j += 1
            if j >= n:
                out.append(re.escape(c))
                continue
            body = seg[i:j].replace("\\", "\\\\")
            negate = body[:1] in ("!", "^")
            body = body[1:] if negate else body
            out.append(f"[^/{body}]" if negate else f"[{body}]")
            i = j + 1
        else:
            out.append(re.escape(c))
    return "".join(out)


def include_regex(norm: str) -> re.Pattern[str]:
    """A normalized include entry as a regex over project-relative POSIX paths: `**` is
    any number of directories, a trailing `**` everything below."""
    parts = norm.rstrip("/").split("/")
    out: list[str] = []
    for i, seg in enumerate(parts):
        last = i == len(parts) - 1
        if seg == "**":
            out.append(".*" if last else "(?:[^/]+/)*")
        else:
            out.append(_segment_regex(seg) + ("" if last else "/"))
    return re.compile("".join(out), re.DOTALL)


def _has_glob(seg: str) -> bool:
    return any(ch in _GLOB_CHARS for ch in seg)


def _dir_filter(parts: list[str]) -> Callable[[str], bool]:
    """Can a directory (project-relative) hold a match? Its first segments must match the
    pattern's, up to the first `**`; deeper dirs of a match are kept (a matching directory
    ships whole). So `*.ckpt` never walks into data/."""
    stop = parts.index("**") if "**" in parts else len(parts)
    seg_rx = [re.compile(_segment_regex(seg), re.DOTALL) for seg in parts[:stop]]

    def keep(drel: str) -> bool:
        segs = drel.split("/")
        return all(rx.fullmatch(seg) for rx, seg in zip(seg_rx, segs, strict=False))

    return keep


@dataclass(slots=True)
class _IncludeScan:
    """What the include patterns matched (relative paths), before the shipping rules."""

    matches: dict[str, str] = field(default_factory=dict)  # rel -> first pattern matching it
    counts: dict[str, int] = field(default_factory=dict)  # pattern -> files matched
    dir_links: list[str] = field(default_factory=list)
    secret_dirs: list[str] = field(default_factory=list)
    unreadable: list[str] = field(default_factory=list)
    through_links: dict[str, str] = field(default_factory=dict)  # pattern -> symlink rel
    vcs: dict[str, str] = field(default_factory=dict)  # pattern -> the VCS dir it names
    visited: int = 0


def _include_walk(
    project: Path,
    base: Path,
    scan: _IncludeScan,
    keep_dir: Callable[[str], bool] | None = None,
) -> Iterator[str]:
    """Every file under `base` (a real directory inside the project) as a project-relative
    path: no symlinked dirs followed, VCS dirs / caches skipped, credential dirs pruned
    (and reported), dirs `keep_dir` rules out not entered, at most MAX_INCLUDE_FILES
    visited in all."""

    def on_error(exc: OSError) -> None:
        where = Path(exc.filename) if exc.filename else base
        if where.is_relative_to(project):
            scan.unreadable.append(where.relative_to(project).as_posix() + "/")

    for root, dirs, files in os.walk(base, followlinks=False, onerror=on_error):
        rel_root = Path(root).relative_to(project).as_posix()
        prefix = "" if rel_root == "." else rel_root + "/"
        kept: list[str] = []
        for d in sorted(dirs):
            if d in INCLUDE_SKIP_DIRS or os.path.isfile(os.path.join(root, d, "pyvenv.cfg")):
                continue
            drel = prefix + d
            if keep_dir is not None and not keep_dir(drel):
                continue
            if os.path.islink(os.path.join(root, d)):
                scan.dir_links.append(drel)
                continue
            if _secret_dirs([part.lower() for part in PurePosixPath(drel).parts]):
                scan.secret_dirs.append(drel + "/")
                continue
            kept.append(d)
        dirs[:] = kept
        for name in sorted(files):
            if _ignored(name, INCLUDE_SKIP_FILES):
                continue
            scan.visited += 1
            if scan.visited > MAX_INCLUDE_FILES:
                raise IncludeError(
                    f"`include:` covers more than {MAX_INCLUDE_FILES} files "
                    f"(stopped in {prefix or './'})",
                    hint="name narrower paths or globs; pass datasets with `data:` "
                    "(uploaded once, cached) instead of shipping them with the code",
                )
            yield prefix + name


def _scan_include(project: Path, patterns: Sequence[str]) -> _IncludeScan:
    scan = _IncludeScan()
    for raw in patterns:
        try:
            norm = normalize_include(raw)
        except ValueError as exc:
            raise IncludeError(
                str(exc), hint="include entries are paths or globs inside the project"
            ) from None
        if norm in scan.counts:
            continue
        scan.counts[norm] = 0
        dir_only = norm.endswith("/")
        parts = norm.rstrip("/").split("/")
        static: list[str] = []
        for seg in parts:
            if _has_glob(seg):
                break
            static.append(seg)
        globbed = len(static) < len(parts)
        # every directory we pass through must be real (a symlink could lead anywhere)
        through = static if globbed else static[:-1]
        link = next(
            (
                "/".join(through[: k + 1])
                for k in range(len(through))
                if (project / "/".join(through[: k + 1])).is_symlink()
            ),
            None,
        )
        if link is not None:
            scan.through_links[norm] = link
            continue
        if any(seg in VCS_DIRS for seg in static):
            scan.vcs[norm] = next(seg for seg in static if seg in VCS_DIRS)
            continue  # .git/config can hold credentials in remote URLs: never ships
        base = project.joinpath(*static) if static else project
        found: list[str] = []
        if not globbed:
            if base.is_dir() and not base.is_symlink():
                found = list(_include_walk(project, base, scan))
            elif os.path.lexists(base) and not dir_only:
                found = ["/".join(static)]  # a file (or a file symlink: checked later)
        elif base.is_dir():
            rx = include_regex(norm)
            depth = len(static)
            for rel in _include_walk(project, base, scan, _dir_filter(parts)):
                rparts = rel.split("/")
                # the file itself, or one of its directories below the static prefix
                cands = [] if dir_only else [rel]
                cands += ["/".join(rparts[:k]) for k in range(depth + 1, len(rparts))]
                if any(rx.fullmatch(c) for c in cands):
                    found.append(rel)
        scan.counts[norm] = len(found)
        for rel in found:
            scan.matches.setdefault(rel, norm)
    return scan


# --------------------------------------------------------------------------- selection

_Kind = Literal["ok", "secret", "outside", "skipped", "gone"]


def _classify(project: Path, rel: str, root_secret: bool) -> tuple[_Kind, ProjectFile | None]:
    """The shipping rules for one listed path (git, walk or include)."""
    if root_secret or looks_secret(rel) or looks_secret((project / rel).as_posix()):
        return "secret", None
    path = project / rel
    try:
        if path.is_symlink():
            target = path.resolve(strict=True)
            inside = target.is_relative_to(project)
            where = target.relative_to(project).as_posix() if inside else str(target)
            if looks_secret(where):
                return "secret", None  # an innocent name for a credential file
            if not inside:
                return "outside", None
            if not target.is_file():
                return "skipped", None
            path = target
        elif not path.is_file():
            # deleted-but-tracked files, submodule gitlinks, sockets: nothing to ship
            return ("skipped" if path.is_dir() else "gone"), None
        st = path.stat()
    except (OSError, RuntimeError):
        return "skipped", None
    return "ok", ProjectFile(
        rel=rel.rstrip("/"), path=path, size=st.st_size, executable=bool(st.st_mode & 0o111)
    )


def _include_warnings(
    scan: _IncludeScan, added: dict[str, ProjectFile], files_by_pattern: dict[str, int]
) -> list[str]:
    out: list[str] = []
    if added:
        size = sum(f.size for f in added.values())
        # sorted, so the warning (and the manifest holding it) ignores the pattern order
        per = ", ".join(f"{p} ({n})" for p, n in sorted(files_by_pattern.items()) if n)
        out.append(
            f"include: shipping {len(added)} file(s), {human_bytes(size)}, that git ignores "
            f"or does not list: {per}"
        )
    for pattern, link in sorted(scan.through_links.items()):
        out.append(
            f"include `{pattern}` goes through the symlink {link}; name the real path "
            "inside the project"
        )
    for pattern, vcs in sorted(scan.vcs.items()):
        out.append(f"include `{pattern}` names {vcs}/, which never ships")
    named = {*scan.through_links, *scan.vcs}
    empty = sorted(p for p, n in scan.counts.items() if n == 0 and p not in named)
    if empty:
        out.append(
            f"include matched no files: {_listing(empty)} (paths are relative to the project root)"
        )
    if scan.unreadable:
        out.append(f"include: could not read {_listing(sorted(set(scan.unreadable)))}")
    return out


def select_files(project_dir: str | Path, include: Sequence[str] = ()) -> FileSelection:
    """List the files to bundle: git's listing (or the walk) plus whatever `include`
    matches. Raises FileNotFoundError / NotADirectoryError for a bad dir, GitError when the
    project is in a git repository and git fails, IncludeError for a bad include entry."""
    project = Path(project_dir)
    if not project.exists():
        raise FileNotFoundError(str(project))
    if not project.is_dir():
        raise NotADirectoryError(str(project))
    project = project.resolve()
    warnings: list[str] = []
    skipped: list[str] = []
    pruned: list[str] = []
    listed = _git_ls_files(project)
    source: Literal["git", "walk"] = "git"
    untracked: set[str] = set()
    if listed is None:
        source = "walk"
        rels, dir_links, pruned = _walk(project)
        skipped.extend(dir_links)
        warnings.append(
            "project is not a git repository; bundling every file except common junk "
            "(.venv, __pycache__, runs/, checkpoints/, ...). run `git init` and add a "
            ".gitignore to control exactly what ships"
        )
    else:
        rels, untracked = listed
    files: list[ProjectFile] = []
    secrets: list[str] = []
    outside: list[str] = []
    # the project's own location counts too: a root at ~/.config ships gh/hosts.yml
    # otherwise (every relative path looks innocent there, D48)
    root_secret = _secret_dirs([part.lower() for part in project.parts])
    for rel in rels:
        kind, pf = _classify(project, rel, root_secret)
        if kind == "secret":
            secrets.append(rel)
        elif kind == "outside":
            outside.append(rel)
        elif kind == "skipped":
            skipped.append(rel)
        elif pf is not None:
            files.append(pf)
    shipped_untracked = sorted(f.rel for f in files if f.rel in untracked)

    added: dict[str, ProjectFile] = {}
    by_pattern: dict[str, int] = {}
    scan = _scan_include(project, include) if include else _IncludeScan()
    if include:
        have = {f.rel for f in files}
        by_pattern = dict.fromkeys(scan.counts, 0)
        for rel, pattern in sorted(scan.matches.items()):
            if rel in have:
                continue  # git ships it anyway
            kind, pf = _classify(project, rel, root_secret)
            if kind == "secret":
                secrets.append(rel)
            elif kind == "outside":
                outside.append(rel)
            elif kind == "skipped":
                skipped.append(rel)
            elif pf is not None:
                added[pf.rel] = pf
                by_pattern[pattern] += 1
        secrets.extend(scan.secret_dirs)
        skipped.extend(scan.dir_links)
        if added:
            files = sorted([*files, *added.values()], key=lambda f: f.rel)
            # a nested repo / directory git only listed as a whole now ships through include
            skipped = [
                s for s in skipped if not any(r.startswith(s.rstrip("/") + "/") for r in added)
            ]

    if shipped_untracked:
        warnings.append(
            f"shipping {len(shipped_untracked)} untracked file(s) that git does not ignore: "
            f"{_listing(shipped_untracked)}. add them to .gitignore to leave them out"
        )
    if secrets:
        warnings.append(
            f"left out {len(secrets)} file(s) that look like credentials: "
            f"{_listing(secrets)}. pass secrets with `secrets:` instead"
        )
    if outside:
        warnings.append(
            f"left out {len(outside)} symlink(s) that point outside the project: "
            f"{_listing(outside)}. copy the file into the project to ship it"
        )
    if skipped:
        warnings.append(
            f"skipped {len(skipped)} path(s) that are not regular files (submodules, nested "
            f"git repos, broken or directory symlinks): {_listing(sorted(skipped))}"
            + ("" if include else ". name a nested repo in `include:` to ship its files")
        )
    if include:
        warnings.extend(_include_warnings(scan, added, by_pattern))
    return FileSelection(
        files=files,
        source=source,
        warnings=warnings,
        excluded_secrets=secrets,
        untracked=shipped_untracked,
        included=sorted(added),
        skipped=sorted(skipped),
        pruned=pruned,
    )


# --------------------------------------------------------------------------- left out (D60)


def _git_ignored(project: Path) -> list[str]:
    """Ignored paths as git reports them, whole ignored directories collapsed ("data/"),
    never descended into. Best effort: [] when git is missing or fails."""
    try:
        proc = subprocess.run(
            [
                "git",
                "-C",
                str(project),
                "ls-files",
                "-o",
                "-i",
                "--exclude-standard",
                "--directory",
                "-z",
            ],
            capture_output=True,
            timeout=GIT_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if proc.returncode != 0:
        return []
    entries = sorted(e for e in proc.stdout.decode("utf-8", "surrogateescape").split("\0") if e)
    out: list[str] = []
    for entry in entries:
        # git also lists a dir whose content is all ignored AND its ignored subdirs
        if out and out[-1].endswith("/") and entry.startswith(out[-1]):
            continue
        out.append(entry)
    return out


def _junk(rel: str) -> bool:
    return any(_ignored(part, LEFT_OUT_JUNK) for part in PurePosixPath(rel).parts)


def _hidden_secret(rel: str) -> bool:
    parts = [part.lower() for part in PurePosixPath(rel.rstrip("/")).parts]
    if rel.endswith("/"):
        return _secret_dirs(parts)
    return looks_secret(rel)


def _tree_size(path: Path, budget: list[int]) -> tuple[int, int, bool]:
    """(bytes, files, complete) under `path`, stat()ing at most budget[0] entries (the
    shared budget is decremented). Symlinks are not followed."""
    if not path.is_dir() or path.is_symlink():
        try:
            return path.lstat().st_size, 1, True
        except OSError:
            return 0, 0, True
    total = files = 0
    stack = [path]
    local = LEFT_OUT_SIZE_BUDGET
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as it:
                for entry in it:
                    if local <= 0 or budget[0] <= 0:
                        return total, files, False
                    local -= 1
                    budget[0] -= 1
                    try:  # counted as include would ship it: no .git, no credentials
                        if entry.is_dir(follow_symlinks=False):
                            low = entry.name.lower()
                            if (
                                entry.name not in INCLUDE_SKIP_DIRS
                                and low not in SECRET_DIRS
                                and not os.path.isfile(os.path.join(entry.path, "pyvenv.cfg"))
                            ):
                                stack.append(Path(entry.path))
                        elif not secret_name(entry.name):
                            total += entry.stat(follow_symlinks=False).st_size
                            files += 1
                    except OSError:
                        continue
        except OSError:
            continue
    return total, files, True


def left_out(
    project_dir: str | Path,
    sel: FileSelection,
    include: Sequence[str] = (),
    *,
    limit: int = MAX_LEFT_OUT,
) -> tuple[list[LeftOut], int]:
    """(paths that do not ship, how many more were not shown), for agents: git-ignored
    paths (whole dirs collapsed, never walked), nested repos and directory symlinks git
    lists only as a whole, and in a non-git project the dirs the walk skips by default.
    Junk (venvs, caches, runs/) and credential-looking paths are never named; a path an
    include entry names whole is not left out. Sizes are measured within a small budget.
    More than `limit` paths are grouped by top-level directory."""
    project = Path(project_dir).resolve()
    candidates: dict[str, str] = {}
    if sel.source == "git":
        for entry in _git_ignored(project):
            candidates[entry] = "ignored"
    else:
        for entry in sel.pruned:
            candidates[entry] = "skipped by default"
    for rel in sel.skipped:
        entry = rel.rstrip("/")
        path = project / entry
        if path.is_symlink():
            if path.is_dir():
                candidates[entry + "/"] = "directory symlink"
        elif (path / ".git").exists():
            candidates[entry + "/"] = "nested git repo"
        elif path.is_dir():
            candidates[entry + "/"] = "not shipped"
    patterns = [
        (include_regex(n), n.endswith("/"))
        for n in (normalize_include(p) for p in include if _valid_include(p))
    ]

    def whole(entry: str) -> bool:
        """Does an include entry match this path or a directory above it (= ships it all)?"""
        parts = entry.rstrip("/").split("/")
        for rx, dir_only in patterns:
            for k in range(1, len(parts) + 1):
                if dir_only and k == len(parts) and not entry.endswith("/"):
                    continue
                if rx.fullmatch("/".join(parts[:k])):
                    return True
        return False

    shipped_dirs: set[str] = set()
    for f in sel.files:
        parts = f.rel.split("/")
        shipped_dirs.update("/".join(parts[:k]) + "/" for k in range(1, len(parts)))
    items: list[tuple[str, str]] = []
    for entry, why in sorted(candidates.items()):
        if _junk(entry) or _hidden_secret(entry):
            continue
        if whole(entry):
            continue  # include ships it whole
        if entry.endswith("/") and entry in shipped_dirs:
            why += "; part ships via include"
        items.append((entry, why))

    grouped: list[tuple[str, str, list[str]]] = []
    if len(items) > limit:  # group by top-level directory
        groups: dict[str, list[tuple[str, str]]] = {}
        for entry, why in items:
            top = entry.split("/", 1)[0] + ("/" if "/" in entry else "")
            groups.setdefault(top, []).append((entry, why))
        for top, members in groups.items():
            if len(members) == 1:
                grouped.append((members[0][0], members[0][1], [members[0][0]]))
            else:
                grouped.append(
                    (top, f"{len(members)} paths inside left out", [m[0] for m in members])
                )
    else:
        grouped = [(entry, why, [entry]) for entry, why in items]

    budget = [LEFT_OUT_TOTAL_BUDGET]
    shown: list[LeftOut] = []
    for shown_path, why, paths in grouped[:limit]:
        total = count = 0
        complete = True
        for rel in paths:
            b, n, done = _tree_size(project / rel.rstrip("/"), budget)
            total, count = total + b, count + n
            complete = complete and done
        shown.append(
            LeftOut(
                path=shown_path,
                why=why,
                bytes=total if complete else None,
                files=count,
                files_at_least=not complete,
            )
        )
    return shown, max(0, len(grouped) - limit)


def _valid_include(pattern: str) -> bool:
    try:
        normalize_include(pattern)
    except ValueError:
        return False
    return True
