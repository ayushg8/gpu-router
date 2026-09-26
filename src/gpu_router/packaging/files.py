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
"""

from __future__ import annotations

import fnmatch
import itertools
import os
import re
import subprocess
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


class GitError(RuntimeError):
    """The project is in a git repository but `git ls-files` failed. `hint` says how to
    fix it; packaging turns this into a BundleError (400 invalid_spec)."""

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

    @property
    def total_bytes(self) -> int:
        return sum(f.size for f in self.files)


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


def _walk(project: Path) -> tuple[list[str], list[str]]:
    """(files, directory symlinks that were not followed)."""
    out: list[str] = []
    dir_links: list[str] = []
    for root, dirs, files in os.walk(project, followlinks=False):
        base = Path(root).relative_to(project)

        def rel(name: str, base: Path = base) -> str:
            return (base / name).as_posix() if str(base) != "." else name

        kept = []
        for d in sorted(dirs):
            if _ignored(d, DEFAULT_IGNORES):
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
    return sorted(out), sorted(dir_links)


def _listing(items: list[str], limit: int = 5) -> str:
    return ", ".join(items[:limit]) + (" ..." if len(items) > limit else "")


def select_files(project_dir: str | Path) -> FileSelection:
    """List the files to bundle. Raises FileNotFoundError / NotADirectoryError for a bad dir,
    GitError when the project is in a git repository and git fails."""
    project = Path(project_dir)
    if not project.exists():
        raise FileNotFoundError(str(project))
    if not project.is_dir():
        raise NotADirectoryError(str(project))
    project = project.resolve()
    warnings: list[str] = []
    skipped: list[str] = []
    listed = _git_ls_files(project)
    source: Literal["git", "walk"] = "git"
    untracked: set[str] = set()
    if listed is None:
        source = "walk"
        rels, dir_links = _walk(project)
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
        if root_secret or looks_secret(rel) or looks_secret((project / rel).as_posix()):
            secrets.append(rel)
            continue
        path = project / rel
        try:
            if path.is_symlink():
                target = path.resolve(strict=True)
                inside = target.is_relative_to(project)
                where = target.relative_to(project).as_posix() if inside else str(target)
                if looks_secret(where):
                    secrets.append(rel)  # an innocent name for a credential file
                    continue
                if not inside:
                    outside.append(rel)
                    continue
                if not target.is_file():
                    skipped.append(rel)
                    continue
                path = target
            elif not path.is_file():
                # deleted-but-tracked files, submodule gitlinks, sockets: nothing to ship
                if path.is_dir():
                    skipped.append(rel)
                continue
            st = path.stat()
        except (OSError, RuntimeError):
            skipped.append(rel)
            continue
        files.append(
            ProjectFile(rel=rel, path=path, size=st.st_size, executable=bool(st.st_mode & 0o111))
        )
    shipped_untracked = sorted(f.rel for f in files if f.rel in untracked)
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
            f"skipped {len(skipped)} path(s) that are not regular files (submodules, broken "
            f"or directory symlinks): {_listing(sorted(skipped))}"
        )
    return FileSelection(
        files=files,
        source=source,
        warnings=warnings,
        excluded_secrets=secrets,
        untracked=shipped_untracked,
    )
