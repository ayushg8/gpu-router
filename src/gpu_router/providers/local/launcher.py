"""Detached launcher for one local run (phase 3). STDLIB ONLY.

It runs as `python -I launcher.py <run_dir> --lock-fd N` under the daemon's interpreter,
imports nothing from gpu_router and starts in milliseconds. adapter.py owns the other side
and imports the layout names below from this module.

The adapter creates `<home>/local/<remote_id>/`, takes an exclusive flock on `alive.lock`,
writes `launch.json` (non-secret plan) + `run.json` (the submit record), and starts this
file with the locked fd passed in (pass_fds), stdout/stderr on `console.log` and the job's
environment (secrets included, never on disk) inherited. Then:

 1. fork. The parent writes `pid.json` and exits at once (the adapter waits for it), so the
    run is re-parented to launchd and never lingers as a zombie of the daemon. The child
    calls setsid(): it leads its own session and process group (pgid == pid), so a daemon
    restart, a closed terminal or Ctrl-C never reaches it (invariant 11).
 2. phase "prepare": extract the bundle archive (or copy the bundle dir) into work/bundle.
 3. phase "env": `env: venv` (default) uses one python env per deps key under
    `<venvs_dir>/<key>/`, shared by every run with the same interpreter + dependencies and
    created and filled once under an flock (uv when available, else `python -m venv` +
    pip); phase "install" while installing. `env: system` uses the configured interpreter
    as it is and installs nothing.
 4. phase "run": exec the bundle's gpu_runner/bootstrap.py with that interpreter and
    --skip-install. exec keeps the pid, the process group and the alive.lock fd, so the
    lock is held exactly as long as the run lives. The adapter's liveness probe is a
    non-blocking shared flock on alive.lock: immune to pid reuse, no daemon memory needed.

A failure before bootstrap writes work/EXIT itself and prints the protocol lines bootstrap
would print: an env/install failure is `install_failed` + exit 90 (INSTALL_FAILED_EXIT,
D26: an environment failure, not a script failure); anything else is a runner error, exit 1.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any

# ---- run-dir layout (adapter.py imports these names)
LAUNCH_JSON = "launch.json"  # plan written by the adapter (no secrets)
RUN_JSON = "run.json"  # submit record; its presence is the commit point of a submit
PID_JSON = "pid.json"  # {"pid": n} of the detached run (launcher, then bootstrap)
ALIVE_LOCK = "alive.lock"  # flock held by the run for its whole life
PHASE_FILE = "phase"  # prepare | env | install | run
ENV_JSON = "env.json"  # which interpreter/venv the run uses (informational)
CANCEL_JSON = "cancel.json"  # written by adapter.cancel() while the run was alive
CONSOLE_LOG = "console.log"  # stdout+stderr of launcher and bootstrap: the run's log
WORK_DIR = "work"  # bootstrap --workdir: EXIT, job.log, bundle/, checkpoints/, outputs/
CKPT_SYNC_DIR = "ckpt-sync"  # bootstrap --checkpoint-sync-dir (file:// checkpoint archives)
EXIT_NAME = "EXIT"
BUNDLE_NAME = "bundle"
OUTPUTS_NAME = "outputs"
CHECKPOINTS_NAME = "checkpoints"
READY_MARKER = ".gpu-router-ready"  # in a venv: holds the deps key once fully installed

PHASE_PREPARE = "prepare"
PHASE_ENV = "env"
PHASE_INSTALL = "install"
PHASE_RUN = "run"

ENV_VENV = "venv"
ENV_SYSTEM = "system"

INSTALL_FAILED_EXIT = 90  # must equal runner/bootstrap.py INSTALL_FAILED_EXIT
RUNNER_ERROR_EXIT = 1
PREFIX = "::gpu:: "  # must equal gpu_router.protocol.PREFIX
LAUNCH_VERSION = 1
DEPS_KEY_VERSION = 1


class LaunchError(Exception):
    """The run cannot start. `code` becomes the EXIT code."""

    def __init__(
        self, message: str, code: int = RUNNER_ERROR_EXIT, install_rc: int | None = None
    ) -> None:
        super().__init__(message)
        self.code = code
        self.install_rc = install_rc


# --------------------------------------------------------------------------- output


def say(text: str) -> None:
    print("gpu-router: " + text, flush=True)


def event(t: str, **fields: Any) -> None:
    body: dict[str, Any] = {"t": t}
    body.update(fields)
    print(PREFIX + json.dumps(body, separators=(",", ":")), flush=True)


def write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def set_phase(run_dir: Path, phase: str) -> None:
    write_atomic(run_dir / PHASE_FILE, phase + "\n")


# --------------------------------------------------------------------------- steps


def prepare_bundle(plan: dict[str, Any], dest: Path) -> Path:
    """work/bundle from the archive (preferred: every attempt gets a clean copy) or the
    extracted bundle dir. Returns the bundle root (holds manifest.json)."""
    archive = plan.get("bundle_archive")
    source_dir = plan.get("bundle_dir")
    if dest.exists():
        shutil.rmtree(dest)
    if archive:
        say(f"unpacking {Path(archive).name}")
        dest.mkdir(parents=True)
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(dest, filter="data")  # the data filter refuses escaping members
    elif source_dir:
        say("copying the bundle")
        shutil.copytree(source_dir, dest, symlinks=True)
    else:
        raise LaunchError("launch.json names no bundle")
    if not (dest / "manifest.json").is_file():
        raise LaunchError(f"{dest} has no manifest.json")
    return dest


def deps_key(plan: dict[str, Any], manifest: dict[str, Any], code_dir: Path) -> str:
    """Same interpreter + same dependencies = same key = same venv (installed once)."""
    deps = manifest.get("deps") or {}
    kind = str(deps.get("kind") or "none")
    req_sha: str | None = None
    if kind == "requirements" and deps.get("file"):
        req = code_dir / str(deps["file"])
        req_sha = hashlib.sha256(req.read_bytes()).hexdigest() if req.is_file() else "missing"
    packages = sorted(str(p) for p in deps.get("packages") or []) if kind == "pyproject" else []
    doc = {
        "v": DEPS_KEY_VERSION,
        "python": plan["python"],
        "installer": "uv" if plan.get("uv") else "pip",
        "base": sorted(str(p) for p in plan.get("base_packages") or []),
        "kind": kind,
        "requirements": req_sha,
        "packages": packages,
    }
    return hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()[:16]


def install_args(plan: dict[str, Any], manifest: dict[str, Any], code_dir: Path) -> list[str]:
    """Installer arguments for base_packages + what the manifest recorded ([] = nothing)."""
    args = [str(p) for p in plan.get("base_packages") or []]
    deps = manifest.get("deps") or {}
    kind = deps.get("kind")
    if kind == "requirements" and deps.get("file"):
        req = code_dir / str(deps["file"])
        if req.is_file():
            args += ["-r", str(req)]
        else:
            say(f"requirements file {deps['file']} is missing from the bundle; skipping it")
    elif kind == "pyproject":
        args += [str(p) for p in deps.get("packages") or []]
    return args


def run_tool(cmd: list[str], cwd: Path) -> int:
    """Run an env/install command with output on our stdout (console.log)."""
    env = dict(os.environ)
    env.pop("VIRTUAL_ENV", None)
    env["UV_NO_PROGRESS"] = "1"
    try:
        return subprocess.run(
            cmd, cwd=cwd, env=env, stdin=subprocess.DEVNULL, check=False
        ).returncode
    except OSError as exc:
        say(f"could not start {cmd[0]}: {exc.strerror or exc}")
        return 127


def _ready(venv: Path, key: str) -> bool:
    try:
        return (venv / READY_MARKER).read_text(encoding="utf-8").strip() == key
    except OSError:
        return False


def ensure_venv(
    plan: dict[str, Any], run_dir: Path, venv: Path, key: str, args: list[str], code_dir: Path
) -> Path:
    """Create + fill the shared venv once (flock per venv). Returns its python."""
    python = venv / "bin" / "python"
    venv.parent.mkdir(parents=True, exist_ok=True)
    set_phase(run_dir, PHASE_ENV)
    with open(venv.parent / f"{venv.name}.lock", "a", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            say("waiting for another run that is setting up the same python env")
            fcntl.flock(lock, fcntl.LOCK_EX)
        if _ready(venv, key) and python.exists():
            say(f"reusing python env {venv.name}")
            return python
        if venv.exists() or venv.is_symlink():
            say(f"python env {venv.name} is incomplete; recreating it")
            shutil.rmtree(venv)
        uv = plan.get("uv")
        base = str(plan["python"])
        say(f"creating python env {venv.name} with {'uv' if uv else 'venv'} from {base}")
        cmd = [uv, "venv", "--python", base, str(venv)] if uv else [base, "-m", "venv", str(venv)]
        rc = run_tool(cmd, run_dir)
        if rc != 0:
            raise LaunchError(
                f"could not create the python env (exit {rc}); the job did not start",
                INSTALL_FAILED_EXIT,
                rc,
            )
        if args:
            set_phase(run_dir, PHASE_INSTALL)
            say(f"installing dependencies: {' '.join(args)}")
            if uv:
                cmd = [uv, "pip", "install", "--python", str(python), *args]
            else:
                cmd = [str(python), "-m", "pip", "install", "--disable-pip-version-check"]
                cmd += ["--no-input", *args]
            rc = run_tool(cmd, code_dir)  # requirements may name ./paths in the project
            if rc != 0:
                raise LaunchError(
                    f"dependency install failed (exit {rc}); the job did not start "
                    f"(exit {INSTALL_FAILED_EXIT} = install failed)",
                    INSTALL_FAILED_EXIT,
                    rc,
                )
        write_atomic(venv / READY_MARKER, key + "\n")
        return python


def resolve_python(
    plan: dict[str, Any], run_dir: Path, root: Path, manifest: dict[str, Any]
) -> tuple[str, dict[str, str]]:
    """(interpreter, extra env) for bootstrap."""
    code_dir = root / "code"
    args = install_args(plan, manifest, code_dir)
    if plan.get("env") == ENV_SYSTEM:
        python = str(plan["python"])
        if args:
            say("env: system, so dependencies are not installed; using the interpreter as it is")
        write_atomic(run_dir / ENV_JSON, json.dumps({"env": ENV_SYSTEM, "python": python}) + "\n")
        return python, {}
    key = deps_key(plan, manifest, code_dir)
    venv = Path(plan["venvs_dir"]) / key
    write_atomic(
        run_dir / ENV_JSON,
        json.dumps({"env": ENV_VENV, "python": plan["python"], "venv": str(venv), "key": key})
        + "\n",
    )
    venv_python = ensure_venv(plan, run_dir, venv, key, args, code_dir)
    path = os.environ.get("PATH", "")
    return str(venv_python), {
        "VIRTUAL_ENV": str(venv),
        "PATH": f"{venv / 'bin'}{os.pathsep}{path}" if path else str(venv / "bin"),
    }


def finish(exit_file: Path, code: int, message: str, install_rc: int | None) -> int:
    """A run that never reached bootstrap: same lines and EXIT file bootstrap would write."""
    say(message)
    if code == INSTALL_FAILED_EXIT:
        event("install_failed", code=install_rc if install_rc is not None else code)
    event("exit", code=code)
    try:
        exit_file.parent.mkdir(parents=True, exist_ok=True)
        write_atomic(exit_file, f"{code}\n")
    except OSError as exc:
        say(f"could not write {exit_file}: {exc.strerror or exc}")
    return code


def launch(run_dir: Path, lock_fd: int) -> int:
    """Everything after the fork. Returns only on failure (success execs bootstrap)."""
    work = run_dir / WORK_DIR
    exit_file = work / EXIT_NAME
    try:
        plan = json.loads((run_dir / LAUNCH_JSON).read_text(encoding="utf-8"))
        if plan.get("v") != LAUNCH_VERSION:
            raise LaunchError(f"launch.json version {plan.get('v')!r} is not {LAUNCH_VERSION}")
        say(f"local run {plan.get('remote_id')} started (pid {os.getpid()})")
        set_phase(run_dir, PHASE_PREPARE)
        work.mkdir(parents=True, exist_ok=True)
        root = prepare_bundle(plan, work / BUNDLE_NAME)
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        python, extra_env = resolve_python(plan, run_dir, root, manifest)
        bootstrap = root / "gpu_runner" / "bootstrap.py"
        if not bootstrap.is_file():
            fallback = plan.get("bootstrap_fallback")
            if not fallback or not Path(fallback).is_file():
                raise LaunchError("the bundle has no gpu_runner/bootstrap.py")
            say("the bundle has no gpu_runner/bootstrap.py; using the installed runner")
            bootstrap = Path(fallback)
        argv = [python, "-u", str(bootstrap), "--bundle", str(root), "--workdir", str(work)]
        argv += ["--skip-install", *[str(a) for a in plan.get("bootstrap_args") or []]]
        env = dict(os.environ)
        env.update(extra_env)
        set_phase(run_dir, PHASE_RUN)
        say(f"running bootstrap with {python}")
        os.set_inheritable(lock_fd, True)  # the lock must survive exec: it IS the liveness
        try:
            os.execve(python, argv, env)  # noqa: S606 - replaces this process, same pid
        except OSError as exc:
            raise LaunchError(
                f"could not start {python}: {exc.strerror or exc}; the job did not start",
                INSTALL_FAILED_EXIT,
            ) from exc
    except LaunchError as exc:
        return finish(exit_file, exc.code, str(exc), exc.install_rc)
    except Exception as exc:
        return finish(exit_file, RUNNER_ERROR_EXIT, f"runner error: {exc}", None)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 3 or args[1] != "--lock-fd" or not args[2].isdigit():
        print("usage: launcher.py <run_dir> --lock-fd N", file=sys.stderr)
        return 2
    run_dir = Path(args[0])
    lock_fd = int(args[2])
    pid = os.fork()
    if pid:
        try:
            write_atomic(run_dir / PID_JSON, json.dumps({"pid": pid}) + "\n")
        finally:
            os._exit(0)
    os.setsid()
    if not (run_dir / PID_JSON).exists():  # the parent's write failed or has not landed yet
        with contextlib.suppress(OSError):
            write_atomic(run_dir / PID_JSON, json.dumps({"pid": os.getpid()}) + "\n")
    code = launch(run_dir, lock_fd)
    sys.stdout.flush()
    os._exit(code)


if __name__ == "__main__":
    sys.exit(main())
