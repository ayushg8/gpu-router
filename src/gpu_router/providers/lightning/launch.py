"""gpu-router's launcher inside a Lightning AI Studio job (uploaded as-is, one per attempt).

The adapter uploads this file with `launch.json` (non-secret parameters), the job bundle
and, when needed, `resume.tar.gz` and `secrets.json` to the teamspace drive folder
`uploads/gpu-router/<job name>/`. The job's command finds that folder (the drive mount
`/teamspace/uploads/...`, else a copy the command downloads with the Studio's own SDK) and
runs `python launch.py`. Here it:

1. copies the bundle into a private work dir (sha256-checked) and pulls out
   gpu_runner/bootstrap.py;
2. reads secrets.json into the runner's environment (the checkpoint storage token goes to
   bootstrap in a 0600 file it deletes after reading, never in an environment), deletes the
   local copy and asks the drive to forget the uploaded one;
3. runs bootstrap (unpack, pip install, run, heartbeats, checkpoint sync, `::gpu:: exit`)
   in its own process group with GPU_OUTPUT_DIR under the work dir, relaying its merged
   output line by line so it lands in the job log;
4. enforces the wall clock: Lightning Jobs have no time limit (`max_runtime` is not one),
   so after `wall_clock_s` it prints WALL_MARK, sends SIGTERM to the group (bootstrap takes
   a final checkpoint), SIGKILL after KILL_GRACE_S. The adapter reads WALL_MARK as "session
   limit" (the job migrates) and also stops a job that runs past it (backstop);
5. packs GPU_OUTPUT_DIR into outputs.tar.gz and delivers it to the drive folder's `out/`
   (upload with the Studio's SDK; copy into the mount when it is writable; a copy under the
   job's working dir so a platform artifact copy has it too);
6. exits with the job's exit code (non-zero -> Lightning marks the job Failed).

The account's LIGHTNING_* credentials the platform puts in the job environment are kept
for this launcher's own drive calls (in bounded subprocesses) and removed from the
environment of the user's job. Python 3.8+, stdlib only.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from typing import IO, Any

PREFIX = "::gpu:: "
WALL_MARK = "gpu-router: wall-clock limit reached"
KILL_GRACE_S = 90.0
SDK_CALL_TIMEOUT_S = 300.0
CREDENTIAL_ENV = ("LIGHTNING_API_KEY", "LIGHTNING_USER_ID", "LIGHTNING_AUTH_TOKEN")

#: Runs in a child interpreter: `<python> -c UPLOAD <teamspace> <local file> <remote path>`.
UPLOAD = """
import os, sys
os.environ["LIGHTNING_DISABLE_VERSION_CHECK"] = "1"
from lightning_sdk import Teamspace
Teamspace(sys.argv[1]).upload_file(sys.argv[2], remote_path=sys.argv[3], progress_bar=False)
"""

#: `<python> -c REMOVE <owner/teamspace> <drive path>`: forget an uploaded file.
REMOVE = """
import os, sys
os.environ["LIGHTNING_DISABLE_VERSION_CHECK"] = "1"
from lightning_sdk.filesystem import Filesystem
Filesystem().rm("lit://%s/%s" % (sys.argv[1], sys.argv[2].strip("/")))
"""


def say(msg: str) -> None:
    print("gpu-router: " + msg, flush=True)


def load_config(folder: str) -> dict[str, Any]:
    with open(os.path.join(folder, "launch.json"), encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError("launch.json is not an object")
    return data


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def gpu_name() -> str | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    lines = [line.strip() for line in out.stdout.splitlines() if line.strip()]
    return lines[0] if out.returncode == 0 and lines else None


def sdk_call(code: str, args: list[str], env: dict[str, str]) -> str | None:
    """A bounded drive call through the Studio's SDK. Returns an error text or None."""
    try:
        res = subprocess.run(
            [sys.executable, "-c", code, *args],
            env=env,
            capture_output=True,
            text=True,
            timeout=SDK_CALL_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return type(exc).__name__
    if res.returncode != 0:
        tail = (res.stderr or "").strip().splitlines()
        return tail[-1][:200] if tail else f"exit {res.returncode}"
    return None


def read_secrets(folder: str, cfg: dict[str, Any], sdk_env: dict[str, str]) -> dict[str, str]:
    """Secret values for the runner (+ the storage token file). The local copy is deleted
    at once; the drive copy is removed through the SDK (the adapter sweeps it too)."""
    path = os.path.join(folder, "secrets.json")
    if not cfg.get("secrets"):
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        say("secrets.json is missing or unreadable; the job runs without its secrets")
        return {}
    values = data.get("values") if isinstance(data, dict) else None
    values = {str(k): str(v) for k, v in (values or {}).items()}
    with contextlib.suppress(OSError):
        os.remove(path)
    drive = cfg.get("drive_dir")
    if drive and cfg.get("teamspace"):
        sdk_call(REMOVE, [cfg["teamspace"], drive + "/secrets.json"], sdk_env)
    return values


def relay(stream: IO[bytes]) -> None:
    for raw in iter(stream.readline, b""):
        sys.stdout.write(raw.decode("utf-8", "replace"))
        sys.stdout.flush()


def run_bootstrap(
    argv: list[str], env: dict[str, str], cwd: str, wall_clock_s: float
) -> tuple[int, bool]:
    """Run the runner; returns (returncode, hit_wall_clock)."""
    proc = subprocess.Popen(
        argv,
        env=env,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    pump = threading.Thread(target=relay, args=(proc.stdout,), daemon=True)
    pump.start()
    deadline = time.monotonic() + wall_clock_s if wall_clock_s else None
    hit = False
    while True:
        wait = 30.0
        if deadline is not None:
            wait = min(wait, max(0.0, deadline - time.monotonic()))
        try:
            proc.wait(timeout=wait)
            break
        except subprocess.TimeoutExpired:
            if deadline is not None and time.monotonic() >= deadline:
                hit = True
                break
    if hit:
        say(f"{WALL_MARK} ({int(wall_clock_s)}s); stopping the job")
        _signal_group(proc, signal.SIGTERM)
        try:
            proc.wait(timeout=KILL_GRACE_S)
        except subprocess.TimeoutExpired:
            _signal_group(proc, signal.SIGKILL)
            proc.wait()
    pump.join(timeout=10)
    return proc.returncode, hit


def _signal_group(proc: subprocess.Popen[bytes], sig: int) -> None:
    with contextlib.suppress(OSError):
        os.killpg(proc.pid, sig)


def pack_outputs(src: str, archive: str) -> tuple[int, int]:
    count = 0
    size = 0
    with tarfile.open(archive, "w:gz") as tar:
        if os.path.isdir(src):
            for root, dirs, files in os.walk(src):
                dirs.sort()
                for name in sorted(files):
                    full = os.path.join(root, name)
                    if os.path.islink(full) or not os.path.isfile(full):
                        continue
                    tar.add(full, arcname=os.path.relpath(full, src), recursive=False)
                    count += 1
                    size += os.path.getsize(full)
    return count, size


def deliver_outputs(
    archive: str,
    folder: str,
    cfg: dict[str, Any],
    sdk_env: dict[str, str],
    count: int,
    size: int,
) -> None:
    """Put outputs.tar.gz where fetch() looks: the drive folder's out/ (SDK upload, or the
    mount when writable) and gpu-router/<name>/ under the working dir (job artifacts)."""
    delivered = []
    drive = cfg.get("drive_dir")
    if drive and cfg.get("teamspace"):
        err = sdk_call(UPLOAD, [cfg["teamspace"], archive, drive + "/out/outputs.tar.gz"], sdk_env)
        if err is None:
            delivered.append("drive")
        else:
            say(f"outputs upload to the drive failed: {err}")
    for target in (os.path.join(folder, "out"), cfg.get("artifacts_dir")):
        if not target:
            continue
        try:
            os.makedirs(target, exist_ok=True)
            shutil.copyfile(archive, os.path.join(target, "outputs.tar.gz"))
            delivered.append(target)
        except OSError:
            pass
    where = ", ".join(delivered) if delivered else "nowhere (fetch will fail)"
    say(f"outputs: {count} file(s), {size} bytes -> {where}")


def main() -> int:
    folder = os.path.dirname(os.path.abspath(__file__))
    cfg = load_config(folder)
    name = cfg["name"]
    work = cfg.get("workdir") or os.path.join(tempfile.gettempdir(), "gpu-router", name)
    os.makedirs(work, exist_ok=True)
    if not cfg.get("artifacts_dir"):
        cfg["artifacts_dir"] = os.path.join(os.getcwd(), "gpu-router", name)
    gpu = gpu_name() or "no GPU"
    # which branch of the job command ran: the /teamspace drive mount, or the copy the
    # Studio's SDK downloaded (NOTES.md "Job command")
    source = "drive mount" if folder.startswith("/teamspace/") else "sdk download"
    say(f"lightning attempt {cfg.get('attempt_key')} starting on {gpu} (files: {source})")

    bundle = os.path.join(work, "bundle.tar.gz")
    shutil.copyfile(os.path.join(folder, "bundle.tar.gz"), bundle)
    if cfg.get("bundle_sha256") and sha256_of(bundle) != cfg["bundle_sha256"]:
        say("the job bundle arrived corrupted (sha256 mismatch)")
        print(PREFIX + json.dumps({"t": "exit", "code": 90}, separators=(",", ":")), flush=True)
        return 90
    with tarfile.open(bundle, "r:gz") as tar:
        member = tar.extractfile("gpu_runner/bootstrap.py")
        if member is None:
            raise RuntimeError("the bundle has no gpu_runner/bootstrap.py")
        boot_src = member.read()
    boot = os.path.join(work, "bootstrap.py")
    with open(boot, "wb") as fh:
        fh.write(boot_src)

    sdk_env = dict(os.environ)
    sdk_env["LIGHTNING_DISABLE_VERSION_CHECK"] = "1"
    env = {k: v for k, v in os.environ.items() if k not in CREDENTIAL_ENV}
    env.update({str(k): str(v) for k, v in (cfg.get("env") or {}).items()})
    values = read_secrets(folder, cfg, sdk_env)
    token = values.pop("GPU_STORAGE_TOKEN", None)
    env.update(values)
    if token:
        token_path = os.path.join(work, ".storage-token")
        fd = os.open(token_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as fh:
            fh.write(token)
        env["GPU_STORAGE_TOKEN_FILE"] = token_path
    outputs = os.path.join(work, "outputs")
    env["GPU_OUTPUT_DIR"] = outputs
    env["GPU_CHECKPOINT_DIR"] = os.path.join(work, "checkpoints")
    env.setdefault("GPU_DATA_DIR", os.path.join(work, "data"))
    env["PYTHONUNBUFFERED"] = "1"
    exit_file = os.path.join(work, "EXIT")
    if os.path.exists(exit_file):
        os.remove(exit_file)
    argv = [
        sys.executable,
        "-u",
        boot,
        "--bundle",
        bundle,
        "--workdir",
        os.path.join(work, "job"),
        "--exit-file",
        exit_file,
        "--checkpoint-sync-dir",
        os.path.join(work, "ckpt-sync"),
        "--ckpt-seq-start",
        str(int(cfg.get("ckpt_seq_start") or 1)),
    ]
    if cfg.get("checkpoint_interval_min") is not None:
        argv += ["--checkpoint-interval-min", str(cfg["checkpoint_interval_min"])]
    resume = os.path.join(folder, "resume.tar.gz")
    if cfg.get("resume_sha256") and os.path.isfile(resume):
        local = os.path.join(work, "resume-src.tar.gz")
        shutil.copyfile(resume, local)
        if sha256_of(local) == cfg["resume_sha256"]:
            argv += ["--resume", local]
        else:
            say("the resume checkpoint arrived corrupted; starting fresh")
    elif cfg.get("resume_note"):
        say(cfg["resume_note"])

    rc, hit = run_bootstrap(argv, env, work, float(cfg.get("wall_clock_s") or 0))
    try:
        with open(exit_file) as fh:
            code = int(fh.read().strip())
    except (OSError, ValueError):
        code = 128 - rc if rc < 0 else rc
        print(PREFIX + json.dumps({"t": "exit", "code": code}, separators=(",", ":")), flush=True)
        say(f"runner ended without an exit file (exit {code})")
    if hit:
        say(WALL_MARK)

    archive = os.path.join(work, "outputs.tar.gz")
    try:
        count, size = pack_outputs(outputs, archive)
        deliver_outputs(archive, folder, cfg, sdk_env, count, size)
    except (OSError, tarfile.TarError) as exc:
        say(f"could not pack outputs: {exc}")
    say(f"finished with exit code {code}")
    return code if 0 <= code < 256 else 1


if __name__ == "__main__":
    sys.exit(main())
