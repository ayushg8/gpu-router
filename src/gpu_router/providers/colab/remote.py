"""Scripts that run ON the Colab VM through `colab exec -f` (phase 3).

Each script is plain Python (3.8+, stdlib only) and runs inside the session's persistent
IPython kernel, so every helper is prefixed `_gr_` and does its work in a function. The
adapter prepends nothing secret: `colab exec` records the code it sends AND its output in
the CLI's history (`$HOME/.config/colab-cli/history/<session>.jsonl`; HOME is the adapter's
private 0700 cli-home and the file is deleted when the session stops, D37), so parameters
are paths, offsets and non-secret env only. Secrets travel as an uploaded 0600 file that
`launch` reads and deletes (NOTES.md "Passing secrets to the remote").

Every script prints exactly one result line, `@@GR:<base64 JSON>`, whatever else the kernel
prints around it: `{"ok": true, "result": ...}` or `{"ok": false, "error": "...", "trace":
"..."}`. `parse_result` finds the last such line in the CLI's stdout.

Remote layout for one attempt (`run_dir` = `<remote_root>/<session>`):

    bundle.tar.gz      uploaded job bundle
    resume.tar.gz      uploaded checkpoint to resume from (optional)
    .secrets.json      uploaded secret env (deleted by launch)
    bootstrap.py       extracted from the bundle by launch
    launched.json      {"pid", "started_at", "gpu"}: written once the runner started
    job.log            runner log (bootstrap --log-file): the log stream the adapter serves
    console.txt        raw stdout/stderr of bootstrap (debugging when job.log is empty)
    EXIT               bootstrap's exit code (written for every outcome it controls)
    RC                 the shell's view of bootstrap's exit (covers SIGKILL / OOM)
    work/              bootstrap workdir: checkpoints/, outputs/, bundle/
    ckpt-sync/         checkpoint archives (bootstrap --checkpoint-sync-dir)
    job.log.gz         packed log for download (binary-exact)
    outputs.tar.gz     packed outputs for download
"""

from __future__ import annotations

import base64
import json
from typing import Any

MARKER = "@@GR:"

_PRELUDE = r"""
import base64 as _gr_b64
import json as _gr_json


def _gr_emit(obj):
    data = _gr_json.dumps(obj, separators=(",", ":")).encode("utf-8")
    print("@@GR:" + _gr_b64.b64encode(data).decode("ascii"), flush=True)


def _gr_read(path):
    try:
        with open(path, "r") as fh:
            return fh.read().strip()
    except (OSError, ValueError):
        return None


def _gr_int(text):
    try:
        return int(text)
    except (TypeError, ValueError):
        return None


def _gr_run(params):
    try:
        _gr_emit({"ok": True, "result": _gr_main(params)})
    except BaseException as exc:  # report everything; the adapter decides
        import traceback

        _gr_emit(
            {
                "ok": False,
                "error": "%s: %s" % (type(exc).__name__, exc),
                "trace": traceback.format_exc()[-3000:],
            }
        )
"""

PREPARE = r"""
def _gr_main(P):
    import os
    import shutil
    import subprocess
    import sys

    run = P["run_dir"]
    os.makedirs(run, exist_ok=True)
    gpu = None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            timeout=20,
        )
        lines = [l.strip() for l in out.stdout.splitlines() if l.strip()]
        if out.returncode == 0 and lines:
            gpu = lines[0]
    except Exception:
        gpu = None
    free_gb = None
    try:
        free_gb = round(shutil.disk_usage(run).free / 1e9, 1)
    except OSError:
        pass
    return {
        "gpu": gpu,
        "python": sys.version.split()[0],
        "free_gb": free_gb,
        "launched": os.path.exists(os.path.join(run, "launched.json")),
    }
"""

LAUNCH = r"""
def _gr_main(P):
    import json
    import os
    import shlex
    import subprocess
    import sys
    import tarfile
    import time

    run = P["run_dir"]
    marker = os.path.join(run, "launched.json")
    if os.path.exists(marker):  # idempotent: a retried launch never starts a second runner
        with open(marker) as fh:
            info = json.load(fh)
        info["already"] = True
        return info
    bundle = os.path.join(run, "bundle.tar.gz")
    boot = os.path.join(run, "bootstrap.py")
    with tarfile.open(bundle, "r:gz") as tf:
        member = tf.extractfile("gpu_runner/bootstrap.py")
        if member is None:
            raise RuntimeError("bundle has no gpu_runner/bootstrap.py")
        data = member.read()
    with open(boot, "wb") as fh:
        fh.write(data)
    work = os.path.join(run, "work")
    env = dict(os.environ)
    env.pop("MPLBACKEND", None)  # the kernel's inline backend breaks plain scripts
    env.update({str(k): str(v) for k, v in P.get("env", {}).items()})
    env["PYTHONUNBUFFERED"] = "1"
    env["GPU_CHECKPOINT_DIR"] = os.path.join(work, "checkpoints")
    env["GPU_OUTPUT_DIR"] = os.path.join(work, "outputs")
    secrets_file = os.path.join(run, ".secrets.json")
    if os.path.exists(secrets_file):
        with open(secrets_file) as fh:
            values = {str(k): str(v) for k, v in json.load(fh).items()}
        os.remove(secrets_file)
        token = values.pop("GPU_STORAGE_TOKEN", None)
        env.update(values)
        if token:
            # never in the environment of the bash wrapper that lives all run (readable in
            # /proc/<pid>/environ): a 0600 file bootstrap deletes after reading it
            token_path = os.path.join(run, ".storage-token")
            fd = os.open(token_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w") as fh:
                fh.write(token)
            env["GPU_STORAGE_TOKEN_FILE"] = token_path
    argv = [
        P.get("python") or sys.executable,
        "-u",
        boot,
        "--bundle",
        bundle,
        "--workdir",
        work,
        "--log-file",
        os.path.join(run, "job.log"),
        "--exit-file",
        os.path.join(run, "EXIT"),
        "--checkpoint-sync-dir",
        os.path.join(run, "ckpt-sync"),
        "--ckpt-seq-start",
        str(int(P.get("ckpt_seq_start", 1))),
        "--heartbeat-s",
        str(float(P.get("heartbeat_s", 60))),
    ]
    if P.get("checkpoint_interval_min") is not None:
        argv += ["--checkpoint-interval-min", str(float(P["checkpoint_interval_min"]))]
    if P.get("resume"):
        argv += ["--resume", os.path.join(run, "resume.tar.gz")]
    cmd = (
        " ".join(shlex.quote(a) for a in argv)
        + " > console.txt 2>&1; echo $? > RC.tmp && mv RC.tmp RC"
    )
    proc = subprocess.Popen(
        ["bash", "-c", cmd],
        cwd=run,
        env=env,
        start_new_session=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    info = {"pid": proc.pid, "started_at": time.time(), "gpu": P.get("gpu")}
    tmp = marker + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(info, fh)
    os.replace(tmp, marker)
    info["already"] = False
    return info
"""

POLL = r"""
def _gr_pack(run, want_outputs):
    import gzip
    import os
    import shutil
    import tarfile

    packed = {}
    log = os.path.join(run, "job.log")
    gz = log + ".gz"
    if os.path.exists(log):
        tmp = gz + ".tmp"
        with open(log, "rb") as src, gzip.open(tmp, "wb", compresslevel=6) as dst:
            shutil.copyfileobj(src, dst)
        os.replace(tmp, gz)
        packed["log_bytes"] = os.path.getsize(log)
        packed["log_gz_bytes"] = os.path.getsize(gz)
    if want_outputs:
        out_dir = os.path.join(run, "work", "outputs")
        arc = os.path.join(run, "outputs.tar.gz")
        tmp = arc + ".tmp"
        files = 0
        total = 0
        with tarfile.open(tmp, "w:gz", compresslevel=1, dereference=True) as tf:
            if os.path.isdir(out_dir):
                for root, dirs, names in os.walk(out_dir):
                    dirs.sort()
                    for name in sorted(names):
                        full = os.path.join(root, name)
                        if not os.path.isfile(full):
                            continue
                        tf.add(full, arcname=os.path.relpath(full, out_dir), recursive=False)
                        files += 1
                        total += os.path.getsize(full)
        os.replace(tmp, arc)
        packed["outputs_files"] = files
        packed["outputs_bytes"] = total
        packed["outputs_archive_bytes"] = os.path.getsize(arc)
    return packed


def _gr_main(P):
    import json
    import os

    run = P["run_dir"]
    if not os.path.isdir(run):
        return {"run_dir": False}
    marker = None
    raw = _gr_read(os.path.join(run, "launched.json"))
    if raw:
        try:
            marker = json.loads(raw)
        except ValueError:
            marker = None
    exit_code = _gr_int(_gr_read(os.path.join(run, "EXIT")))
    rc = _gr_int(_gr_read(os.path.join(run, "RC")))
    alive = False
    if marker and marker.get("pid"):
        pid = int(marker["pid"])
        try:
            os.kill(pid, 0)
            alive = True
        except OSError:
            alive = False
        if alive:
            try:  # an unreaped child of the kernel is a zombie: dead for our purposes
                with open("/proc/%d/stat" % pid) as fh:
                    if fh.read().rsplit(")", 1)[1].split()[0] == "Z":
                        alive = False
            except (OSError, IndexError):
                pass
    try:
        log_bytes = os.path.getsize(os.path.join(run, "job.log"))
    except OSError:
        log_bytes = None
    ckpt = None
    sync = os.path.join(run, "ckpt-sync")
    try:
        names = sorted(
            n for n in os.listdir(sync) if n.startswith("ckpt-") and n.endswith(".tar.gz")
        )
        if names:
            ckpt = {"name": names[-1], "size": os.path.getsize(os.path.join(sync, names[-1]))}
    except OSError:
        pass
    out = {
        "run_dir": True,
        "launched": marker,
        "exit": exit_code,
        "rc": rc,
        "alive": alive,
        "log_bytes": log_bytes,
        "ckpt": ckpt,
    }
    ended = exit_code is not None or rc is not None or (marker is not None and not alive)
    if P.get("pack") and ended:
        code = exit_code if exit_code is not None else rc
        out["packed"] = _gr_pack(run, want_outputs=(code == 0))
    return out
"""

PACK = r"""
def _gr_main(P):
    return _gr_pack(P["run_dir"], want_outputs=bool(P.get("outputs")))
"""

LOGREAD = r"""
def _gr_main(P):
    import os

    path = P["path"]
    off = int(P.get("byte_offset", 0))
    max_bytes = int(P.get("max_bytes", 262144))
    try:
        size = os.path.getsize(path)
    except OSError:
        return {"exists": False, "size": 0, "data": ""}
    if off > size:
        return {"exists": True, "size": size, "data": "", "truncated": True}
    with open(path, "rb") as fh:
        fh.seek(off)
        data = fh.read(max_bytes)
    import base64

    return {"exists": True, "size": size, "data": base64.b64encode(data).decode("ascii")}
"""

SCRIPTS: dict[str, str] = {
    "prepare": PREPARE,
    "launch": LAUNCH,
    "poll": POLL,
    "pack": POLL + PACK,  # pack reuses _gr_pack from the poll script
    "logread": LOGREAD,
}


def build(name: str, params: dict[str, Any]) -> str:
    """The full source for one `colab exec -f` call: prelude, script, call with params."""
    body = SCRIPTS[name]
    payload = json.dumps(params, sort_keys=True, separators=(",", ":"))
    return f"{_PRELUDE}\n{body}\n_gr_run(_gr_json.loads({payload!r}))\n"


class RemoteScriptError(Exception):
    """The script ran but reported an exception (`ok: false`)."""

    def __init__(self, error: str, trace: str = "") -> None:
        super().__init__(error)
        self.error = error
        self.trace = trace


def parse_result(stdout: str) -> Any:
    """The `result` of the last `@@GR:` line. Raises ValueError when there is no marker
    (the script never ran or its output was cut) and RemoteScriptError when it failed."""
    found: str | None = None
    for line in stdout.splitlines():
        idx = line.find(MARKER)
        if idx >= 0:
            found = line[idx + len(MARKER) :].strip()
    if found is None:
        raise ValueError("no result marker in colab exec output")
    try:
        doc = json.loads(base64.b64decode(found, validate=True).decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError(f"unreadable result marker: {exc}") from None
    if not isinstance(doc, dict):
        raise ValueError("result marker is not an object")
    if not doc.get("ok"):
        raise RemoteScriptError(str(doc.get("error") or "remote error"), str(doc.get("trace", "")))
    return doc.get("result")


#: Log lines longer than this are cut into pieces of this size, counted from the start of the
#: line, so the VM-side reader and the harvested copy always cut at the same offsets.
MAX_LINE_BYTES = 64 * 1024


def _decode(raw: bytes) -> str:
    return raw.decode("utf-8", errors="replace").rstrip("\r")


def split_log_bytes(
    data: bytes, *, final: bool, max_line: int = MAX_LINE_BYTES
) -> list[tuple[str, int]]:
    """Split raw log bytes (starting at a line boundary) into `(line, end)` pairs, where
    `end` is the offset in `data` just past the line and its newline.

    A trailing fragment without a newline is held back (it may still be growing) unless
    `final` (the run ended). A line longer than `max_line` bytes comes out as pieces of
    `max_line` bytes. Lines are decoded as UTF-8 with replacement and lose a trailing
    `\\r`. The result depends only on the bytes and where the read started, so reading on
    the VM in windows and reading the downloaded copy in one go give the same lines.
    """
    out: list[tuple[str, int]] = []
    pos = 0
    n = len(data)
    while pos < n:
        nl = data.find(b"\n", pos, pos + max_line + 1)
        if nl >= 0:
            out.append((_decode(data[pos:nl]), nl + 1))
            pos = nl + 1
        elif n - pos > max_line:
            out.append((_decode(data[pos : pos + max_line]), pos + max_line))
            pos += max_line
        else:
            if final:
                out.append((_decode(data[pos:]), n))
            break
    return out
