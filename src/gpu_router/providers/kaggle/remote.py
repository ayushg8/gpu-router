"""What gets pushed to Kaggle for one attempt: kernel naming, kernel-metadata.json and the
generated `run.py` (the kernel's only code file).

`kaggle kernels push` uploads ONLY the `code_file` text (verified in the CLI source), so the
job bundle rides inside run.py as base64 (sha256-checked on arrival). run.py is stdlib-only
and Python 3.8 compatible like the rest of the remote side (D2). On the kernel it:

1. writes the bundle to /tmp/gpu-router/bundle.tar.gz and pulls gpu_runner/bootstrap.py out
2. runs bootstrap (unpack, pip install, run, heartbeats, checkpoint sync, exit line) with
   GPU_OUTPUT_DIR=/kaggle/working/outputs (the only files `fetch` downloads),
   GPU_CHECKPOINT_DIR under /tmp and the checkpoint sync dir under
   /kaggle/working/.gpu-router/ (kept with the kernel's outputs)
3. relays bootstrap's merged output line by line through print(), so the lines land in the
   kernel log whether Kaggle runs the script as a process or inside a notebook kernel
4. prints `::gpu:: {"t":"exit",...}` itself if bootstrap died without writing EXIT, and
   raises on a non-zero exit so Kaggle marks the version as failed (ERROR)

Secrets (phase 5): values never ride in run.py (kept in the kernel's version history).
They travel in a private Kaggle dataset `<user>/gpu-router-secrets` (adapter.py keeps it
current) attached through `dataset_sources`; run.py reads `gpu-router-secrets.json` from
/kaggle/input into the runner's environment. The checkpoint storage token arrives that
way too, but is handed to bootstrap in a 0600 file (GPU_STORAGE_TOKEN_FILE) that bootstrap
deletes after reading, never in an environment (/proc/<pid>/environ). The secrets file on
the read-only /kaggle/input mount itself cannot be removed and stays readable by the job:
that is why remote runs need the separate, bucket-scoped HF_TOKEN_REMOTE (D44).

Naming: one private kernel per attempt, title `gpu-router <job_id> <n>` (>= 5 chars; the
CLI's slugify turns it into the id slug `gpu-router-<job_id>-<n>`, verified), so the attempt
key maps to exactly one kernel (A4, lookup_by_key) and pushing never adds a version to a
kernel that already ran.
"""

from __future__ import annotations

import base64
import json
import re
import textwrap
from collections.abc import Mapping
from typing import Any

__all__ = [
    "CHECKPOINT_SYNC_DIR",
    "KERNEL_WORKDIR",
    "OUTPUT_DIR",
    "OUTPUT_PATTERN",
    "OUTPUT_PREFIX",
    "SECRETS_DATASET_SLUG",
    "SECRETS_FILE",
    "kernel_metadata",
    "kernel_title",
    "render_runner",
    "slug_for_key",
]

KERNEL_WORKDIR = "/tmp/gpu-router"  # noqa: S108 - a path on the Kaggle VM, not on this Mac
OUTPUT_PREFIX = "outputs"
OUTPUT_DIR = f"/kaggle/working/{OUTPUT_PREFIX}"
OUTPUT_PATTERN = rf"^{OUTPUT_PREFIX}/"  # `kaggle kernels output --file-pattern` (re.search)
CHECKPOINT_SYNC_DIR = "/kaggle/working/.gpu-router/checkpoints"
SECRETS_DATASET_SLUG = "gpu-router-secrets"  # private dataset carrying job secrets (phase 5)
SECRETS_FILE = "gpu-router-secrets.json"

_KEY_RE = re.compile(r"^gpu-([0-9a-f]{6,32})-([1-9][0-9]{0,3})$")
_B64_WIDTH = 76


def slug_for_key(attempt_key: str) -> str | None:
    """`gpu-<job_id>-<n>` -> `gpu-router-<job_id>-<n>`; None for anything else."""
    m = _KEY_RE.match(attempt_key)
    if m is None:
        return None
    return f"gpu-router-{m.group(1)}-{m.group(2)}"


def kernel_title(attempt_key: str) -> str:
    m = _KEY_RE.match(attempt_key)
    if m is None:
        raise ValueError(f"not an attempt key: {attempt_key!r}")
    return f"gpu-router {m.group(1)} {m.group(2)}"


def kernel_metadata(
    *,
    owner: str,
    slug: str,
    title: str,
    machine_shape: str | None,
    enable_internet: bool = True,
    dataset_sources: list[str] | None = None,
) -> dict[str, Any]:
    """kernel-metadata.json. is_private / enable_internet / enable_gpu are always explicit
    (CLI 2.2.4 defaults enable_internet to true while the docs say false)."""
    meta: dict[str, Any] = {
        "id": f"{owner}/{slug}",
        "title": title,
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": machine_shape is not None,
        "enable_tpu": False,
        "enable_internet": enable_internet,
        "dataset_sources": list(dataset_sources or []),
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
        "docker_image_pinning_type": "original",
    }
    if machine_shape is not None:
        meta["machine_shape"] = machine_shape
    return meta


def _b64_block(data: bytes) -> str:
    encoded = base64.b64encode(data).decode("ascii")
    return "\n".join(textwrap.wrap(encoded, _B64_WIDTH)) if encoded else ""


_TEMPLATE = '''\
# gpu-router kaggle runner (generated, do not edit). attempt: {attempt_key}
# Python 3.8+, stdlib only. See gpu_router/providers/kaggle/remote.py.
import base64
import hashlib
import json
import os
import subprocess
import sys
import tarfile

ATTEMPT_KEY = {attempt_key!r}
ENV = json.loads({env_json!r})
WORKDIR = {workdir!r}
OUTPUT_DIR = {output_dir!r}
SYNC_DIR = {sync_dir!r}
CKPT_SEQ_START = {ckpt_seq_start!r}
CHECKPOINT_INTERVAL_MIN = {interval!r}
BUNDLE_SHA256 = {bundle_sha!r}
RESUME_SHA256 = {resume_sha!r}
RESUME_NOTE = {resume_note!r}
SECRETS_FILE = {secrets_file!r}
SECRETS_DATASET = {secrets_dataset!r}
INPUT_ROOT = {input_root!r}
PREFIX = "::gpu:: "

BUNDLE_B64 = """
{bundle_b64}
"""

RESUME_B64 = """
{resume_b64}
"""


def say(msg):
    print("gpu-router: " + msg, flush=True)


def load_secrets():
    """Job secrets from the attached private dataset (never from this file's source)."""
    if not SECRETS_DATASET:
        return {{}}
    slug = SECRETS_DATASET.split("/")[-1]
    candidates = [
        os.path.join(INPUT_ROOT, slug, SECRETS_FILE),
        os.path.join(INPUT_ROOT, "datasets", SECRETS_DATASET, SECRETS_FILE),
    ]
    base = INPUT_ROOT.rstrip(os.sep).count(os.sep)
    for root, dirs, files in os.walk(INPUT_ROOT):
        if root.count(os.sep) - base >= 3:
            dirs[:] = []
        if SECRETS_FILE in files:
            candidates.append(os.path.join(root, SECRETS_FILE))
    for path in candidates:
        try:
            with open(path) as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            continue
        values = data.get("values") if isinstance(data, dict) else None
        if isinstance(values, dict):
            return dict((str(k), str(v)) for k, v in values.items())
    say("secrets dataset %s is not attached or unreadable; no secrets" % SECRETS_DATASET)
    return {{}}


def write_blob(b64, path, sha):
    data = base64.b64decode(b64)
    if hashlib.sha256(data).hexdigest() != sha:
        raise RuntimeError("gpu-router: %s arrived corrupted (sha256 mismatch)" % path)
    with open(path, "wb") as fh:
        fh.write(data)


def main():
    os.makedirs(WORKDIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    say("kaggle attempt %s starting" % ATTEMPT_KEY)
    bundle = os.path.join(WORKDIR, "bundle.tar.gz")
    write_blob(BUNDLE_B64, bundle, BUNDLE_SHA256)
    with tarfile.open(bundle, "r:gz") as tar:
        boot_src = tar.extractfile(tar.getmember("gpu_runner/bootstrap.py")).read()
    boot = os.path.join(WORKDIR, "bootstrap.py")
    with open(boot, "wb") as fh:
        fh.write(boot_src)
    exit_file = os.path.join(WORKDIR, "EXIT")
    if os.path.exists(exit_file):
        os.unlink(exit_file)
    env = dict(os.environ)
    env.update(ENV)
    secrets = load_secrets()
    token = secrets.pop("GPU_STORAGE_TOKEN", None)
    env.update(secrets)
    if token:
        # the storage token goes to bootstrap in a 0600 file it deletes after reading: in
        # the environment it would stay readable in /proc/<pid>/environ all run long
        token_path = os.path.join(WORKDIR, ".storage-token")
        fd = os.open(token_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as fh:
            fh.write(token)
        env["GPU_STORAGE_TOKEN_FILE"] = token_path
    env["GPU_OUTPUT_DIR"] = OUTPUT_DIR
    env["GPU_CHECKPOINT_DIR"] = os.path.join(WORKDIR, "checkpoints")
    env.setdefault("GPU_DATA_DIR", os.path.join(WORKDIR, "data"))
    env["PYTHONUNBUFFERED"] = "1"
    argv = [
        sys.executable, "-u", boot,
        "--bundle", bundle,
        "--workdir", os.path.join(WORKDIR, "job"),
        "--exit-file", exit_file,
        "--checkpoint-sync-dir", SYNC_DIR,
        "--ckpt-seq-start", str(CKPT_SEQ_START),
    ]
    if CHECKPOINT_INTERVAL_MIN is not None:
        argv += ["--checkpoint-interval-min", str(CHECKPOINT_INTERVAL_MIN)]
    if RESUME_SHA256:
        resume = os.path.join(WORKDIR, "resume-src.tar.gz")
        write_blob(RESUME_B64, resume, RESUME_SHA256)
        argv += ["--resume", resume]
    elif RESUME_NOTE:
        say(RESUME_NOTE)
    proc = subprocess.Popen(
        argv, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=WORKDIR
    )
    for raw in iter(proc.stdout.readline, b""):
        sys.stdout.write(raw.decode("utf-8", "replace"))
        sys.stdout.flush()
    rc = proc.wait()
    code = None
    try:
        with open(exit_file) as fh:
            code = int(fh.read().strip())
    except (OSError, ValueError):
        code = 128 - rc if rc < 0 else rc
        print(PREFIX + json.dumps({{"t": "exit", "code": code}}, separators=(",", ":")))
        say("runner ended without an exit file (exit %d)" % code)
    sys.stdout.flush()
    if code != 0:
        raise RuntimeError("gpu-router: job exited with code %d" % code)


main()
'''


def render_runner(
    *,
    attempt_key: str,
    bundle: bytes,
    bundle_sha256: str,
    env: Mapping[str, str],
    ckpt_seq_start: int = 1,
    checkpoint_interval_min: int | None = None,
    resume: bytes | None = None,
    resume_sha256: str | None = None,
    resume_note: str | None = None,
    secrets_dataset: str | None = None,
    input_root: str = "/kaggle/input",
    workdir: str = KERNEL_WORKDIR,
    output_dir: str = OUTPUT_DIR,
    sync_dir: str = CHECKPOINT_SYNC_DIR,
) -> str:
    """The run.py text for one attempt. `env` must hold non-secret values only (it is
    stored in the kernel's version history); secrets come from the attached
    `secrets_dataset` (`<user>/<slug>`) at run time. The dirs are overridable for tests
    that run the script on this Mac."""
    if (resume is None) != (resume_sha256 is None):
        raise ValueError("resume and resume_sha256 go together")
    return _TEMPLATE.format(
        attempt_key=attempt_key,
        env_json=json.dumps(dict(env), sort_keys=True),
        workdir=workdir,
        output_dir=output_dir,
        sync_dir=sync_dir,
        ckpt_seq_start=int(ckpt_seq_start),
        interval=checkpoint_interval_min,
        bundle_sha=bundle_sha256,
        resume_sha=resume_sha256,
        resume_note=resume_note,
        secrets_file=SECRETS_FILE,
        secrets_dataset=secrets_dataset,
        input_root=input_root,
        bundle_b64=_b64_block(bundle),
        resume_b64=_b64_block(resume or b""),
    )
