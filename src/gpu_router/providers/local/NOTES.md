# Local Mac (MPS) provider notes

Runs a job bundle on the local Mac with PyTorch MPS. No account, no quota, no login. Verified live
on 2026-09-24 on an M-series Mac (arm64, macOS 27.0, 16 GB unified memory, uv 0.11.29,
CPython 3.12.13, torch 2.14.0).

## How a run works

1. `LocalAdapter.submit` (worker thread, ~50 ms) creates `<home>/local/<attempt key>/`,
   takes an exclusive `flock` on `alive.lock`, writes `launch.json` (plan, no secrets) and
   `run.json` (the commit point), and starts `python -I launcher.py <run dir> --lock-fd N`
   with the locked fd passed in and stdout/stderr on `console.log`. Secrets travel only in
   the child's environment.
2. The launcher forks: the parent writes `pid.json` and exits (the adapter waits for it), the
   child calls `setsid()`. So the run belongs to launchd, never becomes a zombie of the daemon,
   and is out of reach of a daemon restart, a closed terminal or Ctrl-C (invariant 11).
3. The child extracts the bundle archive into `work/bundle` (phase `prepare`), sets up the
   python env (phase `env`, then `install`), and execs the bundle's
   `gpu_runner/bootstrap.py --bundle work/bundle --workdir work --skip-install
   --checkpoint-sync-dir ckpt-sync --ckpt-seq-start <resume seq + 1> [--resume <path>]`
   (phase `run`). exec keeps the pid, the process group and the lock fd.
4. Liveness = "someone holds alive.lock". The adapter probes it with a non-blocking
   *shared* flock: no pid-reuse risk, no daemon memory, works after any restart. bootstrap
   runs children with `close_fds`, so only the runner itself holds the lock.

`status()` mapping, in order: alive + phase `run` = RUNNING, alive otherwise = PENDING (the
message says which setup step); dead + EXIT 0 = SUCCEEDED (a racing cancel does not hide a
finished run); dead + `cancel.json` = CANCELLED; dead + no EXIT = LOST (runner killed, Mac
restarted, or the daemon died before starting the launcher); EXIT 90 = LOST
("install failed", D22/D26: an environment failure, so the engine reroutes); EXIT 128+n for
SIGHUP/SIGINT/SIGKILL/SIGTERM without our cancel = LOST (Mac shutdown or logout, jetsam under
memory pressure, a manual kill); any other EXIT = FAILED with that code.

`cancel()`: writes `cancel.json`, SIGTERMs the run's process group (the launcher and uv while
preparing; bootstrap once running, which forwards to the entrypoint's own session, drains and
writes EXIT), waits up to 15 s, then SIGKILLs the entrypoint's group (found with `ps`: children
of the runner pid) and the runner. Finished or unknown runs: no-op.

Logs: `console.log` is the whole story (launcher lines, uv output, bootstrap tee including
protocol lines). Cursor = byte offset (a string). Only complete lines are handed out while
the run lives; the trailing partial line comes with the final `eof` chunk. `live_logs` is on:
`logs(follow=True)` polls the file every 0.5 s until eof.

Outputs: `fetch()` copies `work/outputs/**` (skips `.gpu-*` bookkeeping and directory
symlinks, reports them in `message`/`partial`), counts files already identical (size +
mtime) without copying again, never deletes anything in `dest`.

## Python environment (D28)

Settings in `config.yaml` under `providers.local` (non-secret; per-job override of `env` and
`python` via `provider_options.local`):

| key | default | meaning |
|---|---|---|
| `env` | `venv` | `venv`: one venv per deps key under `<home>/providers/local/venvs/<key>/`, created and filled once, reused by every later run with the same interpreter + deps. `system`: run with `python` as it is, install nothing (for a conda env or a hand-made venv with torch). |
| `python` | the daemon's base interpreter (`sys._base_executable`, e.g. uv's CPython 3.12) | venv base, or the interpreter itself for `system` (required there). |
| `uv` | found via `$UV`, PATH, `~/.local/bin/uv`, `~/.cargo/bin/uv`, `/opt/homebrew/bin/uv`, `/usr/local/bin/uv` | `false` = stdlib `venv` + pip (slower, ~1.5 s just to create the venv). |
| `base_packages` | `[]` | installed into every venv, e.g. `[torch, numpy]` to mirror the Kaggle/Colab images for scripts without a requirements file. Part of the deps key. |

Deps key = sha256 of (interpreter path, installer uv/pip, base_packages, manifest deps kind,
requirements file bytes or pyproject package list). A venv is marked ready
(`.gpu-router-ready` = key) only after the install succeeded; an unmarked one is deleted and
rebuilt on the next run. Creation and install hold `<key>.lock`, so two runs with the same
deps never install at the same time. Requirements are installed with cwd = the bundle's
`code/`, so `./path` entries resolve. Caveat: a requirements file that pulls in other files
(`-r other.txt`, `./localpkg`) keys on its own bytes only; edit the top file (or bump a
comment) to force a fresh venv after changing what it references.

Environment the job sees (D38, phase-3 review): an **allowlist** of the daemon's
environment, not all of it: `PATH HOME USER LOGNAME SHELL TERM TMPDIR TZ LANG LC_*`,
`__CF_USER_TEXT_ENCODING`, CA bundles (`SSL_CERT_FILE/DIR`, `REQUESTS_CA_BUNDLE`,
`CURL_CA_BUNDLE`), proxies (`HTTP(S)_PROXY`, `NO_PROXY`, `ALL_PROXY`, both cases), cache
knobs (`HF_HOME`, `HF_HUB_CACHE`, `HF_DATASETS_CACHE`, `HF_HUB_OFFLINE`,
`HF_HUB_ENABLE_HF_TRANSFER`, `TRANSFORMERS_CACHE`, `TORCH_HOME`, `XDG_*_HOME`),
`OMP_NUM_THREADS` and `UV_* PIP_* PYTORCH_*`; of those, names that look like secrets
(`models.SECRET_ENV_NAME_STRICT`), values with URL credentials (`://user:pw@`) and
token-shaped values are dropped. A daemon auto-started from a shell (D20) used to hand
every local job that shell's `KAGGLE_KEY`, `AWS_*`, `OPENAI_API_KEY`, ...; launchd- and
shell-started daemons now give jobs the same env. Daemon internals (`GPU_ROUTER_*`,
`VIRTUAL_ENV`, `PYTHONPATH`, `PYTHONHOME`, `__PYVENV_LAUNCHER__`, the daemon's own
`.venv/bin` on PATH) and runner-control variables (`GPU_EXIT_FILE`, `GPU_WORKDIR`, ...,
which would move EXIT or outputs out of the run dir) are still stripped. Then the job's env, then
`GPU_ROUTER_JOB_ID/ATTEMPT/PROTOCOL`, `GPU_CHECKPOINT_DIR`/`GPU_OUTPUT_DIR` (under `work/`),
`GPU_DATA_DIR` = `<home>/providers/local/data` (shared, so datasets download once),
`PYTORCH_ENABLE_MPS_FALLBACK=1` (the job may set 0), then secrets. In `venv` mode the
launcher adds `VIRTUAL_ENV` and puts the venv's `bin` first on PATH.

## Checkpoint storage and datasets (phase 5)

- Runs on the local Mac always use the **local** checkpoint backend, `<home>/storage/`
  (`checkpoint.local_dir`): the engine sets `GPU_STORAGE=file:///.../storage`, and bootstrap
  publishes `jobs/<job>/ckpt-NNNN/` there (stage dir renamed into place, then
  `latest.json`), instead of `ckpt-sync/` archives. No token, no huggingface_hub on the
  runner side. When a job moves between the local Mac and a remote provider the daemon copies
  the checkpoint between `<home>/storage/` and the HF bucket before submitting
  (`CheckpointHub.checkpoint_for`); `_resume_path` prefers the staged `GPU_RESUME_URI`.
- Datasets (`gpu run --data`, gpu.yaml `data:`) are not uploaded for local runs: the runner
  links `$GPU_DATA_DIR/<mount>` (`<home>/providers/local/data/<mount>`) to the path.
- Verified end to end (2026-09-24, real CLI + daemon on a private home): `gpu run --data
  ds=./mydata -p local train.py` linked the dataset, published `ckpt-0001` + `latest.json`
  to `<home>/storage/jobs/<id>/`; `tests/unit/checkpoint/test_local_e2e.py` kills a run's
  runner mid-way and checks the next attempt resumes from the synced checkpoint with the
  next seq.

## Measured live (2026-09-24)

- submit: 46-51 ms. uv venv without deps: ~0.2 s; the whole contract run incl. a fresh venv
  is under 1 s.
- `requirements.txt` = `torch`: torch 2.14.0 + 9 deps installed from the warm uv cache in
  ~0.8 s (the wheel was already in `~/.cache/uv`; a cold download is ~80 MB).
- First torch run: RUNNING after 1.1 s, done after 12.9 s (torch import + MPS init dominate).
  Second run with the same deps: "reusing python env", RUNNING at 0.05 s, done in 3.4 s.
- `torch.backends.mps.is_available()` is True in the job venv; SGD on `mps` + a 2048x2048
  matmul work. torch warns `Failed to initialize NumPy` when numpy is not installed
  (harmless; add numpy to requirements or `base_packages`).
- Daemon SIGKILLed mid-run (`gpu run -p local --json`, then `kill -9`): the next CLI call
  auto-started a new daemon, which reattached to the same run id; job `done` with one
  attempt, 8/8 step lines in `gpu logs` (no repeats), outputs in `runs/<id4>/`.
- Full path `gpu run -p local train.py` through the daemon: done in 15 s, outputs in
  `./runs/<id4>/`, checkpoint 1 recorded.

## Quirks and gotchas

- Port 47291 may already be taken by another daemon on the same Mac; isolated homes need
  `GPU_ROUTER_PORT=0` as well as `GPU_ROUTER_HOME`.
- Raw logs and run dirs (D37, phase-3 review): `console.log` and `work/job.log` hold the
  job's unredacted output (the engine keeps a redacted copy). `logs()` notes when it first
  served a dead run's log to eof in `<home>/providers/local/served/<remote_id>` (scratch,
  so logs() never changes the run dir, A6); 1 h later the sweep deletes both raw logs, and
  7 days after that (or after the submit, if never served) the whole dead run dir (a
  re-fetch after that finds no outputs). The sweep runs at submit() and healthcheck();
  live runs are never touched. Venvs are never deleted.
- A checkpoint the local Mac cannot read (hf:// before phase 5, a `file://` path on a Colab or
  Kaggle VM, a deleted file) no longer raises InvalidJob (which excluded local for the
  rest of the job): the run starts fresh, its log's first line says so, and run.json has
  `resume: unavailable` (D34). A Colab VM path
  `file:///content/gr/<session>/ckpt-sync/ckpt-NNNN.tar.gz` resolves to the Colab adapter's
  Mac mirror `<home>/providers/<name>/runs/<session>/ckpt/` when it exists.
- `max_concurrency: 1` (catalog): MPS memory is shared with the whole Mac. The catalog's
  16 GB is the build machine's unified memory; other Macs differ (`healthcheck().detail.memory_gb`).
- Exit code 90 from the user's own script is read as "install failed" (reserved by D26).

## Integration run (phase-3 integration, 2026-09-24)

`gpu run gpucheck.py --provider local --wait` through the real CLI + auto-started daemon
(private home, port 0), one job: fresh venv for `requirements.txt = torch`
(torch 2.14.0 from the warm uv cache, 10 packages in 0.45 s), `mps available True`, fp32
2048^3 matmul ~4.1 TFLOPS on MPS, done in 7 s wall (10.5 s for the whole CLI call incl.
daemon start), `runs/<id>/gpucheck.json` fetched, `gpu logs <id>` complete. torch warns
"Failed to initialize NumPy" when numpy is not in the job's deps (harmless).
