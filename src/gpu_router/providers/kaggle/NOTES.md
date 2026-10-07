# Kaggle provider notes

verified_at: 2026-09-24 (live runs through the adapter; research 2026-09-23). Kaggle CLI 2.2.4
(`uv tool install kaggle`), kagglesdk 0.1.37 bundled.
Legend: **[V]** verified from CLI help / package source / a live call on the build machine.
**[L]** verified by a live kernel run through `KaggleAdapter` (see "Live runs").
**[D]** from official docs (linked). **[I]** inferred, not verified.

## Summary

| Field | Value | Status |
|---|---|---|
| GPUs | `NvidiaTeslaT4` = 2x Tesla T4 15360 MiB each (driver 580.159.04, torch 2.10.0+cu128); `NvidiaTeslaP100` = 1x P100 16GB | T4x2 [L], P100 name [V], P100 not run |
| Other accelerators | `Tpu1VmV38` (TPU v3-8, 20h/week separate quota) | [V] |
| Free limit | 30 GPU-hours/week (TPU 20h/week, separate) | [V] live `kaggle quota` |
| Reset | weekly, `refreshAt` = Saturday 00:00 (2026-09-26T00:00:00, no tz suffix; UTC assumed) | [V] value, tz [I] |
| Session cap | 12h per GPU/CPU session (TPU 9h); we push `-t` = 12h - 5 min | [D] |
| Concurrency | small number of GPU sessions at once; catalog max_concurrency 1 | [D]/[I] |
| Queue + boot | 16 s from push to RUNNING (2xT4, one sample) | [L] |
| Quota accounting | a 37 s GPU session showed as +0.01 h within 5 s of the end | [L] |
| Cancel | `kernels delete` of a running kernel stops the GPU session at once | [L] |
| Card required | No | [D] |
| Phone verification | Required for GPU and internet in kernels (the test account was verified: GPU + pip worked) | [D]/[L] |
| Quota queryable | Yes: `kaggle quota --format json` | [V] |

## How the adapter uses Kaggle (phase 3)

Code: `adapter.py` (KaggleAdapter), `cli.py` (bounded CLI calls + error mapping), `parse.py`
(pure output parsers), `remote.py` (kernel naming, metadata, generated run.py),
`credentials.py` (Keychain or CLI files). Registry shim: `gpu_router/adapters/kaggle.py`.

- **One private kernel per attempt**: id `<user>/gpu-router-<job_id>-<n>`, title
  `gpu-router <job_id> <n>` (CLI `slugify(title)` == slug [V], >= 5 chars). The attempt key
  maps to exactly one slug, so `lookup_by_key` is a `kernels status` call and a retried
  submit never pushes a second version (a new version = a second run). submit() checks a
  local marker (`<home>/providers/kaggle/submits/<key>.json`), then `kernels status`, then
  (GPU jobs) a live quota pre-check, then pushes.
- **Interrupted push (D35, phase-3 review)**: `submits/<key>.pushing` ({started_at}) is
  written right before `kernels push` and removed when the call returns (subprocess.run
  kills the child at T_PUSH). Only a daemon death leaves it, and then the CLI child may
  still be uploading: for T_PUSH + 60 s from `started_at`, a `kernels status` NotFound
  makes `lookup_by_key` and `submit` raise Unavailable ("may still be uploading") instead
  of "nothing exists" / pushing again. After the window a stale intent is removed and the
  normal path runs (found -> the ref; missing -> None / push).
- **Metadata** always explicit: `is_private: true`, `enable_gpu`, `enable_internet` (true by
  default: pip needs it), `machine_shape` from the router's GPU (`T4` -> `NvidiaTeslaT4`,
  `P100` -> `NvidiaTeslaP100`), `docker_image_pinning_type: original`, no data sources.
- **Code transport**: `kernels push` uploads only `code_file` [V], so run.py carries the job
  bundle as base64 (sha256-checked on arrival) while run.py stays under
  `remote.MAX_INLINE_SOURCE` (900,000 B). **SaveKernel refuses a code file over ~1 MB** [L,
  2026-10-04, CLI 2.2.4, CPU probe kernels: 933,761 B pushed, 1,141,256 B and 3,112,456 B got
  `400 Client Error: Bad Request for url: .../KernelsApiService/SaveKernel`; a 12.9 MB one got
  `Expecting value: line 1 column 1 (char 0)`]. The old 10 MB inline limit was [I] and wrong:
  every project bundle over ~700 KB failed to push between 2026-09-25 and 2026-10-04.
- **Blob datasets** (2026-10-04): a bundle / resume archive that does not fit inline, and
  `data:` datasets when there is no HF storage (`stage_data`), travel as private datasets
  `<user>/gpu-router-{bundle,ckpt,data}-<sha16>` holding one `*.bin` file (a directory is one
  uncompressed tar), attached through `dataset_sources`, sha256-checked (bundle, resume) by
  run.py. Uploaded once per content and reused by later attempts and jobs; records in
  `<home>/providers/kaggle/blobs/`; a background sweep deletes bundle/ckpt blobs unused for 3
  days and data blobs unused for `data_keep_days` (30). Live facts [L, 2026-10-04]: a 3 MB
  `.bin` dataset was `ready` ~5 s after `datasets create` (the first `datasets status` right
  after the create answers 403); it mounts at `/kaggle/input/datasets/<owner>/<slug>/<file>`
  byte-identical (sha256 checked in a CPU kernel). Settings: `blob_datasets` (true; false =
  bundles over ~700 KB are InvalidJob), `max_bundle_mb` (100), `data_keep_days` (30).
- **run.py on the kernel** (stdlib, py3.8+): writes the bundle to /tmp/gpu-router, extracts
  `gpu_runner/bootstrap.py`, runs bootstrap with `GPU_OUTPUT_DIR=/kaggle/working/outputs`,
  `GPU_CHECKPOINT_DIR=/tmp/gpu-router/checkpoints`, `--checkpoint-sync-dir
  /kaggle/working/.gpu-router/checkpoints`, `--ckpt-seq-start` (resume seq + 1); relays
  bootstrap's output through print() (so lines reach the kernel log even if Kaggle runs the
  script inside a notebook kernel); prints its own `::gpu:: exit` line if bootstrap died
  without EXIT; raises on a non-zero exit so Kaggle marks the version ERROR.
- **Status**: QUEUED/NEW_SCRIPT -> pending, RUNNING -> running, CANCEL_* -> cancelled.
  COMPLETE/ERROR: the log's last `::gpu:: exit` line decides (0 succeeded, 90 = dependency
  install failed -> lost so the job reroutes, else failed with that code). No exit line:
  failure message with a time-limit word -> lost ("kaggle session time limit"); "quota" in
  the message, or live GPU quota at 0.00h -> lost with quota_exhausted; COMPLETE without an
  exit line -> lost. COMPLETE **or ERROR** with an empty log -> Unavailable (log not
  published yet: run.py always prints first) for up to 10 min from the first sight
  (`final/<slug>.empty-log`), then judged without it (phase-3 review; that Kaggle delays
  the log after ERROR the way it does after COMPLETE is [I]). The final outcome + log
  lines are cached in `<home>/providers/kaggle/final/<slug>.json`, the lines passed
  through `secrets.redact` first (one out per one in, so line cursors hold; D37).
- **Missing vs forbidden**: every kernel-scoped 401/403 prints `Cannot access kernel ...`,
  the same text as a missing kernel [V]. Before status/lookup act on "missing" (NotFound =
  attempt lost = rerun), the adapter proves auth with `kaggle quota`.
- **Logs**: Kaggle's own log only after the run finishes. While running, phase 5 serves
  the runner's `log-tail.json` from checkpoint storage (see "Checkpoint storage" below);
  without storage `logs()` returns an empty chunk that keeps the cursor. Cursor = lines
  returned (final log) or `t<lines>:<hash>` (live tail).
- **Fetch**: `kernels output <ref> -p <tmp> --file-pattern ^outputs/ -o -q`, then the files
  under `outputs/` are copied (atomically, subdirs kept) into `runs/<id4>/`; nothing in dest
  is deleted; `.gpu-router/` checkpoints and the `<slug>.log` the CLI also writes stay out.
- **Secrets** (phase 5): run.py is kept in the kernel's version history and push metadata
  has no secrets field (kaggle-cli issue #582), so values travel in a **private dataset**
  `<user>/gpu-router-secrets` (file `gpu-router-secrets.json`, `{"v":1,"values":{...}}`)
  attached through `dataset_sources`; run.py reads it from `/kaggle/input` into the
  runner's environment. See "Checkpoint storage" below. `providers.kaggle.secrets_dataset:
  false` turns the channel off (jobs with secrets are then refused, InvalidJob).
- **Resume**: `capabilities.resume = True` (phase 5). With checkpoint storage the engine
  sets `GPU_RESUME_URI` (hf://buckets/...) and the runner downloads the checkpoint itself
  (a checkpoint written on the local Mac is copied into the bucket by the daemon first). Without
  storage the phase-3 path stands: a `file://` archive on the local Mac is embedded and passed
  as `--resume`; anything else starts fresh with a note.
- **Test mode**: with `GPU_ROUTER_TEST_MODE` the adapter never runs the real CLI (health
  DISABLED, calls refused) unless `GPU_ROUTER_REAL_PROVIDERS` lists kaggle (invariant 20).

## Login / auth (headless)

Auth order in `KaggleApi.authenticate()` [V]:
1. Access token: `KAGGLE_API_TOKEN` env (value, or a path to a file holding it), else `~/.kaggle/access_token` (or `.txt`).
2. Legacy key: `KAGGLE_USERNAME` + `KAGGLE_KEY` env, else `~/.kaggle/kaggle.json` (`{"username","key"}`).
   Config dir overridable with `KAGGLE_CONFIG_DIR`.
3. OAuth: `kaggle auth login` (browser) caches `~/.kaggle/credentials.json`.
4. Anonymous (only some read commands).

On the build machine: `~/.kaggle/kaggle.json` exists and works (`auth_method: LEGACY_API_KEY`). [V]

Adapter credentials (`credentials.py`, config `providers.kaggle.credentials`, default `auto`):
- `auto`: Keychain when a secret exists, else the CLI's own files (gpu-router never opens them).
- `keychain`: Keychain only. Secret `kaggle` = the kaggle.json document, or
  `KAGGLE_API_TOKEN` = an access token. Values go to the CLI subprocess as env vars only
  (`KAGGLE_USERNAME`/`KAGGLE_KEY` or `KAGGLE_API_TOKEN`), with `KAGGLE_CONFIG_DIR` pointed
  at an empty private dir so a stale file cannot win; inherited `KAGGLE_*` vars are dropped.
- `cli`: CLI files only.

**Migration to the Keychain**: `gpu secrets set kaggle --stdin < ~/.kaggle/kaggle.json`, run
`gpu providers` (health detail shows `credentials: keychain-json`), then the file may be
deleted. Not done during the build (the build never moves credentials).

Username for kernel ids: settings `providers.kaggle.username`, else the Keychain document,
else `kaggle config view` (prints `- username: <name>`, never the key [V]).

## CLI facts (kaggle 2.2.4)

- Push: `kaggle -W kernels push -p <dir> -t <seconds>`. Success: `Kernel version N
  successfully pushed.  Please check progress at <url>`. Server refusal: `Kernel push
  error: <msg>` with **exit 0** [V source]. Local metadata problems (title < 5 chars, missing
  code file) are ValueErrors printed to stderr, exit 1, nothing sent [V]. `-W` suppresses the
  outdated-version warning.
- Status: `<ref> has status "KernelWorkerStatus.<X>"` + optional `Failure message: "..."` [V].
  Enum: QUEUED, RUNNING, COMPLETE, ERROR, CANCEL_REQUESTED, CANCEL_ACKNOWLEDGED, NEW_SCRIPT [V].
- Missing kernel (or kernel-scoped 401/403): stderr `Cannot access kernel '<ref>'
  (Permission 'kernels.get' was denied). ...`, exit 1 [V live]. `kernels delete` of a
  missing kernel: `403 Client Error: Forbidden for url: .../DeleteKernel`, exit 1 [V live].
- Logs (non-follow) after the run: a JSON array of `{"stream_name", "time", "data"}` (data
  ends with "\n"; stdout and stderr interleaved; time = seconds since session start) [V live].
  Mid-run the same command prints an empty line [L].
- Output: `kernels output` downloads every file whose name matches `--file-pattern`
  (`re.search`), pages through all results, and also writes `<slug>.log` [V source].
- Quota: `kaggle quota --format json` -> `[{"resource":"GPU","used":"0.00h","remaining":
  "30.00h","total":"30.00h","refreshAt":"2026-09-26T00:00:00"}, {TPU...}]` [V]. Hours have 2
  decimals, so the adapter treats `0.00h` remaining as exhausted.
- Network errors are uncaught tracebacks (exit 1); 429 prints `429 Client Error: Too Many
  Requests` [V source]; 401 anywhere prints the auth help text (`Authentication required to
  call the Kaggle API`) [V source].

## Checkpoint storage (phase 5)

Code: `gpu_router/checkpoint/` (hub, storage facade, side channel), the runner's
`gpu_runner/storage.py`, `adapter.py` (`_ensure_secrets_dataset`, `_status_uri`, logs).
Nothing here is verified live yet (no HF token on the build machine, and a build session creates no
cloud resources): every line below is [I] until the first live run.

- **Token path**: the engine adds the job secret `GPU_STORAGE_TOKEN` (Keychain
  `HF_TOKEN_REMOTE`, else `HF_TOKEN`) and non-secret env `GPU_STORAGE=hf://buckets/<you>/
  gpu-router`, `GPU_RESUME_URI`, `GPU_STATUS_PUSH_S`, ... The adapter puts the secret into
  the private dataset `<user>/gpu-router-secrets` (created with `kaggle datasets create -p
  <dir> -q`, private by default; later `kaggle datasets version -p <dir> -m gpu-router -d
  -q`, old versions deleted) and attaches it; the env rides in run.py. The dataset is
  re-uploaded only when the values change (an HMAC of them with a random local salt is
  kept in `<home>/providers/kaggle/secrets/dataset.json`, never the values); the push dir
  holding plaintext values is deleted right after the CLI call. Before the push the adapter
  polls `kaggle datasets status <ref>` until `ready` (up to 6 x 5 s; statuses from
  kagglesdk `DatabundleVersionStatus`: `ready`, `failed`, `deleted`, `blobs_received`, ...).
  At rest the token sits in a private dataset of your Kaggle account: anyone with that
  account's credentials can read it, so use a fine-grained HF token (`gpu login hf
  --remote`) and rotate it by running that command again (the next submit versions the
  dataset and deletes the old version). [I]: that a kernel pushed right after a new
  version mounts the new one, and the mount path (`/kaggle/input/<slug>/` or
  `/kaggle/input/datasets/<owner>/<slug>/`; run.py looks in both and walks 3 levels).
- **Runner on the kernel**: bootstrap pops `GPU_STORAGE_TOKEN` from the environment before
  anything runs (the job and pip never see it), pip-installs `huggingface_hub>=1.32,<2` into
  a private `--target` dir when the image's copy lacks the bucket API (Kaggle's image pins
  an older one for transformers 4.x; the job keeps the image's version), and needs
  `enable_internet` (default true). Checkpoints go to `jobs/<job>/ckpt-NNNN/` in the
  bucket; `latest.json` is written last.
- **Near-live logs**: the runner pushes `jobs/<job>/attempts/<n>/log-tail.json` (last 1000
  lines, absolute line numbers) and `heartbeat.json` every 60 s. `RemoteRef.meta.status_uri`
  records where; `logs()` serves new tail lines while the kernel runs and switches to the
  final log when it ends, aligned on the runner's `hello` line plus a hash of the last
  tail line served, with run.py's own lines (secrets/resume notes, printed before the
  runner starts) delivered once at the switch. Lines that scroll out of the tail between
  two polls become one note line.
- **Planned handoff**: the daemon writes `attempts/<n>/control.json` 30 min before the 12 h
  cap (or before the weekly quota runs out); the runner answers in `control-ack.json` after
  syncing, and the engine then deletes the kernel (cancel) and resumes elsewhere.
- **Cost**: ~5 HF API calls per minute per running kernel (push + control check) plus the
  daemon's tail read per 60 s poll; the free API limit is 1000 per 5 min.

## Live logs (phase 5 input)

`kaggle kernels logs <ref> -f` streams the running session over SSE and **replays from the
start** on every connect [V source]. Live check [L]: 20 s into a run, `-f` had delivered 16
lines (the same lines, in the same order, as the final blob); the non-follow call returned an
empty line. A poll-based live tail is possible: run `-f` with a short read window (the
stream never closes while running, so read ~5-10 s and kill), take whole lines, slice by
the line cursor. Needs a runner that returns partial output on timeout; not built in phase 3
per the phase plan.

## Cancel

The public API cancels a session only by `kernel_session_id`
(`/api/v1/kernels/cancel-session/{id}`), and no CLI response exposes that id [V]. The
adapter's cancel therefore deletes the kernel (`kernels delete <ref> -y`) while it is queued
or running (never after it finished, so outputs survive), leaves a tombstone in
`<home>/providers/kaggle/cancelled/`, and status() reports cancelled once the kernel is gone.
Deleting a RUNNING kernel stops its GPU session: GPU quota stopped accruing the moment the
kernel was deleted and stayed flat for 12 more minutes, past the session's own 600 s
timeout [L, live run 2]. Declared `cancel_confirms=True` (status() confirms once Kaggle no
longer has the kernel). Every push also carries `-t`, so a session can never outlive its
budget even if a delete fails. `lookup_by_key` still returns the ref after a cancel (the
local submit marker stays), which is what the engine wants for an ambiguous submit.

## Quota

Queryable [V]: `kaggle quota --format json` (above). The adapter's `quota()` is live
(`source="live"`); submit() refuses definitively (QuotaExhausted with `resets_at` =
refreshAt) when the GPU row shows 0.00h left; a quota endpoint outage does not block a push.
Kaggle has said the weekly total can float (30h+ depending on demand) [D]; always read live.

## Failure modes -> error taxonomy

| Symptom | Adapter result |
|---|---|
| CLI missing | AuthRequired (hint `uv tool install kaggle`), health auth_required |
| 401 / auth help text | AuthRequired (hint: kaggle.json or Keychain) |
| 429 (`429 Client Error`, `Too Many Requests`, `HTTP Error 429`, ...; never a bare "429": kernel refs are blanked first, since ~1 hex job id in 400 contains "429") | RateLimited (retry_after 60; definitive on push) |
| 5xx, connection error, CLI timeout | Unavailable (ambiguous on push -> lookup_by_key) |
| `Kernel push error` + "quota"/"weekly" | QuotaExhausted |
| `Kernel push error` + "phone"/"verif", or push 403 | AuthRequired (hint: phone verification) |
| `Kernel push error` + "concurrent"/"maximum number" | RateLimited (retry_after 600) |
| `Kernel push error` + "prohibit"/"violat"/"terms of service" | Permanent |
| any other `Kernel push error` | InvalidJob (reroute) |
| push error but the kernel exists anyway | Unavailable (ambiguous) |
| local metadata ValueError | InvalidJob |
| status: missing kernel (auth proven) | NotFound; tombstoned -> cancelled (checked before the rate markers) |
| ERROR, exit line N != 0 | failed, exit_code N |
| exit line 90 | lost ("dependency install failed on kaggle") |
| ERROR, no exit line, time-limit message | lost ("kaggle session time limit") |
| no exit line, quota gone | lost, quota_exhausted |

Exact server strings for quota / concurrency / phone refusals are still [I]; add them here
when first seen.

## ToS

- Free compute is for data science/ML work. Kaggle prohibits crypto mining, and using kernels as
  proxies/servers; long idle or automated abuse can get accounts suspended. [D]
- Automated submission via the official API/CLI is sanctioned (that is what it is for). [D]
- One account per person. Do not rotate accounts to farm quota. [D]
- Kernels accumulate: one private kernel per attempt stays in the account. Cleanup is not
  automated yet (phase 8 `/doctor` candidate: delete finished `gpu-router-*` kernels older
  than N days after their outputs were fetched).

## Live runs

1. 2026-09-24, smoke (`tests/contract/kaggle/test_live_smoke.py::test_live_gpu_smoke`):
   kernel `<user>/gpu-router-<job>-1`, 2xT4. pending at 2.8 s, running at 18.7 s,
   succeeded at 55.6 s after submit. nvidia-smi: `Tesla T4, 15360 MiB, 580.159.04` x2;
   `torch 2.10.0+cu128 cuda_available True devices ['Tesla T4', 'Tesla T4']`; matmul ok.
   22 log lines incl. protocol hello/total/metric/exit; exit 0; fetch returned `gpu.json`.
   GPU quota 0.00h -> 0.01h (~0.6 min at the CLI's 2-decimal resolution). Second submit with
   the same key returned the same kernel without a push; lookup_by_key found it.
2. 2026-09-24, cancel probe (`test_live_cancel_probe`, `KAGGLE_CANCEL_PROBE=1`): kernel
   `<user>/gpu-router-<job>-1`, 2xT4, `-t 600`, a 9-minute sleeper. running at 13.3 s,
   `cancel()` (= `kernels delete -y`) at 75.6 s (~0.017 h of GPU time).
   status right after: `cancelled`. GPU quota used: 0.01 h before submit,
   0.03 h at the cancel, still 0.03 h at 801.7 s after submit (sampled every
   minute; a session that kept running to its timeout would have added ~0.17 h). So a
   deleted kernel's session stops at once. Quota accrues live (0.02 h after ~60 s running).

## Sources

- Package source: kaggle 2.2.4 `kaggle/api/kaggle_api_extended.py` (authenticate, kernels_push,
  kernels_status, kernels_logs, kernels_logs_stream, kernels_output, kernels_delete,
  quota_view), `kaggle/cli.py` (main error handling), `kagglesdk/kernels/types/kernels_enums.py`,
  `kagglesdk/kernels/types/kernels_api_service.py` (CancelKernelSession needs a session id).
- CLI help: `kaggle kernels {push,status,logs,output,pull,init,delete} --help`, `kaggle quota --help`.
- Live: `kaggle quota --format json`, `kaggle config view`, status/logs of an existing kernel
  and of a missing one, delete of a missing one; the two live kernel runs above.
- https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md
- https://www.kaggle.com/docs/efficient-gpu-usage
- https://www.kaggle.com/docs/notebooks
- https://github.com/Kaggle/kaggle-cli/issues/582 (no secrets in push metadata)
- https://www.kaggle.com/discussions/product-feedback/302908 (session runtime limits announcement)
- Note: kaggle.com/docs/* pages are JS-rendered and returned only titles to WebFetch; [D] claims
  about session cap, concurrency and 20GB output come from secondary sources and prior Kaggle
  announcements.

## Integration run (phase-3 integration, 2026-09-24)

`gpu run gpucheck.py --provider kaggle --wait` through the real CLI + daemon (private home,
the CLI's own `~/.kaggle` credentials), kernel
`<user>/gpu-router-<job>-1` (private; still in the account, nothing deletes
finished kernels yet). The router now places `-p kaggle` on 2xT4 (D32). nvidia-smi
`Tesla T4, 15360 MiB, 580.159.04` x2; `torch 2.10.0+cu128 cuda_available True devices
['Tesla T4', 'Tesla T4']`; fp32 2048^3 matmul ~3.5 TFLOPS (one T4). Submit to done 65 s
wall; the run finished inside one 60 s poll, so the job was never seen `running` (hence
D33). `runs/<id>/gpucheck.json` fetched; `gpu logs` complete; live quota 0.03 -> 0.04 h of
30 h (`gpu quota` rounds to whole-ish numbers: "0/30h"; `--json` has 0.04). Kaggle
appends its nbconvert output after the job (`[NbConvertApp] ...` + SyntaxWarnings); logs
are now cut after the runner's exit line (D33, checked against this run's cached log, not
yet on a new live run).
