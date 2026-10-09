# Field test and fixes, 2026-10-04

Trigger: a user project's agent-written gpu-router field log said
Kaggle never ran a job, Lightning was logged out, `data:` could not reach Kaggle, gitignored
folders surprised every agent and weights were re-downloaded on every retry. This file is the
running log of what was checked, what was wrong and what changed. Times are local (PDT).

## Evidence from the live daemon (before any change)

- `gpu history`: 6 Kaggle jobs, 0 ran. Attempts: `kaggle push outcome unknown: kaggle kernels
  push failed: 400 Client Error: Bad Request for url:
  https://api.kaggle.com/v1/kernels.KernelsApiService/SaveKernel` (99e0: 6 attempts, cancelled
  by hand) and `... Expecting value: line 1 column 1 (char 0)` (262c, d102, 3183 attempts 2-3).
- Bundle sizes of those jobs: 1.79 MB (99e0), 1.94 MB (47c1), 3.40 MB (5a0a), 9.66 MB (262c,
  d102). The last Kaggle job that pushed fine (e611, 2026-09-25): 32 KB.
- 47c1/acb1: `dataset 'standin_v1' cannot reach kaggle: it needs Hugging Face storage
  (checkpoint.backend is local)`; HF was declined by the user on 2026-09-25 (config.yaml).
- 3183 (colab q4): 3 Kaggle push failures, then placed on colab, exit 1 from the job's own
  gdown (Drive rate limit). The router's part: each Kaggle failure put Kaggle on a 30 min
  cooldown for every job ("kaggle: cooldown 27m").
- Lightning: `● login needed`, "lightning rejected the credentials: Authentication failed".
  ~/.lightning/credentials.json is from 2026-09-24; the stored key is no longer accepted.
- `gpu route --provider local` places on the Mac (max_concurrency 1): pinning local already
  gives agents a shared queue; nothing told them.

## Root cause 1: Kaggle refuses kernel sources over ~1 MB

The adapter embeds the whole bundle base64 in run.py (`DEFAULT_MAX_EMBED_MB = 10`). Probe,
2026-10-04 22:35, kaggle CLI 2.2.4, CPU-only private script kernels with a padded source:

| run.py size | result |
|---|---|
| 933,761 B | `Kernel version 1 successfully pushed` (kernel deleted right after) |
| 1,141,256 B | `400 Client Error: Bad Request ... SaveKernel` (the exact error from the jobs) |
| 3,112,456 B | same 400 |

So any project whose bundle is over ~700 KB (x 4/3 for base64) could never run on Kaggle, and
the 400 was classed as "outcome unknown" (Unavailable): lookup, provider cooldown, retry.

## Changes and checks

(appended as the work lands)

### Fix 1: Kaggle blob datasets + a definitive 400 (adapter, runner, engine)

- run.py keeps the bundle inline only while it stays under `remote.MAX_INLINE_SOURCE`
  (900,000 B). Bigger bundles and resume archives go into a private content-addressed Kaggle
  dataset `<user>/gpu-router-{bundle,ckpt}-<sha16>` (one `.bin` file, attached via
  `dataset_sources`, sha256-checked on the kernel), uploaded once and reused.
- New optional adapter call `stage_data` (capability `stage_data`, engine timeout
  `timeouts.stage_data` 3600 s): with no HF storage, the driver asks the provider to keep
  the dataset itself. Kaggle implements it as `gpu-router-data-<sha16>` (a directory = one
  uncompressed tar, a file = the file), reused across jobs by content hash, so big weights
  passed as `data=` stop being re-downloaded per job on Kaggle. Notes: `data_uploading`
  (new reason), `data_uploaded`, `data_reused`. Slow uploads wait like storage trouble
  (`checkpoint.storage_wait_s`), refusals exclude the provider for the job.
- `400 Client Error` from SaveKernel is `InvalidJob` (definitive: excluded for this job, no
  provider cooldown, a pinned job fails at once with the reason) instead of "outcome unknown".
- Background sweep deletes blobs unused for 3 days (bundle/ckpt) or 30 days (data).
- Tests: `tests/unit/providers/kaggle/test_blobs.py` (12), `tests/unit/checkpoint/
  test_stage_data.py` (3); SimKaggle now refuses run.py over 1,000,000 B with the live 400,
  answers 403 right after a dataset create, and supports `datasets delete`. Suites: engine,
  checkpoint, providers, contract, api, core, adapters: 1423 passed, 23 skipped.

Live, adapter level (2026-10-05 00:16, real account, `scratchpad/live_kaggle.py`):
fieldtest project (a throwaway field-test project, bundle 1,535,280 B, `data/rows` 3 CSVs)

| step | result |
|---|---|
| stage_data rows | `<user>/gpu-router-data-03e835706eff28df` uploaded in 10 s |
| stage_data again | `uploaded=False` (reused) in 0.8 s |
| submit (bundle blob + push) | 13 s, kernel `gpu-router-fed34fd20103-1` |
| run | queued 30 s, running 2xT4 (`device` line: 2x Tesla T4 15360 MiB), done at +90 s |
| log | "unpacking dataset rows", "dataset rows: 3 csv files", "blob bytes 1500000", exit 0 |
| fetch | `metrics.json`: device cuda, acc 1.0, 16 s of training |

### Agent's-eye test: a separate headless Claude Code session (2026-10-05 00:21-00:27)

`claude -p` (Claude Code 2.1.289, Sonnet, `--max-budget-usd 8`), cwd
a throwaway field-test project, allowed tools = the gpu-router MCP tools + file tools,
`GPU_ROUTER_HOME` = a private tmp home, `GPU_ROUTER_PORT=0` (the MCP server auto-started its
own daemon; the user's daemon and jobs were not touched). Prompt:
`scratchpad/agent_prompt.md`; the agent's own log: `the field-test project/AGENT_LOG.md`.
Cost $0.80, 31 turns, 6m44s.

| run | job | provider | result | submit to done |
|---|---|---|---|---|
| 1 train.py + `data=rows=data/rows` | 8a4a | kaggle 2xT4 | done, cuda, 3 csv files seen | 2m23s |
| 2 same again | 3735 | kaggle 2xT4 | done; `data_reused` (no upload) | 2m09s |
| 3 `--steps 200` | 4a87 | colab T4 | done, cuda | 46s |
| 4 `--steps 100`, `provider=local` | e56a | Mac MPS | done | 25s |
| 5 gpu_route with data | - | would pick colab | (bug, below) | - |

What the agent flagged, and what changed:

- **gpu_route with data= chose colab**, which cannot receive data without HF storage (it
  would have been excluded at submit, burning an attempt). Fixed:
  `RoutingContext.data_unreachable`; both routers reject such providers up front with
  "cannot receive data= without Hugging Face storage (kaggle can: it keeps datasets itself)".
- **"putting dataset rows ... on kaggle" then "reusing it" a second later** read as a
  contradiction. Now only datasets >= 50 MB get the before-note, worded "uploading ...
  unless the same content is already there"; sizes print as `36 B`, not `0.0 MB`.
- **Every gpu_status repeated the full spec + two hashes** (~680 of ~2,800 bytes per poll).
  Polling results now carry a brief spec; `verbose=true` still returns everything.
- **poll_every_s 60 next to a 50 s wait_s cap**: capped at 50 when follow is `wait`.
- `bundle.left_out` listed `data/` although `data/rows` was the job's data=: paths passed as
  data are no longer listed; a parent gets "; data/rows/ is passed as data=".
- Not changed (noted): attempt `remote_message` keeps the last running message after a
  finish ("setting up the python env" on local); colab has no remote_url; the agent's own
  untracked AGENT_LOG.md shipped (correctly warned).

### Fix 2: visibility for people

- `gpu providers` notes moved under the table (in an 80-column pipe the note column was 4
  characters wide); a provider in cooldown now says so: "kaggle: cooling down for 27m00s
  after 3 failed calls in a row; new jobs go elsewhere meanwhile".

### Fix 3: smaller items found on the way

- `gpu route` gained `--data` and `--include` (as `gpu run`); the MCP `gpu_route` had them.
- Lightning's 401 text no longer says "Please run `lightning login`" (gpu-router keeps its
  own Keychain copy of the key, so that alone never fixes it; an agent session told the
  user to run it): "lightning rejected the stored API key (Authentication failed.); sign in
  again with `gpu login lightning`".

### Final check through the user's real daemon (2026-10-05 00:42, launchd daemon restarted on this code)

`gpu run --hours 0.15 --data rows=data/rows train.py --steps 200` from the fieldtest project,
unpinned, as an agent job (CLAUDECODE=1), user's config (`checkpoint.backend: local`):

    placed on kaggle 2xT4 (kaggle: fits 16GB (2xT4), colab cannot receive data= without
      Hugging Face storage (kaggle can: it keeps datasets itself), saved for jobs over 4h)
    dataset rows is already a private kaggle dataset ...-03e835706eff28df; reusing it
    kaggle accepted the run (<user>/gpu-router-fcb719ad3fec-1)
    gpu-router: unpacking bundle-7538362b587d5591.bin
    device cuda Tesla T4 / dataset rows: 3 csv files / done acc 1.0
    ✓ train done in 2m08s on kaggle · 2xT4  → ./runs/fcb7

Cleanup: probe kernels and the probe dataset deleted right after the probes; the two
bundle blobs made by tmp homes (no record, so no sweep) deleted by hand; the blobs the real
home made are swept on schedule (bundle 3 days, data 30 days).

### Independent review of the engine/adapter changes (fresh agent, read-only)

Six findings, all fixed with a regression test each: a transient HF outage could fail data
jobs at routing (and a refusal never re-checked); staging errors from inside the adapter
counted as internal errors and a login problem excluded the provider; Kaggle blob trouble
before the push still cooled Kaggle down (now a 60 s definitive retry); the blob sweep
could race a submit (now locked + re-read); blobs off + a big resume archive excluded
Kaggle (now a fresh start); `HF_TOKEN` without `HF_TOKEN_REMOTE` still let data jobs pick
Colab. Also: a finished attempt no longer keeps its last running message.

### Real use after the fix (2026-10-05, a user project's agents through the live daemon)

| job | provider | result |
|---|---|---|
| bef4 | kaggle | done in 12m |
| 4f90 | kaggle | done in 36m |
| c328 | kaggle | done in 27m; fetch hit a transient network error, the agent's `gpu_fetch` retry got 781 files |
| 5ede | kaggle | done in 29m |
| 8e1c | kaggle | done in 22m |
| 2519 | kaggle | done in 26m |

Kaggle quota went from 0 to 2 of 30 hours used. Found in c328's events: the `fetch_failed`
note held Kaggle's signed download URL (a JWE, `eyJ...` with an empty second part), which no
redaction pattern matched. Fixed: `secrets.TOKEN_PATTERNS` redacts JWT/JWE and the driver
redacts adapter error text before storing it (`tests/unit/core/test_secrets.py`,
`tests/unit/engine/test_flows.py::test_adapter_error_text_is_redacted_before_it_is_stored`).
The stored note holds an expired link and stays (events are append-only).

### The daemon starved on a busy Mac (2026-10-06, D62)

With other projects loading the Mac (load 95-245: Postgres imports, ffmpeg, local model
runs), the daemon got 1 s of CPU in 14 minutes and stopped answering; provider checks timed
out. Cause: the launchd agent ran it with `ProcessType: Background`. Now `Standard`; the
re-installed agent answered ready ~5 s after bootstrap at load 139. `gpu daemon
install-launchd` also retries the bootstrap while launchd still tears down the old daemon
(it had answered error 5 and left the agent unloaded).

### Colab said "login needed" whenever the network dropped (2026-10-08, D63)

37 hours on the new daemon: healthy through 56 wakes from sleep. But four times colab went
`ok -> auth_required` in the same second kaggle failed with `NameResolutionError`, and read
"login needed" for 6 to 45 minutes, until the network was back. The colab CLI prints "No valid
default credentials found" (exit 0) when the credential refresh cannot reach Google, the same
text as an expired sign-in. An auth-looking colab failure now counts as a login problem only
while oauth2.googleapis.com answers; otherwise it is an outage ("the network looks down").

## Still open

- **Lightning needs a fresh sign-in by the user**: `gpu login lightning --browser` (or
  `lightning login` then `gpu login lightning --import`). Not something a build session may
  do.
- Colab cannot take `data=` without HF storage (each attempt is a fresh VM; no per-provider
  store). The router now skips it for such jobs instead of failing there.
- (Fixed 2026-10-06) A fetch that failed on a transient network error was not retried;
  now it is retried twice (20 s, 60 s) before the job says "could not fetch".
- Not verified live: a Kaggle resume archive over the inline limit, the blob sweep against
  the real API, a multi-GB `data=` upload (time depends on the uplink; the stage call allows
  an hour and waits/retries up to `checkpoint.storage_wait_s` after that).
