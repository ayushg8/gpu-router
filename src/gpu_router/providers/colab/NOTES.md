# Colab adapter research notes

verified_at: 2026-09-23. CLI: google-colab-cli 0.7.2 (`colab version`), installed via uv tool at
`~/.local/share/uv/tools/google-colab-cli/`, symlinked to `~/.local/bin/colab`.

Legend: **[V]** verified from CLI help, installed source, local job logs or official docs.
**[I]** inferred, not confirmed.

## Summary

| Field | Value | Basis |
|---|---|---|
| GPUs (CLI flag values) | T4, L4, G4, H100, A100. TPU v5e1, v6e1 | [V] `colab new --help` |
| GPUs actually available on free tier | T4 only | [V] H100 refused on a free account (live job run by an existing Colab batch tool). T4 allocated in several jobs. [I] L4/A100 need Pro |
| VRAM | T4 16 GB (about 15 GB usable). L4 24 GB, A100 40/80 GB, H100 80 GB | [I] hardware specs, not queried |
| Free limit | Unpublished, dynamic. "Colab does not publish these limits" | [V] Colab FAQ |
| Session cap | FAQ: at most 12 h ("depending on availability and your usage patterns"). Pro+: up to 24 h. CLI keep-alive daemon stops itself at 24 h | [V] FAQ; [V] `commands/session.py` `keep_alive()` `max_duration = 24 * 3600` |
| Idle timeout | Exists, length unpublished. The CLI daemon pings keep-alive every 60 s, so the VM is not idle-pruned while the daemon lives | [V] FAQ; [V] `session.py` `time.sleep(60)` |
| Concurrency | Backend 412 means "too many active sessions, or a temporary usage or capacity limit". Free tier: assume 1 GPU session | [V] `client.py:286`, `session.py:200-212`; [I] the limit of 1 |
| Reset schedule | Unknown. The batch tool assumes "roughly a day" after a T4 refusal | [I] the batch tool's usage notes |
| Card required | No, for the free tier | [V] FAQ |
| Phone verification | Not required beyond a Google account | [I] |
| Quota queryable | Only paid compute units: `colab usage` gives balance, rate/hr and active assignments. Free GPU quota is not exposed | [V] `consumption.py`, `commands/usage.py` |

## Login and auth

- Two strategies, set by a global flag placed before the subcommand: `colab --auth=adc ...` or `--auth=oauth2`. [V]
- **Default conflict:** `colab --help` says the default is `oauth2`. README and SKILL.md say `adc`. **Always pass `--auth=adc` explicitly.** The batch tool does this (every call starts `colab --auth=adc`). [V]
- ADC (headless, preferred): the creds file is `~/.config/gcloud/application_default_credentials.json` (0600). It is minted once, by a human, with:
  `gcloud auth application-default login --scopes=openid,https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/userinfo.email,https://www.googleapis.com/auth/colaboratory`.
  If `colaboratory` is missing, keep-alive returns a 403. `colab new` pre-flights keep-alive, unassigns the VM and prints the remediation. [V] `auth.py` ~160-200, `session.py` ~255-275
- oauth2: a copy-paste remote flow (no localhost redirect). The token is cached at `~/.config/colab-cli/token.json`, and the client config is at `~/.colab-cli-oauth-config.json`. It needs a human the first time. [V] `auth.py:54-90`
- Session state: `~/.config/colab-cli/sessions.json` holds, per session: name, endpoint, token, url, kernel_id, keep_alive_pid, accelerator, machine_shape, running, last_execution. It contains **runtime proxy tokens**, so never log it. There is also a lockfile, `settings.json` (update check) and `history/*.jsonl`. [V]
- `colab whoami` (hidden command) shows the email, scopes and expiry. It is a cheap auth probe for `/doctor`, as is `colab sessions`. [V] SKILL.md
- Keychain handoff [I]: the CLI reads ADC through `google.auth.default()`, which honors `GOOGLE_APPLICATION_CREDENTIALS`. To avoid keeping plaintext ADC on disk, the adapter could write the Keychain secret to a 0600 temp file for the call and set that env var. Otherwise, leave the gcloud file alone. The adapter must never read or copy the file itself. Isolate with `--config <path>` per adapter so gpu-router state never collides with sessions other tools created. The daemon inherits `--auth` and `--config`. [V] SKILL.md

## Submit (the pattern proven by the batch tool)

`colab exec --timeout` **defaults to 30 s**, and "a long blocking exec loses its websocket" (a comment in the batch tool). Do not use `colab run` or a blocking `exec` for jobs longer than a few minutes. Use detached launch plus polling. [V]

```
C="colab --auth=adc --config <router_state>/colab-sessions.json"
$C new --gpu T4 -s gr<jobid6>                  # timeout 300 s; failure => classify (below)
$C exec -s S -f mkdir.py                       # os.makedirs('/content/gr')
$C upload -s S job.json /content/gr/job.json   # one file per call; no dir upload
$C upload -s S <entry>.py /content/gr/job.py   # plus each input file
$C upload -s S secrets.env /content/gr/.env    # secrets as a file, never argv (see below)
$C exec -s S -f launch.py                      # Popen(start_new_session=True) -> log.txt, EXIT
loop every 30 s: $C exec -s S -f poll.py --timeout 60   # prints EXIT or RUNNING + last progress lines
$C download -s S /content/gr/out/<f> <local>   # one file per call; timeout 600 s
$C stop -s S                                   # always, in finally (check=False)
```

Reference implementation: the batch tool's job loop, plus its remote `launch.py` (detached `bash -c "python -u job.py > log.txt 2>&1; echo $? > EXIT"`) and `poll.py` (reads EXIT and the log tail, and flags exit 137 as "killed: likely out of RAM" and 139 as a segfault). [V]

Why detach: a dropped websocket or kernel restart can't kill the job (`launch.py` header). [V]

Alternative for short jobs (under ~5 min) [V]: `colab run --gpu T4 --timeout N script.py args` does new, exec and stop in one step. Exit codes propagate, `[colab]` chatter goes to stderr and script stdout to stdout, and the VM is torn down even on error. A missing script fails before any VM is allocated.

## Status, logs, fetch, cancel

- status: `colab sessions` lists server-side assignments as `[name]` lines and prunes stale local entries. Orphans show as `[?]`. `colab status -s S` shows the IDLE/BUSY state and last execution. Job-level state comes from the poll.py output (EXIT file). [V]
- logs: the remote `/content/gr/log.txt` tail via poll. `colab log -s S [-n N] [-t TYPE] [-o f.jsonl]` gives CLI events (executions, file ops, keep_alive_error or keep_alive_stopped with the raw response_body). [V]
- fetch: `colab download -s S REMOTE LOCAL`, one file at a time. For many files, tar them on the VM first, then download the tarball. [I]
- cancel: `colab stop -s S` releases the VM and kills the daemon. Soft cancel: exec a `pkill -f job.py`. [V] for stop, [I] for pkill
- Checkpoints: `/content` dies with the VM. For resumable jobs, write checkpoints to Drive. `colab drivemount` is interactive-only (not agent-runnable), so the practical option is to periodically `download` checkpoints to the Mac, or push them to an HF repo from the job. [V] SKILL.md for drivemount; [I] for the approach

## Quota

- Free GPU quota cannot be queried. Estimate it: record `new --gpu T4` successes and refusals, plus GPU-hours used per rolling 24 h, and treat a refusal as "exhausted until +24 h" (the batch tool's heuristic). [I]
- `colab usage` returns `Current balance: X compute units / Usage rate: Y/hr / Active assignments: N`. Free accounts show a balance of 0, which is normal. Active assignments is useful for the concurrency check. [V]

## Passing secrets to the remote

- `exec`/`run --env KEY=VALUE` exists, but the value lands on the local argv (visible in `ps`) and in `colab log` history. **Avoid it for secrets.** [V] for the flag; [I] that it is recorded in history (history logs executions)
- Batch-tool pattern: `colab upload` the token file to the VM (e.g. `/content/<dir>/.hf_token`), and `launch.py` reads it into the child env. Its comment reads: "sent as a file, never on a command line". Upload runs with `job=None`, so the command isn't logged. Use the same approach: upload a 0600 env file, have launch.py load it and then delete it. [V]

## Failure modes

| Signature (stderr) | Source | Class |
|---|---|---|
| `Backend rejected accelerator 'X'. You may not have quota or entitlement...` (HTTP 400 on assign) | `session.py:215-226`, seen live for H100 | **reroute**. For T4 on free: quota exhausted, mark colab unavailable for ~24 h. For L4/A100/H100: permanent entitlement gap, so drop those GPUs from the plan |
| `Allocation refused (precondition failed)... too many active sessions, or a temporary usage or capacity limit` (412) | `session.py:200-212`, `client.py:286` | **retryable** once after checking `colab sessions` for leaked sessions; otherwise reroute |
| `Keep-alive pre-flight failed: your credentials are missing an OAuth scope` (403) | `session.py:255-275` | **permanent**, needs a human re-auth (`gcloud ... --scopes`) |
| 401 on any call / expired or invalid ADC | auth.py | **permanent**, needs a human |
| `Session 'S' appears to be lost (404/401). Cleaning up.` on exec/poll | `commands/execution.py:226,330,399`, seen live (a batch-tool job while the Mac slept) | **retryable** from the last checkpoint (VM reclaimed: idle, sleep or 12 h cap) |
| `keep_alive_stopped reason=consecutive_4xx_errors` in `colab log` | `session.py:500-562` | scope problem, **permanent**; or VM already gone, **retryable** |
| `keep_alive_stopped reason=time_limit_reached` | 24 h daemon cap | **retryable** with resume |
| Remote EXIT 137 / -9 | `poll.py` | **reroute** to a bigger-RAM provider (T4 VM has ~12 GB system RAM [I]) |
| Remote EXIT 139, or other non-zero exit | `poll.py` | **permanent** (user code) |
| Transient exec failure while polling | the batch tool ("poll hiccup") | **retryable**: keep polling until the deadline |
| `colab new` hangs past 300 s | the batch tool's timeout | retryable once, then reroute |

Mac sleep: the batch tool spawns `caffeinate -i -w <pid>` because "Idle sleep stops the polls, and Colab then reclaims the idle free VM mid-job". The router's poller must do the same. [V] the tool's comment; [I] that polls are the cause. The CLI keep-alive daemon also runs on the Mac, so sleep stops it too.

## ToS

- Free tier prohibits "Remote control such as SSH shells, remote desktops", "Bypassing the notebook UI to interact primarily via a web UI", distributed computing workers, mining, torrents. [V] FAQ
- The CLI is Google's own official tool, so exec, run, upload and download are sanctioned use. **Never use `colab ssh` or `console` on the free tier**, and never run a hosted server or web UI on the VM. Batch jobs only, one at a time, stop when done. [V] FAQ list; [I] the reading that the CLI itself is permitted
- Project rule: Colab free tier is batch jobs only, never a hosted server or web UI.

## Quirks

- An unrecognized `--gpu` value **silently falls back to A100**, which then usually fails. Validate against the enum before calling. [V] SKILL.md
- Always pass `-s NAME`; otherwise the name is a random 6-hex string. [V]
- Kernel state persists across `exec` calls in one session. The working dir is `/content`. [V]
- `colab exec -f` sends the file content, so no upload is needed for the entry script. Data files still need `upload`. [V]
- The CLI prints update-nag lines ("new version of Colab", "colab update", "enable_update_check"). Filter them from logs (the batch tool does). [V]
- `repl`, `console`, `auth` and `drivemount` hang without a TTY. Never call them. [V]
- `--high-mem` needs Pro. [V]
- The batch tool also reuses a warm session between jobs to keep downloaded weights. It is optional for the router. [V]
- At research time `~/.config/colab-cli/sessions.json` held live sessions another tool had created. The router must use its own `--config` and never stop sessions it didn't create. [V]

## Sources

- `colab --help`, `colab <cmd> --help`, `colab readme`, `colab skill` (v0.7.2)
- `~/.local/share/uv/tools/google-colab-cli/lib/python3*/site-packages/colab_cli/{auth.py,client.py,consumption.py,commands/session.py,commands/execution.py,commands/run.py,commands/usage.py}`
- An existing (unpublished) Colab batch tool: its job loop, remote `launch.py` / `poll.py`, and its job logs
- https://research.google.com/colaboratory/faq.html
- https://developers.googleblog.com/introducing-the-google-colab-cli/ (no limits or ToS details)
- https://github.com/googlecolab/google-colab-cli

## Adapter implementation (phase 3)

Code: `providers/colab/{adapter,cli,remote,state}.py`; registry entry point
`adapters/colab.py` (re-export). Tests: `tests/unit/providers/colab/` (simulated CLI
`fake_colab.py`, the real bootstrap runner), `tests/contract/test_colab_contract.py`
(shared contract suite: `colab-sim` always, `colab` opt-in), `test_live.py` (live smoke).

- **One session per attempt.** Name = `gr-<job>-<n>` (the attempt key with `gpu` -> `gr`),
  so `lookup_by_key` needs no CLI call and a restarted daemon finds the run by name.
- **Private session file.** Every call is `colab --auth=adc --config
  <home>/providers/colab/colab-cli/sessions.json ...`. The adapter never reads that file
  (runtime proxy tokens); it chmods it 0600 after `new`. Sessions made by other tools or by
  hand live in `~/.config/colab-cli/sessions.json` and show up only as `[?]` rows in
  `colab sessions`; the adapter only ever stops names it created and recorded.
- **Private CLI home (D37, phase-3 review).** Every call also runs with
  `HOME=<home>/providers/colab/cli-home` (0700) and `CLOUDSDK_CONFIG` = the real
  `~/.config/gcloud` (unless already set), so ADC still resolves (verified live with
  `sessions`, 2026-09-24). The CLI's `colab.log` (urllib3 DEBUG: every contents-API URL
  with its `colab-runtime-proxy-token`) and `history/<session>.jsonl` (every exec's code
  AND output: live log reads come back base64, so no redaction can see them) are created
  there instead of 0644 files under the real `~/.config/colab-cli/`. The history file of
  one of our sessions is deleted as soon as it is stopped; `colab.log` is truncated past
  5 MB. Files the CLI wrote under the real `~/.config/colab-cli/` before this change
  (`history/gr-*.jsonl`, `colab.log`) are not touched by the adapter; delete them by hand.
- **Run record** `<home>/providers/colab/runs/<session>/record.json` (no secrets): written
  BEFORE each remote step (`creating` before `new`, `setup` before uploads, `launched`
  after the detached launch), plus the harvested `job.log`, `outputs.tar.gz` and mirrored
  `ckpt/`. States: creating, setup, launched, exited, cancelled, lost, rejected, abandoned.
- **Submit** (budget 270 s < engine 300 s): validate GPU against `SUPPORTED_GPUS` AND the
  catalog (a typo would silently become A100) -> reap our own finished-but-running
  sessions -> `new --gpu T4 -s NAME` (180 s) -> `exec prepare` (mkdir, `nvidia-smi` must
  show a GPU) -> `upload bundle.tar.gz` (+ `resume.tar.gz`, + `.secrets.json`) -> `exec
  launch` (extracts `gpu_runner/bootstrap.py` from the bundle, `Popen(bash -c "python
  bootstrap.py ... > console.txt 2>&1; echo $? > RC", start_new_session=True)`, writes
  `launched.json`). Launch is idempotent (an existing `launched.json` is returned as is).
  Any failure after `new` stops the session and raises Unavailable (the record says
  `abandoned`, so the engine's `lookup_by_key` resolves it to "no run"). Leftover
  `tmp/.secrets-*.json` / exec scripts (a daemon killed mid-upload) are swept at adapter
  start and at every submit (D37).
- **Orphaned `colab new` (D35).** The CLI runs in its own session, so a `new` that was
  running when the daemon died keeps going and registers the session when the VM is
  assigned. Its pid is in the record (`cli_pid`, set via `ColabCli.run(on_spawn=...)`). A
  restarted daemon keeps such an attempt PENDING while that pid is alive and still a
  `colab ... new ... gr-<name>` (checked with `ps`), or, with no pid, for
  NEW_TIMEOUT_S + 30 s after `created_at`; past STALE_START_S (600 s) it kills the verified
  orphan. Then a "not found" still ends in the idempotent `colab stop -s NAME`, never in
  `stopped=True` on its own. A retried submit of such a record raises Unavailable instead
  of racing a second `new`; cancel kills a verified orphan before stopping.
- **Failure mapping.** 400 "Backend rejected accelerator" -> QuotaExhausted(resets_at =
  now + 24 h, recorded for `quota()`); 412 "Allocation refused" -> Unavailable (the
  account's one free GPU may be held by another tool); missing scope / no or expired ADC /
  missing CLI -> AuthRequired with the fix as hint; `new` timeout or an unknown failure ->
  stop the name, Unavailable (ambiguous). Remote exit 90 (pip install failed) and 137 (OOM)
  -> LOST (reroute); other non-zero exits -> FAILED with the code. Throttling is matched
  on real signatures only ("429 Client Error", "Too Many Requests", RESOURCE_EXHAUSTED,
  ...), with our `gr-...` names blanked first: `new` echoes the session name and about 1
  hex job id in 400 contains "429" (phase-3 review).
- **Status** = one `exec poll` (reads EXIT / RC / pid liveness; `/proc/<pid>/stat` state Z
  counts as dead because the kernel never reaps the detached bash). "Session ... appears to
  be lost (404/401)" or "Session ... not found" -> LOST (VM reclaimed). The first poll that
  sees the exit also packs `job.log.gz` (+ `outputs.tar.gz` for exit 0) in the same exec,
  downloads them (outputs only up to 64 MB inside status), then stops the session. Every
  harvest step leaves 12 s of the call's budget for `colab stop`, and the final checkpoint
  of a SUCCESS is not mirrored (nothing resumes a finished job). Bigger outputs keep the
  session until `fetch()` pulls them; after 1 h without a fetch the session is stopped.
- **Teardown retries (D36).** Whatever the terminal status() could not finish (a failed
  `colab stop`, a short budget) is finished by the run's next `logs()` or `fetch()` (both
  settle whenever the record is final and not stopped, whatever the cached flags say) or
  `cancel()`, and otherwise by a **janitor thread** (every 120 s, started by any call that
  leaves such a record and by the healthcheck the daemon runs at start) until every
  finished session is stopped. It gives up 13 h after the run ended (the 12 h cap has
  ended the VM). `close()` ends it. The reaper at the next submit is still there.
- **Cancel** of a launched run first runs one `exec poll`: if the runner already exited
  (polls are 30 s apart), the run is recorded as exited and settled like a status() would
  (log, outputs, stop), so the engine's "already finished -> fetch, outputs_kept" path
  works. A finished success whose outputs are not pulled yet keeps its session for fetch.
- **Logs**: cursor `"<lines>:<byte offset into job.log>"`. While the VM is up, `exec
  logread` returns up to 256 KiB per read (base64); after harvest the local copy is served.
  Both paths split lines with `remote.split_log_bytes` (a trailing fragment waits until the
  run ended; lines over 64 KiB are cut at 64 KiB from the line start), so the same bytes
  give the same lines either way. The harvested raw `job.log` is deleted 1 h after
  `logs()` first served it to eof (`log_served_at` / `log_purged` in the record; the
  engine keeps its own redacted copy); later reads return eof with no lines (D37).
- **Checkpoints**: bootstrap archives to `<run>/ckpt-sync/ckpt-NNNN.tar.gz` (URI
  `file:///content/gr/<session>/ckpt-sync/...`). The adapter mirrors the newest archive up
  to 64 MB to the Mac on each poll and at exit; `resume_from` resolves to a Mac-local file
  or that mirror and is uploaded; otherwise the attempt starts fresh (record `resume:
  unavailable`, a warning in the daemon log). `--ckpt-seq-start` = resume seq + 1.
- **Test mode**: with `test_mode` on, the adapter never runs the real CLI unless
  `GPU_ROUTER_REAL_PROVIDERS` lists colab or the provider setting `cli` points somewhere
  (the simulator). Inert: healthcheck DISABLED, submit InvalidJob, no subprocess.
- **Provider settings** (config.yaml `providers.colab`, all optional): `cli` (argv or path),
  `remote_root` (default `/content/gr`), `max_bundle_mb` (200), `heartbeat_s` (60).

### CLI facts verified in the 0.7.2 source while building this

- `colab exec` exits 0 even when the code raised in the kernel (errors go to stderr as a
  traceback). Scripts therefore print one `@@GR:<base64 JSON>` result line and the adapter
  trusts only that.
- `colab sessions` with an EMPTY local store swallows auth failures: stderr "No valid
  default credentials found", stdout "No active sessions found on server.", exit 0. The
  healthcheck looks for that stderr line.
- The CLI prints the same "No valid default credentials found" when the credential refresh
  cannot reach Google at all (no network after a wake; reproduced 2026-10-08 with an
  unreachable proxy: exit 0, no network error in the text). So every auth-looking failure is
  checked against `cli.google_reachable()` (a 4 s connection to oauth2.googleapis.com:443,
  on a thread, IPv4 first; True when `requests` would use a proxy, incl. the System Settings
  one): unreachable = Unavailable "the network looks down", not AuthRequired (D63).
- `colab stop -s NAME` for a name not in the store prints "Session 'NAME' not found." and
  exits 0. If the VM is already gone, `unassign` raises (traceback, exit 1) after the
  keep-alive was killed; the adapter treats a 404 there as stopped.
- `upload` into a directory that does not exist fails (Jupyter contents API), so `prepare`
  runs first. Upload/download move whole files as base64 JSON: fine for bundles (capped at
  200 MB) and small outputs, slow for GB-sized files.
- The CLI records every exec's code and outputs in `~/.config/colab-cli/history/
  <session>.jsonl` (regardless of `--config`), and it always logs API request and response
  bodies at DEBUG to `~/.config/colab-cli/colab.log` (`setup_logging` sets the root logger
  to DEBUG), which includes the runtime proxy token from `assign`. Both paths come from
  `os.path.expanduser("~")`, i.e. `$HOME`: the adapter points HOME at its private
  cli-home (see above). Never put secrets in exec code; the adapter uploads them as a file
  that `launch` reads and deletes.
- `colab new` (commands/session.py) echoes "Creating session", assigns, `store.add()`s,
  spawns the keep-alive and only then echoes "Session READY."; with the parent gone that
  last echo fails (EPIPE) after the session exists (D35).
- Error URLs printed by `upload`/`download` failures include
  `colab-runtime-proxy-token=...`; `cli.redact` strips it before any message is built.

### Live smoke, 2026-09-24 (tests/unit/providers/colab/test_live.py, one run)

- Pre-flight: `colab sessions` (private config) showed no active sessions; healthcheck OK.
- `submit` (new + prepare + upload + launch) returned in **28 s**: session
  `gr-<job>-1`, `nvidia-smi`: **Tesla T4, 15360 MiB**; torch 2.11.0+cu128, CUDA 12.8
  preinstalled on the VM, so `deps: none` jobs start at once.
- Each running `status()` (one `exec poll`) took about 1 s. The terminal `status()` (poll +
  pack, download job.log.gz and outputs.tar.gz, `stop`) took about 3 s.
- fp16 4096x4096 matmul sustained **~17.9 TFLOPS** on the T4. The check itself ran 266 s,
  not 30 s: the script timed 1-second steps without synchronizing, so each step drained
  ~1000 queued kernels. Fixed in the test (sync every 8 matmuls); the adapter was not
  involved. Wall time of the whole smoke: 311 s.
- After the run: record `stopped: true`, `colab sessions` (private config) listed nothing,
  a second listing after the test said "No active sessions found on server.", and no
  `colab_cli keep-alive` process was left on the Mac. `fetch` returned the output file.

## Checkpoint storage (phase 5)

Code: `gpu_router/checkpoint/` + the runner's `gpu_runner/storage.py`; the adapter change is
only that an `hf://` resume checkpoint is not uploaded (record `resume: "storage"`). Not
verified live yet (no HF token on the build machine): [I] until the first live run.

- **Token path**: the existing secret channel. The engine adds the job secret
  `GPU_STORAGE_TOKEN` (Keychain `HF_TOKEN_REMOTE`, else `HF_TOKEN`); it goes up in the 0600
  `.secrets.json` that launch reads and deletes (never argv, never exec code), and bootstrap
  pops it from its environment before the job starts. Non-secret env (`GPU_STORAGE`,
  `GPU_RESUME_URI`, `GPU_STATUS_PUSH_S`, ...) rides in the launch params as before.
- **Runner**: pip-installs `huggingface_hub>=1.32,<2` into a private `--target` dir if the
  image's copy lacks the bucket API (the job keeps the image's version), then publishes
  checkpoints to `jobs/<job>/ckpt-NNNN/` in the bucket instead of `ckpt-sync/`, so the Mac
  mirror (64 MB cap) is no longer needed when storage is on; a new VM downloads the
  checkpoint itself. It also pushes heartbeat/log-tail (redundant here: Colab logs are
  live through `exec`) and answers planned-handoff requests 30 min before the 12 h cap.
- Without a token nothing changes from phase 3 (mirror path, `file://` URIs).

## Integration run (phase-3 integration, 2026-09-24)

`gpu run gpucheck.py --provider colab --wait` through the real CLI + daemon (private home,
ADC), one job, session `gr-<job>-1`: placed 08:33:18Z, submit
(new + prepare + upload + launch) confirmed at +17.7 s, running at +19.4 s, done with
outputs fetched and the session stopped at +52.7 s (the done message says 33 s of run). nvidia-smi
`Tesla T4, 15360 MiB, 580.82.07`; Python 3.13.15; `torch 2.11.0+cu128 cuda_available True
devices ['Tesla T4']`; fp32 2048^3 matmul ~3.8 TFLOPS. `pip install -r requirements.txt`
(torch) was all "already satisfied". `runs/<id>/gpucheck.json` fetched; `gpu logs`
complete. Afterwards: record `stopped: true`, `colab sessions` with the private config and
with a throwaway config both said "No active sessions found on server."
