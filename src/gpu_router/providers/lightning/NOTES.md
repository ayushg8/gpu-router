# Lightning AI adapter notes

verified_at: 2026-09-23 (docs + SDK source); live account checks 2026-09-25 ("Live run" below)
SDK checked: `lightning-sdk==2026.9.18.post1` (via `uv run --no-project --with lightning-sdk`).
Account: a free personal account (its teamspace `<user>/<teamspace>` was picked automatically as the
only one), credentials in Keychain via `gpu login lightning --import`.
lightning.ai pricing/docs pages are JS-rendered; WebFetch and curl got only a loading shell. Free-tier
numbers below come from search-engine snippets of lightning.ai docs plus third-party summaries.

Legend: **[V]** verified from installed SDK source or CLI `--help`. **[W]** official lightning.ai text seen
via search snippet. **[3P]** third-party site only. **[I]** inferred, check live at signup.

## How the adapter works (phase 7a, 2026-09-24)

Code: `adapter.py` (LightningAdapter), `sdk.py` (SdkBridge: runs the SDK in its own env),
`driver.py` (runs INSIDE the SDK env, never imports gpu_router), `launch.py` (runs INSIDE the
job), `credentials.py`, `login.py` (`gpu login lightning`). Tests: `tests/unit/providers/lightning/`
(adapter over SimLightning, driver against a fake `lightning_sdk` with the real signatures, the
launcher run for real locally, an offline shape check of the real pinned SDK) and
`tests/contract/lightning/` (every shared contract test over SimLightning; the real account and
`test_live_smoke.py` behind `GPU_ROUTER_REAL_PROVIDERS=lightning`).

- **SDK process.** Each adapter call = one `python -s -u driver.py` in an env with
  `lightning-sdk`: settings `python`, else `~/.local/share/uv/tools/lightning-sdk/bin/python`
  (`uv tool install lightning-sdk`), else `uv run --no-project --python 3.12 --with
  lightning-sdk==2026.9.18.post1 python` (settings `sdk_version`; uv caches it, ~0.3 s + ~2 s SDK
  import per call once cached). New session per call; a timeout kills the whole group (uv and
  the interpreter), so no SDK call outlives the engine's budget. JSON request on stdin, one
  `@@GRL:<base64 JSON>` result line on stdout, SDK chatter on stderr.
- **Credentials.** Keychain `LIGHTNING_USER_ID` + `LIGHTNING_API_KEY` (settings `login_source`:
  auto = Keychain, else the daemon's env, else ~/.lightning/credentials.json read by gpu-router;
  config.yaml refuses the older key name `credentials` as "looks like a secret", D51).
  They reach the child only as env vars; the child env is an allowlist (PATH, locale, CA
  bundles, proxies, UV_*), inherited `LIGHTNING_*` identity vars dropped. Before importing the
  SDK the driver moves HOME, `LIGHTNING_CREDENTIAL_PATH` and `LIGHTNING_SETTINGS_PATH` into
  `<home>/providers/lightning/sdk-home` (0700), disables the version check and sets
  `BROWSER=true`; with no credentials it never imports the SDK (which would open a browser).
  HTTP errors are reported as status + reason + body only: `str(ApiException)` also prints the
  response headers (cookies).
- **Login.** `gpu login lightning` (no-echo prompts; offers ~/.lightning/credentials.json when it
  exists), `--stdin` (two lines, or the credentials.json document), `--import`, `--browser`
  (+ `--no-open`), `--no-check`, `--json`. `--browser` is lightning.ai's own CLI sign-in (the
  flow `lightning login` uses, [V] `AuthServer.login_with_browser`): open
  `https://lightning.ai/sign-in?redirectTo=http://localhost:<port>/login-complete`; lightning.ai
  redirects to it with `token`, `key` (API key) and `userID`. Our one-shot stdlib server on
  127.0.0.1 keeps the pair in memory, logs nothing, answers 303 to `/login-complete/done` (so the
  address bar ends without the values) and nothing is written to ~/.lightning (the SDK's own flow
  writes the file and forwards the key to lightning.ai/me/apps in the URL). Verified live
  2026-09-24: an already signed-in browser redirects at once, no click needed. Verifies with one `whoami` (a rejection stores nothing; unreachable = stored with a
  note; the first check may take a minute while uv fetches the SDK). Never argv, never printed.
- **Teamspace / Studio.** settings `teamspace` (owner/name), else the account's only teamspace
  (cached 24 h per user id); several = auth_required naming them. Jobs run in the environment of
  the Studio settings `studio` (default `gpu-router`), created once with `create_ok` inside
  `skip_studio_setup()` (creating does not start it, and the SDK's keep-alive thread for a
  running Studio is never started). `create_studio: false` = use an existing Studio only.
- **One attempt = one job `gr-<job_id>-<n>`.** Submit uploads to the drive folder
  `uploads/gpu-router/<name>/`: `launch.py`, `launch.json` (env, seq start, checkpoint interval,
  wall clock, sha256s; no secrets), `bundle.tar.gz`, `resume.tar.gz` (a checkpoint FILE on the local
  Mac; hf:// resumes are downloaded by the runner), `secrets.json` (job secrets + the storage
  token; staged 0600, deleted locally right after the call). Then `Job.run(name, machine=T4|L4,
  studio, teamspace, command, env={GPU_ROUTER_ATTEMPT_KEY}, interruptible=False,
  max_run_attempts=1)`. The driver first returns an existing job of that name (Lightning would
  otherwise create `<name>-xyz` [V: `Job._submit` warns "was already taken"]; if that ever happens
  the duplicate is stopped + deleted). Credits below 0.05 = QuotaExhausted before any upload.
  Failures before the create call ("pre"), or a 4xx to it that a lookup right after confirmed
  (no job of that name: "run"), are definitive; anything once Job.run was called ("post": the
  SDK keeps calling the API after its create, e.g. `job.link`, and a status read of the created
  job can fail) is Unavailable and resolved through `lookup_by_key` (the job name) (D54). A `runs/<name>.submitting` intent keeps a
  dead daemon's still-running submit from being doubled (T_SUBMIT + 60 s).
- **Job command.** `D=/teamspace/uploads/gpu-router/<name>`; if `$D/launch.py` is missing, the
  files are downloaded with the Studio's own SDK into `$TMPDIR/gpu-router-dl/<name>`; then `exec
  python -u $D/launch.py`. The launcher copies + checks the bundle, runs bootstrap in its own
  process group (checkpoint sync dir, seq start, interval, `--resume`), relays its output to the
  job log, removes LIGHTNING_* credentials from the job's env, hands the storage token over in a
  0600 file, deletes `secrets.json` (and asks the drive to forget it), and at the end packs
  GPU_OUTPUT_DIR into `outputs.tar.gz` delivered to the drive folder's `out/` (SDK upload, else a
  copy into the mount) and under `<cwd>/gpu-router/<name>/` (for the job's artifacts).
- **Wall clock.** Jobs have no limit of their own, so the catalog now says `session_hours: 4`
  and the launcher stops the runner after 4 h - 5 min (`provider_options.lightning.timeout_s`
  shortens it; recorded as `session_s` in RemoteRef.meta for the engine's handoff planning):
  WALL_MARK line, SIGTERM (bootstrap syncs a final checkpoint), SIGKILL after 90 s; the adapter
  reads it as LOST "wall-clock limit" (the job migrates). Backstop: status() of a job still
  Running 15 min past that stops it and reports LOST once the stop took effect (Stopping,
  Stopped, confirmed); until then it reads as running ("gpu-router is stopping it") and stays
  `stop_pending` in `runs/`, retried by the next status(), healthcheck() and quota() (D54)
  (a deliberate A6 exception like Colab's stop-inside-status, D31).
- **Status.** Pending/NotCreated -> pending, Running -> running, Stopping -> running
  ("stopping"). Terminal: judged once from the log (`::gpu:: exit`, also behind a platform
  prefix) and cached redacted in `final/`: 0 succeeded, 90 lost, other codes failed, WALL_MARK
  lost, Stopped after our cancel cancelled, Stopped with a credits message lost +
  quota_exhausted, interrupted lost (preempted), Stopped by someone else cancelled, Failed with
  no exit line lost. An empty final log waits up to 10 min before a verdict. A whole-log read
  that times out is never an empty log: the driver then reads the last 500 lines
  (`job.logs(tail=500)`, where the exit line and WALL_MARK are) and the verdict is cached with
  `lines_partial` (logs() serves nothing from a tail and upgrades the cache when a whole read
  works); a timeout with no tail = Unavailable, no verdict (D54).
- **Logs.** Snapshot of the job log (`job.logs(follow=False)`, bounded to 20 s in the driver),
  cursor = lines served; final log from the cache. **Fetch.** Drive `out/outputs.tar.gz`, else a
  `gpu-router/.../outputs.tar.gz` among the job's artifacts; safe extract (regular files, no
  absolute or `..` paths) and copied into dest. A success with no archive = partial result.
  **Cancel.** `job.stop()` bounded to 20 s (the SDK's stop loops until terminal) + a local mark;
  Unavailable (the engine retries before it migrates) when the stop is not confirmed and the
  job still reads Running (D54). **Uploads** of one submit must fit T_SUBMIT: bundle +
  resume above (200 - 60) s x `upload_mbps` (default 8 Mbit/s = 140 MB) is InvalidJob, and the
  advertised max_bundle_mb is capped to it. **Machines**: an explicit
  `provider_options.lightning.machine` must be T4, T4_SMALL or L4 and the GPU the router placed
  on (L40S / multi-GPU machines have rates the router does not know). Stale `staging/` dirs
  (plaintext secrets of a daemon killed mid-submit) are removed at adapter start and at each
  submit unless their name has a live `.submitting` intent (D54).
- **Quota.** Credits, `source: live`: the balance from `billing_service_get_user_balance` when
  it answers (used = the month's job costs, limit = balance + used), else the month's job costs in
  the teamspace from Lightning (`total_cost`, USD ~= credits; Studio time not included). Resets
  at the next calendar month UTC. Rates per GPU from `list_machines` in `detail.rates`.
- **Still not verified live:** the exact out-of-credits error text (a free account
  may NOT run L4: the create answers 403 although `list_machines` lists it, D56), the monthly top-up (amount, day), which branch of the job command ran (the launcher now
  logs `(files: drive mount|sdk download)`; the 2026-09-25 run predates that line, but its
  outputs could not be copied into `<folder>/out`, which only fails on the read-only mount).

## Live run (2026-09-25)

One T4 job through `LightningAdapter` (`test_live_smoke.py`'s job; its pytest process died
with the session, so status/logs/fetch/cancel were finished by a fresh adapter on the same
home, then again from an EMPTY home via `lookup_by_key`, as a restarted daemon would):

- job `gr-<job>-1`, Studio `gpu-router` (created by the adapter, left Stopped), created
  07:00:03Z, running 07:00:43Z, Completed 07:03:13Z. Log: "Snapshotting Studio" (11 s),
  "Machine available after 28s", then our launcher: `Tesla T4, 15360 MiB`, driver 580.178.04,
  torch 2.8.0+cu128 with CUDA, matmul ok, 6 steps, `::gpu:: exit 0`, outputs (1 file) ->
  drive; fetch returned `gpu.json` from the drive's `out/outputs.tar.gz`. So: Studio jobs have
  an authenticated SDK (outputs upload worked) and `/teamspace/studios/this_studio` is writable.
- credits: balance 5.0 -> 4.951 (job `total_cost` 0.0433; it read 0.0 at completion and 0.022
  a minute later, so billing lags). Effective ~1 credit/h for a 2.5 min job; list rates from
  `list_machines`: T4 0.55 on-demand / 0.574 interruptible, L4 1.68 / 1.30 credits/h.
- balance API (`billing_service_get_user_balance`) answers for a personal account: `balance`
  (credits, float), `total_spent` (settled spend; lags the balance), `transactions` ([]).
  A new account starts at **5.0 credits**, not 15: the snapshot's limit is now credits left +
  the month's job costs (was max(15, balance), which showed "10 of 15 used" on an untouched
  account).
- the API returns timestamps as ISO strings (`Job.started_at` "2026-09-25T07:00:43Z",
  `V1Job.created_at` with microseconds), not datetimes: the driver parsed only datetimes, so
  status had no start time and the month filter of the quota counted every job. Fixed in
  `driver.epoch`; the fake SDK now returns strings too.
- an orphaned job (test process killed) kept running until the job itself ended: the live
  smoke now sets `timeout_s: 900`, so the in-job wall clock bounds any orphan to 15 min.

Integration run (phases 7-8, 2026-09-25, D53): `gpu run --hours 0.25 gpucheck.py --provider
lightning --wait` through the real CLI + an auto-started daemon on a tmp home (Keychain
credentials): one job (`gr-<job>-1`), accepted 4 s after submit, running after
2m09s (snapshot + machine start), gpucheck ran ~1 min (`pip install -r requirements.txt`
found torch 2.8.0+cu128 already in the Studio env), Tesla T4 15360 MiB, CUDA, 5 matmul
steps, exit 0, `gpucheck.json` in `runs/<id>/`; ~0.046 credits by the balance (4.951 ->
4.905). L4 is priced on its own in the catalog now (`quota_per_gpu_hour_by_gpu: {L4:
1.68}`), still not run live.

## Summary

| Field | Value | Confidence |
|---|---|---|
| GPUs usable on free credits | T4 16GB (`Machine.T4`, slug `lit-t4-1`), L4 24GB (`Machine.L4`, `lit-l4-1`); also A10G/L40S burn faster | machine names [V], free eligibility [3P] |
| Free limit | New account: **5.0 credits** (live 2026-09-25, matches the official "5 free credits upon registration"). 15 credits/month top-up after phone verification is [3P] only. | 5 at signup [V live]; monthly top-up unverified |
| T4 burn | list rate 0.55 credits/hr on-demand, 0.574 interruptible (live `list_machines`); a 30 s job cost 0.043 (snapshot + machine start billed) | [V live] |
| L4 burn | 1.68 credits/hr on-demand, 1.30 interruptible (live `list_machines`), ~3x T4 | [V live]; but see L4 access |
| L4 access | free account: `Job.run(machine=Machine.L4)` -> `jobs_service_create_job_with_http_info ... response: 403` (twice, one job, 2026-09-25; T4 creates work). Catalog offers T4 only; the adapter turns that 403 into InvalidJob (D56) | [V live] |
| Session cap | Free Studio restarts every 4 h. No cap found for Jobs in SDK; `max_runtime` only affects DWS reservations | Studio [3P], Job [V: no SDK cap] |
| Concurrency | "up to 2 concurrent GPUs" on free | [3P]; set router `max_concurrency: 1` |
| Reset | Monthly top-up to 15; credits expire at month end, no rollover. Anchor (calendar month vs signup date) unknown | [3P]/[I] |
| Card required | No (card optional, unlocks bonus credits) | [W]+[3P] |
| Phone verification | Yes, non-virtual number required to unlock free credits | [W] |

## Login / auth (headless)

- Credentials come from env first, then `~/.lightning/credentials.json` [V: `lightning_cloud/login.py`]:
  - `LIGHTNING_USER_ID` + `LIGHTNING_API_KEY` -> HTTP Basic `user_id:api_key`. `LIGHTNING_AUTH_TOKEN` (JWT) wins if set.
  - File path overridable with `LIGHTNING_CREDENTIAL_PATH`; settings at `LIGHTNING_SETTINGS_PATH` (default `~/.lightning/settings.json`).
  - Also read: `LIGHTNING_TEAMSPACE`, `LIGHTNING_ORG`, `LIGHTNING_USERNAME`, `LIGHTNING_CLOUD_URL` (default `https://lightning.ai`), `LIGHTNING_DISABLE_VERSION_CHECK`.
- **Trap [V]:** `Auth.authenticate()` with no env creds and no file **opens a browser and starts a local auth
  server**. The adapter must never call the SDK without creds present; check env first and raise
  `AuthMissing` instead.
- Keychain handoff: user runs one-time `gpu login lightning`, pastes the user id + API key (lightning.ai >
  Settings > Keys [I]) into Keychain via `secrets.py`. The adapter runs the SDK in a subprocess with
  `LIGHTNING_USER_ID`/`LIGHTNING_API_KEY` injected into that child env only, plus
  `LIGHTNING_CREDENTIAL_PATH=<GPU_ROUTER_HOME>/lightning/none.json` so it never writes `~/.lightning`,
  and `LIGHTNING_DISABLE_VERSION_CHECK=1`. Never log env.
- Interactive alternative: `lightning login` (browser) writes `~/.lightning/credentials.json` [V: CLI has `auth`/`login`].

## Submit

Two env modes; exactly one required [V: `Job.run`]:
1. **Studio env** (`studio=`): runs `command` inside a snapshot of an existing Studio. `create_ok=False`, so
   the Studio must exist already. Artifacts land in teamspace drive `jobs/<job name>/` [V: `_artifacts_drive_path`].
2. **Docker image** (`image=`): `command` runs via `sh -c`. Keeps **no artifacts** unless the spec has
   `artifacts_destination`, which `Job.run` does not expose [V]. So outputs must be pushed elsewhere
   (teamspace upload, HF Hub) by the job itself.

Recommended: create one Studio once (`gpu-router`), upload the job bundle to the teamspace drive, run as a Studio job.

```python
from lightning_sdk import Job, Machine, Teamspace, Studio

ts = Teamspace("gpu-router", user="<username>")  # or teamspace="<owner>/<name>"
ts.upload_folder("<bundle_dir>", "gpu-router/<job_id>")  # [V] Teamspace.upload_folder
job = Job.run(
    name="gr-<job_id>",  # must be unique per teamspace [V]
    machine=Machine.T4,  # or Machine.L4
    studio="gpu-router",
    teamspace=ts,
    command="cd /teamspace/uploads/gpu-router/<job_id> && bash run.sh",  # mount path [I]
    env={"GPU_ROUTER_JOB_ID": "<job_id>"},
    interruptible=False,  # True = ~80% cheaper, preemptible
    max_run_attempts=1,  # router owns retries
    tags=["gpu-router"],
)
```
CLI equivalent [V]: `lightning job run --name gr-<id> --machine T4 --studio gpu-router --teamspace <owner>/<ts> --command "..."`.
(`--env`, `--interruptible` flags exist per SDK params [I: not all CLI flags inspected].)

## Status / logs / fetch / cancel [V]

- Reattach: `Job("gr-<id>", teamspace="<owner>/<ts>")` (404 -> `ValueError "does not exist"`).
- `job.status` -> `lightning_sdk.Status`: `NotCreated, Pending, Running, Stopping, Stopped, Completed, Failed`.
  Map: Pending/NotCreated -> queued, Running -> running, Stopping -> cancelling, Completed -> succeeded,
  Failed -> failed, Stopped -> cancelled (or preempted if interruptible [I]).
- `job.started_at`, `job.stopped_at`, `job.current_run_attempt`, `job.link` (web URL).
- Logs: `job.logs(tail=200)` snapshot, `job.logs(follow=True)` iterator, server filters `since/until/query/severity`;
  `job.download_logs()` writes full log file and returns path. CLI: `lightning job logs`.
- Fetch: `job.list_artifacts(recursive=True)`, `job.download_artifacts(target_dir)` (Studio jobs only);
  `job.artifacts_uri` gives `lit://owner/ts/jobs/<name>` for `lightning cp`.
- Cancel: `job.stop()` (no-op if terminal); `job.delete()` to clean up. CLI `lightning job stop|delete`.
- `job.wait()` is a blocking sleep loop; don't use it, poll via the router.

## Quota

- **Queryable, partially [V]:** `Organization.get_monthly_summary()` returns
  `total_credits_remaining/purchased` per month, but only for orgs. Personal-account balance: no public SDK
  method found; `lightning_sdk.api.billing_api` has `get_activity/get_resource_activity/get_session_activity`
  (usage records) [V names only].
- Per-hour rate is queryable: `Teamspace.list_machines()` returns `Machine.cost`, `interruptible_cost`,
  `wait_time` [V fields; units assumed credits/hr I].
- Plan: ledger estimates `elapsed_hours * machine.cost` from the rate at submit; reconcile with billing
  activity when available. Unit = credits, limit 15, reset monthly.

## Passing secrets to the remote

- `Job.run(env={...})` sets env vars in the job [V]. Likely visible in job spec/UI [I]; use only for non-secret values.
- Encrypted secrets: `Teamspace.set_secret(key, value)` / `User.set_secret(key, value)` [V]; keys must be
  `[A-Za-z_][A-Za-z0-9_]*`. Values are write-only after creation. Injected into Studios/Jobs as env vars [I].
  Router: push once from Keychain via `gpu secrets push lightning`, reference by name.

## Failure modes

| Signal | Class |
|---|---|
| No creds -> SDK would open browser | permanent (AuthMissing), guard before call |
| `ApiException` 401/403, `PermissionError` from `raise_access_error_if_not_allowed` | permanent (reauth) |
| Out of credits / phone not verified (exact error unknown [I]) | reroute; mark quota exhausted until reset |
| Name collision on `Job.run` | permanent bug (use unique `gr-<job_id>`) |
| `ValueError` Studio not found / teamspace mismatch | permanent config |
| 5xx, timeouts (`LIGHTNING_CLOUD_READ_TIMEOUT` default 30s), connection errors | retryable |
| Stuck `Pending` past threshold (no GPU capacity; `Machine.wait_time`) | reroute |
| Interruptible job -> `Stopped`/`Failed` without user stop | retryable (resume from checkpoint) |
| `Failed` with user-code non-zero exit | permanent (surface logs) |

## ToS

- One account per person; free credits need a real phone number, virtual numbers rejected [W]. No
  multi-account farming (matches spec hard constraint). Crypto mining banned (standard; not re-verified [I]).

## Quirks

- `import lightning_sdk` triggers a version check; set `LIGHTNING_DISABLE_VERSION_CHECK=1`.
- `Job.run` without `image` constructs a `Studio` object; needs an existing Studio, else ValueError.
- `max_runtime` is NOT a timeout (DWS only). Router must enforce wall-clock limits via `stop()`.
- The SDK is huge (vendored OpenAPI client) and CalVer releases weekly; pin a version in the adapter extra.
- Free Studio 4-h restart applies to interactive Studios; unclear whether it limits Jobs [I].

## Sources

- Local: `lightning_sdk/{job.py,status.py,machine.py,teamspace.py,user.py,organization.py,studio.py,lightning_cloud/login.py,lightning_cloud/env.py}`; `lightning job --help`, `lightning job run --help`.
- https://lightning.ai/pricing (JS-rendered, not readable)
- https://lightning.ai/docs/overview/faq/billing (search snippet only: 5 credits on signup, +25 with card, phone required, non-virtual numbers)
- https://lightning.ai/docs/platform/overview/faq
- https://aicreditmart.com/ai-credits-providers/lightning-ai-free-plan-22-gpu-hours-month-guide-2026/ (15 credits/mo, ~0.68 cr/hr T4, 4-h Studio restart, no card)
- https://www.saasworthy.com/product/lightning-ai/pricing (15 credits/mo, 2 concurrent GPUs, T4/L4/A10G/L40S)
