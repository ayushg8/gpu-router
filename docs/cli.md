# gpu CLI reference (phase 2)

The scriptable CLI. Every command also works for agents and scripts with `--json`.
Source: `src/gpu_router/cli/` (commands in `app.py`, output in `render.py`, exit codes in
`exitcodes.py`); the gpu.yaml schema lives in `src/gpu_router/jobspec.py`.

## Commands

| Command | Does |
|---|---|
| `gpu run [SCRIPT] [ARGS...]` | Submit a job and stream its logs until it finishes |
| `gpu route [SCRIPT]` | Dry run: where it would go, candidates, what was ruled out and why |
| `gpu status [ID]` | Overview (active, finished recently, providers), or one job in detail |
| `gpu jobs [--all] [--here] [-n N] [--before T]` | Running and queued jobs (`--all` adds finished ones) |
| `gpu history [--failed] [--here] [-n N] [--before T]` | Finished jobs, newest first |
| `gpu logs ID [-f] [--attempt N]` | A job's output, all attempts in order |
| `gpu cancel ID` | Stop a job (idempotent) |
| `gpu approve ID [--reason R]`, `gpu deny ID [--reason R]` | Answer an approval request |
| `gpu fetch ID [--dest DIR] [--timeout S]` | Download outputs again into `./runs/<id>/` (and copy to DIR) |
| `gpu quota [--refresh]` | Quota per provider: used, left, reset, live or estimate, and how it was worked out (phase 5 ledger; `--refresh` re-reads every live provider now) |
| `gpu policy [show]`, `gpu policy set KEY VALUE`, `gpu policy reset` | Approval rules: which jobs run automatically and which ask first (saved in config.yaml `policy:`) |
| `gpu providers` | Providers: health, GPUs, session cap, running jobs, quota; then (phase 7b) the ones listed but never routed (`manual`: SageMaker Studio Lab; `verify at signup`: Paperspace, Saturn, with what to check), the excluded services with their reason (Modal: needs a card) and the inference lane |
| `gpu infer -m MODEL [-p P] [-s SYSTEM] [--max-tokens N] [-t T] [--wait S] PROMPT...` | One LLM call on a free inference API (Groq, Cloudflare Workers AI, Gemini, HF), routed by model and daily quota left; the reply on stdout, provider + tokens + quota left on stderr; `-` reads the prompt from stdin; `--dry-run` shows the route only (phase 7b) |
| `gpu infer -m MODEL -f evals.jsonl [-o results.jsonl] [-n N]` | A JSONL eval batch: one `{prompt\|messages, id?, model?, provider?, system?, max_tokens?, temperature?, ...}` per line (other keys are copied to the result's `meta`); stops early after 3 used-up / no-key failures in a row |
| `gpu infer --list` | Inference providers (key stored?), models, and quota left today |
| `gpu login groq\|gemini [--stdin] [--no-check]`, `gpu login cloudflare --account-id ID [--stdin] [--no-check]` | Store an inference key in the Keychain (`INFER_GROQ_API_KEY`, `INFER_GEMINI_API_KEY`, `INFER_CLOUDFLARE_API_TOKEN` + `INFER_CLOUDFLARE_ACCOUNT_ID`) after one free check call (a model list / token verify, never a completion); a rejected key stores nothing. HF inference uses the `gpu login hf` token |
| `gpu secrets set NAME [--stdin]`, `gpu secrets list`, `gpu secrets rm NAME` | Keychain secrets that gpu.yaml lists under `secrets:` (values are never printed) |
| `gpu login hf [--stdin] [--remote] [--import] [--no-check]` | Store a Hugging Face token (Keychain `HF_TOKEN`; `--remote`: `HF_TOKEN_REMOTE`, the least-privilege one remote runtimes get; required for remote storage, D44) so checkpoints and datasets move through a private HF Storage Bucket `<you>/gpu-router` (phase 5) |
| `gpu daemon run\|start\|stop\|status\|install-launchd\|uninstall` | Manage the daemon |
| `gpu status --line` | Claude Code status-line rows (stdlib fast path, <50 ms, never calls the daemon) |
| `gpu doctor [--timeout S] [--only GROUP\|ID ...] [--start] [-v]` | Checks every provider CLI + version, logins (without reading or printing secrets), the daemon (running, version, launchd, state.json), the data dir (0700/0600), disk, the status line, the Claude Code plugin, a competing colab skill, Codex and notifications, and compares live limits with providers.yaml; each row is ok/warn/fail/skip with the exact fix command. Parallel, one deadline (20 s). Never starts the daemon unless `--start`. Exit 0 = nothing failed, 1 = a row failed or a check crashed / did not finish (it verified nothing; `unknown` in `--json`) (phase 8a, D54) |
| `gpu doctor --update-catalog [--yes]` | Write the live limits that drifted (`quota.limit`, `quota.reset_anchor`, `quota.unit`) into `<data dir>/providers.yaml` after showing the diff and asking; no terminal and no `--yes` = exit 2 |
| `gpu notify [status]`, `gpu notify test` | macOS notification settings (config.yaml `notifications:`) and the backend in use; `test` shows one harmless "gpu-router test notification" (exit 1 + why when there is no backend) |
| `gpu setup [--yes] [--only STEP\|ITEM ...] [--dry-run] [--no-check] [--no-smoke] [--again] [--json]` | The first-run wizard (phase 8b; bare `gpu` on a terminal offers it once): 1 tools (missing CLIs via `uv tool install`, one confirmation), 2 logins (kaggle.json -> Keychain after a `kaggle quota` check, colab ADC + the exact gcloud command, Lightning, HF token), 3 launchd agent, 4 Claude Code status line / plugin / Codex MCP entry / colab skill (each shows the exact change and asks, default no), 5 `gpu doctor` + a 10-second GPU smoke job per ready provider (asks: it spends a little quota) + "N providers ready, ~X free GPU hrs/month". Re-running skips what is done; Ctrl-C then `gpu setup` continues the same run (earlier answers stand). `--only` takes a step (tools, logins, launchd, integration, check) or an item (`login.kaggle`, `plugin`, `codex`, `smoke`, ...). `--json`: one JSON document on stdout, the wizard on stderr. Exit 0 finished, 1 an item failed, 2 questions needed a terminal (no `--yes`), 130 Ctrl-C. Run by an agent (CLAUDECODE=1, CODEX_SANDBOX, GPU_ROUTER_AGENT=1), `--yes` never answers the status line, plugin, Codex, colab skill or launchd items: they are asked on a terminal, else left `manual` (D54) |
| `gpu login` | Every login gpu-router uses (kaggle, colab, lightning, hf, groq, gemini, cloudflare) with its state and the command for the missing ones; local facts only (Keychain names, file stats), `--json` |
| `gpu login kaggle [--file PATH \| --stdin] [--no-check]` | Store Kaggle credentials in the Keychain (`kaggle` = the kaggle.json document, or `KAGGLE_API_TOKEN`), from ~/.kaggle or ~/Downloads (asks), a file, stdin or a no-echo prompt; checked first with one `kaggle quota` call using only those credentials (rejected = nothing stored). The file stays where it is |
| `gpu login colab [--run]` | Check colab's Google application-default credentials (present, with the colaboratory scope, via `colab whoami`); when missing, print the exact `gcloud auth application-default login --scopes=...` command and run it with `--run` or after a yes (it opens the browser). gpu-router never reads or copies the ADC file. Exit 1 when not ready |

`ID` is a job id or any unique prefix (`a7f2`). Hex only; an ambiguous prefix is exit 4 (the
human message lists up to 5 matching jobs; `--json` has them in `detail.matches`).

Lists stop at `-n` (jobs 50, history 20, max 500). When there are more, the human view ends
with a `… more:` line, and `--json` has `next_before`; pass it back as `--before` for the
next page.

### gpu run

```
gpu run --vram 16 -p kaggle train.py --epochs 3   # gpu options first, then the script and its args
gpu run train.py --epochs 3 --json               # a few gpu options still work after the script
gpu run --epochs 3                               # script from gpu.yaml, args from the command line
gpu run train.py -- --vram 1                     # everything after -- goes to the script verbatim
gpu run bash scripts/train.sh --fast             # non-.py entrypoint = command run from the project root
```

Flags: `--vram GB`, `--hours H`, `--provider/-p NAME`, `--gpu TYPE`, `--name NAME`,
`--env/-e NAME=VALUE` (repeatable), `--wait/-w` or `--detach/-d`, `--dry-run`,
`--smoke` (phase 5: a quick smoke test, routed to the local Mac first; also on `gpu route`),
`--data [NAME=]PATH|hf://datasets/...` (phase 5, repeatable; merged into gpu.yaml `data:` by
mount, the flag wins), `--include PATH|GLOB` (D60, repeatable, before the script only; added
to gpu.yaml `include:`), `--project/-C DIR`, `--as-agent`, `--json`.

`--as-agent` (D48) submits the job as an AI agent's (`source: agent`): the agent approval
rules apply and datasets / project roots in or around credential stores are refused, as
through the MCP server. It is automatic when `CLAUDECODE=1`, `CODEX_SANDBOX*` or
`GPU_ROUTER_AGENT=1` is in the environment (a note on stderr says so); there is no opt-out.

Argument order (D23): gpu options go **before** the script. After the script only
`--json`, `--wait`, `--detach`, `--dry-run`, `--help`, `--as-agent`, `--vram`, `--hours`
and `--provider/-p` are still read as gpu options (a `-p` value that is not a provider fails
loudly), plus `--gpu TYPE` when TYPE is a GPU type the catalog offers (`--gpu L4`, D56;
`--gpu 0` or `--gpu cuda:0` still go to the script); every other token goes to the script,
including other short flags (`-wd 0.01`, `-e 10`, `-d data`) and `--name` / `--env` /
`--project` (gpu notes on stderr when one of its own option names went to the script). `--`
ends gpu parsing. A first token that looks like an option (`gpu run --epochs 3`) starts the
arguments of gpu.yaml's script.

GPU check (D56): the runner reports what `nvidia-smi` sees; when it is not the GPU the
job was placed on, the timeline gets a `gpu_mismatch` note, `gpu status <job>` shows a
`gpu seen` row and the done message says "ran on a Tesla T4, not the L4 it was placed on".

- Default is `--wait` (stream transitions and log lines, exit with the job's result).
  With `--json` the default is `--detach`; pass `--json --wait` to block and get the final job.
  `--json --wait` prints each state change to stderr (a job waiting for approval says
  `gpu approve <id>` there), and any error after the submit carries `detail.job_id`,
  `detail.short_id` and `detail.state`: the job exists and keeps running.
- Ctrl-C while waiting detaches: the job keeps running (exit 130); with `--json` stdout gets
  `{"job": JobView, "detached": true}`.
- `--dry-run` routes the spec and previews the bundle (files, size, deps, VRAM/runtime
  estimate, warnings) without submitting, plus (D60) a `left out:` line naming the
  git-ignored paths that do not ship (`data/ (2.1 GB, ignored)`, `third_party/ (6 KB,
  nested git repo)`, at most 8, grouped by top-level dir beyond that; venvs, caches, `runs/`
  and credential-looking paths are never named) and the hint to use `include:` or `--data`.
- An unknown `--provider` (or gpu.yaml `provider:`) is exit 4 with a did-you-mean hint;
  nothing is submitted.
- `--data` (phase 5): a dataset appears on the GPU at `$GPU_DATA_DIR/<NAME>`
  (`gpu.data_dir() / NAME`; NAME defaults to the path's basename). Runs on the local Mac get a
  link to the path; remote runs get a copy uploaded once to the HF bucket
  (`datasets/<content sha256>/`, cached in the daemon's `data_cache`: a later job with the
  same content reuses the upload, notes `data_uploaded` / `data_reused`). Remote runs need
  `gpu login hf`; without it a job with data is kept off remote providers (note
  `provider_excluded`). A path that does not exist is exit 2 before anything is submitted.
- The daemon packages the project at submit: git-tracked files plus untracked files git
  does not ignore (named in a bundle warning), plus whatever `include:` / `--include`
  matches (D60), never credential-looking files. `include` entries are paths or globs
  relative to the project root (`third_party/`, `data/crops/*.png`, `experiments/**/out`;
  `*` stays within one directory, `**` spans any number, a trailing `/` matches directories
  only, a matching directory ships whole, nested git repos included); absolute, `~` and
  `..` entries are refused, an entry that matches nothing or goes through a symlink is a
  warning. Inside included trees VCS dirs (`.git`, ...), virtualenvs (`.venv`, `venv`, any dir
  with a `pyvenv.cfg`), `node_modules`, tool caches and `*.pyc` are skipped (an entry naming
  `.git/...` never ships), and the
  credential, outside-symlink and 200 MB rules apply as everywhere; one walk visits at
  most 50,000 files (pass datasets with `--data`). The submit
  request waits up to 300 s for that; if no answer comes back the error is
  `submit_uncertain` (exit 1) with the idempotency key: check `gpu jobs` before retrying.

## Project root and gpu.yaml

Project root = nearest ancestor of the working directory containing `.git`, else the
working directory. `gpu.yaml` is read from the project root. Scripts given on the command
line are resolved against the current directory and stored relative to the project root.

Merge (spec decision 2): start from gpu.yaml, then every flag the user passed wins.
`--env` merges per key over the file's `env`; `--include` adds to the file's `include`;
script arguments on the command line replace
the file's `args`; a script/command on the command line replaces the file's entrypoint and
drops the file's `name` (unless it is the same entrypoint, or `--name` is given).
`provider:` is lowercased like `--provider`. gpu.yaml's `script:` must exist whenever it is
the entrypoint, with or without script arguments. A secret-looking env var is blamed on
whichever of gpu.yaml / `--env` holds it; store it with `gpu secrets set NAME` and list it
under `secrets:`.

gpu.yaml, schema version 1 (all keys optional; unknown keys are errors):

```yaml
version: 1                    # omit = 1; newer than this gpu-router = error
name: train-yolo              # display name (default: script stem)
script: train.py              # entrypoint relative to the project root ...
command: [bash, run.sh]       # ... or an argv run from the project root (not both)
args: [--epochs, "10"]        # string (shell-split) or list
vram: 16                      # GB (alias vram_gb)
hours: 2.5                    # expected runtime
provider: kaggle              # route only here
gpu: T4                       # GPU type constraint
env: {WANDB_MODE: offline}    # NON-secret env; secret-looking names are rejected
secrets: [WANDB_API_KEY]      # Keychain names exposed as env vars (never HF_TOKEN, kaggle, ...)
deps: auto                    # auto | none | requirements.txt | pyproject.toml | {kind:, file:}
data:                         # mounted under $GPU_DATA_DIR/<mount>
  - {mount: coco, path: data/coco}
  - {mount: wiki, uri: "hf://datasets/me/wiki"}
include: [third_party/, "data/crops/*.png"]  # ship even if git ignores it (string or list, D60)
checkpoint_interval_min: 20   # 0 disables checkpoint sync
interactive: false
requires_approval: false      # always ask before running
smoke: false                  # quick smoke test: prefer the local Mac (MPS)
max_attempts: 3
labels: {team: vision}
provider_options: {fake: {duration: 2}}
```

Validation errors name the file, line and key, say what is wrong and what is expected:

```
gpu: gpu.yaml:3: `varm` is not a gpu.yaml setting
  did you mean `vram`?
gpu: gpu.yaml:2: `hours` is 'two'
  `hours` must be a number of hours greater than 0, e.g. 2.5
```

## Exit codes

| Code | Meaning |
|---|---|
| 0 | success; a waited-for job ended `done` |
| 1 | unexpected or internal error; fetch failed or timed out; `submit_uncertain` |
| 2 | usage: unknown command/flag, bad flag value, invalid gpu.yaml or job spec, bad job id |
| 3 | daemon not running and could not be started, or still recovering |
| 4 | no such job (or ambiguous prefix) or provider |
| 5 | not allowed in the job's current state (approve a running job, fetch an unfinished one) |
| 10 | a waited-for job (`run --wait`, `logs -f`) ended `failed` |
| 11 | a waited-for job ended `cancelled` or `denied` |
| 12 | `route` / `run --dry-run`: no provider can ever run this job |
| 130 | interrupted (Ctrl-C); a job being waited on keeps running |

`gpu daemon ...` keeps its own codes: 0 ok, 1 error, 2 usage, 3 not running / already running.
No command ever ends in a Python traceback: usage errors are exit 2 (with `--json`, the
`invalid_request` envelope on stdout), anything unexpected is exit 1 with code `internal`.

## JSON output (`--json`)

Stable and additive-only, like API v1: fields may be added, never renamed or removed.
Exactly one JSON document on stdout (except `logs --json`, which is NDJSON). Progress notes
(such as the daemon auto-start message) go to stderr. Timestamps are ISO-8601 UTC with `Z`.
Model shapes (`JobView`, `JobDetail`, `StatusView`, `RouteDecision`, `ProviderView`,
`QuotaSnapshot`, `LogRecord`) are the API v1 models in `src/gpu_router/api.py`.

| Command | stdout |
|---|---|
| `run` (detach) | `{"job": JobView}` |
| `run --wait` | `{"job": JobView}` (final state); Ctrl-C: `{"job": JobView, "detached": true}` |
| `run --dry-run` | `{"dry_run": true, "spec": JobSpec, "route": RouteDecision, "bundle": Bundle or null}`; Bundle = `{sha256, file_count, size_bytes, code_bytes, cached, deps, estimate, warnings}` + (D60) `included: {files, bytes}` when the spec has `include`, and `left_out: [str]`, `left_out_not_shown`, `hint` when ignored paths do not ship |
| `route` | `{"spec": JobSpec, "route": RouteDecision}` |
| `status` | `StatusView` `{ready, counts, active, recent, providers}` |
| `status ID` | `JobDetail` `{job, attempts, checkpoints, events, route}` |
| `jobs`, `history` | `JobList` `{jobs, next_before}` |
| `logs` | NDJSON `LogRecord`: `{"attempt", "offset", "line"}` ...; with `-f` a final `{"eof": true, "state"}` |
| `cancel`, `approve`, `deny` | `JobView` |
| `fetch` | `{"job": JobView, "fetched": bool, "outputs_dir": str, "dest": str or null, "files": int, "message": str}` (a failed `--dest` copy: `dest` null, the reason in `message`, exit 1) |
| `quota` | `{"quota": [QuotaSnapshot], "inference": [InferQuotaView]}` (`inference` since phase 7b: `{provider, window, window_resets_at, requests_today, counters: [{scope, unit, used, limit, remaining, source live\|estimate, resets_at}], blocked, summary}`); `detail` carries the ledger's `basis` (live, live+history, reset, history, unlimited), `note`, `window` {kind, start, resets_at, length_h, label}, `remaining`, `history_h`, and when set `live_used`, `live_observed_at`, `exhausted_until`, `catalog_limit` |
| `policy`, `policy show/set/reset` | `PolicyView` `{name, editable, policy: {version, agent: Rules, user: Rules}, defaults, config_path}`; Rules = `{auto_max_hours, ask_providers, exempt_providers, max_quota_share, unknown_hours}` |
| `providers` | `{"providers": [ProviderView], "not_routed": {"manual": [Entry], "verify_at_signup": [Entry + "verify_at_signup": [check]], "excluded": [{name, display_name, reason, quote, source, decided, note}]}, "inference": [InferProviderView]}` (phase 7b adds the last two keys; Entry = `{name, display_name, gpus, session_hours, quota, link, note, verified_at}`) |
| `infer` | `InferResult` `{provider, model, model_id, text, finish_reason, usage: {input_tokens, output_tokens, neurons?, usd?}, latency_s, route_reason, fallbacks: [{provider, model_id, code, message}], quota}`; `--dry-run`: `InferRoute` `{outcome place\|wait\|no_fit, model, chosen, candidates: [{provider, model_id, score, reason, left, unlisted}], rejected: [{provider, code, reason, retry_at}], reason, retry_at}` (models in `inference/models.py`) |
| `infer --file` | NDJSON: one `{id, line, ok, provider, model, model_id, output, usage, latency_s, fallbacks, error, meta}` per input line, then `{"summary": {total, answered, failed, not_sent, by_provider, input_tokens, output_tokens, stopped, results}}` (with `-o`, the records go to the file and stdout has the summary only) |
| `infer --list` | `{"providers": [InferProviderView], "quota": [InferQuotaView]}` |
| `login groq\|gemini\|cloudflare` | `{"provider", "stored": [Keychain name], "verified": bool, "note"}` |
| `daemon status` | `{"running", "pid", "port", "version", "ready", "test_mode", "started_at"}` or `{"running": false, "message"}` |
| `daemon start` | `{"running": true, "started": bool, "pid", "port"}` |
| `daemon stop` | `{"stopped": true, "pid"}` or `{"running": false, "stopped": false, "message"}` (exit 3) |
| `daemon install-launchd` | `{"installed": true, "path"}`; with `--print` `{"installed": false, "plist"}` |
| `daemon uninstall` | `{"removed": bool}` |
| `secrets set` / `list` / `rm` | `{"secret", "stored": true}` / `{"secrets": [name]}` / `{"secret", "removed": bool}` |
| `login hf` | `{"secret": "HF_TOKEN"\|"HF_TOKEN_REMOTE", "stored": true, "user", "verified", "source"}` |
| `doctor` | `Report` `{version, home, checked_at, elapsed_ms, ok, counts: {ok, warn, fail, skip}, checks: [{id, group, title, status ok\|warn\|fail\|skip, summary, fix, detail, elapsed_ms}], drift: [{provider, key, catalog, live, note}]}` (groups: daemon, providers, storage, inference, limits, local, integration); with `--update-catalog` also `catalog_update: {changed, path, keys?, reason?, restart?}` (models in `doctor/model.py`) |
| `notify`, `notify status` | `{"notifications": {enabled, configured_backend, backend osascript\|terminal-notifier\|none, events: {finished, failed, approval, migrated}, sound, dedupe_s, max_per_minute, config_path, why_none?, backend_path?}}` |
| `notify test` | `{"sent": true, "backend", "notification": {kind, title, subtitle, body, job_id, sound}}` or `{"sent": false, "backend", "why"}` (exit 1) |
| any error | `{"error": {"code", "message", "hint", "detail"}}` (codes from `errors.ErrorCode`, plus daemon codes such as `invalid_transition`; usage errors are `invalid_request` with `detail.usage: true`) |

RouteDecision (phase 5 adds, all optional): `hours`, `hours_source` (spec | heuristic),
`smoke`, `router` ("scoring"; "simple" for a `--provider` pin); each candidate adds
`quota_left`, `quota_unit`, `quota_share`, `resets_at`; rejection code `reserved` (the local
Mac is kept for smoke tests; `routing.big_vram_providers` for big jobs, empty since Modal was
dropped in phase 7b) and `quota` (nothing or too little left; the job waits for the reset). A
job that needs more VRAM than any free provider has is `no_fit`: "needs 24GB VRAM and no free
provider has more than 16GB (largest: ...)" (data-driven; Lightning's L4 left the catalog in
D56 because the free tier refuses it with a 403).

`gpu policy set` keys: `agent.<rule>` or `user.<rule>`, or a bare `<rule>` for both. Values:
`2` / `1.5h` (hours), `50%` / `0.5` (share), `null` / `off` (no limit), `kaggle,lightning`
(lists; `none` = empty; `ask_providers` is empty by default since phase 7b), `ask` / `auto`. Agent jobs (source `agent`, the MCP server) use the
`agent` rules; `gpu run`, the shell and plain API clients use the `user` rules.

Bundle (dry run): `{sha256, file_count, size_bytes, code_bytes, cached, deps: {kind, file,
packages, python_requires}, estimate: {vram_gb, hours, vram_source, hours_source, mode,
reasons}, warnings: [str]}`.

## The daemon

Commands that need the daemon start it when it is not running: `launchctl kickstart` of
the launchd agent when it is installed and the default data dir is in use, otherwise a
detached `python -m gpu_router daemon run` with the same environment (`GPU_ROUTER_HOME`,
`GPU_ROUTER_PORT`, `GPU_ROUTER_TEST_MODE`). The CLI says so on stderr and waits for
`/v1/health` to report ready (20 s). If the spawned daemon exits right away (port taken, bad
config) the CLI fails at once with the daemon's own message instead of waiting; if it exits
because another CLI's daemon won the start race, the CLI waits for that one and does not
claim to have started it. `GPU_ROUTER_NO_AUTOSTART=1` turns this off (exit 3 instead). `gpu daemon install-launchd` makes it start at login; `--print` shows the plist
without installing it. The agent is global (one per user): with a non-default `GPU_ROUTER_HOME` it
refuses unless `--this-home` (D54).
