# gpu-router: build contract

Injected into every session and subagent working in this repo. Read it fully before editing.
Product spec: `docs/spec.md` (its Decisions table is final). This file says HOW we build it;
the spec says WHAT. When they disagree, the spec's Decisions table wins, then this file.

## Build status

- [x] 0. Design: skeleton, shared interfaces, this contract
- [x] 1. Daemon, SQLite, job state machine, fake provider, tests  (work split below; 651 tests green, integrated 2026-09-23)
- [x] 2. Scriptable CLI: `gpu run / status / logs / cancel / fetch --json`, gpu.yaml, job bundle  (870 tests green, `-m crash` 10, integrated 2026-09-24; phase-2 review fixes D23-D27: 970 green, `-m crash` 10)
- [x] 3. Kaggle and Colab adapters, plus local MPS; remote runner + `gpu` helper  (1267 tests green, `-m crash` 10, live `gpu run --wait` on local/kaggle/colab through the real CLI + daemon, integrated 2026-09-24; D29, D32, D33; phase-3 review fixes D34-D39: 1335 green, 21 skipped, `-m crash` 10)
- [x] 4. Interactive Textual shell, slash commands, footer status bar  (D42; integrated with phase 5 2026-09-24, D43: 1612 passed, 22 skipped, `-m crash` 10; real-pty check from the phase-4 run (D42); 9 run_test screenshots in docs/screenshots/ re-taken with phase-5 data; review fixes D44: 1673 passed, 22 skipped, `-m crash` 10, screenshots re-taken)
- [x] 5. Router (scoring), quota ledger, checkpoint handoff via HF Hub, approval rules  (D40, D41, D43; same run; manual routes/quota/policy/approval on a real daemon, local kill -> resume from local storage, fake resume via `checkpoint.fake_storage`; HF Storage Buckets NOT verified live: no HF token on the dev machine, `tests/unit/checkpoint/test_live_hf.py` is the opt-in check; review fixes D44, same run)
- [x] 6. MCP server, skill, Claude Code plugin, Claude Code status line  (D45, D46, D47; integrated 2026-09-24: 1866 passed, 22 skipped, `-m crash` 10; e2e on a tmp home: MCP stdio (`uv run gpu mcp`) submit -> approval -> `gpu approve` -> running -> done -> fetch, `gpu status --line` captured at each state, the wrapper around a real statusline.sh keeps its 3 lines byte for byte; `gpu status --line` p50 20.6 / p95 22.9 ms; `claude plugin validate` passed; install on the dev machine's real ~/.claude and ~/.codex NOT run: waits for the maintainer, commands under "Phase 6 install"; phase-6 review fixes D48: 1924 passed, 22 skipped, `-m crash` 10, ruff + mypy clean)
- [x] 7. Lightning and Modal adapters (ask about Modal card first), verify-later + inference lanes  (Modal dropped: needs a card, D50; Lightning adapter D49; integrated with phase 8 2026-09-25, D53: 2451 passed, 23 skipped, `-m crash` 10, ruff + mypy clean; phase-7/8 review fixes D54 (25 findings, all fixed): 2505 passed, 23 skipped, `-m crash` 10, ruff + mypy clean; live `gpu run --hours 0.25 gpucheck.py --provider lightning --wait` through the real CLI + daemon on a tmp home: Tesla T4, done, outputs in `runs/d608/`; live inference on Gemini (D50) and Groq (D53). Pending logins: Cloudflare + HF keys, Paperspace / Saturn / Studio Lab accounts; L4 run live 2026-09-25: the free tier refuses it with a 403, so the catalog offers Lightning T4 only, D56)
- [x] 8. Setup wizard polish, notifications, `/doctor`  (D51, D52; same integration run, D53: `gpu doctor` 0 fail on a tmp home, `gpu setup --dry-run` in a sandbox HOME and against a real home with a tmp data dir wrote nothing, `gpu notify test` + daemon notifications (approval, finished) through osascript from a real Lightning job. Pending: `gpu setup` for real = the installs into ~/.claude, ~/.codex, launchd and the logins, under "Phase 6 install")

A phase is done only when it works end to end and `uv run pytest` is green. Tick it here.

## Commands

```bash
uv sync                                   # install (Python 3.12 via uv; never system python3)
uv run pytest                             # all tests (real providers skipped)
uv run pytest tests/unit/core -q          # one area
uv run pytest -m crash                    # crash-recovery harness (spawns + SIGKILLs a daemon)
uv run ruff check src tests && uv run ruff format --check src tests
uv run mypy                               # strict, src/gpu_router (runner/ excluded)
# CI (.github/workflows/ci.yml, macos-latest, every push + PR) runs exactly: uv sync --locked,
# ruff check, ruff format --check, mypy, pytest -q; no provider CLI, Keychain, ~/.claude or
# network needed (D59)
uv run gpu daemon run --foreground        # daemon in the terminal (GPU_ROUTER_HOME=/tmp/x to isolate)
GPU_ROUTER_TEST_MODE=1 uv run gpu daemon run --foreground --port 0   # with the fake provider
```

Never `git commit` unless the user asks. Never print or copy credentials (`~/.kaggle/kaggle.json`,
Keychain items, tokens). Never sign up, log in or create cloud resources from a build session.
Never touch `~/.claude/settings.json` (phase 6 asks the user first).

## Repo layout

`P` = phase that implements it. `Own` = phase-1 owner group (see Work split). "real" = already
real code; everything else has signatures + docstrings and `NotImplementedError` bodies.

| Path (under `src/gpu_router/`) | Responsibility | P | Own |
|---|---|---|---|
| `__init__.py` | version only; imports nothing (inv. 15) | 0 | A |
| `__main__.py` | `python -m gpu_router` = `gpu` | 0 | C |
| `entry.py` | stdlib-only argv dispatch to status line / daemon / shell / CLI (real) | 1 | C |
| `models.py` | domain pydantic models: JobSpec, Job, Attempt, JobEvent, ... (real); `SECRET_ENV_NAME` (validator) + `SECRET_ENV_NAME_STRICT` / `secret_env_problem` (intake, D39) | 1 | A |
| `agent.py` | agent-job intake (D48): `agent_marker` (CLAUDECODE=1, CODEX_SANDBOX*, GPU_ROUTER_AGENT=1), `mark_agent`, `check_agent_spec` (no credential store / home dir / dataset symlink into a store); used by `mcp/tools.py` and `gpu run` | 6 | - |
| `statemachine.py` | job + attempt states, TRANSITIONS, Reason codes, RECOVERY (real) | 1 | A |
| `errors.py` | all exceptions: adapter taxonomy, API errors, envelope codes (real; phase-2 review added client-side `SubmitUncertain`, code `submit_uncertain`) | 1 | A |
| `store.py` | the only reader/writer of gpu.db; transactional transitions; `data_cache` get/put/touch/delete (phase 5, D40); `quota_snapshots` pruned to the newest 500 per provider (D43) | 1 | A |
| `db/connection.py` | sqlite connect + pragmas, `transaction()` | 1 | A |
| `db/migrate.py` | discover + apply `migrations/NNNN_*.sql` | 1 | A |
| `db/migrations/0001_initial.sql` | schema v1 (real) | 1 | A |
| `statefile.py` | state.json status-line cache: shape (real) + coalescing writer | 1 | A |
| `config.py` | config.yaml models (real), load/save/migrate | 1 | A |
| `paths.py` | data-dir layout, stdlib only (real) | 1 | A |
| `clock.py` | Clock protocol, SystemClock, FakeClock (real); FakeClock sleeps on its monotonic time and `suspend(dt)` models a Mac asleep (D57) | 1 | A |
| `ids.py` | job ids, short ids, attempt keys | 1 | A |
| `lock.py` | single-instance flock | 1 | A |
| `log.py` | JSON structured logging + redaction filter | 1 | A |
| `secrets.py` | Keychain access + redaction | 1 | A |
| `protocol.py` | `::gpu::` line protocol + stdout fallback parser (real) | 1 | B |
| `providers/catalog.py` | providers.yaml models (real) + loader/merge | 1 | B |
| `providers/providers.yaml` | packaged catalog: GPUs, VRAM, session caps, quotas (data); phase 7b (D50): `status: verify_at_signup` (paperspace, saturn) + `manual_only` (sagemaker_studio_lab) entries are listed, never registered; `excluded:` (modal + the spec's list, with reason/quote/source/decided); `inference:` (the inference lane's endpoints, limits, model aliases) | 1,7 | B |
| `providers/<name>/NOTES.md` | per-provider login steps, quirks, commands | 3,7 | - |
| `providers/cli.py` | shared `CommandRunner` (not built: kaggle and colab each ship a bounded runner in `providers/<name>/cli.py`; merge when a third CLI adapter lands) | 3,7 | - |
| `providers/local/adapter.py` | `LocalAdapter` (kind local): runs bundles on this Mac with MPS; run dir `<home>/local/<attempt key>/`, liveness = flock on `alive.lock`, venv per deps key (D28); job env = allowlist (D38); raw-log / run-dir sweep (D37); unreadable resume checkpoint = fresh start (D34) | 3 | - |
| `providers/local/launcher.py` | stdlib-only detached launcher (`python -I`): fork + setsid, unpack, venv/install (uv, else venv+pip), exec the bundle's bootstrap holding the lock; owns the run-dir file names | 3 | - |
| `adapters/local.py` | registry entry point: re-exports `providers.local.adapter.LocalAdapter` | 3 | - |
| `providers/kaggle/adapter.py` | `KaggleAdapter` (kind kaggle): one private script kernel per attempt `<user>/gpu-router-<job_id>-<n>`, bundle base64-embedded in run.py, post-run logs (cached redacted, D37), fetch of `outputs/`, cancel = delete kernel (stops the session, verified live), live quota; push intent file (D35); test-mode offline guard (D30); phase 5: secrets via the private dataset `<user>/gpu-router-secrets`, live logs from the storage log tail, `resume=True` (D40) | 3 | - |
| `providers/kaggle/{cli,parse,remote,credentials}.py` | bounded `kaggle -W` calls + error taxonomy; pure parsers for CLI 2.2.4 output; kernel naming/metadata/generated run.py; Keychain-or-CLI credentials | 3 | - |
| `adapters/kaggle.py` | registry entry point: re-exports `providers.kaggle.adapter.KaggleAdapter` | 3 | - |
| `providers/colab/adapter.py` | `ColabAdapter` (kind colab): one T4 session per attempt named `gr-<job>-<n>`, detached launch + `exec` polls, harvest-then-stop, private `--config` session file (D31); orphaned-`new` handling (D35); teardown retries + janitor thread (D36); local hygiene sweep (D37) | 3 | - |
| `providers/colab/cli.py` | `ColabCli` (bounded `colab --auth=adc --config ...` calls with a private `HOME`, timeout kills the group, `on_spawn` pid hook), failure classification, token redaction | 3 | - |
| `providers/colab/remote.py` | VM-side scripts (prepare, launch, poll, pack, logread) as strings, `@@GR:` result parser, `split_log_bytes` | 3 | - |
| `providers/colab/state.py` | `RunRecord`/`RunStore`: per-session JSON record under `<home>/providers/colab/runs/` (incl. `cli_pid`, `log_served_at`, `log_purged`) | 3 | - |
| `adapters/colab.py` | registry entry point: re-exports `providers.colab.adapter.ColabAdapter` | 3 | - |
| `providers/lightning/adapter.py` | `LightningAdapter` (kind lightning): one Studio job `gr-<job>-<n>` per attempt in the Studio `gpu-router`, files via the teamspace drive `uploads/gpu-router/<name>/`, in-job wall clock + status() backstop, final-log verdicts cached redacted, live credits (D49) | 7 | - |
| `providers/lightning/{sdk,driver}.py` | `SdkBridge`: one bounded `driver.py` process per call in the SDK's own env (uv tool env or `uv run --with lightning-sdk==<pin>`), process-group kill, scrubbed env, private SDK HOME; `driver.py` runs inside that env (never imports gpu_router), JSON on stdin, one `@@GRL:` line out | 7 | - |
| `providers/lightning/{launch,credentials,login}.py` | in-job launcher (py3.8, stdlib: bundle, secrets file, storage token file, bootstrap, wall clock, outputs.tar.gz delivery); Keychain/env/file credentials; `gpu login lightning` logic (the Typer command is in `cli/login.py`) | 7 | - |
| `adapters/lightning.py` | registry entry point: re-exports `providers.lightning.adapter.LightningAdapter` | 7 | - |
| `adapters/base.py` | Adapter ABC, RemoteRef/RemoteStatus/LogChunk/Health, Capabilities (real) | 1 | B |
| `adapters/registry.py` | name -> adapter instance, built from catalog + config (real); test mode keeps real providers out unless opted in (D29) | 1 | B |
| `adapters/fake.py` | FakeAdapter: disk-backed simulated remote, directives; with `checkpoint.fake_storage` (test mode) its simulated runner publishes checkpoints to local storage and resumes from `GPU_RESUME_URI`; quota honours `reset_anchor` (D43) | 1 | B |
| `adapters/lightning.py` | real adapter (no modal adapter: Modal was dropped because it needs a card, D50) | 7 | - |
| `inference/{catalog,keys,clients,ledger,router,service,models,remote,batch,errors}.py` | phase 7b inference lane (D50): typed `inference:` catalog; `INFER_*` Keychain keys + `gpu login` checks; one OpenAI-compatible httpx client (Groq/Gemini/Cloudflare/HF) + error/header rules; daily-quota ledger `<home>/inference/ledger.json` (live/est); router by model + quota left with one-line reasons; `InferenceService` (daemon-owned, `DaemonRuntime.inference`); API models; client helpers; JSONL batches; see `inference/NOTES.md` | 7 | - |
| `daemon/lanes.py` | phase 7b endpoints `/v1/infer`, `/v1/infer/route`, `/v1/infer/quota`, `/v1/infer/providers` (worker threads, never the Store) | 7 | - |
| `cli/infer.py`, `cli/lanes.py`, `shell/infer.py` | phase 7b: `gpu infer` (+ `--file` JSONL evals, `--dry-run`, `--list`), `gpu login groq\|gemini\|cloudflare`; the not-routed / excluded / inference listings in `gpu providers`, `gpu quota`, /providers, /quota; the shell's /infer | 7 | - |
| `router/base.py` | Router protocol, RoutingContext, RouteDecision, Candidate (real) | 1 | C |
| `router/simple.py` | phase-1 first-fit router (equal-VRAM tie-break prefers more GPUs, D32) | 1 | C |
| `router/scoring.py` | phase-5 `ScoringRouter`, the daemon default: filter (session/handoff, ledger quota, `reserved` Mac/modal), score, one-line reasons; `--provider` pins go to `SimpleRouter` (D41); resumed jobs need only what is left, "(est)" labels, 0-left share sentinel (D44) | 5 | - |
| `router/settings.py` | `routing:` section of config.yaml, typed (`RoutingSettings`: provider roles, smoke threshold, handoff minimum, `quota` knobs) | 5 | - |
| `policy.py` | approval policy protocol, phase-1 `SpecOnlyPolicy`, phase-5 `RulesPolicy` (default; `policy:` in config.yaml, agent vs user rules, `apply_policy_setting` for `gpu policy set`, D41); `ApprovalDecision.always` = asks even after an approval (quota rule, D43) | 1,5 | C |
| `api.py` | HTTP API request/response models shared by daemon + client (real) | 1 | C |
| `engine/deps.py` | `EngineDeps` bundle handed to supervisor/drivers (real) | 1 | C |
| `engine/backoff.py` | backoff schedule | 1 | C |
| `engine/calls.py` | `AdapterCaller`: thread pool, timeouts, contract enforcement | 1 | C |
| `engine/capture.py` | log capture: attempt log files, protocol parsing, metrics/ckpt | 1 | C |
| `engine/driver.py` | `JobDriver`: one asyncio task per non-terminal job; phase 5 (D40): storage env/data per attempt, checkpoint reconcile before a migration, planned handoff (session cap / quota), progress-aware max_attempts; re-asks the policy after an approval, in place (D43) | 1 | C |
| `engine/supervisor.py` | `Supervisor`: recovery, drivers, user actions, health loop (backoff re-checks of unhealthy providers, wake-from-sleep detection + grace, `ProviderView` "re-checking in ..." note, D57) | 1 | C |
| `engine/crashpoints.py` | test-mode crash injection (`GPU_ROUTER_CRASH_AT`) | 1 | C |
| `engine/context.py` | builds the router input (RoutingContext), shared by drivers and `/v1/route` dry runs | 1 | C |
| `engine/_obs.py` | logging wrapper so a logging failure never stops a job | 1 | C |
| `daemon/__main__.py` | `gpu daemon run/start/status/stop/install(-launchd)/uninstall` (argparse; every subcommand but `run` takes `--json`) | 1 | C |
| `daemon/runtime.py` | `DaemonRuntime`: assemble lock, config, store, engine, writer | 1 | C |
| `daemon/app.py` | FastAPI app factory, middleware, error envelope | 1 | C |
| `daemon/routes.py` | `/v1` endpoints | 1 | C |
| `daemon/auth.py` | bearer token file, loopback Host/Origin guard | 1 | C |
| `daemon/events.py` | `EventBus`: store listener -> long-poll waiters | 1 | C |
| `daemon/server.py` | uvicorn runner, signals, daemon.json | 1 | C |
| `daemon/launchd.py` | launchd plist writer / load / unload | 1 | C |
| `daemon/spawn.py` | background start + wait for `/v1/health` (CLI auto-start, `gpu daemon start`); watches the spawned child (early exit = fail fast with its launchd.log line; exit 3 = lost the start race, keep waiting, `started=False`) | 2 | - |
| `client.py` | sync HTTP client used by CLI, shell, MCP (submit: 300 s timeout, retries only connections that never reached the daemon, else `SubmitUncertain`) | 1 | C |
| `packaging/files.py` | which files ship: `git ls-files -co --exclude-standard -t` (untracked ones named in a warning), walk only when no `.git` ancestor, `GitError` when git fails inside a repo, credential deny list (names, dirs, symlink targets), outside-project symlinks refused; `include:` patterns (`normalize_include`, `IncludeError`) ship ignored paths through the same rules, `left_out` names what git ignores, cheaply (D60) | 2 | - |
| `packaging/deps.py` | `DepsSpec` -> `DepsInfo` (requirements.txt / pyproject deps parsed with tomllib / none); `[tool.uv.sources]` git/url -> PEP 508 direct refs, in-project path -> `./path`, index -> warning, others -> `DepsError` | 2 | - |
| `packaging/estimate.py` | VRAM + hours heuristics (labelled spec/heuristic, with reasons) | 2 | - |
| `packaging/bundle.py` | `build_bundle`, deterministic tar.gz + sha256 (fsynced), cache `bundles/<sha>.tar.gz` (a rebuild always replaces the entry), `materialize`, `BundleBuilder`; `bundle_summary` = what ships + `left_out` without the archive, for gpu_submit/gpu_route (D60) | 2 | - |
| `jobspec.py` | gpu.yaml schema v1 (validation errors with file:line + hint), project root, gpu.yaml + flags merge -> JobSpec; strict env intake check (D39) | 2 | - |
| `cli/app.py` | Typer commands (run, route, status, jobs, history, logs, cancel, approve, deny, fetch, quota, providers, secrets, daemon), `split_run_argv` (D23), `main(argv)` (catches both click copies, D24); `run --data` merge (`_with_data`, phase 5) | 2 | - |
| `cli/render.py` | rich output: visual language (⚡ ⏸ ✓ ✗ ↪, green/yellow/red/dim), tables, empty states | 2 | - |
| `cli/exitcodes.py` | exit codes (0/1/2/3/4/5/10/11/12/130) + error/state mapping | 2 | - |
| `cli/bundling.py` | `run --dry-run` bundle preview via `packaging.build_bundle` | 2 | - |
| `runner/gpu.py` | `import gpu` helper (py3.8, stdlib; shipped as `gpu_runner/gpu.py`); `atomic_checkpoint(name)`; `checkpoint_requested()` (phase 5 handoff) | 2 | - |
| `runner/bootstrap.py` | remote entrypoint: unpack, pip install, resume restore, tee, heartbeat, ckpt sync, EXIT (py3.8, stdlib; see D26); phase 5: storage publish/restore, status channel (heartbeat + log tail), checkpoint requests, datasets, storage-token hygiene (D40) | 2 | - |
| `runner/storage.py` | THE checkpoint/data/status storage layout (py3.8, stdlib, lazy huggingface_hub; shipped as `gpu_runner/storage.py`, imported by the daemon too): `LocalStore` (`file://`), `HfBucketStore` (`hf://buckets/<ns>/<name>`), keys, `publish_checkpoint`, `claim`, datasets (D40) | 5 | - |
| `shell/app.py` | `GpuShell` (Textual): job panel, transcript, `>` prompt + `/` popup, footer; tokens/theme; `run()` for bare `gpu` on a TTY (D42) | 4 | - |
| `shell/feed.py` | background poller: `/v1/status` every 1 s, `/v1/quota` every 60 s on its own thread (restarted whenever it ended), metric history from `logs?protocol=true` (first read: the log's last 5000 lines), checkpoint URI -> HF Hub / local storage / `<provider> disk` (D43); auto-start on first connect, then reconnect every 3 s; a status timeout keeps the last snapshot (D44) | 4 | - |
| `shell/commands.py` | slash commands (run route jobs status logs watch cancel fetch approve deny quota history providers login doctor policy config help clear exit); reuse `cli.app` helpers + `cli.render` through a `Collector` console; `shellify` turns `gpu x` hints into `/x`; phase 5: `/run --smoke --data`, `/route --smoke`, `/quota --refresh`, `/policy show/set/reset`, `policy_keys()` (D43) | 4 | - |
| `shell/complete.py` | popup/tab candidates: commands, job ids (approvals first), scripts + dirs, providers, `--data` paths, `/policy` subcommands + rule keys (D43) | 4 | - |
| `shell/{panel,chart,metrics,state}.py` | pure formatting: panel rows, footer, empty/down states, notices; /watch area chart; `MetricHistory` (helper beats stdout), sparkline, trend; immutable `Snapshot` | 4 | - |
| `shell/widgets.py` | `JobPanel`, `StatusFooter`, `CommandBlock`, live `LogsBlock` (reads at most the last ~5000 lines per attempt, writes in 250-line chunks, pgup scrolls it, output released when old) / `WatchBlock` (own poll thread, esc stops), `CommandPopup` (`moved`), `Prompt` (D44) | 4 | - |
| `checkpoint/hub.py` | `CheckpointHub` (daemon, `EngineDeps.checkpoints`): backend per adapter kind (local dir for runs on this Mac, HF bucket for remote with a Keychain token, else none + reason/hint), per-attempt env + storage-token secret, owner claim, checkpoint copy between backends, `latest`, handoff control/ack, dataset digest/upload; `call()` runs blocking work in its own pools (D40); `fake` is a local kind when test mode + `checkpoint.fake_storage` (D43) | 5 | - |
| `checkpoint/storage.py`, `checkpoint/tokens.py` | typed facade over `runner/storage.py` + `StorageError`; HF tokens from the Keychain (`HF_TOKEN`, `HF_TOKEN_REMOTE`), never env/file implicitly | 5 | - |
| `checkpoint/data.py`, `checkpoint/sidechannel.py` | dataset content hash (+ signature index); log-tail reads, `t<n>:<hash>` cursors and tail -> final-log alignment for adapters without live logs; `set_active_hub`/`active_hub` in `checkpoint/__init__.py` | 5 | - |
| `cli/login.py` | `gpu login hf [--stdin] [--remote] [--import] [--no-check]` (token -> Keychain, never argv; whoami check; namespace cache) | 5 | - |
| `quota/windows.py` | reset windows from providers.yaml `reset` + `reset_anchor` (fixed calendar week/month/day in UTC, rolling fallbacks), pure date math | 5 | - |
| `quota/ledger.py` | per-provider ledger view (live reading when fresh, else live + history / history estimate, labelled; `remaining`, unit conversion incl. per-GPU rates `quota_per_gpu_hour_by_gpu`, D53) computed from the store (D41) | 5 | - |
| `quota/service.py` | `QuotaService` (daemon): ledger views for GET /v1/quota and state.json, background refresh of live readings (TTL, due `early_s` before it, failure backoff, injected clock; D44), never on the routing path | 5 | - |
| `quota/settings.py` | `routing.quota` knobs (ttl_s, refresh_s, wait_s, retry_failed_s, unknown_window_hours) | 5 | - |
| `mcp/server.py`, `mcp/tools.py` | `gpu mcp`: FastMCP stdio server, the spec's 7 tools, no approve tool (`server.py` = agent-facing text + error envelope; `tools.py` = logic over `GpuClient`, no MCP imports); see "Agent integration (phase 6a)" | 6 | - |
| `.claude-plugin/marketplace.json` (repo root) | marketplace `gpu-router` for `claude plugin marketplace add ayushg8/gpu-router`, one entry with source `./plugin` (D58) | - | - |
| `.github/workflows/ci.yml` (repo root) | CI on macos-latest: uv sync --locked, ruff check + format --check, mypy, pytest -q (D59) | - | - |
| `plugin/` (repo root) | Claude Code plugin + local marketplace `gpu-router-local`: `.mcp.json`, `skills/gpu-router/SKILL.md`, `commands/gpu-{run,status,approve,statusline}.md`, `statusline/gpu-statusline.sh` (6b's wrapper, shipped with the plugin, activated only by `gpu statusline install`) | 6 | - |
| `docs/codex/AGENTS-section.md` | Codex: setup (`codex mcp add`, config.toml) + the AGENTS.md section (the skill, adapted) | 6 | - |
| `statusline/fast.py` | `gpu status --line`: 0-2 rows from state.json in the user's status-line style (stdlib only, never calls the daemon, ~21 ms p50 incl. interpreter); see "Claude Code status line (phase 6b)" | 6 | - |
| `statusline/{samples,install,cli}.py`, `statusline/gpu-statusline.sh` | sample snapshots per row state (preview + goldens); `gpu statusline install\|uninstall\|preview\|status` (argparse, dispatched in entry.py; Typer only forwards); the wrapper (same bytes as `plugin/statusline/gpu-statusline.sh`) | 6 | - |
| `notify/{settings,format,backends,service,cli}.py` | phase 8a macOS notifications: config.yaml `notifications:` (typed here), which events notify + their text, osascript / terminal-notifier / null backends, the daemon's `Notifier` on the EventBus (`DaemonRuntime.notifier`), `gpu notify [status\|test]`; see "Notifications and doctor (phase 8a)" | 8 | - |
| `doctor/{probe,checks,runner,model,render,catalog_fix,cli}.py` | phase 8a `gpu doctor` / `/doctor`: injectable `ProbeEnv`, one function per check, parallel runner with one deadline, `--json` Report, rich rendering shared by CLI and shell, `--update-catalog` | 8 | - |
| `setup/{firstrun,state,context,ui,base,providers,tools,logins,service,integrations,check,wizard,cli}.py` | phase 8b `gpu setup` wizard (D52): `firstrun` (stdlib only; bare `gpu` offers the wizard while `<home>/setup.json` has no finished/dismissed run), `state` (setup.json: answers, outcomes, smoke results; resumable), `context.SetupEnv` (every path/subprocess/launchctl/prompt injectable; `probe()` = a doctor ProbeEnv over the same fakes), one module per step, `wizard` (order, `--only`, exit codes), `cli` (`gpu setup`, `gpu login kaggle\|colab`, bare `gpu login`) | 8 | - |

Tests: `tests/conftest.py` (shared fixtures), `tests/unit/core/` (A), `tests/unit/adapters/` and
`tests/contract/` (B), `tests/unit/engine/`, `tests/api/`, `tests/crash/` (C),
`tests/unit/cli/` (jobspec, render, exit codes) + `tests/cli/` (CliRunner against one real
daemon subprocess per module on the fake provider, auto-start disabled via
`GPU_ROUTER_NO_AUTOSTART=1`; launchctl is always monkeypatched) (phase 2 CLI),
`tests/cli/test_e2e_subprocess.py` (phase-2 end to end through the real `gpu` executable:
auto-start on a private home, run/status/logs/fetch/cancel/jobs/history `--json`, outputs in
`runs/<id4>/`, then the daemon-materialized bundle executed by `bootstrap.py`),
`tests/unit/packaging/` + `tests/unit/runner/` + `tests/api/test_bundle_api.py` (phase 2
packaging/runner; `tests/unit/packaging/helpers.py` makes throwaway git projects with the user's
git config isolated). Phase-2 review regressions: `tests/cli/test_review_fixes.py` (usage
errors through the real executable, `run --json --wait` contract, fetch --dest, unknown
provider, paging, secrets, daemon --json, spawn early exit), `tests/unit/cli/test_run_argv.py`
(D23 argv split), `tests/unit/runner/test_bootstrap_review.py` (D26),
`tests/unit/packaging/test_files_review.py` (D25, uv sources, cache healing). `test_runs_on_python38` runs bootstrap under a real 3.8 when
`uv python find 3.8` succeeds (`uv python install 3.8`), else skips.
End-to-end: `tests/api/test_e2e_flows.py` (HTTP API flows: done, rate limit, migration + resume,
cancel, approve/deny) and `tests/crash/test_sigkill_midrun.py` (plain SIGKILL mid-run and during
provisioning; restart must reattach to the same remote id).
Local provider (phase 3): `tests/unit/providers/local/` (real launcher/bootstrap/job
processes under the tmp home, `env: system` on the test interpreter or a fake `uv` script;
one test uses the real uv; cancel/SIGKILL/SIGTERM/reattach/interrupted-submit cases;
`helpers.py` builds projects, bundles, jobs) and `tests/contract/local/` (re-runs every
shared contract test: target `local` always, `local-venv` = default uv venv settings when
`GPU_ROUTER_REAL_PROVIDERS=local`; fake-directive tests skip).
Colab provider (phase 3): `tests/unit/providers/colab/` (`fake_colab.py` simulates the
`colab` CLI + VM on the local filesystem with fault injection via `sim-control.json`; the
runner is the real bootstrap from a real bundle; `helpers.py` has `ColabSim`, demo bundles,
`make_adapter`; `test_live.py` is the opt-in live T4 smoke) and
`tests/contract/test_colab_contract.py` (re-collects every shared contract test with its own
fixtures: `colab-sim` always, `colab` = the real account when
`GPU_ROUTER_REAL_PROVIDERS=colab`, ~15 T4 sessions, teardown stops each; fake-directive
tests skip).
Shell (phase 4): `tests/shell/` (`conftest.py` runs a real daemon IN the test process:
uvicorn on its own thread + event loop, `DaemonRuntime.create(configure_logging=False)`
built on that thread, fake providers, port 0, auto-start off; `test_shell_app.py` drives
`GpuShell` with Textual's pilot: popup, tab completion, approve/deny, /run + live logs,
/watch, every CLI-backed command, notices, empty / daemon-down / too-many states, history,
quitting; `test_format.py` + `test_complete_commands.py` are pure; `test_screenshots.py`
renders the SVGs behind `docs/screenshots/shell-*.png`, with `GPU_SHELL_SHOTS=<dir>`).
Lightning provider (phase 7a, D49): `tests/unit/providers/lightning/` (`test_adapter.py` over
`tests/contract/lightning/sim.py` = SimLightning, a simulated SDK driver on a FakeClock with
fake directives; `test_driver.py` runs the real driver.py in a subprocess against
`fake_sdk/lightning_sdk` (the real 2026.9.18.post1 names/signatures, state in a JSON file);
`test_launch.py` runs launch.py + the real bootstrap on a real bundle, incl. the wall clock;
`test_real_sdk_shape.py` checks the pinned real SDK from uv's cache with `--offline`, skips
without it; sdk/credentials/login tests) and `tests/contract/lightning/` (every shared
contract test: `lightning-sim` always, `lightning` = the real account when
`GPU_ROUTER_REAL_PROVIDERS=lightning`; `test_live_smoke.py` = ONE live T4 job, creds from env
or ~/.lightning/credentials.json since the tests' keyring is in memory).
Phase-3 integration: `tests/unit/adapters/test_registry.py` (D29 test-mode rule),
`tests/unit/router/test_simple.py::test_equal_vram_prefers_more_gpus` (D32),
`tests/unit/engine/test_flows.py::test_run_finished_between_polls_gets_a_start_time` (D33).
Phase-3 review regressions (D34-D39): `tests/unit/providers/{kaggle,colab,local}/test_review_fixes.py`
(429 in names, orphaned `new` / interrupted push, teardown retries + janitor, cancel harvest,
private CLI home, tmp secrets sweep, raw-log purge, empty ERROR log, env allowlist),
`tests/unit/providers/local/test_local_run.py::test_unreadable_resume_checkpoint_starts_fresh_with_a_note`
and `::test_a_colab_checkpoint_resumes_from_its_mirror_on_this_mac` (D34),
`tests/unit/core/test_env_intake.py` + `tests/api/test_review_fixes_misc.py::test_new_specs_with_secret_looking_env_are_refused` (D39).
The colab unit tests switch the janitor off (`JANITOR_ENABLED`, autouse fixture) except the
tests about it; the contract fixtures `close()` their adapters.
Live end to end through the real `gpu` CLI + daemon (private home, `GPU_ROUTER_PORT=0`,
2026-09-24): `gpu run gpucheck.py --provider local|kaggle|colab --wait` all done with
outputs in `runs/<id4>/`, logs via `gpu logs`, Kaggle quota live in `gpu quota`; numbers in
each provider's NOTES.md "Integration run" line.
Kaggle provider (phase 3): `tests/unit/providers/kaggle/` (parsers, CLI wrapper, generated
run.py executed for real on a real bundle, adapter over SimKaggle or scripted CLI output,
credentials, test-mode guard) and `tests/contract/kaggle/` (`sim.py` = SimKaggle, a
simulated kaggle CLI on a FakeClock that honours fake directives; `test_contract.py`
re-runs every shared contract test: target `kaggle-sim` always, `kaggle` = real CLI when
`GPU_ROUTER_REAL_PROVIDERS=kaggle`, one private kernel per submitting test);
`test_live_smoke.py` = ONE live 2xT4 kernel end to end (opt-in, real_provider), plus a
cancel probe behind `KAGGLE_CANCEL_PROBE=1`.
Phase 5a (router, quota ledger, approval policy): `tests/unit/router/test_scoring.py`
(table-driven routing cases on the packaged catalog: 6h -> kaggle, 20 min -> colab, 24GB ->
a configured big-VRAM provider (synthetic since D50) or no-fit, `--smoke` -> local, kaggle exhausted -> colab with the reason, handoff,
waits, reservations, pins), `tests/unit/quota/` (windows incl. the Saturday 00:00 UTC
boundary, ledger bases, `QuotaService` TTL/slow/failed providers), `tests/unit/policy/`
(thresholds per audience, rule order, `gpu policy set` parsing),
`tests/api/test_policy_quota_api.py` (/v1/policy persisted + driving approvals, /v1/quota
ledger, /v1/route estimate, placement detail) and `tests/cli/test_phase5_cli.py`.
Phase 5 checkpoint storage (D40): `tests/unit/checkpoint/` (`fake_hfapi.py` = in-memory
HfApi with the 1.32 bucket call shapes + fault injection; `test_runner_storage.py` both
backends; `test_hub.py` backend choice, tokens, degrade reasons, copies, control files,
datasets; `test_bootstrap_storage.py` the real runner publishing/resuming/answering
handoff requests over local storage, HF downloads in-process; `test_engine_storage.py`
FakeClock handoff before the cap / quota, no answer, restart, reconcile, progress-aware
max_attempts, dataset upload once + reuse, no-token degrade; `test_local_e2e.py` the real
local provider killed mid-run resuming on attempt 2 with the next seq;
`test_cli_login_data.py`; `test_sidechannel.py`; `test_live_hf.py` opt-in live bucket
round trip with `HF_TOKEN` + `GPU_ROUTER_REAL_PROVIDERS=hf`) and
`tests/unit/providers/kaggle/test_storage_channel.py` (secrets dataset over SimKaggle's new
`datasets` commands, storage resume, live tail -> final log).
Phases 4-5 integration (D43): `tests/unit/engine/test_approval_reask.py` (re-ask when the
re-route after an approval lands on another provider over half its quota, same provider
places, non-`always` rules keep the approval, timeout restarts at the re-ask),
`tests/unit/adapters/test_fake_storage.py` + `tests/api/test_fake_storage_e2e.py` (a fake job
dies and resumes on attempt 2 from `<home>/storage/jobs/<id>/ckpt-NNNN` through a real
DaemonRuntime with `checkpoint.fake_storage`),
`tests/unit/core/test_store.py::test_quota_snapshots_are_pruned_per_provider`, shell:
`test_shell_app.py::test_phase5_flags_reach_the_router_and_the_ledger`,
`test_complete_commands.py::test_phase5_completion_data_paths_and_policy_keys`,
`test_format.py::test_checkpoint_where_names_the_storage_backend`,
`test_screenshots.py::test_shot_route`.
Phases 4-5 review regressions (D44): `tests/unit/router/test_review_fixes.py` (reserved Mac
vs all-excluded, 0-left share, "(est)" labels, resumed-job hours, default runtime guess),
`tests/unit/engine/test_review_fixes_phase5.py` (freed slot wakes a queued job, quota-reset
waits vs max_queue_wait_s, invalid_job with the reserved Mac),
`tests/api/test_review_fixes_phase5.py` (agent job without hours asks, real bundle),
`tests/unit/quota/test_service.py::test_background_refresh_keeps_the_reading_live_every_period`,
`tests/unit/checkpoint/test_review_fixes.py` (placement waits for unreachable storage and
its bound, locked Keychain, HF_TOKEN_REMOTE only, refused claim, verified copies + fallback,
reconcile retry, cleanup, dataset LRU, unreadable data file, session deadline anchor,
symlinked dataset dirs, fsync), `tests/unit/checkpoint/test_bootstrap_review.py` (runner
resume fallback / torn checkpoint / exit 90, landed publish + seq bump, ack retries,
handoff upload retry, moved download, restore OSError, low-disk direct publish, py<3.10,
token file), `tests/unit/providers/kaggle/test_remote.py::test_runner_hands_the_storage_token_over_in_a_file`,
`tests/unit/providers/colab/test_adapter.py::test_the_storage_token_never_sits_in_a_long_lived_environment`,
`tests/shell/test_review_fixes.py` (Enter vs the popup, /policy set, history draft,
bounded /logs + feed read, quitting mid-command, notices, /jobs paging, transcript bound,
pgup, quota thread, thinning, completion speed, one visual language).
MCP (phase 6a): `tests/mcp/` (`conftest.py` reuses the shell suite's `InProcDaemon` + a
FastMCP in-memory `Client(build_server())`; `test_mcp_tools.py`: exactly the 7 tools and no
approve, submit -> status -> logs(tail/since) -> fetch, approval-needed guidance with the
`/gpu-approve` text, cancel, quota, route + approval preview, bad refs / project dirs /
secret env, logs across attempts, `gpu mcp --help`, and the real stdio transport through
`python -m gpu_router mcp`).
Plugin (phase 6 integration): `tests/unit/test_plugin.py` (manifests agree with pyproject's
version, `.mcp.json` = `gpu mcp`, the server registers exactly the 7 spec tools + gpu_infer (D50), the 4
commands' frontmatter, human-only `gpu-approve`/`gpu-statusline` and their allowed-tools
(never `install`), the preview states exist, the skill's "never call ..." rule, the plugin's
wrapper = the packaged one and executable).
Status line (phase 6b): `tests/unit/statusline/` (`conftest.py` pins TZ=America/Los_Angeles;
`test_fast.py` goldens for every row state in plain text + stripped ANSI + raw ANSI codes, the
30-col grid vs the reference script's row 2, long/wide names, 2-row cap + counts, windows, stale pid /
overdue heartbeat, missing / broken / oversized files, stdin cwd, stdlib-only import;
`test_user_style.py` re-reads the reference status-line script
`tests/fixtures/statusline/statusline.sh` (tokens, COL, meter, `when()`) and runs a copy in a
tmp HOME: `week` lands on our column 2 and the wrapper keeps its lines byte for byte (skips
without jq); `test_wrapper.py` the bash wrapper (stdin bytes,
order, no gpu, failing original, installed layout, recursion guard); `test_install.py`
install/uninstall on tmp settings only (diff, prompt, --yes, --dry-run, refusal without a
tty, change-while-asking, backup, symlink, mode, restore byte-identical, real
`~/.claude/settings.json` unchanged); `test_statefile_phase6.py` the daemon side (trend,
route hours, resume seq, migrating, recent fields, heartbeat writer) and a real
DaemonRuntime's state.json rendered through the fast path; `test_timing.py` p50 < 50 ms).
Phase 7b (providers cleanup + inference lane, D50): `tests/unit/adapters/test_catalog_lanes.py`
(Modal excluded with quote/source/date, a user file cannot re-add it, verify-at-signup and
manual entries listed but never registered even when config enables them, no big-VRAM
provider by default) and `tests/unit/inference/` (`fakes.py` = one httpx.MockTransport
answering like Groq / Cloudflare / the Gemini OpenAI endpoint / HF's router by host +
test keys in the in-memory keyring; `test_clients.py` request shape per provider, Groq
live headers, the error taxonomy incl. per-minute vs used-up day, keys never in messages,
the login check endpoints; `test_ledger.py` windows incl. Pacific DST, estimates, live
readings, blocks, rollover, atomic file; `test_router.py` model availability, keys,
quota, waits, pins, passthrough, reasons; `test_service.py` fallbacks, blocks, one wait,
bad requests, test-mode guard, views, no prompt in logs; `test_batch.py`;
`test_frontends.py` a real in-process daemon with the service swapped for a mocked one:
/v1/infer*, `gpu infer` single / --json / --dry-run (exit 12 on no_fit) / --file / --list,
`gpu providers|quota --json` additions, `gpu login groq|gemini|cloudflare`, MCP
`gpu_infer`, shell /infer + /login hint). Updated for the Modal drop: router, policy,
plugin, MCP, shell footer/empty-state/screenshot and phase-5 CLI tests.
Notifications + doctor (phase 8a): `tests/unit/notify/` (`test_format.py` event -> kind table and texts, `test_backends_settings.py` config validation, argv-only AppleScript, terminal-notifier flags, backend choice incl. never-real-under-pytest, `test_service.py` no Store read in the commit callback, dedupe, rate limit + summary, slow / stuck / failing backends never block, `test_daemon_wiring.py` a real DaemonRuntime notifying finished / failed / approval / migrated once each (recorder backend), `test_cli.py` `gpu notify`) and `tests/unit/doctor/` (`conftest.py` = ProbeEnv over tmp dirs with a scripted runner, fake which, fake daemon client, fake HF whoami; `test_checks.py` every check's ok/warn/fail/skip + fix, incl. lightning `login_source` modes and the inference-keys row (secret values never read); `test_runner_catalog.py` parallelism, deadline, crash rows, drift, `--update-catalog`; `test_cli.py` the real `gpu doctor --json` limited with `--only` to checks inside the tmp home, against a stopped and an in-process daemon). No doctor test probes a real provider, ~/.claude, ~/.codex or the Keychain; only `--only`-limited runs use the real ProbeEnv.
Setup wizard (phase 8b, D52): `tests/unit/setup/` (`conftest.py` = a `Sandbox`: tmp user home, scripted subprocess runner for launchctl / uv / claude / kaggle / colab, fake `which`, `ScriptedUi` that fails on any unplanned question, fake HF whoami + Lightning bridge + browser sign-in, `gpu_router.paths.DEFAULT_HOME` pointed at the tmp data dir, and an autouse guard that the real ~/.claude settings/plugins, ~/.codex/config.toml, the launchd plist and ~/.claude/skills/colab did not change; `test_firstrun_state.py` stdlib-only import, first-run modes, setup.json resume/subset rules, bare-`gpu` dispatch; `test_tools.py`, `test_logins.py`, `test_service.py`, `test_integrations.py` one step each incl. declines, no terminal, `--yes`, dry run, failures, secrets never printed; `test_check.py` free-hours math, the smoke script run for real, smoke jobs through the shell suite's `InProcDaemon` on the fakes; `test_wizard.py` a full first run then an idempotent re-run, Ctrl-C + resume, `--yes` without a terminal, dry run, `--only`; `test_cli.py` the Typer front ends; `test_ui.py` the terminal prompts).
Phases 7-8 integration (D53): `tests/unit/router/test_gpu_rates.py` (Lightning L4 priced at
its own rate: share, quota rejection, pins; `gpu_name`), `tests/unit/statusline/test_install_perms.py`
(record dirs 0700, a loose one tightened, nothing above the data dir touched),
`tests/shell/test_login_steps.py` (`/login kaggle|lightning` keeps `gpu login ...`).
Sleep/wake recovery (D57): `tests/unit/engine/test_health_recheck.py` (60/120/240/480/900/900
backoff, reset on a healthy answer, the view's "re-checking in 40s" + `next_healthcheck_at`, a
driver-marked login problem re-checked in 1m, `FakeClock.suspend` and a long loop stall both
count as a wake, the grace before the post-wake checks, a post-wake failure retried in 1m, a
queued job's quota-reset wait cut short by the wake, the loop surviving a bookkeeping error,
a real-thread healthcheck timeout) and `tests/unit/core/test_clock.py::test_suspend_*`.
GitHub marketplace + CI (D58, D59): `tests/unit/test_plugin.py::test_repo_root_marketplace_*`,
`tests/unit/doctor/test_checks.py::test_plugin_installed_from_github`,
`tests/unit/setup/test_integrations.py::test_plugin_from_github_*`,
`tests/unit/core/test_secrets.py::test_child_processes_never_reach_the_macos_keychain`;
`tests/cli/test_review_fixes.py` keeps its GpuClient patches in `MonkeyPatch.context()`
(never `monkeypatch.undo()`, which also undoes the autouse `gpu_home`).

## Core invariants

Code comments cite these numbers. Never renumber; append new ones at the end.

1. **Only the daemon opens gpu.db.** `InstanceLock` (flock on `daemon.lock`) is acquired before the
   DB is opened; `Store.open` refuses without a held lock. Clients never import `store`.
2. **One owner of state.** CLI, shell, MCP and status line are thin clients of the daemon's HTTP API
   (or of state.json, read-only). Agents and front ends never call provider CLIs directly.
3. **Every job state change goes through `Store.transition`** (or `place`/`create_job`), which
   calls `statemachine.check_transition` first. No code writes `jobs.state` any other way.
4. **Every transition is durable and explained**: the state update and its `job_events` row
   commit in one transaction; after commit, one `job.transition` log line. Messages say what
   happened and what the tool does next (UX principle 4).
5. **At most one live attempt per job** (`submitting|submitted|running`), enforced by the unique
   partial index `attempts_one_live_per_job`.
6. **No double submits.** The attempt row (state `submitting`, `attempt_key`) commits BEFORE
   `adapter.submit()`. Only `DEFINITIVE_SUBMIT_ERRORS` prove nothing started; any other outcome
   is ambiguous and is resolved via `lookup_by_key` before the job is placed anywhere else. An
   adapter without `lookup_by_key` gets the attempt marked `abandoned` + a user warning.
7. **Adapters raise only the `AdapterError` taxonomy** (rule A3). The engine wraps anything else
   in `AdapterContractViolation` and treats it like `Unavailable`.
8. **Rate limits and outages never fail a job.** They cool the provider down and reroute with
   backoff; only budgets (`max_attempts`, `max_placements`, `max_queue_wait_s`) end in `failed`.
9. **The Store is synchronous and used only from the event-loop thread.** Adapter calls are
   blocking and run in `AdapterCaller`'s worker threads; they never touch the Store.
10. **Adapters are stateless toward gpu-router**: everything they need arrives as arguments
    (`Job`, `AttemptContext`, `RemoteRef`). Remote runs are tagged with the attempt key so they
    can be found again after a crash. Adapter-private scratch lives in `providers/<name>/`.
11. **The daemon never stops remote runs on its own shutdown or crash.** On restart, `Supervisor`
    applies `statemachine.RECOVERY` per state and reattaches. Killing the daemon loses nothing.
12. **Secrets only through `gpu_router.secrets`.** Never in config, SQLite, logs, events, bundles,
    state.json, exception messages or reprs. Carry values as `SecretStr`. `JobSpec.env` rejects
    secret-looking names.
13. **Injected time.** Engine, store, adapters, router and policy use the `Clock` they are given;
    `time.time`/`datetime.now` are banned by ruff (TID251) outside `clock.py`, `runner/`,
    `statusline/fast.py`.
14. **Stdlib-only import paths**: `entry.py`, `paths.py`, `statefile.py` (runtime imports),
    `statusline/fast.py`, `runner/*`. Heavy imports (pydantic, fastapi, httpx, typer, textual)
    happen after dispatch.
15. **`gpu_router/__init__.py` imports nothing.**
16. **The status line never calls the daemon or providers.** It reads state.json, returns in
    <50 ms, never raises, prints nothing when idle or on any error.
17. **API v1 is additive-only.** New fields/endpoints/Reason codes are fine; never rename,
    repurpose or remove. Errors use the envelope `{"error": {code, message, hint, detail}}`.
18. **The daemon binds 127.0.0.1 only**, requires the bearer token on every route except
    `GET /v1/health`, and rejects non-loopback `Host` and any browser `Origin` header.
19. **Free tiers only, one account per provider.** No adapter may create accounts, rotate
    accounts or use a provider that needs a card (spec hard constraints).
20. **Tests never touch real providers, the real Keychain or the user's data dir.** `conftest.py`
    points `GPU_ROUTER_HOME` at a tmp dir and installs an in-memory keyring for every test
    (child processes get `PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring`, D59);
    real-provider tests need `@pytest.mark.real_provider` + `GPU_ROUTER_REAL_PROVIDERS`.
21. **Migrations and config versions are append-only once a phase ships.** Change the schema with
    `0002_*.sql`; change config shape with `CONFIG_MIGRATIONS[n]`. 0001 is editable until phase 1
    is ticked.
22. **Outputs land in `<project_dir>/runs/<id[:4]>/`**, fixed at job creation (`jobs.outputs_dir`).

## Job state machine (source of truth: `statemachine.py`)

| From | Legal to | Typical reasons |
|---|---|---|
| (new) | queued | submitted |
| queued | routing, failed, cancelled | routing_started; gave_up; user_cancel |
| routing | queued, awaiting_approval, provisioning, failed, cancelled | no_capacity; approval_required; placed; no_provider_fits/gave_up |
| awaiting_approval | provisioning, queued, failed, denied, cancelled | placed (after `approved` note); no_capacity; denied/approval_expired |
| provisioning | running, queued, done, failed, cancelling, cancelled | started; rate_limited/provider_unavailable/quota_exhausted/auth_required/invalid_for_provider/provision_timeout/lost_before_start; completed; script_failed/provider_permanent |
| running | checkpointing, done, failed, migrating, cancelling, cancelled | checkpoint_begin; completed; script_failed/interactive_lost; session_lost/status_lost/quota_exhausted/handoff; user_cancel; remote_cancelled |
| checkpointing | running, done, failed, migrating, cancelling, cancelled | checkpoint_end/checkpoint_stalled; as running |
| migrating | provisioning, queued, awaiting_approval, done, failed, cancelling, cancelled | placed (resume from latest checkpoint); no_capacity; ... |
| cancelling | cancelled | cancelled; cancel_unconfirmed |
| done, failed, cancelled, denied | (terminal) | |

`cancel_target(state, has_live_attempt)`: live attempt in a REMOTE_STATE -> `cancelling`, else
`cancelled`; terminal or already cancelling -> no-op. Recovery after restart (`RECOVERY`):
queued RESUME, routing REROUTE, awaiting_approval WAIT, provisioning RESOLVE_ATTEMPT,
running/checkpointing REATTACH, migrating REMIGRATE, cancelling RECANCEL, terminal NONE.

Attempt states: `submitting -> submitted -> running -> {succeeded, failed, lost, cancelled}`;
`submitting -> rejected` (definitive submit error); any live state `-> abandoned`. Terminal
attempt states stamp `ended_at`. Remote `pending` maps to attempt `submitted`.

### Driver semantics (engine, phase 1)

- **Route**: queued -> routing -> router. PLACE: policy check, then `store.place()` (-> provisioning).
  WAIT: -> queued with `not_before` = earliest cooldown end (reason no_capacity). NO_FIT: -> failed
  (no_provider). Budgets checked before each routing: accepted attempts < `max_attempts`, attempt
  rows < `max_placements`, now - `waiting_since` < `max_queue_wait_s`; else gave_up.
- **Approval**: policy says required -> awaiting_approval (approval_reason, provider/route_reason set).
  Approve = `record_approval` note, then the driver re-routes and places without asking again.
  Deny -> denied. `approval_timeout_s` -> denied (approval_expired).
- **Submit** (worker thread): success -> `record_submission`. Definitive error -> attempt
  `rejected` (+`error_kind`), provider reaction per the errors.py table, job -> queued with
  backoff, InvalidJob excludes the provider for this job. Ambiguous -> note submit_ambiguous,
  `lookup_by_key` until resolved (found -> record_submission; NotFound -> rejected).
- **Poll** every `poll_interval_s` (catalog, config override): `status()`, capture logs
  (`logs(follow=False, since=log_cursor)`), parse protocol, then map: pending (provisioning, until
  `provision_timeout_s` -> cancel + reroute), running (-> running, reason started), succeeded
  (fetch outputs to `outputs_dir`, -> done), failed with exit code (-> failed, user_error,
  script_failed), lost (-> migrating, session_lost; interactive jobs -> failed interactive_lost),
  cancelled not by us (-> cancelled, remote_cancelled). `status()` errors: note after
  `status_stale_after_s`, attempt lost after `unreachable_lost_after_s`.
- **Migrate**: ensure the old attempt is terminal (cancel if needed), then route again with
  `resume_checkpoint_id` = latest checkpoint.
- **Cancel**: user -> `cancel_target`; cancelling: `adapter.cancel()` (idempotent; NotFound = done),
  confirm via status, -> cancelled; after `cancel_timeout_s` -> cancelled (cancel_unconfirmed).
  If the remote had already succeeded, fetch outputs and add note outputs_kept.
- **Internal errors**: an unexpected exception in a driver step adds an internal_error note and
  retries with backoff; `internal_error_limit` consecutive -> failed (internal).

## Data model (`db/migrations/0001_initial.sql`)

| Table | Holds |
|---|---|
| `schema_version` | applied migrations |
| `meta` | instance_id, last_start_at, last_clean_shutdown_at, last_pid |
| `jobs` | one row per job: state, spec_json (+hash), provider/gpu/current attempt, budgets counters, route/approval reasons, progress, last metrics, checkpoint count, outputs, failure_kind, `version` |
| `job_events` | append-only: every transition (`kind=transition`) and fact (`kind=note`); global `seq` is the event-feed cursor |
| `attempts` | one row per placement: provider, `attempt_key`, state, remote ids/meta, errors, log cursor, timings; doubles as the usage ledger |
| `checkpoints` | `<job>.c<seq>` from runner `ckpt_end` lines; seq monotonic per job across attempts |
| `provider_state` | health, cooldown_until, exhausted_until, consecutive_failures |
| `quota_snapshots` | live/estimated quota observations (phase 5) |
| `data_cache` | dataset content hash -> HF Hub URI (phase 5) |

CHECK constraints mirror the enums; `tests/unit/core/test_schema.py` keeps them in sync with
`statemachine`/`models`. Timestamps: REAL epoch seconds in SQLite/Python, ISO-8601 `Z` in JSON.
Job id: 12 hex; display `short_id` >= 4 chars; attempt id `<job>.<n>`; key `gpu-<job>-<n>`.

## Daemon HTTP API (`daemon/routes.py`, models in `api.py`)

Base `http://127.0.0.1:<port>`; port in `daemon.json` (default 47291). Header
`Authorization: Bearer <contents of daemon.token>` (0600, created by the daemon). Responses carry
`X-Gpu-Router-Version`. Success bodies are the resource itself; errors are
`{"error": {"code", "message", "hint", "detail"}}` with `code` from `errors.ErrorCode` and the
exception's `http_status`. Validation errors -> 400 `invalid_request`/`invalid_spec`. During
recovery, mutating routes return 503 `not_ready`. `{ref}` = full id or unique prefix.

| Method + path | Body -> response |
|---|---|
| `GET /v1/health` (no auth) | -> `HealthView` {ok, version, api_version, ready, pid, started_at} |
| `GET /v1/status` | -> `StatusView` {counts, active jobs, providers} |
| `POST /v1/jobs` | `SubmitRequest` {spec}; header `Idempotency-Key` -> 201 `JobView` (200 if replay; 409 `conflict` if the key was used for a different spec) |
| `GET /v1/jobs?state=&project_dir=&limit=&before=` | -> `JobList` {jobs, next_before} |
| `GET /v1/jobs/{ref}` | -> `JobDetail` {job, attempts, checkpoints, events (last 50)} |
| `GET /v1/jobs/{ref}/events?after=` | -> `EventList` {events, next} |
| `GET /v1/jobs/{ref}/logs?attempt=&offset=&follow=&protocol=` | NDJSON stream of `LogRecord` (`protocol=true` includes `::gpu::` lines) |
| `POST /v1/jobs/{ref}/cancel` | -> `JobView` (idempotent) |
| `POST /v1/jobs/{ref}/approve` / `deny` | `DecisionRequest` {reason?} -> `JobView` |
| `POST /v1/jobs/{ref}/fetch` | -> `JobView` (re-fetch outputs; async, note fetch_requested) |
| `POST /v1/route` | `SubmitRequest` -> `RouteDecision` (dry run, no DB writes) |
| `GET /v1/providers`, `GET /v1/providers/{name}` | -> `ProviderView` list / one |
| `POST /v1/providers/{name}/healthcheck` | -> `ProviderView` |
| `GET /v1/quota?refresh=` | -> list of `QuotaSnapshot` (phase 5: quota ledger views; stale live readings are refreshed first, waiting at most `routing.quota.wait_s`; `refresh=true` re-reads every live provider) |
| `POST /v1/infer`, `POST /v1/infer/route` | `InferRequest` {model, prompt\|messages, system?, provider?, max_tokens?, temperature?, wait_s} -> `InferResult` / `InferRoute` (phase 7b inference lane, `inference/models.py`, D50) |
| `GET /v1/infer/quota`, `GET /v1/infer/providers` | -> list of `InferQuotaView` (daily-quota ledger, live/est) / `InferProviderView` (key stored?, models, limits) |
| `GET /v1/policy`, `PUT /v1/policy` | -> `PolicyView` {name, editable, policy, defaults, config_path}; PUT takes a full `PolicyConfig`, persists it to config.yaml `policy:` and applies it at once (phase 5) |
| `GET /v1/events?after=&timeout=` | long poll (max 30 s) -> `EventList` |
| `POST /v1/daemon/shutdown` | -> 202 |

**Log streaming**: `application/x-ndjson`, one `LogRecord` per line: `{"attempt": n, "offset": i,
"line": "..."}`; `offset` is the 0-based line index in that attempt's log file so clients resume
with `offset=`. Without `attempt`, all attempts in order. `follow=true` keeps streaming until the
job is terminal, then sends `{"eof": true, "state": "<job state>"}`; a `{"heartbeat": true}` line
every 15 s keeps idle connections alive. Lines are already redacted.

## Agent integration (phase 6a; owner: mcp)

**MCP server** `gpu mcp` (entry.py dispatches it before typer, so nothing reaches stdout but
the transport; also a Typer `mcp` command for `gpu --help`, and `python -m gpu_router.mcp`):
FastMCP 4 on stdio, server `gpu-router`, `show_banner=False` (the banner is also fastmcp's
only update check). Exactly `gpu_submit gpu_status gpu_logs gpu_fetch gpu_cancel gpu_quota
gpu_route`, plus `gpu_infer` (phase 7b, additive: the inference lane, D50); no approve/deny tool. Each call connects through `daemon.spawn.connect`
(auto-start like the CLI, `GPU_ROUTER_NO_AUTOSTART` honoured) with client name `mcp`, so
the daemon records actor `agent`. Specs come from `cli.app._build` (gpu.yaml + tool args =
`gpu run -C <project_dir>`), then `source=agent` + label `via: mcp` (agent approval rules).
Results reuse the CLI `--json` shapes with nulls dropped and lists trimmed (`verbose=true`
= full documents): `{"job", "guidance", "submitted"}`, JobDetail keys (last 3 attempts /
checkpoints, last N events), `{"spec", "route", "approval"}`, the `gpu fetch --json`
payload + `outputs` listing, `{"quota", "summary"}`; the status overview uses compact rows.
Every job result has `guidance` {state, meaning, finished, route, next, follow
(`wait` for jobs of <= 15 min, `report_and_stop` for longer/unknown ones, `end_turn` while
waiting for approval; D48), poll_every_s | needs_approval, why, tell_user ("Job <id>
needs your approval ...: <reason>. It would run on <provider GPU>. Declared runtime: ...
/gpu-approve <id> ... `gpu approve <id>`": only gpu-router's own text, never the job
name), rules | outputs_dir} and `untrusted` (job names, messages, metric names, output
names and logs are job-controlled). Metrics are cut to 12 per job (`metrics_not_shown`),
the overview to 20 active / 10 recent rows (approvals first; `active_not_shown`). gpu_submit
is idempotent (D48): `request_id` = key `mcp-req-<id>`; without it the key is sha256 of the
spec, chained past identical jobs that ended (found in the project's job list), so a retry
after a client timeout returns the same job (`submitted: false`, `duplicate_of`).
Errors are tool errors whose text is the CLI envelope `{"error": {...}}`.
Bounds: status/submit `wait_s` <= 50 (Codex times tools out at 60 s), fetch waits <= 300
(default 45; `in_progress` then), logs `tail` <= 1000, lines cut at 2000 chars, 60k chars
per call, cursor `next_since` = `"<attempt>:<line offset>"` crossing attempts with a
`── attempt N on <provider> ──` line. Refs pass `ids.normalize_ref` before connecting (a bad
ref never starts a daemon or reaches a URL; the client quotes again). `gpu_route` adds an
approval preview: the daemon's rules (GET /v1/policy) evaluated locally on a placeholder
Job; the daemon decides again at placement; it also carries
`hours`, `hours_source` and `stopped_for_approval_after_h` (D48).

**Plugin** `plugin/` is both the checkout's marketplace (`.claude-plugin/marketplace.json`, name
`gpu-router-local`, plugin source `./`) and the plugin (`.claude-plugin/plugin.json`,
`.mcp.json` = `gpu mcp`, which needs `gpu` on PATH; `skills/gpu-router/SKILL.md`, the one
skill: the Codex copy in `docs/codex/AGENTS-section.md` must be kept in sync;
`commands/gpu-run.md`, `gpu-status.md`, `gpu-approve.md` with `disable-model-invocation:
true` + `allowed-tools: Bash(gpu approve:*), Bash(gpu status:*)`, a human-only command that
validates the hex id before running `gpu approve <id> --json`; `commands/gpu-statusline.md`,
also human-only, `allowed-tools: Bash(gpu statusline status:*), Bash(gpu statusline
preview:*)`: shows whether the rows are installed plus a preview, and tells the user to run
`gpu statusline install` in their terminal (it never installs); `statusline/` is 6b's
`gpu-statusline.sh`, byte-identical to the packaged copy that `gpu statusline install`
copies to `<home>/statusline/` (a plugin cannot set `statusLine`, and the versioned plugin
cache path would break on every plugin update, D47)). The repo root carries a second
marketplace, `.claude-plugin/marketplace.json` (name `gpu-router`, one entry, source
`./plugin`, same description/version as the local one; `tests/unit/test_plugin.py` keeps them
in step), so users install from GitHub without a clone (D58). Install (run by hand; never from
a build session):

```bash
# from GitHub (what the README leads with; `gpu` must be on PATH for the MCP server)
claude plugin marketplace add ayushg8/gpu-router
claude plugin install gpu-router@gpu-router
# from a checkout (repo root): its own marketplace, plugin edits need no push
uv tool install --editable .              # puts `gpu` on PATH
claude plugin marketplace add ./plugin
claude plugin install gpu-router@gpu-router-local
# after editing the plugin: claude plugin marketplace update gpu-router-local
#                           claude plugin update gpu-router@gpu-router-local
# MCP only, no plugin:      claude mcp add gpu-router -- gpu mcp
# try without installing:   claude --plugin-dir ./plugin
```

Checks (read-only): `claude plugin validate plugin` (validates the marketplace when both
manifests exist), `claude plugin validate --strict .` (the repo-root marketplace),
`claude plugin validate --strict plugin/.claude-plugin/plugin.json`,
`claude --plugin-dir plugin plugin details gpu-router` (4 commands + 1 skill + 1 MCP server,
~258 tokens always on as of the phase-6 integration). The add + install commands above were
run against a throwaway `CLAUDE_CONFIG_DIR` (Claude Code 2.1.281): installed, enabled, same
inventory, `statusline/gpu-statusline.sh` in the cache with its exec bit.
**Codex**: `docs/codex/AGENTS-section.md` (`codex mcp add gpu-router -- gpu mcp`, or the
config.toml entry with `tool_timeout_sec = 330`, both checked with `codex mcp get` 0.154.0
on a throwaway `CODEX_HOME`; the AGENTS.md section sits between `gpu-router:begin/end`
markers).

**Phase 6 install: all of it is run by hand, never from a build session** (none was run on the
dev machine's real `~/.claude` or `~/.codex`; run from the repo root):

```bash
# 1. gpu on PATH (needed by the plugin's MCP server, the status line and Codex)
uv tool install --editable .
# 2. Claude Code plugin: MCP server + skill + /gpu-run /gpu-status /gpu-approve /gpu-statusline
#    (or from GitHub: claude plugin marketplace add ayushg8/gpu-router &&
#     claude plugin install gpu-router@gpu-router)
claude plugin marketplace add ./plugin
claude plugin install gpu-router@gpu-router-local
# 3. status line rows (prints the settings.json diff, asks [y/N]; undo: gpu statusline uninstall)
gpu statusline install
# 4. Codex: prefer the config.toml entry in docs/codex/AGENTS-section.md (tool_timeout_sec =
#    330), else `codex mcp add gpu-router -- gpu mcp`; then paste that file's marked section
#    into AGENTS.md
# 5. the global ~/.claude/skills/colab skill also claims "run this on a GPU" and drives the
#    colab CLI directly (the spec keeps Colab's skill inside the adapter, D48). Offer to
#    disable it (it is kept, just moved out of the skills dir):
[ -d ~/.claude/skills/colab ] && read -r -p "disable the colab skill for job running? [y/N] " a \
  && [ "$a" = y ] && mv ~/.claude/skills/colab ~/.claude/skills-disabled-colab
```
(`gpu doctor` / `/doctor` detect `~/.claude/skills/colab` and print this command as the fix, phase 8a.)

## Adapter contract (`adapters/base.py`)

Adding a provider = adapter + `providers.yaml` entry + passing `tests/contract/`.

| Call | Returns | Notes |
|---|---|---|
| `submit(job, ctx)` | `RemoteRef` | ctx: `AttemptContext` (attempt id/key/n, bundle, resume ckpt, env, secrets) |
| `status(ref)` | `RemoteStatus` | phase pending/running/succeeded/failed/cancelled/lost |
| `logs(ref, follow, since)` | `Iterator[LogChunk]` | `since` = cursor from a previous chunk |
| `fetch(ref, dest)` | `FetchResult` | outputs into `dest` |
| `cancel(ref)` | `None` | idempotent |
| `quota()` | `QuotaSnapshot` | used, limit, resets_at, source live/estimate |
| `healthcheck()` | `Health` | ok or the reason not |
| `lookup_by_key(key)` | `RemoteRef or None` | crash recovery; only if `capabilities.lookup_by_key` |
| `stage_data(path, sha256, files)` | `StagedData` {uri, uploaded, where} | a `data:` dataset kept in the provider's own store, once per content (D61); only if `capabilities.stage_data`; bounded below `engine.timeouts.stage_data` (3600) |

Rules: **A1** implement all calls above; declare `Capabilities` honestly. **A2** methods are
blocking and run in worker threads; bound every subprocess/network call below
`engine.timeouts.<call>`. **A3** raise only `errors.AdapterError` subclasses; from `submit`,
raise a `DEFINITIVE_SUBMIT_ERRORS` class only when certain nothing was created, else `Unavailable`.
**A4** `submit` is idempotent per `ctx.attempt_key` and tags the remote run with it. **A5**
`cancel` of a finished or unknown run returns normally. **A6** `status`/`logs`/`quota`/
`healthcheck` have no side effects. **A7** `logs(since=c)` never repeats lines before cursor `c`;
cursors are opaque strings that survive a daemon restart. **A8** no DB access, no secret values in
exceptions, logs or `remote_meta`. **A9** `fetch` is re-runnable and never deletes existing files
in `dest` outside what it writes. **A10** use only the injected `clock` for time.

## Job bundle, remote runner, `gpu` helper

**Built by the daemon at submit** (`Supervisor.submit`, via `EngineDeps.bundler` =
`packaging.BundleBuilder`, set in `DaemonRuntime`; engine unit tests leave it None): 1.
`build_bundle(spec.project_dir, spec)` in a worker thread, BEFORE `create_job`, so a bad project
(missing dir, over `DEFAULT_MAX_BUNDLE_MB`=200, bad explicit deps file) is a 400 `invalid_spec`
(`BundleError`/`BundleTooLarge` subclass `InvalidSpec`, with hint) and never becomes a job;
2. `create_job`; 3. `update_job(bundle_sha256=...)`; 4. `materialize` -> `jobs/<id>/bundle.tar.gz`
(hard link to `<home>/bundles/<sha>.tar.gz`) + `jobs/<id>/bundle/` (extracted); 5. start the
driver. If the daemon dies between 3 and 4, `JobDriver._attempt_context` re-materializes from the
cache. The CLI never packages for real; `gpu run --dry-run` may call `build_bundle` to preview.

Archive layout: `manifest.json` + `code/<rel>` + `gpu_runner/{gpu.py,bootstrap.py}`. Files: git
repos ship `git ls-files -co --exclude-standard` (tracked + untracked-not-ignored, every
untracked file that ships named in a warning; run from `project_dir`, so a repo subdir ships
only its subtree); dirs with no `.git` in themselves or any parent fall back to a walk with
`files.DEFAULT_IGNORES` and a warning (dir symlinks reported). Inside a repo, a failing git
(no Command Line Tools, dubious ownership, timeout) is a `BundleError` with git's message and
a hint, never a silent walk (D25). Credential-looking files never ship, even if tracked
(invariant 12): `files.SECRET_PATTERNS` on the name (`.env*`, `*.env`, `.envrc`, keys, `.npmrc`,
`token.json`, `client_secret*.json`, `secrets.toml`, ...), `SECRET_DIRS` on any path part
(`.aws/`, `.ssh/`, `.kaggle/`, `.huggingface/`, `.config/gcloud/`, ...), and for symlinks the
target too. Symlinks to files inside the project are dereferenced; symlinks resolving outside
it, dir symlinks and submodules are skipped with a warning. Deterministic: sorted
paths, mtime 1980-01-01, uid/gid 0, modes 0644/0755, gzip mtime 0, manifest JSON sorted and
timestamp-free (manifest is the last member so `files.tree_sha256` is computed in one pass).
Manifest keys: `manifest_version`, `name`, `entrypoint` {script, command, args}, `deps` {kind,
file, packages, python_requires}, `estimate` {vram_gb, hours, vram_source, hours_source,
model_params_b, mode, reasons}, `files` {count, bytes, source git|walk, tree_sha256,
untracked}, `checkpoint_interval_min`, `runner`, `warnings` (includes deps warnings). A
missing or git-ignored entry script is a warning, not an error (the CLI checks gpu.yaml's
script up front, with or without script args). The archive is fsynced before it enters the
cache and a rebuild replaces an existing entry (a truncated entry heals). If `materialize`
fails in `Supervisor.submit`, the driver is started anyway and materializes before submit.

**bootstrap.py** (`python gpu_runner/bootstrap.py` inside an extracted bundle, or
`--bundle x.tar.gz --workdir d`): removes a stale EXIT, rotates `job.log` to `job.log.prev`,
prints `hello`; safe-extracts into `<workdir>/bundle` (stamped with the archive sha256; a
different archive is re-extracted into a clean dir); claims `GPU_CHECKPOINT_DIR` and
`GPU_OUTPUT_DIR` for `GPU_ROUTER_JOB_ID` (sibling stamp `.<dir>.gpu-router-job`; a dir stamped
by another job is emptied); `--resume <tar.gz|dir>` restores into a cleared `GPU_RESUME_DIR`
(default `<workdir>/resume`) and seeds an empty `GPU_CHECKPOINT_DIR` with real copies (never
hard links); `pip install -r <file>` or `pip install <pyproject deps>` (skip: `--skip-install`
/ `GPU_SKIP_INSTALL=1`; install failure = job not started: `::gpu:: {"t":"install_failed",
"code":<pip>}` and exit `INSTALL_FAILED_EXIT` = 90); runs the entrypoint from `code/` in its
own process group with `PYTHONPATH=gpu_runner:code`, stdout+stderr merged and teed to stdout +
`<workdir>/job.log` (tqdm `\r` updates become separate lines, at most one per 5 s; a protocol
line glued behind a `\r` segment is split back out); when the entrypoint exits, the group gets
`DRAIN_GRACE_S` (2 s) to release the pipe, then SIGKILL; SIGTERM/SIGINT go to the whole
group; `::gpu:: {"t":"heartbeat",...}` every `--heartbeat-s` (60; `protocol.parse_line`
ignores unknown types, so neither heartbeat nor install_failed is a protocol change); with
`--checkpoint-sync-dir` archives the checkpoint dir (gzip
level 1, symlinks dereferenced, `.gpu-*` bookkeeping excluded) every `checkpoint_interval_min`
on its own thread when it changed, plus once at exit, emitting `ckpt_begin`/`ckpt_end` (uri
`file://...`, seq from `--ckpt-seq-start`, which the adapter sets to latest seq + 1; keeps the
newest 3 archives); with `--storage`/`GPU_STORAGE` (phase 5, D40) it publishes the same
fingerprinted files instead to checkpoint storage (`gpu_runner/storage.py`: staged copy
re-verified, `jobs/<job>/ckpt-NNNN/` + manifest, then `latest.json`; seq =
max(`--ckpt-seq-start`, latest + 1); newest `GPU_CKPT_KEEP` kept; stops when `owner.json`
names a later attempt), restores `GPU_RESUME_URI` (`hf://` downloaded; checked against its
manifest; gone or torn = storage's latest.json, then older intact ones, fresh start only when
none exists; missing while latest.json is unreadable, a restore OSError, or repeated
download failure = exit 90, D44), links/downloads `GPU_DATA` datasets under
`GPU_DATA_DIR/<mount>` (failure = exit 90), pushes `heartbeat.json` + `log-tail.json` every
`GPU_STATUS_PUSH_S` and answers `control.json` checkpoint requests (`<ckpt dir>/.gpu-checkpoint-request`,
wait, sync, `control-ack.json`, then frozen after a handoff); the storage token is read from
`GPU_STORAGE_TOKEN_FILE` (0600, deleted after reading; Kaggle/Colab launchers, D44) or
`GPU_STORAGE_TOKEN` is popped
from the environment first (job and pip never see it); finally `exit` {code} and `<workdir>/EXIT` = code (signals -> 128+n), for
every outcome including bad arguments and a SIGTERM outside the entrypoint. Every flag also
reads an env var (`GPU_BUNDLE`, `GPU_WORKDIR`, `GPU_RESUME_SRC`, `GPU_LOG_FILE`,
`GPU_EXIT_FILE`, `GPU_HEARTBEAT_S`, `GPU_CHECKPOINT_SYNC_DIR`, `GPU_CKPT_SEQ_START`; bad
numbers fall back to the default with a note). Relative dirs (flags and `GPU_*_DIR`) are
resolved against bootstrap's cwd before use and export.

Remote env: `GPU_ROUTER_JOB_ID`, `GPU_ROUTER_ATTEMPT`, `GPU_ROUTER_PROTOCOL=1`,
`GPU_CHECKPOINT_DIR` (save checkpoints here), `GPU_RESUME_DIR` (latest checkpoint restored here,
absent on first attempt), `GPU_OUTPUT_DIR` (fetched to `runs/<id4>/`), `GPU_DATA_DIR` (`/data`).
Phase 5 (engine-set through the adapter, D40): `GPU_STORAGE`, `GPU_RESUME_URI`,
`GPU_STATUS_PUSH_S`, `GPU_CONTROL_POLL_S`, `GPU_CKPT_KEEP`, `GPU_SECRET_NAMES`, `GPU_DATA`
(JSON), secret `GPU_STORAGE_TOKEN`; test/dev knob `GPU_CHECKPOINT_INTERVAL_S`.

User helper (`import gpu`, D2/spec decision 3): `gpu.log(step=i, **metrics)`,
`gpu.total_steps(n)`, `gpu.checkpoint_dir()`, `gpu.resume_dir()` (None on a fresh start),
`gpu.latest_checkpoint()`, `gpu.is_resumed()`, `gpu.output_dir()`, `gpu.data_dir()`,
`gpu.enabled()`, `gpu.checkpoint_requested()` (phase 5: True while a planned handoff waits
for a save), `with gpu.atomic_checkpoint("last.pt") as tmp: torch.save(s, tmp)` (writes
under `checkpoint_dir()/.gpu-tmp/`, renamed into place on success; re-raises). Dirs are `pathlib.Path`s and are created. Accepts numpy/torch scalars (`.item()`),
drops non-finite/non-numeric values. Outside a gpu-router run (`GPU_ROUTER_PROTOCOL` unset) it
prints plain `step=10/100 loss=0.41` text (which the stdout fallback parses), uses ./checkpoints
and ./outputs, and never fails.

**Line protocol** (`protocol.py`, real): a line `::gpu:: <compact JSON>` with `"t"` one of
`hello` {v, runner}, `total` {steps}, `metric` {step?, total?, metrics{name: float}},
`ckpt_begin` {seq}, `ckpt_end` {seq, uri, step?, size?, sha256?}, `exit` {code}. Protocol lines
are hidden from `gpu logs` by default. **Stdout fallback** (`protocol.StdoutMetricParser`):
`loss=0.41`, `loss: 0.41`, `step 10/100`, `step=10`, `Epoch 3/10`, tqdm `45%|...| 45/100`. Once
a job emits any protocol metric, fallback results are ignored (`progress_source=helper`).

## Claude Code status line (phase 6b; owner: statusline)

Source of truth for the look: a reference status-line script, checked in as
`tests/fixtures/statusline/statusline.sh` (installed as settings.json `statusLine` =
`{"type": "command", "command": "bash \"$HOME/.claude/statusline.sh\"", "refreshInterval":
2}`), read 2026-09-24. Copied exactly (`tests/unit/statusline/test_user_style.py` re-reads the
script and fails on drift):

| Token | Value in the script | Used for |
|---|---|---|
| `COL` | `30` visible columns; `cell()` pads text to it with at least 1 space (row 1 uses 2) | column 2 starts at col 30 on every row; a gpu left cell is at most 28 wide (`GUTTER` 2, like row 1) |
| meter | 10 cells, `filled = (pct*10+99)/100` (ceil), `█` in the state colour, then `░` in dim, then `\033[0m` (no reset between the two runs) | provider quota bar and job progress bar |
| `fg` | `\033[38;5;252m` (light grey, NOT bold; the spec's "bold white %" is this) | values, percentages |
| `dim` | `\033[38;5;243m` | labels (`gpu`, provider name), `↻` resets, `·` separators, secondary text |
| `warn` / `crit` | `\033[38;5;179m` at >= 70%, `\033[38;5;174m` at >= 90% (meter fill + number) | provider quota only (a budget, like `week`); job progress is never coloured |
| `accent` | `\033[38;5;75m` blue | the model name only; never used by gpu rows |
| off | `\033[0m` | after every coloured run |
| reset time | `when()`: under 22 h -> `%-I%p`, else `%a %-I%p`, piped through `tr 'AMP' 'amp'` (so `1pm`, `Sat 5pm`, and Monday prints `mon`) | `↻` on the provider quota |
| row 2 | `session` + ` ` + meter + ` ` + `20%` + ` ↻1pm`, padded to COL, then `week` + ` ` + meter + ` ` + `43%` + ` ↻Tue 4pm`; `<1%` for 0 < x < 1 | the running row mirrors it: `gpu` + bar + % + `1:50 left`, then provider + quota bar + % + `↻reset` (`~73%` = ledger estimate) |
| row 3 | `big` fg + ` ` + dim `·` + ` ` + dim `now`, fitted to COL-1 (drop the live half, then cut to COL-2 + `…`), padded, then the ETA | the running detail row: job name · GPU, then `loss 0.412 ↓  ckpt 3m ago`; every left cell uses the same fit rule (a dropped fact moves to the right column, never lost) |
| output | rows joined by `\n`, no trailing newline | the wrapper appends `\n` + gpu rows |

State icons are the only other colour: `⏸` warn yellow, `✗` crit red, `✓` `\033[38;5;108m`
(muted green; the script has no green). `↪` is fg. Everything is width 1 (no `⚡`).

**Rows** (`fast.build_rows`, pure; goldens in `test_fast.py`, `gpu statusline preview` shows
them all): running = `gpu <bar> 38% 1:50 left  kaggle <quota> 73% ↻Sat 5pm` + `train_yolo ·
2×T4  loss 0.412 ↓  ckpt 3m ago` (bar = steps when a total is known, from the helper or a
parsed `step i/N`, else elapsed vs the session cap `(1:42 of 12h)`, else `1:42 elapsed`;
remaining = daemon `eta_s` minus the file's age, 10h+ drops the word `left`); approval =
`gpu ⏸ eval.py → colab T4 · ~20m  /gpu-approve` (id added when 2+ wait); finished (for
`finished_visible_s`) = `gpu ✓ name · 3h12m  → ./runs/a7f2` (`<project>/runs/..` when the
session's cwd, from the stdin JSON, is elsewhere; `gpu fetch <id>` if outputs were not
fetched); migrated (for `migrated_visible_s`) = `gpu ↪ name  colab → kaggle  resumed · ckpt
4`, then the bar row while running (`from colab` / `resuming` before placement); failed =
`gpu ✗ name · 14m  exit 1 · gpu logs <id>`; provisioning/queued = `gpu name → kaggle 2×T4
starting` / `queued · retry in 2m`; cancelled/denied/idle print nothing. Order: approval >
failed > running/migrated > done > starting; the first block fills up to 2 rows, a second
job gets its first row if room is left, the rest become `+1 running · +2 queued · +1 done`
(dim) at the end of the last row. Stale: daemon pid gone -> nothing, or `gpu daemon not
running  gpu daemon start` (dim) when active jobs were listed; heartbeat older than
max(5 x `heartbeat_s`, 300 s) with active jobs -> `gpu daemon not responding`.

**state.json (additive, schema 1)**: ActiveJob + `script`, `project_dir`, `attempt_n`,
`resumed_from_seq`, `route_hours`/`route_hours_source` (spec hours, else the approval
event's router estimate), `migrate_reason`, `metric.trend` (tail of metrics.jsonl, shell's
trend rule), `migrated_from` also while still `migrating`; RecentJob + `project_dir`,
`outputs_path`, `outputs_fetched`, `failure_kind`, `exit_code`, `provider`, `gpu`;
ProviderSummary + `unlimited`, `remaining` (ledger); snapshot + `finished_visible_s`,
`migrated_visible_s`, `heartbeat_s`; ActiveJob/RecentJob + `origin` (D56: each Claude
Code status line draws only its own session's jobs). `StateFileWriter(heartbeat_s=60)` rewrites while the
last snapshot listed active jobs (ETA / quota stay fresh, a wedged daemon is detectable);
it checks WALL time every `HEARTBEAT_TICK_S` (5 s; the loop clock stops while a Mac
sleeps, D48). `eta_s` is this attempt's own rate (D48): two+ metrics.jsonl points of the
current attempt, else `elapsed/step` for a fresh first attempt only, else one point plus
the resume checkpoint's step, else null; fast.py has no local fallback. Names, metric
names, scripts, messages and reasons are stripped of C0/C1 controls and bidi overrides
(`statefile.clean_text`; fast.py `_str` again).

**Wrapper + install** (D48 layout): one record per settings file,
`<home>/statusline/<sha1(realpath(settings))[:12]>/` (wrapper copy, original-command,
original.json with the `settings` path it belongs to, gpu-bin, gpu-home). Claude Code runs
`f="$HOME/Library/Application Support/gpu-router/statusline/<key>/gpu-statusline.sh"; if
[ -f "$f" ]; then bash "$f"; else eval '<original command>'; fi` (no original: `true`), so a
deleted data dir leaves the user's own line and settings.json keeps the original command.
The wrapper reads stdin once, starts `gpu status --line --stdin` on fd 3 (first line = its
pid: `sh -c 'echo $$; exec gpu ...'`), runs the original command (`$GPU_STATUSLINE_ORIGINAL`,
else `original-command` beside it) through `/bin/sh -c` with the same bytes, prints its
output unchanged at once, then waits at most ~1 s (`read -t 1`, bash 3.2) for the rows
(timeout: no rows, the gpu process is killed) and prints `\n` + rows; gpu/home from
`$GPU_ROUTER_BIN`/`$GPU_ROUTER_HOME`, else `gpu-bin`/`gpu-home` beside it; never recurses
into itself; always exits 0. `gpu statusline install` = diff + `[y/N]` (or `--yes`; no tty
and no `--yes` = exit 2; `--dry-run`), refuses a settings.json that changed while asking or
whose directory is not writable, refuses a statusLine that already runs a wrapper from
another data dir / record (names it), writes the record, a
`settings.json.gpu-router-<stamp>[-N].bak` (O_EXCL, never overwrites), then settings.json
atomically (symlink target, mode kept, other keys untouched); a failed write removes the
record and reports "cannot write <path>: <why>; nothing changed in settings.json".
`uninstall` only acts when the command runs THIS home's record for THIS file (else names
the data dir), restores that record's statusLine object exactly (the command embedded in
settings.json when the record is gone), removes only that record.
NOT run in phase 6b (installing is left to the maintainer): only `--dry-run` against a copy
of the statusLine key. Measured (200 runs, M-series, 2026-09-24): `gpu status --line` p50 20.9 ms / p95
23.5 ms (python -c pass: 16.6 / 18.5); the wrapper around a sandboxed copy of the reference
script adds ~12 ms p50 (45.0 -> 57.3 ms).

## Notifications and doctor (phase 8a; owner: notify, doctor)

**Notifications** (spec UX 5). `DaemonRuntime.create` calls `notify.service.attach_notifier`
(a bad `notifications:` section is a ConfigError at start). `Notifier.on_change` is an
EventBus subscriber: it only classifies (`format.classify`: -> done = finished, -> failed =
failed, -> awaiting_approval or the D43 re-ask note = approval, -> migrating = migrated
except `hours_exceeded`, whose approval follows; cancelled/denied never notify) and
`call_soon`s `_drain`, which reads the job (never inside the commit callback), dedupes
(finished/failed once per job; approval once per job+provider+reason within `dedupe_s`;
migrated once per job within `dedupe_s`), rate-limits (`max_per_minute`, the rest fold into
one "N more job updates" notification) and `put_nowait`s onto a bounded queue; one daemon
worker thread runs the backend with `timeout_s`. Backends: terminal-notifier (`-group
gpu-router.<job>`), else `/usr/bin/osascript` with an `on run argv` script (text is argv,
never AppleScript source), else null. `auto` is OFF in test mode; nothing real is ever sent
under pytest unless `GPU_ROUTER_NOTIFY_REAL=1`. Texts: title `gpu-router`, subtitle `✓ name
finished` / `✗ name failed` / `⏸ name needs your approval` / `↪ name is moving off colab`,
body with time, provider GPU, outputs (`proj/runs/a7f2`), `gpu logs|approve <id>`; sounds
Basso (failed) and Glass (approval) when `sound: true`. Config (all optional):

```yaml
notifications:
  enabled: true
  backend: auto          # auto | terminal-notifier | osascript | off
  events: {finished: true, failed: true, approval: true, migrated: true}
  sound: true
  dedupe_s: 600
  max_per_minute: 6
  timeout_s: 10
```

`gpu notify [status] [--json]` shows the settings and the backend; `gpu notify test` sends
one "gpu-router test notification" from the CLI process (exit 1 + why when the backend is
none). Sent once for real on 2026-09-24 through osascript (exit 0).

**Doctor** (`gpu doctor [--json] [--timeout 20] [--only GROUP|ID ...] [--start] [-v]`,
`gpu doctor --update-catalog [--yes]`, shell `/doctor` with the same flags). Groups: daemon
(running, version vs this CLI, launchd agent + `launchctl print`, state.json freshness),
providers (per enabled provider: CLI + version from its uv tool env's dist-info, else
`--version`; credentials by stat only: Keychain names from secrets.index, ~/.kaggle files +
mode, ADC file + `colab --auth=adc whoami` for the colaboratory scope in a throwaway HOME,
Lightning Keychain/env/file following `providers.lightning.login_source` like
`credentials.resolve` (LIGHTNING_AUTH_TOKEN is not a source); live = the daemon's healthcheck (`kaggle quota`, `colab
sessions`); Modal's exclusion with its quote from providers.yaml `excluded:`, leftover
modal config/secrets; the verify-at-signup lane as one row), storage (huggingface_hub
version, HF_TOKEN whoami, HF_TOKEN_REMOTE), inference (one optional row: which D50 lane
providers have their Keychain names in secrets.index, half-set-up entries, broken catalog
entries, "key rejected" blocks from the daemon's GET /v1/infer/quota; skip + `gpu login
groq` when none), limits (live quota limit / unit / reset anchor
vs providers.yaml -> DriftItems; colab GPUs from `colab new --help`), local (config +
policy/routing/notifications sections + user providers.yaml, data dir 0700 with token/db/
config 0600 (daemon.json and state.json are 0644 by design), disk, git, uv), integration
(`gpu` on PATH, status-line wrapper via `statusline.install.status`, plugin in
installed_plugins.json + enabledPlugins + version, `~/.claude/skills/colab` with the
confirm-guarded move as fix, Codex `[mcp_servers.gpu-router]`, notification backend).
Each row: ok/warn/fail/skip + summary + exact fix command (paths shell-quoted); fix lines
print unwrapped so they copy whole. Exit 1 when any row fails or a check crashed / did not
finish (`Report.unknown`: it verified nothing, D54; warnings exit 0). Rules: the
daemon is never started unless `--start`; providers that are not enabled are not probed;
provider accounts are only touched through the daemon (healthcheck, GET /v1/quota), doctor
itself runs only local quota-free probes; credential files are stat()ed, never opened; one
daemon thread per check, results by the deadline, the rest become "did not finish" rows.
`--update-catalog` writes live `quota.limit` / `quota.reset_anchor` / `quota.unit` into
`<home>/providers.yaml` (diff first, `[y/N]`, no terminal and no `--yes` = exit 2; validated
by loading it; atomic, 0600; the daemon needs a restart). Live on the dev machine (tmp home,
real daemon, 2026-09-24): 1.4 s for 34 checks; kaggle 2.2.4 / colab 0.7.2 found, colab ADC has
the colaboratory scope, kaggle live 30 h/week resetting sat 00:00 UTC = providers.yaml, a
user providers.yaml saying 25 h / fri was flagged and `--update-catalog --yes` fixed it.
Re-run 2026-09-25 (tmp home, real daemon): 1.4 s for 35 checks, 0 fail; lightning login ok
from ~/.lightning/credentials.json (0600), lightning live answered with 15 credits/month =
providers.yaml (pre-fix math: since D49's live fix it reports the account's real limit, 4.99
credits, and doctor flags the drift from providers.yaml's 15, D53); inference row skip (no keys); `gpu notify test` sent through osascript.

## Setup wizard (phase 8b; owner: setup)

`gpu setup [--yes] [--only STEP|ITEM ...] [--dry-run] [--no-check] [--no-smoke] [--again]
[--json]`; bare `gpu` on a terminal offers it first while setup never finished (`setup/firstrun`:
no `<home>/setup.json`, or an interrupted run; never in test mode or with
`GPU_ROUTER_NO_SETUP=1`; "not now" is remembered as `dismissed_at`). Steps and items
(`--only` takes either; an item's last part works when unique):
1 `tools` (`tools.uv|gpu|kaggle|colab|lightning|hf`: detected like doctor; missing ones for
enabled providers installed with `uv tool install` after ONE confirmation; lightning-sdk
pinned to `providers/lightning/sdk.DEFAULT_SDK_VERSION`; uv itself is never installed);
2 `logins` (`login.kaggle`: kaggle.json from ~/.kaggle or ~/Downloads, or a pasted token ->
Keychain `kaggle` / `KAGGLE_API_TOKEN` after one `kaggle -W quota` with ONLY those
credentials and an empty KAGGLE_CONFIG_DIR, rejected = nothing stored; `login.colab`:
doctor's `check_colab_login`, else the exact ADC gcloud command, offered to run attached to
the terminal (browser), never without a human, the ADC file is never read; `login.lightning`:
`providers/lightning/login.run_login` import / browser / paste; `login.hf` and optional
`login.hf_remote`: token files are only stat()ed until the user says yes);
3 `launchd` (doctor's row decides; `daemon.launchd.install` with launchctl through
`SetupEnv.run`; never for a custom GPU_ROUTER_HOME: the label is global);
4 `integration` (`integration.statusline` = `statusline.install.run_install` with the
wizard's confirm, `integration.plugin` = `claude plugin marketplace add <repo>/plugin` +
`claude plugin install gpu-router@gpu-router-local` (or enable / update, from doctor's fix),
`integration.codex` = the docs/codex config.toml block appended as text (every existing
byte kept; the result must parse; backup; refused when the file changed while asking),
`integration.colab_skill` = move to skills-disabled-colab; each shows the change and asks,
default no); 5 `check` (`check.doctor` in-process against the daemon, which the wizard starts
like any command (a dry run never starts it); `check.smoke`: one `setup smoke <provider>`
job per enabled + healthy provider through the daemon (`<home>/setup/smoke/gpu_smoke.py`,
0700: nvidia-smi / CUDA or MPS matmul, `gpu-smoke: {json}` line, outputs/gpu.json, exit 3 =
no GPU), asks first, passed providers are not re-run (`--again`), unfinished ones are
cancelled after 15 min, approval waits are named and left (`manual` + `gpu approve <id>`,
never failed; job ids go into setup.json right after the submit, so a resumed run reattaches
instead of submitting again, D54); `check.summary`: ready providers
and free GPU hrs/month = the ledger's (else providers.yaml's) limit in GPU hours x resets per
month (weekly 30.44/7, monthly 1, daily 30.44), colab (unknown) and the Mac named, never
counted). Every yes/no goes through `base.Ctx.confirm`: `--dry-run` never asks or writes,
a resumed run keeps its "no"s (unless `--only` names the item), `--yes` answers yes except
human-only steps without a terminal (browser sign-ins, pasted tokens: `manual`) and, when an
agent runs setup (`agent.agent_marker()`), except `base.AGENT_GUARDED` (integration.*,
launchd: asked on a terminal, else `manual`, D54), no terminal and no `--yes` = not asked
(`manual` + the command; exit 2; the paste-only logins too, D54). setup.json (0600,
atomic, no secrets) records answers, item outcomes and smoke results; an `--only` run
neither resumes nor ends the recorded run. Exit 0 finished, 1 an item failed, 2 unanswered
(wins over 1), 130 Ctrl-C (resumable). `gpu login` (bare) lists every login from local
facts; `gpu login kaggle [--file|--stdin] [--no-check]`; `gpu login colab [--run]`.

## Paths and config

Data dir `~/Library/Application Support/gpu-router/` (override `GPU_ROUTER_HOME`); full layout
in `paths.py`. Config `config.yaml` (`version: 1`), precedence env > file > defaults
(`GPU_ROUTER_PORT`, `GPU_ROUTER_LOG_LEVEL`, `GPU_ROUTER_TEST_MODE`). Provider facts
(VRAM, session caps, quotas, resets, card_required, verified_at) are data in the packaged
`providers/providers.yaml`, overridable by `<home>/providers.yaml`, never prose. The fake
providers (`fake`, `fake-b`) are `test_only` and registered only when `test_mode` is on. In
test mode the real providers (`local`, `kaggle`, `colab`, ...) are registered only when
`providers.<name>.enabled: true` is set explicitly or `GPU_ROUTER_REAL_PROVIDERS` lists them
(D29). Set that env var only with a provider's own test paths (e.g. `tests/contract/local`):
on a whole-suite run it also registers the provider in the daemon/API/CLI tests that expect
only the fakes.

## Logging and secrets

JSON lines to `logs/daemon.jsonl` (shape and event names in `log.py`); use
`log_event(logger, "job.transition", msg, job_id=..., ...)`, never bare f-string logs for
state changes. Every handler redacts (`secrets.redact`), and the engine's captured job logs
(`jobs/<id>/logs/attempt-<n>.log`) are redacted before they hit disk. Provider-side raw
copies are short-lived (D37): Kaggle caches its final log redacted; Colab's harvested
`job.log` and local's `console.log`/`work/job.log` are deleted 1 h after `logs()` served
them to eof; Colab's CLI history of a session is deleted when it is stopped. Secrets: Keychain service `gpu-router`, via `secrets.get_secret/set_secret` only.
Jobs reference secrets by name (`JobSpec.secrets`); the engine resolves them into
`AttemptContext.secrets` (SecretStr) right before submit.

## Testing strategy

- **Unit** (`tests/unit/<area>/`): pure logic with `FakeClock` and `Store.open_memory()`.
- **Contract** (`tests/contract/`): one suite, parametrized over adapters through
  `ContractTarget`s; the fake always runs; real providers only with `real_provider` marker +
  env opt-in + passing healthcheck.
- **Engine** (`tests/unit/engine/`): Supervisor + FakeAdapter + in-memory store + FakeClock;
  drive time with `clock.advance(); await settle()`. Cover every row of the transition table,
  rate-limit reroute, quota exhaustion, migration with resume checkpoint, cancel, approval.
- **API** (`tests/api/`): `httpx.ASGITransport` against `create_app()` with a real runtime on a
  tmp home; auth, envelope, idempotency, log streaming.
- **Crash** (`tests/crash/`, `-m crash`): spawn `gpu daemon run --foreground --port 0` with
  `GPU_ROUTER_TEST_MODE=1`, submit to the fake with sub-second directives, SIGKILL at a crash
  point (`GPU_ROUTER_CRASH_AT=<name>`), restart, assert the job finishes exactly once.
- Every test dir is a package (`__init__.py`), so shared helpers import as
  `from tests.contract.harness import ...`; `uv run pytest --collect-only -q` must stay clean.
- Fixtures in `tests/conftest.py`: `gpu_home` (autouse tmp `GPU_ROUTER_HOME`), `paths`, `clock`,
  `test_config`, in-memory keyring (autouse). Add area fixtures in your area's `conftest.py`.

## Conventions

- Python 3.12, `from __future__ import annotations`, full type hints, mypy strict clean.
- pydantic v2 for anything crossing a boundary (API, DB json, config); dataclasses inside.
- Enums are `StrEnum`; compare with members, serialize values.
- Line length 100; ruff rules in pyproject. No new runtime dependency without a Decisions entry.
- User-facing text: lowercase-first short sentences, no emoji, says what happened + what next.
- Don't edit a file owned by another group during phase 1; ask the orchestrator for an
  interface change instead. Shared interface files (real code) change only with a note here.

## Phase 1 work split

Each file has exactly one owner. Shared interfaces marked (real) are frozen: implementers may add
private helpers and fix bugs but must not change public signatures without the orchestrator.

**Group A: core data layer.** Owns `models.py`, `statemachine.py`, `errors.py`, `store.py`,
`db/__init__.py`, `db/connection.py`, `db/migrate.py`, `db/migrations/__init__.py`,
`db/migrations/0001_initial.sql`, `statefile.py`, `config.py`, `paths.py`, `ids.py`, `lock.py`,
`log.py`, `secrets.py`, `clock.py`, `__init__.py`, `py.typed`; tests `tests/__init__.py`,
`tests/conftest.py`, `tests/unit/__init__.py`, `tests/unit/core/**` (test_statemachine, test_models, test_schema, test_migrate, test_store,
test_ids, test_config, test_paths, test_clock, test_lock, test_log, test_secrets, test_statefile).

**Group B: adapters.** Owns `protocol.py`, `adapters/__init__.py`, `adapters/base.py`,
`adapters/registry.py`, `adapters/fake.py`, `providers/__init__.py`, `providers/catalog.py`,
`providers/providers.yaml`; tests `tests/unit/adapters/**` (test_fake, test_registry,
test_protocol, test_catalog) and `tests/contract/**` (pre-written: `harness.py`, `conftest.py`,
`test_submit.py`, `test_status.py`, `test_logs.py`, `test_fetch_cancel.py`,
`test_quota_health.py`; extend, don't weaken).

**Group C: engine, daemon, client.** Owns `engine/**`, `daemon/**`, `client.py`, `api.py`,
`policy.py`, `router/__init__.py`, `router/base.py`, `router/simple.py`, `entry.py`,
`__main__.py`; tests `tests/unit/engine/**`, `tests/unit/router/**`, `tests/unit/test_entry.py`,
`tests/api/**`, `tests/crash/**`.

Dependencies: B and C build against A's real models/errors/statemachine/clock/paths now; C uses
`Store.open_memory()` as soon as A lands it (until then, C writes engine tests and code against
the documented Store API). C drives the FakeAdapter only through `adapters/base.Adapter`.
Integration order: A store -> B fake + contract green -> C engine tests -> C API -> crash tests.

Phase 1 is done when: all unit/contract/API tests pass; `-m crash` passes for every named
crash point in `engine/crashpoints.py`; `GPU_ROUTER_TEST_MODE=1 gpu daemon run --foreground`
accepts a fake job over HTTP and it reaches `done` with outputs in `runs/<id4>/`; ruff and
mypy are clean. All met at integration (2026-09-23): 651 passed, `-m crash` 10 passed
(~28 s), ruff check/format and mypy strict clean, foreground smoke test done.

## Decisions log

- **D1** Spec decisions adopted as final: name gpu-router / `gpu`; job spec = gpu.yaml + flags
  (flags win); metrics = helper + stdout fallback; status line adds max 2 rows while active;
  Modal card question deferred to phase 7.
- **D2** Python 3.12 managed by uv; `uv_build` backend; src layout; the remote side (`runner/`)
  stays Python 3.8-compatible and stdlib-only because provider images vary.
- **D3** Provider SDKs are NOT project dependencies. Adapters shell out to provider CLIs installed
  in isolated tool envs via `uv tool install <pkg>` (kaggle, google-colab-cli, lightning-sdk,
  modal), through `providers/cli.py:CommandRunner`. Keeps our env small and conflict-free.
- **D4** One daemon, SQLite in WAL mode, synchronous Store on the event-loop thread; blocking
  adapter calls in a bounded thread pool (per-provider concurrency limit).
- **D5** Time is epoch-seconds floats everywhere internally; ISO-8601 `Z` only at the JSON edge.
- **D6** Daemon on 127.0.0.1:47291 with a 0600 bearer-token file; port 0 in tests/dev, clients
  read `daemon.json`.
- **D7** Adapter methods take a `RemoteRef` (remote_id + url + adapter-private meta) rather than a
  bare remote id, so adapters that need extra handles (Colab session name, Kaggle slug) stay
  stateless (invariant 10). `ref.remote_id` is the "remote id" of the spec's contract table.
- **D8** Log capture is poll-based in phase 1 (`logs(follow=False, since=cursor)` each tick):
  crash-safe, one code path for every provider. `live_logs` capability is for UI tailing later.
- **D9** The fake provider keeps its "remote" under `<home>/fake/` and derives state from the
  injected clock and submit time, so it survives daemon SIGKILL and is deterministic under
  FakeClock.
- **D10** Approval approves the job, not a specific provider; after approval the driver re-routes
  and places without asking again (phase 5 may re-ask for the >50%-of-quota rule; it does, D43).
- **D11** Packaged provider facts live in `src/gpu_router/providers/providers.yaml`, loaded with
  `importlib.resources`; the user file overrides per key.
- **D12** (phase-1 integration) `errors.Conflict` (409, code `conflict`) is the generic
  state-conflict error; Idempotency-Key reuse with a different spec raises it. The client
  rebuilds any `conflict` envelope (incl. StaleState) as `Conflict`.
- **D13** (phase-1 integration) Store listeners may define an optional
  `provider_changed(provider)` hook, called after `upsert_provider_state` commits. `EventBus`
  implements it by notifying subscribers with `job_id=""` and no events, so provider health
  changes reach state.json without waiting for a job event.
- **D14** (phase-1 integration) Fake-adapter semantics fixed by group B: `rate_limit_n` /
  `unavailable_n` count per job; `set_health` also makes submit/status raise; a retried submit
  with a known key returns the existing run even during a simulated outage; `error_kind` on an
  attempt is the exception class name (e.g. `RateLimited`).
- **D15** (phase 2, packaging) The daemon builds bundles at submit, before `create_job`
  (`EngineDeps.bundler`, optional so engine unit tests need no project on disk). Bundles are
  content-addressed in `<home>/bundles/<sha256>.tar.gz` and linked into `jobs/<id>/`; resubmitting
  an unchanged project reuses the cached archive. `Paths` is unchanged: `bundles_dir(paths)` lives
  in `packaging/bundle.py`.
- **D16** (phase 2, packaging) Deps are resolved on the Mac (tomllib) and recorded in the
  manifest as a package list, so the py3.8 runner never parses TOML and never builds the user's
  project as a package. `[dependency-groups]`/dev deps are not installed remotely.
- **D17** (phase 2, packaging) VRAM/hours estimates are recorded in `manifest.json["estimate"]`
  and logged (`job.bundle`), but NOT written into the stored spec: the phase-1 router filters only
  on explicit `spec.vram_gb`, so a wrong guess can never block a job. Phase 5's scoring router
  reads the estimate. No new `Reason` code was added (schema 0001 is frozen), so bundle facts are
  in the manifest and the daemon log, not in job events.
- **D18** (phase 2, runner) Runner heartbeats use protocol type `heartbeat`, which
  `protocol.parse_line` ignores (unknown types are skipped by design); checkpoint sync targets a
  local dir with `file://` URIs until phase 5 adds HF Hub; ckpt seq start is passed in by the
  adapter so seq stays monotonic per job across attempts.
- **D19** (phase 2, CLI) CLI reference, JSON shapes and exit codes are in `docs/cli.md` and are
  additive-only like API v1. `--json` prints one document on stdout (NDJSON for `logs`), errors
  as `{"error": {...}}` on stdout with the daemon's raw code (e.g. `invalid_transition`), notes on
  stderr. `gpu run` waits by default (poll loop: job detail + events + per-attempt log offsets,
  terminal events printed after the last log lines); with `--json` it detaches by default.
  Ctrl-C detaches (exit 130), never cancels.
- **D20** (phase 2, CLI) `gpu daemon ...` stays on the argparse path in `entry.py` (no typer
  import); phase 2 added `start` and `install-launchd` (alias of `install`, `--print` shows the
  plist). The Typer `daemon` command only forwards there. Auto-start (`daemon/spawn.py`) uses
  `launchctl kickstart` only when the agent is installed AND `GPU_ROUTER_HOME` is unset (the
  agent serves the default home); otherwise a detached `python -m gpu_router daemon run`
  inheriting the environment. `GPU_ROUTER_NO_AUTOSTART=1` disables it.
- **D21** (phase 2, CLI) gpu.yaml lives at the project root (nearest `.git` ancestor, else cwd).
  Flags win per field; `--env` merges per key; CLI script args replace file `args`; a first
  positional that starts with `-` is a script arg for gpu.yaml's script. Unknown gpu.yaml keys
  are errors (difflib suggestion). The CLI never builds bundles for submit (D15: the daemon
  does); `run --dry-run` previews one through `packaging.build_bundle`.
- **D22** (phase-2 integration) The fake provider simulates a run; it never executes the
  bundle, so fake-job outputs are the fake's own files (`model.txt`, `result.json`). Real
  execution of the bundle is proven by running `jobs/<id>/bundle/gpu_runner/bootstrap.py`
  directly (`tests/cli/test_e2e_subprocess.py`); the phase-3 local adapter is the first
  provider that runs it for real. Outputs dir stays `runs/<id[:4]>/` (invariant 22, spec
  mockup `./runs/a7f2`). `run --dry-run` bundle warnings are `bundle.warnings` only (it
  already contains the deps warnings). Open for later phases: bundle warnings after a real
  submit reach only the manifest and daemon log (needs an additive `JobDetail` field); an
  Idempotency-Key replay rebuilds the bundle first, so a replay after the project dir is
  deleted returns 400 instead of the existing job; `gpu run --wait` exits 3 if the daemon
  restarts mid-wait (no reconnect, but the error names the job since D24); phase-3 adapters
  must pass `--ckpt-seq-start` and a checkpoint sync target to bootstrap, and should treat
  exit 90 / `install_failed` as an environment failure (reroute), not a script failure (D26). `gpu status --line` measured at integration (no
  `statusline/fast.py` yet, so it is the entry.py dispatch cost): median 18.6 ms, p95 20.8 ms
  over 50 runs vs 17.4 ms for a bare `python -c pass`.
- **D23** (phase-2 review, supersedes D21's "flags anywhere") `gpu run` options go before the
  script. After it, only `--json --wait --detach --dry-run --help --vram --hours
  --provider/-p` are still gpu's (the spec's `gpu run train.py --json` and the common
  `--vram`/`-p` overrides keep working; a `-p` value that is no provider fails loudly);
  everything else, other short flags and `--name/--gpu/--env/--project` included, goes to
  the script, with a stderr note when a gpu option name went there. `--` ends gpu parsing.
  Implemented as `cli/app.split_run_argv` in a `TyperCommand.parse_args` override, so click
  only ever sees gpu's part. Reason: `-wd 0.01`, `-e 10`, `--name exp1`, `--gpu 0` are common
  training args and were silently eaten. A pure `allow_interspersed_args=False` was rejected
  because it would send `--json` after the script to the script.
- **D24** (phase-2 review) typer >= 0.27 vendors click (`typer._click`); its exceptions do not
  subclass the `click` package's, so `cli/app.py` catches both (`CLICK_USAGE_ERRORS`,
  `CLICK_ABORTS`, `CLICK_EXITS`). Usage errors: `gpu: <message>` + hint, exit 2; with `--json`
  the `invalid_request` envelope (`detail.usage: true`). `guarded()` turns any other exception
  into code `internal`, exit 1, so `--json` stdout always parses. Submit waits
  `client.SUBMIT_TIMEOUT_S` (300 s, the daemon bundles in-request) and only retries a
  connection that never reached the daemon; otherwise `errors.SubmitUncertain` (code
  `submit_uncertain`, exit 1, `detail.idempotency_key`). Timeouts no longer say the daemon is
  down, and hints say `gpu daemon start`. After `run` has submitted, every error envelope
  carries `detail.job_id/short_id/state`; Ctrl-C in `run --json --wait` prints `{"job",
  "detached": true}`; `--json --wait` notes state changes (approval) on stderr. Unknown
  providers are exit 4 before submit (CLI-side check against `GET /v1/providers`; the daemon
  still accepts the spec, so other clients see `no_provider_fits`).
- **D25** (phase-2 review) Bundles keep shipping untracked-not-ignored files (a script an
  agent just wrote must run before it is committed; the spec's "git-tracked only" line is not
  in its Decisions table, and this contract chose `-co` in phase 2), but name them in a
  warning and `manifest.files.untracked`. Invariant 12 is enforced by a much broader deny
  list (names, credential dirs anywhere on the path, symlink targets) and by refusing
  symlinks that resolve outside the project. The walk fallback runs only when there is no
  `.git` in the project or a parent; a failing git inside a repo is a BundleError (fail
  closed: the walk ignores .gitignore).
- **D26** (phase-2 review, runner) Checkpoints are never shared by hard link (in-place saves
  truncate every link); the checkpoint syncer publishes only files unchanged for 2 s
  (postponing up to 120 s), re-fingerprints after archiving and drops a changed archive
  before `ckpt_begin`, skips the final sync after a non-zero exit when files changed within
  10 s (a kill mid-save), dereferences symlinks, uses gzip level 1, runs periodic syncs on
  their own thread and keeps the newest 3 archives. Dependency install failure exits 90 with
  an `install_failed` line (ignored by `protocol.parse_line`). A reused workdir is scoped
  per run (EXIT removed, log rotated, bundle re-extracted on a sha change, checkpoint/output
  dirs emptied when stamped by another job id). The entrypoint runs in its own process group.
- **D27** (phase-2 review) `gpu secrets set NAME [--stdin] | list | rm NAME` exists (Keychain
  via `gpu_router.secrets`, names only in output); every hint that says `gpu secrets set`
  now points at a real command. `gpu daemon stop/install/uninstall` take `--json`.
- **D28** (phase 3, local provider) Local runs use a **uv-managed venv per deps key** by
  default: `<home>/providers/local/venvs/<sha16 of interpreter, installer, base_packages,
  deps kind, requirements bytes / pyproject packages>/`, created from the daemon's base
  interpreter and filled once under an flock (marked ready only after a clean install), so
  later runs with the same deps start in ~50 ms. `providers.local.env: system` +
  `python: <path>` runs a user's own interpreter as-is (nothing installed);
  `base_packages` installs e.g. torch/numpy into every venv; `uv: false` falls back to
  stdlib venv + pip. `env`/`python` can be overridden per job in `provider_options.local`.
  The remote is `<home>/local/<attempt key>/` (remote id = attempt key, so A4 and
  lookup_by_key are directory checks). A detached, stdlib-only launcher (fork + setsid,
  re-parented to launchd) unpacks the archive, prepares the env and execs the bundle's own
  bootstrap with `--skip-install`, keeping an flock on `alive.lock` that the adapter took
  before `run.json` was written: liveness = lock held, which survives daemon restarts and
  is immune to pid reuse. EXIT 90 and signal deaths (HUP/INT/KILL/TERM) we did not cause
  map to LOST (reroute/migrate, D22); other non-zero exits to FAILED. `GPU_DATA_DIR` =
  `<home>/providers/local/data`; `PYTORCH_ENABLE_MPS_FALLBACK=1` unless the job sets it.
  Details and live numbers: `providers/local/NOTES.md`.
- **D29** (phase-3 integration) `registry.is_enabled`: in test mode a non-`test_only`
  provider is registered only when `providers.<name>.enabled` is explicitly true or
  `GPU_ROUTER_REAL_PROVIDERS` (comma-separated, `registry.real_providers_opted_in()`) lists
  it; an explicit `enabled: false` still wins. Before this, the phase-3 adapters (on by
  default) registered in every test-mode daemon: 9 shared tests failed and test daemons
  healthchecked the real Kaggle/Colab CLIs (invariant 20). The adapters keep their own
  test-mode guards on top. `tests/contract/conftest.py` no longer parametrizes providers
  that have their own contract suite (`OWN_SUITES`: local, kaggle, colab), so opting one in
  no longer yields "no contract builder" skips.
- **D30** (phase 3, kaggle) One **private script kernel per attempt**, id
  `<user>/gpu-router-<job_id>-<n>`, title `gpu-router <job_id> <n>` (slugify(title) == slug),
  so the attempt key is the lookup key and a retried submit never pushes a second version
  (= a second run); submit checks a local marker, then `kernels status`, then (GPU) a live
  quota pre-check before `kaggle -W kernels push -t <12h - 5 min>`. The bundle rides base64
  inside run.py (push uploads only the code file; limit `max_embed_mb`, default 10); run.py
  runs bootstrap with outputs in `/kaggle/working/outputs` (the only files fetched) and
  checkpoint sync in `/kaggle/working/.gpu-router/`. The runner's `::gpu:: exit` line in the
  post-run log is the exit code (90 -> lost); no exit line -> lost (time limit / quota /
  Kaggle's message). Logs only after the run (live `-f` SSE tail verified, left for phase
  5). Cancel = `kernels delete` while queued/running + local tombstone (the API cancels
  only by a session id no response exposes); verified live to stop the GPU session at once,
  so `cancel_confirms=True`.
  Secrets are refused (InvalidJob) and `resume=False` until phase 5. Credentials: Keychain
  `kaggle` (kaggle.json doc) or `KAGGLE_API_TOKEN` via env only, else the CLI's own files.
  A test-mode adapter never runs the real CLI unless `GPU_ROUTER_REAL_PROVIDERS` lists
  kaggle (invariant 20). Details and live numbers: `providers/kaggle/NOTES.md`.
- **D31** (phase 3, colab) **One fresh Colab session per attempt**, named `gr-<job>-<n>`
  (the attempt key with `gpu` -> `gr`), so the name is the remote id and `lookup_by_key` is
  a local record check. Every CLI call is `colab --auth=adc --config
  <home>/providers/colab/colab-cli/sessions.json`: a session file only gpu-router uses, so
  another tool's (e.g. an existing Colab batch tool) or a human's sessions are never visible as
  ours and never stopped. Submit =
  `new --gpu T4` (GPU checked against the CLI's list AND the catalog; a typo would silently
  become A100) -> `exec prepare` (nvidia-smi must show a GPU) -> upload bundle (+ resume
  archive, + secrets as a 0600 file that launch reads and deletes: never argv, never exec
  code, which the CLI records in its history) -> `exec launch` (detached bootstrap via
  `bash -c ... ; echo $? > RC`; a long blocking exec loses its websocket). A per-session
  JSON record is written before every remote step, so a crash mid-submit is found again and
  cleaned up (stale `setup` -> LOST + stop, or LAUNCHED if the runner did start). Status =
  one short `exec poll`; on the first poll that sees the exit, the adapter packs and
  downloads `job.log` (and outputs of a successful run, up to 64 MB) and **stops the
  session inside status()** (the contract has no release call; a finished Colab session
  keeps the GPU and the account's free quota until stopped); larger outputs keep the
  session until `fetch()`; later logs/fetch/cancel calls, a janitor thread (D36) and a
  reaper at the next submit stop anything left. 400 "Backend
  rejected accelerator" -> QuotaExhausted (resets_at = now + 24 h); 412 -> Unavailable;
  scope/ADC problems -> AuthRequired; exit 90 / 137 -> LOST; session "lost (404/401)" ->
  LOST. Log cursor = `"<lines>:<byte offset>"` into job.log; lines over 64 KiB are cut the
  same way on the VM and in the harvested copy. Checkpoints up to 64 MB are mirrored to the
  Mac so a new VM can resume (phase 5's HF Hub replaces this). Quota is an estimate (GPU
  time in the last 24 h + last refusal). A test-mode adapter never runs the real CLI unless
  `GPU_ROUTER_REAL_PROVIDERS` lists colab or the `cli` setting points at the simulator
  (invariant 20). Details and live numbers: `providers/colab/NOTES.md`.
- **D32** (phase-3 integration) The first-fit router picks, among a provider's fitting
  offers, the least per-GPU VRAM and then the MOST GPUs (was: fewest). Only Kaggle has a
  tie today (P100 x1 vs T4 x2, both 16 GB): `gpu run -p kaggle` now gets 2xT4 (the
  shape proven live) instead of the P100, which was never run live and may not be usable
  from Kaggle's torch 2.10+cu128 image (Pascal sm_60 support in cu128 wheels is doubtful;
  unverified). Same free quota either way.
  `--gpu P100` still selects it. Phase 5's scoring router supersedes this.
- **D33** (phase-3 integration) A run that ends before any poll sees it `running` (Kaggle
  polls every 60 s; the live gpucheck finished inside one interval) and whose adapter
  reports no `started_at` is stamped with the attempt's `submitted_at` as its start, on
  both the attempt and the job, in the done/failed transition (`JobDriver._run_start`,
  `_start_stamp`). Without it `gpu status` showed "took -", `gpu jobs` kept counting from
  creation, and the phase-5 usage ledger (attempts with `started_at`) would skip the run.
  Charging from submit over-counts queue time, the safe side for a quota ledger; the done
  message then says "within <d> of submit". Kaggle logs are also trimmed after the
  runner's last `::gpu:: exit` line (`kaggle.parse.trim_post_run`): Kaggle appends its own
  nbconvert output (`[NbConvertApp] ...`, SyntaxWarnings) after the job ends.
- **D34** (phase-3 review) The engine hands every placement the job's latest checkpoint,
  whichever provider wrote it. A checkpoint an adapter cannot read is never InvalidJob
  (that excludes the provider for the rest of the job): local now starts fresh with a
  note as the first log line and `resume: unavailable` in run.json, like Colab and
  Kaggle already did, and resolves a Colab VM path
  (`file:///content/gr/<session>/ckpt-sync/<name>`) to the Colab adapter's Mac mirror
  `<home>/providers/<colab>/runs/<session>/ckpt/<name>` when it exists. Also: CLI error
  classification never matches a bare "429" (kaggle and colab): the searched text
  carries our kernel/session names and about 1 hex job id in 400 contains "429", which
  turned a missing kernel into RateLimited forever (definitive on submit, so the job
  could never run on Kaggle) and a crashed `colab new` into a definitive rejection
  without a stop. Names are blanked first, then real signatures ("429 Client Error",
  "Too Many Requests", RESOURCE_EXHAUSTED, ...) are matched; Kaggle checks the
  missing-kernel markers before the rate markers.
- **D35** (phase-3 review) A provider CLI call can outlive the daemon (colab runs every
  call with `start_new_session=True`; a kaggle push child survives a plain SIGKILL), so
  "not found" right after a restart is not proof that nothing will exist. Colab records
  the `colab new` pid in the run record (`ColabCli.run(on_spawn=...)`); a restarted
  daemon keeps a `creating` attempt PENDING while that pid is alive and its `ps` command
  line names the session (no pid: NEW_TIMEOUT_S + 30 s after `created_at`), kills a
  verified orphan past STALE_START_S (600 s), and then always runs the idempotent
  `colab stop -s NAME` instead of setting `stopped` on a "not found"; a retried submit of
  such a record raises Unavailable (ambiguous) rather than racing a second `new`. Kaggle
  writes `submits/<key>.pushing` right before `kernels push` and removes it when the call
  returns; while one younger than T_PUSH + 60 s exists, a NotFound from `kernels status`
  makes `lookup_by_key` and `submit` raise Unavailable ("may still be uploading") instead
  of returning None / pushing a second version (invariant 6).
- **D36** (phase-3 review) Colab teardown no longer depends on the next Colab submit.
  `logs()` and `fetch()` settle whenever the record is final and not stopped, whatever
  the cached flags say (logs() computes its chunks and settles BEFORE yielding: the
  engine stops iterating after max_lines); every harvest step leaves STOP_RESERVE_S
  (12 s) of the call's budget for `colab stop`, a harvest try is counted only when it
  really starts, and the final checkpoint of a success is not mirrored. What still is
  not stopped is retried by a per-adapter **janitor thread** (daemon thread, every
  JANITOR_INTERVAL_S = 120 s; started by any call that leaves such a record, by logs()
  when a raw log starts its purge clock, and by the healthcheck the daemon runs at start;
  exits when nothing is left; gives up 13 h after the run ended; `close()` ends it). It
  only stops sessions whose runs are over, so invariant 11 holds. `cancel()` of a
  launched run first polls once: a runner that already exited is recorded and settled
  (log, outputs, stop) so the engine's outputs_kept path works; a success whose outputs
  are not pulled yet keeps its session for fetch.
- **D37** (phase-3 review) Raw job output and CLI side files stay private and short-lived.
  Colab CLI calls run with `HOME=<home>/providers/colab/cli-home` (0700) and
  `CLOUDSDK_CONFIG` = the real gcloud dir (ADC verified live with `colab sessions`), so the
  CLI's `colab.log` (proxy tokens in URLs) and `history/<session>.jsonl` (exec output,
  incl. base64 live log reads no redaction can see) no longer land as 0644 files in
  `~/.config/colab-cli/`; a stopped session's history is deleted, `colab.log` is
  truncated past 5 MB. (Files already written under the real `~/.config/colab-cli/` by
  earlier runs are left for the user to delete.) Leftover `tmp/.secrets-*.json` and exec
  scripts are swept at adapter start and every submit. Colab's harvested `job.log` and
  local's `console.log` + `work/job.log` are deleted RAW_LOG_GRACE_S (1 h) after `logs()`
  first served them to eof (the engine has its redacted copy; later reads return eof
  with no lines, so A7 holds); local dead run dirs go RETENTION_S (7 d) after that (sweep
  at submit/healthcheck; the served marker lives in adapter scratch so logs() never
  changes a run dir, A6). Kaggle's cached final log is redacted on write, line for line.
  Not done: redacting Colab/local raw copies in place (their cursors are byte offsets).
- **D38** (phase-3 review) Local jobs get an allowlist of the daemon's environment (PATH,
  HOME, user, shell, TERM, TMPDIR, TZ, LANG/LC_*, CA bundles, proxies, HF/torch/XDG cache
  dirs, OMP_NUM_THREADS, UV_*/PIP_*/PYTORCH_*), minus secret-looking names, values with
  URL credentials and token-shaped values; the job's env and its listed secrets are added
  on top. Before, an auto-started daemon (D20) passed the shell's whole env
  (KAGGLE_KEY, AWS_*, OPENAI_API_KEY ...) to every local job, unregistered for redaction.
- **D39** (phase-3 review) New specs get a stricter secret-env rule than stored ones:
  `models.secret_env_problem` refuses names matching `SECRET_ENV_NAME_STRICT` (the old
  rule plus `*_KEY`, `*_PASS`, webhooks, DSNs, cookies and credential-bearing URL names
  like DATABASE_URL) and values `secrets.redact` would change, in `Supervisor.submit` and
  `dry_route` (400 invalid_spec before bundling) and in `jobspec.build_spec` (with the
  gpu.yaml line or `--env`). The JobSpec validator keeps the original `SECRET_ENV_NAME`,
  because Store and every API view re-validate stored specs: widening it would make jobs
  stored earlier unloadable. env is published verbatim into Kaggle kernel source (kept
  in version history) and Colab exec code.
- **D40** (phase 5b: checkpoint handoff + data movement) **Storage = HF Storage Buckets**
  (private, mutable, no history; docs/notes/hf-hub.md) for remote runs and a **local
  directory** `<home>/storage/` for runs on this Mac; one layout for both, implemented once
  in `runner/storage.py` (py3.8, ships in every bundle as `gpu_runner/storage.py`, imported
  by the daemon through `checkpoint/storage.py`; mypy `follow_imports=skip` for
  `gpu_router.runner.*`). Checkpoints are raw files in `jobs/<job>/ckpt-NNNN/` (Xet dedup),
  manifest then `latest.json` last; `owner.json` (written by the engine before every
  submit) stops an older runner that is still alive; the runner keeps 3. The engine
  (`EngineDeps.checkpoints` = `CheckpointHub`, built by `DaemonRuntime`, None in engine unit
  tests = phase-3 behaviour) adds per attempt: env `GPU_STORAGE`, `GPU_RESUME_URI`
  (copying the checkpoint between backends when the job crosses between this Mac and a
  remote provider), `GPU_DATA`, push/poll cadences, and the job secret `GPU_STORAGE_TOKEN`
  (Keychain `HF_TOKEN_REMOTE` only since D44, never the admin `HF_TOKEN`; bootstrap pops it before the job, pip or
  any child runs). Adapters pass env + secrets through their existing channels (local env,
  colab `.secrets.json`); **Kaggle gets a secrets channel**: a private dataset
  `<user>/gpu-router-secrets` attached via `dataset_sources`, re-versioned (old versions
  deleted) only when the values' HMAC changes, so job secrets now work on Kaggle
  (`providers.kaggle.secrets_dataset: false` restores the refusal); Kaggle `resume=True`;
  Kaggle `logs()` serves the runner's `log-tail.json` while the kernel runs (`t<n>:<hash>`
  cursors, aligned onto the final log on the runner's hello line). **No token** = remote
  runs get no storage (phase-3 behaviour) and a one-time `storage_unavailable` note with
  the `gpu login hf` hint; `backend: off|local|hf|auto` in config `checkpoint:`; test mode
  never reads the Keychain for HF unless `GPU_ROUTER_REAL_PROVIDERS` lists `hf`.
  **Handoff**: (a) before routing a migrating job the engine records a newer checkpoint
  that only storage knew (note `checkpoint_found`); (b) at `session_deadline -
  handoff_margin_min` (30) or when the latest quota snapshot says the free GPU hours end
  within the margin, the engine writes `control.json` (action `handoff`, `wait_s` =
  `handoff_wait_min` 10 min, note `handoff_requested`); the runner flags
  `gpu.checkpoint_requested()`, waits for a settled new save, syncs, answers
  `control-ack.json` and publishes nothing more; the engine records the acked checkpoint,
  marks a quota-driven provider exhausted and moves running -> migrating (reason
  `handoff`), whose step cancels the attempt and resumes elsewhere; no answer within
  wait + slack = note `handoff_skipped`, the job runs to the session end. **max_attempts**
  now counts only attempts without progress (no checkpoint beyond the one they resumed
  from); max_placements still bounds all. **Datasets**: local runs get a symlink; remote
  runs get `datasets/<content sha256>/` uploaded once (manifest last), cached in
  `data_cache` (Store get/put/touch/delete, no schema change), reused on later runs
  (notes `data_uploaded` / `data_reused`); a remote run without HF storage excludes that
  provider for the job. New Reason codes (notes): handoff_requested, handoff_skipped,
  checkpoint_found, checkpoint_copied, storage_unavailable, data_uploaded, data_reused.
  `gpu login hf` stores tokens (stdin/getpass/`--import`, never argv). Not verified live:
  no HF token existed on the dev machine; `tests/unit/checkpoint/test_live_hf.py` is the
  opt-in check.
- **D41** (phase 5a: router, quota ledger, approval policy; D40 is the phase-5 hub work)
  **Router**: `ScoringRouter` is the daemon default; `--provider` pins still go through
  `SimpleRouter` (reason "pinned with --provider", ledger quota facts added so the policy
  sees them). Job hours = spec, else the bundle estimate (`RoutingContext.estimate`, read
  from `jobs/<id>/bundle/manifest.json`; POST /v1/route computes it from the project in a
  thread, 10 s cap); explicit VRAM is a hard filter, a heuristic VRAM only scores (D17
  holds). SESSION rejects only jobs that cannot hand off (handoff = checkpoint_interval_min
  > 0, not interactive, `capabilities.resume`). QUOTA (ledger: nothing left, or less than
  all hours without handoff / `handoff_min_hours` 0.5 with it) is temporary: the job waits
  for the reset. New reject code `reserved`: the local Mac (catalog kind `local`) runs only
  smoke tests and is a last resort only when it is the sole registered provider (a busy
  or unsuitable Mac never makes a cloud job wait, and a cloud job that fits nowhere fails
  loudly instead of running for hours on MPS); `routing.big_vram_providers` (modal: 16)
  only for jobs needing more (explicit VRAM, else the estimate), but a last resort when
  nothing else can take the job. Smoke = `JobSpec.smoke` (new additive field; gpu.yaml
  `smoke:`, `--smoke` on run/route) or an explicit `hours` at most `smoke_max_minutes` (5)
  with no GPU type, no VRAM ask, not interactive. Scores (provider roles are data in config
  `routing:`): save_for_long_jobs [kaggle] +30 over long_job_hours (4) else -30 "saved for
  jobs over 4h", but +10 "use it or lose it" when more quota is left than hours to its
  reset; short_job_providers [colab] +20 short/interactive (unknown length = short), -10
  long; big-VRAM provider +25; smoke on the Mac +100; heuristic VRAM above every offer -40;
  reset soonest up to +15 over the last 7 days; over half of what is left -10; priority
  tie-break. The chosen reason names a ruled-out better fit, else the runner-up's handicap
  when it outweighs the chosen one's best point. `RouteDecision` gains `hours`,
  `hours_source`, `smoke`; `Candidate` gains `quota_left`, `quota_unit`, `quota_share`,
  `resets_at` (additive).
  **Ledger** (`quota/`): views are computed from the store, never a provider call on the
  routing path: the latest live reading if younger than `ttl_s` (1800) and before its own
  resets_at; an older one + our GPU time since; one from before its reset = our GPU time
  since that reset; else history in the catalog window (fixed when `reset_anchor` parses,
  e.g. Kaggle `sat 00:00 UTC`; rolling 7/30/1 d with no anchor, so Lightning/Modal with an
  unknown reset day are rolling; rolling 24 h for `unknown` = Colab; unlimited for `none` =
  local). Adapter estimates (Colab's) are ignored in favour of the ledger's own;
  credits/usd need catalog option `quota_per_gpu_hour`, else the limit is reported as None
  rather than a fake 0 used; `exhausted_until` forces 0 left. Routing, provider views and
  state.json all use it (`engine/context.quota_views`). `QuotaService` refreshes
  live-capable providers every `refresh_s` (1800) in the background (never the unlimited
  Mac), skips disabled and logged-out ones, waits `retry_failed_s` (300) after a failure
  (each reading is one quota_snapshots row, ~48/day for Kaggle; the Store has no prune
  yet); GET /v1/quota waits at most
  `wait_s` (8) for stale readings, `?refresh=true` forces.
  **Policy**: `RulesPolicy` is the default, from config.yaml `policy:` with its own
  `version: 1` (`routing:`/`policy:` were free-form, unused dicts, so typing them needs no
  config migration; CONFIG_VERSION stays 1; bad sections are a ConfigError at daemon
  start). Two audiences: agent jobs (source agent, the phase-6 MCP server) get the spec
  defaults (auto up to 1 h, modal asks, over 50% of a provider's quota left asks, unknown
  runtime asks); user jobs (cli, shell, api) have no hours limit (typing `gpu run --hours 6`
  is the approval) but modal and the 50% rule still ask, unknown runtime runs; `local` is
  exempt for both. Hours are wall hours of one session (how free tiers meter). GET/PUT
  /v1/policy; PUT persists via `config.set_config_section` and swaps the rules in memory
  (hand edits to config.yaml need a daemon restart); `gpu policy [show|set KEY VALUE|
  reset]`. Open for the engine owner: an approved job is placed without asking the policy
  again (`step_awaiting_approval` passes ask_policy=False), so the "over 50% always asks"
  rule cannot re-ask when a re-route after approval lands on another provider.
  `RulesPolicy` already scopes an approval to `job.provider`, but the driver would also
  need a same-state re-ask path (awaiting_approval -> awaiting_approval is not a legal
  transition: update the approval fields in place + an approval_required note; done in D43).
  Policy edits do not re-evaluate jobs already waiting for approval.
- **D42** (phase 4: interactive shell) Bare `gpu` opens `shell.app.GpuShell` only when
  stdin AND stdout are terminals (`entry._is_tty`); a pipe, CI or an agent gets `gpu
  --help`, never a TUI and never a daemon start. The shell is a thin client (invariant 2):
  one feed thread polls `GET /v1/status` every 1 s (auto-start via `daemon.spawn.connect`
  on the first try only, then plain reconnects every 3 s: a failed start is not retried in
  a loop, but every command the user types tries it again like the CLI), `GET /v1/quota`
  every 60 s on its own thread. Metric history for the panel sparkline and /watch comes
  from the job's captured log read with `logs?protocol=true` (additive `protocol=` kwarg on
  `GpuClient.logs`), parsed like engine/capture.py (helper `::gpu::` metrics win; stdout
  fallback only until the first helper metric, whose arrival drops the fallback's guesses);
  no metrics endpoint was added. Commands reuse `cli.app` helpers (`_build`,
  `split_run_argv`, `_check_provider`, `_print_submitted`, `_action_result`, `_wait_fetch`,
  `_copy_outputs`, `_with_quota`, `_split_events`, `_print_final`, `_print_policy`) and
  `cli.render` through `Collector`, a rich Console whose print() feeds the transcript;
  `shellify` rewrites `gpu approve a7f2` hints to `/approve a7f2`, and every command also
  answers without the slash or with a leading `gpu `. /run streams logs by default like
  `gpu run` (`-d` detaches; esc/ctrl+c detach, never cancel). /login = live healthcheck +
  the provider's login steps (no interactive login from the shell); /doctor = daemon
  health, CLIs on PATH, parallel healthchecks, live limits vs providers.yaml; /policy =
  `gpu policy` (show/set/reset over /v1/policy) + jobs waiting; /config = effective
  config, `edit` suspends the app for $EDITOR and validates on return. Visual language:
  colour only on state icons/words (green/yellow/red, rich names mapped by the shell's
  ANSI theme), labels dim, direction as ↓↑→ glyphs, icons padded to 2 cells (⚡ is double
  width), neutral grey chrome (theme primary is grey, not blue), 0 ms motion. The
  transcript is height:auto up to the room left, so the prompt sits under the panel on an
  empty screen (mockup); rule + footer are docked. CLI list tables (4+ columns, e.g.
  `render.jobs_table`) are fitted to the transcript width by `commands.fit_table` (one row
  per job: notes cut with an ellipsis, capped name columns shrink first) instead of rich
  folding the notes word by word at 80 columns. Screenshots: SVG from Textual, PNG via
  `uv run --with resvg-py` (Menlo), no browser. Verified in a real pty: bare `gpu`
  auto-started a private daemon (tmp home, port 0), drew the panel, ran /jobs, ctrl+d
  exited 0.
- **D43** (phases 4-5 integration, 2026-09-24) (a) **Re-ask after an approval** (closes D41's
  open item and D10's "may re-ask"): `step_awaiting_approval` re-routes an approved job and
  asks the policy again (`_apply_decision(after_approval=True)`), honouring only decisions
  with the new additive `ApprovalDecision.always`, which `RulesPolicy` sets on the
  quota_share rule (it already exempts an approval given for the same provider). Every
  other rule, and policies that never set it (a test "always ask" policy), keep D10: the
  approval covers the job. awaiting_approval -> awaiting_approval is not legal, so the
  re-ask resets approval_reason/approved_at/approved_by/provider/gpu/route_reason/message
  in place with an `approval_required` note (`detail.reask`), and the approval timeout
  counts from the latest approval_required event (`_approval_asked_at`). Migrations already
  asked the policy and are unchanged. (b) `Store.record_quota_snapshot` prunes each
  provider to the newest `QUOTA_SNAPSHOTS_KEEP` (500) rows; only the latest is ever read.
  (c) **Fake + storage**: config `checkpoint.fake_storage` (additive, default false, test
  mode only) makes `fake` a local kind in `CheckpointHub.from_config`, so fake attempts get
  `GPU_STORAGE=file://<home>/storage`, owner claims and `GPU_RESUME_URI`; the fake's
  simulated runner publishes each checkpoint (runner/storage.py layout, a small
  state.json, `created_at` = simulated time via the new `publish_checkpoint(created_at=)`)
  when status()/logs() observe its ckpt_end time (D9 style), stops once owner.json names a
  later attempt, reports storage URIs in ckpt_end, and checks GPU_RESUME_URI at submit
  (missing = fresh start with a log line). Off by default, so no other test changed. The
  fake's quota() now honours `reset_anchor`. (d) **Shell uses phase 5**: `/run --smoke
  --data`, `/route --smoke --hours`, `/quota --refresh`, `/policy` without the pre-phase-5
  fallback; tab completion for `--data` paths (a `NAME=` prefix is kept) and `/policy
  show|set|reset` + rule keys; the panel places a checkpoint by URI: `hf://` -> HF Hub, a
  storage key `jobs/<id>/ckpt-N` under file:// -> local storage, else this Mac /
  `<provider> disk`. (e) Screenshots re-taken with phase-5 data: posed providers get GPU
  offers and kaggle's Saturday anchor (shown in local time, `↻Fri` in PDT); jobs are
  placed by the scoring router (only lightning is pinned), the approval is an agent job
  over the 1h auto limit, checkpoints go through local storage, and `shell-route.png`
  (new) shows /route reasons and the /quota ledger. Checked by hand on tmp homes with port
  0: a real daemon routed 6h -> kaggle, 20 min -> colab ("kaggle saved for jobs over 4h"),
  `--vram 24` -> no fit (exit 12), `--smoke` -> local MPS; `gpu quota` showed kaggle live
  0/30h, colab estimated, local unlimited; `gpu run --hours 20` waited for approval ("would
  use 67% of kaggle's remaining 30h quota") and was denied with 0 attempts (its gpu.yaml
  named an unset secret, so nothing could have reached Kaggle); a real local job that
  writes into `gpu.checkpoint_dir()` was SIGKILLed mid-run, attempt 2 restored checkpoint
  2 from local storage and printed "resumed from step 15", done; a test-mode daemon with
  `fake_storage` ran a fake job that died after checkpoint 3 and resumed on attempt 2 from
  `storage/jobs/<id>/ckpt-0003`. Not verified: HF Storage Buckets live (no token on the
  dev machine), and the re-ask against a real provider.
- **D44** (phases 4-5 review fixes, 2026-09-24; 37 findings: 34 fixed, 3 in part, none
  rejected; each has a regression test, listed under Repo layout). **Engine/router/policy**:
  (a) a WAIT with any rejection that has no `until` (capacity, a login) sleeps at most the
  engine backoff, whatever a far-off quota reset says; the driver reports when a step ended
  a live attempt (`on_attempt_ended`) and the supervisor wakes queued drivers, and a woken
  queued job now routes at once (before, a wake just re-slept until not_before, which also
  made the health-OK wake a no-op for queued jobs). Router `retry_at` keeps its meaning
  (earliest known `until`). (b) A WAIT held back only by quota resets (QUOTA/EXHAUSTED with
  `until`) sets `waiting_since` to the reset, so max_queue_wait_s counts from there (D41's
  "waits for the reset" no longer ends in gave_up). (c) RulesPolicy: without `--hours`, a
  bundle-heuristic runtime under the auto limit counts as unknown (`unknown_hours`; the
  estimate falls back to 1h = the agent limit); a guess over the limit keeps the hours rule.
  (d) RESERVED (and OVERRIDE) rejections are ignored by the all-excluded/all-VRAM checks in
  `_no_fit_reason` and the driver's invalid_job classification. (e) A known 0 left gives
  `quota_share` = 9.99 (`SHARE_NOTHING_LEFT`) in `_pinned` and `_score`; the policy says
  "quota is used up". (f) `QuotaService.stale()` refreshes `early_s` (max(60, 10% of
  refresh_s), at most ttl/2) before the TTL; its loop sleeps on the injected clock. (g) The
  planned quota handoff and the exhausted_until it sets use the ledger
  (`engine/context.quota_left_hours`: other attempts count, a fresh reading minus our GPU
  time since it). (h) `Candidate.quota_source` (additive) and "(est)" on every "left"
  fragment and the policy's quota reason. (i) `RoutingContext.resume_step` (additive): a
  resumed job's hours = hours x (1 - checkpoint step / total), at least 5%, for quota need,
  share, scoring and policy; the chosen reason says "resuming: ~1h left of 10h". (j) Session
  deadlines anchor at the adapter's started_at, else the attempt's submit, and use min(catalog
  cap, `remote_meta["session_s"]`); Kaggle records its `-t` there.
  **Checkpoint storage**: (k) `CheckpointHub.prepare_attempt/checkpoint_for/latest` raise
  StorageError(retryable) while HF is down for a reason that should pass (network, 5xx, a
  locked Keychain: `hf_down_for_now`); the driver waits (one note) up to the new
  `checkpoint.storage_wait_s` (3600) per attempt, then calls with `degrade=True` (no storage
  / start over, with a note). Refusals (no HF_TOKEN_REMOTE, 403 on owner.json, a full storage
  quota) return None = no storage, never an endless retry; a dataset that cannot be stored
  excludes the provider. A locked Keychain at the 300 s recheck keeps the cached client. The
  migrating step's reconcile marks itself done only after a successful read (retries within
  the same bound). (l) Cross-backend copies are checked against the manifest (sizes, sha256
  when it is one) and fall back to storage's newer latest.json or the newest intact older
  checkpoint; the target's latest.json is merged forward. (m) Cleanup: a finished job's
  `jobs/<id>/` leaves every backend we can reach (`checkpoint.cleanup`, default true);
  uploads evict datasets unused for `checkpoint.dataset_keep_days` (30; `Store.
  unused_data_cache`). (n) Remote storage needs Keychain `HF_TOKEN_REMOTE` (the admin
  HF_TOKEN is never sent: the job runs as the runner's user); Kaggle's run.py and Colab's
  launcher hand the token to bootstrap in a 0600 file (`GPU_STORAGE_TOKEN_FILE`) it deletes
  after reading, never in an environment; Kaggle's secrets file on /kaggle/input stays
  readable by the job (documented in providers/kaggle/remote.py and docs/notes/hf-hub.md).
  (o) Datasets follow symlinked directories (ancestor loops skipped); `digest` OSErrors are a
  data problem naming the file (user_error), not an internal error.
  **Runner** (`bootstrap.py`, `storage.py`, py3.8): (p) resume candidates = GPU_RESUME_URI,
  storage's latest.json, older intact checkpoints; each verified against its manifest
  (`storage.verify_checkpoint`); none = fresh start; a missing checkpoint while latest.json is
  unreadable, or a restore OSError, = exit 90. (q) After a failed publish, latest.json is
  re-read: this attempt's seq landed = published; a seq it names is never reused. Pruning
  deletes every ckpt below the kept window (`list_checkpoints`). (r) Handoff answers: freeze,
  write the ack with 5 tries, unfreeze + re-answer at the next poll on failure; a failed
  handoff upload is retried until 120 s past the wait. (s) A downloaded resume is renamed
  into the resume dir; a sync that would not fit the disk uploads from the checkpoint dir
  and checks the fingerprint before the manifest (`publish_checkpoint(skip_upload=)`).
  (t) LocalStore fsyncs files and directories (F_FULLFSYNC for write_many on macOS).
  (u) Python < 3.10 is told HF storage needs 3.10+ without a pip attempt.
  **Shell**: (v) Enter takes a popup candidate only when the user moved the highlight or
  typed part of the word; never a default id for /cancel /deny /approve; candidates that need
  more (`Candidate.more`: `/policy set`, rule keys) keep editing. (w) Commands run on daemon
  threads; `shell.app.run()` os._exits when one is still running after the app ends.
  (x) Notices diff against the last UP snapshot across outages; a status timeout keeps the
  last snapshot; "daemon started" once per pid; no finish notice for a job a /logs or /watch
  block showed ending (`LiveBlock.saw_end`). (y) /logs reads at most ~5000 lines per attempt
  (a dim "… N earlier lines" line), writes 250-line chunks, keeps the reader's scroll;
  pgup/pgdn scroll the newest /logs block first; the transcript keeps 300 blocks and 3
  stopped /logs outputs; the feed's first metric read is the last 5000 lines; its quota
  thread restarts whenever it ended. (z) /jobs and /history call `cli.app.list_jobs` (shared
  with the CLI: empty states, "… more" hint) and accept `--before`; metric thinning keeps
  the first point; script completion is one scandir, prefix-filtered (50k files: 390 ms ->
  32 ms); ↑↓ history restores the draft; one state word (`render.state_label`: "needs
  approval") in panel, tables, popup, live heads and notices, and daemon text is shellified
  (`panel.shell_words`) with the /logs hint not repeated. Screenshots re-taken. In part /
  not done: `gpu storage gc` (jobs that ended before D44 keep their storage), the runner's
  memory stays readable by same-user processes (inherent), the engine cannot tell a
  runner's Python version before it runs (a <3.10 image still gets HF storage assigned; the
  runner degrades), heartbeat-derived session starts. Verified: 1673 passed, 22 skipped,
  `-m crash` 10, ruff + mypy clean. Not verified live: HF buckets, Kaggle, Colab.
- **D45** (phase 6a: MCP server, skill, plugin, Codex; details under "Agent integration")
  The MCP server has no approve/deny tool, and `/gpu-approve` is a plugin command with
  `disable-model-invocation: true` that runs `gpu approve` through Bash only because the
  user typed it (hex id checked first). Approval-needed results carry
  `guidance.tell_user` with the reason, the one-line route reason and both commands. MCP
  results reuse the CLI `--json` shapes, trimmed by default (nulls dropped, last 3
  attempts/checkpoints, last 5 events, compact overview rows; measured: a JobDetail with 7
  events is ~6 KB and 2 fake ProviderViews ~4.8 KB, which a polling agent pays every call);
  `verbose=true` returns the full documents. Blocking is bounded for MCP clients' tool
  timeouts (Codex: 60 s): `wait_s` <= 50, fetch <= 300 with `in_progress`. For agent specs
  (tool args and the project's gpu.yaml) the MCP layer also refuses dataset paths the
  bundle deny list calls credentials (`packaging.files.looks_secret`, any `SECRET_DIRS`
  part) and a project root or data path that is the home directory or above it: a job log
  an agent read could ask it to upload `~/.ssh` as a dataset. `gpu mcp` is dispatched in
  entry.py (like `daemon`) so typer never touches stdout; cli/app.py only gained a
  forwarding `mcp` command for `--help`. Verified: `tests/mcp` 24 passed; `claude plugin
  validate` (plugin + marketplace) passed; install commands run against a throwaway
  `CLAUDE_CONFIG_DIR`; one headless `claude -p` (Claude Code 2.1.281, `--mcp-config` =
  `uv run gpu mcp`, tmp `GPU_ROUTER_HOME`, test mode, port 0) called gpu_route ->
  gpu_submit -> gpu_status(wait_s=50) -> gpu_logs(tail=5): job done on `fake` in 6 s,
  outputs in `runs/<id4>/`, 6 turns, ~$0.18; its auto-started daemon was stopped after.
  Open for the daemon owner: `POST /v1/jobs/{ref}/approve` accepts any client, including
  `X-Gpu-Router-Client: mcp` (actor `agent`); the header is self-declared, so refusing it
  there would be defence in depth only.
- **D46** (phase 6b: Claude Code status line; details under "Claude Code status line")
  The reference script wins over the spec's prose where they differ: percentages are
  `38;5;252` (not bold), labels `38;5;243`, the grid is COL=30 with a 10-cell ceil meter,
  resets use the script's `when()` incl. its `tr 'AMP' 'amp'` (Monday = `mon`). Deliberate
  deviations from the spec's mockups: labels are `gpu ` + 1 space (the mock's 8-wide
  `gpu     ` cannot fit the 30-col grid), `~1:50 left` becomes `1:50 left` (keeps a 2-space
  gutter with 2-digit %), the provider quota meter keeps the script's 70/90% colours (it is
  a budget like `week`; the spec's "colour only on icons" holds for everything else, and
  job progress is never coloured), an estimated quota shows `~73%`, and when 2 rows allow
  it a second job's first row is shown (an approval above a running bar) before counting
  the rest. `gpu statusline` is dispatched in entry.py (argparse, like `daemon`/`mcp`);
  cli/app.py only forwards for `--help`. State.json changes are additive (schema stays 1)
  plus a writer heartbeat; runtime.py passes the migrated window, `paths.job_metrics` and
  the ledger's `unlimited`/`remaining`. Verified: `tests/unit/statusline` 156 passed (incl.
  a real DaemonRuntime job rendered running -> done -> expired -> approval), the reference
  script run in a sandbox with the wrapper, dry-run diff on a copy of the statusLine
  key. Not done: install on the dev machine's real settings.json (left to the maintainer:
  `gpu statusline install`); a live Claude Code session showing the rows.
- **D47** (phase 6 integration; details under "Agent integration" and "Claude Code status line")
  The plugin ships the status-line wrapper (`plugin/statusline/gpu-statusline.sh`) but never
  activates it: plugins cannot set `statusLine`, the plugin cache path is versioned (it would
  break on every plugin update), and the spec says to ask before touching settings.json. The
  in-Claude-Code entry point is a 4th, human-only command `/gpu-statusline`
  (`disable-model-invocation: true`, allowed-tools limited to `gpu statusline status` and
  `preview`) that shows the state and a preview and tells the user to run `gpu statusline
  install` in their terminal; the model never runs install/uninstall. `tests/unit/test_plugin.py`
  pins the plugin's shape. The daemon coalesces state.json writes to one per second, so a
  status line can lag a transition by up to ~1 s (0.54 s measured for awaiting_approval); with
  Claude Code's 2 s refresh that is invisible, and scripted checks wait for state.json before
  capturing. E2E (tmp GPU_ROUTER_HOME, test mode, port 0, fastmcp stdio client spawning `uv run
  gpu mcp`; a real `~/.claude/statusline.sh` (the reference script) run through a sandbox HOME
  symlink so its cache writes stay out of ~/.claude, plus one real-HOME run with a payload that makes it only
  read): approval `gpu ⏸ train.py → fake T4      ~2h  /gpu-approve`, running `gpu ██░░░░░░░░ 20%
  0:02 left  fake ░░░░░░░░░░ 0% ↻Thu 6am` + `train_yolo · T4  loss 0.802 ↓  ckpt <1m ago`,
  finished `gpu ✓ train_yolo · 1m         → ./runs/a0c6`; the script's own lines were a
  byte-exact prefix every time, the appended rows equalled `gpu status --line`, `gpu statusline install
  --yes` / `uninstall --yes` on a sandbox settings.json holding only the original statusLine key
  restored it exactly. Latency (M-series, 200 runs): `gpu status --line --stdin` p50 20.6 /
  p95 22.9 / max 23.7 ms (python -c pass 16.2 / 18.1); the reference script alone p50 56.4 ms vs
  the wrapper 69.2 ms (+12.8 ms). Plugin installed into a throwaway CLAUDE_CONFIG_DIR: 4
  commands, 1 skill, 1 MCP server, ~258 tokens always on.
- **D48** (phase-6 review fixes, 17 findings, all fixed; regression tests in
  `tests/{unit/engine,mcp,cli,unit/statusline}/test_review_fixes_phase6.py` and
  `tests/unit/checkpoint/test_overrun_handoff.py`). (1) `models.RESERVED_SECRET_NAMES`
  (kaggle, KAGGLE_API_TOKEN/_KEY/_USERNAME, HF_TOKEN, HF_TOKEN_REMOTE, GPU_STORAGE_TOKEN,
  MODAL_*, LIGHTNING_*, colab; case-insensitive) are refused at intake (gpu.yaml with
  file:line, daemon submit/route `_check_new_spec`) and again in `_attempt_context` (a
  stored spec fails with `ReservedSecret`, never exports); not in the JobSpec validator so
  stored specs still load (D39 pattern). Policy rules are additive with per-audience
  defaults filled by a before-validator (`_AUDIENCE_DEFAULTS`): `ask_secrets` (agent: a job
  listing `secrets:` asks, the names are in the reason and so in tell_user) and
  `enforce_hours`. (2) `gpu run` is an agent job with `--as-agent` or an agent marker
  (`agent.py`), with the same intake checks; the skill, /gpu-run and the Codex section no
  longer offer a CLI fallback (stop and report instead). (3) Declared hours are enforced for
  agent jobs: past `policy.overrun_limit_s` = max(1.5x, +15 min) of running time over all
  attempts, the driver asks the runner for a checkpoint (handoff code `overrun`; no storage
  or no answer = stop anyway), moves the job running -> migrating (new Reason
  `hours_exceeded`) and `_apply_decision` asks for approval ("ran past its declared 30m");
  approving lets it finish (no second stop); exempt providers and user jobs are never
  stopped. `gpu_route`'s preview shows hours, hours_source, stopped_for_approval_after_h.
  (4) `packaging.files.credential_path_problem` (every SECRET_DIRS part incl. the last,
  pairs, credential names, `HOME_CREDENTIAL_STORES` inside/containing, the gpu-router data
  dir) guards agent project dirs, roots and datasets; dataset symlinks into a store are
  refused for agents (walk capped at 20k entries); `checkpoint.data.scan` never lists
  credential files or enters linked stores for any job; `select_files` checks the absolute
  path too; deny list + auth.json, .credentials.json, *.keychain(-db), .codex, .lightning,
  .password-store. (5) The skill description says "Use this skill, not the colab skill";
  INSTRUCTIONS too; Phase 6 install step 5 offers to disable ~/.claude/skills/colab (not
  run). (6)+(7) `protocol.METRIC_NAME` `[A-Za-z0-9_.:/@%+-]{1,64}`, `MAX_METRICS` 32 per line
  and per job (`merge_metrics`, primary names win); JobSpec.name drops controls; state.json
  and fast.py strip C0/C1/bidi; MCP metrics <= 12 + `metrics_not_shown`, overview <= 20/10
  rows, `untrusted` on status/fetch/submit/cancel. (8) guidance.follow: `end_turn` for
  approvals, `report_and_stop` past 15 min or unknown, `wait` otherwise; INSTRUCTIONS, tool
  descriptions and docs match. (9) tell_user carries no job name. (10) idempotent gpu_submit
  (see Agent integration); SubmitUncertain says the same call is safe; Codex doc leads with
  config.toml. (11)(14)(15)(16) statusline records per settings file, O_EXCL backups,
  self-degrading command, writability check + OSError reporting + record rollback. (13)
  wrapper waits <= ~1 s for gpu and kills a stuck one; the user's lines print first. (12)
  per-attempt ETA. (17) wall-clock heartbeat tick. Not done: a `job:` keyring namespace
  (reserved names instead); `/doctor` colab-skill check (phase 8); agents can still run
  `gpu approve` through Bash (Claude Code's permission prompt is the boundary; /gpu-approve
  needs it). Measured: wrapper around a trivial command p50 32.9 ms with the real `gpu
  status --line` p50 20.9 ms.
- **D49** (phase 7a: Lightning AI adapter; details in `providers/lightning/NOTES.md` "How the
  adapter works") The SDK never runs in gpu-router's interpreter (D3): each call is one
  bounded `driver.py` process in an isolated env (settings `python`, else the `uv tool install
  lightning-sdk` env, else `uv run --no-project --python 3.12 --with lightning-sdk==2026.9.18.post1`),
  own session, killed as a group on timeout; credentials (Keychain `LIGHTNING_USER_ID` /
  `LIGHTNING_API_KEY`, else the daemon env, else ~/.lightning/credentials.json read by us) go
  into that child's env only, whose HOME / LIGHTNING_CREDENTIAL_PATH / LIGHTNING_SETTINGS_PATH
  point at `<home>/providers/lightning/sdk-home` (~/.lightning is never read or written by the
  SDK), version check off, no SDK import without credentials (it would open a browser).
  `gpu login lightning [--stdin|--import|--browser [--no-open]] [--no-check] [--json]`
  (prompt/stdin/file, or lightning.ai's CLI sign-in redirect caught by a one-shot 127.0.0.1
  server that logs nothing and 303s to a value-free URL; one whoami check, never argv; the
  credential mode setting is `login_source` because config.yaml refuses `credentials`, D51). One attempt = one Studio-env job `gr-<job>-<n>` (Studio
  `gpu-router`, created once, never started by us); code, params, resume file and a secrets
  file travel through the teamspace drive `uploads/gpu-router/<name>/`; the job command runs
  `launch.py` from the `/teamspace` mount or a copy the Studio's SDK downloads. Catalog:
  `enabled_by_default: true` (not logged in = auth_required with no SDK process),
  `session_hours: 4` = gpu-router's own per-job wall clock (Jobs have none; the launcher
  enforces it in the job, status() stops a job 15 min past it as a backstop, an A6 exception
  like D31), `reset_anchor: day 1 00:00 UTC` and `quota_per_gpu_hour: 0.68` (both [3P]).
  Quota is live credits (balance API, else the month's job costs). L4 (24 GB) stays in the
  catalog, so 17-24 GB jobs can route to Lightning (credits burn faster). Tests changed
  outside the area: `tests/unit/adapters/test_registry.py` (lightning on by default),
  `tests/unit/quota/test_ledger.py` (the no-rate case strips the new catalog rate),
  `tests/contract/conftest.py` (OWN_SUITES lightning). Verified: 164 lightning tests
  (unit + contract over SimLightning, the driver against a fake SDK, the launcher + real
  bootstrap incl. the wall clock, the pinned real SDK's names offline), ruff + mypy clean.
  Live 2026-09-25 (a free account, Keychain creds via `gpu login lightning --import`):
  healthcheck + quota, and one T4 job through the adapter (submit, status,
  logs, fetch of `gpu.json`, cancel; reattached from an empty home via `lookup_by_key`): Tesla
  T4 seen, exit 0, 0.043 credits, nothing left running. Fixes from it: ISO-string timestamps
  (`driver.epoch`), live quota limit = credits left + the month's job costs (a new account
  has 5.0 credits, not the catalog's 15 [3P]), the smoke's `timeout_s: 900` bounds an orphan.
  Not verified live: L4, the out-of-credits error text, the monthly top-up.
- **D50** (phase 7b: providers catalog cleanup, verify-later + manual lanes, inference lane;
  2026-09-24) (1) **Modal is dropped** (decided at phase 7): modal.com/docs/guide/billing says
  "Note that you must have a payment method on file in order to use Modal." (no-card rule,
  invariant 19). It is under providers.yaml `excluded:` (reason, quote, source, decided;
  the spec's other exclusions too), not in `ADAPTER_KINDS`, and a user providers.yaml that
  lists an excluded name is a ConfigError. Defaults that named it are empty now:
  `routing.big_vram_providers` {} and `policy.*.ask_providers` () (a persisted config that
  still says modal is harmless). Nothing free has more than 16GB, so an explicit VRAM ask
  above every provider is no_fit with "needs 24GB VRAM and no free provider has more than
  16GB (largest: kaggle); `gpu providers` lists the excluded ones" (data-driven: Lightning's
  L4 24GB counts when it is enabled). The skill, Codex section, MCP text, /gpu-run and the
  shell no longer offer Modal; `providers/modal/NOTES.md` is kept as history with a
  DROPPED banner. Redaction + reserved secret names for Modal stay (harmless, safer).
  (2) **Lanes**: `ProviderEntry.status` (active | verify_at_signup), `link`, `note`,
  `verify_at_signup` (what to check), `lane` (gpu | manual | verify); only the gpu lane is
  ever registered (registry) or shown by `Supervisor.provider_views`; paperspace and saturn
  are verify_at_signup with their checklists, sagemaker_studio_lab is `manual_only` (T4,
  4h/day, no API). The task's `enabled: false` / `manual: true` map to the existing
  `enabled_by_default: false` / `manual_only: true` keys. Clients list the not-routed lanes
  from the catalog files (like /doctor), so `/v1/providers` did not change.
  (3) **Inference lane** (evals, LLM calls; not GPU jobs; details in `inference/NOTES.md`,
  limits verified 2026-09-24 from the official pages): Groq (per model: 1K req/day, 200K
  tok/day, 30 RPM on the free plan), Cloudflare Workers AI (10k neurons/day, 00:00 UTC),
  Gemini API (free models listed; per-model limits only in AI Studio, so `null` = unknown,
  RPD at midnight Pacific), HF Inference Providers ($0.10/month credits, the `gpu login hf`
  token), HF ZeroGPU (5 min/day, Spaces only: listed, never routed). All speak the OpenAI
  chat API, so one httpx client (no new dependency). The **daemon owns it** (invariant 2):
  `InferenceService` on `DaemonRuntime.inference`, endpoints in `daemon/lanes.py`, calls in
  worker threads; the ledger is `<home>/inference/ledger.json` (0600, atomic, one lock), not
  gpu.db: per-window counters written from worker threads (the Store is event-loop only,
  inv. 9) need no migration (and no 0002 number race with other phase-7/8 work). Live =
  Groq's requests-per-day headers; else our own counts (est); 429s that say the day is used
  up block the model / provider until its reset, per-minute ones cool down, 402 = HF
  credits gone, 401/403 = key rejected (blocked 10 min), 5xx = 60 s cooldown (of the model
  on a per-model provider, of the provider when nothing answered); the next candidate is
  tried; a bad request is not. Router: candidates by alias / listed id /
  passthrough regex, filters (key, blocks, quota for requests/tokens/neurons/usd at model
  and provider scope), score = 40 x share left after the request (+20 when it resets within
  a day, -0.25 x priority, unknown limit 40 x 0.5 - 5, unlisted -15), one-line reason.
  Keys: `INFER_GROQ_API_KEY`, `INFER_GEMINI_API_KEY`, `INFER_CLOUDFLARE_API_TOKEN` +
  `INFER_CLOUDFLARE_ACCOUNT_ID` (prefixed so a job secret named GROQ_API_KEY stays the
  user's; added to `models.RESERVED_SECRET_NAMES`); Groq/Google key shapes added to
  `secrets.TOKEN_PATTERNS`. Prompts and replies are never logged or stored. A test-mode
  daemon refuses real calls unless GPU_ROUTER_REAL_PROVIDERS names the provider (inv. 20).
  (4) **gpu_infer** is an 8th MCP tool, additive to the spec's seven (the spec lists the
  inference lane but no tool for it; agents need it for evals): plugin tests assert
  "spec tools + gpu_infer", the skill and Codex section document it, the plugin
  description names it. `gpu infer` prints the reply alone on stdout (pipeable) and the
  provider/tokens/quota on stderr; `--dry-run` no_fit exits 12 like `gpu route`. The
  shell's /login for an inference provider prints the terminal command (no echo-free
  prompt in the TUI). docs/screenshots/shell-*.png re-taken without modal
  (GPU_SHELL_SHOTS + resvg-py, D42). (5) **Live Gemini, 2026-09-25** (the first inference
  key on the dev machine, stored by the orchestrator): `gpu infer` through a real daemon on a tmp
  home answered on gemini-3.8-flash, 3.5-flash and 3.5-flash-lite; the 2.5 models said "no
  longer available to new users", so the catalog lists the three 3.x models and passes
  `gemini-*` / `gemma-*` ids through. Fixes from that run: a 5xx from a per-model provider
  cools only that model (Gemini's 503 "high demand" had blocked every Gemini model for
  60 s); an error after fallbacks shows the route that chose the provider, not the
  re-route that skipped it ("no free inference provider serves ..."); `gpu infer` warns
  on stderr when `finish_reason` is `length` (thinking models return an empty reply under a
  small `--max-tokens`). The Cloudflare key check is now `accounts/{id}/ai/models/search`
  (works for user and account-owned tokens; `tokens/verify` is split between /user and
  /accounts). tests/shell/test_screenshots.py `test_shot_running` pins prep_data.py to
  colab: on the day before kaggle's Saturday reset the scoring router sent it to kaggle
  ("use it or lose it") and the training job then went to lightning. Not verified live:
  Groq, Cloudflare, HF (no key yet); no Paperspace / Saturn / Studio Lab account.
- **D51** (phase 8a: notifications + doctor; number chosen to leave D50 to phase 7b; details
  under "Notifications and doctor (phase 8a)") (1) Notifications are daemon-side on the
  EventBus and never on the engine's path: the subscriber only classifies, the job is read on
  the next loop turn (a commit callback may not read the Store), a bounded queue + one daemon
  worker thread run osascript/terminal-notifier with a timeout; dedupe + rate limit in the
  daemon. `config.Config.notifications` is a new free-form section typed in
  `notify/settings.py` (like routing/policy; no config migration, CONFIG_VERSION stays 1);
  `DaemonRuntime.notifier` is a new field; shell `/doctor` now calls `doctor.cli.shell_doctor`
  (the phase-4 `_cli_rows`/`_check` helpers are gone) and prints fix lines unshellified
  (`gpu login hf` is not `/login`). (2) `gpu doctor` observes: it never starts the daemon
  (`--start` does), a stopped idle daemon is a warning (fail when state.json lists active jobs
  or launchd should be running it), providers that are not enabled are not probed, provider
  accounts are only touched through the daemon (healthcheck, quota ledger; invariant 2), and
  its own probes are local and quota-free (`--version`/dist-info, `colab whoami` for scopes,
  `colab new --help` for the GPU list). Credential files are judged by stat() only. (3) Drift
  = live quota limit / unit / reset anchor vs providers.yaml; `--update-catalog` writes only
  those keys into `<home>/providers.yaml` after a diff + confirmation. (4) Found while
  building: `providers.<name>.credentials` (kaggle's and lightning's documented credential
  mode setting) is refused by `config._reject_secret_keys` ("looks like a secret"), so those
  modes can only be "auto" today; doctor treats a refused config as auto (lightning later
  took `login_source`, which doctor now honours). Not done: a notification click action (no
  `-execute`/`-open`: nothing runs from a notification). Verified: `tests/unit/notify` 71 +
  `tests/unit/doctor` 73 passed, the shell's `/doctor` test, ruff + mypy clean, one real
  osascript notification, live `gpu doctor` against a real daemon on a tmp home.
  (5) 2026-09-25 follow-up: an `inference` doctor group (additive to the --json GROUPS) with
  one optional row for the D50 lane's keys (names only, never values), and the lightning
  login row mirrors `credentials.resolve` (`login_source` modes; LIGHTNING_AUTH_TOKEN no
  longer counts: the adapter clears it). Verified: `tests/unit/doctor` 79 + `tests/unit/notify`
  71 passed, the shell renderer test, ruff + mypy clean, live doctor re-run (0 fail).
- **D52** (phase 8b: setup wizard; details under "Setup wizard (phase 8b)") (1) The wizard
  decides "done" with doctor's own checks through a `ProbeEnv` built from the same
  injectable `SetupEnv`, so `gpu setup` and `gpu doctor` never disagree and tests fake one
  world. (2) Additive changes outside `setup/`: `daemon.launchd.install(..., launchctl=,
  environ=)` (injectable; defaults unchanged), `cli.login._cache_namespace(..., home=)`,
  `cli/login.py`'s `login_app` now has `invoke_without_command` + a callback (bare `gpu
  login` = the overview instead of help), `cli/app.py` registers `setup.cli.register`
  (`gpu setup`, `gpu login kaggle|colab`), entry.py's bare-`gpu` branch calls `_first_run()`
  before the shell (stdlib `setup.firstrun` decides; any wizard error still opens the shell;
  "no" to "open the gpu shell now?" exits 0). (3) Kaggle credentials are checked with the
  kaggle CLI itself before they are stored (the adapter's `quota` call), not through the
  daemon: `gpu login` runs before a daemon may exist, like `gpu login hf|lightning`. (4) The
  Codex entry is appended as text (no TOML writer dependency); an inline `mcp_servers =
  {...}` table would make the result invalid, so that case falls back to `codex mcp add`.
  (5) Smoke tests are ordinary daemon jobs (source cli, label `via: gpu setup`, hours 0.1,
  max_attempts 1, checkpoints off) so the ledger counts them (invariant 2). Verified:
  `tests/unit/setup` (104 passed), the adjacent suites (entry, login, doctor, statusline install, inference front ends, tests/cli: 70), ruff + mypy clean, a live `gpu setup --dry-run` against the dev machine's real home
  with a tmp data dir (nothing written: settings.json / config.toml hashes unchanged, no data dir
  created), and `gpu setup --yes` twice through the real executable in a sandbox HOME with
  stub uv / claude / codex and a test-mode daemon (installs, status line, plugin, Codex entry,
  colab skill, 2 fake smoke jobs, "2 providers ready"; the second run changed nothing), plus
  the prompts in a real pty (the `[y/N]` answer, and bare `gpu`'s first-run prompt -> "not
  now" -> the shell). Not run for real: any step against the real ~/.claude, ~/.codex,
  launchd, uv tools, Keychain or providers (left to the maintainer: `gpu setup`).
- **D53** (phases 7-8 integration, 2026-09-25) (1) **Lightning's L4 stays routable** (the
  spec lists Lightning as "T4 / L4"; free credits, no card; D49 kept it): a job asking for
  17-24GB goes to Lightning L4, more than 24GB is no_fit "needs 32GB VRAM and no free provider
  has more than 24GB (largest: lightning)" (data-driven, D50; with Lightning logged out it
  says 16GB / kaggle). The "no free provider has more than 16GB" wording in the skill, the
  Codex section (kept in sync), the MCP instructions, the modal `excluded:` note, modal's
  NOTES banner and docs/cli.md now names the L4; doctor's modal row computes it from the
  catalog (`checks._largest_free`). (2) **Per-GPU quota rates**: catalog option
  `quota_per_gpu_hour_by_gpu` (Lightning `{L4: 1.68}`, the live list rate; T4 keeps the 0.68
  effective rate); `ledger.to_gpu_hours(entry, amount, gpu=None)` + `ledger.gpu_name`
  ("2xT4" -> "T4"); the scoring router converts credits with the offer it would place on
  (quota rejection, score/share, `--provider` pins) and the engine's quota handoff with
  `attempt.gpu` (`context.quota_left_hours(..., gpu)`), so 4.95 credits are 2.9h of L4, not
  7.3h: a 2h L4 job shows "uses 68% of the 2.9h left" (the 50% approval rule asks) and a 3h
  L4 job that cannot checkpoint waits for the reset. Ledger history keeps the default rate
  (the live balance, re-read every 30 min, bounds that). (3) `gpu statusline install`
  creates its record dirs 0700 (`install._private_dirs`, also tightens a 0755
  `<home>/statusline` from an older install): doctor's data-dir check had flagged it after
  install (found by the 8b run). (4) Kaggle login hints say `gpu login kaggle` (doctor fixes,
  `kaggle.cli.LOGIN_HINT`, shell /login steps); shell `/login <provider>` prints its steps
  unshellified, as phase 7b's inference steps already did (`gpu login lightning` had read
  `/login lightning`, which only re-checks). (5) The catalog's Lightning limit stays 15 [3P]:
  live accounts use the live limit (credits left + the month's job costs = 4.97 on the test
  account), so `gpu doctor` shows a drift warning ("providers.yaml 15, lightning reports
  4.99") until `gpu doctor --update-catalog` or a top-up; lowering it would guess the other
  way. Known: doctor judges Keychain logins by `<home>/secrets.index` (names per data dir),
  so on a fresh GPU_ROUTER_HOME it reports the Lightning login from
  ~/.lightning/credentials.json and "no inference keys" while the daemon uses the global
  Keychain (the default home's index lists them; only tmp homes differ). Verified: 2451
  passed, 23 skipped, `-m crash` 10, ruff check/format + mypy clean. Manual, tmp home + real
  daemon + port 0: `gpu providers` (colab/kaggle/lightning/local up; sagemaker_studio_lab
  manual, paperspace/saturn verify at signup; modal excluded with its quote), `gpu route
  --vram 24` -> lightning L4 and `--vram 32` -> no_fit exit 12, `gpu doctor` 23 ok / 8 warn /
  0 fail (warns: 2 HF tokens, lightning limit drift, 5 installs not run yet),
  `gpu setup --help`, `gpu setup --dry-run` (sandbox HOME: nothing written, no data dir;
  real HOME + tmp data dir: settings.json / config.toml / installed_plugins.json /
  known_marketplaces.json hashes unchanged, no launchd plist, "4 providers ready, ~153 free
  GPU hrs/month"), `gpu infer --help` + one live Groq call (gpt-oss-20b answered "4", live
  header 999/1000 requests left), `gpu notify test` (osascript, exit 0). Live Lightning
  through the real CLI: `gpu run gpucheck.py --provider lightning --wait` from this session
  is an agent job (CLAUDECODE=1, D48), so without `--hours` it waited for approval; it was
  cancelled, not self-approved, and re-run with `--hours 0.25`: job d608, one Studio
  job, Tesla T4 15360 MiB driver 580.178.04, torch 2.8.0+cu128 CUDA, 5 matmul
  steps (3.9-7.0 TFLOPS fp32), exit 0, `gpucheck.json` fetched into `runs/d608/`, ~0.046
  credits by the balance (4.951 -> 4.905), the remote job Completed (checked through the
  adapter afterwards); the daemon sent the approval and finished notifications through
  osascript (8a's unverified path). Not verified: L4 live, Cloudflare/HF inference, HF
  buckets, the Lightning monthly top-up, `gpu setup` for real.
- **D54** (phases 7-8 review fixes, 2026-09-25; 25 findings: 25 fixed, 0 rejected; regression
  tests next to each area's tests) Lightning: (1) the driver's submit stages are "pre" (before
  Job.run), "run" (Job.run raised and a lookup right after found no job) and "post" (the job
  may exist); an unstaged submit failure defaults to "post", Job.run errors after the create
  (`job.link`) return the created job, and a failed status read of the new job no longer
  fails the submit (invariant 6). (2) The backstop reports LOST only once the stop took
  effect (Stopping/Stopped/confirmed); else running + `stop_pending` in `runs/`, retried by
  status, healthcheck and quota (T_HEALTH/T_QUOTA 42 s so pending stop 15 + call stays under
  60). (3) A whole-log read that times out is never an empty log: the driver falls back to
  `job.logs(tail=500)` (flagged `timeout` + `tail`), the verdict is cached `lines_partial`
  (logs() serves nothing from a tail and upgrades once a whole read works), a timeout with no
  tail = Unavailable without starting the empty-log clock. (4) cancel() raises Unavailable
  when the stop is not confirmed and the job still reads Running. (5) Stale `staging/` dirs
  (plaintext secrets) are swept at adapter start and each submit unless a live `.submitting`
  intent names them (the intent is now written before the staging dir). (6) Uploads must fit
  T_SUBMIT: bundle + resume above (T_SUBMIT - 60 s) x `upload_mbps` (setting, default 8
  Mbit/s = 140 MB) = InvalidJob (the engine places the job elsewhere), and
  capabilities.max_bundle_mb is capped to it. (7) An explicit `provider_options.lightning.
  machine` must be T4 / T4_SMALL / L4 (`EXPLICIT_MACHINES`) and the GPU the router placed on.
  (8) Browser login: the redirect carries a random `state` (query, so a path allowlist on
  lightning.ai still matches; a naive second `?` is parsed), the callback needs it plus a
  localhost Host and no non-navigate Sec-Fetch-Mode, and on a terminal the verified account
  is confirmed [y/N] before storing (`run_login(confirm_account=)`). NOT verified live: that
  lightning.ai keeps the query in redirectTo (if it drops it, the browser sign-in fails with
  "not started by gpu-router" and `gpu login lightning` paste still works). Inference: (9) a
  400 saying the API key is not valid (Gemini's answer, captured live with a made-up key) is
  auth in `classify()` and `verify()`. (10) `InferRequest.deadline_s` (additive; remote.infer
  sends 400 s), no round with under 5 s left, read timeout `max(120, 30 + max_tokens/100)`
  capped by the time left, a read timeout after the request went out records the estimate
  (est); `/v1/infer` runs on `InferenceService.executor` (16 threads) and a provider slot is
  waited for at most 30 s (then the next provider, no ledger block). Setup/doctor: (11) an
  agent's `--yes` never answers `base.AGENT_GUARDED` (integration.*, launchd) and an agent is
  not told "add --yes"; (12) paste-only logins without a terminal count as unanswered (exit
  2) via `Ctx.needs_keyboard`; (13) smoke approval waits are `manual` (ok=None, pending) and
  job ids are recorded at submit (resume reattaches, a waiting job is not re-submitted);
  (14) the launchd step diffs an existing plist, backs it up to `<home>/backups/` and prints
  the absolute plist path; (15) notifications: terminal-notifier is also found in
  /opt/homebrew/bin and /usr/local/bin (a launchd PATH has neither), the daemon logs its
  backend (`notify.backend`), `HealthView.notifications` (additive) says which one it uses
  and doctor reports that instead of recomputing it; (16) env-only kaggle / lightning
  credentials are WARN in doctor (the launchd daemon has no such env) and the wizard offers
  a Keychain import of the Lightning pair; (17) `check_launchd` returns SKIP for an agent
  serving another dir before judging its program, a custom home with an unreadable plist is
  SKIP, and `gpu daemon install-launchd` refuses a non-default GPU_ROUTER_HOME unless
  `--this-home`; (18) a daemon cleanly stopped under launchd is WARN (FAIL only for a nonzero
  `last exit code` or active jobs); (19) the colab skill fix is `gpu setup --only
  integration.colab_skill` (the old `read -r -p` failed in zsh); (20) `gpu` installed in the
  uv tool bin dir but off PATH -> `uv tool update-shell`; (21) HF rows check the token role
  (`cli.login.token_role`: read-only = WARN) and verify HF_TOKEN_REMOTE with whoami
  (`ProbeEnv/SetupEnv.hf_role`, injectable), and store_hf notes a read-only token; (22)
  crashed / unfinished doctor rows are `Report.unknown` (additive), never "everything works",
  and `gpu doctor` exits 1 for them; (23) the colab probe's subprocess ends 1 s before the
  deadline so its throwaway HOME is removed, with an atexit sweep. (24) docs/spec.md: Decisions
  row 5 says Modal is dropped (the billing quote) and its body lines are marked superseded
  (the spec's table wins over this file, so the drop now lives where precedence puts it).
  Verified: 2505 passed, 23 skipped, `-m crash` 10, ruff check/format + mypy clean; the
  Gemini bad-key bodies were captured live with a made-up key (no account). Not verified
  live: the Lightning browser sign-in with the `state` query (above), the tail log read
  and the backstop/cancel paths against the real Lightning API (fake SDK + SimLightning).
- **D55** (status line visibility, decided 2026-09-25): GPU rows show only while a job
  is in use (queued, awaiting approval, provisioning, running, checkpointing, migrating).
  `statusline.finished_visible_s` now defaults to 0, so a finished or failed job drops off
  as soon as it ends; the macOS notification still reports the ending. The spec's "just
  finished (visible 10 min)" row is opt-in: set `statusline.finished_visible_s: 600` in
  config.yaml. `statusline/fast.py` now honours 0 instead of treating it as "missing".
- **D56** (Lightning L4 bug, found live 2026-09-25) Root cause of "asked for L4, ran on a
  Tesla T4, reported done on T4": `gpu run gpucheck.py --provider lightning --gpu L4 ...`
  put `--gpu L4` after the script, where D23's split hands `--gpu` to the script (job e940:
  `spec.gpu` null, `args ["--gpu","L4"]`), so the router placed the default T4 and the
  adapter asked for `Machine.T4`; the adapter's map (`L4` -> `Machine.L4`, slug `lit-l4-1`
  in lightning-sdk 2026.9.18.post1) was right and Lightning substituted nothing. (1) After
  the script, `--gpu X` is gpu's when X is a catalog GPU type or looks like a GPU model
  (`cli/app._GPU_MODEL`: T4, L4, A100-40GB, H100, RTX4090, `2xT4`); `--gpu 0`, `cuda:0`,
  `all`, `gpu0` stay the script's (D23's reason) with the stderr note. Also in shell /run.
  (2) Live L4 through the fixed CLI (job 1253): Lightning answered the L4 create with
  `jobs_service_create_job_with_http_info ... response: 403` twice (T4 creates work on the
  same account; `list_machines` still lists L4 at 1.68 credits/h), no job created, no
  credits spent. So the free tier cannot provide L4: providers.yaml offers Lightning T4 only
  (the L4 rate `quota_per_gpu_hour_by_gpu` stays for a user catalog that re-adds L4 on a plan
  that runs it; `MACHINES` keeps L4), the router's no-fit text is data-driven (now "no free
  provider has more than 16GB"), and the skill, Codex section, MCP instructions, modal's
  `excluded:` note and NOTES banner, docs/cli.md and doctor's docstring say 16GB. A 403 to
  the create call (stage "run") for a non-T4 machine is `InvalidJob` ("lightning refused to
  create a L4 job (403 Forbidden): this account's plan does not include L4 machines (the
  free tier runs T4 only)", hint `--gpu T4`) instead of AuthRequired: that one had said
  "needs login", cooled Lightning down for every job and made the pinned job retry forever
  (1253 was cancelled by hand after 3 attempts). (3) General GPU check, all adapters:
  bootstrap prints `::gpu:: {"t":"device","gpus":["Tesla T4, 15360 MiB"]}` right after
  hello (nvidia-smi, 15 s bound, nothing without it; protocol type `device`, additive);
  capture collects it; the driver compares it with `attempt.gpu` (`capture.gpu_mismatch`:
  every token of the placed model among the seen name's tokens, so an L40S is not an L4,
  and at least as many GPUs as placed) and adds one note per attempt, reason
  `gpu_mismatch` (detail placed/seen); the done message appends "; ran on a Tesla T4, not
  the L4 it was placed on" and `gpu status <job>` shows a `gpu seen` row. Fake directive
  `device` drives it in tests. Tests: `tests/unit/engine/test_gpu_seen.py`,
  `tests/unit/cli/test_run_argv.py` (gpu-type cases), `tests/unit/runner/test_bootstrap.py::
  test_device_line_reports_what_nvidia_smi_sees`, `tests/unit/providers/lightning/
  test_adapter.py` (403 for L4, T4-only catalog); `tests/unit/router/test_gpu_rates.py` now
  re-adds L4 as a user catalog would. Not verified live: an L4 run (the free tier refuses
  it), the device line on a real provider (it ships with the next remote job).
- **D56** (per-session status line, requirement added 2026-09-25: "the gpu status line should
  only show in the current terminal that has the gpu being used in it"). `gpu run` and the
  MCP server tag every job with its Claude Code origin as two `JobSpec.labels` (existing
  free-form, non-secret field, so no model, schema or API change): `claude_session` =
  CLAUDE_CODE_SESSION_ID, `claude_pid` = CLAUDE_PID (the MCP server, a direct child of
  claude, falls back to `os.getppid()` when CLAUDE_PID is missing but the env is a Claude
  Code child). Helper: `gpu_router/origin.py` (`claude_origin`, `with_origin`). Jobs from a
  plain terminal or the `gpu` shell carry no origin. The origin is display routing only:
  never secret, never auth. state.json (additive, schema 1): active and recent entries get
  `origin: {claude_session, claude_pid}` when the job has one. `statusline/fast.py`: with a
  `session_id` in the stdin JSON (the wrapper always passes it) only jobs whose origin
  matches are drawn: `claude_session` equals the stdin session id or this process's
  CLAUDE_CODE_SESSION_ID, or `claude_pid` equals this process's CLAUDE_PID (keeps the rows
  after /clear, which changes the session id; subagents share the parent's id and pid).
  Everything downstream (the `+N ...` counts, the approval id rule, the daemon-down hint)
  sees only that session's jobs; no-origin jobs show in no Claude status line (they are in
  `gpu`, `gpu jobs`). Without a stdin session id (`gpu status --line` by hand) every job is
  drawn, as before; `--session ID` scopes a manual run. Verified on the dev machine: the status
  line's `gpu status --line` process env carries CLAUDE_PID and CLAUDE_CODE_SESSION_ID
  (temporary probe across 8 live sessions, removed), the running MCP server's parent is its
  claude process. No wrapper change. Known limit: a pid reused by a later claude process
  could match an old session's still-active job (only while that job is active).
  Tests: `tests/unit/statusline/test_session_scope.py`.
- **D57** (sleep/wake recovery, found live 2026-09-29) After a night asleep on a flaky
  network, the healthchecks right after the wake timed out (`kaggle --version` > 15 s,
  `lightning whoami` > 42 s, `colab sessions`); colab happened to recover 26 s later, kaggle and
  lightning stayed unavailable until the flat `health_recheck_s` (900 s) re-check. (1)
  **Backoff**: an unhealthy provider is re-checked `engine.health_recheck_min_s` (60) after its
  first failed check, doubling up to `health_recheck_s` (900, now the cap): 60, 120, 240, 480,
  900, 900; a healthy answer resets it. Counts and due times live in the Supervisor (a
  restarted daemon checks every provider anyway); a provider a driver marked (AuthRequired,
  no healthcheck) is picked up one short backoff after its `last_healthcheck_at`; due checks
  run concurrently (`_check_many`), so one slow CLI no longer delays the rest; a bookkeeping
  error in a turn is logged (`engine.bug`) and the loop goes on. (2) **Wake**: the loop ticks
  every `health_tick_s` (30, or sooner when a re-check is due). A wake = wall time across its
  sleep > 2 x the sleep + 60 s, or wall time ahead of the monotonic clock by > 60 s since the
  last turn (asyncio's clock is `mach_absolute_time`, which stops while macOS sleeps, D48; the
  second signal also catches a sleep during a check). Then: log `daemon.wake` ("woke from sleep
  after about 5h; re-checking every provider in 30s, once the network is back"), reset every
  backoff, wait `wake_grace_s` (30) so the first check does not fail on a network that is still
  coming back, check every provider (healthy ones too), and wake queued drivers (a quota-reset
  wait sleeps on the stopped clock as well). A post-wake failure is retried 60 s later. (3)
  **What happened + what next**: a failed check (a timeout included: the adapters' own bounds,
  or the caller's `<p> healthcheck timed out after 60s`) stores `unavailable` with the
  provider's reason as before; `ProviderView.health_reason` for unavailable / auth_required /
  degraded ends with "; re-checking in 1m", computed when the view is built so it never goes
  stale, and the additive `ProviderView.next_healthcheck_at` carries the time; the stored
  reason stays the provider's own words (router messages quote it). `provider.health` lines
  say "kaggle: ok -> unavailable (kaggle --version did not answer within 15s); re-checking in
  1m"; a check that fails again logs at debug. (4) `FakeClock` keeps sleep deadlines on its
  monotonic time and gains `suspend(dt)` (wall time jumps; monotonic time and pending sleeps
  do not); `advance`/`set` move both, so existing tests see no change. The four config keys are
  additive (CONFIG_VERSION stays 1). Also: the Lightning SDK now runs from its `uv tool` env
  (`~/.local/share/uv/tools/lightning-sdk/bin/python`, which `SdkBridge` prefers), so a
  driver call no longer resolves `uv run --with lightning-sdk` over the network first.
  Verified: 2587 passed, 23 skipped (whole suite, crash harness included), ruff check/format +
  mypy clean; the launchd daemon restarted on this code: `gpu providers` colab / kaggle /
  lightning / local up, `/v1/providers` carries `next_healthcheck_at`, `gpu doctor` 31 ok /
  0 warn / 0 fail. Not verified live: a real sleep/wake (the FakeClock tests model it).
- **D58** (plugin install from GitHub) (1) A repo-root `.claude-plugin/marketplace.json`,
  marketplace `gpu-router`, lists the one plugin with source `./plugin`, so `claude plugin
  marketplace add ayushg8/gpu-router` + `claude plugin install gpu-router@gpu-router` works
  without a clone (the shorthand clones the repo and reads exactly that file). `plugin/`'s own
  marketplace `gpu-router-local` stays for checkouts and `gpu setup`; the two entries share
  name, description and version (`tests/unit/test_plugin.py`). (2) Doctor knows both installs:
  not installed and not running from a checkout, the fix is the GitHub one-liner (it was
  `claude plugin install gpu-router@gpu-router-local`, which fails without that marketplace);
  an outdated install is updated through its own marketplace; the wizard's enable/update
  commands take the plugin key from doctor's fix, so a GitHub install is never switched to
  gpu-router-local. (3) README leads with the GitHub commands and keeps the clone-based ones.
  Verified with Claude Code 2.1.283, each in a throwaway `CLAUDE_CONFIG_DIR`: `claude plugin
  validate --strict .` passed (0 warnings); `marketplace add <repo root>` + `install
  gpu-router@gpu-router`: installed, enabled, v0.1.0, 4 commands + 1 skill + the MCP server
  (`plugin details` lists commands as skills now: 5; ~305 tokens always on), the cache holds
  only plugin/ with the wrapper's exec bit; a git-cloned marketplace (a throwaway commit of
  the tree served over local smart HTTP, the clone path `owner/repo` takes) installed the
  same; `marketplace add ./plugin` + `install gpu-router@gpu-router-local` still works;
  `marketplace add ayushg8/gpu-router` against the public repo before this change is pushed
  fails with "Marketplace file not found at .../.claude-plugin/marketplace.json", the file
  this adds. Not verified: the GitHub install after the push (the maintainer pushes).
- **D59** (CI, 2026-09-29) `.github/workflows/ci.yml`: push + pull_request on `macos-latest`
  (the tool is macOS-only), `actions/checkout@v7`, `astral-sh/setup-uv@v10.2.0` (setup-uv
  publishes no major tag, so the full version is pinned) with its cache, then `uv sync
  --locked` (a stale uv.lock fails), `ruff check src tests`, `ruff format --check src tests`,
  `mypy`, `pytest -q`; `contents: read`, 30 min timeout, superseded runs cancelled; actionlint
  1.7.12 clean. README carries the badge. Hermetic tests: (1) child processes now get
  `PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring` from the autouse `gpu_home` fixture:
  `memory_keyring` only covered the test process, so a daemon subprocess or the real `gpu`
  executable used the macOS Keychain backend (a runner has nobody to answer a prompt; on the
  dev machine a job listing secrets would have read real items). (2) Found by the local CI
  run (a probe plugin listing what each test adds under an empty HOME):
  `tests/cli/test_review_fixes.py::test_run_json_wait_{error_after_submit,ctrl_c}...` called
  `monkeypatch.undo()` mid-test, which also undid `gpu_home` and the cli fixture (GPU_ROUTER_HOME,
  test mode, GPU_ROUTER_NO_AUTOSTART), so their `cli("cancel")` went to the DEFAULT
  data dir: in the CI run it auto-started a second daemon there (port taken, it exited), on the
  dev machine that is the real daemon's (a cancel of a job id it does not know); the patches
  now live in `pytest.MonkeyPatch.context()`. One colab test ran the fake
  CLI without the adapter's private CLI home (the fake refuses the real home, so harmless) and
  now passes `home=colab.cli_home`. After the fixes the probe finds nothing added under HOME
  besides uv's own cache. Local CI simulation: a fresh copy of the tree (tracked +
  untracked-not-ignored files) committed into a new git repo, `env -i` with HOME = an empty dir,
  PATH = /usr/bin:/bin + a dir holding only uv, CI=true: uv downloaded CPython 3.12.13 and the
  locked deps, ruff / format / mypy passed, pytest 2582 passed, 28 skipped in 6m32s (the 5
  skips beyond the dev machine's 23 are environmental: no Python 3.8, no ~/.claude/settings.json,
  3 real-SDK-shape checks with the pinned lightning-sdk absent from uv's offline cache).
- **D60** (field test 2026-10-04: agents were surprised that git-ignored `third_party/`,
  `data/...` crops and `experiments/**/out/` never reached the GPU, and nothing told them)
  (1) `JobSpec.include` (additive, `exclude_if` empty: a spec without it dumps and hashes
  exactly as before and an older daemon still accepts it): paths or globs relative to the
  project root, normalized + deduplicated by the validator (`files.normalize_include`:
  absolute, `~`, `..`, `.` / `**` alone refused; <= 64 entries). gpu.yaml `include:`
  (string or list), `gpu run --include` (repeatable, before the script only, D23), shell
  `/run --include`, MCP `include` on gpu_submit / gpu_route; flags and tool args add to the
  file's. (2) `select_files(project, include)`: `*` stays in one dir, `**` spans dirs, a
  trailing `/` = dirs only, a matching dir ships whole; walks only real dirs (a pattern
  through a symlink warns), never enters dirs that cannot match (`*.ckpt` skips data/),
  skips VCS dirs (an entry naming `.git/...` warns and ships nothing), virtualenvs (`.venv`,
  `venv`, any dir with `pyvenv.cfg`: a field-test project's third_party clone had one with 50k+ files),
  node_modules, tool caches and `*.pyc`, prunes credential dirs, stops at
  `MAX_INCLUDE_FILES` (50k: BundleError, hint `data:`). Every match goes through the same
  `_classify` as git's listing (deny list, symlink targets, outside-project refusal) and
  counts toward the 200 MB cap (the hint names include's share); a nested repo git lists
  only as a whole now ships its files; zero-match entries warn; include warnings are sorted
  so the archive never depends on pattern order. Manifest `files.included` {count, bytes}.
  (3) `bundle.bundle_summary(spec)`: the same selection without the archive plus
  `files.left_out`: `git ls-files -o -i --exclude-standard --directory` (ignored dirs
  collapsed, never walked), nested repos / dir symlinks from the selection, the walk's
  default-skipped dirs in a non-git project; venvs, caches, `runs/`, build output and
  credential-looking paths are never named, paths an include entry covers are dropped,
  sizes within a 5k-entry budget per path (20k in all, else "N+ files"), more than 8
  grouped by top dir. gpu_route and a creating gpu_submit return it as `bundle` {files,
  bytes, size, included?, left_out?, left_out_not_shown?, hint?, warnings? (<= 6, 300
  chars)}; `gpu run --dry-run`'s Bundle gains `included` / `left_out` / `hint` and a `left
  out:` line. Computed client-side (MCP server, CLI), not by the daemon: the engine was
  being edited in parallel and POST /v1/jobs has no field for it (D22's open point stays
  open). `left_out` is deliberately not in the manifest: ignored dirs like `runs/` change
  with every job and would change every bundle's sha. (4) SKILL.md and the Codex section
  (sync test in `tests/unit/test_plugin.py`), the MCP instructions / descriptions and
  docs/cli.md say: ignored files do not ship, read `bundle.left_out`, use `include` or
  `data`; and `provider="local"` runs on this Mac through gpu-router's one-at-a-time queue
  (max_concurrency 1), while unpinned the Mac takes only smoke tests (D41). Tests:
  `tests/unit/packaging/test_include.py`, `tests/unit/cli/test_include_flags.py`,
  `tests/mcp/test_include_bundle.py`, `tests/cli/test_include_cli.py`.
- **D61** (field test 2026-10-04, `docs/notes/field-test-2026-10-04.md`: 6 Kaggle jobs, 0
  ran) (1) **Kaggle's SaveKernel refuses a code file over ~1 MB** (probed live: 933,761 B
  pushed, 1,141,256 B `400 Bad Request`); the phase-3 inline limit (10 MB, marked [I]) was
  never true, so every bundle over ~700 KB failed. run.py stays under
  `remote.MAX_INLINE_SOURCE` (900,000 B); a bigger bundle or resume archive travels as a
  private content-addressed dataset `<user>/gpu-router-{bundle,ckpt}-<sha16>` (one `.bin`,
  `dataset_sources`, sha256-checked by run.py, mount `/kaggle/input/datasets/<owner>/
  <slug>/` seen live), uploaded once inside submit's 270 s budget and reused by later
  attempts and jobs (`_ensure_blob`, records in `providers/kaggle/blobs/`; 403 right after
  a create = not visible yet). A `400 Client Error` from the push is InvalidJob (nothing was
  saved): no provider cooldown, a pinned job fails at once with the reason (was "outcome
  unknown": cooldowns of 2-30 min and 6 retries). Settings `blob_datasets` (true),
  `max_bundle_mb` (100), `data_keep_days` (30); a background sweep deletes blobs unused for
  3 days (bundle/ckpt) or data_keep_days. (2) **Optional adapter call `stage_data`**
  (`Capabilities.stage_data`, `StagedData`, `AdapterCaller.stage_data`, config
  `engine.timeouts.stage_data` 3600, all additive): with no HF storage for the attempt the
  driver digests a `data:` path and asks the adapter to keep it; Kaggle stores a directory
  as one uncompressed tar (`data-<sha16>.tar.bin`, extracted by run.py under
  /tmp/gpu-router/data-src/) and a file as itself, GPU_DATA gets `kaggle://<owner>/<slug>/
  <file>` which run.py rewrites to `local` items for bootstrap. Reused by content hash
  across jobs (the cross-job cache for weights the field-test jobs re-downloaded per retry).
  Transient trouble = `_StagePending` (a StorageError: waits up to
  checkpoint.storage_wait_s, then excludes), refusals exclude the provider. New Reason
  `data_uploading` (note, only for datasets >= 50 MB); `data_uploaded`/`data_reused` say
  where. (3) **Routing knows where data can go**: `RoutingContext.data_unreachable`
  (additive) names providers a job's local `data:` paths cannot reach (no HF storage by the
  hub's cached state, `CheckpointHub.remote_data_possible`, and no stage_data); both
  routers reject them as EXCLUDED with that reason, and an all-excluded no-fit names it
  (`gpu_route` with data had picked colab, which would have been excluded at submit). (4)
  `gpu providers`: notes print under the table (an 80-column pipe squeezed the 7th column
  to 4 chars a line) and a cooldown is named ("cooling down for 27m00s after 3 failed
  calls in a row"). (5) MCP polling results (gpu_status / gpu_fetch, not verbose) carry a
  brief spec (`SPEC_BRIEF`) without spec_hash / bundle_sha256 (~25% of each poll), and
  `poll_every_s` never exceeds the 50 s wait_s cap when follow is `wait`. The bundle
  summary does not list a path the job passes as `data=` as left out (a parent dir gets
  "; data/rows/ is passed as data="). (6) Skill + Codex section: weights and datasets
  used every run go in `data`. Verified live: adapter-level 1.5 MB bundle + staged data on
  real Kaggle 2xT4 (done in 90 s, metrics fetched); a separate headless Claude Code
  session (Sonnet, MCP tools only, private daemon on a tmp home) ran 5 runs: Kaggle with
  data twice (second reused the upload), Colab, local MPS through the queue, a route:
  all done, $0.80. Not verified live: a Kaggle resume archive over the limit, the sweep
  against the real API, a big (GB) data upload. (7) Independent review fixes (6 findings,
  all fixed): `remote_data_possible` is True while HF is down for a reason that should
  pass (the placement waits, D44) and again once a refusal's pause ends (only placements
  call hf(), so a new `gpu login hf` would never count), and False while remote runners
  have no HF_TOKEN_REMOTE (`_remote_missing_at`, re-checked after RETRY_PERMANENT_S);
  `_stage_on_provider` treats AdapterContractViolation like Unavailable (invariant 7) and
  applies AuthRequired as a definitive submit error (provider health, requeue) instead of
  excluding the provider; Kaggle blob trouble before the push is RateLimited
  (`BLOB_RETRY_S` 60: definitive, no ambiguous lookup, no outage cooldown); the sweep
  re-reads a record under its blob lock and counts `uploading_at` as use; blobs off + a
  big resume archive = fresh start with a note (not InvalidJob); OSErrors while packing
  are Unavailable (A3). (8) Real use after the fix (a user project, 2026-10-05): 6 jobs done
  on Kaggle (12-36 min each), 0 before. One fetch hit a transient network error and the
  agent's `gpu_fetch` retry got all 781 files, but the stored `fetch_failed` note carried
  Kaggle's signed download URL: a JWE (`eyJ...` with an empty second part) no pattern knew.
  `secrets.TOKEN_PATTERNS` now redacts JWT/JWE, and the driver's `_err_text` redacts every
  adapter error text it stores (invariant 12, defence in depth). The note already stored
  holds an expired link and stays (events are append-only). (9) Output fetches retry
  transient failures (Unavailable, RateLimited, contract violations) after 20 s and 60 s
  (`driver.FETCH_RETRY_S`, A9 makes fetch re-runnable) before the `fetch_failed` note, which
  then says "after 3 tries"; definitive errors are not retried; the supervisor's manual
  re-fetch redacts its error text too.
