# Modal adapter notes

> **DROPPED 2026-09-24 (decided in phase 7b). There is no Modal adapter.** The official
> billing doc (https://modal.com/docs/guide/billing) says: "Note that you must have a payment
> method on file in order to use Modal." That breaks the no-card rule (spec hard constraints,
> invariant 19). Modal is listed under `excluded:` in `providers/providers.yaml`, is not in
> `adapters/registry.ADAPTER_KINDS`, and nothing routes to or displays it as available. It was
> the only free-tier option above 16GB of VRAM (Lightning's free tier refuses L4, D56), and the
> router says plainly that a job needing more has no free provider. The notes below are kept as research history.

verified_at: 2026-09-23. Modal client inspected: `modal` 1.5.5 (via `uvx modal`). No Modal account was logged in on the build machine (`~/.modal.toml` absent, no `MODAL_*` env vars), so nothing here was run against a live workspace.

Legend: **[V-cli]** verified from `modal --help` / installed package source. **[V-doc]** verified from an official modal.com page fetched today. **[3P]** third-party only. **[I]** inferred, untested.

## Summary

| Field | Value |
|---|---|
| GPUs (type string: VRAM) | T4: 16, L4: 24, A10: 24, L40S: 48, A100-40GB: 40, A100-80GB: 80, H100: 80 (`H100` may auto-upgrade to H200; use `H100!` to pin), H200: 141, B200: 180, B300, RTX-PRO-6000 [V-doc for names and L40S/A100/H200 VRAM; the other VRAM figures are standard NVIDIA specs, I] |
| Max GPUs per container | 8 (A10: 4). More than 2 means longer waits [V-doc] |
| Free limit | Starter plan: "$30 / month free compute" [V-doc pricing] |
| GPU concurrency | "10 GPU concurrency", 100 containers [V-doc pricing] |
| Session cap | Function `timeout` 1 s to 24 h, default 300 s. Each retry gets a fresh timeout [V-doc timeouts] |
| Reset | Monthly, probably aligned to the calendar month (billing cycle is monthly; `billing summary` works on month-aligned intervals) [V-doc/V-cli; exact anchor I] |
| Card required | **Likely YES.** Official billing doc: "Note that you must have a payment method on file in order to use Modal." Third-party blogs say "no credit card required" [3P]. The official doc wins, but signup is still unconfirmed. See "Card question" |
| Phone verification | Unknown. Signup is OAuth only (GitHub/Google/SSO) [3P] |
| Hours per $30 (GPU line only; CPU and RAM add a bit) | T4 $0.59/h ≈ 50 h · L4 $0.80/h ≈ 37 h · A10 $1.10/h ≈ 27 h · L40S $1.95/h ≈ 15 h · A100-40 $2.10/h ≈ 14 h · A100-80 $2.50/h ≈ 12 h · H100 $3.95/h ≈ 7.6 h · H200 $4.54/h ≈ 6.6 h · B200 $6.25/h ≈ 4.8 h [computed from V-doc per-second prices] |

Per-second prices (pricing page, 2026-09-23): T4 0.000164, L4 0.000222, A10 0.000306, L40S 0.000542, A100-40GB 0.000583, A100-80GB 0.000694, H100 0.001097, H200 0.001261, B200 0.001736, B300 0.001972 USD/s.

## Card question (spec Decision 5, ask at phase 7)

- Official, and the strongest source: https://modal.com/docs/guide/billing says "Note that you must have a payment method on file in order to use Modal."
- Pricing page: no card wording either way.
- Third party (eesel.ai, aicreditmart, studentoffers) claims "no credit card required". Treat as low trust.
- Verdict: **unclear at signup, likely required before use.** The spec's hard constraint is "if signup asks for one, drop that provider", so the most likely outcome is that Modal gets dropped. Keep `enabled_by_default: false` and ask at phase 7; signing up settles it.

## Login / auth (headless)

- Credentials live in `~/.modal.toml` (per-profile `token_id` / `token_secret`). `MODAL_CONFIG_PATH` overrides the path. The env vars `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` take precedence over the file [V-cli: modal/config.py].
- First login needs a browser once: `modal setup` or `modal token new` opens a web session and writes the toml [V-cli].
- Headless or scripted: `modal token set --token-id … --token-secret … --profile gpu-router --no-activate`. Never pass secrets on argv, because they end up in shell history and `ps`. Prefer injecting the env vars into the subprocess.
- Keychain handoff (proposal, see `secrets.py`): store the token id and secret in the macOS Keychain under the gpu-router service. The adapter reads them and sets `MODAL_TOKEN_ID`/`MODAL_TOKEN_SECRET` only in the child process env. It never writes `~/.modal.toml` itself. If `~/.modal.toml` already exists, let the SDK use it.
- Health check: `modal token info` (shows the current token/workspace) [V-cli]. An AuthError or failure here means the provider is "not logged in", which is permanent until the user fixes it.

## Submit

Two viable patterns. **Recommended: A** (no deploy step, one app per job, and app-level logs and stop map cleanly onto a job).

A. Detached ephemeral app [V-cli]
```
# the adapter generates job_<id>.py: App("gpu-router-<job_id>"), Image (debian_slim + pip reqs + add_local_dir(project)),
# @app.function(gpu="T4", timeout=<=86400, volumes={"/out": Volume.from_name("gpu-router", create_if_missing=True)})
# def run(): subprocess(cmd); write outputs to /out/<job_id>/ ; volume.commit()
# @app.local_entrypoint() def main(): run.remote()
modal run --detach --name gpu-router-<job_id> -q job_<id>.py
```
`--detach` means "Don't stop the app if the local process dies or disconnects". Parse the app ID (`ap-…`) from the output, or find it later with `modal app list --json` by name [V-cli; parsing I].

B. Deployed function plus spawn (Python API) [V-doc job-queue]
`modal deploy` once, then `modal.Function.from_name(app, fn).spawn(args)` returns a FunctionCall whose `object_id` is `fc-…`. Later: `modal.FunctionCall.from_id(id).get(timeout=0)` polls (raises TimeoutError while still running). Results stay retrievable "for up to 7 days after completion". Spawn requires a deployed app.

## Status / logs / fetch / cancel

- Status: `modal app list --json` (running, deployed or recently stopped apps; the state field is in the JSON) [V-cli; field names I]. For B: `FunctionCall.get(timeout=0)`: result means done, `TimeoutError` means running, a raised remote exception means failed, `OutputExpiredError` means past 7 days. `get_call_graph()` gives per-input status [V-cli source].
- Logs: `modal app logs <ap-id>` (last 100 lines), `--tail N`, `-f` follow, `--since 2h`, `--source stderr`, `--timestamps` [V-cli]. Also `modal container logs <ta-id>`.
- Fetch outputs: write to a `modal.Volume`, then `modal volume get gpu-router <job_id>/ <local_dir> --force` [V-cli]. Checkpoints go to the same volume, so the chaining or resume logic can hand them to the next provider. `modal run -w path` only works for a str/bytes return value.
- Cancel: `modal app stop <ap-id> -y` (A). `FunctionCall.cancel(terminate_containers=True)` (B). `modal container stop <id>` sends SIGINT; `--graceful` lets the current input finish [V-cli].

## Quota

- **Queryable** [V-cli]: `modal billing summary --for "this month" --json`, `modal billing report --for "this month" --show-resources --json` (per GPU type), `modal billing rates --json`. The same data is available through `modal.Workspace.billing.summary/report`. Whether the summary shows credits remaining on Starter is untested [I].
- Fallback estimate: seconds × per-second rate from `billing rates`, kept in the router ledger. Unit: usd, limit 30, reset monthly.
- Reports cover "full intervals only", so today's spend lags. The ledger should add its own in-flight estimate on top [V-cli text; I].
- Budgets/spend limits exist (https://modal.com/docs/guide/budgets). Recommend setting the workspace spend limit to $0 net so overage can never be charged [I, not fetched].

## Passing secrets to the remote

- `modal secret create gpu-router-<name> --from-dotenv <file>` (or `--from-json`), then `secrets=[modal.Secret.from_name(...)]` on the function. Values arrive as env vars [V-cli]. Avoid `KEY=VALUE` argv form.
- Per-job ephemeral: `modal.Secret.from_dict({...})` in the generated script, with values read from Keychain at submit time and never written to disk [I; standard API].

## Failure modes

| Signal | Class |
|---|---|
| `AuthError`, `PermissionDeniedError`, no token | permanent (needs user) |
| Credits exhausted / spend limit hit / billing blocked (probably `ResourceExhaustedError` or a PermissionDenied message) | reroute (quota exhausted until next month) [I] |
| GPU type unavailable, long queue (container never starts past `startup_timeout`) | reroute, or retry with a GPU fallback list like `["L4","A10"]` [V-doc fallbacks] |
| `ConnectionError`, `InternalError`, `ServiceError`, grpc UNAVAILABLE | retryable |
| `FunctionTimeoutError` (hit `timeout`, at most 24 h) | reroute/chain from checkpoint |
| `ImageBuildError` (pip or apt fail) | permanent (user spec) |
| User code nonzero exit / `RemoteError` / `ExecutionError` | permanent (job failure, no retry) |
| `OutputExpiredError` (spawn result older than 7 d) | permanent (fetch outputs from the Volume instead) |
| Container preempted (Modal reschedules inputs) | retryable, handled by Modal; code must resume from checkpoint [V-cli container stop text; I] |

## ToS (https://modal.com/legal/terms)

- Prohibited: crypto mining, DoS, P2P, file-hosting/media-serving. No reselling, and no "third party direct access to or use of the Service".
- The main terms have no explicit multi-account clause; the spec's one-account rule covers it anyway. The router is personal use, which is fine [I].

## Quirks

- `modal` subcommand help needs real word-splitting (zsh `$c` loops break). Call the binary directly.
- Default function timeout is 5 min: the adapter **must** set `timeout` explicitly (cap 86400).
- `gpu="H100"` may silently run on H200 (billed as H200?). Use `H100!` to pin [V-doc]. `A100` may auto-upgrade to 80GB.
- Image builds are cached per definition. The first run pays the build time (billed CPU).
- `modal run` without `--detach` kills the app when the local process exits. Always pass `--detach`.
- Billing is per second, including container startup and idle time in the scaledown window.

## Sources (fetched 2026-09-23)

- https://modal.com/pricing (GPU per-second rates, "$30 / month free compute", "10 GPU concurrency", 100 containers)
- https://modal.com/docs/guide/billing ("must have a payment method on file in order to use Modal")
- https://modal.com/docs/guide/timeouts (1 s to 24 h, default 300 s)
- https://modal.com/docs/guide/job-queue (spawn/from_id, 7-day result retention, deploy required)
- https://modal.com/docs/guide/gpu (GPU strings, counts, fallbacks, auto-upgrades)
- https://modal.com/legal/terms
- `uvx modal` 1.5.5 `--help` for run, app, billing, token, secret, volume, container; package source `modal/config.py`, `modal/_functions.py`, `modal/exception.py`
- Third party (low trust): https://www.eesel.ai/blog/modal-ai-pricing, https://aicreditmart.com/ai-credits-providers/modal-free-tier-how-to-get-30-month-in-compute-credits-2026/
