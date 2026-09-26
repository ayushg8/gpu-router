# hf-hub adapter notes (storage lane: datasets, checkpoints, logs/metrics; plus ZeroGPU)

verified_at: 2026-09-23. Tags: **[V]** verified against official docs or installed package source today;
**[I]** inferred or unverified; treat as a hypothesis until a live test confirms it.
Installed: `huggingface_hub` 1.32.0 in `.venv` (CLI is `hf`; `huggingface-cli` is gone) [V].

HF Hub is **not a training compute provider** in gpu-router. It is the storage and handoff layer
(spec: "Checkpoint and handoff", "Code and data movement"). ZeroGPU is a separate inference-only lane.

## 1. Summary table

| Item | Storage (Hub repos + Buckets) | ZeroGPU (inference lane) |
|---|---|---|
| GPUs + VRAM | n/a | NVIDIA RTX Pro 6000 Blackwell: `large` = half card, 48 GB (default, 1x quota); `xlarge` = full card, 96 GB (2x quota) [V] |
| Free limit | 100 GB **private** storage per free user/org, shared across models, datasets and buckets. Public storage "best-effort" [V] | 5 min GPU/day free account; 2 min unauthenticated; PRO 40 min [V] |
| Session cap | none; per-call HTTP commit timeout 60 s server-side [V] | 60 s default per `@spaces.GPU` call, raise with `duration=` [V]; max duration not documented [I] |
| Concurrency | API rate limit 1,000 req / 5 min (free), resolvers 5,000 / 5 min, pages 200 / 5 min [V]. Commit and repo-creation rate limits exist but are **undocumented** [V] | free account may host max 2 ZeroGPU Spaces (verified email, account older than 30 days) [V] |
| Reset schedule | rate limits: fixed 5-min windows [V]; storage: none (quota reflects squash within 36 h) [V] | 24 h after **first GPU use** (rolling, not midnight) [V] |
| Card required | no for free tier [V]. Pay-as-you-go private storage above 1 TB needs PRO [V] | no for free quota; overage credits need PRO ($1 / 10 min) [V] |
| Phone verification | not required per docs [I] | not mentioned; hosting needs verified email [V] |

## 2. Login / auth (headless)

- Token types: `read`, `write`, `fine-grained` (scoped to specific repos/orgs) [V]. Tokens are created
  **only in the web UI** at huggingface.co/settings/tokens; no user-token creation API (Trusted Publishers
  OIDC is CI-only; OAuth token exchange is Enterprise-only) [V].
- Whether fine-grained tokens can be scoped to a single **bucket** is not documented [I]. If not, the
  remote runtime needs a user-wide write token, which is the main blast-radius problem (see quirks).
- Default CLI login `hf auth login --token ...` writes plaintext to `$HF_HOME/token`
  (`~/.cache/huggingface/token`, override `HF_TOKEN_PATH`) [V]. **gpu-router must not call `hf auth login`.**
- Keychain handoff (per `secrets.py`): store under keyring service `gpu-router`, name `HF_TOKEN`.
  Read via `get_secret("HF_TOKEN")` -> `SecretStr`; pass explicitly as `HfApi(token=...)` in the daemon,
  and as env `HF_TOKEN` to child processes only. Nothing on disk. Build machine at the time: no token file, no
  `HF_TOKEN` env (checked presence only) [V].
- Validate with `HfApi().whoami(token=...)`, but cache the result: `/whoami-v2` is "heavily rate-limited"
  and 429s by design [V]. `hf auth whoami` is the CLI equivalent.
- `secrets.TOKEN_PATTERNS` already redacts `hf_[A-Za-z0-9]{30,}` [V].

## 3. "Submit" = write paths (exact calls)

Recommended layout (one private bucket per user, content-addressed datasets, per-job prefixes):

```
hf://buckets/<user>/gpu-router/
  datasets/<sha256>/...          # immutable by convention, written once
  jobs/<job-id>/ckpt/latest/...  # overwritten every ~20 min
  jobs/<job-id>/logs/*.jsonl     # metrics side channel
  jobs/<job-id>/outputs/...
```

Buckets (non-versioned, mutable, Xet chunk-dedup, S3-like; available to all users) [V]:

```bash
hf buckets create gpu-router --private                    # idempotent via Python exist_ok=True
hf sync ./ckpt hf://buckets/<user>/gpu-router/jobs/<id>/ckpt/latest --delete
hf buckets cp ./metrics.jsonl hf://buckets/<user>/gpu-router/jobs/<id>/logs/metrics.jsonl
hf buckets info <user>/gpu-router                          # size + total_files
hf buckets rm <user>/gpu-router/jobs/<id>/ --recursive [--dry-run]
```

```python
from huggingface_hub import HfApi
api = HfApi(token=tok)                        # tok from Keychain
api.create_bucket("gpu-router", private=True, exist_ok=True)
api.sync_bucket("./ckpt", "hf://buckets/u/gpu-router/jobs/j1/ckpt/latest", delete=True, quiet=True)
api.batch_bucket_files("u/gpu-router", add=[("./m.jsonl", "jobs/j1/logs/m.jsonl")], delete=[...])
api.download_bucket_files("u/gpu-router", files=[("jobs/j1/ckpt/latest/model.safetensors", "./x")])
api.bucket_info("u/gpu-router")               # BucketInfo(id, private, created_at, size, total_files)
api.copy_files("hf://datasets/org/ds/data", "hf://buckets/u/gpu-router/datasets/<sha>")  # server-side, Xet files only
```

Git repos (versioned) fallback / publish path [V]:

```python
api.create_repo("u/gpu-router-ckpt", repo_type="model", private=True, exist_ok=True)
api.upload_folder(repo_id=..., folder_path="./ckpt", path_in_repo="jobs/j1", commit_message="step 1200")
api.snapshot_download(repo_id=..., allow_patterns="jobs/j1/*", local_dir="./runs/j1")
api.delete_folder("jobs/j1", repo_id=...)      # history keeps the bytes
api.super_squash_history(repo_id=...)          # destructive; quota drops within 36 h
```

`CommitScheduler(every=minutes)` / `hf upload --every` exist for periodic push into git repos [V],
but each tick is a commit (see failure modes).

## 4. Status / logs / fetch / cancel

- Status: `bucket_info` (size, files); `list_bucket_tree` / `get_bucket_paths_info` to find the newest
  checkpoint [V]. Live change feed: `GET /api/buckets/<owner>/<name>/events` (SSE; ~15 min replay window,
  reconnect ~every 20 min, 30 s pings, 503 + Retry-After when unavailable) [V]. The daemon can tail
  `jobs/<id>/logs/` through this instead of polling (resolver/API budget friendly).
- Fetch outputs: `hf sync hf://buckets/u/gpu-router/jobs/<id>/outputs ./runs/<id>/` [V].
- Cancel: n/a for storage. Cleanup = `hf buckets rm ... --recursive`; deletes are immediate and permanent [V].

## 5. Quota

- Queryable [V]: `bucket_info().size`; repo info `used_storage` (`expand=["usedStorage"]`). Account-wide
  total and limit are shown on huggingface.co/settings/billing; no documented API for the account total [I].
  Ledger approach: sum `bucket_info().size` + `used_storage` of gpu-router repos vs 100 GB.
- Rate limits are observable per response via `RateLimit: "api";r=<remaining>;t=<secs>` and
  `RateLimit-Policy: "fixed window";"api";q=<limit>;w=300` headers; `huggingface_hub.utils.parse_ratelimit_headers`
  parses them, and the library auto-sleeps on 429 (>= 1.2.0) for downloads and paginated calls [V].
- ZeroGPU remaining quota is not exposed by any documented API found [I]; estimate from ledger (5 min per 24 h rolling).

## 6. Passing secrets to the remote

- The remote needs `HF_TOKEN` to push checkpoints. Inject through each provider's secret channel
  (Kaggle secrets, Colab userdata/env, Modal Secret, Lightning env) as env var `HF_TOKEN`; the library
  reads it automatically [V]. Never bake it into the code bundle or notebook source.
- Prefer a dedicated **fine-grained write token** used only by remote runtimes, separate from the local
  admin token, so it can be rotated after any leak [V: recommended practice]. Scope it to the gpu-router
  checkpoint repo; bucket scoping unknown [I].
- Downloads of public data should still carry the token (anonymous rate limits are per IP and shared on
  Colab/Kaggle egress IPs) [V: docs say "always pass HF_TOKEN"; shared-IP effect I].

## 7. Failure modes -> class

| Failure | Signal | Class |
|---|---|---|
| 429 API/resolver rate limit | HTTP 429 + `RateLimit` header `t=` | retryable (sleep `t`) |
| Commit/repo-creation action limit | 429 with "exceeded our hourly quotas for action ..." text [I] | retryable after ~1 h; switch to bucket path |
| 5xx / timeout / network | 500-504, 408 | retryable (lib retries 408/429/5xx) |
| Commit timeout client-side but applied server-side | 60 s HTTP timeout [V] | retryable, idempotent check (list files before re-commit) |
| Private storage full | 403/402-style quota error at upload [I] | permanent until GC; run GC then retry |
| Invalid / revoked token | 401 | permanent (user action: new token) |
| Token lacks scope / org policy | 403 | permanent |
| Bucket server-side copy across regions | copy falls back / fails [V: same-region required] | reroute to download+upload |
| SSE feed `reset` (cursor > 15 min old) | event `reset` | retryable: re-list, re-follow |
| ZeroGPU quota exhausted | Space returns quota error [I] | reroute to another inference provider |

## 8. ToS / policy

- Free public storage is "best-effort" and meant for content "useful to the community"; large public
  dumps of private scratch data are discouraged [V]. **Always create private.**
- Datasets hosted publicly at scale require a card and reuse intent [V]; irrelevant for private buckets.
- ZeroGPU on free tier: hosting needs a Space (Gradio SDK only) [V]; using it as a batch backend by
  scripting calls to a self-hosted Space is not addressed by docs [I]; keep usage demo-shaped and small.

## 9. Quirks

- Git repos keep every checkpoint version: 20-min cadence = 72 commits/job/day; a 1 GB checkpoint can
  eat the 100 GB free quota in ~1.5 days unless squashed (Xet dedup helps only for unchanged chunks) [I math on V facts].
  UX degrades after "a few thousand commits" [V]. **Use Buckets for checkpoints and logs.**
- Buckets have no versioning: `sync --delete` onto `latest/` mid-write can leave a torn checkpoint if the
  runtime dies during upload [I]. Write to `ckpt/step-<n>/` then update a small `ckpt/LATEST` pointer
  file last; GC older steps (keep 2).
- Bucket -> repo server-side copy is not available yet (repo -> bucket is) [V].
- Limits: < 10k entries per folder, < 100k files per repo, 500 GB hard max per file, ~50-100 files per
  commit (upload_folder auto-splits) [V; folder/file limits documented for git repos, bucket note says
  that section does not apply to buckets].
- `HF_HUB_ENABLE_HF_TRANSFER` is deprecated; use `HF_XET_HIGH_PERFORMANCE=1` [V].
- ZeroGPU: no `torch.compile` (AoT only), PyTorch 2.8+, Python 3.10/3.12, Gradio only; models must be
  moved to cuda at import time [V]. Not usable for training.

## 10. Sources

- https://huggingface.co/docs/hub/storage-limits (fetched 2026-09-23)
- https://huggingface.co/docs/hub/rate-limits (tiers "as of September '25")
- https://huggingface.co/docs/hub/storage-buckets
- https://huggingface.co/storage (bucket pricing: $18/TB/mo private, $12 public; egress included to 8:1)
- https://huggingface.co/docs/hub/security-tokens
- https://huggingface.co/docs/hub/spaces-zerogpu
- `.venv/.../huggingface_hub/hf_api.py` (create_bucket, sync_bucket, batch_bucket_files, bucket_info,
  delete_folder, super_squash_history), `_buckets.py` (BucketInfo), `utils/_http.py` (RateLimitInfo),
  `constants.py` (HF_TOKEN_PATH), `hf --help`, `hf buckets --help`, `hf auth login --help`

## 11. What phase 5 built (2026-09-24)

Code: `src/gpu_router/checkpoint/` (daemon side), `src/gpu_router/runner/storage.py` (the one
implementation of the layout, shipped in every bundle as `gpu_runner/storage.py`), bootstrap's
storage path, `gpu login hf`, `gpu run --data`. Decision D40 in CLAUDE.md.

- **Buckets, not git repos**, as recommended above: one private bucket `<ns>/gpu-router`
  (`checkpoint.bucket`; namespace from one cached whoami), created with
  `create_bucket(private=True, exist_ok=True)`.
- **Layout** (both backends): `jobs/<job>/ckpt-NNNN/<files>` + `ckpt-NNNN/.gpu-ckpt.json`
  manifest, then `jobs/<job>/latest.json` last (the torn-upload guard from section 9);
  `jobs/<job>/owner.json` (attempt allowed to publish); `jobs/<job>/attempts/<n>/`
  `heartbeat.json`, `log-tail.json`, `control.json` (daemon -> runner), `control-ack.json`;
  `datasets/<sha256>/<files>` + `.gpu-data.json` written last. The runner keeps the newest 3
  checkpoints per job (`checkpoint.keep`).
- **Calls used**: `batch_bucket_files(add=[(bytes|path, dest)], delete=[...])` for every write,
  `get_bucket_paths_info` for stat/exists, `download_bucket_files` with the `BucketFile` from
  paths-info or `list_bucket_tree(prefix, recursive=True)`, `whoami`, `create_bucket`.
  Checkpoint files are uploaded raw (not tarred), so Xet dedup applies across checkpoints.
- **Tokens**: Keychain `HF_TOKEN` (daemon) and `HF_TOKEN_REMOTE` (what remote runtimes
  get, as job secret `GPU_STORAGE_TOKEN`). Since D44 remote storage needs `HF_TOKEN_REMOTE`
  (the admin token is never sent): the job and anything pip installs run as the same user
  as the runner and can read what it holds. The launchers hand the token to bootstrap in a
  0600 file it deletes after reading (never in an environment, which stays readable in
  /proc/<pid>/environ), but on Kaggle the secrets dataset on the read-only /kaggle/input
  mount stays readable by the job for the whole run. Use a fine-grained token that can
  only write your buckets. `$HF_TOKEN` and `~/.cache/huggingface/token` are never read
  implicitly; `gpu login hf --import` copies one into the Keychain on request.
- **Python**: `huggingface_hub>=1.32` (the bucket API) needs Python 3.10+. A runner on an
  older image gets no HF storage (it says so; an hf:// resume exits 90). Kaggle and Colab
  images are 3.10+.
- **Local backend**: `<data dir>/storage/`, used by local Mac runs; the daemon copies a
  checkpoint between it and the bucket when a job moves across.
- **Not verified live**: no token existed on the build machine (checked: no Keychain `gpu-router/HF_TOKEN`,
  no `$HF_TOKEN`, no `~/.cache/huggingface/token`). Unit tests drive the code against a fake
  HfApi with the 1.32 call shapes. To verify live:
  1. create a token at huggingface.co/settings/tokens (write access; for remote runtimes a
     second fine-grained one), then `gpu login hf` (and `gpu login hf --remote`);
  2. `HF_TOKEN=<token> GPU_ROUTER_REAL_PROVIDERS=hf uv run pytest tests/unit/checkpoint/test_live_hf.py`
     (a round trip in a `<you>/gpu-router-selftest` bucket, cleaned up after);
  3. a real handoff: set `checkpoint: {handoff_margin_min: 715}` in config.yaml (due a few
     minutes into Kaggle's 12 h session), run a job that saves to `gpu.checkpoint_dir()`
     when `gpu.checkpoint_requested()` is true, and watch `gpu status <id>` show
     handoff_requested -> handoff -> the next attempt resuming from that checkpoint.
