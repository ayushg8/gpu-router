# Inference lane notes (phase 7b)

Evals and LLM calls on free inference APIs, separate from GPU jobs. Facts (limits, model
ids, resets) are data in `providers/providers.yaml` under `inference:`; this file is login
steps, quirks and sources. Checked 2026-09-24 from the official pages listed below with a
few web fetches, no browser, no account. Live since: Gemini only (2026-09-25, see Quirks);
Groq, Cloudflare and HF had no key on the build machine yet.

| Provider | Free allowance (official) | Resets | Endpoint (OpenAI-compatible) | Key |
|---|---|---|---|---|
| Groq | per model, free plan: gpt-oss-20b / gpt-oss-120b / qwen3.8-27b 30 RPM, 1K requests/day, 8K TPM, 200K tokens/day | "RPD" (their headers give the time left) | `https://api.groq.com/openai/v1/chat/completions` | `gpu login groq` -> `INFER_GROQ_API_KEY` |
| Cloudflare Workers AI | 10,000 neurons/day, free and paid plans | 00:00 UTC | `https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1/chat/completions` | `gpu login cloudflare --account-id ID` -> `INFER_CLOUDFLARE_API_TOKEN`, `INFER_CLOUDFLARE_ACCOUNT_ID` |
| Google AI Studio (Gemini API) | free tier; live on a free key 2026-09-25: gemini-3.8-flash, 3.5-flash, 3.5-flash-lite answer, the 2.5 models say "no longer available to new users"; per-model RPM/RPD are shown only in AI Studio | RPD at midnight Pacific | `https://generativelanguage.googleapis.com/v1beta/openai/chat/completions` | `gpu login gemini` -> `INFER_GEMINI_API_KEY` |
| HF Inference Providers | $0.10 of credits a month for a free account ("subject to change"); past it HF refuses until credits are bought | monthly | `https://router.huggingface.co/v1/chat/completions` | the `HF_TOKEN` from `gpu login hf` |
| HF ZeroGPU Spaces | 5 min GPU/day for a free account | 24 h after the first use | Gradio Spaces, no chat endpoint: listed, never routed | - |

Sources: https://console.groq.com/docs/rate-limits ·
https://developers.cloudflare.com/workers-ai/platform/pricing/ ·
https://developers.cloudflare.com/workers-ai/configuration/open-ai-compatibility/ ·
https://ai.google.dev/gemini-api/docs/rate-limits · https://ai.google.dev/gemini-api/docs/pricing ·
https://ai.google.dev/gemini-api/docs/openai · https://huggingface.co/docs/inference-providers/pricing ·
https://huggingface.co/docs/hub/spaces-zerogpu

## Login (the user runs these; keys never go on argv or into the chat)

```bash
gpu login groq                                  # key from console.groq.com/keys, no echo
gpu login gemini                                # key from aistudio.google.com/apikey
gpu login cloudflare --account-id <account id>  # API token with Workers AI read access
gpu login hf                                    # already there for checkpoints; reused
gpu infer --list                                # which ones have a key, models, quota
```

Each login makes one free check call (Groq and Gemini: the model list; Cloudflare:
`accounts/{id}/ai/models/search?per_page=1`, which works for user and account-owned tokens
and proves the account id and the Workers AI Read permission) and stores nothing when the provider rejects the key.
`--no-check` skips it; an unreachable provider stores the key with a note.

## Quirks

- **Groq**: every response carries `x-ratelimit-{limit,remaining,reset}-requests` = the
  model's requests per day ("Always refers to Requests Per Day") and `...-tokens` = tokens
  per minute. The ledger records the requests ones as a live reading per model. A 429 with
  `remaining-requests: 0` or "per day" in the text = the day is used up for that model
  (blocked until `reset-requests`); otherwise a cooldown for `retry-after`. The docs table
  listed only gpt-oss and qwen chat models on 2026-09-24; your org's exact limits are at
  console.groq.com/settings/limits.
- **Cloudflare**: neurons are not in the response; the ledger computes them from token
  counts with the per-model rates from the pricing page (`neurons_per_mtok`), labelled
  est. An unlisted `@cf/...` model is estimated at the dearest listed rates. The free plan
  refuses requests past 10,000 neurons (error 4006, "daily free allocation"): blocked
  until 00:00 UTC.
- **Gemini (live 2026-09-25, a free key)**: gemini-3.8-flash and
  3.5-flash think before they answer, so a small `--max-tokens` (20) comes back empty with
  `finish_reason: length` (`gpu infer` says so on stderr); leave it unset or give it room.
  A 503 "This model is currently experiencing high demand" and a 500 on a `gemma-*`
  passthrough were each about one model: 5xx answers from a per-model provider (Groq,
  Gemini) cool that model down for 60 s, only a missing answer (network, timeout) cools the
  whole provider. `gemma-*` ids pass through to Gemini (`-m gemma-4-31b-it`).
- **Gemini**: the docs no longer publish free-tier numbers ("can be viewed in Google AI
  Studio"), so the catalog has no limit: the router still uses it (ranked after providers
  with a known allowance) and a 429 whose quota id says `PerDay` blocks that model until
  midnight Pacific; `PerMinute` = cooldown for `RetryInfo.retryDelay`. Free-tier prompts
  and outputs may be used to improve Google's products (pricing page: "Content used to
  improve our products: Yes" on the free tier). Do not send private data through it.
- **HF**: credits are dollars, not requests; cost depends on which provider HF routes to,
  so usage is estimated at `usd_per_mtok` (0.5, rough) and labelled est. A 402 means the
  month's credits are gone: blocked until the 1st. Any Hub model id (`org/name`) is
  accepted as a passthrough; unsupported ones come back as "does not serve" and the next
  provider is tried.
- **Gemini bad key (live 2026-09-25, a made-up key)**: `400 INVALID_ARGUMENT "Please pass a
  valid API key"` on both /v1beta/openai/models and /chat/completions (the native API says
  "API key not valid" + reason API_KEY_INVALID), never 401/403: `classify()` and
  `verify()` treat those 400s as auth (so `gpu login gemini` stores nothing) (D54).
- **Deadlines (D54)**: a request has an overall budget (`deadline_s`; `remote.infer` sends
  400 s, 20 s under its own HTTP timeout); no round starts with under 5 s left, and each
  call's read timeout is `max(120, 30 + max_tokens/100)` s (capped at 600 and at the time
  left). A read that times out after the request went out records the request's estimated
  usage (est) before the next provider is tried; a connect timeout records nothing.
  `/v1/infer` runs on the service's own 16-thread executor, and a request waits at most 30 s
  for one of a provider's 4 slots before trying the next provider (nothing is blocked in the
  ledger for that).
- **Test mode**: a test-mode daemon never calls a real provider unless
  `GPU_ROUTER_REAL_PROVIDERS` names it (invariant 20); tests inject an `httpx.MockTransport`.

## Model aliases

The same alias on several providers lets the router choose by quota: `gpt-oss-20b` and
`gpt-oss-120b` (groq, cloudflare, hf), `llama-3.1-8b`, `llama-3.3-70b` (cloudflare, hf),
`gemini-*` (gemini). `gpu infer --list` prints the current map. A provider's own id works
too (`@cf/...`, `gemini-...`, `org/name` on hf), or anything with `-p <provider>`.
