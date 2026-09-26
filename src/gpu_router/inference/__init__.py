"""The inference lane (phase 7b): evals and LLM calls on free inference APIs, separate from
GPU jobs.

    catalog.py   the `inference:` section of providers.yaml, typed (models, limits, resets)
    keys.py      Keychain names per provider (`gpu login <provider>`), presence, verification
    clients.py   one OpenAI-compatible chat client (httpx) + per-provider error / header rules
    ledger.py    the daily-quota ledger (requests / tokens / neurons / usd, live or est)
    router.py    picks a provider by model availability and quota left, with a one-line reason
    service.py   `InferenceService`: the daemon's owner of the ledger and the calls
    models.py    request / result / view models shared by the daemon API and its clients
    batch.py     JSONL eval files: parse lines, shape results (CLI `gpu infer --file`, /infer)

The daemon owns the ledger (invariant 2); the CLI, the shell's /infer and the MCP tool
`gpu_infer` call `POST /v1/infer`. Nothing here imports heavy modules at package import.
"""
