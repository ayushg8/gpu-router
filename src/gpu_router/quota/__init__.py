"""Quota ledger (phase 5): how much free quota each provider has left, live or estimated.

  `windows`  reset windows from providers.yaml (`reset` + `reset_anchor`), pure date math
  `settings` the `routing.quota` knobs of config.yaml (TTL, refresh cadence, rolling window)
  `ledger`   per-provider view = latest live reading (fresh) or an estimate from job history
  `service`  daemon-side cache refresher: calls adapters' quota() in the background

Routing never calls a provider: it reads the ledger view, which is computed from the store
(quota_snapshots, attempts, provider_state). Importing this package imports nothing.
"""
