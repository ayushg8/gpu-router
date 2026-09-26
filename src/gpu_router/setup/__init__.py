"""`gpu setup`: the first-run wizard (phase 8b; spec "Setup and quality bar", UX principle 1).

Steps, in order (each one detects what is already done and skips it; `--only` picks some):

1. tools        install missing CLIs with `uv tool install` (asks once for the batch)
2. logins       kaggle (kaggle.json -> Keychain), colab (ADC + the exact gcloud command),
                lightning (Keychain), Hugging Face token (Keychain)
3. launchd      the daemon's launchd agent (asks)
4. integration  Claude Code status line, Claude Code plugin, Codex MCP entry, the global
                colab skill (each shows exactly what changes and asks; default no)
5. check        `gpu doctor`, a 10-second GPU smoke test per ready provider (asks: it
                spends a little quota), then "N providers ready, ~X free hrs/month"

Modules: `firstrun` (stdlib only: does bare `gpu` launch the wizard?), `state`
(`<home>/setup.json`: answers, outcomes, smoke results; resumable), `context` (everything
injectable: user home, environ, subprocess runner, launchctl, prompts), `ui`, one module per
step (`tools`, `logins`, `service`, `integrations`, `check`), `wizard` (order, `--only`,
`--yes`, resume, exit codes) and `cli` (`gpu setup`, `gpu login kaggle|colab`, bare
`gpu login`).

This package imports nothing at import time (entry.py imports `firstrun` on the bare-`gpu`
path, invariant 14).
"""
