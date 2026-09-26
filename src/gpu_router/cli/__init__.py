"""Scriptable CLI: `gpu run / route / status / jobs / logs / cancel / fetch / approve / deny /
quota / history / providers / daemon` (phase 2, Typer + rich).

Reached from `entry.py` for any argv that is not `status --line`, `daemon ...` or empty.
`app.py` holds the commands; `render.py` the human output (spec UX principle 6: green
running, yellow waiting, red failed, dim idle; icons ⚡ ⏸ ✓ ✗ ↪); `exitcodes.py` the exit
codes. Every command takes `--json`; the JSON shapes and exit codes are documented in
docs/cli.md and are stable (additive-only, like API v1).

Importing this package imports nothing heavy; `app.py` imports typer and rich.
"""
