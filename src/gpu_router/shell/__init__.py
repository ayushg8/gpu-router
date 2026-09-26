"""Interactive Textual shell (phase 4): `gpu` with no arguments in a terminal.

A thin client of the daemon's HTTP API (invariant 2), like the CLI whose logic it reuses:

- `app.py`       GpuShell (layout, keys, prompt, popup, transcript) and `run()`.
- `feed.py`      background poller: status, quota, metric history -> `Snapshot`.
- `commands.py`  slash commands; each reuses gpu_router.cli helpers and render functions.
- `complete.py`  popup + tab completion for commands, job ids, scripts, providers.
- `metrics.py`   metric history parsed from `::gpu::` lines (stdout fallback), sparklines.
- `panel.py`     the live job panel and the footer status bar (pure formatting).
- `chart.py`     the /watch chart (pure formatting).
- `widgets.py`   transcript blocks, live /logs and /watch blocks, the popup.

Nothing here opens SQLite or calls a provider.
"""
