"""The gpu-router daemon (phase 1; owner: group C).

- __main__.py  `gpu daemon run|status|stop|install|uninstall` (argparse, no typer)
- runtime.py   DaemonRuntime: builds and owns every long-lived object
- app.py       FastAPI app factory: middleware (auth, loopback guard, version header),
               error envelope handlers
- routes.py    /v1 endpoints (CLAUDE.md "Daemon HTTP API")
- auth.py      bearer token file + request guards
- events.py    EventBus: StoreListener fan-out to long-poll waiters and the state.json writer
- server.py    uvicorn runner, signal handling, daemon.json lifecycle
- launchd.py   ~/Library/LaunchAgents plist writer, bootstrap/bootout

Importing this package imports nothing.
"""
