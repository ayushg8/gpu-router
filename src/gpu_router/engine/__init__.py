"""The job engine (phase 1; owner: group C).

- deps.py        EngineDeps: the collaborators every engine piece receives
- backoff.py     retry/cooldown schedule
- calls.py       AdapterCaller: runs blocking adapter calls in worker threads, with timeouts,
                 per-provider concurrency and contract enforcement (invariants 7, 9)
- capture.py     LogCapture: attempt log files, redaction, protocol/metric parsing
- driver.py      JobDriver: one asyncio task per non-terminal job; the state machine in motion
- supervisor.py  Supervisor: recovery on start, owns drivers, user actions, provider health
- crashpoints.py test-mode crash injection for the crash-recovery harness

Importing this package imports nothing.
"""
