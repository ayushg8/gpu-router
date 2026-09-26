"""Remote side of gpu-router (phase 2). STDLIB ONLY, Python 3.8+ (invariant 14, D2).

`gpu.py` is the user-facing helper (`import gpu` inside a job); `bootstrap.py` is the remote
entrypoint. Both are copied verbatim into every job bundle under `gpu_runner/`, so nothing
here may import `gpu_router` or any third-party package.
"""
