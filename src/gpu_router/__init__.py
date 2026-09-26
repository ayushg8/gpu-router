"""gpu-router: route GPU jobs across free-tier cloud providers behind one local daemon.

Invariant 15: this file imports nothing. `gpu status --line` imports the package on its
<50ms fast path, so any import added here is paid by every status-line render.
"""

__version__ = "0.1.0"
