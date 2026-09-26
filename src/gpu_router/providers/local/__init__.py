"""Local Mac provider (phase 3): runs a job bundle on this Mac with PyTorch MPS.

`adapter.LocalAdapter` is what the registry builds (kind "local"). `launcher.py` is the
detached, stdlib-only process that prepares the bundle and the python env and then execs
the bundle's `gpu_runner/bootstrap.py`. Facts and quirks are in NOTES.md next to this file.
Importing this package imports nothing.
"""
