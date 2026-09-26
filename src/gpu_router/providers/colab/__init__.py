"""Google Colab adapter (phase 3). See NOTES.md for the verified CLI behaviour.

`adapter.ColabAdapter` drives the official `colab` CLI (google-colab-cli, D3) with a
dedicated session file under `<home>/providers/colab/`, so it never sees or stops sessions
created by anything else (another tool, a human). Importing this package imports nothing.
"""
