"""Claude Code status line (phase 6b).

- `fast.py`     `gpu status --line`: 0-2 gpu rows from state.json (stdlib only, <50 ms).
- `samples.py`  sample snapshots for every row state (preview + golden tests).
- `install.py`  `gpu statusline install|uninstall|status`: wrap the user's status line.
- `cli.py`      argparse front end for `gpu statusline ...` (dispatched by entry.py).
- `gpu-statusline.sh`  the wrapper Claude Code runs (same bytes as plugin/statusline/).

This file imports nothing: `gpu status --line` imports the package on its fast path.
"""
