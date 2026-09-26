---
description: Preview the gpu rows for your status line and see whether they are installed (you type this, not Claude)
disable-model-invocation: true
allowed-tools: Bash(gpu statusline status:*), Bash(gpu statusline preview:*)
---

The user typed `/gpu-statusline`. Show them what gpu-router adds to their Claude Code
status line and whether it is installed. This command only looks; it changes nothing.

Steps:

1. Run `gpu statusline status`.
2. Run `gpu statusline preview --plain --state running --state approval --state finished`
   and show its output in a code block, unchanged.
3. Answer in two or three lines:
   - Not installed: the rows appear only while a GPU job is active, under the user's own
     lines, which never change. To add them, the user runs `gpu statusline install` in
     their own terminal: it prints the settings.json diff and asks `[y/N]` before writing
     anything. Undo with `gpu statusline uninstall`.
   - Installed: say which command it wraps, and that `gpu statusline uninstall` restores
     the original exactly.

Never run `gpu statusline install` or `gpu statusline uninstall` yourself (with or without
`--yes`), and never edit `settings.json`: changing the status line is the user's decision,
made in their terminal.
