---
description: Run a script on a free cloud GPU through gpu-router
argument-hint: "[--hours H] [--vram GB] [--smoke] <script> [script args...]"
---

Run a script on a free GPU with gpu-router. The user's request: `$ARGUMENTS`

Follow the gpu-router skill. Steps:

1. Parse the request. gpu options come before the script: `--hours H`, `--vram GB`,
   `--gpu TYPE`, `--provider NAME`, `--smoke`, `--name NAME`. The first other token is the
   script (a path relative to the project); every token after it is a script argument.
   With no script, use gpu.yaml's `script:` (read gpu.yaml at the project root first).
2. `project_dir` is the absolute path of the current project (the git root, or the
   working directory when there is no repo).
3. When no `--hours` was given, estimate the runtime honestly from the script (epochs,
   dataset size, model size) and pass it as `hours`, saying the estimate in one line. If
   it cannot be estimated, submit without it: the user will be asked to approve. A job
   still running well past its hours is stopped and waits for the user's approval.
4. Call the `gpu_submit` tool (from the gpu-router MCP server) with those values. If the
   gpu-router MCP tools are unavailable, say that the gpu-router MCP server is not running
   (`claude mcp list`; it needs `gpu` on PATH) and stop. Do not submit with the `gpu` CLI.
5. Report in two or three lines: job short id, where it runs (provider, GPU) and the
   route reason. If `guidance.needs_approval` is true, show `guidance.tell_user` word for
   word and end your turn: only the user approves, with `/gpu-approve <id>`.
6. Follow `guidance.follow`: `wait` (about 15 minutes or less) = keep following it with
   `gpu_status(ref, wait_s=50)` until it finishes, then report the result, the last
   metrics and the outputs directory; `report_and_stop` = say how to check later
   (`/gpu-status <id>`) and stop.

Never call kaggle, colab or lightning directly (nor the colab skill), and never
approve the job yourself.
