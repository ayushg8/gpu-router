"""MCP server for agents (phase 6): `gpu mcp` serves gpu_submit, gpu_status, gpu_logs,
gpu_fetch, gpu_cancel, gpu_quota and gpu_route on stdio.

`tools.py` holds the tool logic (plain functions over `gpu_router.client`, no MCP imports);
`server.py` registers them with FastMCP and runs the stdio transport. There is no approve
tool on purpose: only a human approves a job (`/gpu-approve` or `gpu approve`).
"""
