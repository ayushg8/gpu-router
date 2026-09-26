"""`python -m gpu_router.mcp` = `gpu mcp` (stdio MCP server)."""

import sys

from gpu_router.mcp.server import main

sys.exit(main())
