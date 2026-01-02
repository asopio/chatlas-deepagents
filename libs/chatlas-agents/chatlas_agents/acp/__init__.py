"""Agent-Client Protocol (ACP) support for ChATLAS agents.

This module provides ACP server implementation for ChATLAS agents,
enabling integration with third-party interfaces like IDEs and chatbot apps.

The ACP implementation follows the same middleware-based pattern as MCP integration,
keeping all new code in the chatlas_agents package without modifying upstream packages.

Example usage:
    # Start ACP server from command line
    $ chatlas-acp
    
    # Or programmatically
    from chatlas_agents.acp import run_acp_server
    
    run_acp_server()
"""

from chatlas_agents.acp.server import ChATLASACP, run_acp_server
from chatlas_agents.acp.config import ACPConfig

__all__ = [
    "ChATLASACP",
    "run_acp_server",
    "ACPConfig",
]
