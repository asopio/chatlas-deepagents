"""Example: Running ChATLAS ACP Server

This example demonstrates how to start and configure the ChATLAS ACP server.

The ACP server enables integration with third-party interfaces like:
- IDEs (VSCode, JetBrains, Zed, etc.)
- Chatbot applications
- Custom client applications

The server communicates via stdio using the Agent-Client Protocol.
"""

import asyncio
import logging

from chatlas_agents.acp import ACPConfig, run_acp_server

logging.basicConfig(level=logging.INFO)


async def main():
    """Example: Start ACP server with custom configuration."""
    
    # Option 1: Use default configuration from environment
    print("Starting ChATLAS ACP server with default configuration...")
    print("Configuration will be loaded from environment variables:")
    print("  CHATLAS_ACP_* variables")
    print("  CHATLAS_MCP_* variables")
    print("")
    
    # Option 2: Create custom configuration
    config = ACPConfig(
        agent_id="example-agent",
        model="gpt-4",
        mcp_url="https://chatlas-mcp.app.cern.ch/mcp",
        mcp_timeout=120,
        sandbox_type=None,  # Local execution (no sandbox)
        enable_memory=True,
        enable_skills=True,
        enable_shell=True,
        verbose=True,
    )
    
    print("Custom configuration:")
    print(f"  Agent ID: {config.agent_id}")
    print(f"  Model: {config.model}")
    print(f"  MCP URL: {config.mcp_url}")
    print(f"  Sandbox: {config.sandbox_type or 'None (local)'}")
    print("")
    
    # Start the server
    print("Starting ACP server...")
    print("The server will communicate via stdio.")
    print("Connect an ACP client to interact with the agent.")
    print("")
    
    # Run the server (this blocks until terminated)
    # In production, this is typically called by an ACP client that
    # launches the server as a subprocess
    run_acp_server(config)


if __name__ == "__main__":
    # Note: The ACP server is normally started by an ACP client
    # This example is for demonstration purposes only
    
    print("=" * 60)
    print("ChATLAS ACP Server Example")
    print("=" * 60)
    print("")
    print("IMPORTANT:")
    print("The ACP server communicates via stdin/stdout.")
    print("It should be started by an ACP client (IDE, chatbot app).")
    print("")
    print("For production use, run:")
    print("  chatlas-acp")
    print("")
    print("Or start from your ACP client application.")
    print("=" * 60)
    print("")
    
    # Uncomment to actually start the server:
    # asyncio.run(main())
