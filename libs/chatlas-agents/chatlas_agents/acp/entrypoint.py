"""CLI entrypoint for ChATLAS ACP server.

This module provides the command-line interface for starting the ChATLAS ACP server.
"""

from __future__ import annotations

import argparse
import logging
import sys

from rich.console import Console
from rich.logging import RichHandler

from chatlas_agents import __version__ as CHATLAS_VERSION
from chatlas_agents.acp.config import ACPConfig
from chatlas_agents.acp.server import run_acp_server

# Configure line buffering for subprocess communication
sys.stdout.reconfigure(line_buffering=True)  # type: ignore
sys.stderr.reconfigure(line_buffering=True)  # type: ignore
sys.stdin.reconfigure(line_buffering=True)  # type: ignore

console = Console()


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration.

    Args:
        verbose: Enable verbose (DEBUG level) logging
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="ChATLAS Agent-Client Protocol (ACP) Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment Variables:
  CHATLAS_ACP_AGENT_ID       Agent identifier (default: chatlas-acp)
  CHATLAS_ACP_WORKDIR        Working directory (default: current directory)
  CHATLAS_ACP_MODEL          LLM model (default: gpt-4)
  CHATLAS_MCP_URL            MCP server URL
  CHATLAS_MCP_TIMEOUT        MCP timeout in seconds (default: 120)
  CHATLAS_ACP_SANDBOX_TYPE   Sandbox type: docker, apptainer, or none
  CHATLAS_ACP_SANDBOX_IMAGE  Container image (default: python:3.13-slim)
  CHATLAS_ACP_ENABLE_MEMORY  Enable memory (default: true)
  CHATLAS_ACP_ENABLE_SKILLS  Enable skills (default: true)
  CHATLAS_ACP_ENABLE_SHELL   Enable shell (default: true)
  CHATLAS_ACP_DEFAULT_MODE   Default mode: default, auto, research
  CHATLAS_ACP_VERBOSE        Enable verbose logging (default: false)

Examples:
  # Start ACP server with default configuration
  chatlas-acp

  # Start with verbose logging
  chatlas-acp --verbose

  # Configure via environment variables
  export CHATLAS_ACP_MODEL=gpt-4-turbo
  export CHATLAS_MCP_URL=https://chatlas-mcp.app.cern.ch/mcp
  chatlas-acp
        """,
    )

    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"chatlas-acp {CHATLAS_VERSION}",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    parser.add_argument(
        "--setup",
        action="store_true",
        help="Run initial setup and configuration wizard",
    )

    return parser.parse_args()


def run_setup() -> None:
    """Run setup wizard to configure ChATLAS ACP server.

    This creates a .env file with configuration variables.
    """
    console.print("[bold]ChATLAS ACP Server Setup[/bold]\n")

    config_content = """# ChATLAS ACP Server Configuration
# Environment variables for chatlas-acp

# Agent Configuration
CHATLAS_ACP_AGENT_ID=chatlas-acp
# CHATLAS_ACP_WORKDIR=/path/to/workdir

# LLM Configuration
CHATLAS_ACP_MODEL=gpt-4
# OPENAI_API_KEY=your-api-key-here
# ANTHROPIC_API_KEY=your-api-key-here

# MCP Server Configuration
CHATLAS_MCP_URL=https://chatlas-mcp.app.cern.ch/mcp
CHATLAS_MCP_TIMEOUT=120

# Sandbox Configuration (optional)
# CHATLAS_ACP_SANDBOX_TYPE=docker
# CHATLAS_ACP_SANDBOX_IMAGE=python:3.13-slim

# Agent Features
CHATLAS_ACP_ENABLE_MEMORY=true
CHATLAS_ACP_ENABLE_SKILLS=true
CHATLAS_ACP_ENABLE_SHELL=true

# Agent Behavior
CHATLAS_ACP_DEFAULT_MODE=default
# CHATLAS_ACP_VERBOSE=false
"""

    output_file = ".env.chatlas-acp"
    with open(output_file, "w") as f:
        f.write(config_content)

    console.print(f"[green]✓ Created configuration file: {output_file}[/green]\n")
    console.print("[dim]Next steps:[/dim]")
    console.print(f"[dim]1. Edit {output_file} and add your API keys[/dim]")
    console.print(f"[dim]2. Load the configuration:[/dim]")
    console.print(f"[dim]   export $(cat {output_file} | xargs)[/dim]")
    console.print("[dim]3. Start the ACP server:[/dim]")
    console.print("[dim]   chatlas-acp[/dim]\n")
    console.print("[yellow]Note: The ACP server communicates via stdio.[/yellow]")
    console.print(
        "[yellow]It should be started by an ACP client (IDE, chatbot app).[/yellow]"
    )


def main() -> None:
    """Main entry point for the chatlas-acp command."""
    args = parse_arguments()

    # Setup logging
    setup_logging(verbose=args.verbose)

    # Handle setup command
    if args.setup:
        run_setup()
        sys.exit(0)

    # Load configuration from environment
    config = ACPConfig.from_env()

    # Start ACP server
    run_acp_server(config)


if __name__ == "__main__":
    main()
