"""Configuration for ChATLAS ACP server."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class AgentModeType(str, Enum):
    """Available agent modes for ChATLAS ACP server."""

    DEFAULT = "default"
    """Default mode with human-in-the-loop for destructive operations."""

    AUTO = "auto"
    """Auto-approve mode that skips permission requests."""

    RESEARCH = "research"
    """Research mode focused on information retrieval."""


class ACPConfig(BaseModel):
    """Configuration for ChATLAS ACP server.

    This configuration defines how the ChATLAS ACP server is initialized,
    including LLM settings, MCP integration, sandbox configuration, and
    agent behavior.

    Attributes:
        agent_id: Identifier for the agent instance (affects memory storage)
        workdir: Working directory for agent operations
        model: LLM model identifier (e.g., "gpt-4", "claude-sonnet-4")
        mcp_url: URL of the ChATLAS MCP server
        mcp_timeout: Timeout for MCP server connections in seconds
        sandbox_type: Type of sandbox for code execution ("docker", "apptainer", or None)
        sandbox_image: Container image for sandbox
        enable_memory: Enable persistent conversation memory
        enable_skills: Enable custom skills system
        enable_shell: Enable shell access (local mode only)
        default_mode: Default agent mode for new sessions
        verbose: Enable verbose logging
    """

    agent_id: str = Field(
        default="chatlas-acp",
        description="Agent identifier for memory storage",
    )

    workdir: Path = Field(
        default_factory=lambda: Path.cwd(),
        description="Working directory for agent operations",
    )

    # LLM Configuration
    model: str = Field(
        default="gpt-4",
        description="LLM model identifier",
    )

    # MCP Configuration
    mcp_url: str = Field(
        default="https://chatlas-mcp.app.cern.ch/mcp",
        description="ChATLAS MCP server URL",
    )

    mcp_timeout: int = Field(
        default=120,
        description="MCP server timeout in seconds",
    )

    # Sandbox Configuration
    sandbox_type: Optional[str] = Field(
        default=None,
        description="Sandbox type: 'docker', 'apptainer', or None for local",
    )

    sandbox_image: str = Field(
        default="python:3.13-slim",
        description="Container image for sandbox",
    )

    # Agent Features
    enable_memory: bool = Field(
        default=True,
        description="Enable persistent conversation memory",
    )

    enable_skills: bool = Field(
        default=True,
        description="Enable custom skills system",
    )

    enable_shell: bool = Field(
        default=True,
        description="Enable shell access (local mode only)",
    )

    # Agent Behavior
    default_mode: AgentModeType = Field(
        default=AgentModeType.DEFAULT,
        description="Default agent mode for new sessions",
    )

    verbose: bool = Field(
        default=False,
        description="Enable verbose logging",
    )

    @classmethod
    def from_env(cls) -> ACPConfig:
        """Load configuration from environment variables.

        Environment variables follow the pattern CHATLAS_ACP_<FIELD_NAME>.
        For example: CHATLAS_ACP_MODEL, CHATLAS_ACP_MCP_URL, etc.

        Returns:
            ACPConfig instance loaded from environment
        """
        import os

        return cls(
            agent_id=os.getenv("CHATLAS_ACP_AGENT_ID", "chatlas-acp"),
            workdir=Path(os.getenv("CHATLAS_ACP_WORKDIR", Path.cwd())),
            model=os.getenv("CHATLAS_ACP_MODEL", "gpt-4"),
            mcp_url=os.getenv("CHATLAS_MCP_URL", "https://chatlas-mcp.app.cern.ch/mcp"),
            mcp_timeout=int(os.getenv("CHATLAS_MCP_TIMEOUT", "120")),
            sandbox_type=os.getenv("CHATLAS_ACP_SANDBOX_TYPE"),
            sandbox_image=os.getenv(
                "CHATLAS_ACP_SANDBOX_IMAGE", "python:3.13-slim"
            ),
            enable_memory=os.getenv("CHATLAS_ACP_ENABLE_MEMORY", "true").lower()
            == "true",
            enable_skills=os.getenv("CHATLAS_ACP_ENABLE_SKILLS", "true").lower()
            == "true",
            enable_shell=os.getenv("CHATLAS_ACP_ENABLE_SHELL", "true").lower()
            == "true",
            default_mode=AgentModeType(
                os.getenv("CHATLAS_ACP_DEFAULT_MODE", "default")
            ),
            verbose=os.getenv("CHATLAS_ACP_VERBOSE", "false").lower() == "true",
        )

    def get_auto_approve(self, mode: AgentModeType) -> bool:
        """Get auto-approve setting for a given mode.

        Args:
            mode: The agent mode

        Returns:
            True if the mode should auto-approve tool calls
        """
        return mode == AgentModeType.AUTO
