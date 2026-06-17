"""Tests for ChATLAS ACP server implementation."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from acp.schema import (
    AllowedOutcome,
    InitializeRequest,
    NewSessionRequest,
    PromptRequest,
    RequestPermissionRequest,
    RequestPermissionResponse,
    TextContentBlock,
)
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage

from chatlas_agents.acp.config import ACPConfig, AgentModeType
from chatlas_agents.acp.server import ACPSession, ChATLASACP


class FakeAgentSideConnection:
    """Fake ACP connection for testing."""

    def __init__(self) -> None:
        """Initialize the fake connection."""
        self.calls: list[dict[str, Any]] = []
        self.permission_requests: list[RequestPermissionRequest] = []
        self.permission_response: RequestPermissionResponse | None = None

    async def sessionUpdate(self, notification: Any) -> None:
        """Track sessionUpdate calls."""
        self.calls.append(notification)

    async def requestPermission(
        self, request: RequestPermissionRequest
    ) -> RequestPermissionResponse:
        """Track permission requests and return response."""
        self.permission_requests.append(request)
        if self.permission_response:
            return self.permission_response
        # Default: approve
        return RequestPermissionResponse(
            outcome=AllowedOutcome(
                outcome="selected",
                optionId="allow-once",
            )
        )


@pytest.fixture
def acp_config() -> ACPConfig:
    """Create test ACP configuration."""
    return ACPConfig(
        agent_id="test-agent",
        model="gpt-4",
        mcp_url="http://test-mcp.local/mcp",
        mcp_timeout=30,
        verbose=True,
    )


@pytest.fixture
def fake_connection() -> FakeAgentSideConnection:
    """Create fake ACP connection."""
    return FakeAgentSideConnection()


class TestACPConfig:
    """Tests for ACPConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = ACPConfig()

        assert config.agent_id == "chatlas-acp"
        assert config.model == "gpt-4"
        assert config.mcp_url == "https://chatlas-mcp.app.cern.ch/mcp"
        assert config.mcp_timeout == 120
        assert config.enable_memory is True
        assert config.enable_skills is True
        assert config.enable_shell is True
        assert config.default_mode == AgentModeType.DEFAULT

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = ACPConfig(
            agent_id="custom-agent",
            model="gpt-4-turbo",
            mcp_url="http://custom-mcp.local/mcp",
            sandbox_type="docker",
            default_mode=AgentModeType.AUTO,
        )

        assert config.agent_id == "custom-agent"
        assert config.model == "gpt-4-turbo"
        assert config.mcp_url == "http://custom-mcp.local/mcp"
        assert config.sandbox_type == "docker"
        assert config.default_mode == AgentModeType.AUTO

    def test_auto_approve_mode(self) -> None:
        """Test auto-approve mode detection."""
        config = ACPConfig()

        assert config.get_auto_approve(AgentModeType.DEFAULT) is False
        assert config.get_auto_approve(AgentModeType.AUTO) is True
        assert config.get_auto_approve(AgentModeType.RESEARCH) is False

    def test_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test loading configuration from environment."""
        monkeypatch.setenv("CHATLAS_ACP_AGENT_ID", "env-agent")
        monkeypatch.setenv("CHATLAS_ACP_MODEL", "gpt-3.5-turbo")
        monkeypatch.setenv("CHATLAS_MCP_URL", "http://env-mcp.local/mcp")
        monkeypatch.setenv("CHATLAS_ACP_DEFAULT_MODE", "auto")

        config = ACPConfig.from_env()

        assert config.agent_id == "env-agent"
        assert config.model == "gpt-3.5-turbo"
        assert config.mcp_url == "http://env-mcp.local/mcp"
        assert config.default_mode == AgentModeType.AUTO


class TestChATLASACP:
    """Tests for ChATLASACP server."""

    async def test_initialize(
        self, acp_config: ACPConfig, fake_connection: FakeAgentSideConnection
    ) -> None:
        """Test agent initialization."""
        acp = ChATLASACP(fake_connection, acp_config)

        init_request = InitializeRequest(
            protocolVersion="0.6.0",
            clientCapabilities={},
        )

        response = await acp.initialize(init_request)

        assert response.protocolVersion == "0.6.0"
        assert response.agentInfo.name == "@chatlas/chatlas-agents"
        assert response.agentInfo.title == "ChATLAS Agent"
        assert response.agentCapabilities.loadSession is False
        assert response.agentCapabilities.promptCapabilities.embeddedContext is True

    async def test_new_session(
        self, acp_config: ACPConfig, fake_connection: FakeAgentSideConnection
    ) -> None:
        """Test creating a new session."""
        with patch(
            "chatlas_agents.acp.server.ChATLASACP._create_agent_graph"
        ) as mock_create:
            # Mock agent graph creation
            mock_graph = MagicMock()
            mock_create.return_value = mock_graph

            acp = ChATLASACP(fake_connection, acp_config)

            new_session_request = NewSessionRequest(
                cwd="/tmp/test",
                mcpServers=[],
            )

            response = await acp.newSession(new_session_request)

            # Verify response
            assert response.sessionId is not None
            assert response.modes.currentModeId == "default"
            assert len(response.modes.availableModes) == 3
            assert response.models.currentModelId == "gpt-4"

            # Verify session was created
            assert response.sessionId in acp._sessions
            session = acp._sessions[response.sessionId]
            assert session.mode == AgentModeType.DEFAULT
            assert session.agent == mock_graph

    async def test_set_session_mode(
        self, acp_config: ACPConfig, fake_connection: FakeAgentSideConnection
    ) -> None:
        """Test setting session mode."""
        with patch(
            "chatlas_agents.acp.server.ChATLASACP._create_agent_graph"
        ) as mock_create:
            mock_graph = MagicMock()
            mock_create.return_value = mock_graph

            acp = ChATLASACP(fake_connection, acp_config)

            # Create a session
            new_session_response = await acp.newSession(
                NewSessionRequest(cwd="/tmp/test", mcpServers=[])
            )
            session_id = new_session_response.sessionId

            # Change mode to auto
            from acp.schema import SetSessionModeRequest

            mode_response = await acp.setSessionMode(
                SetSessionModeRequest(
                    sessionId=session_id,
                    modeId="auto",
                )
            )

            assert mode_response is not None
            assert acp._sessions[session_id].mode == AgentModeType.AUTO

    async def test_cancel_session(
        self, acp_config: ACPConfig, fake_connection: FakeAgentSideConnection
    ) -> None:
        """Test cancelling a session."""
        with patch(
            "chatlas_agents.acp.server.ChATLASACP._create_agent_graph"
        ) as mock_create:
            mock_graph = MagicMock()
            mock_create.return_value = mock_graph

            acp = ChATLASACP(fake_connection, acp_config)

            # Create a session
            new_session_response = await acp.newSession(
                NewSessionRequest(cwd="/tmp/test", mcpServers=[])
            )
            session_id = new_session_response.sessionId

            # Create a mock task
            mock_task = AsyncMock()
            mock_task.done.return_value = False
            acp._sessions[session_id].task = mock_task

            # Cancel the session
            from acp.schema import CancelNotification

            await acp.cancel(CancelNotification(sessionId=session_id))

            # Verify task was cancelled
            mock_task.cancel.assert_called_once()
            assert acp._sessions[session_id].task is None

    async def test_build_text_prompt(
        self, acp_config: ACPConfig, fake_connection: FakeAgentSideConnection
    ) -> None:
        """Test building text prompt from ACP content blocks."""
        acp = ChATLASACP(fake_connection, acp_config)

        # Test with TextContentBlock objects
        prompt = [
            TextContentBlock(text="Hello", type="text"),
            TextContentBlock(text=" world", type="text"),
        ]

        result = acp._build_text_prompt(prompt)
        assert result == "Hello world"

        # Test with dict-like objects
        prompt_dict = [
            {"text": "Test"},
            {"text": " prompt"},
        ]

        result = acp._build_text_prompt(prompt_dict)
        assert result == "Test prompt"


class TestACPSession:
    """Tests for ACPSession model."""

    def test_session_creation(self) -> None:
        """Test creating an ACPSession."""
        mock_agent = MagicMock()

        session = ACPSession(
            id="test-session",
            agent=mock_agent,
            thread_id="test-thread",
            mode=AgentModeType.DEFAULT,
        )

        assert session.id == "test-session"
        assert session.agent == mock_agent
        assert session.thread_id == "test-thread"
        assert session.mode == AgentModeType.DEFAULT
        assert session.task is None


@pytest.mark.asyncio
class TestACPIntegration:
    """Integration tests for ACP server (require full setup)."""

    @pytest.mark.skip(reason="Requires full agent setup with MCP server")
    async def test_full_prompt_flow(self) -> None:
        """Test complete prompt flow with agent execution.

        This test is skipped by default as it requires:
        - Valid MCP server connection
        - LLM API keys
        - Full agent dependencies
        """
        pass
