"""ChATLAS ACP server implementation.

This module implements the Agent-Client Protocol server for ChATLAS agents,
enabling integration with third-party interfaces like IDEs and chatbot apps.

The implementation follows patterns from:
- DeepAgents ACP: libs/acp/deepagents_acp/server.py
- Mistral Vibe ACP: https://github.com/mistralai/mistral-vibe/tree/main/vibe/acp
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any, Literal

from acp import (
    Agent,
    AgentSideConnection,
    PROTOCOL_VERSION,
    stdio_streams,
)
from acp.schema import (
    AgentCapabilities,
    AgentMessageChunk,
    AgentPlanUpdate,
    AgentThoughtChunk,
    AllowedOutcome,
    CancelNotification,
    ContentToolCallContent,
    DeniedOutcome,
    Implementation,
    InitializeRequest,
    InitializeResponse,
    LoadSessionRequest,
    LoadSessionResponse,
    ModelInfo,
    NewSessionRequest,
    NewSessionResponse,
    PlanEntry,
    PermissionOption,
    PromptCapabilities,
    PromptRequest,
    PromptResponse,
    RequestPermissionRequest,
    SessionModelState,
    SessionModeState,
    SessionNotification,
    SetSessionModeRequest,
    SetSessionModeResponse,
    SetSessionModelRequest,
    SetSessionModelResponse,
    TextContentBlock,
    ToolCall as ACPToolCall,
    ToolCallProgress,
)
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from langchain_core.messages.content import ToolCall
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, Interrupt
from pydantic import BaseModel, ConfigDict

from chatlas_agents import __version__ as CHATLAS_VERSION
from chatlas_agents.acp.config import ACPConfig, AgentModeType

logger = logging.getLogger(__name__)


class ACPSession(BaseModel):
    """Session state for an ACP agent session.

    Attributes:
        id: Unique session identifier
        agent: Compiled LangGraph StateGraph for the agent
        thread_id: Thread ID for agent execution
        mode: Current agent mode (default, auto, research)
        task: Optional asyncio task for concurrent execution
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    agent: CompiledStateGraph
    thread_id: str
    mode: AgentModeType
    task: asyncio.Task[None] | None = None


class ChATLASACP(Agent):
    """ACP Agent implementation for ChATLAS agents.

    This class implements the Agent-Client Protocol for ChATLAS agents,
    providing a bridge between ACP clients (IDEs, chatbot apps) and
    ChATLAS agent functionality (MCP tools, sandboxes, skills, memory).

    The implementation handles:
    - Session management and lifecycle
    - User prompt processing and response streaming
    - Tool call progress updates
    - Permission requests (human-in-the-loop)
    - Plan updates for TODO lists
    - Agent mode switching
    - Model switching

    Attributes:
        config: ACP server configuration
        connection: ACP connection for client communication
        sessions: Active agent sessions by ID
        tool_calls: Mapping of tool call IDs to ToolCall objects
    """

    def __init__(
        self,
        connection: AgentSideConnection,
        config: ACPConfig,
    ) -> None:
        """Initialize the ChATLAS ACP agent.

        Args:
            connection: The ACP connection for communicating with the client
            config: Configuration for the ACP server
        """
        self._connection = connection
        self._config = config
        self._sessions: dict[str, ACPSession] = {}
        # Track tool calls by ID for matching with ToolMessages
        # Maps tool_call_id -> ToolCall TypedDict
        self._tool_calls: dict[str, ToolCall] = {}
        # Cache for agent graphs by configuration
        self._agent_cache: dict[str, CompiledStateGraph] = {}

    async def initialize(
        self,
        params: InitializeRequest,
    ) -> InitializeResponse:
        """Initialize the agent and return capabilities.

        Args:
            params: Initialization request from the client

        Returns:
            InitializeResponse with agent capabilities and info
        """
        logger.info("Initializing ChATLAS ACP server")

        return InitializeResponse(
            protocolVersion=PROTOCOL_VERSION,
            agentCapabilities=AgentCapabilities(
                loadSession=False,  # Session persistence not yet implemented
                promptCapabilities=PromptCapabilities(
                    audio=False,
                    embeddedContext=True,  # Support embedded file context
                    image=False,
                ),
            ),
            agentInfo=Implementation(
                name="@chatlas/chatlas-agents",
                version=CHATLAS_VERSION,
                title="ChATLAS Agent",
            ),
        )

    async def newSession(
        self,
        params: NewSessionRequest,
    ) -> NewSessionResponse:
        """Create a new session with a ChATLAS agent.

        Args:
            params: New session request from the client

        Returns:
            NewSessionResponse with session ID and initial state
        """
        logger.info(f"Creating new session in directory: {params.cwd}")

        session_id = str(uuid.uuid4())
        thread_id = str(uuid.uuid4())

        # Create or retrieve cached agent graph
        agent_graph = await self._create_agent_graph(params.cwd)

        # Store session state
        session = ACPSession(
            id=session_id,
            agent=agent_graph,
            thread_id=thread_id,
            mode=self._config.default_mode,
        )
        self._sessions[session_id] = session

        logger.info(f"Created session {session_id} with mode={session.mode}")

        # Prepare available modes
        available_modes = [
            {
                "modeId": AgentModeType.DEFAULT.value,
                "name": "Default",
                "description": "Default mode with human-in-the-loop for destructive operations",
            },
            {
                "modeId": AgentModeType.AUTO.value,
                "name": "Auto",
                "description": "Auto-approve mode that skips permission requests",
            },
            {
                "modeId": AgentModeType.RESEARCH.value,
                "name": "Research",
                "description": "Research mode focused on information retrieval",
            },
        ]

        # Available models (single model for now, could be extended)
        available_models = [
            ModelInfo(
                modelId=self._config.model,
                name=self._config.model,
            )
        ]

        return NewSessionResponse(
            sessionId=session_id,
            modes=SessionModeState(
                currentModeId=session.mode.value,
                availableModes=available_modes,
            ),
            models=SessionModelState(
                currentModelId=self._config.model,
                availableModels=available_models,
            ),
        )

    async def _create_agent_graph(self, workdir: str) -> CompiledStateGraph:
        """Create or retrieve cached agent graph.

        Args:
            workdir: Working directory for the agent

        Returns:
            Compiled LangGraph StateGraph
        """
        # For now, create a new graph each time
        # TODO: Implement caching based on configuration
        from chatlas_agents.config import MCPServerConfig, load_config_from_env
        from chatlas_agents.middleware import MCPMiddleware
        from chatlas_agents.sandbox import (
            ApptainerSandboxBackend,
            DockerSandboxBackend,
            SandboxBackendType,
        )
        from deepagents_cli.agent import create_cli_agent
        from deepagents_cli.config import create_model, settings
        from deepagents_cli.tools import fetch_url, http_request, web_search

        logger.info(
            f"Creating agent graph with MCP URL: {self._config.mcp_url}"
        )

        # Load MCP middleware
        mcp_config = MCPServerConfig(
            url=self._config.mcp_url,
            timeout=self._config.mcp_timeout,
        )
        mcp_middleware = await MCPMiddleware.create(mcp_config)

        # Setup sandbox if configured
        sandbox_backend = None
        sandbox_type_str = None
        if self._config.sandbox_type:
            sandbox_type_str = self._config.sandbox_type.lower()
            sandbox_type = SandboxBackendType(sandbox_type_str)
            image = self._config.sandbox_image

            if sandbox_type == SandboxBackendType.DOCKER:
                sandbox_backend = DockerSandboxBackend(image=image)
            elif sandbox_type == SandboxBackendType.APPTAINER:
                if not image.startswith(("docker://", "oras://", "library://", "/")):
                    image = f"docker://{image}"
                sandbox_backend = ApptainerSandboxBackend(image=image)

        # Create LLM model
        # Set model in environment for create_model()
        import os

        if self._config.model:
            os.environ["OPENAI_MODEL"] = self._config.model

        llm_model = create_model()

        # Prepare standard tools
        standard_tools = [http_request, fetch_url]
        if settings.has_tavily:
            standard_tools.append(web_search)

        # Create CLI agent
        agent, composite_backend = create_cli_agent(
            model=llm_model,
            assistant_id=self._config.agent_id,
            tools=standard_tools,
            sandbox=sandbox_backend,
            sandbox_type=sandbox_type_str,
            auto_approve=False,  # Managed via ACP permissions
            enable_memory=self._config.enable_memory,
            enable_skills=self._config.enable_skills,
            enable_shell=self._config.enable_shell and (sandbox_backend is None),
            additional_middleware=[mcp_middleware],
        )

        logger.info(
            f"Agent graph created with {len(standard_tools)} standard tools "
            f"+ {len(mcp_middleware.tools)} MCP tools"
        )

        return agent

    def _get_session(self, session_id: str) -> ACPSession:
        """Get session by ID or raise error.

        Args:
            session_id: Session identifier

        Returns:
            ACPSession instance

        Raises:
            RuntimeError: If session not found
        """
        session = self._sessions.get(session_id)
        if session is None:
            raise RuntimeError(f"Session not found: {session_id}")
        return session

    async def prompt(
        self,
        params: PromptRequest,
    ) -> PromptResponse:
        """Handle a user prompt and stream responses.

        Args:
            params: Prompt request from the client

        Returns:
            PromptResponse with stop reason
        """
        session = self._get_session(params.sessionId)

        logger.info(f"Processing prompt for session {params.sessionId}")

        # Extract text from prompt content blocks
        prompt_text = self._build_text_prompt(params.prompt)

        # Stream the agent's response
        agent = session.agent
        thread_id = session.thread_id
        config = {"configurable": {"thread_id": thread_id}}

        # Start with the initial user message
        stream_input: dict[str, Any] | Command = {
            "messages": [{"role": "user", "content": prompt_text}]
        }

        # Determine auto-approve based on mode
        auto_approve = self._config.get_auto_approve(session.mode)

        # Loop until there are no more interrupts
        while True:
            # Stream and collect any interrupts
            interrupts = await self._stream_and_handle_updates(
                params, agent, stream_input, config
            )

            # If no interrupts, we're done
            if not interrupts:
                break

            # If auto-approve mode, automatically approve all interrupts
            if auto_approve:
                all_decisions = []
                for interrupt in interrupts:
                    # Auto-approve all actions in the interrupt
                    interrupt_data = interrupt.value
                    action_requests = interrupt_data.get("action_requests", [])
                    for _ in action_requests:
                        all_decisions.append({"type": "approve"})
                stream_input = Command(resume={"decisions": all_decisions})
                continue

            # Process each interrupt and collect decisions from user
            all_decisions = []
            for interrupt in interrupts:
                decisions = await self._handle_interrupt(params, interrupt)
                all_decisions.extend(decisions)

            # Prepare to resume with the collected decisions
            stream_input = Command(resume={"decisions": all_decisions})

        logger.info(f"Prompt completed for session {params.sessionId}")
        return PromptResponse(stopReason="end_turn")

    def _build_text_prompt(self, acp_prompt: list[Any]) -> str:
        """Build text prompt from ACP content blocks.

        Args:
            acp_prompt: List of content blocks from ACP

        Returns:
            Combined text prompt
        """
        prompt_text = ""
        for block in acp_prompt:
            if hasattr(block, "text"):
                prompt_text += block.text
            elif isinstance(block, dict) and "text" in block:
                prompt_text += block["text"]
        return prompt_text

    async def _stream_and_handle_updates(
        self,
        params: PromptRequest,
        agent: Any,
        stream_input: dict[str, Any] | Command,
        config: dict[str, Any],
    ) -> list[Interrupt]:
        """Stream agent execution and handle updates.

        Args:
            params: Prompt request parameters
            agent: The agent graph to stream from
            stream_input: Input for agent.astream
            config: Agent configuration

        Returns:
            List of interrupts that occurred
        """
        interrupts = []

        async for stream_mode, data in agent.astream(
            stream_input,
            config=config,
            stream_mode=["messages", "updates"],
        ):
            if stream_mode == "messages":
                # Handle streaming message chunks
                message, metadata = data
                if isinstance(message, AIMessageChunk):
                    await self._handle_ai_message_chunk(params, message)

            elif stream_mode == "updates":
                # Handle completed node updates
                for node_name, update in data.items():
                    # Check for interrupts
                    if node_name == "__interrupt__":
                        interrupts.extend(update)
                        continue

                    # Only process model and tools nodes
                    if node_name not in ("model", "tools"):
                        continue

                    # Handle todos from tools node
                    if node_name == "tools" and "todos" in update:
                        todos = update.get("todos", [])
                        if todos:
                            await self._handle_todo_update(params, todos)

                    # Get messages from the update
                    messages = update.get("messages", [])
                    if not messages:
                        continue

                    # Process the last message from this node
                    last_message = messages[-1]

                    # Handle completed AI messages from model node
                    if node_name == "model" and isinstance(last_message, AIMessage):
                        if last_message.tool_calls:
                            await self._handle_completed_tool_calls(
                                params, last_message
                            )

                    # Handle tool execution results from tools node
                    elif node_name == "tools" and isinstance(
                        last_message, ToolMessage
                    ):
                        tool_call = self._tool_calls.get(last_message.tool_call_id)
                        if tool_call:
                            await self._handle_tool_message(
                                params, tool_call, last_message
                            )

        return interrupts

    async def _handle_ai_message_chunk(
        self,
        params: PromptRequest,
        message: AIMessageChunk,
    ) -> None:
        """Handle an AIMessageChunk and send appropriate notifications.

        Args:
            params: Prompt request parameters
            message: AIMessageChunk from streaming response
        """
        for block in message.content_blocks:
            block_type = block.get("type")

            if block_type == "text":
                text = block.get("text", "")
                if not text:
                    continue
                await self._connection.sessionUpdate(
                    SessionNotification(
                        update=AgentMessageChunk(
                            content=TextContentBlock(text=text, type="text"),
                            sessionUpdate="agent_message_chunk",
                        ),
                        sessionId=params.sessionId,
                    )
                )
            elif block_type == "reasoning":
                reasoning = block.get("reasoning", "")
                if not reasoning:
                    continue
                await self._connection.sessionUpdate(
                    SessionNotification(
                        update=AgentThoughtChunk(
                            content=TextContentBlock(text=reasoning, type="text"),
                            sessionUpdate="agent_thought_chunk",
                        ),
                        sessionId=params.sessionId,
                    )
                )

    async def _handle_completed_tool_calls(
        self,
        params: PromptRequest,
        message: AIMessage,
    ) -> None:
        """Handle completed tool calls from an AIMessage.

        Args:
            params: Prompt request parameters
            message: AIMessage containing tool_calls
        """
        if not message.tool_calls:
            return

        for tool_call in message.tool_calls:
            tool_call_id = tool_call["id"]
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            if tool_call_id is None:
                continue

            # Skip todo tool calls (handled separately)
            if tool_name == "todo":
                continue

            # Send tool call progress update
            await self._connection.sessionUpdate(
                SessionNotification(
                    update=ToolCallProgress(
                        sessionUpdate="tool_call_update",
                        toolCallId=tool_call_id,
                        title=tool_name,
                        rawInput=tool_args,
                        status="pending",
                    ),
                    sessionId=params.sessionId,
                )
            )

            # Store the tool call for later matching
            self._tool_calls[tool_call_id] = tool_call

    async def _handle_tool_message(
        self,
        params: PromptRequest,
        tool_call: ToolCall,
        message: ToolMessage,
    ) -> None:
        """Handle a ToolMessage and send appropriate notifications.

        Args:
            params: Prompt request parameters
            tool_call: Original ToolCall that this message responds to
            message: ToolMessage containing tool execution result
        """
        # Determine status
        status: Literal["completed", "failed"] = "completed"
        if hasattr(message, "status") and message.status == "error":
            status = "failed"

        # Build content blocks
        content_blocks = []
        for content_block in message.content_blocks:
            if content_block.get("type") == "text":
                text = content_block.get("text", "")
                if text:
                    content_blocks.append(
                        ContentToolCallContent(
                            type="content",
                            content=TextContentBlock(text=text, type="text"),
                        )
                    )

        # Send tool call progress update
        await self._connection.sessionUpdate(
            SessionNotification(
                update=ToolCallProgress(
                    sessionUpdate="tool_call_update",
                    toolCallId=message.tool_call_id,
                    title=tool_call["name"],
                    content=content_blocks,
                    rawOutput=message.content,
                    status=status,
                ),
                sessionId=params.sessionId,
            )
        )

    async def _handle_todo_update(
        self,
        params: PromptRequest,
        todos: list[dict[str, Any]],
    ) -> None:
        """Handle todo list updates.

        Args:
            params: Prompt request parameters
            todos: List of todo dictionaries
        """
        entries = []
        for todo in todos:
            content = todo.get("content", "")
            status = todo.get("status", "pending")

            if status not in ("pending", "in_progress", "completed"):
                status = "pending"

            entry = PlanEntry(
                content=content,
                status=status,  # type: ignore
                priority="medium",
            )
            entries.append(entry)

        await self._connection.sessionUpdate(
            SessionNotification(
                update=AgentPlanUpdate(
                    sessionUpdate="plan",
                    entries=entries,
                ),
                sessionId=params.sessionId,
            )
        )

    async def _handle_interrupt(
        self,
        params: PromptRequest,
        interrupt: Interrupt,
    ) -> list[dict[str, Any]]:
        """Handle a LangGraph interrupt and request permission.

        Args:
            params: Prompt request parameters
            interrupt: Interrupt from LangGraph

        Returns:
            List of decisions to pass to Command(resume={...})
        """
        interrupt_data = interrupt.value
        action_requests = interrupt_data.get("action_requests", [])
        review_configs = interrupt_data.get("review_configs", [])

        # Create mapping of action names to allowed decisions
        allowed_decisions_map = {}
        for review_config in review_configs:
            action_name = review_config.get("action_name")
            allowed_decisions = review_config.get("allowed_decisions", [])
            allowed_decisions_map[action_name] = allowed_decisions

        # Collect decisions for all action requests
        decisions = []

        for action_request in action_requests:
            tool_name = action_request.get("name")
            tool_args = action_request.get("args", {})

            # Get allowed decisions
            allowed_decisions = allowed_decisions_map.get(
                tool_name, ["approve", "reject"]
            )

            # Build permission options
            options = []
            if "approve" in allowed_decisions:
                options.append(
                    PermissionOption(
                        optionId="allow-once",
                        name="Allow once",
                        kind="allow_once",
                    )
                )
            if "reject" in allowed_decisions:
                options.append(
                    PermissionOption(
                        optionId="reject-once",
                        name="Reject",
                        kind="reject_once",
                    )
                )

            # Generate tool call ID
            tool_call_id = f"perm_{uuid.uuid4().hex[:8]}"

            # Create ACP ToolCall
            acp_tool_call = ACPToolCall(
                toolCallId=tool_call_id,
                title=tool_name,
                rawInput=tool_args,
                status="pending",
            )

            # Request permission from client
            response = await self._connection.requestPermission(
                RequestPermissionRequest(
                    sessionId=params.sessionId,
                    toolCall=acp_tool_call,
                    options=options,
                )
            )

            # Convert ACP response to LangGraph decision
            outcome = response.outcome

            if isinstance(outcome, AllowedOutcome):
                option_id = outcome.optionId
                if option_id == "allow-once":
                    decisions.append({"type": "approve"})
                elif option_id == "edit":
                    # Edit option - for now, just approve
                    # TODO: Implement actual edit functionality
                    decisions.append({"type": "approve"})
            elif isinstance(outcome, DeniedOutcome):
                decisions.append(
                    {
                        "type": "reject",
                        "message": "Action rejected by user",
                    }
                )

        return decisions

    async def cancel(self, params: CancelNotification) -> None:
        """Cancel a running session.

        Args:
            params: Cancel notification from client
        """
        session = self._get_session(params.sessionId)
        if session.task and not session.task.done():
            session.task.cancel()
            session.task = None
            logger.info(f"Cancelled session {params.sessionId}")

    async def loadSession(
        self,
        params: LoadSessionRequest,
    ) -> LoadSessionResponse | None:
        """Load an existing session (not implemented).

        Args:
            params: Load session request

        Returns:
            None (not implemented)
        """
        # TODO: Implement session persistence
        return None

    async def setSessionMode(
        self,
        params: SetSessionModeRequest,
    ) -> SetSessionModeResponse | None:
        """Set session mode.

        Args:
            params: Set session mode request

        Returns:
            SetSessionModeResponse if successful, None otherwise
        """
        session = self._get_session(params.sessionId)

        try:
            new_mode = AgentModeType(params.modeId)
        except ValueError:
            logger.warning(f"Invalid mode ID: {params.modeId}")
            return None

        session.mode = new_mode
        logger.info(f"Session {params.sessionId} mode set to {new_mode}")

        return SetSessionModeResponse()

    async def setSessionModel(
        self,
        params: SetSessionModelRequest,
    ) -> SetSessionModelResponse | None:
        """Set session model (not supported).

        Args:
            params: Set session model request

        Returns:
            None (not supported - model is fixed at server start)
        """
        # Model is configured at server start and cannot be changed per session
        logger.warning("setSessionModel not supported - model is fixed at server start")
        return None

    async def authenticate(self, params: Any) -> Any | None:
        """Authenticate (optional, not implemented).

        Args:
            params: Authentication parameters

        Returns:
            None (authentication not required)
        """
        return None

    async def extMethod(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """Handle extension methods (not implemented).

        Args:
            method: Extension method name
            params: Method parameters

        Raises:
            NotImplementedError: Always, as no extension methods are supported
        """
        raise NotImplementedError(f"Extension method {method} not supported")

    async def extNotification(self, method: str, params: dict[str, Any]) -> None:
        """Handle extension notifications (not implemented).

        Args:
            method: Extension notification name
            params: Notification parameters
        """
        pass


async def _run_acp_server(config: ACPConfig | None = None) -> None:
    """Run the ACP server with given configuration.

    Args:
        config: Optional ACP configuration (defaults to loading from environment)
    """
    if config is None:
        config = ACPConfig.from_env()

    logger.info(f"Starting ChATLAS ACP server v{CHATLAS_VERSION}")
    logger.info(f"Configuration: {config.model_dump_json(indent=2)}")

    reader, writer = await stdio_streams()
    AgentSideConnection(lambda conn: ChATLASACP(conn, config), writer, reader)
    await asyncio.Event().wait()


def run_acp_server(config: ACPConfig | None = None) -> None:
    """Synchronous entry point for the ACP server.

    Args:
        config: Optional ACP configuration (defaults to loading from environment)
    """
    try:
        asyncio.run(_run_acp_server(config))
    except KeyboardInterrupt:
        logger.info("ACP server terminated by user")
    except Exception as e:
        logger.error(f"ACP server error: {e}", exc_info=True)
        raise
