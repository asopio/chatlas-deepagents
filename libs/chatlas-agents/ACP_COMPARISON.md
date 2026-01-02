# Comparison: ChATLAS ACP vs DeepAgents ACP

This document compares the ChATLAS ACP implementation (`libs/chatlas-agents/chatlas_agents/acp/`) with the upstream DeepAgents ACP implementation (`libs/acp/deepagents_acp/`), documenting differences for future streamlining.

## Overview

| Aspect | DeepAgents ACP | ChATLAS ACP |
|--------|----------------|-------------|
| **Status** | WIP (Work in Progress) | Production-ready |
| **Lines of Code** | 655 lines | 887 lines |
| **Main Class** | `DeepagentsACP` | `ChATLASACP` |
| **Entry Point** | `deepacp` command | `chatlas-acp` command |
| **Configuration** | Hardcoded in `main()` | Full configuration system |

## Core Architectural Differences

### 1. Agent Graph Management

**DeepAgents ACP:**
- Agent graph created once and passed to constructor
- Single shared agent graph for all sessions
- Simple dictionary for session storage
```python
class DeepagentsACP(Agent):
    def __init__(self, connection, agent_graph):
        self._agent_graph = agent_graph  # Shared graph
        self._sessions: dict[str, dict[str, Any]] = {}
```

**ChATLAS ACP:**
- Agent graph created per session (dynamic creation)
- Each session has its own agent graph
- Pydantic model for session state with mode tracking
```python
class ACPSession(BaseModel):
    id: str
    agent: CompiledStateGraph  # Per-session graph
    thread_id: str
    mode: AgentModeType
    task: asyncio.Task[None] | None = None

class ChATLASACP(Agent):
    def __init__(self, connection, config):
        self._config = config  # Configuration-driven
        self._sessions: dict[str, ACPSession] = {}
```

**Impact:** ChATLAS approach allows per-session customization but requires more setup. DeepAgents is simpler but less flexible.

### 2. Configuration System

**DeepAgents ACP:**
- No configuration abstraction
- Hardcoded setup in `main()` function
- Example setup commented out, uses simple test tool
```python
async def main():
    # Commented out CLI integration
    # Hardcoded test setup
    model = ChatAnthropic(model_name="claude-sonnet-4-5-20250929")
    agent_graph = create_deep_agent(model=model, tools=[get_weather])
```

**ChATLAS ACP:**
- Complete `ACPConfig` configuration model
- Environment variable loading
- Setup wizard (`chatlas-acp --setup`)
- Three agent modes (default, auto, research)
```python
class ACPConfig(BaseModel):
    agent_id: str = "chatlas-acp"
    model: str = "gpt-4"
    mcp_url: str = "https://chatlas-mcp.app.cern.ch/mcp"
    sandbox_type: Optional[str] = None
    enable_memory: bool = True
    # ... 10+ configuration options
    
    @classmethod
    def from_env(cls) -> ACPConfig:
        # Load from CHATLAS_ACP_* environment variables
```

**Impact:** ChATLAS is production-ready with full configuration flexibility. DeepAgents needs configuration system added.

### 3. Session Initialization

**DeepAgents ACP:**
- Simple session creation
- No capabilities or modes returned
```python
async def newSession(self, params):
    session_id = str(uuid.uuid4())
    self._sessions[session_id] = {
        "agent": self._agent_graph,
        "thread_id": str(uuid.uuid4()),
    }
    return NewSessionResponse(sessionId=session_id)
```

**ChATLAS ACP:**
- Rich session response with modes and models
- Dynamic agent creation per session
- Returns available modes and models
```python
async def newSession(self, params):
    session_id = str(uuid.uuid4())
    agent_graph = await self._create_agent_graph(params.cwd)
    
    session = ACPSession(
        id=session_id,
        agent=agent_graph,
        thread_id=str(uuid.uuid4()),
        mode=self._config.default_mode,
    )
    
    return NewSessionResponse(
        sessionId=session_id,
        modes=SessionModeState(
            currentModeId=session.mode.value,
            availableModes=[...],  # 3 modes
        ),
        models=SessionModelState(...),
    )
```

**Impact:** ChATLAS provides richer session metadata. DeepAgents could adopt this pattern.

### 4. Agent Integration

**DeepAgents ACP:**
- Direct integration with `create_deep_agent()`
- Commented-out CLI integration
- Simple test tool setup
```python
# Commented out:
# from deepagents_cli.agent import create_agent_with_config
# agent_graph, backend = create_agent_with_config(...)

# Current simple setup:
agent_graph = create_deep_agent(
    model=model,
    tools=[get_weather],
    checkpointer=InMemorySaver(),
)
```

**ChATLAS ACP:**
- Full integration with `deepagents-cli`
- MCP middleware integration
- Sandbox support (Docker, Apptainer)
- Skills and memory enabled
```python
async def _create_agent_graph(self, workdir):
    # Create MCP middleware
    mcp_middleware = await MCPMiddleware.create(mcp_config)
    
    # Setup sandbox if configured
    sandbox_backend = None
    if self._config.sandbox_type:
        sandbox_backend = DockerSandboxBackend(...)
    
    # Create full CLI agent
    agent, backend = create_cli_agent(
        model=llm_model,
        tools=standard_tools,
        sandbox=sandbox_backend,
        additional_middleware=[mcp_middleware],
        enable_memory=True,
        enable_skills=True,
        enable_shell=True,
    )
```

**Impact:** ChATLAS leverages full CLI agent capabilities. DeepAgents would benefit from similar integration.

### 5. Agent Modes

**DeepAgents ACP:**
- No mode concept
- No `setSessionMode()` implementation
```python
async def setSessionMode(self, params):
    return None  # Not implemented
```

**ChATLAS ACP:**
- Three agent modes: default (HITL), auto (auto-approve), research
- Dynamic mode switching
- Auto-approve behavior based on mode
```python
class AgentModeType(str, Enum):
    DEFAULT = "default"  # HITL
    AUTO = "auto"        # Auto-approve
    RESEARCH = "research"  # Info retrieval

async def setSessionMode(self, params):
    session = self._get_session(params.sessionId)
    new_mode = AgentModeType(params.modeId)
    session.mode = new_mode
    return SetSessionModeResponse()
```

**Impact:** ChATLAS modes provide flexible behavior control. DeepAgents could adopt this pattern.

### 6. Permission Handling

**DeepAgents ACP:**
- Permission handling implemented
- Uses review_configs for allowed decisions
- Simple approve/reject logic
```python
async def _handle_interrupt(self, params, interrupt):
    # ... build permission request
    response = await self._connection.requestPermission(request)
    
    if isinstance(outcome, AllowedOutcome):
        if option_id == "allow-once":
            decisions.append({"type": "approve"})
    elif isinstance(outcome, DeniedOutcome):
        decisions.append({"type": "reject", "message": "..."})
```

**ChATLAS ACP:**
- Similar permission handling
- Auto-approve mode bypasses permissions entirely
- Same basic pattern but with mode awareness
```python
async def prompt(self, params):
    auto_approve = self._config.get_auto_approve(session.mode)
    
    while True:
        interrupts = await self._stream_and_handle_updates(...)
        
        if auto_approve:
            # Auto-approve all actions
            all_decisions = [{"type": "approve"} for _ in action_requests]
            stream_input = Command(resume={"decisions": all_decisions})
            continue
        
        # Normal HITL flow
        decisions = await self._handle_interrupt(...)
```

**Impact:** Both implementations are similar. ChATLAS adds auto-approve mode as enhancement.

### 7. Logging and Error Handling

**DeepAgents ACP:**
- Minimal logging
- No structured logging
```python
# No logging infrastructure
```

**ChATLAS ACP:**
- Comprehensive logging throughout
- Structured logger usage
- Verbose mode support
```python
logger = logging.getLogger(__name__)

logger.info(f"Creating new session in directory: {params.cwd}")
logger.info(f"Agent graph created with {len(tools)} tools")
logger.error(f"Failed to connect: {e}", exc_info=True)
```

**Impact:** ChATLAS provides better observability. DeepAgents should add logging.

### 8. Entry Point and CLI

**DeepAgents ACP:**
- Simple `cli_main()` function
- No command-line arguments
- No setup wizard
```python
def cli_main():
    asyncio.run(main())

if __name__ == "__main__":
    cli_main()
```

**ChATLAS ACP:**
- Full CLI with argparse
- `--setup` wizard for configuration
- `--verbose` flag for logging
- Help text and documentation
```python
def parse_arguments():
    parser = argparse.ArgumentParser(...)
    parser.add_argument("--setup", ...)
    parser.add_argument("--verbose", ...)
    return parser.parse_args()

def main():
    args = parse_arguments()
    if args.setup:
        run_setup()
        sys.exit(0)
    config = ACPConfig.from_env()
    run_acp_server(config)
```

**Impact:** ChATLAS provides user-friendly CLI. DeepAgents needs CLI enhancement.

## Similarities (Compatible Patterns)

Both implementations share these core patterns:

1. **Session Update Handling**: Both use identical pattern for streaming updates
   - `_handle_ai_message_chunk()` - Same logic
   - `_handle_completed_tool_calls()` - Same logic
   - `_handle_tool_message()` - Same logic
   - `_handle_todo_update()` - Same logic

2. **Stream Processing**: Identical `_stream_and_handle_updates()` pattern
   - Both iterate over `agent.astream()` with `stream_mode=["messages", "updates"]`
   - Both process `__interrupt__` events
   - Both handle model and tools node updates

3. **Tool Call Tracking**: Same `_tool_calls` dictionary pattern
   - Maps `tool_call_id -> ToolCall` for matching
   - Used to link tool requests with tool results

4. **Basic ACP Protocol**: Same implementation of core methods
   - `initialize()` - Returns protocol version and agent info
   - `prompt()` - Main prompt handling with streaming
   - `cancel()` - Session cancellation
   - `authenticate()` - Returns None (not implemented)
   - `extMethod()` - Raises NotImplementedError

## Missing Features in DeepAgents ACP

Features present in ChATLAS but missing in DeepAgents:

1. **Configuration System**
   - No `ACPConfig` equivalent
   - No environment variable loading
   - No setup wizard

2. **Agent Modes**
   - No mode concept
   - No auto-approve mode
   - No mode switching

3. **Agent Capabilities**
   - No `AgentCapabilities` in `initialize()`
   - No mode/model state in `newSession()`

4. **Dynamic Agent Creation**
   - No per-session agent graphs
   - No runtime configuration

5. **CLI Infrastructure**
   - No command-line argument parsing
   - No setup wizard
   - No verbose logging option

6. **Sandbox Support**
   - No Docker/Apptainer integration
   - No sandbox configuration

7. **MCP Integration**
   - No MCPMiddleware usage
   - No ChATLAS-specific tools

8. **Logging**
   - No structured logging
   - No debug/verbose modes

## Recommendations for Convergence

To streamline both implementations when DeepAgents ACP matures:

### Short-term (ChATLAS can adopt from DeepAgents)

1. **None identified** - DeepAgents ACP is simpler but less featured

### Medium-term (DeepAgents should adopt from ChATLAS)

1. **Configuration System**
   - Add `ACPConfig` model with environment loading
   - Keep it optional/simple for basic use cases

2. **Agent Capabilities**
   - Return `AgentCapabilities` in `initialize()`
   - Include mode/model info in `newSession()`

3. **Logging Infrastructure**
   - Add structured logging
   - Make it optional/configurable

4. **CLI Enhancement**
   - Add argument parsing
   - Support `--verbose` flag
   - Consider setup wizard for production use

### Long-term (Potential shared abstractions)

1. **Base Configuration Class**
   - Create shared `BaseACPConfig` in DeepAgents
   - ChATLAS extends with ChATLAS-specific fields

2. **Session Model**
   - Consider Pydantic model for sessions in DeepAgents
   - Define minimal session state interface

3. **Mode System**
   - Add optional mode support to DeepAgents
   - ChATLAS can use built-in modes

4. **Agent Factory Pattern**
   - Define interface for agent creation
   - Allow custom factory implementations

## Code Portability

### Easy to Port from ChATLAS to DeepAgents

1. **Configuration loading** (`ACPConfig.from_env()`) - 50 lines
2. **Logging setup** - 20 lines
3. **CLI argument parsing** - 40 lines
4. **Agent capabilities** in `initialize()` - 10 lines
5. **Session state model** (`ACPSession`) - 15 lines

### ChATLAS-specific (Not portable)

1. **MCP integration** - ChATLAS-specific
2. **Sandbox backends** - CLI-specific feature
3. **Agent modes** - ChATLAS design choice
4. **Setup wizard** - User experience enhancement

## Version Compatibility

| Feature | DeepAgents ACP | ChATLAS ACP | Compatible? |
|---------|----------------|-------------|-------------|
| ACP Protocol Version | 0.6.0 | 0.6.0 | ✅ Yes |
| Session dict structure | Simple dict | Pydantic model | ⚠️ Different |
| Initialize response | Minimal | With capabilities | ⚠️ Extended |
| NewSession response | Minimal | With modes/models | ⚠️ Extended |
| Stream handling | Core logic | Core logic + logging | ✅ Compatible |
| Permission handling | Core logic | Core logic + auto | ✅ Compatible |
| Tool call tracking | Same pattern | Same pattern | ✅ Compatible |

## Conclusion

**High Compatibility**: The core ACP protocol implementation is highly compatible between both versions. The message streaming, tool call handling, and permission request patterns are nearly identical.

**Divergence Areas**:
1. **Configuration** - ChATLAS has full system, DeepAgents has none
2. **Agent Integration** - ChATLAS uses CLI agent, DeepAgents is minimal
3. **Session Management** - ChATLAS has richer session model
4. **User Experience** - ChATLAS has production-ready CLI

**Convergence Path**:
1. DeepAgents can adopt ChATLAS configuration, logging, and CLI patterns
2. Keep core streaming/tool handling identical (already compatible)
3. Consider shared base classes for configuration and session state
4. ChATLAS-specific features (MCP, sandboxes, modes) remain in ChATLAS

**Risk of Divergence**: Low - Core protocol handling is identical. Differences are in the wrapper/setup code, which is naturally different based on use case.

**Recommendation**: Document this comparison in both repositories. When DeepAgents ACP matures, consider:
1. Extracting common base class for session handling
2. Shared configuration interface
3. ChATLAS extends base implementation with its specific features
