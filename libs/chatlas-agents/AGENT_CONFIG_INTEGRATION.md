# Agent Configuration Comparison & Integration Guide

## Overview

This document explains how the benchmark agent configuration compares to the default ChATLAS CLI agent, and how to integrate optimized configurations back into the CLI and DeepAgents framework.

## Agent Configuration Comparison

### 1. Default ChATLAS CLI Agent (Interactive Mode)

**Location**: `chatlas_agents/cli.py` → `_run_interactive_session()`

**Configuration Sources**:
- Environment variables via `Settings` class (prefix: `CHATLAS_`)
- YAML config files via `load_config_from_yaml()`
- Command-line overrides (model, MCP URL, timeout, etc.)

**Agent Creation Flow**:
```python
# 1. Load configuration
config = load_config_from_env()  # or load_config_from_yaml()

# 2. Create LLM
model = create_llm_from_config(config.llm)

# 3. Create MCP middleware
mcp_middleware = await MCPMiddleware.create(config.mcp)

# 4. Create agent via deepagents-cli
agent, backend = create_cli_agent(
    model=model,
    assistant_id="chatlas-agent",
    tools=[],  # Additional tools like web_search, fetch_url
    sandbox=sandbox_backend,  # Optional: Docker/Apptainer
    sandbox_type=sandbox_type_str,
)
```

**Key Components**:
- **Model**: From `config.llm` (provider, model name, temperature, etc.)
- **Middleware**: MCPMiddleware (for ChATLAS tools) + built-in DeepAgents middleware
  - TodoListMiddleware (planning)
  - FilesystemMiddleware (file operations)
  - SubAgentMiddleware (task delegation)
  - ShellMiddleware (command execution)
  - SkillsMiddleware (custom skills)
  - AgentMemoryMiddleware (conversation persistence)
- **System Prompt**: Default coding instructions + agent.md content + sandbox-specific paths
- **Tools**: MCP tools + web_search, fetch_url, http_request
- **Backend**: FilesystemBackend or SandboxBackend (Docker/Apptainer)
- **Checkpointer**: MemorySaver for conversation state

### 2. Benchmark Evaluation Agent

**Location**: `chatlas_agents/benchmark/evaluate.py` → `run_benchmark()`

**Configuration Sources**:
- YAML config file (same as CLI)
- Environment variables (same Settings class)
- Command-line arguments specific to benchmarking

**Agent Creation Flow**:
```python
# 1. Load configuration (identical to CLI)
config = load_config_from_yaml(str(agent_config_file))  # or load_config_from_env()

# 2. Create LLM (identical to CLI)
agent_llm = create_llm_from_config(config.llm)

# 3. Create MCP middleware (identical to CLI)
mcp_middleware = await MCPMiddleware.create(config.mcp)

# 4. Create agent via DeepAgents directly
agent = create_deep_agent(
    model=agent_llm,
    middleware=[mcp_middleware],  # Only MCP, gets TodoList/Filesystem/SubAgent automatically
    system_prompt="You are a helpful AI assistant for ATLAS experiment documentation and queries.",
)
```

**Key Differences from CLI**:
- Uses `deepagents.create_deep_agent()` directly, not `deepagents_cli.create_cli_agent()`
- Simpler middleware stack (only MCPMiddleware added explicitly)
- Generic system prompt instead of coding-specific instructions
- No sandbox integration
- No checkpointer (stateless evaluation)
- No shell/skills/memory middleware
- Decorated with `@track` for Opik tracing

### Side-by-Side Comparison

| Aspect | CLI Agent | Benchmark Agent |
|--------|-----------|-----------------|
| **Creation Function** | `create_cli_agent()` | `create_deep_agent()` |
| **LLM Config** | From `config.llm` | From `config.llm` |
| **MCP Tools** | Via MCPMiddleware | Via MCPMiddleware |
| **System Prompt** | Coding-focused + agent.md | Generic ATLAS assistant |
| **Middleware** | MCP + Shell + Skills + Memory + Todo + FS + SubAgent | MCP + (auto: Todo + FS + SubAgent) |
| **Tools** | MCP + web_search + fetch_url + http_request | MCP only |
| **Backend** | Filesystem or Sandbox | Default (StateBackend) |
| **Checkpointer** | MemorySaver | None |
| **Sandbox** | Optional Docker/Apptainer | Not used |
| **Tracing** | None | @track decorator |
| **Use Case** | Interactive development | Automated evaluation |

## How DeepAgents Configuration Works

Based on LangChain/DeepAgents documentation research:

### `create_deep_agent()` Parameters

```python
from deepagents import create_deep_agent

agent = create_deep_agent(
    model="claude-sonnet-4-5-20250929",  # or BaseChatModel instance
    tools=[custom_tool_1, custom_tool_2],  # Additional tools beyond built-ins
    system_prompt="Your custom instructions",  # Appended to default prompt
    middleware=[CustomMiddleware()],  # Additional middleware (auto middleware still added)
    subagents=[research_subagent, code_subagent],  # Custom subagent definitions
    interrupt_on={"tool_name": {"allowed_decisions": [...]}},  # HITL config
    backend=FilesystemBackend(root_dir="/path"),  # Storage backend
)
```

### Automatic Middleware

**DeepAgents always adds** (from documentation):
1. **TodoListMiddleware**: Planning with `write_todos` and `read_todos` tools
2. **FilesystemMiddleware**: File operations (`ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`, `execute`*)
3. **SubAgentMiddleware**: Task delegation with `task` tool

*`execute` only available if backend implements `SandboxBackendProtocol`

### Middleware Architecture

From LangChain docs, middleware provides:
- **Tool injection**: Add tools dynamically
- **System prompt modification**: Inject instructions about tools
- **Request/response hooks**: `beforeModel()` and `afterModel()` methods
- **State management**: Access and modify agent state
- **Dynamic behavior**: Change agent behavior based on context

### Custom Middleware Pattern

```python
from langchain.agents.middleware import AgentMiddleware

class CustomMetricsMiddleware(AgentMiddleware):
    """Track agent metrics during execution."""
    
    def before_agent(self, state):
        """Called before agent execution."""
        # Add custom state, inject tools, modify prompt
        return state
    
    def wrap_model_call(self, request, handler):
        """Intercept model calls."""
        # Log, modify request, track metrics
        response = handler(request)
        # Process response
        return response
```

## Integrating Optimized Configuration

### Strategy 1: YAML Configuration Files

**Best for**: Storing optimized agent configurations for reuse

**Implementation**:

1. **After benchmark optimization**, save the best-performing config:

```python
# In benchmark evaluation
best_config = {
    "llm": {
        "provider": "anthropic",
        "model": "claude-3-5-sonnet-20241022",
        "temperature": 0.5,  # Optimized via benchmarking
        "max_tokens": 4096,
    },
    "mcp": {
        "url": "https://chatlas-mcp.app.cern.ch/mcp",
        "timeout": 180,  # Increased based on benchmarks
    },
    "agent": {
        "name": "chatlas-optimized",
        "max_iterations": 15,  # Tuned for complex queries
    }
}

# Save to YAML
import yaml
with open("configs/optimized-agent.yaml", "w") as f:
    yaml.dump(best_config, f)
```

2. **Use in CLI**:

```bash
chatlas --config configs/optimized-agent.yaml
```

3. **Use in benchmark**:

```bash
chatlas benchmark \\
  --csv-file benchmarks/test.csv \\
  --config configs/optimized-agent.yaml
```

### Strategy 2: Environment Variable Defaults

**Best for**: Organization-wide defaults

**Implementation**:

1. **Update Settings defaults** in `chatlas_agents/config/__init__.py`:

```python
class Settings(BaseSettings):
    # Update defaults based on benchmark results
    llm_provider: str = Field(default="anthropic")  # Changed from "openai"
    llm_model: str = Field(default="claude-3-5-sonnet-20241022")  # Optimized model
    llm_temperature: float = Field(default=0.5)  # Reduced for accuracy
    
    mcp_timeout: int = Field(default=180)  # Increased based on benchmarks
    
    agent_max_iterations: int = Field(default=15)  # Tuned value
```

2. **Set in .env**:

```bash
# .env.production
CHATLAS_LLM_PROVIDER=anthropic
CHATLAS_LLM_MODEL=claude-3-5-sonnet-20241022
CHATLAS_LLM_TEMPERATURE=0.5
CHATLAS_MCP_TIMEOUT=180
CHATLAS_AGENT_MAX_ITERATIONS=15
```

### Strategy 3: Middleware-Based Configuration

**Best for**: Dynamic behavior based on task type

**Implementation**:

1. **Create ConfigurationMiddleware**:

```python
# chatlas_agents/middleware/config.py
from langchain.agents.middleware import AgentMiddleware, wrap_model_call

class OptimizedConfigMiddleware(AgentMiddleware):
    """Apply benchmark-optimized configuration."""
    
    def __init__(self, task_type: str = "general"):
        super().__init__()
        self.task_type = task_type
        
        # Configuration from benchmarks
        self.configs = {
            "qa": {
                "temperature": 0.3,  # Lower for factual accuracy
                "max_tokens": 2048,
                "system_prompt_suffix": "Be concise and accurate.",
            },
            "research": {
                "temperature": 0.7,  # Higher for creative synthesis
                "max_tokens": 4096,
                "system_prompt_suffix": "Provide comprehensive analysis.",
            },
            "code": {
                "temperature": 0.5,
                "max_tokens": 8192,
                "system_prompt_suffix": "Write clean, well-documented code.",
            },
        }
    
    @wrap_model_call
    def apply_config(self, request, handler):
        """Apply task-specific configuration."""
        config = self.configs.get(self.task_type, {})
        
        # Modify request with optimized params
        if "temperature" in config:
            request = request.override(
                temperature=config["temperature"]
            )
        
        if "system_prompt_suffix" in config:
            current_prompt = request.system_prompt or ""
            request = request.override(
                system_prompt=f"{current_prompt}\\n\\n{config['system_prompt_suffix']}"
            )
        
        return handler(request)
```

2. **Use in agent creation**:

```python
from chatlas_agents.middleware import OptimizedConfigMiddleware

# Create agent with optimized config for Q&A tasks
agent = create_deep_agent(
    model=agent_llm,
    middleware=[
        mcp_middleware,
        OptimizedConfigMiddleware(task_type="qa"),
    ],
)
```

### Strategy 4: System Prompt Optimization

**Best for**: Improving task-specific performance

**Implementation**:

1. **Store optimized prompts** based on benchmark results:

```python
# chatlas_agents/prompts.py
OPTIMIZED_PROMPTS = {
    "atlas_qa": '''
You are an expert ATLAS physicist assistant. When answering questions:

1. ACCURACY: Provide factually correct information about ATLAS
2. RELEVANCE: Stay focused on the specific question asked
3. COVERAGE: Mention key aspects but don't overwhelm with details
4. CONCISENESS: Be clear and direct, avoiding unnecessary verbosity

For technical questions, cite specific ATLAS components or data formats when relevant.
''',
    
    "data_analysis": '''
You are a data analysis expert for ATLAS. When helping with analysis:

1. Recommend appropriate tools (ROOT, PyROOT, uproot, etc.)
2. Explain data formats (AOD, DAOD, PHYSLITE)
3. Provide working code examples
4. Explain physics context when relevant
''',
}
```

2. **Apply in CLI**:

```python
# In cli.py
from chatlas_agents.prompts import OPTIMIZED_PROMPTS

system_prompt = OPTIMIZED_PROMPTS.get(task_type, DEFAULT_PROMPT)

agent = create_deep_agent(
    model=agent_llm,
    middleware=[mcp_middleware],
    system_prompt=system_prompt,
)
```

### Strategy 5: Subagent Specialization

**Best for**: Complex multi-step tasks

**Implementation**:

1. **Define specialized subagents** based on benchmark performance:

```python
# chatlas_agents/subagents.py
OPTIMIZED_SUBAGENTS = [
    {
        "name": "atlas-expert",
        "description": "Expert in ATLAS physics and detector questions",
        "prompt": OPTIMIZED_PROMPTS["atlas_qa"],
        "tools": [],  # Uses parent tools
        "model": "claude-3-5-sonnet-20241022",  # Best model from benchmarks
    },
    {
        "name": "data-analyst",
        "description": "Expert in ATLAS data analysis and formats",
        "prompt": OPTIMIZED_PROMPTS["data_analysis"],
        "tools": [],
        "model": "gpt-5-mini",  # Cost-effective for code tasks
    },
]
```

2. **Use in agent creation**:

```python
agent = create_deep_agent(
    model=agent_llm,
    middleware=[mcp_middleware],
    subagents=OPTIMIZED_SUBAGENTS,
)
```

## Recommendation: Hybrid Approach

For ChATLAS, the best integration strategy combines multiple approaches:

### 1. **Configuration Files** (Primary)
- Store benchmark-optimized configs in `configs/` directory
- Provide templates: `configs/qa-optimized.yaml`, `configs/research-optimized.yaml`, etc.
- Users can select via `--config` flag

### 2. **Updated Defaults** (Secondary)
- Update `Settings` defaults based on overall best performance
- Ensures new users get optimized experience out-of-the-box

### 3. **Task-Specific Middleware** (Advanced)
- Create OptimizedConfigMiddleware for automatic task detection
- Apply best configuration based on query type
- Enables dynamic optimization without user intervention

### 4. **Documentation** (Essential)
- Document benchmark results and optimal configurations
- Provide usage examples for different task types
- Explain when to use which configuration

## Implementation Steps

1. **Run comprehensive benchmarks** on different task types
2. **Analyze results** to identify optimal configurations per task
3. **Create configuration templates** in `configs/` directory
4. **Update Settings defaults** with overall best configuration
5. **Document findings** in README and dedicated guide
6. **Add CLI flag** for task-type selection: `--task-type qa|research|code`
7. **Create OptimizedConfigMiddleware** for automatic optimization
8. **Integrate with deepagents-cli** as an optional feature

## Example Usage After Integration

```bash
# Use pre-optimized configuration
chatlas --config configs/qa-optimized.yaml

# Automatic task-type optimization
chatlas --task-type research

# Benchmark with specific configuration
chatlas benchmark \\
  --csv-file benchmarks/qa-test.csv \\
  --config configs/qa-optimized.yaml \\
  --experiment-name "qa-optimized-v2"

# Compare configurations
chatlas benchmark --csv-file benchmarks/test.csv --config configs/baseline.yaml
chatlas benchmark --csv-file benchmarks/test.csv --config configs/optimized.yaml
```

## References

- DeepAgents Documentation: https://docs.langchain.com/oss/python/deepagents/overview
- Middleware Guide: https://docs.langchain.com/oss/python/deepagents/middleware
- Customization: https://docs.langchain.com/oss/python/deepagents/customization
- Subagents: https://docs.langchain.com/oss/python/deepagents/subagents
