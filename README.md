# ChATLAS Agents

This repository is a fork of the LangChain `deepagents` library, modified to integrate with ChATLAS. Extends the functionality of deep agents in the following ways:
- **Native MCP Support**: MCPMiddleware for seamless integration with Model Context Protocol servers without modifying upstream packages
- **ChATLAS MCP** search ChATLAS vector stores by connecting to the MCP server.
- **ATLAS software** compatible through SetupATLAS (on Lxplus).
- **HTCondor integration** submit agent sandboxes to the HTCondor batch farm.

ChATLAS-specific features can be found in `libs/chatlas-agents`.

## Proof of Concept feature plan (v0.3.0)
- Provide a suite of skills, MCP tools to let agent users query ATLAS data sources (AMI, Rucio, Indico)
- Create benchmarks to evaluate agent performance
  - Information retrieval w/ mutli-dimensional LLM as judge scoring (relevance, accuracy, completeness, conciseness) -- use ChATLAS RAG bench dataset (already available) -- compare score to basic RAG approach + commercial agent Copilot CLI
  - More "real-world" agent task: generate review comments on previous ATLAS paper drafts -- use LLM judge multi-dimensional scoring again -- compare to Copilot CLI + human comments scraped from CDS -- Also use LLM judge to test AI comment coverage vs. human comments

## TODO list for ChatLAS Agents
### v0.3
- [x] Fix timeout issues with MCP server -- increased timeout client side and provided more pods on the server. Should be able to handle many concurrent requests now and return answers more quickly.
- [ ] Fix known bugs:
  - [ ] Agent seems to get stuck sometimes when using MCP tools in interactive mode. Needs investigation.
  - [x] Not all tools seem to be available / configured properly with the chatlas agent. Web search tool seems to be missing, for example. Fixed by modifying MCPMiddleware and adding web search tools to CLI.
- [ ] Properly set up docker and apptainer sandbox.
  - [x] Sandboxes set up with new CLI and MCP middleware.
  - [ ] Need to understand how to handle file transfers between host and sandbox. Implement this.
  - [ ] Set up and test HTCondor submission.
  - [x] Alternative container solution: set up registry with chatlas-deepagents packages pre-installed, mount workdir into sandbox & tell agent to copy files there. -> Docker container has been set up on gitlab (`gitlab-registry.cern.ch/asopio/chatlas-deepagents/chatlas_deepagents`). Can be run with either docker (`docker runn -it`) or apptainer (`apptainer shell --docker-login`).
- [ ] Interface with ATLAS software stack. Create local MCP, tools for ATLAS data sources: AMI, Rucio, Upcoming indico meetings
  - [x] Simple, preliminary solution: use deepagents skills to wrap command line tools that access ATLAS data sources.
  - [ ] Longer term: create proper MCP server with tools for ATLAS data sources (can interface this with other agent providers eg. Copilot).

### v0.4+
- [ ] Add GitLab remote. Set up CI/CD. Would be cool to have agents running in GitLab runners, eg. to produce automated reviews of paper latex sources.
  - Example: [Qwen-code GitHub actions](https://github.com/QwenLM/qwen-code-action) provides automated workflow for delegating tasks to agents, triggered thorugh local CLI commands or issue requests, and automatically places pull request on completion. Could be adapted to equivalent gitlab feaures through [GitLab MCP tools](https://docs.gitlab.com/user/gitlab_duo/model_context_protocol/mcp_server_tools/).
- [ ] Integration of CLI with IDEs and other high-level interfaces through _Agent Client Protocol_ (see, for eaxmple, [__Qwen Code__ integration in Zed IDE](https://qwenlm.github.io/qwen-code-docs/en/users/integration-zed/) which can be used with own Open AI API keys, or [__Mistral Vibe__ simpler python-based ACP](https://github.com/mistralai/mistral-vibe/tree/main/vibe/acp).

## Quick Start

```python
from chatlas_agents.middleware import MCPMiddleware
from chatlas_agents.config import MCPServerConfig
from deepagents import create_deep_agent

# Create MCP middleware
mcp_config = MCPServerConfig(url="https://chatlas-mcp.app.cern.ch/mcp", timeout=60)
mcp_middleware = await MCPMiddleware.create(mcp_config)

# Create agent with MCP support
agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-5-20250929",
    middleware=[mcp_middleware],
)
```

## ChATLAS CLI Usage

The `chatlas` command-line interface provides an interactive AI assistant with access to ChATLAS MCP tools for searching ATLAS documentation and resources.

### Installation

```bash
# Install from repository
cd libs/chatlas-agents
pip install -e .

# Or install with uv (recommended)
uv pip install -e .
```

### Quick Start

Simply run `chatlas` to start an interactive session:

```bash
chatlas
```

This launches an interactive agent session with:
- **ChATLAS MCP tools** for searching ATLAS documentation
- **DeepAgents capabilities** (file operations, planning, sub-agents)
- **Skills system** for custom tools
- **Memory** for conversation persistence
- **Human-in-the-loop** approval for destructive operations

### Configuration

Initialize a configuration file with your API keys:

```bash
chatlas init
```

This creates a `.env` file. Edit it to add your API keys:

```bash
# .env
CHATLAS_MCP_URL=https://chatlas-mcp.app.cern.ch/mcp
CHATLAS_MCP_TIMEOUT=120

CHATLAS_LLM_PROVIDER=openai
CHATLAS_LLM_MODEL=gpt-5-mini

OPENAI_API_KEY=your-api-key-here
```

Load the configuration:

```bash
export $(cat .env | xargs)
chatlas
```

### Usage Examples

**Basic interactive session:**
```bash
chatlas
```

**Use a custom agent name (for separate memory):**
```bash
chatlas --agent my-research-agent
```

**Override MCP server:**
```bash
chatlas --mcp-url https://custom-mcp.example.com/mcp
```

**Use a different model:**
```bash
chatlas --model gpt-5-mini
```

**Enable Docker sandbox for isolated code execution:**
```bash
chatlas --sandbox docker
```

**Use Apptainer sandbox (for HPC environments like lxplus):**
```bash
chatlas --sandbox apptainer --sandbox-image docker://python:3.13-slim
```

**Auto-approve all tool calls (non-interactive mode):**
```bash
chatlas --auto-approve
```

**Enable verbose logging:**
```bash
chatlas --verbose
```

**Use YAML configuration file:**
```bash
chatlas --config my-config.yaml
```

### Sandbox Execution

ChATLAS supports isolated code execution in containers:

- **Docker sandbox**: Uses Docker containers for code execution
- **Apptainer sandbox**: Uses Apptainer/Singularity (ideal for HPC environments like CERN lxplus)

Sandbox execution provides:
- Isolated environment for running code
- Secure execution boundaries
- Support for custom container images
- File upload/download capabilities

Example with Apptainer on lxplus:
```bash
# SSH to lxplus
ssh lxplus.cern.ch

# Run ChATLAS with Apptainer sandbox
chatlas --sandbox apptainer --sandbox-image docker://python:3.13-slim
```

### CLI Commands

- **`chatlas`** - Start interactive session (default)
- **`chatlas init`** - Create configuration file
- **`chatlas version`** - Show version information
- **`chatlas --help`** - Show help for all options

### Documentation

**For Developers & AI Agents:**
- **[AGENTS.md](AGENTS.md)** - Quick reference for coding agents working on this repository
- **[.github/copilot-instructions.md](.github/copilot-instructions.md)** - GitHub Copilot specific guidance

**Technical Documentation:**
- **[.github/MCP_INTEGRATION.md](.github/MCP_INTEGRATION.md)** - Comprehensive guide to MCP integration approaches and architecture
- **[.github/MCP_APPROACHES_COMPARISON.md](.github/MCP_APPROACHES_COMPARISON.md)** - Quick comparison of different integration strategies
- **[.github/DEPENDENCY_ANALYSIS.md](.github/DEPENDENCY_ANALYSIS.md)** - Module dependency analysis and setup
- **[.github/IMPLEMENTATION_SUMMARY_MCP.md](.github/IMPLEMENTATION_SUMMARY_MCP.md)** - MCP implementation summary

**Examples:**
- **[examples/mcp_middleware_example.py](libs/chatlas-agents/examples/mcp_middleware_example.py)** - Working example with deepagents
- **[examples/mcp_cli_integration_example.py](libs/chatlas-agents/examples/mcp_cli_integration_example.py)** - CLI integration patterns

**Module Documentation:**
- **[libs/chatlas-agents/README.md](libs/chatlas-agents/README.md)** - ChATLAS agents module documentation
- **[libs/chatlas-agents/SETUP.md](libs/chatlas-agents/SETUP.md)** - Detailed setup instructions

### ATLAS Software Tools Skills

ChATLAS includes specialized skills for working with ATLAS experiment software tools on LXPlus:

- **[AMI Query](libs/chatlas-agents/skills/ami-query/SKILL.md)** - Query ATLAS Metadata Interface for dataset information and metadata
- **[Rucio Management](libs/chatlas-agents/skills/rucio-management/SKILL.md)** - Download and manage ATLAS grid data using Rucio DDM
- **[ATLAS Run Query](libs/chatlas-agents/skills/atlas-runquery/SKILL.md)** - Query run information, data quality, and luminosity records

**Overview:** See [ATLAS_SKILLS.md](libs/chatlas-agents/skills/ATLAS_SKILLS.md) for detailed documentation on using these skills.

These skills provide guidance for:
- Finding and downloading ATLAS datasets from the grid
- Querying dataset metadata and production information
- Managing data quality and run selection for physics analysis
- Working with distributed data management (Rucio)

**Prerequisites:** Users must initialize their ATLAS environment in their shell **before** starting the agent:
```bash
setupATLAS
lsetup pyami              # For AMI queries
localSetupRucioClients    # For Rucio data management
voms-proxy-init -voms atlas
```

**Note:** Not all commands are needed for all skills. See individual skill prerequisites for details.

The skills are designed to work on the CERN LXPlus cluster with the full ATLAS software stack available via CVMFS. The agent can verify it's on LXPlus by checking `echo $HOSTNAME` (should match `lxplus*.cern.ch`).

---

# 🚀🧠 Deep Agents

<div align="center">
  <a href="https://docs.langchain.com/oss/python/deepagents/overview#deep-agents-overview">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset=".github/images/logo-dark.svg">
      <source media="(prefers-color-scheme: light)" srcset=".github/images/logo-light.svg">
      <img alt="Deep Agents Logo" src=".github/images/logo-dark.svg" width="50%">
    </picture>
  </a>
</div>

<div align="center">
  <h3>The batteries-included agent harness.</h3>
</div>

<div align="center">
  <a href="https://opensource.org/licenses/MIT" target="_blank"><img src="https://img.shields.io/pypi/l/deepagents" alt="PyPI - License"></a>
  <a href="https://pypistats.org/packages/deepagents" target="_blank"><img src="https://img.shields.io/pepy/dt/deepagents" alt="PyPI - Downloads"></a>
  <a href="https://pypi.org/project/deepagents/#history" target="_blank"><img src="https://img.shields.io/pypi/v/deepagents?label=%20" alt="Version"></a>
  <a href="https://x.com/langchain" target="_blank"><img src="https://img.shields.io/twitter/url/https/twitter.com/langchain.svg?style=social&label=Follow%20%40LangChain" alt="Twitter / X"></a>
</div>

<br>

Deep Agents is an agent harness. An opinionated, ready-to-run agent out of the box. Instead of wiring up prompts, tools, and context management yourself, you get a working agent immediately and customize what you need.

**What's included:**

- **Planning** — `write_todos` for task breakdown and progress tracking
- **Filesystem** — `read_file`, `write_file`, `edit_file`, `ls`, `glob`, `grep` for reading and writing context
- **Shell access** — `execute` for running commands (with sandboxing)
- **Sub-agents** — `task` for delegating work with isolated context windows
- **Smart defaults** — Prompts that teach the model how to use these tools effectively
- **Context management** — Auto-summarization when conversations get long, large outputs saved to files

> [!NOTE]
> Looking for the JS/TS library? Check out [deepagents.js](https://github.com/langchain-ai/deepagentsjs).

## Quickstart

```bash
pip install deepagents
# or
uv add deepagents
```

```python
from deepagents import create_deep_agent

agent = create_deep_agent()
result = agent.invoke({"messages": [{"role": "user", "content": "Research LangGraph and write a summary"}]})
```

The agent can plan, read/write files, and manage its own context. Add tools, customize prompts, or swap models as needed.

> [!TIP]
> For developing, debugging, and deploying AI agents and LLM applications, see [LangSmith](https://docs.langchain.com/langsmith/home).

## Customization

Add your own tools, swap models, customize prompts, configure sub-agents, and more. See the [documentation](https://docs.langchain.com/oss/python/deepagents/overview) for full details.

```python
from langchain.chat_models import init_chat_model

agent = create_deep_agent(
    model=init_chat_model("openai:gpt-4o"),
    tools=[my_custom_tool],
    system_prompt="You are a research assistant.",
)
```

MCP is supported via [`langchain-mcp-adapters`](https://github.com/langchain-ai/langchain-mcp-adapters).

## Deep Agents CLI

<p align="center">
  <img src="libs/cli/images/cli.png" alt="Deep Agents CLI" width="600"/>
</p>

```bash
curl -LsSf https://raw.githubusercontent.com/langchain-ai/deepagents/main/libs/cli/scripts/install.sh | bash
```

Web search, remote sandboxes, persistent memory, human-in-the-loop approval, and more. See the [CLI README](libs/cli/) for the full feature set.

## LangGraph Native

`create_deep_agent` returns a compiled [LangGraph](https://docs.langchain.com/oss/python/langgraph/overview) graph. Use it with streaming, Studio, checkpointers, or any LangGraph feature.

## FAQ

### Why should I use this?

- **100% open source** — MIT licensed, fully extensible
- **Provider agnostic** — Works with any Large Language Model that supports tool calling, including both frontier and open models
- **Built on LangGraph** — Production-ready runtime with streaming, persistence, and checkpointing
- **Batteries included** — Planning, file access, sub-agents, and context management work out of the box
- **Get started in seconds** — `uv add deepagents` and you have a working agent
- **Customize in minutes** — Add tools, swap models, tune prompts when you need to

---

## Documentation

- [docs.langchain.com](https://docs.langchain.com/oss/python/deepagents/overview) – Comprehensive documentation, including conceptual overviews and guides
- [reference.langchain.com/python](https://reference.langchain.com/python/deepagents/) – API reference docs for Deep Agents packages
- [Chat LangChain](https://chat.langchain.com/) – Chat with the LangChain documentation and get answers to your questions

**Discussions**: Visit the [LangChain Forum](https://forum.langchain.com) to connect with the community and share all of your technical questions, ideas, and feedback.

## Additional resources

- **[Examples](examples/)** — Working agents and patterns
- [Contributing Guide](https://docs.langchain.com/oss/python/contributing/overview) – Learn how to contribute to LangChain projects and find good first issues.
- [Code of Conduct](https://github.com/langchain-ai/langchain/?tab=coc-ov-file) – Our community guidelines and standards for participation.

---

## Acknowledgements

This project was primarily inspired by Claude Code, and initially was largely an attempt to see what made Claude Code general purpose, and make it even more so.

## Security

<<<<<<< HEAD
### `subagents`

The main agent can delegate work to sub-agents via the `task` tool (see [Built-in Tools](#built-in-tools)). You can supply custom sub-agents for context isolation and custom instructions:

```python
from deepagents import create_deep_agent

research_subagent = {
    "name": "research-agent",
    "description": "Used to research in-depth questions",
    "system_prompt": "You are an expert researcher",
    "tools": [internet_search],
    "model": "openai:gpt-5-mini",  # Optional, defaults to main agent model
}

agent = create_deep_agent(subagents=[research_subagent])
```

For complex cases, pass a pre-built LangGraph graph:

```python
from deepagents import CompiledSubAgent, create_deep_agent

custom_graph = create_agent(model=..., tools=..., system_prompt=...)

agent = create_deep_agent(
    subagents=[CompiledSubAgent(
        name="data-analyzer",
        description="Specialized agent for data analysis",
        runnable=custom_graph
    )]
)
```

See the [subagents documentation](https://docs.langchain.com/oss/python/deepagents/subagents) for more details.

### `interrupt_on`

Some tools may be sensitive and require human approval before execution. Deepagents supports human-in-the-loop workflows through LangGraph’s interrupt capabilities. You can configure which tools require approval using a checkpointer.

These tool configs are passed to our prebuilt [HITL middleware](https://docs.langchain.com/oss/python/langchain/middleware#human-in-the-loop) so that the agent pauses execution and waits for feedback from the user before executing configured tools.

```python
from langchain_core.tools import tool
from deepagents import create_deep_agent

@tool
def get_weather(city: str) -> str:
    """Get the weather in a city."""
    return f"The weather in {city} is sunny."

agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-20250514",
    tools=[get_weather],
    interrupt_on={
        "get_weather": {
            "allowed_decisions": ["approve", "edit", "reject"]
        },
    }
)
```

See the [human-in-the-loop documentation](https://docs.langchain.com/oss/python/deepagents/human-in-the-loop) for more details.

### `backend`

Deep agents use pluggable backends to control how filesystem operations work. By default, files are stored in the agent's ephemeral state. You can configure different backends for local disk access, persistent cross-conversation storage, or hybrid routing.

```python
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend

agent = create_deep_agent(
    backend=FilesystemBackend(root_dir="/path/to/project"),
)
```

Available backends include:

- **StateBackend** (default): Ephemeral files stored in agent state
- **FilesystemBackend**: Real disk operations under a root directory
- **StoreBackend**: Persistent storage using LangGraph Store
- **CompositeBackend**: Route different paths to different backends

See the [backends documentation](https://docs.langchain.com/oss/python/deepagents/backends) for more details.

### Long-term Memory

Deep agents can maintain persistent memory across conversations using a `CompositeBackend` that routes specific paths to durable storage.

This enables hybrid memory where working files remain ephemeral while important data (like user preferences or knowledge bases) persists across threads.

```python
from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from langgraph.store.memory import InMemoryStore

agent = create_deep_agent(
    backend=CompositeBackend(
        default=StateBackend(),
        routes={"/memories/": StoreBackend(store=InMemoryStore())},
    ),
)
```

Files under `/memories/` will persist across all conversations, while other paths remain temporary. Use cases include:

- Preserving user preferences across sessions
- Building knowledge bases from multiple conversations
- Self-improving instructions based on feedback
- Maintaining research progress across sessions

See the [long-term memory documentation](https://docs.langchain.com/oss/python/deepagents/long-term-memory) for more details.

## Built-in Tools

<img src=".github/images/deepagents_tools.png" alt="deep agent" width="600"/>

Every deep agent created with `create_deep_agent` comes with a standard set of tools:

| Tool Name | Description | Provided By |
|-----------|-------------|-------------|
| `write_todos` | Create and manage structured task lists for tracking progress through complex workflows | TodoListMiddleware |
| `read_todos` | Read the current todo list state | TodoListMiddleware |
| `ls` | List all files in a directory (requires absolute path) | FilesystemMiddleware |
| `read_file` | Read content from a file with optional pagination (offset/limit parameters) | FilesystemMiddleware |
| `write_file` | Create a new file or completely overwrite an existing file | FilesystemMiddleware |
| `edit_file` | Perform exact string replacements in files | FilesystemMiddleware |
| `glob` | Find files matching a pattern (e.g., `**/*.py`) | FilesystemMiddleware |
| `grep` | Search for text patterns within files | FilesystemMiddleware |
| `execute`* | Run shell commands in a sandboxed environment | FilesystemMiddleware |
| `task` | Delegate tasks to specialized sub-agents with isolated context windows | SubAgentMiddleware |

The `execute` tool is only available if the backend implements `SandboxBackendProtocol`. By default, it uses the in-memory state backend which does not support command execution. As shown, these tools (along with other capabilities) are provided by default middleware:

See the [agent harness documentation](https://docs.langchain.com/oss/python/deepagents/harness) for more details on built-in tools and capabilities.

## Built-in Middleware

`deepagents` uses middleware under the hood. Here is the list of the middleware used.

| Middleware | Purpose |
|------------|---------|
| **TodoListMiddleware** | Task planning and progress tracking |
| **FilesystemMiddleware** | File operations and context offloading (auto-saves large results) |
| **SubAgentMiddleware** | Delegate tasks to isolated sub-agents |
| **SummarizationMiddleware** | Auto-summarizes when context exceeds 170k tokens |
| **AnthropicPromptCachingMiddleware** | Caches system prompts to reduce costs (Anthropic only) |
| **PatchToolCallsMiddleware** | Fixes dangling tool calls from interruptions |
| **HumanInTheLoopMiddleware** | Pauses execution for human approval (requires `interrupt_on` config) |

## Built-in prompts

The middleware automatically adds instructions about the standard tools. Your custom instructions should **complement, not duplicate** these defaults:

#### From [TodoListMiddleware](https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain/agents/middleware/todo.py)

- Explains when to use `write_todos` and `read_todos`
- Guidance on marking tasks completed
- Best practices for todo list management
- When NOT to use todos (simple tasks)

#### From [FilesystemMiddleware](libs/deepagents/deepagents/middleware/filesystem.py)

- Lists all filesystem tools (`ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`, `execute`*)
- Explains that file paths must start with `/`
- Describes each tool's purpose and parameters
- Notes about context offloading for large tool results

#### From [SubAgentMiddleware](libs/deepagents/deepagents/middleware/subagents.py)

- Explains the `task()` tool for delegating to sub-agents
- When to use sub-agents vs when NOT to use them
- Guidance on parallel execution
- Subagent lifecycle (spawn → run → return → reconcile)

## Security Considerations

### Trust Model

Deep Agents follows a "trust the LLM" model. The agent can do anything its tools allow. Enforce boundaries at the tool/sandbox level, not by expecting the model to self-police. See the [security policy](https://github.com/langchain-ai/deepagents?tab=security-ov-file) for more information.

