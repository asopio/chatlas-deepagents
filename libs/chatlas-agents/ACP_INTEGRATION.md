# ACP Integration Guide for ChATLAS Agents

This document provides a comprehensive guide for integrating ChATLAS agents with third-party applications using the Agent-Client Protocol (ACP).

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Configuration](#configuration)
4. [Agent Modes](#agent-modes)
5. [Protocol Details](#protocol-details)
6. [Client Integration](#client-integration)
7. [Troubleshooting](#troubleshooting)
8. [API Reference](#api-reference)

## Overview

The ChATLAS ACP server implements the Agent-Client Protocol v0.6.0, enabling seamless integration with:

- **IDEs**: VSCode, JetBrains, Zed, and other ACP-compatible editors
- **Chatbot Applications**: Custom chat interfaces and applications
- **Automation Tools**: CI/CD pipelines, code review bots, etc.

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│              ACP Client (IDE, App, etc.)                 │
│  - User interface                                        │
│  - Session management                                    │
│  - Permission prompts                                    │
└──────────────────┬──────────────────────────────────────┘
                   │ JSON-RPC over stdio
                   │
┌──────────────────▼──────────────────────────────────────┐
│              ChATLAS ACP Server (chatlas-acp)            │
│  - Session lifecycle                                     │
│  - Message streaming                                     │
│  - Tool execution                                        │
│  - Permission handling                                   │
└──────────────────┬──────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────┐
│              DeepAgents CLI Agent                        │
│  - Planning & reasoning                                  │
│  - File operations                                       │
│  - Sub-agent spawning                                    │
│  - Memory & skills                                       │
└──────────────────┬──────────────────────────────────────┘
                   │
          ┌────────┴────────┐
          │                 │
┌─────────▼───────┐ ┌──────▼──────────┐
│   MCP Tools     │ │  Standard Tools │
│  - ChATLAS RAG  │ │  - HTTP request │
│  - ATLAS docs   │ │  - Web search   │
│                 │ │  - File ops     │
└─────────────────┘ └─────────────────┘
```

## Quick Start

### Installation

```bash
# Install chatlas-agents package
cd libs/chatlas-agents
uv sync

# Verify installation
chatlas-acp --version
```

### Basic Usage

```bash
# 1. Set up configuration
chatlas-acp --setup

# 2. Edit .env.chatlas-acp with your API keys
vim .env.chatlas-acp

# 3. Load environment
export $(cat .env.chatlas-acp | xargs)

# 4. Start ACP server
chatlas-acp
```

The server will wait for stdio communication from an ACP client.

### Testing with a Simple Client

```python
import asyncio
import subprocess
import json

async def test_acp_server():
    """Simple ACP client for testing."""
    # Start the server
    process = await asyncio.create_subprocess_exec(
        "chatlas-acp",
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    
    # Send initialize request
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "0.6.0",
            "clientCapabilities": {},
        },
    }
    
    process.stdin.write(json.dumps(request).encode() + b"\n")
    await process.stdin.drain()
    
    # Read response
    response = await process.stdout.readline()
    print("Server response:", response.decode())
    
    # Terminate
    process.terminate()
    await process.wait()

asyncio.run(test_acp_server())
```

## Configuration

### Environment Variables

The ACP server is configured entirely through environment variables:

#### Agent Settings

```bash
# Agent identifier (affects memory storage)
export CHATLAS_ACP_AGENT_ID=chatlas-acp

# Working directory (defaults to current directory)
export CHATLAS_ACP_WORKDIR=/path/to/workdir
```

#### LLM Settings

```bash
# LLM model to use
export CHATLAS_ACP_MODEL=gpt-4

# API keys (at least one required)
export OPENAI_API_KEY=sk-...
# OR
export ANTHROPIC_API_KEY=sk-ant-...
# OR
export GROQ_API_KEY=gsk_...
```

#### MCP Server Settings

```bash
# ChATLAS MCP server URL
export CHATLAS_MCP_URL=https://chatlas-mcp.app.cern.ch/mcp

# Connection timeout in seconds
export CHATLAS_MCP_TIMEOUT=120
```

#### Sandbox Settings

```bash
# Sandbox type: docker, apptainer, or leave unset for local execution
export CHATLAS_ACP_SANDBOX_TYPE=docker

# Container image for sandbox
export CHATLAS_ACP_SANDBOX_IMAGE=python:3.13-slim
```

#### Agent Features

```bash
# Enable persistent conversation memory
export CHATLAS_ACP_ENABLE_MEMORY=true

# Enable custom skills system
export CHATLAS_ACP_ENABLE_SKILLS=true

# Enable shell access (local mode only)
export CHATLAS_ACP_ENABLE_SHELL=true
```

#### Agent Behavior

```bash
# Default agent mode: default, auto, or research
export CHATLAS_ACP_DEFAULT_MODE=default

# Enable verbose logging
export CHATLAS_ACP_VERBOSE=false
```

### Programmatic Configuration

```python
from chatlas_agents.acp import ACPConfig, run_acp_server

# Create custom configuration
config = ACPConfig(
    agent_id="my-agent",
    model="gpt-4-turbo",
    mcp_url="https://chatlas-mcp.app.cern.ch/mcp",
    mcp_timeout=120,
    sandbox_type="docker",
    sandbox_image="python:3.13-slim",
    enable_memory=True,
    enable_skills=True,
    enable_shell=False,
    default_mode="default",
    verbose=True,
)

# Start server with custom config
run_acp_server(config)
```

## Agent Modes

The ACP server supports three operational modes that can be switched during a session:

### Default Mode

**Human-in-the-loop mode with permission prompts**

- Requests user approval for destructive operations
- Safe for production use
- Recommended for most use cases

```bash
export CHATLAS_ACP_DEFAULT_MODE=default
```

**Behavior:**
- File writes/deletions require approval
- Shell commands require approval
- Read operations execute automatically

### Auto Mode

**Auto-approve mode that skips permission requests**

- All operations execute automatically
- Use with caution in production
- Useful for trusted automation workflows

```bash
export CHATLAS_ACP_DEFAULT_MODE=auto
```

**Behavior:**
- All operations execute without prompts
- Faster execution
- Higher risk of unintended changes

### Research Mode

**Information retrieval focused mode**

- Optimized for searching and reading
- Limited write operations
- Ideal for documentation exploration

```bash
export CHATLAS_ACP_DEFAULT_MODE=research
```

**Behavior:**
- Emphasizes MCP search tools
- Minimal file system changes
- Focus on information gathering

### Switching Modes

Modes can be changed during a session via the `setSessionMode` request:

```json
{
  "jsonrpc": "2.0",
  "id": 42,
  "method": "setSessionMode",
  "params": {
    "sessionId": "session-uuid",
    "modeId": "auto"
  }
}
```

## Protocol Details

### Supported ACP Methods

#### Required Methods

- `initialize` - Initialize agent and return capabilities
- `newSession` - Create a new agent session
- `prompt` - Send user prompt and receive streaming response
- `cancel` - Cancel a running session

#### Optional Methods

- `setSessionMode` - Change session mode (default/auto/research)
- `loadSession` - Load existing session (not yet implemented)
- `setSessionModel` - Change LLM model (not supported - model is fixed)

### Session Lifecycle

```
┌────────────┐
│ Initialize │ - Exchange capabilities
└─────┬──────┘
      │
┌─────▼──────┐
│ New Session│ - Create agent session with thread ID
└─────┬──────┘
      │
      ├─────────────────────┐
      │                     │
┌─────▼──────┐      ┌──────▼────────┐
│   Prompt   │ ───> │ Set Mode/Model│ (optional)
└─────┬──────┘      └───────────────┘
      │
      │ (repeat)
      │
┌─────▼──────┐
│   Cancel   │ (if needed)
└────────────┘
```

### Update Types

The server streams various update types to the client:

#### AgentMessageChunk

Text content from the agent's response:

```json
{
  "sessionUpdate": "agent_message_chunk",
  "content": {
    "type": "text",
    "text": "I'll help you with that..."
  }
}
```

#### AgentThoughtChunk

Reasoning/thinking content:

```json
{
  "sessionUpdate": "agent_thought_chunk",
  "content": {
    "type": "text",
    "text": "First, I need to understand..."
  }
}
```

#### ToolCallProgress

Tool execution progress:

```json
{
  "sessionUpdate": "tool_call_update",
  "toolCallId": "call_123",
  "title": "search_chatlas",
  "rawInput": {"query": "ATLAS detector"},
  "status": "pending"
}
```

Then when complete:

```json
{
  "sessionUpdate": "tool_call_update",
  "toolCallId": "call_123",
  "title": "search_chatlas",
  "content": [{
    "type": "content",
    "content": {"type": "text", "text": "Found 5 results..."}
  }],
  "rawOutput": "Found 5 results...",
  "status": "completed"
}
```

#### AgentPlanUpdate

TODO list updates:

```json
{
  "sessionUpdate": "plan",
  "entries": [
    {
      "content": "Search ATLAS documentation",
      "status": "completed",
      "priority": "medium"
    },
    {
      "content": "Analyze results",
      "status": "in_progress",
      "priority": "medium"
    }
  ]
}
```

### Permission Requests

When human-in-the-loop approval is needed:

```json
{
  "method": "requestPermission",
  "params": {
    "sessionId": "session-uuid",
    "toolCall": {
      "toolCallId": "perm_abc123",
      "title": "write_file",
      "rawInput": {"path": "output.txt", "content": "..."},
      "status": "pending"
    },
    "options": [
      {
        "optionId": "allow-once",
        "name": "Allow once",
        "kind": "allow_once"
      },
      {
        "optionId": "reject-once",
        "name": "Reject",
        "kind": "reject_once"
      }
    ]
  }
}
```

Client responds:

```json
{
  "jsonrpc": "2.0",
  "id": ...,
  "result": {
    "outcome": {
      "outcome": "selected",
      "optionId": "allow-once"
    }
  }
}
```

## Client Integration

### Example: Custom Python Client

```python
import asyncio
import json
import subprocess
from typing import AsyncIterator, Any

class ChATLASACPClient:
    """Simple ACP client for ChATLAS agents."""
    
    def __init__(self):
        self.process = None
        self.request_id = 0
    
    async def start(self):
        """Start the ACP server process."""
        self.process = await asyncio.create_subprocess_exec(
            "chatlas-acp",
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    
    async def send_request(self, method: str, params: dict) -> dict:
        """Send JSON-RPC request and get response."""
        self.request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": method,
            "params": params,
        }
        
        self.process.stdin.write(json.dumps(request).encode() + b"\n")
        await self.process.stdin.drain()
        
        response = await self.process.stdout.readline()
        return json.loads(response.decode())
    
    async def initialize(self) -> dict:
        """Initialize the agent."""
        return await self.send_request("initialize", {
            "protocolVersion": "0.6.0",
            "clientCapabilities": {},
        })
    
    async def new_session(self, cwd: str = ".") -> str:
        """Create a new session and return session ID."""
        response = await self.send_request("newSession", {
            "cwd": cwd,
            "mcpServers": [],
        })
        return response["result"]["sessionId"]
    
    async def prompt(self, session_id: str, text: str) -> AsyncIterator[dict]:
        """Send prompt and stream responses."""
        await self.send_request("prompt", {
            "sessionId": session_id,
            "prompt": [
                {"type": "text", "text": text}
            ],
        })
        
        # Stream session updates
        while True:
            line = await self.process.stdout.readline()
            if not line:
                break
            
            update = json.loads(line.decode())
            if update.get("method") == "sessionUpdate":
                yield update["params"]["update"]
            elif "result" in update:
                # Prompt completed
                break
    
    async def stop(self):
        """Stop the server process."""
        if self.process:
            self.process.terminate()
            await self.process.wait()

# Usage example
async def main():
    client = ChATLASACPClient()
    await client.start()
    
    # Initialize
    init_response = await client.initialize()
    print(f"Agent: {init_response['result']['agentInfo']['title']}")
    
    # Create session
    session_id = await client.new_session()
    print(f"Session: {session_id}")
    
    # Send prompt
    async for update in client.prompt(session_id, "What is the ATLAS detector?"):
        if update["sessionUpdate"] == "agent_message_chunk":
            print(update["content"]["text"], end="", flush=True)
    print()
    
    await client.stop()

asyncio.run(main())
```

## Troubleshooting

### Common Issues

#### Server doesn't start

```bash
# Check if all dependencies are installed
uv sync

# Verify API key is set
echo $OPENAI_API_KEY

# Test with verbose logging
chatlas-acp --verbose
```

#### Connection timeout to MCP server

```bash
# Increase timeout
export CHATLAS_MCP_TIMEOUT=300

# Test MCP connectivity
curl https://chatlas-mcp.app.cern.ch/mcp
```

#### Permission errors in sandbox

```bash
# For Docker sandbox
docker info  # Verify Docker is running

# For Apptainer sandbox
apptainer --version  # Verify Apptainer is installed
```

### Debug Mode

Enable verbose logging to see detailed execution:

```bash
export CHATLAS_ACP_VERBOSE=true
chatlas-acp
```

This will show:
- Configuration loading
- Agent graph creation
- Tool loading
- Session lifecycle
- Message streaming
- Error stack traces

## API Reference

### ACPConfig

Configuration model for the ACP server.

**Fields:**
- `agent_id: str` - Agent identifier (default: "chatlas-acp")
- `workdir: Path` - Working directory (default: current directory)
- `model: str` - LLM model (default: "gpt-4")
- `mcp_url: str` - MCP server URL
- `mcp_timeout: int` - MCP timeout in seconds (default: 120)
- `sandbox_type: str | None` - Sandbox type ("docker", "apptainer", or None)
- `sandbox_image: str` - Container image (default: "python:3.13-slim")
- `enable_memory: bool` - Enable conversation memory (default: True)
- `enable_skills: bool` - Enable skills system (default: True)
- `enable_shell: bool` - Enable shell access (default: True)
- `default_mode: AgentModeType` - Default mode (default: "default")
- `verbose: bool` - Verbose logging (default: False)

**Methods:**
- `from_env() -> ACPConfig` - Load from environment variables
- `get_auto_approve(mode: AgentModeType) -> bool` - Check if mode auto-approves

### ChATLASACP

Main ACP server implementation.

**Methods:**
- `initialize(params: InitializeRequest) -> InitializeResponse`
- `newSession(params: NewSessionRequest) -> NewSessionResponse`
- `prompt(params: PromptRequest) -> PromptResponse`
- `cancel(params: CancelNotification) -> None`
- `setSessionMode(params: SetSessionModeRequest) -> SetSessionModeResponse | None`
- `setSessionModel(params: SetSessionModelRequest) -> SetSessionModelResponse | None`
- `loadSession(params: LoadSessionRequest) -> LoadSessionResponse | None`

### Helper Functions

- `run_acp_server(config: ACPConfig | None = None) -> None` - Start ACP server

---

For more information, see:
- [ACP Specification](https://agentclientprotocol.com/)
- [ChATLAS Agents README](README.md)
- [Example Code](examples/acp_server_example.py)
