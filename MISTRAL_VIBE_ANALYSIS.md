# Mistral Vibe 2.0 Feature Analysis for ChATLAS DeepAgents

**Date:** January 27, 2026  
**Version:** Mistral Vibe 2.0.0  
**Repository:** https://github.com/mistralai/mistral-vibe

## Executive Summary

Mistral Vibe 2.0 introduces several significant features that could enhance chatlas-deepagents. This document analyzes these features, their implementation in Mistral Vibe, and provides recommendations for adapting them to our architecture.

## Key Features in Mistral Vibe 2.0

### 1. 🎯 AskUserQuestion Tool

**What it does:**  
Interactive tool allowing agents to ask users clarifying questions during execution, with multi-choice options and free-text fallback.

**Implementation Details:**
- **Location:** `vibe/core/tools/builtins/ask_user_question.py`
- **Key Components:**
  - Pydantic models: `Question`, `Choice`, `Answer`, `AskUserQuestionResult`
  - Supports 1-4 questions displayed as tabs
  - Each question has 2-4 options plus automatic "Other" for free text
  - Multi-select mode support
  - Cancellation handling
- **UI Integration:** Custom TUI widget in `vibe/cli/textual_ui/widgets/question_app.py`
- **Tool Permission:** Always allowed (no approval needed)

**Adaptation for ChATLAS DeepAgents:**

**Priority:** HIGH  
**Effort:** LOW-MEDIUM  
**Value:** HIGH - Significantly improves interactive workflows

**Implementation Plan:**
```python
# 1. Create new tool in libs/chatlas-agents/chatlas_agents/tools/ask_user_question.py
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

class Choice(BaseModel):
    label: str = Field(description="Short label (1-5 words)")
    description: str = Field(default="", description="Optional explanation")

class Question(BaseModel):
    question: str
    header: str = Field(default="", max_length=12)
    options: list[Choice] = Field(min_length=2, max_length=4)
    multi_select: bool = False

class AskUserQuestionInput(BaseModel):
    questions: list[Question] = Field(min_length=1, max_length=4)

class AskUserQuestionTool(BaseTool):
    name = "ask_user_question"
    description = "Ask user clarifying questions with multi-choice options"
    
    def _run(self, questions: list[Question]) -> dict:
        # Implementation using callback pattern
        pass
```

**Integration Points:**
- Add to CLI's interactive mode with Textual UI widget
- Support callback mechanism for question handling
- Enable in both local and sandbox modes

---

### 2. 🤖 Agent Configuration System

**What it does:**  
TOML-based agent profiles with nested config overrides, enabling multiple specialized agent personas (e.g., plan-only, auto-approve, red-team).

**Implementation Details:**
- **Location:** `vibe/core/agents/models.py`, `vibe/core/agents/manager.py`
- **Built-in Agents:**
  - `default`: Standard agent with approval required
  - `plan`: Read-only for exploration (auto-approves safe tools)
  - `accept-edits`: Auto-approves file edits only
  - `auto-approve`: Auto-approves all tools
  - `explore`: Subagent for read-only codebase exploration
- **TOML Structure:**
  ```toml
  display_name = "Custom Agent"
  description = "Agent description"
  safety = "neutral"  # safe, neutral, destructive, yolo
  agent_type = "agent"  # or "subagent"
  
  [overrides]
  auto_approve = true
  enabled_tools = ["grep", "read_file"]
  active_model = "mistral-large-latest"
  ```
- **Features:**
  - Deep merge of agent config with base config
  - Agent discovery from `~/.vibe/agents/` and `.vibe/agents/`
  - Runtime agent switching
  - Safety level indicators

**Adaptation for ChATLAS DeepAgents:**

**Priority:** HIGH  
**Effort:** MEDIUM  
**Value:** HIGH - Enables specialized workflows

**Implementation Plan:**
```python
# 1. Create agent configuration system
# Location: libs/chatlas-agents/chatlas_agents/agents/

from enum import Enum
from pydantic import BaseModel
from typing import Optional, Dict, Any

class AgentType(str, Enum):
    AGENT = "agent"
    SUBAGENT = "subagent"

class SafetyLevel(str, Enum):
    SAFE = "safe"          # Read-only operations
    NEUTRAL = "neutral"    # Requires approval for writes
    DESTRUCTIVE = "destructive"  # Explicit approval needed
    YOLO = "yolo"         # Auto-approve everything

class AgentProfile(BaseModel):
    name: str
    display_name: str
    description: str
    safety: SafetyLevel = SafetyLevel.NEUTRAL
    agent_type: AgentType = AgentType.AGENT
    overrides: Dict[str, Any] = {}
    
    @classmethod
    def from_yaml(cls, path: str) -> "AgentProfile":
        # Load from YAML file
        pass

class AgentManager:
    def __init__(self, config_paths: list[str]):
        self.profiles: Dict[str, AgentProfile] = {}
        self._discover_agents()
    
    def _discover_agents(self):
        # Scan for *.yaml in config paths
        pass
    
    def get_profile(self, name: str) -> AgentProfile:
        pass
    
    def get_subagents(self) -> list[AgentProfile]:
        return [p for p in self.profiles.values() 
                if p.agent_type == AgentType.SUBAGENT]
```

**Built-in Agents to Create:**
1. `default`: Standard ChATLAS agent with MCP tools
2. `explore`: Read-only agent for codebase exploration
3. `atlas-expert`: Specialized for ATLAS software queries
4. `document-writer`: Optimized for documentation tasks

**Configuration Location:**
- Global: `~/.chatlas/agents/`
- Local: `.chatlas/agents/`
- Built-in: `libs/chatlas-agents/chatlas_agents/agents/builtin/`

---

### 3. 📦 Enhanced Session Management

**What it does:**  
Separates session metadata from messages, enabling better session resumption and statistics tracking.

**Implementation Details:**
- **Location:** `vibe/core/session/session_logger.py`, `session_loader.py`
- **File Structure:**
  - `meta.json`: Session metadata (ID, timestamps, model, agent, stats)
  - `messages.jsonl`: Message history (system prompt excluded)
- **Metadata Fields:**
  ```json
  {
    "session_id": "abc12345",
    "session_start_time": "2026-01-27T19:00:00Z",
    "session_prefix": "session",
    "agent_name": "default",
    "model": "mistral-large-latest",
    "statistics": {
      "total_turns": 15,
      "total_cost": 0.45,
      "total_tokens": 12000
    }
  }
  ```
- **Benefits:**
  - Metadata survives message compaction
  - Better session browsing and filtering
  - Cost tracking per session
  - Resume by session ID prefix (first 8 chars)

**Adaptation for ChATLAS DeepAgents:**

**Priority:** MEDIUM  
**Effort:** MEDIUM  
**Value:** MEDIUM - Improves session management

**Implementation Plan:**
```python
# Location: libs/chatlas-agents/chatlas_agents/session/

from datetime import datetime
from pydantic import BaseModel
from pathlib import Path
import json

class SessionMetadata(BaseModel):
    session_id: str
    session_start_time: datetime
    session_end_time: Optional[datetime] = None
    agent_name: str
    model: str
    mcp_server: Optional[str] = None
    statistics: Dict[str, Any] = {}

class SessionManager:
    def __init__(self, session_dir: Path):
        self.session_dir = session_dir
    
    def create_session(self, agent: str, model: str) -> str:
        session_id = self._generate_id()
        metadata = SessionMetadata(
            session_id=session_id,
            session_start_time=datetime.now(),
            agent_name=agent,
            model=model
        )
        self._save_metadata(session_id, metadata)
        return session_id
    
    def save_message(self, session_id: str, message: dict):
        # Append to messages.jsonl
        pass
    
    def load_session(self, session_id_prefix: str) -> tuple[SessionMetadata, list]:
        # Load metadata + messages
        pass
```

**Session Storage:**
- Location: `~/.chatlas/sessions/{session_id}/`
- Files: `meta.json`, `messages.jsonl`
- Cleanup: Delete sessions older than 30 days (configurable)

---

### 4. 🔧 Skill System Improvements

**What it does:**  
Markdown-based skill files with YAML frontmatter for custom tools and slash commands.

**Implementation Details:**
- **Location:** `vibe/core/skills/`
- **File Format:** `{skill-name}/SKILL.md` with YAML frontmatter
- **Frontmatter Schema:**
  ```yaml
  ---
  name: my-skill
  description: Custom skill description
  license: MIT
  compatibility: Python 3.12+
  user-invocable: true
  allowed_tools:
    - read_file
    - grep
  metadata:
    category: analysis
  ---
  
  # Skill Documentation
  This skill helps with...
  ```
- **Discovery Paths:**
  1. Config paths (from config.toml)
  2. Local `.vibe/skills/`
  3. Global `~/.vibe/skills/`
- **Skill Management:**
  - Enable/disable via patterns: `enabled_skills = ["code-review", "test-*"]`
  - Skill content injected into system prompt
  - Supports slash command creation

**Adaptation for ChATLAS DeepAgents:**

**Priority:** MEDIUM  
**Effort:** LOW  
**Value:** MEDIUM - Already have skill system, improve it

**Current State:**
- Already have skills in `libs/chatlas-agents/skills/`
- Current format: `SKILL.md` with description
- Examples: ami-query, rucio-management, atlas-runquery

**Improvements to Make:**
1. Add YAML frontmatter support
2. Implement skill discovery from multiple paths
3. Add enable/disable patterns in config
4. Support `user-invocable` flag for slash commands
5. Add metadata field for categorization

**Implementation:**
```python
# Update: libs/chatlas-agents/chatlas_agents/skills/manager.py

import frontmatter
from pathlib import Path

class SkillMetadata(BaseModel):
    name: str
    description: str
    license: Optional[str] = None
    compatibility: Optional[str] = None
    user_invocable: bool = True
    allowed_tools: list[str] = []
    metadata: Dict[str, Any] = {}

class SkillManager:
    def __init__(self, skill_paths: list[Path]):
        self.skill_paths = skill_paths
        self.skills: Dict[str, Skill] = {}
    
    def discover_skills(self):
        for path in self.skill_paths:
            for skill_dir in path.glob("*/"):
                skill_file = skill_dir / "SKILL.md"
                if skill_file.exists():
                    self._load_skill(skill_file)
    
    def _load_skill(self, path: Path):
        with open(path) as f:
            post = frontmatter.load(f)
            metadata = SkillMetadata(**post.metadata)
            content = post.content
            # Store skill
```

**Skill Paths:**
- Built-in: `libs/chatlas-agents/skills/`
- User global: `~/.chatlas/skills/`
- Project local: `.chatlas/skills/`

---

### 5. 🔌 MCP Configuration Enhancements

**What it does:**  
Enhanced MCP server configuration with environment variables, custom timeouts, and multiple transport types.

**Implementation Details:**
- **Location:** `vibe/core/config.py`, `vibe/core/tools/mcp.py`
- **Configuration Schema:**
  ```toml
  [[mcp_servers]]
  name = "chatlas"
  transport = "http"
  url = "https://chatlas-mcp.app.cern.ch/mcp"
  headers = { "Authorization" = "Bearer token" }
  api_key_env = "CHATLAS_API_KEY"
  api_key_header = "Authorization"
  api_key_format = "Bearer {token}"
  startup_timeout_sec = 15
  tool_timeout_sec = 120
  
  [[mcp_servers]]
  name = "local_tools"
  transport = "stdio"
  command = "python"
  args = ["-m", "mcp_server.tools"]
  env = { "DEBUG" = "1" }
  ```
- **Transport Types:**
  1. HTTP: Direct HTTP connection
  2. Streamable-HTTP: HTTP with streaming
  3. Stdio: Local process via stdin/stdout
- **Features:**
  - Custom timeout per server
  - Environment variable injection for stdio
  - API key from environment variable
  - Customizable authentication headers

**Adaptation for ChATLAS DeepAgents:**

**Priority:** MEDIUM  
**Effort:** LOW  
**Value:** MEDIUM - Improves MCP flexibility

**Current State:**
- Have `MCPServerConfig` in `chatlas_agents/config/__init__.py`
- Basic URL and timeout support
- No environment variable or stdio support

**Improvements:**
```python
# Update: libs/chatlas-agents/chatlas_agents/config/__init__.py

from enum import Enum

class MCPTransport(str, Enum):
    HTTP = "http"
    STREAMABLE_HTTP = "streamable-http"
    STDIO = "stdio"

class MCPServerConfig(BaseModel):
    name: str
    transport: MCPTransport = MCPTransport.HTTP
    
    # HTTP/Streamable-HTTP fields
    url: Optional[str] = None
    headers: Dict[str, str] = {}
    api_key_env: Optional[str] = None
    api_key_header: str = "Authorization"
    api_key_format: str = "Bearer {token}"
    
    # Stdio fields
    command: Optional[str] = None
    args: list[str] = []
    env: Dict[str, str] = {}
    
    # Common fields
    startup_timeout_sec: int = 10
    tool_timeout_sec: int = 60
    
    def get_effective_url(self) -> str:
        """Resolve URL with API key if needed."""
        if self.api_key_env and self.api_key_env in os.environ:
            # Add auth header or format URL
            pass
        return self.url
    
    def get_env_vars(self) -> Dict[str, str]:
        """Get environment variables for stdio."""
        return {**os.environ, **self.env}
```

**Configuration File:**
```yaml
# configs/agent_config.yaml
mcp_servers:
  - name: chatlas
    transport: http
    url: https://chatlas-mcp.app.cern.ch/mcp
    startup_timeout_sec: 15
    tool_timeout_sec: 120
    
  - name: local_ami
    transport: stdio
    command: python
    args: ["-m", "atlas_mcp.ami"]
    env:
      ATLAS_AUTH: "grid"
```

---

### 6. 🔄 Auto-Update Feature

**What it does:**  
Automatic version checking with changelog display on updates.

**Implementation Details:**
- **Location:** `vibe/cli/update_notifier/`
- **Components:**
  - `UpdateGateway`: Abstract interface for version checking
  - `PyPIUpdateGateway`: Check PyPI for latest version
  - `UpdateCacheRepository`: Filesystem cache to avoid frequent checks
  - `check_update()`: Daily version check
  - `do_update()`: Subprocess upgrade with pip/uv
- **What's New Display:**
  - Static markdown file: `vibe/whats_new.md`
  - Shown once per version bump
  - Tracks seen versions in cache
- **Configuration:**
  ```toml
  enable_auto_update = true  # default
  ```

**Adaptation for ChATLAS DeepAgents:**

**Priority:** LOW  
**Effort:** LOW  
**Value:** LOW-MEDIUM - Nice to have

**Implementation Plan:**
```python
# Location: libs/chatlas-agents/chatlas_agents/update/

from packaging import version
import httpx
from datetime import datetime, timedelta
from pathlib import Path

class UpdateChecker:
    def __init__(self, package_name: str = "chatlas-agents"):
        self.package_name = package_name
        self.cache_path = Path.home() / ".chatlas" / "update_cache.json"
    
    async def check_update(self) -> Optional[str]:
        """Check PyPI for new version. Returns new version or None."""
        if not self._should_check():
            return None
        
        current = self._get_current_version()
        latest = await self._get_latest_version()
        
        self._update_cache()
        
        if version.parse(latest) > version.parse(current):
            return latest
        return None
    
    async def _get_latest_version(self) -> str:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"https://pypi.org/pypi/{self.package_name}/json")
            data = resp.json()
            return data["info"]["version"]
    
    def _should_check(self) -> bool:
        """Check at most once per day."""
        if not self.cache_path.exists():
            return True
        cache = json.loads(self.cache_path.read_text())
        last_check = datetime.fromisoformat(cache["last_check"])
        return datetime.now() - last_check > timedelta(days=1)
```

**What's New:**
- Create `libs/chatlas-agents/WHATS_NEW.md`
- Display in CLI on version bump
- Track in `~/.chatlas/seen_versions.json`

---

### 7. 👥 Subagent Support (Future Enhancement)

**What it does:**  
Specialized agents that can be delegated work by the main agent, running in isolated contexts.

**Implementation Details:**
- **Location:** `vibe/core/tools/builtins/task.py`, `vibe/core/agents/`
- **Key Concepts:**
  - Subagents defined with `agent_type = "subagent"` in TOML
  - Tool restrictions per subagent type
  - Built-in "Explore" subagent: read-only, uses only `grep` and `read_file`
  - Delegation via `task()` tool
  - Separate context window per subagent
  - Auto-approval inherits from parent agent
- **Task Tool:**
  ```python
  task(
      task="Analyze project structure",
      agent="explore"  # Subagent name
  )
  ```

**Adaptation for ChATLAS DeepAgents:**

**Priority:** LOW (Future)  
**Effort:** HIGH  
**Value:** HIGH - Powerful but complex

**Notes:**
- DeepAgents already has subagent support via `task` tool
- Would need to integrate with agent configuration system
- Requires careful context isolation
- Consider as Phase 2 enhancement

**Recommended Subagents:**
1. `explore`: Read-only codebase exploration
2. `atlas-query`: Specialized for ATLAS data queries (AMI, Rucio)
3. `document-generator`: Documentation writing
4. `code-reviewer`: Code review tasks

---

## Implementation Roadmap

### Phase 1: Quick Wins (1-2 weeks)

**1. AskUserQuestion Tool**
- [ ] Create Pydantic models
- [ ] Implement tool class
- [ ] Add CLI callback support
- [ ] Create Textual UI widget
- [ ] Add to default tool set
- [ ] Write tests
- [ ] Update documentation

**2. MCP Configuration Enhancements**
- [ ] Extend `MCPServerConfig` with new fields
- [ ] Add environment variable resolution
- [ ] Support multiple transport types
- [ ] Update configuration examples
- [ ] Test with ChATLAS MCP server

**3. Skill System Improvements**
- [ ] Add YAML frontmatter parsing
- [ ] Implement multi-path skill discovery
- [ ] Add enable/disable patterns
- [ ] Update existing skills with frontmatter
- [ ] Document skill creation guide

### Phase 2: Core Features (2-4 weeks)

**4. Agent Configuration System**
- [ ] Create `AgentProfile` and `AgentManager`
- [ ] Define built-in agent profiles
- [ ] Implement config merging
- [ ] Add agent switching in CLI
- [ ] Create agent discovery system
- [ ] Add safety level indicators
- [ ] Document custom agent creation

**5. Enhanced Session Management**
- [ ] Create `SessionMetadata` model
- [ ] Implement dual-file storage
- [ ] Add session resumption by ID
- [ ] Track statistics (cost, tokens)
- [ ] Add session cleanup
- [ ] Update CLI to show session info

### Phase 3: Nice to Have (Future)

**6. Auto-Update Feature**
- [ ] Implement `UpdateChecker`
- [ ] Create version cache
- [ ] Add "What's New" display
- [ ] Add config flag
- [ ] Test upgrade flow

**7. Subagent Support**
- [ ] Design delegation architecture
- [ ] Implement context isolation
- [ ] Create built-in subagents
- [ ] Add tool filtering per agent
- [ ] Integrate with agent config system

---

## Architectural Patterns from Mistral Vibe

### 1. Manager Pattern
All major components use a Manager class for discovery, caching, and lifecycle:
- `AgentManager`: Agent discovery and switching
- `SkillManager`: Skill discovery and loading
- `ToolManager`: Tool registration and permission handling
- `SessionManager`: Session storage and resumption

### 2. Callback Pattern
Interactive features use callbacks for UI integration:
- `InvokeContext.user_input_callback`: For `ask_user_question`
- `InvokeContext.approval_callback`: For tool approval
- `InvokeContext.progress_callback`: For progress updates

### 3. Pydantic Everywhere
All configuration and data structures use Pydantic:
- Strong typing
- Automatic validation
- JSON/YAML serialization
- Clear documentation via Field descriptions

### 4. Config Merging
Agent configs use deep merge for overrides:
```python
def _deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
```

### 5. Pattern-based Filtering
Tools, skills, and agents support pattern matching:
- Exact match: `"grep"`
- Glob: `"mcp_*"`
- Regex: `"re:^atlas_.*$"`

---

## Recommendations

### High Priority (Implement First)
1. **AskUserQuestion Tool** - Low effort, high value, improves UX significantly
2. **Agent Configuration System** - Medium effort, high value, enables specialized workflows
3. **MCP Configuration Enhancements** - Low effort, medium value, better flexibility

### Medium Priority (Implement Next)
4. **Enhanced Session Management** - Medium effort, medium value, better tracking
5. **Skill System Improvements** - Low effort, medium value, better organization

### Low Priority (Future)
6. **Auto-Update Feature** - Low effort, low value, nice to have
7. **Subagent Support** - High effort, high value, but already partially supported in DeepAgents

### Not Recommended
- **Trust Folder System** - Not needed in our environment
- **External Editor Integration** - IDE integration via ACP is better approach
- **Terminal Theme System** - Not a priority

---

## Testing Strategy

### Unit Tests
- Test Pydantic models for validation
- Test manager discovery logic
- Test config merging
- Test pattern matching

### Integration Tests
- Test agent switching with config changes
- Test skill loading from multiple paths
- Test MCP server connection with env vars
- Test session save/load with metadata

### E2E Tests
- Test full interactive session with AskUserQuestion
- Test agent profile switching in CLI
- Test session resumption

---

## Documentation Updates

### User Documentation
- [ ] Add "Agent Profiles" section to README
- [ ] Document AskUserQuestion tool usage
- [ ] Update MCP server configuration guide
- [ ] Create skill creation guide
- [ ] Add session management guide

### Developer Documentation
- [ ] Update AGENTS.md with new patterns
- [ ] Document manager classes
- [ ] Add architecture decision records
- [ ] Update API reference

---

## Conclusion

Mistral Vibe 2.0 provides excellent patterns for improving chatlas-deepagents. The most impactful features to adopt are:

1. **AskUserQuestion Tool** - Enables better interactive workflows
2. **Agent Configuration System** - Provides flexibility for different use cases
3. **Enhanced MCP Configuration** - Improves tool integration

These features align well with our existing architecture and can be implemented incrementally without breaking changes. The middleware pattern we already use makes integration straightforward.

**Next Steps:**
1. Review this analysis with the team
2. Prioritize features based on user needs
3. Create implementation tickets
4. Begin with Phase 1 quick wins

---

**References:**
- Mistral Vibe Repository: https://github.com/mistralai/mistral-vibe
- Mistral Vibe 2.0 Changelog: https://github.com/mistralai/mistral-vibe/blob/main/CHANGELOG.md
- Agent Skills Specification: https://agentskills.io/specification
