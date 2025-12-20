# ChATLAS Agent Skills

This directory contains skills for ChATLAS agents built on the DeepAgents framework.

## Available Skills

### chatlas-search

Search ATLAS experiment documentation using the ChATLAS RAG system.

**Features:**
- Query multiple ATLAS knowledge bases (Twiki, CDS, Indico, ATLAS-TALK, mkdocs)
- Semantic search using RAG (Retrieval-Augmented Generation)
- Configurable number of results
- Formatted output with metadata

**Usage:**
```bash
python3 ~/.deepagents/agent/skills/chatlas-search/chatlas_search.py \
  "photon calibration" \
  --vectorstore twiki_prod \
  --ndocs 5
```

See `chatlas-search/SKILL.md` for detailed documentation.

## Using ChATLAS Skills

### Installation

When you install chatlas-agents, you can copy the skills to your DeepAgents agent directory:

```bash
# For user-level skills (available to all agents)
mkdir -p ~/.deepagents/agent/skills
cp -r libs/chatlas-agents/skills/chatlas-search ~/.deepagents/agent/skills/

# For project-level skills (specific to your project)
mkdir -p .deepagents/skills
cp -r libs/chatlas-agents/skills/chatlas-search .deepagents/skills/
```

### Discovery

Once installed, skills are automatically discovered by the DeepAgents framework:

```bash
# List all available skills
deepagents skills list

# Get detailed information about a skill
deepagents skills info chatlas-search
```

### Agent Usage

Agents can use skills in two ways:

1. **Direct invocation** via the bash/execute tool:
   ```python
   # Agent decides to use the skill
   bash("python3 ~/.deepagents/agent/skills/chatlas-search/chatlas_search.py 'query' --vectorstore twiki_prod")
   ```

2. **Skill middleware** (if configured):
   Skills with SKILL.md files are automatically loaded and their instructions are added to the agent's context.

## Creating New Skills

To create a new ChATLAS skill:

1. Create a directory: `libs/chatlas-agents/skills/my-skill/`
2. Add a `SKILL.md` file with YAML frontmatter:
   ```markdown
   ---
   name: my-skill
   description: Short description of what the skill does
   ---
   
   # My Skill
   
   ## When to Use
   - When the user needs...
   
   ## How to Use
   ...
   ```
3. Add any supporting files (Python scripts, configs, etc.)
4. Add tests to verify the skill works

## Testing Skills

Each skill should include tests to verify it works correctly:

```bash
# Run skill tests
python3 libs/chatlas-agents/skills/chatlas-search/test_chatlas_search.py
```

Tests should query the actual MCP server or services to ensure end-to-end functionality.
