---
name: chatlas-search
description: Search ATLAS experiment documentation using ChATLAS RAG system with multiple vectorstores
---

# ChATLAS Search Skill

This skill provides access to the ChATLAS RAG (Retrieval-Augmented Generation) system for querying ATLAS experiment documentation across multiple knowledge bases.

## When to Use This Skill

Use this skill when you need to:
- Search ATLAS Twiki pages for technical documentation
- Find ATLAS papers, notes, and talks from CERN Document Server
- Locate information from ATLAS Indico meetings and presentations
- Search ATLAS-TALK forum posts for technical discussions
- Access ATLAS public documentation

## Available Vectorstores

- **twiki_prod**: ATLAS Twiki pages with technical documentation (Run 1, 2, 3)
- **cds_v1**: CERN Document Server - papers, notes, and talks
- **indico_prod_v1**: ATLAS Indico - meeting agendas, minutes, and slides
- **atlas_talk_prod**: ATLAS-TALK forum posts on software and technical topics
- **mkdocs_prod_v1**: ATLAS public documentation hosted on mkdocs

## How to Use

The skill provides a Python script that queries the ChATLAS MCP server and returns formatted results.

### Basic Usage

**Note:** This skill should be run using `uv run` from the chatlas-agents directory to ensure all dependencies are available.

```bash
cd [CHATLAS_AGENTS_DIR]
uv run [YOUR_SKILLS_DIR]/chatlas-search/chatlas_search.py "your query" --vectorstore twiki_prod [--ndocs N]
```

Where:
- `[CHATLAS_AGENTS_DIR]`: Path to the chatlas-agents package (e.g., `libs/chatlas-agents`)
- `[YOUR_SKILLS_DIR]`: Absolute path to your skills directory (shown in the system prompt)

**Arguments:**
- `query` (required): The search query string (e.g., "photon calibration", "trigger menu")
- `--vectorstore` (required): Which knowledge base to search (see list above)
- `--ndocs` (optional): Number of documents to retrieve (default: 5, max: 10)

### Examples

Search ATLAS Twiki for photon calibration:
```bash
cd libs/chatlas-agents
uv run ~/.deepagents/agent/skills/chatlas-search/chatlas_search.py "photon calibration" --vectorstore twiki_prod --ndocs 5
```

Find papers about Higgs to diphoton:
```bash
cd libs/chatlas-agents
uv run ~/.deepagents/agent/skills/chatlas-search/chatlas_search.py "Higgs diphoton" --vectorstore cds_v1
```

Search Indico for trigger meetings:
```bash
cd libs/chatlas-agents
uv run ~/.deepagents/agent/skills/chatlas-search/chatlas_search.py "trigger menu" --vectorstore indico_prod_v1
```

Search ATLAS-TALK for software questions:
```bash
cd libs/chatlas-agents
uv run ~/.deepagents/agent/skills/chatlas-search/chatlas_search.py "athena framework" --vectorstore atlas_talk_prod
```

## Output Format

The script returns formatted results with:
- **Source information**: Document metadata (title, date, URL if available)
- **Content**: Retrieved text chunks relevant to your query
- **Child and Parent documents**: Both specific chunks and broader context

Results are ordered by relevance to the query.

## Features

- **Multiple knowledge bases**: Access different ATLAS documentation sources
- **RAG-based retrieval**: Semantic search for better relevance
- **Configurable results**: Control number of documents returned
- **Date filtering**: Some vectorstores include temporal information

## Dependencies

This skill is designed to run with the `uv` package manager from the chatlas-agents directory.

**Required packages** (managed by uv):
- `httpx`: For HTTP requests to the MCP server
- `langchain-mcp-adapters`: For MCP protocol communication
- `chatlas-agents`: For MCP client functionality

All dependencies are automatically available when using `uv run` from the chatlas-agents directory.

If running outside of uv (not recommended), the skill will detect missing packages and show installation instructions.

## Environment Variables

The skill requires:
- `CHATLAS_MCP_URL`: URL of the ChATLAS MCP server (default: https://chatlas-mcp.app.cern.ch/mcp)
- `CHATLAS_MCP_TIMEOUT`: Timeout in seconds (default: 60)

## Notes

- Twiki documentation may include outdated information from earlier runs
- Check the 'date' field in metadata when available
- CDS includes both public and internal ATLAS documents
- Indico results include PDF presentations which are text-extracted
- For best results, use specific technical terms familiar to ATLAS
