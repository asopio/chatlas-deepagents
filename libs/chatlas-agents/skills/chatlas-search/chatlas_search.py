#!/usr/bin/env python3
"""ChATLAS Search Skill.

Searches ATLAS experiment documentation using the ChATLAS MCP server.
"""

import argparse
import asyncio
import os
import sys


def search_chatlas(query: str, vectorstore: str, ndocs: int = 5) -> str:
    """Query ChATLAS MCP server for ATLAS documentation.

    Parameters
    ----------
    query : str
        The search query string.
    vectorstore : str
        The vectorstore to search (twiki_prod, cds_v1, indico_prod_v1, 
        atlas_talk_prod, mkdocs_prod_v1).
    ndocs : int
        Number of documents to retrieve (default: 5, max: 10).

    Returns
    -------
    str
        The formatted search results or an error message.
    """
    # Check for required packages
    try:
        import httpx
        from langchain_mcp_adapters.client import MultiServerMCPClient
        from langchain_mcp_adapters.tools import load_mcp_tools
        from datetime import timedelta
    except ImportError as e:
        missing_pkg = str(e).split("'")[1] if "'" in str(e) else "unknown"
        return (
            f"Error: Required package '{missing_pkg}' not installed.\n"
            f"Install with: pip install httpx langchain-mcp-adapters chatlas-agents"
        )

    # Get MCP server configuration from environment
    mcp_url = os.environ.get("CHATLAS_MCP_URL", "https://chatlas-mcp.app.cern.ch/mcp")
    mcp_timeout = int(os.environ.get("CHATLAS_MCP_TIMEOUT", "60"))

    async def _search():
        """Async search function."""
        try:
            # Create connection configuration
            connection = {
                "url": mcp_url,
                "timeout": timedelta(seconds=mcp_timeout),
                "transport": "streamable_http",
            }

            # Load tools from MCP server
            tools = await load_mcp_tools(
                session=None,
                connection=connection,
                server_name="chatlas",
            )

            # Find the search_chatlas tool
            search_tool = None
            for tool in tools:
                if getattr(tool, 'name', '') == 'search_chatlas':
                    search_tool = tool
                    break

            if not search_tool:
                return "Error: search_chatlas tool not found on MCP server"

            # Invoke the tool
            result = await search_tool.ainvoke({
                "query": query,
                "vectorstore": vectorstore,
                "ndocs": str(min(ndocs, 10)),  # Cap at 10
            })

            # Format the results
            if isinstance(result, list) and len(result) > 0:
                # Extract text from the result
                output_lines = []
                for item in result:
                    if isinstance(item, dict) and 'text' in item:
                        output_lines.append(item['text'])
                
                if output_lines:
                    return "\n\n".join(output_lines)
                else:
                    return "No results found or unexpected result format."
            else:
                return "No results found."

        except Exception as e:
            return f"Error querying ChATLAS: {type(e).__name__}: {str(e)}"

    # Run the async function
    try:
        return asyncio.run(_search())
    except Exception as e:
        return f"Error: {type(e).__name__}: {str(e)}"


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Search ATLAS documentation using ChATLAS"
    )
    parser.add_argument("query", type=str, help="Search query string")
    parser.add_argument(
        "--vectorstore",
        type=str,
        required=True,
        choices=["twiki_prod", "cds_v1", "indico_prod_v1", "atlas_talk_prod", "mkdocs_prod_v1"],
        help="Vectorstore to search",
    )
    parser.add_argument(
        "--ndocs",
        type=int,
        default=5,
        help="Number of documents to retrieve (default: 5, max: 10)",
    )

    args = parser.parse_args()

    # Execute search and print results
    result = search_chatlas(args.query, args.vectorstore, args.ndocs)
    print(result)


if __name__ == "__main__":
    main()
