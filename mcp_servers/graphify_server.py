"""MCP server `graphify-vault`: anchor-resolved <=depth-2 subgraph over Obsidian notes + code graph."""
import json

from _common import run  # noqa: F401

from mcp.server.fastmcp import FastMCP

from ua.contracts import GraphSubgraphIn, Meta
from ua.services import GraphService

mcp = FastMCP("graphify-vault")
svc = GraphService()


@mcp.tool(description="Return the minimal connected subgraph (<= depth 2, <= max_tokens) around the query anchor.")
def graph_subgraph(args: GraphSubgraphIn, meta: Meta) -> dict:
    return svc.call("graph_subgraph", args.model_dump(), meta.model_dump())


@mcp.resource("vault://notes/{note_id}")
def note(note_id: str) -> str:
    return json.dumps(svc.get_note(note_id) or {"error": "not found"})


if __name__ == "__main__":
    run(mcp, 8082)
