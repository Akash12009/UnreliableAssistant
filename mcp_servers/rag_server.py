"""MCP server `hybrid-rag`: 3-stage retrieval (BM25 || bi-encoder -> RRF -> cross-encoder). Tool: hybrid_retrieve; resource rag://chunks/{id}."""
import json

from _common import run  # noqa: F401  (also puts src/ on sys.path)

from mcp.server.fastmcp import FastMCP

from ua.contracts import HybridRetrieveIn, Meta
from ua.services import RagService

mcp = FastMCP("hybrid-rag")
svc = RagService()


@mcp.tool(description="3-stage hybrid retrieval. Returns <=4 cross-encoder-verified chunks or an abstention.")
def hybrid_retrieve(args: HybridRetrieveIn, meta: Meta) -> dict:
    return svc.call("hybrid_retrieve", args.model_dump(), meta.model_dump())


@mcp.resource("rag://chunks/{chunk_id}")
def chunk(chunk_id: str) -> str:
    return json.dumps(svc.get_chunk(chunk_id) or {"error": "not found"})


if __name__ == "__main__":
    run(mcp, 8080)
