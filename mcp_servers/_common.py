import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def run(mcp, default_port: int):
    """stdio by default (dev); UA_TRANSPORT=streamable-http for distributed deployment (ADR-007)."""
    transport = os.environ.get("UA_TRANSPORT", "stdio")
    if transport != "stdio":
        mcp.settings.host = os.environ.get("UA_HOST", "127.0.0.1")
        mcp.settings.port = int(os.environ.get("UA_PORT", default_port))
    mcp.run(transport=transport)
