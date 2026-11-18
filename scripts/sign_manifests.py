"""Sign MCP server manifests (script hash + tool schema hashes). Run after any change to mcp_servers/*.py."""
import asyncio, json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from ua import security

SERVERS = {"hybrid-rag": "rag_server.py", "state-store": "state_server.py", "graphify-vault": "graphify_server.py"}


async def tools_of(script: Path):
    p = StdioServerParameters(command=sys.executable, args=[str(script)], env={**os.environ, "UA_SQLITE": ":memory:"})
    async with stdio_client(p) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            return [dict(name=t.name, schema=t.inputSchema) for t in (await s.list_tools()).tools]


async def main():
    security.MANIFEST_DIR.mkdir(exist_ok=True)
    for name, f in SERVERS.items():
        script = ROOT / "mcp_servers" / f
        body = security.manifest_body(name, script, await tools_of(script))
        (security.MANIFEST_DIR / f"{name}.manifest.json").write_text(json.dumps(security.sign_manifest(body), indent=1))
        print("signed", name, [t["name"] for t in body["tools"]])

asyncio.run(main())
