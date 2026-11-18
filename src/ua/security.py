"""HMAC-signed request metadata (tenant integrity across the MCP boundary) + signed server manifests (Gate 13)."""
from __future__ import annotations

import hashlib
import hmac
import json
import os
from pathlib import Path

from .config import ROOT
from .errors import UAError

_DEV_SECRET = "dev-only-secret-change-me"


def _secret(name: str) -> bytes:
    return os.environ.get(name, _DEV_SECRET).encode()


def sign_meta(meta) -> str:
    msg = f"{meta.tenant_id}|{meta.user_id}|{meta.session_id}|{meta.role}|{meta.subagent}".encode()
    return hmac.new(_secret("UA_MCP_SECRET"), msg, hashlib.sha256).hexdigest()


def verify_meta(meta) -> None:
    """Raises TENANT_MISMATCH (-32006) when the claimed identity was not signed by the gateway."""
    if not meta.auth or not hmac.compare_digest(meta.auth, sign_meta(meta)):
        raise UAError(-32006, "request identity signature invalid (tenant/user/role tampering)")


# ------------------------------------------------------------------ manifests
MANIFEST_DIR = ROOT / "mcp_manifests"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest_body(server: str, script: Path, tools: list[dict]) -> dict:
    return dict(server=server, script=str(script.relative_to(ROOT)), script_sha256=_sha(script),
                tools=[dict(name=t["name"], schema_sha256=hashlib.sha256(
                    json.dumps(t["schema"], sort_keys=True).encode()).hexdigest()) for t in tools])


def sign_manifest(body: dict) -> dict:
    msg = json.dumps(body, sort_keys=True).encode()
    return dict(body=body, sig=hmac.new(_secret("UA_MANIFEST_KEY"), msg, hashlib.sha256).hexdigest())


def verify_manifest(server: str) -> tuple[bool, str]:
    f = MANIFEST_DIR / f"{server}.manifest.json"
    if not f.exists():
        return False, "manifest missing"
    m = json.loads(f.read_text())
    msg = json.dumps(m["body"], sort_keys=True).encode()
    if not hmac.compare_digest(m["sig"], hmac.new(_secret("UA_MANIFEST_KEY"), msg, hashlib.sha256).hexdigest()):
        return False, "manifest signature invalid"
    script = ROOT / m["body"]["script"]
    if _sha(script) != m["body"]["script_sha256"]:
        return False, f"server script changed since signing ({script.name})"
    return True, "ok"
