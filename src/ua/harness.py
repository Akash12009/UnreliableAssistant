"""ToolExecutionHarness: every MCP tool call passes through
   (1) Pydantic v2 strict validation -> (2) deterministic regex auto-repair -> (3) RBAC tier check ->
   (4) UUIDv4 idempotency enforcement -> circuit breaker -> dispatch.
No LLM is involved in repair: it is a pure function, so it can never 'repair' into something unreviewed.
"""
from __future__ import annotations

import hashlib
import json
import re
import uuid
from typing import Any

from pydantic import ValidationError

from .contracts import TIER_RANK, TOOLS, Meta, Tier, ToolResult
from .errors import Block, UAError
from .mcp_host import McpHost
from .plugins.registry import LifecycleRegistry
from .telemetry import RECORDER, span

_SKU = re.compile(r"(?i)^\s*sku[\s\-_#:]*0*(\d{1,3})\s*$")
_LOT = re.compile(r"(?i)^\s*lot[\s\-_#:]*(\d{1,5})\s*$")
ID_FIELDS = {"sku": "SKU", "lot": "LOT", "entity_id": None}

# role -> highest tier the *user* may request (Tier 3 still goes through HITL)
ROLE_MAX_TIER = {"viewer": Tier.TIER_1_READ, "operator": Tier.TIER_3_MUTATE, "reviewer": Tier.TIER_3_MUTATE}


def repair_id(value: Any, kind: str | None) -> Any:
    """SKU99 -> SKU-099 ; lot 44 -> LOT-00044. Returns the input unchanged when it cannot be repaired unambiguously."""
    if not isinstance(value, str):
        return value
    if kind in (None, "SKU") and (m := _SKU.match(value)):
        return f"SKU-{int(m.group(1)):03d}"
    if kind in (None, "LOT") and (m := _LOT.match(value)):
        return f"LOT-{int(m.group(1)):05d}"
    return value


def repair_args(model, args: dict) -> dict:
    out = dict(args)
    props = model.model_fields
    for k, v in args.items():
        if k in ID_FIELDS:
            out[k] = repair_id(v, ID_FIELDS[k])
        elif k in props and isinstance(v, str):
            ann = props[k].annotation
            s = v.strip()
            if ann in (int, "int") or str(ann) in ("int", "int | None", "<class 'int'>"):
                if re.fullmatch(r"-?\d+", s):
                    out[k] = int(s)
            elif ann in (float, "float") or str(ann) in ("float", "float | None", "<class 'float'>"):
                if re.fullmatch(r"-?\d+(\.\d+)?", s):
                    out[k] = float(s)
            elif ann is bool and s.lower() in ("true", "false"):
                out[k] = s.lower() == "true"
    if "payload" in args and isinstance(args["payload"], dict):
        out["payload"] = {k: v for k, v in args["payload"].items()}
    return out


def det_uuid4(*parts: str) -> str:
    """Deterministic UUIDv4-format key: identical logical call within a session replays instead of duplicating."""
    h = bytearray(hashlib.sha256("|".join(parts).encode()).digest()[:16])
    h[6] = (h[6] & 0x0F) | 0x40
    h[8] = (h[8] & 0x3F) | 0x80
    return str(uuid.UUID(bytes=bytes(h)))


class ToolCall:
    def __init__(self, name: str, args: dict, meta: Meta):
        self.name, self.args, self.meta = name, args, meta


class ToolExecutionHarness:
    def __init__(self, host: McpHost, plugins: LifecycleRegistry):
        self.host, self.plugins = host, plugins

    async def execute(self, tool: str, raw_args: dict, meta: Meta, timeout_ms: float | None = None) -> ToolResult:
        spec = TOOLS.get(tool)
        if not spec:
            return self._err(UAError(-32601, f"unknown tool {tool}"))
        model = spec["model"]
        repaired_from = None
        # (1) strict validation
        try:
            args = model.model_validate(raw_args).model_dump()
            RECORDER.incr("harness.first_pass_ok")
        except ValidationError as e1:
            RECORDER.incr("harness.first_pass_fail")
            # (2) deterministic auto-repair, then re-validate
            fixed = repair_args(model, raw_args)
            try:
                args = model.model_validate(fixed).model_dump()
                repaired_from = {k: raw_args[k] for k in raw_args if raw_args.get(k) != fixed.get(k)}
                RECORDER.incr("harness.repair_ok")
            except ValidationError as e2:
                RECORDER.incr("harness.repair_fail")
                fields = ",".join(".".join(map(str, x["loc"])) for x in e2.errors())[:120]
                return self._err(UAError(-32602, f"invalid arguments for {tool}: {fields}"))
        # (3) RBAC: subagent allow-list + user role tier
        if meta.subagent not in spec["subagents"]:
            return self._err(UAError(-32002, f"subagent '{meta.subagent}' may not call {tool}"))
        if TIER_RANK[spec["tier"]] > TIER_RANK[ROLE_MAX_TIER.get(meta.role, Tier.TIER_1_READ)]:
            return self._err(UAError(-32002, f"role '{meta.role}' may not call {spec['tier'].value} tool {tool}"))
        # (4) idempotency (UUIDv4) for Tier 2/3
        if TIER_RANK[spec["tier"]] >= 2 and not meta.idempotency_key:
            key = det_uuid4(meta.tenant_id, meta.session_id, tool, json.dumps(args, sort_keys=True, default=str))
            meta = meta.model_copy(update={"idempotency_key": key})
        call = ToolCall(tool, args, meta)
        # circuit breaker / dispatch plugins
        try:
            self.plugins.dispatch(call)
        except Block as b:
            return self._err(b)
        try:
            with span("mcp.call", tool=tool):
                raw = await self.host.call(tool, args, meta, timeout_ms)
        except UAError as e:
            res = self._err(e)
            self.plugins.observe(call, res)
            return res
        ok = bool(raw.get("ok"))
        status = raw.get("status", "OK") if ok else "REJECTED"
        res = ToolResult(ok=ok, data=raw.get("data"),
                         status="REPAIRED" if (repaired_from and status == "OK") else status,
                         repaired_from=repaired_from, error=raw.get("error"), latency_ms=raw.get("_rt_ms", 0.0))
        self.plugins.observe(call, res)
        cb = self.plugins.get("circuit_breaker")
        if not res.ok and cb is not None and cb.key(call) in cb.open:
            res.tripped = True
        return res

    @staticmethod
    def _err(e: UAError) -> ToolResult:
        return ToolResult(ok=False, status="REJECTED", error=e.to_obj())
