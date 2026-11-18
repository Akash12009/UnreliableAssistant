"""MCP server business logic. Shared by mcp_servers/*.py (real MCP transports) and the in-process transport.

Every tool: verifies signed identity -> validates input (Pydantic strict) -> tenant-scoped execution -> returns a
ToolResult-shaped dict. Servers never raise across the wire; errors are structured (code/name/message).
"""
from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Callable

from .config import SETTINGS
from .contracts import (TOOLS, AuditAppendIn, AuditVerifyIn, HitlListIn, GetFactsIn, GraphSubgraphIn, HitlDecideIn, HybridRetrieveIn, LookupItemIn,
                        Meta, SessionFactIn, SubmitFormIn, TransitionIn)
from .errors import UAError
from .security import verify_meta
from .store import StateStore, canonical


def _result(ok=True, data=None, status="OK", err: UAError | None = None, t0: float | None = None) -> dict:
    return dict(ok=ok, data=data, status=status, error=err.to_obj().model_dump() if err else None,
                server_ms=round((time.perf_counter() - t0) * 1000, 3) if t0 else 0.0)


class BaseService:
    tools: dict[str, Callable]

    def call(self, tool: str, args: dict, meta: dict, verify: bool = True) -> dict:
        t0 = time.perf_counter()
        try:
            m = Meta(**meta)
            if verify:
                verify_meta(m)
            model = TOOLS[tool]["model"]
            a = model(**args) if tool not in () else model(**args)
            out = self.tools[tool](a, m)
            if isinstance(out, dict) and "ok" in out:
                out["server_ms"] = round((time.perf_counter() - t0) * 1000, 3)
                return out
            return _result(True, out, t0=t0)
        except UAError as e:
            return _result(False, err=e, t0=t0)
        except Exception as e:                       # validation / internal errors -> structured, never raw tracebacks
            code = -32602 if e.__class__.__name__ == "ValidationError" else -32603
            return _result(False, err=UAError(code, f"{e.__class__.__name__}: {str(e)[:300]}"), t0=t0)


class RagService(BaseService):
    def __init__(self, engine=None):
        from .rag.engine import RagEngine
        self.engine = engine or RagEngine()
        self.tools = {"hybrid_retrieve": self.hybrid_retrieve}

    def hybrid_retrieve(self, a: HybridRetrieveIn, m: Meta):
        return self.engine.retrieve(a, debug=a.debug).model_dump()

    def get_chunk(self, chunk_id: str) -> dict | None:
        return self.engine.get_chunk(chunk_id)


class GraphService(BaseService):
    def __init__(self, graph=None, path: str | None = None):
        from .graphify import Graph
        self.graph = graph or Graph.load(path or SETTINGS.graph_path)
        self.tools = {"graph_subgraph": self.subgraph}

    def subgraph(self, a: GraphSubgraphIn, m: Meta):
        return self.graph.subgraph(a.query, a.anchor, a.depth, a.max_tokens)

    def get_note(self, note_id: str) -> dict | None:
        nid = self.graph.by_ref.get(note_id.upper())
        return self.graph.nodes.get(nid) if nid else None


class StateService(BaseService):
    def __init__(self, store: StateStore | None = None):
        self.store = store or StateStore(SETTINGS.sqlite_path)
        self.tools = {"lookup_item": self.lookup_item, "get_session_facts": self.get_facts,
                      "put_session_fact": self.put_fact, "submit_form": self.mutation("submit_form"),
                      "transition_workflow": self.mutation("transition_workflow"), "audit_append": self.audit,
                      "hitl_decide": self.hitl_decide, "hitl_list": self.hitl_list, "audit_verify": self.audit_verify}
        self._seeded: set[str] = set()

    def _ensure_seed(self, tenant: str) -> None:
        """DEMO ONLY: lazily give every tenant its own copy of data/items.json (per-tenant rows, tenant_id in every query)."""
        if tenant in self._seeded:
            return
        from .config import ROOT
        with self.store.lock:
            if not self.store.db.execute("SELECT 1 FROM items WHERE tenant_id=? LIMIT 1", (tenant,)).fetchone():
                for it in json.loads((ROOT / "data" / "items.json").read_text()):
                    self.store.upsert_item(tenant, it["item_id"], it["kind"], it["state"], it["data"])
        self._seeded.add(tenant)

    # ---- Tier 1
    def hitl_list(self, a: HitlListIn, m: Meta):
        if m.role != "reviewer":
            raise UAError(-32002, "hitl_list requires reviewer role")
        return {"items": self.store.hitl_list(m.tenant_id, None if a.status == "ALL" else a.status)}

    def audit_verify(self, a: AuditVerifyIn, m: Meta):
        ok, n, bad = self.store.verify_chain(m.tenant_id)
        out = {"ok": ok, "records": n, "first_bad_seq": bad}
        if a.tail:
            out["tail"] = self.store.audit_tail(m.tenant_id, a.tail)
        return out

    def lookup_item(self, a: LookupItemIn, m: Meta):
        self._ensure_seed(m.tenant_id)
        iid = a.sku or a.lot
        it = self.store.get_item(m.tenant_id, iid)
        if not it:
            raise UAError(-32011, f"{iid} not found")
        return it

    def get_facts(self, a: GetFactsIn, m: Meta):
        return {"facts": self.store.get_facts(m.tenant_id, m.session_id)}

    # ---- Tier 2 (idempotent)
    def _idem(self, m: Meta, tool: str, args: dict, fn: Callable[[], dict]) -> dict:
        if not m.idempotency_key:
            raise UAError(-32602, f"{tool} requires idempotency_key (UUIDv4)")
        h = hashlib.sha256(canonical([tool, args]).encode()).hexdigest()
        prev = self.store.idem_get(m.tenant_id, m.idempotency_key)
        if prev:
            if prev[0] != h:
                raise UAError(-32003, "idempotency_key reused with different arguments")
            return {**json.loads(prev[1]), "replayed": True}
        out = fn()
        self.store.idem_put(m.tenant_id, m.idempotency_key, h, canonical(out))
        return out

    def put_fact(self, a: SessionFactIn, m: Meta):
        return self._idem(m, "put_session_fact", a.model_dump(),
                          lambda: (self.store.put_fact(m.tenant_id, m.session_id, a.key, a.value) or {"stored": a.key}))

    def audit(self, a: AuditAppendIn, m: Meta):
        r = self.store.audit_append(m.tenant_id, a.event, {**a.detail, "session": m.session_id, "user": m.user_id})
        return r

    # ---- Tier 3: NEVER executes directly; always routed through the HITL gate (KPI-19 = 100%)
    def mutation(self, tool: str):
        def run(a, m: Meta):
            def create():
                aid = self.store.hitl_create(m.tenant_id, tool, a.model_dump(), m.user_id, m.idempotency_key)
                return {"approval_id": aid}
            data = self._idem(m, tool, a.model_dump(), create)
            return dict(ok=True, status="PENDING_APPROVAL", data=data)
        return run

    def hitl_decide(self, a: HitlDecideIn, m: Meta):
        if m.role != "reviewer" or m.subagent != "reviewer":
            raise UAError(-32002, "hitl_decide requires an authorised reviewer")
        try:
            rec = self.store.hitl_decide(m.tenant_id, a.approval_id, a.decision, m.user_id)
        except PermissionError as e:
            raise UAError(-32002, str(e))
        if rec is None:
            raise UAError(-32011, "approval not pending (unknown, expired or already decided)")
        result = None
        if rec["status"] == "APPROVED":
            result = self._execute(m.tenant_id, rec["tool"], rec["args"], m.user_id)
        return dict(approval_id=a.approval_id, status=rec["status"], executed=result)

    def _execute(self, tenant: str, tool: str, args: dict, reviewer: str) -> dict:
        if tool == "transition_workflow":
            ok = self.store.set_item_state(tenant, args["entity_id"], args["to_state"])
            res = {"entity": args["entity_id"], "state": args["to_state"], "applied": ok}
        elif args["form_type"] == "safety_release":
            ok = self.store.set_item_state(tenant, args["entity_id"], "released")
            res = {"entity": args["entity_id"], "state": "released", "applied": ok}
        elif args["form_type"] == "price_change":
            it = self.store.get_item(tenant, args["entity_id"])
            ok = bool(it)
            if ok and "price" in args.get("payload", {}):
                it["data"]["price"] = args["payload"]["price"]
                self.store.upsert_item(tenant, it["item_id"], it["kind"], it["state"], it["data"])
            res = {"entity": args["entity_id"], "applied": ok}
        else:
            res = {"entity": args["entity_id"], "form": args["form_type"], "applied": True}
        self.store.audit_append(tenant, "mutation.executed", {"tool": tool, "entity": args["entity_id"], "approved_by": reviewer})
        return res
