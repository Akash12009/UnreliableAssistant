"""FastAPI service: POST /api/v1/agent/query (+ HITL + admin/dev endpoints). Stateless workers; all state in Redis/DB."""
from __future__ import annotations

import asyncio
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

from .config import SETTINGS
from .contracts import ID_RE, Meta
from .orchestrator import QueryRequest
from .runtime import Runtime, build_runtime
from .telemetry import RECORDER, new_traceparent

ADMIN_TOKEN = os.environ.get("UA_ADMIN_TOKEN", "dev-admin-token")
DEV = os.environ.get("UA_DEV_MODE", "1") == "1"
MAX_INFLIGHT = int(os.environ.get("UA_MAX_INFLIGHT", 512))


class QueryBody(BaseModel):
    tenant_id: str = Field(pattern=ID_RE)
    user_id: str = Field(pattern=ID_RE)
    query: str = Field(min_length=1, max_length=4000)
    session_id: str | None = Field(default=None, pattern=ID_RE)
    role: str = Field(default="operator", pattern=r"^(viewer|operator)$")   # reviewers use the HITL endpoints
    debug: bool = False


class DecideBody(BaseModel):
    tenant_id: str = Field(pattern=ID_RE)
    reviewer_id: str = Field(pattern=ID_RE)
    decision: str = Field(pattern=r"^(approve|reject)$")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.rt = await build_runtime()
    app.state.sem = asyncio.Semaphore(MAX_INFLIGHT)
    app.state.inflight = 0
    yield
    await app.state.rt.close()


app = FastAPI(title="The Unreliable Assistant (enterprise)", lifespan=lifespan)


def rt_of(r: Request) -> Runtime:
    return r.app.state.rt


def _admin(token: str | None):
    if not DEV or token != ADMIN_TOKEN:
        raise HTTPException(403, "admin endpoints disabled or bad token")


@app.get("/health")
async def health(r: Request):
    rt = rt_of(r)
    ups = {s["server"]: s["state"] for s in rt.host.snapshot()}
    ok = all(v == "up" for v in ups.values())
    return JSONResponse({"status": "ok" if ok else "degraded", "servers": ups}, status_code=200 if ok else 503)


@app.post("/api/v1/agent/query")
async def query(body: QueryBody, request: Request, x_tenant_id: str | None = Header(default=None),
                traceparent: str | None = Header(default=None)):
    if x_tenant_id is not None and x_tenant_id != body.tenant_id:
        RECORDER.incr("security.tenant_header_mismatch")
        return JSONResponse({"error": {"code": -32006, "name": "TENANT_MISMATCH", "message": "X-Tenant-ID does not match body"}}, status_code=403)
    sem: asyncio.Semaphore = request.app.state.sem
    if sem.locked():                                            # load shedding: fail fast instead of queueing past the deadline
        RECORDER.incr("req.shed")
        return JSONResponse({"error": "overloaded"}, status_code=503, headers={"Retry-After": "1"})
    async with sem:
        try:
            resp = await rt_of(request).orch.handle(QueryRequest(**body.model_dump()), traceparent)
        except Exception as e:                                  # never leak internals
            RECORDER.incr("req.5xx")
            RECORDER.incr(f"req.err.{type(e).__name__}")
            return JSONResponse({"error": "internal_error"}, status_code=500)
    if resp["status"] == "blocked":
        return JSONResponse(resp, status_code=403)
    return resp


# ------------------------------------------------------------------ HITL (reviewer) endpoints
def _rmeta(tenant: str, reviewer: str) -> Meta:
    return Meta(traceparent=new_traceparent(), tenant_id=tenant, user_id=reviewer, session_id="review", role="reviewer", subagent="reviewer")


@app.get("/api/v1/hitl/pending")
async def hitl_pending(tenant_id: str, reviewer_id: str, request: Request):
    r = await rt_of(request).harness.execute("hitl_list", {"status": "PENDING_APPROVAL"}, _rmeta(tenant_id, reviewer_id))
    return r.model_dump()


@app.post("/api/v1/hitl/{approval_id}/decide")
async def hitl_decide(approval_id: str, body: DecideBody, request: Request):
    r = await rt_of(request).harness.execute("hitl_decide", {"approval_id": approval_id, "decision": body.decision},
                                             _rmeta(body.tenant_id, body.reviewer_id))
    return JSONResponse(r.model_dump(), status_code=200 if r.ok else 409)


# ------------------------------------------------------------------ ops
@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    lines = []
    for n in sorted(RECORDER._lat):
        for q in (50, 95, 99):
            v = RECORDER.percentile(n, q)
            if v is not None:
                lines.append(f'ua_span_ms{{name="{n}",quantile="{q / 100}"}} {v:.3f}')
    for k, v in sorted(RECORDER._counters.items()):
        lines.append(f'ua_counter{{name="{k}"}} {v}')
    return "\n".join(lines) + "\n"


@app.get("/api/v1/admin/status")
async def status(request: Request, x_admin_token: str | None = Header(default=None)):
    _admin(x_admin_token)
    rt = rt_of(request)
    return dict(servers=rt.host.snapshot(), plugins=rt.plugins.status(), cache=dict(hit_ratio=rt.cache.hit_ratio(), entries=rt.cache.backend.size()),
                runbooks=dict(active=list(rt.runbooks.active), alerts=rt.runbooks.evaluate_alerts()))


@app.post("/api/v1/admin/runbook/{name}")
async def runbook(name: str, request: Request, target: str | None = None, x_admin_token: str | None = Header(default=None)):
    _admin(x_admin_token)
    return {"result": await rt_of(request).runbooks.apply(name, target)}
