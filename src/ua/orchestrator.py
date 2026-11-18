"""Request orchestration: pre-hooks -> cache -> triage -> subagent ReAct loop -> verification -> audit -> egress."""
from __future__ import annotations

import asyncio
import hashlib
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .agents import Evidence, ExtractiveLLM, compress_observation
from .cache import SemanticCache
from .config import SETTINGS, scaled
from .contracts import Meta, Route
from .errors import Block, UAError
from .harness import ToolExecutionHarness
from .loop import S, GoalAnchorBlock, HotMemory, ReActLoop
from .plugins.registry import LifecycleRegistry, RequestContext
from .telemetry import RECORDER, new_traceparent, span, start_trace
from .textnorm import approx_tokens, norm_query

PRICE_IN, PRICE_OUT = float(os.environ.get("UA_PRICE_IN_PER_M", 0.25)), float(os.environ.get("UA_PRICE_OUT_PER_M", 1.25))
ABSTAIN_MSG = "I can't answer that from the verified knowledge base, so I won't guess. [{reason}]"
HELP_MSG = ("I answer questions about products (SKU), batches (LOT), policies (POL) and SOPs from the verified knowledge base, "
            "and can submit change requests that a human reviewer must approve.")
OOS_MSG = "That request is outside what I can help with (enterprise catalog, inventory, policy and SOP questions)."


@dataclass
class Draft:
    text: str
    chunks: list[dict]
    abstained: bool = False
    faithfulness: float = 1.0
    unsupported: list[str] = field(default_factory=list)
    veto: str | None = None


@dataclass
class QueryRequest:
    tenant_id: str
    user_id: str
    query: str
    session_id: str | None = None
    role: str = "operator"
    debug: bool = False


class Orchestrator:
    def __init__(self, harness: ToolExecutionHarness, plugins: LifecycleRegistry, cache: SemanticCache, triage, embedder,
                 memory: HotMemory | None = None, llm=None):
        self.h, self.plugins, self.cache, self.triage, self.emb = harness, plugins, cache, triage, embedder
        self.mem = memory or HotMemory(SETTINGS.loop.hot_turns)
        self.loop = ReActLoop(harness, self.mem)
        self.llm = llm or ExtractiveLLM()
        self.bg: set[asyncio.Task] = set()

    def _bg(self, coro) -> None:
        t = asyncio.create_task(coro)
        self.bg.add(t)
        t.add_done_callback(self.bg.discard)

    async def _generate(self, q, ev, pairs, strict, qvec):
        """Returns (text, abstain_reason|None, usage dict, fallback flag). LLM failure -> extractive fallback (FMEA #4)."""
        if hasattr(self.llm, "agenerate"):
            o = await self.llm.agenerate(q, ev, pairs, strict=strict)
            if not o.error:
                if o.abstain:
                    return "", o.abstain_reason or "llm_abstain", dict(prompt=o.prompt_tokens, completion=o.completion_tokens), False
                return o.text, None, dict(prompt=o.prompt_tokens, completion=o.completion_tokens), False
            RECORDER.incr("llm.fallback")
            RECORDER.incr(f"llm.err.{o.error}")
            return ExtractiveLLM().generate(q, ev, pairs, strict=strict, qvec=qvec, emb=self.emb), None, {}, True
        return self.llm.generate(q, ev, pairs, strict=strict, qvec=qvec, emb=self.emb), None, {}, False

    async def handle(self, req: QueryRequest, traceparent: str | None = None) -> dict:
        t0 = time.perf_counter()
        tp = traceparent or new_traceparent()
        trace = start_trace(tp)
        deadline_t = t0 + scaled(SETTINGS.loop.deadline_ms) / 1000
        session = req.session_id or f"s-{req.user_id}"
        RECORDER.incr("req.total")
        ctx = RequestContext(req.tenant_id, req.user_id, session, req.role, req.query, traceparent=tp)
        base = dict(tenant_id=req.tenant_id, user_id=req.user_id, session_id=session, trace_id=trace.trace_id)

        def finish(resp: dict, route: str = "") -> dict:
            ms = (time.perf_counter() - t0) * 1000
            resp.update(base, route=route or resp.get("route", ""), latency_ms=round(ms, 2))
            RECORDER.finish_trace(trace.spans)
            from .telemetry import add_span
            add_span("request.e2e", ms, route=resp["route"], status=resp["status"])
            RECORDER.sample("e2e_ms", ms)
            if resp["status"] != "blocked":
                RECORDER.incr(f"req.status.{resp['status']}")
            if req.debug:
                resp["trace"] = [dict(name=s.name, ms=round(s.ms, 2)) for s in trace.spans]
            return resp

        # 1) pre-execution plugins (PII -> injection), fail-closed
        try:
            await asyncio.to_thread(self.plugins.pre, ctx) if False else self.plugins.pre(ctx)
        except Block as b:
            RECORDER.incr(f"req.blocked.{b.name}")
            self._bg(self._audit(req.tenant_id, session, req.user_id, "request.blocked", dict(code=b.code, trace=trace.trace_id)))
            return finish(dict(status="blocked", answer="", error=b.to_obj().model_dump(), citations=[], abstained=False), "")
        q = ctx.query
        nq = norm_query(q)

        # 2) semantic cache (tenant/user namespaced; never crosses tenants)
        with span("cache.lookup"):
            qvec = await asyncio.to_thread(self.emb.query, nq)
            hit = self.cache.lookup(req.tenant_id, req.user_id, q)
        if hit is not None:
            if hit.get("tenant_id") != req.tenant_id:                 # defence in depth: must never happen
                RECORDER.incr("cache.leak_detected")
                hit = None
        if hit is not None:
            RECORDER.incr("cache.hit")
            r = {k: v for k, v in hit.items() if k not in ("_cache", "scope")}
            r.update(cache=hit["_cache"], steps=0)
            return finish(r)
        RECORDER.incr("cache.miss")

        # 3) triage (<= 25 ms, temperature 0)
        with span("triage.classify"):
            tri = self.triage.classify(q, qvec)
        route = tri.route
        meta_base = dict(traceparent=tp, tenant_id=req.tenant_id, user_id=req.user_id, session_id=session, role=req.role)
        meta = Meta(**meta_base)
        info = dict(intent=tri.intent, domain=tri.domain, risk_tier=tri.risk_tier, complexity=tri.complexity,
                    confidence=round(tri.confidence, 3), fallback=tri.fallback)

        if route == Route.REJECT_OOS:
            return finish(dict(status="rejected_oos", answer=OOS_MSG, citations=[], abstained=True, triage=info, steps=0), route.value)
        if route == Route.DIRECT_ANSWER:
            return finish(dict(status="answered", answer=HELP_MSG, citations=[], abstained=False, triage=info, steps=0), route.value)

        # 4) bounded ReAct loop (subagents bound to their MCP servers)
        ents = tuple(dict.fromkeys(m.upper() for m in __import__("re").findall(r"\b(?:SKU|LOT|POL|SOP)-\d+\b", q, __import__("re").I)))
        anchor = GoalAnchorBlock(q, ents)
        lr = await self.loop.run(route.value, q, meta, anchor, deadline_t)

        steps, trace_steps = lr.steps, lr.trace
        out: dict[str, Any] = dict(triage=info, steps=steps, react_steps=sum(1 for t in trace_steps if t.get("calls")), loop_state=lr.state.value, backend=lr.backend,
                                   degraded=lr.degraded)
        if req.debug:
            out["react_trace"] = trace_steps
            out["contexts"] = [dict(src=e.src, text=e.text) for e in lr.evidence]
            out["anchor"] = anchor.render()

        # 5) route-specific completion + verification
        if lr.degraded and route == Route.HYBRID_RAG:     # FMEA #3: retrieval guards did not run -> deterministic abstain, never graph-only
            RECORDER.incr("req.degraded_abstain")
            out.update(status="abstained", answer=ABSTAIN_MSG.format(reason="retrieval_unavailable"), citations=[], abstained=True,
                       abstain_reason="retrieval_unavailable")
            return finish(out, route.value)
        if lr.state in (S.HALTED, S.HALTED_BREAKER):
            RECORDER.incr("loop.halted")
            if lr.state == S.HALTED_BREAKER:
                RECORDER.incr("loop.halted_breaker")
            out.update(status="halted", answer=f"I stopped: {lr.halted_reason}. Check the identifiers in your request and try again.",
                       citations=[], abstained=False, halted_reason=lr.halted_reason)
            self._bg(self._audit(req.tenant_id, session, req.user_id, "loop.halted", dict(reason=lr.halted_reason[:80], trace=trace.trace_id)))
            return finish(out, route.value)

        if route == Route.TRANSACTION_EXEC:
            tx = lr.tx_result
            if tx and tx.get("status") == "PENDING_APPROVAL":
                out.update(status="pending_approval", citations=[], abstained=False, approval_id=tx.get("approval_id"),
                           answer=f"Submitted for human approval (id {tx.get('approval_id')}). No change is made until an authorized reviewer approves.")
                RECORDER.incr("hitl.pending")
            else:
                out.update(status="needs_clarification", citations=[], abstained=False,
                           answer="I could not turn that into a supported change request. Please name the item (e.g. LOT-88001) and the action (release, quarantine, recall, archive, price change).")
            return finish(out, route.value)

        # HYBRID_RAG: ground or abstain
        hard = lr.abstain_reason and (lr.abstain_reason.startswith(("unknown_entity", "unsupported_term")))
        if lr.degraded and not lr.evidence:
            RECORDER.incr("req.degraded_abstain")
            out.update(status="abstained", answer=ABSTAIN_MSG.format(reason="retrieval_unavailable"), citations=[], abstained=True,
                       abstain_reason="retrieval_unavailable")
            return finish(out, route.value)
        if lr.quarantined:
            RECORDER.incr("indirect.quarantined", len(lr.quarantined))
            out["quarantined"] = lr.quarantined
            self._bg(self._audit(req.tenant_id, session, req.user_id, "evidence.quarantined", dict(n=len(lr.quarantined), srcs=",".join(lr.quarantined)[:80], trace=trace.trace_id)))
        if lr.retrieval_abstain or not lr.evidence or hard:
            RECORDER.incr("req.abstained")
            out.update(status="abstained", answer=ABSTAIN_MSG.format(reason=lr.abstain_reason or "low_confidence"), citations=[],
                       abstained=True, abstain_reason=lr.abstain_reason or ("evidence_quarantined" if lr.quarantined else "low_confidence"))
            self._bg(self._audit(req.tenant_id, session, req.user_id, "answer.abstained", dict(reason=(lr.abstain_reason or "low_confidence")[:40], trace=trace.trace_id)))
            return finish(out, route.value)

        ev = lr.evidence
        from .telemetry import add_span as _as
        _as("ttft", (time.perf_counter() - t0) * 1000)           # non-streaming generator: TTFT = time until generation starts
        with span("llm.generate"):
            text, llm_abstain, usage, fell_back = await self._generate(q, ev, lr.pairs, SETTINGS.strict_extractive, qvec)
        if fell_back:
            out["llm_fallback"] = True
        if llm_abstain:                                           # LLM answerability judge: evidence lacks the requested fact
            RECORDER.incr("req.abstained")
            RECORDER.incr("req.abstained.llm")
            out.update(status="abstained", answer=ABSTAIN_MSG.format(reason=f"llm_abstain:{llm_abstain}"), citations=[], abstained=True,
                       abstain_reason=f"llm_abstain:{llm_abstain}")
            self._bg(self._audit(req.tenant_id, session, req.user_id, "answer.abstained", dict(reason="llm_abstain", trace=trace.trace_id)))
            return finish(out, route.value)
        draft = Draft(text, [dict(text=e.text, source_id=e.src) for e in ev])
        with span("hook.post.ragas"):
            self.plugins.post(ctx, draft)
        if draft.veto:                                             # Verification subagent vetoes ungrounded draft
            RECORDER.incr("verify.vetoed")
            text2, ab2, usage2, fb2 = await self._generate(q, ev, lr.pairs, True, qvec)
            d2 = Draft(text2, draft.chunks)
            if ab2 or not text2:
                d2.veto = "llm_abstain_on_retry"
            for k_, v_ in usage2.items():
                usage[k_] = usage.get(k_, 0) + v_
            self.plugins.post(ctx, d2)
            if d2.veto:
                out.update(status="abstained", answer=ABSTAIN_MSG.format(reason="verification_failed"), citations=[], abstained=True,
                           abstain_reason="verification_failed")
                return finish(out, route.value)
            draft = d2
        cites = list(dict.fromkeys(e.src for e in ev if f"[{e.src}]" in draft.text))
        ctx_tokens = approx_tokens(anchor.render()) + approx_tokens(lr.observation) + approx_tokens(q)
        base_tokens = approx_tokens(q) + lr.tokens_raw
        out.update(status="answered", answer=draft.text, citations=cites, abstained=False, faithfulness=round(draft.faithfulness, 3),
                   tokens=dict(baseline=base_tokens, context=ctx_tokens, output=approx_tokens(draft.text),
                               **({"llm_prompt": usage["prompt"], "llm_completion": usage["completion"]} if usage else {})))
        RECORDER.sample("tok_base", base_tokens)
        RECORDER.sample("tok_ctx", ctx_tokens)
        RECORDER.sample("tok_out", approx_tokens(draft.text))
        RECORDER.sample("faithfulness", draft.faithfulness)
        # 6) cache store: tenant scope only for shared, non-personalised, <= tier-2 content (ADR-003)
        shared = all(e.risk_tier <= 2 for e in ev) and not ctx.redactions and not self.mem.get(req.tenant_id, session)[:-1]
        self.cache.store(req.tenant_id, req.user_id, q, {k: out[k] for k in ("status", "answer", "citations", "abstained", "faithfulness",
                                                                            "triage", "tokens", "route") if k in out} | dict(route=route.value),
                         "tenant" if shared else "user")
        # 7) off-critical-path: audit + facts + drift
        self._bg(self._after(req, session, q, qvec, draft, ents, trace.trace_id, route.value))
        return finish(out, route.value)

    # ------------------------------------------------------------------------------------------------ background
    async def _audit(self, tenant, session, user, event, detail) -> None:
        try:
            m = Meta(traceparent=new_traceparent(), tenant_id=tenant, user_id=user, session_id=session, subagent="verification", role="operator")
            await self.h.execute("audit_append", dict(event=event, detail={k: (v if isinstance(v, (str, int, float, bool)) else str(v)) for k, v in detail.items()}), m)
        except Exception:
            RECORDER.incr("audit.failed")

    async def _after(self, req, session, q, qvec, draft: Draft, ents, trace_id, route) -> None:
        qh = hashlib.sha256(q.encode()).hexdigest()[:16]          # hash only: no raw text in the audit chain
        await self._audit(req.tenant_id, session, req.user_id, "answer.completed",
                          dict(route=route, q_hash=qh, faith=round(draft.faithfulness, 3), trace=trace_id))
        m = Meta(traceparent=new_traceparent(), tenant_id=req.tenant_id, user_id=req.user_id, session_id=session, subagent="verification")
        if ents and draft.faithfulness >= 0.92:
            await self.h.execute("put_session_fact", dict(key="last_entities", value=",".join(ents)[:200]), m)
        # goal-alignment drift (KPI-18): cosine(goal, final answer) - computed off the critical path
        try:
            av = await asyncio.to_thread(self.emb.query, draft.text[:400])
            RECORDER.sample("drift", float(np.dot(qvec, av)))
        except Exception:
            pass
