"""Multi-tab Streamlit developer workbench (Gate 18).  Run:  .venv/bin/streamlit run app.py

Everything here drives the REAL runtime (plugins, MCP host, harness, orchestrator, state store, cache) in-process -
no mocked widgets. Charts that need a 50K-VU run read the last real measurement from eval/results/.
"""
from __future__ import annotations

import asyncio
import json
import re
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="Unreliable Assistant - Developer Workbench", layout="wide")


# ------------------------------------------------------------------------------------------------ runtime handle
class Handle:
    """Owns a background asyncio loop + the runtime; Streamlit reruns the script but this survives."""

    def __init__(self):
        from ua.runtime import build_runtime
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, daemon=True).start()
        self.rt = self.run(build_runtime(sqlite_path=str(Path(tempfile.gettempdir()) / "ua_workbench.db")))

    def run(self, coro, timeout=120):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)


@st.cache_resource(show_spinner="Starting runtime (models, MCP host, plugins)...")
def handle() -> Handle:
    return Handle()


H = handle()
RT = H.rt
from ua.config import SETTINGS  # noqa: E402
from ua.contracts import TOOLS, Meta  # noqa: E402
from ua.orchestrator import QueryRequest  # noqa: E402
from ua.telemetry import RECORDER, new_traceparent  # noqa: E402
from ua.textnorm import norm_query  # noqa: E402

RES = ROOT / "eval" / "results"


def jload(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def M(tenant, user, session="wb", **kw) -> Meta:
    return Meta(traceparent=new_traceparent(), tenant_id=tenant, user_id=user, session_id=session, **kw)


st.title("The Unreliable Assistant - Enterprise Developer Workbench")
st.caption(f"profile `{SETTINGS.model_profile}` - budgets x{SETTINGS.budget_scale:g} on this CPU box (KPI thresholds are never scaled) - "
           f"retrieval backend `{RT.services['hybrid-rag'].engine.backend}` - **no LLM connected** (deterministic extractive generator)")
tabs = st.tabs(["1 Architecture & Workflow", "2 3-Stage Hybrid RAG Lab", "3 MCP & Plugin Inspector", "4 Memory & Graphify Tokens",
                "5 Red Team & Circuit Breaker", "6 36-KPI Scorecard & Load"])

# ================================================================================================ TAB 1
with tabs[0]:
    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("design.md contract checklist")
        dm = (ROOT / "design.md").read_text()
        checks = [("1.1 System Charter & Hard Boundaries", "## 1.1"), ("1.2 MCP Tool & Data Contracts", "## 1.2"), ("1.3 Distributed Topology & Data Flow", "## 1.3"),
                  ("1.4 End-to-End Latency Budget", "## 1.4"), ("1.5 KPI & SLA Gate Matrix (36 KPIs)", "## 1.5"), ("1.6 FMEA (12 rows)", "## 1.6"), ("1.7 ADRs", "## 1.7")]
        for name, key in checks:
            st.checkbox(name, value=key in dm, disabled=True, key=f"chk{key}")
        kpi_rows = len(re.findall(r"^\| KPI-\d\d", dm, re.M))
        fmea_rows = len(re.findall(r"^\| \d+ \| \*\*", dm, re.M))
        st.checkbox(f"36 KPIs present ({kpi_rows})", value=kpi_rows == 36, disabled=True, key="k36")
        st.checkbox(f"12 FMEA rows present ({fmea_rows})", value=fmea_rows == 12, disabled=True, key="f12")
        st.checkbox("design.md LOCKED", value="**LOCKED**" in dm, disabled=True, key="lock")
        png = ROOT / "docs" / "workflow_orchestration.png"
        if png.exists():
            from PIL import Image
            Image.MAX_IMAGE_PIXELS = None
            with Image.open(png) as im:
                st.checkbox(f"workflow_orchestration.png present ({im.size[0]}x{im.size[1]}, {round((im.info.get('dpi') or (0,))[0])} DPI)", value=True, disabled=True, key="png")
    with c2:
        st.subheader("Live triage console")
        q1 = st.text_input("Query", "Please quarantine lot 88008", key="tri_q")
        if q1:
            nq = norm_query(RT.plugins.get("pii_redactor").redact(q1)[0])
            t0 = time.perf_counter()
            qv = RT.embedder.query(nq)
            from ua.triage import Triage
            tr = Triage().classify(nq, qv)
            st.json(dict(route=tr.route.value, intent=tr.intent, domain=tr.domain, risk_tier=tr.risk_tier, complexity=tr.complexity,
                         confidence=round(tr.confidence, 3), low_conf_fallback_to_RAG=tr.fallback, ms=round((time.perf_counter() - t0) * 1000, 2)))
    st.image(str(png), caption="docs/workflow_orchestration.png (Graphviz, 300 DPI)") if png.exists() else st.warning("diagram missing")

    st.subheader("ReAct step-by-step trace")
    cc = st.columns([3, 1, 1, 1])
    q2 = cc[0].text_input("Run the full pipeline", "Who signs off on big purchases of the Tundra Gauze?", key="run_q")
    tenant = cc[1].text_input("tenant", "wbT")
    user = cc[2].text_input("user", "alice")
    if cc[3].button("Run", type="primary"):
        st.session_state["last"] = H.run(RT.orch.handle(QueryRequest(tenant, user, q2, f"wb-{uuid.uuid4().hex[:6]}", debug=True)))
    last = st.session_state.get("last")
    if last:
        st.markdown(f"**status** `{last['status']}` - **route** `{last.get('route')}` - **{last['latency_ms']} ms** - steps {last.get('steps')} - cache `{last.get('cache')}`")
        st.info(last.get("answer") or json.dumps(last.get("error")))
        if last.get("anchor"):
            st.code(last["anchor"], language="text")
        for s in last.get("react_trace", []):
            with st.expander(f"Step {s['step']}: {s['thought']}  [{' -> '.join(s['state'])}]"):
                st.write(f"prompt[0] = `{s['prompt_head']}` - prompt tokens {s['prompt_tokens']}")
                st.dataframe(pd.DataFrame(s["calls"]) if s["calls"] else pd.DataFrame([{"-": "no tool call (final)"}]), width="stretch")
                if s.get("observation"):
                    st.text(f"observation ({s['observation_tokens']} tok):\n{s['observation']}")
                if s.get("quarantined"):
                    st.error(f"quarantined evidence: {s['quarantined']}")

    st.subheader("Tier 3 HITL approve / reject queue")
    hc = st.columns([1, 1, 1])
    ht = hc[0].text_input("tenant", "wbT", key="h_t")
    rv = hc[1].text_input("reviewer id", "bob", key="h_r")
    if hc[2].button("Refresh queue"):
        pass
    lr = H.run(RT.harness.execute("hitl_list", {"status": "PENDING_APPROVAL"}, M(ht, rv, role="reviewer", subagent="reviewer")))
    items = (lr.data or {}).get("items", []) if lr.ok else []
    if not items:
        st.caption("No pending approvals. Try: *Please quarantine lot 88008* above (as alice), then approve here as bob.")
    for it in items:
        a, b, c = st.columns([5, 1, 1])
        a.write(f"`{it['approval_id'][:8]}` **{it['tool']}** {json.dumps(it['args'])} - requested by *{it['requested_by']}*")
        for col, dec in ((b, "approve"), (c, "reject")):
            if col.button(dec.title(), key=f"{dec}{it['approval_id']}"):
                r = H.run(RT.harness.execute("hitl_decide", {"approval_id": it["approval_id"], "decision": dec}, M(ht, rv, role="reviewer", subagent="reviewer")))
                st.toast(f"{dec}: ok={r.ok} {r.error.message if r.error else json.dumps(r.data)[:80]}")
                st.rerun()

# ================================================================================================ TAB 2
with tabs[1]:
    st.subheader("Stage 1 BM25 | Stage 2 bi-encoder | RRF | Stage 3 cross-encoder")
    eng = RT.services["hybrid-rag"].engine
    c = st.columns(6)
    q = st.text_input("Query", "What is the unit price of SKU-007?", key="lab_q")
    k1 = c[0].slider("BM25 k1", 0.2, 3.0, SETTINGS.rag.bm25_k1, 0.05)
    b = c[1].slider("BM25 b", 0.0, 1.0, SETTINGS.rag.bm25_b, 0.05)
    rrfk = c[2].slider("RRF k", 1, 200, SETTINGS.rag.rrf_k)
    thr = c[3].slider("CE threshold", 0.0, 1.0, SETTINGS.rag.threshold, 0.01)
    pool = c[4].slider("rerank pool", 4, 40, SETTINGS.rag.pool)
    topk = c[5].slider("top_k (each)", 5, 50, 50)
    if st.button("Retrieve", key="lab_go"):
        keep = (SETTINGS.rag.bm25_k1, SETTINGS.rag.bm25_b)
        SETTINGS.rag.bm25_k1, SETTINGS.rag.bm25_b = k1, b
        try:
            from ua.contracts import HybridRetrieveIn
            out = eng.retrieve(HybridRetrieveIn(query=q, rrf_k=rrfk, threshold=thr, pool=pool, top_k_sparse=topk, top_k_dense=topk), debug=True)
        finally:
            SETTINGS.rag.bm25_k1, SETTINGS.rag.bm25_b = keep
        cd = out.candidates
        ids = list(dict.fromkeys([x[0] for x in cd["rerank"]] + [x[0] for x in cd["rrf"]]))
        bm, de, rr, ce = (dict(cd[k]) for k in ("bm25", "dense", "rrf", "rerank"))
        rank = lambda d, i: (list(d).index(i) + 1) if i in d else None
        kept = {c_.chunk_id for c_ in out.chunks}
        df = pd.DataFrame([dict(chunk=i, **{"BM25 score": bm.get(i), "BM25 rank": rank(bm, i), "Bi-enc cosine": de.get(i), "Dense rank": rank(de, i),
                                            "RRF fused (0-1)": rr.get(i), "RRF rank": rank(rr, i), "Cross-enc score": ce.get(i),
                                            "kept": "yes" if i in kept else ""}) for i in ids])
        st.dataframe(df, width="stretch", hide_index=True)
        st.write(f"**abstain = {out.abstain}** ({out.abstain_reason}) - top score {out.top_score} - threshold used {cd['threshold']:.2f} - degraded {cd['degraded']}")
        sm = out.stage_ms
        fig = go.Figure(go.Bar(x=["BM25 (<12)", "Bi-enc (<25)", "RRF (<3)", "Rerank (<45)"], y=[sm["bm25"], sm["biencoder"], sm["rrf"], sm["rerank"]],
                               marker_color=["#2a7", "#2a7", "#2a7", "#c52"]))
        fig.update_layout(title="Stage latency (ms) - budgets in labels; rerank is hardware-bound on CPU", height=280)
        st.plotly_chart(fig, width="stretch")
        for ch in out.chunks:
            st.markdown(f"**{ch.chunk_id}** (CE {ch.ce_score}) - {ch.text[:300]}")

# ================================================================================================ TAB 3
with tabs[2]:
    st.subheader("MCP servers")
    st.dataframe(pd.DataFrame(RT.host.snapshot()), width="stretch", hide_index=True)
    rc = st.columns(3)
    srv = rc[0].selectbox("server", list(RT.host.status))
    if rc[1].button("Restart worker"):
        H.run(RT.host.restart(srv))
        st.rerun()
    if rc[2].button("Refresh"):
        st.rerun()
    st.subheader("JSON-RPC 2.0 payload inspector")
    pc = st.columns([1, 1, 1, 1])
    tool = pc[0].selectbox("tool", list(TOOLS))
    tn = pc[1].text_input("tenant", "wbT", key="i_t")
    sub = pc[2].selectbox("subagent", ["retrieval", "execution", "verification", "reviewer"], index=1)
    role = pc[3].selectbox("role", ["operator", "viewer", "reviewer"])
    sample = {"hybrid_retrieve": {"query": "unit price of SKU-007"}, "graph_subgraph": {"query": "Tundra Gauze"}, "lookup_item": {"sku": "SKU7"},
              "submit_form": {"form_type": "safety_release", "entity_id": "LOT 88008", "payload": {}}, "get_session_facts": {}}.get(tool, {})
    args_txt = st.text_area("arguments (raw JSON - the harness validates/repairs)", json.dumps(sample), height=90)
    if st.button("Send via ToolExecutionHarness"):
        try:
            args = json.loads(args_txt)
            meta = M(tn, "inspector", role=role, subagent=sub)
            env = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": tool, "arguments": {"args": args, "meta": meta.model_dump(mode="json")}}}
            r = H.run(RT.harness.execute(tool, args, meta))
            a, b2 = st.columns(2)
            a.markdown("**request (JSON-RPC envelope)**")
            a.json(env)
            b2.markdown("**harness result**")
            b2.json(json.loads(r.model_dump_json()))
        except Exception as e:
            st.error(f"{type(e).__name__}: {e}")
    st.subheader("Lifecycle plugins (pluggy) - hot toggle")
    for e in RT.plugins.status():
        cc = st.columns([2, 2, 1, 1, 1])
        cc[0].write(f"**{e['name']}**")
        cc[1].write(f"{e['stage']}  - priority {e['priority']}")
        cc[2].write("fail-closed" if e["fail_closed"] else "")
        new = cc[3].toggle("enabled", e["enabled"], key=f"tg{e['name']}")
        if new != e["enabled"]:
            ok = RT.plugins.set_enabled(e["name"], new)
            st.toast(f"{e['name']} -> {new}" if ok else "locked outside dev mode")
            st.rerun()
    st.caption("pii_redactor / injection_shield are hot-toggleable only because UA_DEV_MODE=1; production locks them (design 1.2).")

# ================================================================================================ TAB 4
with tabs[3]:
    gc = st.columns([1, 1])
    with gc[0]:
        st.subheader("Graphify subgraph (<= depth 2)")
        gq = st.text_input("Query / anchor", "How long is the return window for the Tundra Gauze?")
        depth = st.slider("depth", 1, 2, 2)
        sg = RT.services["graphify-vault"].graph.subgraph(gq, None, depth, 320)
        if sg["nodes"]:
            dot = "digraph G { rankdir=LR; node [shape=box, style=rounded];\n" + "".join(
                f'"{n}" [style="rounded,filled", fillcolor="{"#ffd" if n in sg["anchors"] else "#eef"}"];\n' for n in sg["nodes"]) + "".join(f'"{a}" -> "{b}";\n' for a, b in sg["edges"]) + "}"
            st.graphviz_chart(dot)
            st.caption(f"anchors {sg['anchors']} - {sg['tokens']} tokens (budget 320) - depth reached {sg['depth_reached']}")
            st.code(sg["text"], language="text")
        else:
            st.info("No anchor entity resolved for this query.")
    with gc[1]:
        st.subheader("Obsidian atomic note viewer")
        notes = sorted(p.stem for p in (ROOT / "knowledge_vault").glob("*.md"))
        nid = st.selectbox("note", notes, index=notes.index("SKU-007") if "SKU-007" in notes else 0)
        st.code((ROOT / "knowledge_vault" / f"{nid}.md").read_text(), language="markdown")
    st.subheader("Token savings & cost")
    tb, tc, to_ = RECORDER.window("tok_base", 1e9), RECORDER.window("tok_ctx", 1e9), RECORDER.window("tok_out", 1e9)
    if tb:
        red = 1 - sum(tc) / sum(tb)
        from ua.orchestrator import PRICE_IN, PRICE_OUT
        c1, c2 = st.columns(2)
        fig = go.Figure(go.Indicator(mode="gauge+number", value=red * 100, number={"suffix": "%"}, title={"text": f"prompt-token reduction (gate >= 45%, target 55%) over {len(tb)} requests"},
                                     gauge={"axis": {"range": [0, 100]}, "threshold": {"line": {"color": "red", "width": 3}, "value": 45}, "bar": {"color": "#2a7"}}))
        c1.plotly_chart(fig, width="stretch")
        base_cost = sum(tb) * PRICE_IN / 1e6 + sum(to_) * PRICE_OUT / 1e6
        our_cost = sum(tc) * PRICE_IN / 1e6 + sum(to_) * PRICE_OUT / 1e6
        c2.metric("modelled cost / 1,000 queries", f"${our_cost / len(tb) * 1000:.4f}", delta=f"{(our_cost / base_cost - 1) * 100:.0f}% vs naive RAG prompt", delta_color="inverse")
        c2.caption("Modelled at configurable $/M-token prices; no LLM is billed here.")
    else:
        st.info("Run a few queries in Tab 1 to populate token statistics.")
    st.subheader("Session state browser (SQLite / cache)")
    sc = st.columns(3)
    bt = sc[0].text_input("tenant", "wbT", key="b_t")
    bs = sc[1].text_input("session id", "wb", key="b_s")
    store = RT.services["state-store"].store
    st.write("**session_facts**", store.get_facts(bt, bs) or "(none)")
    st.write("**hot memory (last 3 turns)**", RT.orch.mem.get(bt, bs) or "(none)")
    keys = [k for k in RT.cache.backend.all_keys() if k.startswith(f"{{{bt}}}:")]
    st.write(f"**cache keys for tenant `{bt}`** ({len(keys)}; cache hit ratio {RT.cache.hit_ratio():.2f})")
    st.code("\n".join(keys[:15]) or "(none)")
    ok, n, bad = store.verify_chain(bt)
    st.write(f"**audit hash chain** for `{bt}`: {'VALID' if ok else f'BROKEN at seq {bad}'} ({n} records)")
    st.dataframe(pd.DataFrame(store.audit_tail(bt, 8)), width="stretch", hide_index=True)

# ================================================================================================ TAB 5
with tabs[4]:
    from eval import kpi_suites
    from eval.redteam_data import CRESCENDO, DIRECT, INDIRECT, PII_SMUGGLING

    st.subheader("One-click adversarial runner (real pipeline)")
    rc = st.columns(5)
    sid = lambda: f"rt-{uuid.uuid4().hex[:8]}"
    if rc[0].button("Direct injections"):
        rows = [dict(prompt=p[:80], blocked=H.run(RT.orch.handle(QueryRequest("rtT", "atk", p, sid())))["status"] == "blocked") for p in DIRECT]
        st.session_state["rt_res"] = ("Direct jailbreaks", rows)
    if rc[1].button("Indirect (poisoned chunk)"):
        svc, rows = RT.services["hybrid-rag"], []
        orig = svc.tools["hybrid_retrieve"]
        for i, t in enumerate(INDIRECT[:10]):
            def poisoned(a, m, _t=t):
                d = orig(a, m)
                if d["chunks"]:
                    d["chunks"][0]["text"] = _t
                return d
            svc.tools["hybrid_retrieve"] = poisoned
            r = H.run(RT.orch.handle(QueryRequest(f"rtI{i}", "victim", "What is the unit price of SKU-007?", sid())))
            rows.append(dict(poisoned_text=t[:70], quarantined=bool(r.get("quarantined")), unsafe_action=r["status"] == "pending_approval"))
        svc.tools["hybrid_retrieve"] = orig
        st.session_state["rt_res"] = ("Indirect injection in retrieved chunks", rows)
    if rc[2].button("Obfuscated PII"):
        red = RT.plugins.get("pii_redactor")
        rows = []
        for qq, bad in PII_SMUGGLING:
            o, _ = red.redact(qq)
            rows.append(dict(input=qq[:60], redacted=o[:60], clean=not any(b.lower() in o.lower() for b in bad)))
        st.session_state["rt_res"] = ("PII / homoglyph smuggling", rows)
    if rc[3].button("Infinite SKU retry loop"):
        from ua.agents import ChaosBrain
        from ua.loop import GoalAnchorBlock
        rows = []
        for kind in ("same_sku", "alternating", "unique", "refetch"):
            lr = H.run(RT.orch.loop.run("TRANSACTION_EXEC", "x", M("rtC", "atk", f"c-{kind}-{sid()}", subagent="execution"), GoalAnchorBlock("x", ()), time.perf_counter() + 60, brain=ChaosBrain(kind)))
            rows.append(dict(chaos=kind, end_state=lr.state.value, iterations=lr.steps, reason=lr.halted_reason))
        st.session_state["rt_res"] = ("Runaway tool loops", rows)
    if rc[4].button("Cross-tenant cache probe"):
        res = H.run(kpi_suites.isolation_probe(RT, tenants=20, users=3))
        st.session_state["rt_res"] = ("Cross-tenant cache probe", [res])
    if "rt_res" in st.session_state:
        title, rows = st.session_state["rt_res"]
        st.markdown(f"**{title}**")
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    st.subheader("Circuit breaker")
    cb = RT.plugins.get("circuit_breaker")
    snap = cb.snapshot()
    bc = st.columns(4)
    bc[0].metric("open circuits", len(snap["open"]))
    bc[1].metric("total trips", snap["trips"])
    if bc[2].button("Reset all breakers"):
        cb.reset()
        st.rerun()
    if bc[3].button("Trip: 3x identical failing call"):
        for _ in range(3):
            H.run(RT.harness.execute("lookup_item", {"sku": "SKU-999"}, M("rtB", "atk", "brk", subagent="execution")))
        st.rerun()
    st.dataframe(pd.DataFrame(snap["open"]) if snap["open"] else pd.DataFrame([{"state": "all circuits closed"}]), width="stretch", hide_index=True)

    st.subheader("Chaos & runbooks")
    cc = st.columns(4)
    hang = cc[0].toggle("Hang hybrid-rag", "hybrid-rag" in RT.host.fail_injected)
    (RT.host.fail_injected.add if hang else RT.host.fail_injected.discard)("hybrid-rag")
    rr_fail = cc[1].toggle("Reranker failure", RT.services["hybrid-rag"].engine.reranker_failing)
    RT.services["hybrid-rag"].engine.reranker_failing = rr_fail
    lat = cc[2].slider("inject state-store latency (ms)", 0, 500, int(RT.host.latency_injected.get("state-store", 0)))
    RT.host.latency_injected["state-store"] = lat
    rb = st.columns(5)
    for i, name in enumerate(["latency_breach", "hallucination_surge", "mcp_hang", "cache_stampede"]):
        if rb[i].button(f"Apply {name}"):
            st.toast(H.run(RT.runbooks.apply(name, "hybrid-rag" if name == "mcp_hang" else "wbT")))
    if rb[4].button("Revert all"):
        [RT.runbooks.revert(n) for n in list(RT.runbooks.active)]
    st.dataframe(pd.DataFrame(RT.runbooks.evaluate_alerts()), width="stretch", hide_index=True)

# ================================================================================================ TAB 6
with tabs[5]:
    st.subheader("OpenTelemetry span waterfall (last traced request)")
    trs = RECORDER.recent_traces(1)
    if trs and trs[0]:
        sp = sorted(trs[0], key=lambda s: s.start_ns)
        t0 = sp[0].start_ns
        fig = go.Figure(go.Bar(y=[s.name for s in sp][::-1], x=[s.ms for s in sp][::-1], base=[(s.start_ns - t0) / 1e6 for s in sp][::-1], orientation="h"))
        fig.update_layout(height=60 + 26 * len(sp), xaxis_title="ms since request start")
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("Run a query in Tab 1.")
    sc = jload(RES / "kpi_scorecard.json")
    st.subheader("RAGAS-style triad (golden set, last scorecard run)")
    if sc:
        g, h = sc["suites"]["ragas_golden"], sc["suites"]["ragas_heldout"]
        gg = st.columns(5)
        for col, (lab, key, gate) in zip(gg, [("Faithfulness", "faithfulness", 0.92), ("Answer relevancy", "answer_relevancy", 0.90), ("Ctx precision", "context_precision", 0.88),
                                              ("Ctx recall", "context_recall", 0.85), ("Entity recall", "context_entity_recall", 0.85)]):
            v = g[key]
            col.plotly_chart(go.Figure(go.Indicator(mode="gauge+number", value=v, title={"text": lab}, gauge={"axis": {"range": [0, 1]}, "threshold": {"value": gate, "line": {"color": "red", "width": 3}},
                                                                                                         "bar": {"color": "#2a7" if v >= gate else "#c33"}})).update_layout(height=200, margin=dict(t=40, b=0)),
                             width="stretch")
            col.caption(f"held-out: {h[key]:.2f}")
        st.caption("RAGAS-STYLE deterministic proxies (no LLM judge). Held-out = hand-written paraphrases not used for tuning.")
    st.subheader("Load test: latency histogram vs 200-600 ms SLA")
    ld = jload(RES / "load.json")
    if ld and ld.get("hist"):
        e, cn = ld["hist"]["edges"], ld["hist"]["counts"]
        fig = go.Figure(go.Bar(x=[(e[i] + e[i + 1]) / 2 for i in range(len(cn))], y=cn, name="requests"))
        for val, lab, col in ((ld["p50_ms"], "p50", "#27a"), (ld["p95_ms"], "p95", "#e90"), (ld["p99_ms"], "p99", "#c33")):
            fig.add_vline(x=val, line_color=col, annotation_text=f"{lab} {val:.0f}")
        fig.add_vrect(x0=200, x1=600, fillcolor="green", opacity=0.08, annotation_text="200-600 ms SLA")
        fig.update_layout(height=320, xaxis_title="ms")
        st.plotly_chart(fig, width="stretch")
        st.warning(f"Measured with **{ld['vus']} VUs** on this 4-vCPU box ({ld['rps']} RPS). This is NOT a 50,000-VU result and cannot certify KPI-31..36; "
                   f"scripts for the real run: load_tests/k6_50k_load.js, load_tests/jmeter_50k_plan.jmx.")
    else:
        st.info("No load.json yet - run `python -m eval.run_kpis`.")
    st.subheader("36-KPI release scorecard")
    if sc:
        df = pd.DataFrame(sc["scorecard"])[["id", "stage", "name", "op", "threshold", "value", "status", "risk", "note"]]
        cnt = df["status"].str.split(" ").str[0].value_counts().to_dict()
        st.write(cnt)
        st.dataframe(df.style.map(lambda s: "background-color:#fcc" if str(s).startswith("FAIL") else ("background-color:#cfc" if str(s).startswith("PASS") else "background-color:#eee"), subset=["status"]),
                     width="stretch", hide_index=True, height=700)
    if st.button("Re-run scorecard (background, ~12 min)"):
        subprocess.Popen([sys.executable, "-m", "eval.run_kpis"], cwd=ROOT)
        st.toast("started - refresh when done")
