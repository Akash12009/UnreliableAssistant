"""Generate docs/workflow_orchestration.png (Gate 2) with Graphviz at 300 DPI.

Usage: python scripts/generate_workflow_diagram.py
Requires the Graphviz `dot` binary and the `graphviz` Python package.
"""
from pathlib import Path

from graphviz import Digraph

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "workflow_orchestration"

C = {  # one fill colour per layer
    "sec": "#FDE2E2", "triage": "#E3F0FF", "mcp": "#E5F6E3",
    "ctrl": "#FFF3D6", "store": "#EDE7F6", "edge": "#F2F2F2",
}

g = Digraph("workflow_orchestration", format="png")
g.attr(rankdir="TB", dpi="300", fontname="Helvetica", fontsize="22", labelloc="t",
       label="The Unreliable Assistant → Enterprise Grade — Workflow Orchestration\n"
             "p95 ≤ 450 ms · p99 ≤ 590 ms @ 50K VUs · MCP over JSON-RPC 2.0",
       nodesep="0.35", ranksep="0.55", compound="true", pad="0.4")
g.attr("node", shape="box", style="rounded,filled", fontname="Helvetica", fontsize="13",
       fillcolor="white", margin="0.18,0.10")
g.attr("edge", fontname="Helvetica", fontsize="11")


def node(graph, name, label, fill, **kw):
    graph.node(name, label, fillcolor=fill, **kw)


# ---------------------------------------------------------------- 1. Ingress & Security
with g.subgraph(name="cluster_ingress") as s:
    s.attr(label="1 · INGRESS & SECURITY LAYER", style="rounded,filled", fillcolor=C["sec"],
           fontsize="16", fontname="Helvetica-Bold")
    node(s, "client", "Client / k6 / JMeter\n(tenant_id, user_id, traceparent)", C["edge"], shape="oval")
    node(s, "gw", "NGINX / Envoy API Gateway\n→ stateless FastAPI workers", "white")
    node(s, "pii", "Pre-Execution Plugin 1\nPII Redactor (fail-closed)\nKPI-02 ≥ 99.8%", "white")
    node(s, "inj", "Pre-Execution Plugin 2\nInjection Classifier (fail-closed)\nKPI-03 ≥ 99.5%", "white")
    node(s, "cache", "Redis Semantic Cache Check\nkey = SHA256(tenant:user:norm_query)\nTTL 300s ± 45s · KPI-28: 0 leaks", C["store"],
         shape="cylinder")
    s.edge("client", "gw")
    s.edge("gw", "pii")
    s.edge("pii", "inj", label=" hook.pre ≤ 15 ms")
    s.edge("inj", "cache")

# ---------------------------------------------------------------- 2. Triage & Subagents
with g.subgraph(name="cluster_triage") as s:
    s.attr(label="2 · TRIAGE & SUBAGENT ROUTING", style="rounded,filled", fillcolor=C["triage"],
           fontsize="16", fontname="Helvetica-Bold")
    node(s, "triage", "Triage Agent  (temperature = 0.0, ≤ 25 ms)\n→ (intent, domain, risk_tier, complexity)", "white",
         shape="diamond", margin="0.05,0.05")
    node(s, "direct", "Direct Response\n(DIRECT_ANSWER)", "white")
    node(s, "oos", "Reject\n(REJECT_OOS)", "white")
    node(s, "retr", "Retrieval Subagent\nread-only · HYBRID_RAG", "white")
    node(s, "exec", "Execution Subagent\nno RAG / web · TRANSACTION_EXEC", "white")
    node(s, "audit", "Audit (Verification) Subagent\nNLI claim check · independent context", "white")

# ---------------------------------------------------------------- 3. MCP & RAG
with g.subgraph(name="cluster_mcp") as s:
    s.attr(label="3 · MCP (JSON-RPC 2.0)  &  3-STAGE HYBRID RAG PIPELINE", style="rounded,filled",
           fillcolor=C["mcp"], fontsize="16", fontname="Helvetica-Bold")
    node(s, "harness", "ToolExecutionHarness\n① Pydantic v2 strict → ② regex auto-repair\n③ RBAC tier → ④ UUIDv4 idempotency", "white")
    with s.subgraph(name="cluster_rag") as r:
        r.attr(label="mcp-hybrid-rag  (hybrid_retrieve · rag://chunks/{id})", style="rounded,filled",
               fillcolor="#F6FFF4", fontsize="13")
        node(r, "bm25", "Stage 1 · BM25Okapi\nk1=1.25 b=0.75 top-50 · < 12 ms", "white")
        node(r, "dense", "Stage 2 · Bi-Encoder + FAISS HNSW\nefSearch=64 top-50 · < 25 ms", "white")
        node(r, "rrf", "RRF Fusion  k=60\ndedup chunk_hash · pool 40 · < 3 ms", "white")
        node(r, "ce", "Stage 3 · Cross-Encoder (ONNX FP16)\nrerank_k=4 · threshold > 0.42 · < 45 ms", "white")
        r.edge("bm25", "rrf")
        r.edge("dense", "rrf")
        r.edge("rrf", "ce")
    node(s, "graph", "mcp-graphify-vault\ndepth ≤ 2 subgraph (~320 vs ~1,400 tokens)", "white")
    node(s, "state", "mcp-state-store\nsession_facts · audit · HITL", "white")

node(g, "pg", "PostgreSQL + pgvector\n(tenant_id on every query)\nhash-chained audit log", C["store"], shape="cylinder")
node(g, "abstain", "Deterministic Abstention\nCE top score < 0.42 → no LLM\n< 90 ms from retrieval start", "#FFE0B2")

# ---------------------------------------------------------------- 4. Control plane & Egress
with g.subgraph(name="cluster_ctrl") as s:
    s.attr(label="4 · CONTROL PLANE & EGRESS", style="rounded,filled", fillcolor=C["ctrl"],
           fontsize="16", fontname="Helvetica-Bold")
    node(s, "anchor", "GoalAnchorBlock @ index 0\nObservation compress ≤ 180 tok · last 3 turns hot", "white")
    node(s, "react", "Bounded ReAct Loop  (max_steps = 5)\nINIT→TRIAGE→REASONING→TOOL_CALL→OBSERVATION→VERIFY→DONE/HALTED\nstep 120 ms · loop 450 ms", "white")
    node(s, "cb", "Circuit Breaker\n(tool, args_hash) fails ≥ 2 → HALTED_BREAKER", "white")
    node(s, "hitl", "Tier 3 HITL Gate\nPENDING_APPROVAL → human approve/reject", "#FFD6D6", shape="octagon")
    node(s, "llm", "LLM (vLLM / TRT-LLM)\ncontinuous batching · prefix cache · TTFT ≤ 180 ms", "white")
    node(s, "post", "Post-Execution\nRAGAS (async/sampled) + NLI veto", "white")
    node(s, "otel", "OpenTelemetry Export\nW3C traceparent → Collector → Prometheus · Tempo · Grafana", "white")
    node(s, "resp", "Response to client\n(cited, grounded)", C["edge"], shape="oval")

# ---------------------------------------------------------------- Edges (cross-cluster)
g.edge("cache", "triage", label=" miss")
g.edge("cache", "resp", label=" hit (≥ 35%)", style="dashed", constraint="false")
g.edge("triage", "direct", label="DIRECT_ANSWER")
g.edge("triage", "oos", label="REJECT_OOS")
g.edge("triage", "retr", label="HYBRID_RAG")
g.edge("triage", "exec", label="TRANSACTION_EXEC")
g.edge("triage", "audit", label="audit", style="dotted")

g.edge("retr", "harness", label=" tools/call")
g.edge("exec", "harness")
g.edge("harness", "bm25", label=" parallel")
g.edge("harness", "dense", label=" parallel")
g.edge("harness", "graph")
g.edge("harness", "state")
g.edge("ce", "abstain", label=" score < 0.42", color="#E65100", fontcolor="#E65100")
g.edge("state", "pg")

g.edge("ce", "anchor", label=" top-4 verified chunks")
g.edge("graph", "anchor", label=" subgraph")
g.edge("direct", "anchor", style="dashed")
g.edge("anchor", "react")
g.edge("react", "cb")
g.edge("cb", "llm", label=" ok")
g.edge("cb", "hitl", label=" Tier 3 call", color="#C62828", fontcolor="#C62828")
g.edge("hitl", "pg", label=" PENDING_APPROVAL", style="dashed")
g.edge("llm", "react", label=" observation", constraint="false", style="dashed")
g.edge("llm", "post")
g.edge("audit", "post", style="dotted")
g.edge("post", "react", label=" veto / re-verify", constraint="false", color="#6A1B9A", fontcolor="#6A1B9A")
g.edge("post", "otel", style="dashed")
g.edge("post", "resp")
g.edge("abstain", "resp", color="#E65100")
g.edge("oos", "resp")

if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    path = g.render(filename=str(OUT), cleanup=True)
    from PIL import Image  # stamp 300 DPI into PNG metadata (pHYs chunk)
    with Image.open(path) as im:
        im.load()
        im.save(path, dpi=(300, 300))
    print("wrote", path)
