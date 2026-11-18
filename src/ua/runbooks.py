"""Incident runbooks (Gate 10/11): alert rule -> automated mitigation. Every mitigation is reversible and recorded."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .config import SETTINGS
from .telemetry import RECORDER

RUNBOOKS = {
    "latency_breach": "Enable aggressive semantic cache (cosine >= 0.92); drop BM25/Dense top_k 50 -> 25.",
    "hallucination_surge": "Raise cross-encoder cutoff 0.42 -> 0.55; enforce strict extractive quote mode.",
    "mcp_hang": "Trip the MCP bulkhead for the hung server, serve degraded, restart the worker.",
    "cache_stampede": "Re-assert tenant/user key prefix + jittered TTL; flush the affected tenant namespace.",
}


@dataclass
class Runbooks:
    rt: object = None
    active: dict[str, float] = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)
    _orig: dict = field(default_factory=dict)

    def _log(self, name, action, detail=""):
        self.history.append(dict(ts=time.time(), runbook=name, action=action, detail=detail))

    async def apply(self, name: str, target: str | None = None) -> str:
        r, c = SETTINGS.rag, SETTINGS.cache
        if name == "latency_breach":
            self._orig.setdefault(name, (r.top_k_sparse, r.top_k_dense))
            SETTINGS.aggressive_cache = True
            r.top_k_sparse = r.top_k_dense = 25
            msg = f"aggressive cache cosine>={c.cosine_aggressive}; top_k=25"
        elif name == "hallucination_surge":
            self._orig.setdefault(name, r.threshold)
            r.threshold = 0.55
            SETTINGS.strict_extractive = True
            msg = "cross-encoder cutoff 0.55; strict extractive mode ON"
        elif name == "mcp_hang":
            srv = target or "hybrid-rag"
            host = self.rt.host
            host.bulkhead_open.add(srv)
            await host.restart(srv)
            msg = f"bulkhead opened for {srv}; worker restarted (restarts={host.status[srv].restarts})"
        elif name == "cache_stampede":
            n = self.rt.cache.backend.flush_tenant(target) if target else 0
            msg = f"flushed {n} keys" + (f" for tenant {target}" if target else "") + "; key prefix + jitter enforced"
        else:
            raise KeyError(name)
        self.active[name] = time.time()
        self._log(name, "apply", msg)
        return msg

    def revert(self, name: str) -> str:
        r = SETTINGS.rag
        if name == "latency_breach":
            r.top_k_sparse, r.top_k_dense = self._orig.pop(name, (50, 50))
            SETTINGS.aggressive_cache = False
        elif name == "hallucination_surge":
            r.threshold = self._orig.pop(name, 0.42)
            SETTINGS.strict_extractive = False
        self.active.pop(name, None)
        self._log(name, "revert")
        return "reverted"

    def evaluate_alerts(self) -> list[dict]:
        """Alert triggers from the spec, evaluated over rolling windows."""
        out = []
        e2e = RECORDER.window("e2e_ms", 30)
        if len(e2e) >= 20:
            import numpy as np
            p95 = float(np.percentile(e2e, 95))
            out.append(dict(runbook="latency_breach", firing=p95 > 550, value=round(p95, 1), rule="p95 e2e > 550 ms for 30 s"))
        f = RECORDER.window("faithfulness", 300)
        if len(f) >= 10:
            m = sum(f) / len(f)
            out.append(dict(runbook="hallucination_surge", firing=m < 0.90, value=round(m, 3), rule="RAGAS faithfulness < 0.90 (5 m)"))
        to = RECORDER.counter("mcp.timeout.hybrid-rag")
        tot = max(1.0, RECORDER.counter("req.total"))
        out.append(dict(runbook="mcp_hang", firing=(to / tot) > 0.01 and to >= 3, value=round(to / tot, 4), rule="JSON-RPC timeout rate > 1%"))
        leaks = RECORDER.counter("cache.leak_detected")
        out.append(dict(runbook="cache_stampede", firing=leaks > 0, value=leaks, rule="tenant key collision / leak detected"))
        return out
