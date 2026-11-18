"""IR benchmark over the golden set: recall@50 (pre-rerank), precision@4, MRR@4, nDCG@4, abstention accuracy.

usage: python -m eval.ir_bench [--pool N] [--quant int8|fp32] [--out eval/results/ir.json]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ua.config import SETTINGS  # noqa: E402
from ua.contracts import HybridRetrieveIn  # noqa: E402
from ua.rag import index as idxmod  # noqa: E402
from ua.rag.encoders import load_models  # noqa: E402
from ua.rag.engine import RagEngine  # noqa: E402

from .metrics import context_precision, mrr_at_k, ndcg_at_k, pctl, recall_at_k  # noqa: E402


def load_golden(name: str = "golden_set"):
    return [json.loads(l) for l in (ROOT / "data" / f"{name}.jsonl").read_text().splitlines() if l.strip()]


def run(engine: RagEngine, golden, pool: int) -> dict:
    rows, lat = [], []
    for g in golden:
        out = engine.retrieve(HybridRetrieveIn(query=g["question"], pool=pool), debug=True)
        lat.append(out.stage_ms)
        gold = set(g["gold_chunks"])
        c = out.candidates
        pre = [x[0].split("#")[0] for x in c["rrf"]]                    # fused candidates before rerank (pool)
        sparse = [x[0].split("#")[0] for x in c["bm25"]]
        dense = [x[0].split("#")[0] for x in c["dense"]]
        post = [ch.source_id for ch in out.chunks]
        ranked_all = [x[0].split("#")[0] for x in c["rerank"]]          # reranker ordering regardless of threshold
        rows.append(dict(qid=g["qid"], type=g["type"], gold=sorted(gold), abstain=out.abstain, top=out.top_score,
                         recall50=recall_at_k(list(dict.fromkeys(sparse + dense)), gold, 100),
                         recall_pool=recall_at_k(pre, gold, len(pre)),
                         p4=context_precision(post, gold) if gold else None,
                         mrr4=mrr_at_k(ranked_all, gold, 4) if gold else None,
                         ndcg4=ndcg_at_k(ranked_all, gold, 4) if gold else None,
                         post=post, reason=c.get("abstain_reason")))
    ans = [r for r in rows if r["type"] != "negative"]
    neg = [r for r in rows if r["type"] == "negative"]
    agg = dict(
        n=len(rows), backend=engine.backend, pool=pool,
        recall50=sum(r["recall50"] for r in ans) / len(ans),
        recall_pool=sum(r["recall_pool"] for r in ans) / len(ans),
        context_precision4=sum(r["p4"] for r in ans) / len(ans),
        mrr4=sum(r["mrr4"] for r in ans) / len(ans),
        ndcg4=sum(r["ndcg4"] for r in ans) / len(ans),
        abstain_accuracy=sum(r["abstain"] for r in neg) / len(neg),
        false_abstain_rate=sum(r["abstain"] for r in ans) / len(ans),
        latency_p95_ms={k: pctl([l[k] for l in lat], 95) for k in lat[0]},
        latency_p50_ms={k: pctl([l[k] for l in lat], 50) for k in lat[0]},
        by_type={t: dict(recall_pool=sum(r["recall_pool"] for r in ans if r["type"] == t) / max(1, sum(1 for r in ans if r["type"] == t)),
                         ndcg4=sum(r["ndcg4"] for r in ans if r["type"] == t) / max(1, sum(1 for r in ans if r["type"] == t)),
                         false_abstain=sum(r["abstain"] for r in ans if r["type"] == t) / max(1, sum(1 for r in ans if r["type"] == t)))
                 for t in ("factoid", "complex", "multihop", "summary")})
    return dict(aggregate=agg, rows=rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, default=SETTINGS.rag.pool)
    ap.add_argument("--quant", default=None)
    ap.add_argument("--set", default="golden_set")
    ap.add_argument("--out", default=str(ROOT / "eval" / "results" / "ir.json"))
    a = ap.parse_args()
    enc, rr = load_models(quant=a.quant)
    eng = RagEngine(encoder=enc, reranker=rr)
    t = time.time()
    res = run(eng, load_golden(a.set), a.pool)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res["aggregate"], indent=1), f"\n({time.time() - t:.0f}s)")
