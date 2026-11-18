"""RAGAS-STYLE evaluation of the full pipeline over the golden set (design Gate 8).

IMPORTANT: no judge LLM is available offline, so these are deterministic PROXIES of the RAGAS metrics, not the `ragas`
library's LLM-judged scores. Definitions (all computed on the real orchestrator output):
  faithfulness            = fraction of answer claims supported by the retrieved contexts (lexical NLI proxy, ids/numbers exact)
  answer_relevancy        = key-fact coverage of the answer (answerable questions)
  context_precision       = RAGAS rank-weighted precision of contexts vs gold notes
  context_recall          = fraction of reference key-facts present in the contexts
  context_entity_recall   = fraction of reference entities (gold note ids + key facts) present in the contexts
A faithful-by-construction extractive generator inflates faithfulness; do not read it as LLM quality.
"""
from __future__ import annotations

import asyncio
import json
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ua.orchestrator import QueryRequest  # noqa: E402
from ua.verify import claim_support, split_claims  # noqa: E402

from .metrics import context_precision  # noqa: E402


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().replace(",", ""))


def _has(text: str, fact: str) -> bool:
    t, f = _norm(text), _norm(fact)
    return f in t or f.replace(" months", "").replace(" days", "").replace(" years", "") in t


async def evaluate(rt, name: str = "golden_set", limit: int | None = None) -> dict:
    G = [json.loads(l) for l in (ROOT / "data" / f"{name}.jsonl").read_text().splitlines() if l.strip()]
    if limit:
        G = G[:limit]
    rows = []
    for g in G:
        r = await rt.orch.handle(QueryRequest(f"ev_{name}_{g['qid']}", "evaluator", g["question"], f"sess-{g['qid']}", debug=True))
        ctx = r.get("contexts", [])
        ctx_text = " \n ".join(c["text"] for c in ctx)
        ans = r.get("answer", "")
        answered = r["status"] == "answered"
        if g["type"] == "negative":
            rows.append(dict(qid=g["qid"], type="negative", abstained=r["status"] in ("abstained", "rejected_oos", "blocked"), status=r["status"],
                             faith=claim_support(ans, [c["text"] for c in ctx])[0] if answered else 1.0,
                             n_claims=len(split_claims(ans)) if answered else 0,
                             n_unsupported=len(claim_support(ans, [c["text"] for c in ctx])[1]) if answered else 0))
            continue
        kf = g["key_facts"]
        ents = set(g["gold_chunks"]) | set(kf)
        faith, unsup = claim_support(ans, [c["text"] for c in ctx]) if answered else (1.0, [])
        rows.append(dict(
            qid=g["qid"], type=g["type"], status=r["status"], answered=answered,
            faith=faith, n_claims=len(split_claims(ans)) if answered else 0, n_unsupported=len(unsup),
            relevancy=(sum(_has(ans, f) for f in kf) / len(kf)) if (answered and kf) else 0.0,
            ctx_precision=context_precision([c["src"] for c in ctx], set(g["gold_chunks"])),
            ctx_recall=(sum(_has(ctx_text, f) for f in kf) / len(kf)) if kf else 1.0,
            ent_recall=(sum(_has(ctx_text, e) for e in ents) / len(ents)) if ents else 1.0,
            steps=r.get("steps", 0), latency=r["latency_ms"], answer=ans[:200]))
    ans_rows = [x for x in rows if x["type"] != "negative"]
    neg = [x for x in rows if x["type"] == "negative"]
    answered = [x for x in rows if x.get("status") == "answered"]
    mean = lambda k, rs: sum(x[k] for x in rs) / len(rs) if rs else float("nan")
    agg = dict(
        set=name, n=len(rows), n_answerable=len(ans_rows), n_negative=len(neg),
        faithfulness=mean("faith", answered) if answered else 1.0,
        answer_relevancy=mean("relevancy", ans_rows), context_precision=mean("ctx_precision", ans_rows),
        context_recall=mean("ctx_recall", ans_rows), context_entity_recall=mean("ent_recall", ans_rows),
        abstention_accuracy=sum(x["abstained"] for x in neg) / len(neg) if neg else float("nan"),
        false_abstain_rate=sum(1 for x in ans_rows if not x["answered"]) / len(ans_rows),
        hallucination_rate=(sum(x["n_unsupported"] for x in answered) / max(1, sum(x["n_claims"] for x in answered))),
        mean_steps=mean("steps", [x for x in ans_rows if x["answered"]]),
        by_type={t: dict(relevancy=mean("relevancy", [x for x in ans_rows if x["type"] == t]),
                         ctx_recall=mean("ctx_recall", [x for x in ans_rows if x["type"] == t]),
                         ctx_precision=mean("ctx_precision", [x for x in ans_rows if x["type"] == t]),
                         ent_recall=mean("ent_recall", [x for x in ans_rows if x["type"] == t])) for t in ("factoid", "complex", "multihop", "summary")})
    return dict(aggregate=agg, rows=rows)


async def _main():
    from ua.runtime import build_runtime
    rt = await build_runtime(sqlite_path=f"{tempfile.mkdtemp()}/e.db")
    out = {}
    for nm in ("golden_set", "heldout_set"):
        out[nm] = await evaluate(rt, nm)
        a = out[nm]["aggregate"]
        print(nm, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in a.items() if k != "by_type"})
        print("   by type", {t: {k: round(v, 2) for k, v in d.items()} for t, d in a["by_type"].items()})
    (ROOT / "eval" / "results").mkdir(exist_ok=True)
    (ROOT / "eval" / "results" / "ragas.json").write_text(json.dumps(out, indent=1))
    await asyncio.sleep(0.5)
    await rt.close()

if __name__ == "__main__":
    asyncio.run(_main())
