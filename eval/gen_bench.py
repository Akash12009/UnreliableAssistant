"""Generator-only benchmark: GOLD evidence in -> key-fact coverage out (isolates generation from retrieval)."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from ua.agents import Evidence, ExtractiveLLM, compress_observation
from ua.embedder import Embedder
from ua.rag.encoders import load_models
from ua.rag.index import parse_note
from ua.textnorm import approx_tokens
from ua.verify import claim_support

def main(name="golden_set", prefixed=True):
    enc, _ = load_models(); emb = Embedder(enc)
    notes = {p.stem: parse_note(p) for p in (ROOT / "knowledge_vault").glob("*.md")}
    G = [json.loads(l) for l in (ROOT / "data" / f"{name}.jsonl").read_text().splitlines() if l.strip()]
    llm, cov, per, faith, toks = ExtractiveLLM(), [], {}, [], []
    for g in G:
        if g["type"] == "negative": continue
        ev = [Evidence(i, notes[i]["body"], notes[i]["risk_tier"], r) for r, i in enumerate(g["gold_chunks"])]
        qv = enc.encode([g["question"]], is_query=prefixed)[0]
        txt, pairs, raw = compress_observation(g["question"], ev, qvec=qv, emb=emb)
        ans = llm.generate(g["question"], ev, pairs, qvec=qv, emb=emb)
        hit = [f.lower() in ans.lower() or f.lower().replace(" months", "") in ans.lower() for f in g["key_facts"]]
        c = sum(hit) / len(hit) if hit else 1.0
        cov.append(c); per.setdefault(g["type"], []).append(c); toks.append(approx_tokens(ans))
        faith.append(claim_support(ans, [e.text for e in ev])[0])
        if c < 1 and len(sys.argv) > 1 and sys.argv[1] == "-v": print(f"  MISS {g['qid']} {g['question']} facts={g['key_facts']} -> {ans[:140]}")
    print(f"{name} prefixed={prefixed}: key-fact coverage {sum(cov)/len(cov):.3f} | by type", {k: round(sum(v)/len(v), 2) for k, v in per.items()},
          f"| faithfulness {sum(faith)/len(faith):.3f} | mean answer tokens {sum(toks)/len(toks):.0f}")

if __name__ == "__main__":
    for nm in ("golden_set", "heldout_set"):
        for pf in (False, True):
            main(nm, pf)
