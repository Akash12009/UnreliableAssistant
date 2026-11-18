"""Train the triage classifier on data/triage_train.jsonl; evaluate on the disjoint-template data/triage_test.jsonl.
C=20 was fixed a priori (not tuned on the test set)."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from sklearn.linear_model import LogisticRegression
from ua.rag.encoders import load_models
from ua.textnorm import norm_query
from ua.triage import MODEL_PATH, ROUTES, Triage

def load(f): return [json.loads(l) for l in (ROOT / "data" / f).read_text().splitlines() if l.strip()]
enc, _ = load_models()
tr, te = load("triage_train.jsonl"), load("triage_test.jsonl")
lab = [r.value for r in ROUTES]
E = lambda rows: enc.encode([norm_query(x["text"]) for x in rows], is_query=False)
Xtr, Xte = E(tr), E(te)
clf = LogisticRegression(C=20, max_iter=3000).fit(Xtr, [lab.index(x["label"]) for x in tr])
W = np.zeros((len(ROUTES), Xtr.shape[1])); b = np.zeros(len(ROUTES))
for ci, cls in enumerate(clf.classes_): W[cls] = clf.coef_[ci]; b[cls] = clf.intercept_[ci]
np.savez(MODEL_PATH, W=W, b=b, encoder=enc.name)
t = Triage()
pred = [t.classify(x["text"], Xte[i]).route.value for i, x in enumerate(te)]
acc = np.mean([p == x["label"] for p, x in zip(pred, te)])
print(f"triage held-out accuracy {acc:.3f} on {len(te)} (train {len(tr)}, disjoint templates, encoder={enc.name})")
for p, x in zip(pred, te):
    if p != x["label"]: print("  miss:", x["text"], x["label"], "->", p)
