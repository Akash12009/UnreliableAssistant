"""IR + RAGAS-style metrics. RAGAS-style = deterministic proxies (no LLM judge available); see eval/README."""
from __future__ import annotations

import math


def recall_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    return len(set(ranked[:k]) & gold) / len(gold) if gold else 1.0


def mrr_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    for r, c in enumerate(ranked[:k], 1):
        if c in gold:
            return 1.0 / r
    return 0.0


def ndcg_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    dcg = sum(1 / math.log2(r + 1) for r, c in enumerate(ranked[:k], 1) if c in gold)
    idcg = sum(1 / math.log2(r + 1) for r in range(1, min(len(gold), k) + 1))
    return dcg / idcg if idcg else 0.0


def context_precision(ranked: list[str], gold: set[str]) -> float:
    """RAGAS context-precision: rank-weighted precision (average of precision@i at each relevant i)."""
    hits, acc = 0, 0.0
    for i, c in enumerate(ranked, 1):
        if c in gold:
            hits += 1
            acc += hits / i
    return acc / hits if hits else 0.0


def pctl(xs: list[float], q: float) -> float:
    import numpy as np
    return float(np.percentile(xs, q)) if xs else float("nan")
