"""Triage agent: distilled linear classifier over the (shared) bge bi-encoder query embedding, temperature 0 (argmax).

The embedding is computed ONCE per request and reused by the semantic cache, triage and drift audit, so triage itself
is a 384x4 matrix product (~0.02 ms). Output: (intent, domain, risk_tier, complexity) + route.
Trained offline (scripts/train_triage.py, scikit-learn); inference is numpy-only. Low confidence routes to
HYBRID_RAG (read-only, safest default; FMEA #9) - never to TRANSACTION_EXEC.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import ROOT, SETTINGS
from .contracts import Route

ROUTES = [Route.DIRECT_ANSWER, Route.HYBRID_RAG, Route.TRANSACTION_EXEC, Route.REJECT_OOS]
MODEL_PATH = ROOT / "models" / "triage.npz"


@dataclass
class TriageResult:
    route: Route
    intent: str
    domain: str
    risk_tier: int
    complexity: str
    confidence: float
    fallback: bool = False


class Triage:
    def __init__(self, path: Path = MODEL_PATH):
        z = np.load(path)
        self.W, self.b = z["W"], z["b"]
        self.encoder_name = str(z["encoder"]) if "encoder" in z else ""

    def classify(self, text: str, qvec: np.ndarray) -> TriageResult:
        logits = self.W @ qvec.astype(np.float64) + self.b
        p = np.exp(logits - logits.max())
        p /= p.sum()
        k = int(np.argmax(p))                        # temperature 0.0 == argmax
        conf = float(p[k])
        route, fb = ROUTES[k], False
        if conf < SETTINGS.triage_min_confidence:
            route, fb = Route.HYBRID_RAG, True
        ids = [m.group(0).lower() for m in re.finditer(r"\b(sku|lot|pol|sop)[\s\-_#]*\d+\b", text, re.I)]
        kinds = {re.match(r"[a-z]+", i).group(0) for i in ids}
        domain = ("inventory" if "lot" in kinds else "catalog" if "sku" in kinds else "policy" if "pol" in kinds
                  else "sop" if "sop" in kinds else "general")
        risk = 3 if route == Route.TRANSACTION_EXEC else 2 if kinds & {"lot", "pol", "sop"} else 1
        cx = "multi" if (len(set(ids)) > 1 or re.search(r"\b(and|then|also)\b.*\b(who|how|when|what)\b", text, re.I)) else "simple"
        return TriageResult(route, route.value.lower(), domain, risk, cx, conf, fb)
