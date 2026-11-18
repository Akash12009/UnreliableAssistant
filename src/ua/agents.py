"""Deterministic reasoning components that sit behind swappable interfaces.

NO LLM IS CONNECTED in this build (none is available offline). `ExtractiveLLM` and `RuleBrain` are deterministic
stand-ins that obey the same contracts a real model would (`generate(...)`, `plan(...)`):
  * generation is strictly extractive + cited, so faithfulness is high *by construction* - it says nothing about the
    quality of a real LLM. The verification layer (claim_support, RAGAS auditor) is what protects a real model later.
  * to plug in vLLM / TensorRT-LLM implement `LLM.generate` and `Brain.plan`; harness, loop limits, guards and
    plugins are unchanged.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .textnorm import approx_tokens
from .verify import content_stems, stem

_WIKI = re.compile(r"\[\[([^\]]+)\]\]")
_ID = re.compile(r"\b(?:SKU|LOT|POL|SOP)-\d+\b", re.I)
SYN = {"cost": "price", "costs": "price", "sells": "price", "sell": "price", "much": "price", "expensive": "price",
       "expire": "expir", "expiry": "expir", "company": "supplier", "buy": "supplier", "supplies": "supplier",
       "held": "warehouse", "kept": "warehouse", "stocks": "warehouse", "cleared": "status", "role": "approv",
       "approves": "approv", "sign": "approv", "signs": "approv", "send": "return", "alert": "escalat",
       "notified": "escalat", "notify": "escalat", "who": ""}
SUMMARY_RX = re.compile(r"\b(summar\w+|overview|everything|know about|details? (on|about)|describe|tell me about)\b", re.I)


def clean(text: str) -> str:
    t = _WIKI.sub(r"\1", text)
    return re.sub(r"\s+", " ", t).strip()


def clauses(text: str) -> list[str]:
    t = clean("\n".join(l for l in text.splitlines() if not l.lstrip().startswith("#")))   # drop markdown title line
    out = []
    for c in re.split(r"(?<=[.;])\s+", t):
        c = c.strip().rstrip(".;").strip()
        if len(c) > 6:
            out.append(c)
    return out


def q_stems(question: str) -> set[str]:
    q = _ID.sub(" ", question)
    out = set()
    for w in re.findall(r"[a-z0-9]+", q.lower()):
        w2 = SYN.get(w, w)
        if w2:
            out |= content_stems(w2)
    return out


def score_clause(qs: set[str], question: str, clause: str, src: str, dense: float = 0.0, identity: bool = False) -> float:
    """dense = bi-encoder cosine min-max normalised within the candidate set (0..1): handles 'stay good' ~ 'keeps for'.
    identity = the note's first clause (id/name/category): every question mentions it, so it must not win by default."""
    cs = content_stems(clause)
    ov = len(qs & cs) / max(1, len(qs)) if qs else 0.0
    sc = 0.6 * ov + 1.5 * dense
    if re.search(r"[\d$°]", clause):
        sc += 0.15          # factual clauses (numbers / currency / units) beat boilerplate
    if identity:
        sc -= 0.8
    if _pointer(clause, question):
        sc -= 0.5
    return sc


def _norm(xs: list[float], skip: set[int] | None = None) -> list[float]:
    """Min-max normalise, ignoring `skip` indices when finding the range (identity clauses would stretch it)."""
    if not xs:
        return xs
    vals = [x for i, x in enumerate(xs) if not skip or i not in skip] or xs
    lo, hi = min(vals), max(vals)
    return [0.0 if hi - lo < 1e-9 else max(0.0, min(1.0, (x - lo) / (hi - lo))) for x in xs]


def _pointer(clause: str, question: str) -> bool:
    """A clause that only cross-references other notes ('Return rules are in POL-07 ...') is navigation, not an answer."""
    ids = {m.upper() for m in _ID.findall(clause)}
    return bool(ids) and not (ids & {m.upper() for m in _ID.findall(question)}) and len(clause.split()) < 16


@dataclass
class Evidence:
    src: str
    text: str
    risk_tier: int = 1
    rank: int = 0


def _dense_scores(emb, qvec, cl: list[str]) -> list[float]:
    if emb is None or qvec is None or not cl:
        return [0.0] * len(cl)
    return [float(x) for x in emb.texts(cl) @ qvec]


def compress_observation(question: str, evidence: list[Evidence], budget: int = 180, qvec=None, emb=None) -> tuple[str, list[tuple[str, str]], int]:
    """Extractive compression: strip JSON/boilerplate, keep the most question-relevant clauses until <= budget tokens.
    Returns (text, [(src, clause)], raw_tokens_before)."""
    qs = q_stems(question)
    raw = sum(approx_tokens(e.text) for e in evidence)
    scored = []
    allc = [(e, i, c) for e in evidence for i, c in enumerate(clauses(e.text))]
    dn = _norm(_dense_scores(emb, qvec, [c for _, _, c in allc]), {k for k, (_, i, _) in enumerate(allc) if i == 0})
    for (e, i, c), d in zip(allc, dn):
        scored.append((score_clause(qs, question, c, e.src, d, identity=(i == 0)) - 0.01 * e.rank - 0.001 * i, e, c))
    scored.sort(key=lambda x: -x[0])
    kept, used = [], 0
    for sc, e, c in scored:
        t = approx_tokens(c) + 4
        if used + t > budget:
            continue
        kept.append((e, c, sc))
        used += t
    order = {e.src: e.rank for e in evidence}
    kept.sort(key=lambda x: (order.get(x[0].src, 99), clauses(x[0].text).index(x[1]) if x[1] in clauses(x[0].text) else 0))
    pairs = [(e.src, c) for e, c, _ in kept]
    return "\n".join(f"[{s}] {c}" for s, c in pairs), pairs, raw


class ExtractiveLLM:
    name = "extractive-deterministic"

    def generate(self, question: str, evidence: list[Evidence], pairs: list[tuple[str, str]] | None = None,
                 strict: bool = False, max_clauses: int = 3, qvec=None, emb=None) -> str:
        if not evidence:
            return ""
        if SUMMARY_RX.search(question) and not strict:
            top = evidence[0]
            return f"{'. '.join(clauses(top.text))}. [{top.src}]"
        qs = q_stems(question)
        cand = pairs if pairs is not None else [(e.src, c) for e in evidence for c in clauses(e.text)]
        rank = {e.src: e.rank for e in evidence}
        first = {e.src: (clauses(e.text) or [""])[0] for e in evidence}
        dense = _norm(_dense_scores(emb, qvec, [c for _, c in cand]), {k for k, (s_, c_) in enumerate(cand) if c_ == first.get(s_)})
        scored = sorted(((score_clause(qs, question, c, s, d, identity=(c == first.get(s))) - 0.01 * rank.get(s, 0), s, c)
                         for (s, c), d in zip(cand, dense)), key=lambda x: -x[0])
        take = 1 if strict else max_clauses
        best = scored[0][0]
        # keep clauses close to the best one: a second clause must be nearly as relevant, not just present
        picked = [(s, c) for sc, s, c in scored[:take] if sc >= best - (0.25 if strict else 0.4)] or [(scored[0][1], scored[0][2])]
        seen, out = set(), []
        for s, c in picked:
            if c not in seen:
                seen.add(c)
                out.append(f"{c}. [{s}]")
        return " ".join(out)


# ----------------------------------------------------------------------------------------------- planner
@dataclass
class ToolSpec:
    tool: str
    args: dict
    subagent: str


@dataclass
class Plan:
    thought: str
    calls: list[ToolSpec] = field(default_factory=list)
    final: bool = False


_ENT = re.compile(r"\b(sku|lot)[\s\-_#]*\d{1,5}\b", re.I)
TO_STATE = [("quarantin", "quarantined"), ("recall", "recalled"), ("archiv", "archived"), ("retire", "archived"),
            ("deactivat", "archived"), ("on hold", "quarantined"), ("release", "released")]


def parse_transaction(q: str) -> dict | None:
    """Deterministic NL -> (tool,args). Entity ids are passed RAW so the harness auto-repair is what normalises them."""
    m = _ENT.search(q)
    if not m:
        return None
    raw_id = m.group(0)
    ql = q.lower()
    if "safety release" in ql:
        return dict(tool="submit_form", args=dict(form_type="safety_release", entity_id=raw_id, payload={}), entity=raw_id)
    if "price" in ql:
        num = re.search(r"(\d+(?:\.\d+)?)\s*$", re.sub(r"[?.!]+$", "", q.strip()))
        payload = {"price": float(num.group(1))} if num and not _ENT.fullmatch(num.group(1)) else {}
        return dict(tool="submit_form", args=dict(form_type="price_change", entity_id=raw_id, payload=payload), entity=raw_id)
    if "amend" in ql or "order" in ql:
        return dict(tool="submit_form", args=dict(form_type="order_amend", entity_id=raw_id, payload={}), entity=raw_id)
    for kw, state in TO_STATE:
        if kw in ql:
            return dict(tool="transition_workflow", args=dict(entity_id=raw_id, to_state=state), entity=raw_id)
    return None


class RuleBrain:
    """Deterministic planner. Replace with an LLM-backed Brain exposing the same `plan`."""

    def plan(self, route: str, question: str, step: int, state: dict) -> Plan:
        if route == "HYBRID_RAG":
            if step == 1:
                return Plan("retrieve evidence (3-stage RAG) and resolve the entity subgraph in parallel", [
                    ToolSpec("hybrid_retrieve", dict(query=question, pool=state["pool"], threshold=state["threshold"]), "retrieval"),
                    ToolSpec("graph_subgraph", dict(query=question), "retrieval")])
            return Plan("evidence sufficient; compose cited answer", final=True)
        if route == "TRANSACTION_EXEC":
            tx = parse_transaction(question)
            if not tx:
                return Plan("could not parse a supported transaction", final=True)
            state["tx"] = tx
            if step == 1:
                key = "sku" if tx["entity"].lower().startswith("sku") else "lot"
                return Plan("verify the target entity exists (Tier 1) before mutating",
                            [ToolSpec("lookup_item", {key: tx["entity"]}, "execution")])
            if step == 2 and state.get("entity_ok"):
                return Plan("target verified; submit Tier-3 mutation (HITL-gated)", [ToolSpec(tx["tool"], tx["args"], "execution")])
            return Plan("done", final=True)
        return Plan("nothing to do", final=True)


class ChaosBrain:
    """Adversarial brains for KPI-16 / Streamlit Tab 5: try to loop forever."""

    def __init__(self, kind: str = "same_sku", seed: int = 0):
        self.kind, self.seed = kind, seed

    def plan(self, route: str, question: str, step: int, state: dict) -> Plan:
        if self.kind == "same_sku":          # identical failing call forever -> circuit breaker
            return Plan("retry the lookup", [ToolSpec("lookup_item", {"sku": f"SKU-9{self.seed % 90 + 10}"}, "execution")])
        if self.kind == "alternating":       # A,B,A,B with distinct args -> max_iterations / no-progress bound
            n = (step + self.seed) % 2
            return Plan("try the other id", [ToolSpec("lookup_item", {"sku": f"SKU-9{n}{self.seed % 9}"}, "execution")])
        if self.kind == "unique":            # always new args -> max_iterations bound
            return Plan("try yet another id", [ToolSpec("lookup_item", {"sku": f"SKU-{800 + step * 7 + self.seed % 97}"}, "execution")])
        if self.kind == "refetch":           # tier-1 retrieval with no state change -> no-progress bound
            return Plan("fetch again", [ToolSpec("hybrid_retrieve", dict(query=question, pool=4), "retrieval")])
        return Plan("done", final=True)
