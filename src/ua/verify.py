"""Lexical claim-to-evidence verification (NLI proxy, ADR-006).

A claim (answer sentence) is SUPPORTED iff (a) every number / entity-id in it appears in the evidence and
(b) >= `min_overlap` of its content-word stems appear in the evidence. This is deliberately conservative: it
catches fabricated figures/IDs, which is the dominant hallucination mode for this domain. It is NOT a
semantic entailment model; a learned NLI model can replace `claim_support` behind the same signature.
"""
from __future__ import annotations

import re

from nltk.stem import PorterStemmer

_ps = PorterStemmer()
_STOP = set("""a an the and or of to in on for with is are was were be been by at as it its this that these those
from into than then so such not no do does did has have had can could should would may might will shall
which who whom what when where why how also any all each per via about above below between over under""".split())
_CITE = re.compile(r"\[(?:[A-Za-z0-9_\-:#]+)\]")
_ID = re.compile(r"\b(?:SKU|LOT|POL|SOP)-\d+\b", re.I)
_NUM = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)?(?![\w])")
_stem_cache: dict[str, str] = {}


def stem(w: str) -> str:
    s = _stem_cache.get(w)
    if s is None:
        s = _stem_cache[w] = _ps.stem(w)
    return s


def content_stems(text: str) -> set[str]:
    return {stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP and len(w) > 1}


def split_claims(answer: str) -> list[str]:
    body = _CITE.sub("", answer)
    parts = re.split(r"(?<=[.!?])\s+|\n+", body)
    return [p.strip(" -•\t") for p in parts if len(p.strip()) > 3]


def claim_support(answer: str, evidence: list[str], min_overlap: float = 0.8) -> tuple[float, list[str]]:
    claims = split_claims(answer)
    if not claims:
        return 1.0, []
    ev_text = " \n ".join(evidence)
    ev_stems = content_stems(ev_text)
    ev_ids = {m.upper() for m in _ID.findall(ev_text)}
    ev_nums = set(_NUM.findall(ev_text))
    unsupported = []
    for c in claims:
        ids = {m.upper() for m in _ID.findall(c)}
        nums = set(_NUM.findall(c))
        cs = content_stems(c)
        overlap = len(cs & ev_stems) / len(cs) if cs else 1.0
        if not ids <= ev_ids or not nums <= ev_nums or overlap < min_overlap:
            unsupported.append(c)
    return 1 - len(unsupported) / len(claims), unsupported
