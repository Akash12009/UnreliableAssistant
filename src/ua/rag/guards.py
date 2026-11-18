"""Deterministic grounding guards used by the retrieval engine.

* entity anchoring : explicit IDs / known product names in the query are resolved exactly (no model needed)
* unsupported term : a distinctive query term that appears NOWHERE in the knowledge base (and is not a generic
                     question word) means the KB cannot answer it -> deterministic abstention.

WARNING - STRICT MODE ONLY. Measured: this guard reached 100% abstention accuracy on the golden set but falsely
refused ~70% of held-out paraphrases (vocabulary mismatch: "company" vs "supplier"). It therefore runs ONLY in
`strict_extractive` mode (the runbook response to a hallucination surge), trading recall for safety. See
eval/README.md ("Rejected approaches").
"""
from __future__ import annotations

import re

from ..verify import _STOP, content_stems, stem

NEUTRAL_WORDS = """acceptable why explain explained current currently now right shipped ship shipping customer customers meet
total would give tell show everything summary summarize summarise overview details detail information info need new
employee many much long often please help happen happens mean means describe description based relevant also still just
again today status cost costs price priced pay paid cheap expensive amount quantity units unit item items product products
number numbers use used using work works way ways step steps list kind type types thing things place places person people
someone anyone whom whose within without before after during between about around across over under above below more most
less least same different similar other another each every both either neither any some many few several first second third
last next previous good best better worse bad able possible impossible required require requires must may might shall
can cannot get got give gave take took make made keep kept hold held know known see seen look find found check called
name names named stored stores store storing kept warehouse stock stocks stocked hold holds when where what which who
whose how does did done doing been being having has have""".split()
NEUTRAL = {stem(w) for w in NEUTRAL_WORDS}
_ID_TOK = re.compile(r"\b(?:sku|lot|pol|sop)-?\d+\b|\b\d+(?:[.,]\d+)?\b", re.I)


# Unit markers present in the corpus imply the concept words users use to ask about them.
UNIT_CONCEPTS = {
    "°c": "temperature degree degrees celsius cold warm hot",
    "$": "price cost dollar dollars usd sell sells selling sold pay charge",
}


def build_vocab(chunks: list[dict]) -> set[str]:
    v: set[str] = set()
    blob = " ".join(c["text"].lower() for c in chunks)
    for c in chunks:
        v |= content_stems(c["text"])
        v |= content_stems(c.get("title", ""))
    for marker, words in UNIT_CONCEPTS.items():
        if marker in blob:
            v |= {stem(w) for w in words.split()}
    return v


def unsupported_terms(query: str, vocab: set[str]) -> list[str]:
    q = _ID_TOK.sub(" ", query.lower())
    bad = []
    for w in re.findall(r"[a-z]+", q):
        s = stem(w)
        if len(w) > 2 and w not in _STOP and w not in _STOP_EXTRA and s not in vocab and s not in NEUTRAL:
            bad.append(w)
    return bad


_STOP_EXTRA = {"the", "and", "for", "are", "was", "with", "from", "this", "that", "its", "has", "not", "you", "your", "can"}
