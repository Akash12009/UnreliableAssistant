"""BM25Okapi with Porter stemming and an inverted index (pre-tokenised). k1/b adjustable at query time."""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

import numpy as np
from nltk.stem import PorterStemmer

_ps = PorterStemmer()
_cache: dict[str, str] = {}
_TOK = re.compile(r"[a-z]+-\d+|[a-z0-9]+")
_STOP = frozenset("a an the of to in on for and or is are was were be by at as it this that with from what which who how when".split())


def _stem(w: str) -> str:
    s = _cache.get(w)
    if s is None:
        s = _cache[w] = w if "-" in w or w.isdigit() else _ps.stem(w)
    return s


def tokenize(text: str) -> list[str]:
    """lowercase -> id-preserving split -> stopword drop -> Porter stem. IDs ('sku-007') stay whole."""
    return [_stem(w) for w in _TOK.findall(text.lower()) if w not in _STOP]


class BM25Index:
    EPS = 0.25   # rank_bm25 default floor for negative idf

    def __init__(self, docs: list[str]):
        self.n = len(docs)
        toks = [tokenize(d) for d in docs]
        self.dl = np.array([len(t) for t in toks], dtype=np.float64)
        self.avgdl = float(self.dl.mean()) if self.n else 0.0
        post: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, t in enumerate(toks):
            for w, c in Counter(t).items():
                post[w].append((i, c))
        self.post = {w: (np.array([d for d, _ in v], dtype=np.int64), np.array([c for _, c in v], dtype=np.float64))
                     for w, v in post.items()}
        idf = {w: math.log((self.n - len(v[0]) + 0.5) / (len(v[0]) + 0.5)) for w, v in self.post.items()}
        avg_idf = sum(idf.values()) / len(idf) if idf else 0.0
        self.idf = {w: (x if x >= 0 else self.EPS * avg_idf) for w, x in idf.items()}

    def scores(self, query: str, k1: float = 1.25, b: float = 0.75) -> np.ndarray:
        s = np.zeros(self.n)
        for w in tokenize(query):
            p = self.post.get(w)
            if p is None:
                continue
            docs, tf = p
            norm = k1 * (1 - b + b * self.dl[docs] / (self.avgdl or 1.0))
            s[docs] += self.idf[w] * tf * (k1 + 1) / (tf + norm)
        return s

    def top(self, query: str, k: int, k1: float = 1.25, b: float = 0.75) -> list[tuple[int, float]]:
        sc = self.scores(query, k1, b)
        if k < self.n:
            idx = np.argpartition(-sc, k)[:k]
        else:
            idx = np.arange(self.n)
        idx = idx[np.argsort(-sc[idx], kind="stable")]
        return [(int(i), float(sc[i])) for i in idx if sc[i] > 0]
