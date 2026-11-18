"""Shared embedder: one query embedding per request (cache + triage + clause scoring + drift), memoised clause vectors."""
from __future__ import annotations

import threading
from pathlib import Path

import numpy as np


class Embedder:
    def __init__(self, encoder):
        self.enc = encoder
        self.memo: dict[str, np.ndarray] = {}
        self.lock = threading.Lock()

    def query(self, text: str) -> np.ndarray:
        return self.texts([text])[0]

    def texts(self, items: list[str]) -> np.ndarray:
        with self.lock:
            need = [t for t in dict.fromkeys(items) if t not in self.memo]
        if need:
            vecs = self.enc.encode(need, is_query=False)
            with self.lock:
                for t, v in zip(need, vecs):
                    self.memo[t] = v
                if len(self.memo) > 50000:          # bound memory: drop oldest quarter
                    for k in list(self.memo)[:12500]:
                        self.memo.pop(k, None)
        return np.stack([self.memo[t] for t in items])

    def warm(self, vault_dir: str) -> int:
        from .agents import clauses
        from .rag.index import parse_note
        cl = []
        for f in sorted(Path(vault_dir).glob("*.md")):
            cl += clauses(parse_note(f)["body"])
        self.texts(cl)
        return len(cl)
