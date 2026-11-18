"""3-stage hybrid retrieval engine (BM25 || dense -> RRF -> cross-encoder) with deterministic abstention."""
from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import numpy as np

from ..config import RagParams, SETTINGS
from ..contracts import Chunk, HybridRetrieveIn, HybridRetrieveOut
from . import index as idxmod
from ..textnorm import approx_tokens
from .bm25 import BM25Index
from .guards import build_vocab, unsupported_terms
from .encoders import MicroBatcher, load_models

ID_RX = re.compile(r"\b(?:SKU-\d{3}|LOT-\d{5}|POL-\d{2}|SOP-\d{2})\b", re.I)


def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000


class RagEngine:
    def __init__(self, index_dir: str | None = None, encoder=None, reranker=None, pool: ThreadPoolExecutor | None = None):
        d = index_dir or SETTINGS.index_dir
        self.encoder, self.reranker = (encoder, reranker) if encoder else load_models()
        if reranker is not None:
            self.reranker = reranker
        self.chunks, self.faiss, self.manifest, self.from_lkg = idxmod.load(d)
        self.faiss.hnsw.efSearch = SETTINGS.rag.ef_search
        self.bm25 = BM25Index([c["text"] for c in self.chunks])
        self.by_id = {c["chunk_id"]: i for i, c in enumerate(self.chunks)}
        self.known_sources = {c["source_id"].upper() for c in self.chunks}
        self.vocab = build_vocab(self.chunks)
        self.src_idx: dict[str, list[int]] = {}
        self.name_idx: dict[str, list[int]] = {}
        for i, c in enumerate(self.chunks):
            self.src_idx.setdefault(c["source_id"].upper(), []).append(i)
            nm = (c.get("name") or "").lower()
            if len(nm) >= 8 and c["source_id"].startswith(("SKU", "SOP", "POL")):
                self.name_idx.setdefault(nm, []).append(i)
        self.term_guard = None      # None -> follow SETTINGS.strict_extractive (runbook: hallucination surge)
        self.entity_anchor = True
        self.pool = pool or ThreadPoolExecutor(SETTINGS.rag_pool, thread_name_prefix="rag")
        self.batcher = MicroBatcher(self.reranker.score_pairs, SETTINGS.rag.rerank_batch, 8.0)
        self.backend = f"{self.encoder.backend}+{self.reranker.backend}"
        self.reranker_failing = False         # chaos hook (Streamlit Tab 5 / FMEA #2)

    # -- stage 1 / 2 ------------------------------------------------------
    def _sparse(self, q: str, p: RagParams):
        t0 = time.perf_counter()
        r = self.bm25.top(q, p.top_k_sparse, p.bm25_k1, p.bm25_b)
        return r, _ms(t0)

    def _dense(self, q: str, p: RagParams):
        t0 = time.perf_counter()
        v = self.encoder.encode([q], is_query=True)
        self.faiss.hnsw.efSearch = p.ef_search
        D, I = self.faiss.search(v, p.top_k_dense)
        r = [(int(i), float(s)) for i, s in zip(I[0], D[0]) if i >= 0]
        return r, _ms(t0)

    @staticmethod
    def rrf(sparse, dense, k: int, pool: int, w_bm25=1.0, w_dense=1.0):
        s: dict[int, float] = {}
        rank_b = {i: r for r, (i, _) in enumerate(sparse, 1)}
        rank_d = {i: r for r, (i, _) in enumerate(dense, 1)}
        for i, r in rank_b.items():
            s[i] = s.get(i, 0.0) + w_bm25 / (k + r)
        for i, r in rank_d.items():
            s[i] = s.get(i, 0.0) + w_dense / (k + r)
        ordered = sorted(s.items(), key=lambda x: -x[1])
        return ordered, rank_b, rank_d

    def _dedup(self, ordered):
        seen, out = set(), []
        for i, sc in ordered:
            h = self.chunks[i]["chunk_hash"]
            if h in seen:
                continue
            seen.add(h)
            out.append((i, sc))
        return out

    def entity_chunks(self, q: str) -> list[int]:
        """Chunks explicitly named in the query (by ID or by exact product/SOP name)."""
        out: list[int] = []
        for m in ID_RX.findall(q):
            out += self.src_idx.get(m.upper(), [])
        ql = q.lower()
        for nm, idxs in self.name_idx.items():
            if nm in ql:
                out += idxs
        return list(dict.fromkeys(out))

    # -- orchestration ----------------------------------------------------
    def retrieve(self, req: HybridRetrieveIn | dict, debug: bool = False) -> HybridRetrieveOut:
        if isinstance(req, dict):
            req = HybridRetrieveIn(**req)
        p = replace(SETTINGS.rag, top_k_sparse=req.top_k_sparse, top_k_dense=req.top_k_dense, rrf_k=req.rrf_k,
                    pool=req.pool, rerank_k=req.rerank_k, threshold=req.threshold)
        q = req.query
        t0 = time.perf_counter()
        fb = self.pool.submit(self._sparse, q, p)
        fd = self.pool.submit(self._dense, q, p)
        (sparse, ms_b), (dense, ms_d) = fb.result(), fd.result()

        t1 = time.perf_counter()
        ordered, rank_b, rank_d = self.rrf(sparse, dense, p.rrf_k, p.pool, SETTINGS.rag.w_bm25, SETTINGS.rag.w_dense)
        ordered = self._dedup(ordered)
        ents = self.entity_chunks(q) if self.entity_anchor else []
        if ents:                                                 # entity-anchored candidates always enter the rerank pool
            fused_map = dict(ordered)
            floor = ordered[-1][1] if ordered else 0.0
            ordered = [(i, fused_map.get(i, floor)) for i in ents] + [(i, s) for i, s in ordered if i not in set(ents)]
        ordered = ordered[:max(p.pool, len(ents))]
        top = max((s for _, s in ordered), default=1.0)
        fused = [(i, sc / top) for i, sc in ordered]          # scale-normalised to [0,1]
        ent_set = set(ents)
        ms_rrf = _ms(t1)

        bm = dict(sparse)
        cos = dict(dense)
        t2 = time.perf_counter()
        degraded = False
        thr = p.threshold
        if fused:
            pairs = [(q, self.chunks[i]["text"]) for i, _ in fused]
            try:
                if self.reranker_failing:
                    raise TimeoutError("reranker unavailable (injected)")
                scores = self.batcher.submit(pairs)
            except Exception:                                   # FMEA #2: fall back to fused ranks, stricter bar
                degraded = True
                bmax = max(bm.values()) if bm else 1.0
                scores = np.array([0.5 * min(1.0, bm.get(i, 0.0) / (bmax or 1.0)) + 0.5 * max(0.0, cos.get(i, 0.0))
                                   for i, _ in fused])
                thr = min(1.0, thr + 0.13)
        else:
            scores = np.array([])
        ms_rr = _ms(t2)

        # final order: entity-anchored first (exact), then by cross-encoder score
        order = sorted(range(len(scores)), key=lambda j: (fused[j][0] not in ent_set, -scores[j])) if len(scores) else []
        ranked = [(fused[j][0], float(scores[j]), fused[j][1]) for j in order]
        out_chunks: list[Chunk] = []
        rr_rank = {i: r for r, (i, _) in enumerate(ordered, 1)}
        for i, ce, fs in ranked[:p.rerank_k]:
            if ent_set and i not in ent_set:
                continue            # entity-anchored query: unrelated near-duplicate notes would only pollute the context
            if ce <= thr and i not in ent_set:
                continue
            c = self.chunks[i]
            out_chunks.append(Chunk(chunk_id=c["chunk_id"], chunk_hash=c["chunk_hash"], text=c["text"], source_id=c["source_id"],
                                    risk_tier=c["risk_tier"], links=c["links"], bm25=bm.get(i), cosine=cos.get(i),
                                    rrf_rank=rr_rank.get(i), rrf_score=round(fs, 4), ce_score=round(ce, 4)))
        top_score = float(max((r[1] for r in ranked), default=0.0))
        abstain, reason = (not out_chunks or (top_score <= thr and not ent_set)), "low_confidence"
        if not abstain and (SETTINGS.strict_extractive if self.term_guard is None else self.term_guard):
            bad = unsupported_terms(q, self.vocab)
            if bad:                                             # asks about something the KB never mentions
                abstain, reason = True, f"unsupported_term:{bad[0]}"
        if not abstain:
            unknown = {m.upper() for m in ID_RX.findall(q)} - self.known_sources
            if unknown:                                         # deterministic grounding: named entity does not exist
                abstain, reason = True, f"unknown_entity:{sorted(unknown)[0]}"
        if abstain:
            out_chunks = []
        cand = None
        if debug:
            cand = dict(bm25=[(self.chunks[i]["chunk_id"], round(s, 3)) for i, s in sparse[:20]],
                        dense=[(self.chunks[i]["chunk_id"], round(s, 3)) for i, s in dense[:20]],
                        rrf=[(self.chunks[i]["chunk_id"], round(s, 4)) for i, s in fused[:20]],
                        rerank=[(self.chunks[i]["chunk_id"], round(ce, 4)) for i, ce, _ in ranked[:20]],
                        abstain_reason=reason if abstain else None, degraded=degraded, threshold=thr)
        return HybridRetrieveOut(chunks=out_chunks, top_score=round(top_score, 4), abstain=abstain, backend=self.backend,
                                 abstain_reason=reason if abstain else None,
                                 pool_tokens=sum(approx_tokens(self.chunks[i]['text']) for i, _ in fused),
                                 stage_ms=dict(bm25=round(ms_b, 3), biencoder=round(ms_d, 3), rrf=round(ms_rrf, 3),
                                               rerank=round(ms_rr, 3), total=round(_ms(t0), 3)),
                                 candidates=cand)

    def get_chunk(self, chunk_id: str) -> dict | None:
        i = self.by_id.get(chunk_id)
        return self.chunks[i] if i is not None else None
