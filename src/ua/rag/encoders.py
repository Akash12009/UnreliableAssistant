"""ONNX bi-encoder / cross-encoder wrappers + dependency-free fallbacks.

Profiles (design ADR-002):
  dev  : bge-small-en-v1.5 (384d) + ms-marco-MiniLM-L-6-v2   -> CPU-feasible, used on this machine
  spec : bge-large-en-v1.5 (1024d) + bge-reranker-v2-m3 FP16 -> needs GPU to hit the latency budget
Fallback (`hash`/`lexical`) is used ONLY when model files are absent; `backend` is reported on every result so
reports can never silently claim neural-model numbers.
"""
from __future__ import annotations

import hashlib
import math
import os
import re
import threading
import time
from pathlib import Path
from queue import Empty, Queue

import numpy as np

from ..config import ROOT, SETTINGS

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

PROFILES = {
    "dev": dict(encoder=ROOT / "models" / "bge-small-en-v1.5", reranker=ROOT / "models" / "ms-marco-MiniLM-L-6-v2",
                query_prefix=BGE_QUERY_PREFIX, rerank_file="model.onnx"),
    "spec": dict(encoder=ROOT / "models" / "bge-large-en-v1.5", reranker=ROOT / "models" / "bge-reranker-v2-m3",
                 query_prefix=BGE_QUERY_PREFIX, rerank_file="model_fp16.onnx"),
}


def _session(path: Path, threads: int | None = None):
    import onnxruntime as ort
    so = ort.SessionOptions()
    # 2 threads/session: >2 oversubscribes a 4-vCPU box when encoder + reranker run concurrently (measured)
    so.intra_op_num_threads = threads or int(os.environ.get("UA_ORT_THREADS", 2))
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def _tokenizer(path: Path, max_len: int):
    from tokenizers import Tokenizer
    t = Tokenizer.from_file(str(path))
    t.enable_truncation(max_length=max_len)
    t.enable_padding()
    return t


def _feeds(sess, enc):
    ids = np.array([e.ids for e in enc], dtype=np.int64)
    mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
    feeds = {"input_ids": ids, "attention_mask": mask}
    names = {i.name for i in sess.get_inputs()}
    if "token_type_ids" in names:
        feeds["token_type_ids"] = np.array([e.type_ids for e in enc], dtype=np.int64)
    return {k: v for k, v in feeds.items() if k in names}


class OnnxEncoder:
    backend = "onnx-bi-encoder"

    def __init__(self, model_dir: Path, max_len: int = 512, query_prefix: str = BGE_QUERY_PREFIX, filename: str = "model.onnx"):
        self.sess = _session(model_dir / filename)
        self.file = filename
        self.tok = _tokenizer(model_dir / "tokenizer.json", max_len)
        self.prefix = query_prefix
        self.name = model_dir.name
        self.dim = self.sess.get_outputs()[0].shape[-1] if isinstance(self.sess.get_outputs()[0].shape[-1], int) else 384

    def encode(self, texts: list[str], is_query: bool = False, batch: int = 32) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch):
            part = [(self.prefix + t) if is_query else t for t in texts[i:i + batch]]
            enc = self.tok.encode_batch(part)
            h = self.sess.run(None, _feeds(self.sess, enc))[0]
            v = h[:, 0, :].astype(np.float32)               # CLS pooling (BGE)
            v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
            out.append(v)
        return np.vstack(out) if out else np.zeros((0, self.dim), np.float32)


class HashEncoder:
    """Fallback: hashed word + char-trigram embedding. Lexical-ish, NOT semantic. Used only if no model files."""
    backend = "hash-fallback"
    name = "hash-512"
    dim = 512

    def encode(self, texts, is_query=False, batch=0):
        M = np.zeros((len(texts), self.dim), np.float32)
        for r, t in enumerate(texts):
            t = t.lower()
            toks = re.findall(r"[a-z0-9\-]+", t) + [t[i:i + 3] for i in range(len(t) - 2)]
            for tok in toks:
                h = int(hashlib.md5(tok.encode()).hexdigest()[:8], 16)
                M[r, h % self.dim] += 1.0 if (h >> 16) & 1 else -1.0
        M /= np.linalg.norm(M, axis=1, keepdims=True) + 1e-12
        return M


class OnnxReranker:
    backend = "onnx-cross-encoder"

    def __init__(self, model_dir: Path, filename: str = "model.onnx", max_len: int = 512):
        # reranker is the heavy stage and runs alone after the encoder -> give it all cores
        self.sess = _session(model_dir / filename, int(os.environ.get("UA_RERANK_THREADS", os.cpu_count() or 2)))
        self.file = filename
        self.tok = _tokenizer(model_dir / "tokenizer.json", max_len)
        self.name = model_dir.name

    def score_pairs(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        enc = self.tok.encode_batch(pairs)
        logits = self.sess.run(None, _feeds(self.sess, enc))[0].reshape(len(pairs), -1)[:, 0].astype(np.float64)
        return 1.0 / (1.0 + np.exp(-logits))        # sigmoid


class LexicalReranker:
    """Fallback reranker: query-term coverage. NOT a neural cross-encoder."""
    backend = "lexical-fallback"
    name = "lexical"

    def score_pairs(self, pairs):
        from ..verify import content_stems
        out = []
        for q, p in pairs:
            qs, ps = content_stems(q), content_stems(p)
            out.append(len(qs & ps) / len(qs) if qs else 0.0)
        return np.array(out)


class MicroBatcher:
    """Dynamic micro-batching across concurrent requests: batch <= 32, max_wait = 8 ms (design Gate 15).

    Waiting only happens when other requests are in flight; a lone request is flushed immediately so the
    8 ms window never inflates single-request latency.
    """

    def __init__(self, fn, max_batch: int = 32, max_wait_ms: float = 8.0):
        self.fn, self.max_batch, self.max_wait = fn, max_batch, max_wait_ms / 1000
        self.q: Queue = Queue()
        self.inflight = 0
        self.lock = threading.Lock()
        self.batches = 0
        self.items = 0
        self.t = threading.Thread(target=self._run, daemon=True, name="rerank-batcher")
        self.t.start()

    def submit(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        with self.lock:
            self.inflight += 1
        slots = [dict(pair=p, ev=threading.Event(), out=None, err=None) for p in pairs]
        for s in slots:
            self.q.put(s)
        try:
            for s in slots:
                s["ev"].wait()
                if s["err"]:
                    raise s["err"]
            return np.array([s["out"] for s in slots])
        finally:
            with self.lock:
                self.inflight -= 1

    def _run(self):
        while True:
            first = self.q.get()
            batch = [first]
            deadline = time.perf_counter() + self.max_wait
            while len(batch) < self.max_batch:
                try:
                    batch.append(self.q.get_nowait())
                    continue
                except Empty:
                    pass
                with self.lock:
                    others = self.inflight > 1
                remain = deadline - time.perf_counter()
                if not others or remain <= 0:
                    break
                try:
                    batch.append(self.q.get(timeout=remain))
                except Empty:
                    break
            try:
                scores = self.fn([b["pair"] for b in batch])
                for b, s in zip(batch, scores):
                    b["out"] = float(s)
            except Exception as e:
                for b in batch:
                    b["err"] = e
            self.batches += 1
            self.items += len(batch)
            for b in batch:
                b["ev"].set()


def load_models(profile: str | None = None, quant: str | None = None):
    """quant: 'int8' (CPU path, ADR-002) | 'fp32' | 'fp16' (spec profile on GPU). Env UA_QUANT overrides default."""
    profile = profile or SETTINGS.model_profile
    prof = PROFILES[profile]
    quant = quant or os.environ.get("UA_QUANT", "int8" if profile == "dev" else "fp16")
    enc_file = "model_int8.onnx" if quant == "int8" else "model.onnx"
    rr_file = "model_int8.onnx" if quant == "int8" else prof["rerank_file"]
    if not (prof["encoder"] / enc_file).exists():
        enc_file = "model.onnx"
    if not (prof["reranker"] / rr_file).exists():
        rr_file = prof["rerank_file"]
    enc: object
    rr: object
    if (prof["encoder"] / enc_file).exists():
        enc = OnnxEncoder(prof["encoder"], query_prefix=prof["query_prefix"], filename=enc_file)
    else:
        enc = HashEncoder()
    if (prof["reranker"] / rr_file).exists():
        rr = OnnxReranker(prof["reranker"], rr_file)
    else:
        rr = LexicalReranker()
    return enc, rr
