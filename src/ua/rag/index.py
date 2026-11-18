"""Vault -> chunks -> BM25 + FAISS-HNSW index, with sha256 manifest (FMEA #11: refuse corrupt index)."""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
from pathlib import Path

import faiss
import numpy as np

from ..config import SETTINGS

FM = re.compile(r"\A---\n(.*?)\n---\n(.*)\Z", re.S)
WIKI = re.compile(r"\[\[([A-Za-z0-9\-_ ]+)\]\]")


def parse_note(path: Path) -> dict:
    raw = path.read_text()
    m = FM.match(raw)
    meta, body = {}, raw
    if m:
        for line in m.group(1).splitlines():
            k, _, v = line.partition(":")
            meta[k.strip()] = v.strip()
        body = m.group(2)
    title = next((l[2:].strip() for l in body.splitlines() if l.startswith("# ")), path.stem)
    return dict(id=meta.get("id", path.stem), domain=meta.get("domain", ""), risk_tier=int(meta.get("risk_tier", 1)),
                version=int(meta.get("version", 1)), title=title, body=body.strip(), links=WIKI.findall(body))


def chunk_note(n: dict, size: int = 512, overlap: int = 64) -> list[dict]:
    words = n["body"].split()
    out, i, k = [], 0, 0
    while True:
        # single-chunk notes keep their original text (newlines => markdown heading stays on its own line)
        piece = n["body"].strip() if len(words) <= size else " ".join(words[i:i + size])
        name = re.sub(r"^(?:SKU|LOT|POL|SOP)-\d+\s*", "", n["title"]).strip()
        out.append(dict(chunk_id=f"{n['id']}#{k}", source_id=n["id"], text=piece, risk_tier=n["risk_tier"],
                        title=n["title"], name=name,
                        links=n["links"], version=n["version"], domain=n["domain"],
                        chunk_hash=hashlib.sha256(re.sub(r"\s+", " ", piece.lower()).encode()).hexdigest()))
        k += 1
        if i + size >= len(words):
            return out
        i += size - overlap


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def build(vault: str | Path, out: str | Path, encoder, params=None) -> dict:
    p = params or SETTINGS.rag
    out = Path(out)
    lkg = out.with_name(out.name + ".lkg")
    if out.exists():                       # keep last-known-good snapshot
        if lkg.exists():
            shutil.rmtree(lkg)
        shutil.copytree(out, lkg)
    out.mkdir(parents=True, exist_ok=True)
    chunks = []
    for f in sorted(Path(vault).glob("*.md")):
        chunks += chunk_note(parse_note(f), p.chunk_tokens, p.chunk_overlap)
    texts = [c["text"] for c in chunks]
    emb = encoder.encode(texts, is_query=False)
    idx = faiss.IndexHNSWFlat(emb.shape[1], p.hnsw_m, faiss.METRIC_INNER_PRODUCT)
    idx.hnsw.efConstruction = p.ef_construction
    idx.add(emb)
    (out / "chunks.jsonl").write_text("\n".join(json.dumps(c) for c in chunks) + "\n")
    faiss.write_index(idx, str(out / "dense.faiss"))
    np.save(out / "embeddings.npy", emb)
    man = dict(built_at=time.time(), n_chunks=len(chunks), dim=int(emb.shape[1]), encoder=getattr(encoder, "name", "?"),
               encoder_backend=getattr(encoder, "backend", "?"),
               files={f: _sha(out / f) for f in ("chunks.jsonl", "dense.faiss", "embeddings.npy")})
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    return man


def verify(dirpath: Path) -> bool:
    try:
        man = json.loads((dirpath / "manifest.json").read_text())
        return all(_sha(dirpath / f) == h for f, h in man["files"].items())
    except Exception:
        return False


def load(dirpath: str | Path):
    """Load index; if checksum fails fall back to the last-known-good snapshot (never serve a corrupt index)."""
    d = Path(dirpath)
    chosen = d if verify(d) else (d.with_name(d.name + ".lkg") if verify(d.with_name(d.name + ".lkg")) else None)
    if chosen is None:
        raise RuntimeError(f"index at {d} failed checksum and no valid last-known-good snapshot exists")
    chunks = [json.loads(l) for l in (chosen / "chunks.jsonl").read_text().splitlines() if l.strip()]
    idx = faiss.read_index(str(chosen / "dense.faiss"))
    man = json.loads((chosen / "manifest.json").read_text())
    return chunks, idx, man, chosen != d
