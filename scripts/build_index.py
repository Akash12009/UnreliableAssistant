"""Build the retrieval index: python scripts/build_index.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from ua.config import SETTINGS
from ua.rag import index
from ua.rag.encoders import load_models

enc, rr = load_models()
man = index.build(SETTINGS.vault_dir, SETTINGS.index_dir, enc)
print(f"index built: {man['n_chunks']} chunks, dim={man['dim']}, encoder={man['encoder']} ({man['encoder_backend']}), reranker={rr.name} ({rr.backend})")
