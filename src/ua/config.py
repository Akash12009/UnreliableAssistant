"""Central configuration. Values mirror design.md; runbook knobs are mutable at runtime."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(os.environ.get("UA_ROOT", Path(__file__).resolve().parents[2]))


@dataclass
class RagParams:
    bm25_k1: float = 1.25
    bm25_b: float = 0.75
    top_k_sparse: int = 50          # runbook: drop to 25 on latency breach
    top_k_dense: int = 50
    rrf_k: int = 60
    pool: int = 40
    rerank_k: int = 4
    threshold: float = 0.42         # runbook: raise to 0.55 on hallucination surge
    rerank_batch: int = 32
    max_len: int = 512
    chunk_tokens: int = 512
    chunk_overlap: int = 64
    hnsw_m: int = 32
    ef_construction: int = 200
    ef_search: int = 64
    w_bm25: float = 1.0
    w_dense: float = 1.0


@dataclass
class LoopParams:
    max_iterations: int = 5
    per_step_timeout_ms: int = 120
    total_loop_budget_ms: int = 450
    deadline_ms: int = 590          # absolute request deadline (p99 gate)
    egress_reserve_ms: int = 30
    observation_max_tokens: int = 180
    hot_turns: int = 3


@dataclass
class CacheParams:
    cosine_normal: float = 0.95
    cosine_aggressive: float = 0.92   # runbook: latency breach
    ttl_s: int = 300
    ttl_jitter_s: int = 45
    max_entries_per_namespace: int = 200


@dataclass
class Settings:
    rag: RagParams = field(default_factory=RagParams)
    loop: LoopParams = field(default_factory=LoopParams)
    cache: CacheParams = field(default_factory=CacheParams)
    triage_timeout_ms: int = 25
    triage_min_confidence: float = 0.6
    mcp_timeout_ms: int = 150
    rag_pool: int = 64
    db_pool: int = 128
    llm_pool: int = 256
    strict_extractive: bool = False   # runbook: hallucination surge
    degraded: bool = False
    aggressive_cache: bool = False
    # model profile: "dev" (CPU small models) or "spec" (bge-large + bge-reranker-v2-m3 FP16)
    model_profile: str = os.environ.get("UA_MODEL_PROFILE", "dev")
    sqlite_path: str = os.environ.get("UA_SQLITE", str(ROOT / "data" / "state.db"))
    redis_url: str = os.environ.get("UA_REDIS_URL", "")
    index_dir: str = os.environ.get("UA_INDEX_DIR", str(ROOT / "index"))
    vault_dir: str = str(ROOT / "knowledge_vault")
    graph_path: str = str(ROOT / "graph.json")
    otlp_endpoint: str = "http://localhost:4317"
    # Runtime timeouts/deadlines are multiplied by this in the CPU dev profile so the system can FUNCTION on hardware
    # that cannot meet the spec latencies. KPI thresholds are NEVER scaled; the scorecard still judges against spec.
    budget_scale: float = float(os.environ.get("UA_BUDGET_SCALE", 6.0 if os.environ.get("UA_MODEL_PROFILE", "dev") == "dev" else 1.0))

    def __post_init__(self):
        # CPU dev profile: rerank pool 12 instead of 40. Measured on this 4-vCPU AVX2 box a 6-layer cross-encoder
        # needs ~16 ms/pair, so pool 40 cannot meet the 45 ms stage budget (nor can any CPU). Spec value is
        # restored automatically for the `spec` profile. Reported honestly by the KPI scorecard.
        if self.model_profile == "dev" and "UA_POOL" not in os.environ:
            self.rag.pool = 12
        if "UA_POOL" in os.environ:
            self.rag.pool = int(os.environ["UA_POOL"])


SETTINGS = Settings()


def scaled(ms: float) -> float:
    return ms * SETTINGS.budget_scale


def load_mcp_config(path: str | Path | None = None) -> dict:
    p = Path(path or os.environ.get("UA_MCP_CONFIG", ROOT / "mcp_config.json"))
    return json.loads(p.read_text())
