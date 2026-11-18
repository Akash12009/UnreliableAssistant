"""Pydantic v2 MCP tool & data contracts (design.md 1.2). All models are strict + extra=forbid."""
from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .errors import ErrorObj

SKU_RE = r"^SKU-\d{3}$"
LOT_RE = r"^LOT-\d{5}$"
UUID4_RE = r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
ID_RE = r"^[A-Za-z0-9_\-]{1,64}$"
TRACEPARENT_RE = r"^00-[0-9a-f]{32}-[0-9a-f]{16}-0[01]$"

STRICT = ConfigDict(strict=True, extra="forbid")
# wire models: JSON has no enum/tuple types, so strict=False for coercion of JSON primitives only
WIRE = ConfigDict(extra="forbid")


class Tier(StrEnum):
    TIER_1_READ = "TIER_1_READ"
    TIER_2_COMPUTE = "TIER_2_COMPUTE"
    TIER_3_MUTATE = "TIER_3_MUTATE"


TIER_RANK = {Tier.TIER_1_READ: 1, Tier.TIER_2_COMPUTE: 2, Tier.TIER_3_MUTATE: 3}


class Route(StrEnum):
    DIRECT_ANSWER = "DIRECT_ANSWER"
    HYBRID_RAG = "HYBRID_RAG"
    TRANSACTION_EXEC = "TRANSACTION_EXEC"
    REJECT_OOS = "REJECT_OOS"


class Meta(BaseModel):
    model_config = WIRE
    traceparent: str = Field(pattern=TRACEPARENT_RE)
    tenant_id: str = Field(pattern=ID_RE)
    user_id: str = Field(pattern=ID_RE)
    session_id: str = Field(pattern=ID_RE)
    idempotency_key: str | None = Field(default=None, pattern=UUID4_RE)
    deadline_ms: int = Field(default=450, ge=1, le=600)
    role: str = "operator"          # calling user's role: viewer | operator | reviewer
    subagent: str = "retrieval"     # retrieval | execution | verification | reviewer
    auth: str | None = None         # HMAC over (tenant,user,session,role,subagent); verified by every MCP server


# ------------------------------------------------------------------ tool inputs
class HybridRetrieveIn(BaseModel):
    model_config = STRICT
    query: str = Field(min_length=1, max_length=2000)
    top_k_sparse: int = Field(default=50, ge=1, le=100)
    top_k_dense: int = Field(default=50, ge=1, le=100)
    rrf_k: int = Field(default=60, ge=1, le=200)
    pool: int = Field(default=40, ge=4, le=100)
    rerank_k: int = Field(default=4, ge=1, le=10)
    threshold: float = Field(default=0.42, ge=0.0, le=1.0)
    debug: bool = False


class Chunk(BaseModel):
    chunk_id: str
    chunk_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    text: str
    source_id: str
    risk_tier: int = Field(ge=1, le=3)
    links: list[str] = Field(default_factory=list)
    bm25: float | None = None
    cosine: float | None = None
    rrf_rank: int | None = None
    rrf_score: float | None = None
    ce_score: float = 0.0


class HybridRetrieveOut(BaseModel):
    chunks: list[Chunk] = Field(max_length=10)
    top_score: float
    abstain: bool
    stage_ms: dict[str, float]
    abstain_reason: str | None = None
    pool_tokens: int = 0                       # tokens of the full rerank pool (naive-RAG prompt baseline, KPI-26)
    candidates: dict[str, Any] | None = None   # debug payload for Streamlit Tab 2
    backend: str = ""


class LookupItemIn(BaseModel):
    model_config = STRICT
    sku: str | None = Field(default=None, pattern=SKU_RE)
    lot: str | None = Field(default=None, pattern=LOT_RE)

    @model_validator(mode="after")
    def _exactly_one(self):
        if (self.sku is None) == (self.lot is None):
            raise ValueError("exactly one of sku/lot required")
        return self


class SubmitFormIn(BaseModel):
    model_config = STRICT
    form_type: Literal["safety_release", "price_change", "order_amend"]
    entity_id: str = Field(pattern=r"^(SKU-\d{3}|LOT-\d{5})$")
    payload: dict[str, str | int | float | bool] = Field(default_factory=dict, max_length=32)


class TransitionIn(BaseModel):
    model_config = STRICT
    entity_id: str = Field(pattern=r"^(SKU-\d{3}|LOT-\d{5})$")
    to_state: Literal["released", "quarantined", "recalled", "archived"]


class SessionFactIn(BaseModel):
    model_config = STRICT
    key: str = Field(pattern=r"^[a-z_][a-z0-9_]{0,63}$")
    value: str = Field(max_length=1000)


class GetFactsIn(BaseModel):
    model_config = STRICT


class GraphSubgraphIn(BaseModel):
    model_config = STRICT
    query: str = Field(min_length=1, max_length=2000)
    anchor: str | None = Field(default=None, max_length=64)
    depth: int = Field(default=2, ge=1, le=2)
    max_tokens: int = Field(default=320, ge=50, le=800)


class AuditAppendIn(BaseModel):
    model_config = STRICT
    event: str = Field(pattern=r"^[a-z_\.]{1,48}$")
    detail: dict[str, str | int | float | bool | None] = Field(default_factory=dict, max_length=24)


class HitlDecideIn(BaseModel):
    model_config = STRICT
    approval_id: str = Field(pattern=UUID4_RE)
    decision: Literal["approve", "reject"]


class HitlListIn(BaseModel):
    model_config = STRICT
    status: Literal["PENDING_APPROVAL", "APPROVED", "REJECTED", "EXPIRED", "ALL"] = "PENDING_APPROVAL"


class AuditVerifyIn(BaseModel):
    model_config = STRICT
    tail: int = Field(default=0, ge=0, le=200)


class ToolResult(BaseModel):
    ok: bool
    data: dict[str, Any] | None = None
    status: Literal["OK", "PENDING_APPROVAL", "REPAIRED", "REJECTED"] = "OK"
    repaired_from: dict[str, Any] | None = None
    error: ErrorObj | None = None
    latency_ms: float = 0.0
    tripped: bool = False          # this call's failure opened the circuit breaker (-> HALTED_BREAKER)


# tool registry: name -> (server, input model, tier, allowed subagents)
TOOLS: dict[str, dict[str, Any]] = {
    "hybrid_retrieve": dict(server="hybrid-rag", model=HybridRetrieveIn, tier=Tier.TIER_1_READ,
                            subagents={"retrieval"}),
    "graph_subgraph": dict(server="graphify-vault", model=GraphSubgraphIn, tier=Tier.TIER_1_READ,
                           subagents={"retrieval"}),
    "lookup_item": dict(server="state-store", model=LookupItemIn, tier=Tier.TIER_1_READ,
                        subagents={"execution", "retrieval"}),
    "get_session_facts": dict(server="state-store", model=GetFactsIn, tier=Tier.TIER_1_READ,
                              subagents={"execution", "retrieval", "verification"}),
    "put_session_fact": dict(server="state-store", model=SessionFactIn, tier=Tier.TIER_2_COMPUTE,
                             subagents={"execution", "verification"}),
    "submit_form": dict(server="state-store", model=SubmitFormIn, tier=Tier.TIER_3_MUTATE,
                        subagents={"execution"}),
    "transition_workflow": dict(server="state-store", model=TransitionIn, tier=Tier.TIER_3_MUTATE,
                                subagents={"execution"}),
    "audit_append": dict(server="state-store", model=AuditAppendIn, tier=Tier.TIER_2_COMPUTE,
                         subagents={"verification"}),
    "hitl_decide": dict(server="state-store", model=HitlDecideIn, tier=Tier.TIER_3_MUTATE,
                        subagents={"reviewer"}),
    "hitl_list": dict(server="state-store", model=HitlListIn, tier=Tier.TIER_1_READ, subagents={"reviewer", "verification"}),
    "audit_verify": dict(server="state-store", model=AuditVerifyIn, tier=Tier.TIER_1_READ,
                         subagents={"reviewer", "verification"}),
}
