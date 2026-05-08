"""Pydantic request/response schemas for the retriever service API (spec §8)."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# Health (§8.8)
# ---------------------------------------------------------------------------

class ComponentStatus(BaseModel):
    chroma: str
    tantivy: str
    relational: str
    reranker: Optional[str] = None
    normalizer: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    components: ComponentStatus
    version: str


# ---------------------------------------------------------------------------
# Retrieve (§8.1)
# ---------------------------------------------------------------------------

class RetrieveFilters(BaseModel):
    good_law: Optional[bool] = None
    min_ocr_confidence: Optional[float] = None
    min_pagerank_percentile: Optional[float] = None
    exclude_section_types: list[str] = Field(default_factory=list)
    exclude_opinion_types: list[str] = Field(default_factory=list)
    decision_date_gte: Optional[str] = None
    decision_date_lte: Optional[str] = None
    court_name: Optional[list[str]] = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {}
        if self.good_law is not None:
            d["good_law"] = self.good_law
        if self.min_ocr_confidence is not None:
            d["min_ocr_confidence"] = self.min_ocr_confidence
        if self.min_pagerank_percentile is not None:
            d["min_pagerank_percentile"] = self.min_pagerank_percentile
        if self.exclude_section_types:
            d["exclude_section_types"] = self.exclude_section_types
        if self.exclude_opinion_types:
            d["exclude_opinion_types"] = self.exclude_opinion_types
        if self.decision_date_gte:
            d["decision_date_gte"] = self.decision_date_gte
        if self.decision_date_lte:
            d["decision_date_lte"] = self.decision_date_lte
        if self.court_name:
            d["court_name"] = self.court_name
        return d


class RetrieveRequest(BaseModel):
    query: str
    k: int = 10
    session_id: Optional[str] = None
    corpora: list[str] = Field(default_factory=lambda: ["ny_case_law"])
    filters: RetrieveFilters = Field(default_factory=RetrieveFilters)
    rerank: bool = True
    return_debug: bool = False
    retrievers: list[str] = Field(
        default_factory=lambda: ["bm25", "dense"],
        description="Which retrieval legs to activate. Subset of ['bm25', 'dense'].",
    )

    @field_validator("retrievers")
    @classmethod
    def validate_retrievers(cls, v: list[str]) -> list[str]:
        valid = {"bm25", "dense"}
        unknown = set(v) - valid
        if unknown:
            raise ValueError(f"Unknown retrievers: {unknown}. Valid: {valid}")
        if not v:
            raise ValueError("retrievers must contain at least one of: bm25, dense")
        return v

    @field_validator("query")
    @classmethod
    def query_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must be non-empty.")
        return v

    @field_validator("session_id")
    @classmethod
    def session_id_non_empty_if_provided(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not v.strip():
            raise ValueError("session_id must be non-empty when provided.")
        return v

    @field_validator("corpora")
    @classmethod
    def validate_corpora(cls, v: list[str]) -> list[str]:
        valid = {"ny_case_law", "user_workspace"}
        for corpus in v:
            if corpus not in valid:
                raise ValueError(f"Unknown corpus '{corpus}'. Valid: {valid}")
        return v

    def model_post_init(self, __context: Any) -> None:
        if "user_workspace" in self.corpora and not self.session_id:
            raise ValueError(
                "session_id is required when corpora includes 'user_workspace'."
            )


class ChunkSource(BaseModel):
    case_name: Optional[str] = None
    citation_official: Optional[str] = None
    decision_date: Optional[str] = None
    court_name: Optional[str] = None
    corpus: Optional[str] = None
    section_type: Optional[str] = None
    opinion_type: Optional[str] = None
    opinion_author: Optional[str] = None
    pagerank_percentile: Optional[float] = None
    ocr_confidence: Optional[float] = None


class ChunkRanks(BaseModel):
    bm25: Optional[int] = None
    dense: Optional[int] = None
    rrf: Optional[int] = None
    rerank: Optional[int] = None


class ScoreComponents(BaseModel):
    reranker_logit: Optional[float] = None
    pagerank_boost: Optional[float] = None


class RetrieveResult(BaseModel):
    chunk_id: str
    case_id: Optional[int] = None
    text: str
    score: float
    source: ChunkSource
    ranks: Optional[ChunkRanks] = None
    score_components: Optional[ScoreComponents] = None


class RetrieveResponse(BaseModel):
    results: list[RetrieveResult]
    degraded: bool = False
    retrievers_used: list[str] = Field(default_factory=list)
    latency_ms: Optional[dict[str, int]] = None


# ---------------------------------------------------------------------------
# Ingest (§8.2)
# ---------------------------------------------------------------------------

class IngestDocument(BaseModel):
    text: str
    source: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestRequest(BaseModel):
    corpus: str
    session_id: Optional[str] = None
    documents: list[IngestDocument]

    @field_validator("documents")
    @classmethod
    def documents_non_empty(cls, v: list) -> list:
        if not v:
            raise ValueError("documents list must be non-empty.")
        return v


class IngestResponse(BaseModel):
    chunks_indexed: int
    duplicates_skipped: int
    corpus: str
    status: str = "done"
    job_id: Optional[str] = None


class IngestJobResponse(BaseModel):
    job_id: str
    status: str
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Session document management (§8.3)
# ---------------------------------------------------------------------------

class DeleteSessionDocumentsResponse(BaseModel):
    chunks_deleted: int
    log_rows_deleted: int


class DeleteDocumentResponse(BaseModel):
    chunks_deleted: int
    source: str


class DocumentInfo(BaseModel):
    source: str
    chunks: int
    ingested_at: Optional[str] = None
    size_bytes: Optional[int] = None


class SessionDocumentTotals(BaseModel):
    documents: int
    chunks: int
    size_bytes: int


class ListDocumentsResponse(BaseModel):
    session_id: str
    documents: list[DocumentInfo]
    totals: SessionDocumentTotals


# ---------------------------------------------------------------------------
# Cases (§8.4)
# ---------------------------------------------------------------------------

class CaseResponse(BaseModel):
    case_id: int
    case_name: Optional[str] = None
    citation_official: Optional[str] = None
    decision_date: Optional[str] = None
    court_name: Optional[str] = None
    jurisdiction: Optional[str] = None
    pagerank_percentile: Optional[float] = None
    ocr_confidence: Optional[float] = None
    good_law: bool = True
    corpus: str = "ny_case_law"
    chunk_count: int = 0


# ---------------------------------------------------------------------------
# Case edges (§8.5)
# ---------------------------------------------------------------------------

class CaseEdgeEntry(BaseModel):
    case_id: int
    case_name: Optional[str] = None
    pagerank_percentile: Optional[float] = None


class CaseEdgesResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    case_id: int
    out: list[CaseEdgeEntry] = Field(default_factory=list)
    in_: list[CaseEdgeEntry] = Field(default_factory=list, alias="in")


# ---------------------------------------------------------------------------
# Subgraph (§8.6)
# ---------------------------------------------------------------------------

class SubgraphRequest(BaseModel):
    seed_case_ids: list[int]
    depth: int = 1
    include_external_neighbors: bool = False
    max_neighbors_per_seed: int = 20

    @field_validator("depth")
    @classmethod
    def cap_depth(cls, v: int) -> int:
        return min(v, 2)

    @field_validator("seed_case_ids")
    @classmethod
    def non_empty_seeds(cls, v: list) -> list:
        if not v:
            raise ValueError("seed_case_ids must be non-empty.")
        return v


class SubgraphNode(BaseModel):
    case_id: int
    case_name: Optional[str] = None
    pagerank_percentile: Optional[float] = None
    is_seed: bool = False


class SubgraphEdge(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    from_: int = Field(alias="from")
    to: int


class SubgraphResponse(BaseModel):
    nodes: list[SubgraphNode]
    edges: list[SubgraphEdge]


# ---------------------------------------------------------------------------
# Anchors (§8.7)
# ---------------------------------------------------------------------------

class AnchorEntry(BaseModel):
    case_id: int
    case_name: Optional[str] = None
    hits: int
    first_retrieved_at: Optional[str] = None
    last_retrieved_at: Optional[str] = None
    pagerank_percentile: Optional[float] = None
    anchor_score: float


class AnchorsResponse(BaseModel):
    session_id: str
    anchors: list[AnchorEntry]


# ---------------------------------------------------------------------------
# Admin stats (§Phase 7.2)
# ---------------------------------------------------------------------------

class RetrievalStatDay(BaseModel):
    date: str
    query_count: int
    chunk_count: int
    unique_cases: int
    sessions: int
    top_rank_1_ratio: float


class RetrievalStatsResponse(BaseModel):
    since: Optional[str] = None
    rows: list[RetrievalStatDay]
