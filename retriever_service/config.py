from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


# ---------------------------------------------------------------------------
# Nested config models — mirror config.yaml structure exactly.
# ---------------------------------------------------------------------------

class HNSWConfig(BaseModel):
    M: int = 16
    construction_ef: int = 200
    search_ef: int = 100
    space: str = "cosine"


class BM25Config(BaseModel):
    k1: float = 1.2
    b: float = 0.75


class RRFConfig(BaseModel):
    k: int = 60
    candidate_pool: int = 100


class RerankerConfig(BaseModel):
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    max_length: int = 512
    batch_size: int = 16
    enabled_default: bool = True
    top_n_input: int = 50
    top_n_output: int = 10
    pagerank_boost_weight: float = 0.0


class FiltersConfig(BaseModel):
    default_min_ocr_confidence: float = 0.3


class GraphConfig(BaseModel):
    subgraph_max_depth: int = 2
    subgraph_max_neighbors: int = 50


class AnchorsConfig(BaseModel):
    default_min_hits: int = 2
    default_weight_by_pagerank: bool = True


class CaseMetadataCacheConfig(BaseModel):
    max_entries: int = 1024
    ttl_seconds: int = 10


class ContextualPrefixConfig(BaseModel):
    enabled: bool = True


class HeadMatterStructuralConfig(BaseModel):
    detect_held_points: bool = True
    detect_attorneys_block: bool = True
    held_points_max: int = 20


class OpinionStructuralConfig(BaseModel):
    detect_roman_headers: bool = True
    detect_letter_subheaders: bool = True
    detect_named_headers: bool = True
    headers_max: int = 15


class StructuralSplitConfig(BaseModel):
    head_matter: HeadMatterStructuralConfig = Field(default_factory=HeadMatterStructuralConfig)
    opinion: OpinionStructuralConfig = Field(default_factory=OpinionStructuralConfig)


class ChunkingConfig(BaseModel):
    target_tokens: int = 256
    overlap_tokens: int = 40
    min_chunk_tokens: int = 30
    max_chunk_tokens: int = 320
    tokenizer_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    sentence_splitter: str = "pysbd"
    contextual_prefix: ContextualPrefixConfig = Field(default_factory=ContextualPrefixConfig)
    structural_split: StructuralSplitConfig = Field(default_factory=StructuralSplitConfig)


class RelationalStoreConfig(BaseModel):
    database_path: str = "./legal_sessions.db"
    retrieval_log_retention_days: int = 365


class TimeoutsConfig(BaseModel):
    bm25: int = 2000
    dense: int = 2000
    rerank: int = 3000


class CircuitBreakerConfig(BaseModel):
    consecutive_failures: int = 5
    cooldown_seconds: int = 60


class ServiceConfig(BaseModel):
    port: int = 8001
    chroma_path: str = "./chroma_db"
    tantivy_path: str = "./tantivy_index"


# ---------------------------------------------------------------------------
# Root settings — loaded once at import time from config.yaml.
# Environment variable RETRIEVER_CONFIG_PATH overrides the yaml location.
# ---------------------------------------------------------------------------

class RetrieverSettings(BaseModel):
    hnsw: HNSWConfig = Field(default_factory=HNSWConfig)
    bm25: BM25Config = Field(default_factory=BM25Config)
    rrf: RRFConfig = Field(default_factory=RRFConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    filters: FiltersConfig = Field(default_factory=FiltersConfig)
    graph: GraphConfig = Field(default_factory=GraphConfig)
    anchors: AnchorsConfig = Field(default_factory=AnchorsConfig)
    case_metadata_cache: CaseMetadataCacheConfig = Field(default_factory=CaseMetadataCacheConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    relational_store: RelationalStoreConfig = Field(default_factory=RelationalStoreConfig)
    timeouts_ms: TimeoutsConfig = Field(default_factory=TimeoutsConfig)
    circuit_breaker: CircuitBreakerConfig = Field(default_factory=CircuitBreakerConfig)
    service: ServiceConfig = Field(default_factory=ServiceConfig)


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_settings(config_path: Optional[str] = None) -> RetrieverSettings:
    """Load settings from YAML file, falling back to defaults for missing keys."""
    resolved = Path(
        config_path
        or os.environ.get("RETRIEVER_CONFIG_PATH", "")
        or Path(__file__).parent / "config.yaml"
    )
    data = _load_yaml(resolved) if resolved.exists() else {}
    return RetrieverSettings(**data)


# Module-level singleton — imported everywhere as `from retriever_service.config import settings`
settings: RetrieverSettings = load_settings()
