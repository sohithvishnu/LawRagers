"""Dense (HNSW) retriever backed by ChromaDB + sentence-transformer embedder (spec §7).

The embedder (sentence-transformers/all-MiniLM-L6-v2) is loaded once at service
startup and shared across all requests.  MPS device is used on Apple Silicon
with CPU fallback (spec §3 / CLAUDE.md tech stack).

Single chokepoint for all dense retrieval.  Enforces mandatory session_id
filter on user_workspace queries (spec §4.5.2).
"""

from __future__ import annotations

import logging
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from retriever_service.stores.chroma_store import ChromaStore

from retriever_service.stores.chroma_store import COLLECTION_CASE_LAW, COLLECTION_USER_WORKSPACE
from retriever_service.config import RetrieverSettings, settings as default_settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------

class Embedder:
    """Sentence-transformer embedding wrapper.

    Loads the model onto MPS (Apple Silicon) with CPU fallback.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        logger.info("Loading embedder %s on device=%s", model_name, device)
        self._model = SentenceTransformer(model_name, device=device)
        self._model_name = model_name
        self.device = device

    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> Any:
        """Encode a list of texts; returns a numpy array of shape (N, dim)."""
        return self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    def encode_query(self, query: str) -> list[float]:
        """Encode a single query string; returns a plain Python list."""
        vector = self._model.encode(
            [query],
            batch_size=1,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vector[0].tolist()


# ---------------------------------------------------------------------------
# Dense retriever
# ---------------------------------------------------------------------------

class DenseRetriever:
    """Dense retriever backed by ChromaDB HNSW."""

    def __init__(
        self,
        chroma_store: "ChromaStore",
        embedder: Embedder,
    ) -> None:
        self._chroma = chroma_store
        self._embedder = embedder

    def query_case_law(
        self,
        query_text: str,
        k: int,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """Query the global ny_case_law collection.

        Returns list of {chunk_id, text, metadata, distance}.
        """
        embedding = self._embedder.encode_query(query_text)
        where = _build_chroma_where(filters or {})
        results = self._chroma.query(
            collection=COLLECTION_CASE_LAW,
            embedding=embedding,
            k=k,
            where=where or None,
        )
        return results

    def query_user_workspace(
        self,
        session_id: str,
        query_text: str,
        k: int,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """Query user_workspace, enforcing mandatory session_id isolation.

        Raises ValueError if session_id is empty (spec §4.5.2 rule 3).
        """
        if not session_id:
            raise ValueError(
                "session_id is required and must be non-empty for user_workspace queries."
            )
        embedding = self._embedder.encode_query(query_text)
        where_dict = _build_chroma_where(
            filters or {},
            extra_clauses=[{"session_id": {"$eq": session_id}}],
        )
        results = self._chroma.query(
            collection=COLLECTION_USER_WORKSPACE,
            embedding=embedding,
            k=k,
            where=where_dict,
        )
        return results


# ---------------------------------------------------------------------------
# Chroma where-clause builder
# ---------------------------------------------------------------------------

def _build_chroma_where(
    filters: dict[str, Any],
    extra_clauses: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    """Convert retrieval filter primitives to a ChromaDB where clause."""
    clauses: list[dict[str, Any]] = []

    if "min_ocr_confidence" in filters:
        clauses.append({"ocr_confidence": {"$gte": float(filters["min_ocr_confidence"])}})

    if "min_pagerank_percentile" in filters:
        clauses.append({"pagerank_percentile": {"$gte": float(filters["min_pagerank_percentile"])}})

    if "good_law" in filters:
        val = 1 if filters["good_law"] else 0
        clauses.append({"good_law": {"$eq": val}})

    if "decision_date_gte" in filters:
        clauses.append({"decision_date": {"$gte": filters["decision_date_gte"]}})

    if "decision_date_lte" in filters:
        clauses.append({"decision_date": {"$lte": filters["decision_date_lte"]}})

    for section_type in filters.get("exclude_section_types", []):
        clauses.append({"section_type": {"$ne": section_type}})

    for opinion_type in filters.get("exclude_opinion_types", []):
        clauses.append({"opinion_type": {"$ne": opinion_type}})

    court_names: list[str] = filters.get("court_name", [])
    if court_names:
        clauses.append({"court_name": {"$in": court_names}})

    if extra_clauses:
        clauses.extend(extra_clauses)

    if not clauses:
        return {}
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}
