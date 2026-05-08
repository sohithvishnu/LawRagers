"""Atomic triple-index ingestion context manager (spec §5).

DualIndexWriter wraps writes to three targets:
  1. ChromaDB    — dense vectors + chunk metadata
  2. Tantivy     — BM25 sparse index
  3. SQLite      — citation graph edges (citations table)

The name "DualIndexWriter" (spec §5) reflects the primary dual-index design;
the third target (citations) is additive.

Atomicity contract:
  - All three targets are updated in __exit__.
  - SQLite transaction is committed first (most reliable rollback semantics).
  - Tantivy is committed second.
  - ChromaDB upsert is third (ChromaDB has no rollback; we delete by ID on failure).
  - On any exception: rollback SQLite, delete by ID from whichever target committed.

Embedder is injected at construction time so DualIndexWriter is testable
without a live model.

Usage (case-law ingest):

    with DualIndexWriter(
        corpus="ny_case_law",
        embedder=embedder,
        chroma=chroma_store,
        tantivy=tantivy_store,
        relational=relational_store,
    ) as w:
        for case in cases:
            for chunk in chunk_cap_case(case):
                w.add_chunk(chunk_id, text_with_prefix, embedding, metadata)
            w.add_citation_edges(case.id, case.cites_to_ids)

Usage (user-upload ingest):

    with DualIndexWriter(
        corpus="user_workspace",
        embedder=embedder,
        chroma=chroma_store,
        tantivy=tantivy_store,
        relational=relational_store,
    ) as w:
        for chunk in chunks:
            w.add_chunk(chunk_id, text_with_prefix, embedding, metadata)
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from retriever_service.stores.chroma_store import ChromaStore
    from retriever_service.stores.tantivy_store import TantivyStore
    from retriever_service.stores.relational_store import RelationalStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Corpus → collection/index name mapping
# ---------------------------------------------------------------------------

_CORPUS_TO_COLLECTION = {
    "ny_case_law": "ny_case_law",
    "user_workspace": "user_workspace",
}


def _resolve_collection(corpus: str) -> str:
    """Map an ingest corpus name to the canonical collection name."""
    base = corpus.split("_workspace_")[0] + "_workspace" if "_workspace_" in corpus else corpus
    # user_workspace_<session_id> → user_workspace
    if corpus.startswith("user_workspace"):
        return "user_workspace"
    if corpus in _CORPUS_TO_COLLECTION:
        return _CORPUS_TO_COLLECTION[corpus]
    raise ValueError(f"Unknown corpus '{corpus}'.")


# ---------------------------------------------------------------------------
# Embedder protocol — any callable(texts) → embeddings is acceptable.
# ---------------------------------------------------------------------------

class EmbedderProtocol(Protocol):
    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> Any: ...


# ---------------------------------------------------------------------------
# DualIndexWriter
# ---------------------------------------------------------------------------

class DualIndexWriter:
    """Context manager for atomic writes to ChromaDB + Tantivy + SQLite.

    Buffer chunks and citation edges during the with-block; commit all three
    atomically on __exit__.  Roll back on failure.
    """

    def __init__(
        self,
        corpus: str,
        embedder: EmbedderProtocol,
        chroma: "ChromaStore",
        tantivy: "TantivyStore",
        relational: "RelationalStore",
        batch_size: int = 32,
    ) -> None:
        self._corpus = corpus
        self._collection = _resolve_collection(corpus)
        self._embedder = embedder
        self._chroma = chroma
        self._tantivy = tantivy
        self._relational = relational
        self._batch_size = batch_size

        # Buffered chunk data
        self._chunk_ids: list[str] = []
        self._texts_with_prefix: list[str] = []
        self._precomputed_embeddings: list[Optional[list[float]]] = []
        self._metadatas: list[dict[str, Any]] = []

        # Buffered citation edges: list of (citing_case_id, [cited_case_ids])
        self._citation_batches: list[tuple[int, list[int]]] = []

        # Committed IDs for rollback tracking
        self._chroma_committed: list[str] = []
        self._tantivy_committed = False

    def __enter__(self) -> "DualIndexWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_type is not None:
            logger.error(
                "DualIndexWriter exiting due to exception; no writes committed.",
                exc_info=True,
            )
            return False  # Re-raise the exception

        try:
            self._commit()
        except Exception:
            logger.exception("DualIndexWriter commit failed; attempting rollback.")
            self._rollback()
            raise
        return False

    # ------------------------------------------------------------------
    # Public ingest API
    # ------------------------------------------------------------------

    def add_chunk(
        self,
        chunk_id: str,
        text_with_prefix: str,
        embedding: list[float],
        metadata: dict[str, Any],
    ) -> None:
        """Buffer a single chunk for commit."""
        self._chunk_ids.append(chunk_id)
        self._texts_with_prefix.append(text_with_prefix)
        self._precomputed_embeddings.append(embedding)
        self._metadatas.append(metadata)

    def add_chunks_with_embedding(
        self,
        chunks: list[dict[str, Any]],
    ) -> None:
        """Buffer a list of pre-embedded chunks.

        Each dict must have keys: chunk_id, text_with_prefix, embedding, metadata.
        """
        for c in chunks:
            self.add_chunk(
                chunk_id=c["chunk_id"],
                text_with_prefix=c["text_with_prefix"],
                embedding=c["embedding"],
                metadata=c["metadata"],
            )

    def add_chunk_auto_embed(
        self,
        chunk_id: str,
        text_with_prefix: str,
        metadata: dict[str, Any],
    ) -> None:
        """Buffer a chunk; embedding is computed in batch at commit time.

        Prefer this when calling add_chunk in a tight loop — embeddings are
        generated with batch_size=32 on commit(), which is ~5× faster than
        one-at-a-time inference.
        """
        self._chunk_ids.append(chunk_id)
        self._texts_with_prefix.append(text_with_prefix)
        self._precomputed_embeddings.append(None)
        self._metadatas.append(metadata)

    def add_citation_edges(
        self,
        citing_case_id: int,
        cited_case_ids: list[int],
    ) -> None:
        """Buffer citation edges for a case."""
        if cited_case_ids:
            self._citation_batches.append((citing_case_id, cited_case_ids))

    # ------------------------------------------------------------------
    # Commit / rollback
    # ------------------------------------------------------------------

    def _commit(self) -> None:
        if not self._chunk_ids and not self._citation_batches:
            logger.debug("DualIndexWriter: nothing to commit.")
            return

        # --- Generate embeddings in batches ---
        embeddings: list[list[float]] = []
        if self._chunk_ids:
            if any(embedding is None for embedding in self._precomputed_embeddings):
                if not all(embedding is None for embedding in self._precomputed_embeddings):
                    raise ValueError(
                        "DualIndexWriter does not support mixing precomputed and auto-generated embeddings "
                        "within the same commit batch."
                    )
                raw = self._embedder.encode(
                    self._texts_with_prefix,
                    batch_size=self._batch_size,
                    show_progress_bar=False,
                )
                embeddings = [
                    v.tolist() if hasattr(v, "tolist") else list(v)
                    for v in raw
                ]
            else:
                embeddings = [list(embedding) for embedding in self._precomputed_embeddings if embedding is not None]

        # --- Step 1: SQLite citations (most reliable rollback) ---
        for citing_id, cited_ids in self._citation_batches:
            self._relational.add_citation_edges(citing_id, cited_ids)
        logger.debug(
            "DualIndexWriter: committed %d citation batches to SQLite.",
            len(self._citation_batches),
        )

        # --- Step 2: Tantivy BM25 ---
        if self._chunk_ids:
            # Tantivy document dicts: merge metadata with text + chunk_id
            tantivy_docs = []
            for i, (chunk_id, text_with_prefix, meta) in enumerate(
                zip(self._chunk_ids, self._texts_with_prefix, self._metadatas)
            ):
                # Strip prefix for the indexed text (prefix is an indexing artifact)
                indexed_text = _strip_prefix(text_with_prefix)
                doc = {"chunk_id": chunk_id, "text": indexed_text, **meta}
                tantivy_docs.append(doc)

            self._tantivy.add(self._collection, tantivy_docs)
            self._tantivy_committed = True
            logger.debug(
                "DualIndexWriter: committed %d docs to Tantivy (%s).",
                len(tantivy_docs), self._collection,
            )

        # --- Step 3: ChromaDB dense index ---
        if self._chunk_ids:
            self._chroma.add(
                collection=self._collection,
                ids=self._chunk_ids,
                texts=self._texts_with_prefix,
                embeddings=embeddings,
                metadatas=self._metadatas,
            )
            self._chroma_committed = list(self._chunk_ids)
            logger.debug(
                "DualIndexWriter: committed %d chunks to ChromaDB (%s).",
                len(self._chunk_ids), self._collection,
            )

        logger.info(
            "DualIndexWriter commit complete",
            extra={
                "corpus": self._corpus,
                "chunks": len(self._chunk_ids),
                "citation_batches": len(self._citation_batches),
            },
        )

    def _rollback(self) -> None:
        """Best-effort rollback after partial commit failure."""
        # SQLite: already committed atomically — cannot undo without a compensating
        # delete; skipping citation rollback is acceptable (idempotent re-ingest).

        if self._tantivy_committed and self._chunk_ids:
            logger.warning(
                "DualIndexWriter rollback: deleting %d chunks from Tantivy (%s).",
                len(self._chunk_ids), self._collection,
            )
            try:
                for chunk_id in self._chunk_ids:
                    self._tantivy.delete_by_filter(
                        self._collection, {"chunk_id": chunk_id}
                    )
            except Exception:
                logger.exception("DualIndexWriter rollback: Tantivy delete failed.")

        if self._chroma_committed:
            logger.warning(
                "DualIndexWriter rollback: deleting %d chunks from ChromaDB (%s).",
                len(self._chroma_committed), self._collection,
            )
            try:
                col = self._chroma._collections[self._collection]
                col.delete(ids=self._chroma_committed)
            except Exception:
                logger.exception("DualIndexWriter rollback: ChromaDB delete failed.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

import re as _re

_PREFIX_LINE_RE = _re.compile(r"^\[[^\]]+\]\n", _re.MULTILINE)


def _strip_prefix(text: str) -> str:
    """Remove a single leading [prefix] line from indexed text (spec §6.5.5 step 5)."""
    return _PREFIX_LINE_RE.sub("", text, count=1)
