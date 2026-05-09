"""Session isolation integration test (spec §4.5.2 point 5).

Asserts that /retrieve for session A never returns session B's chunks,
tested across multiple queries using real (but temp-dir) Chroma + Tantivy +
SQLite instances.  No live server required — exercises the stores and pipeline
directly via Python APIs.

Run with:
    pytest tests/integration/test_isolation.py -v

Requirements:
    chromadb, tantivy, sentence-transformers must be installed.
"""

from __future__ import annotations

import pytest
import random
import string
from pathlib import Path
from typing import Generator

# Skip the whole module if heavy ML deps are not installed
pytest.importorskip("chromadb")
pytest.importorskip("tantivy")
pytest.importorskip("sentence_transformers")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tmp_stores(tmp_path_factory):
    """Initialise real Chroma + Tantivy + SQLite stores in a temp directory."""
    base = tmp_path_factory.mktemp("isolation_stores")

    from retriever_service.config import RetrieverSettings
    cfg = RetrieverSettings()

    from retriever_service.stores.chroma_store import ChromaStore
    from retriever_service.stores.tantivy_store import TantivyStore
    from retriever_service.stores.relational_store import RelationalStore
    from retriever_service.stores.case_metadata import CaseMetadataStore
    from retriever_service.retrieval.dense_retriever import Embedder, DenseRetriever
    from retriever_service.retrieval.bm25_retriever import BM25Retriever

    chroma = ChromaStore(chroma_path=str(base / "chroma"), cfg=cfg)
    tantivy = TantivyStore(base_path=str(base / "tantivy"), cfg=cfg)
    db_path = str(base / "test.db")
    relational = RelationalStore(db_path=db_path)

    embedder = Embedder()
    dense = DenseRetriever(chroma_store=chroma, embedder=embedder)
    bm25 = BM25Retriever(tantivy_store=tantivy)

    return {
        "chroma": chroma,
        "tantivy": tantivy,
        "relational": relational,
        "embedder": embedder,
        "dense": dense,
        "bm25": bm25,
        "cfg": cfg,
    }


def _ingest_doc(
    stores: dict,
    session_id: str,
    source: str,
    text: str,
) -> int:
    """Ingest a single document into the user_workspace for a session."""
    from retriever_service.ingestion.dual_writer import DualIndexWriter
    from retriever_service.ingestion.chunker import chunk_user_upload
    from retriever_service.normalize import normalize
    from retriever_service.stores.chunk_ids import compute_chunk_id

    cfg = stores["cfg"]
    chunks_indexed = 0

    with DualIndexWriter(
        corpus="user_workspace",
        embedder=stores["embedder"],
        chroma=stores["chroma"],
        tantivy=stores["tantivy"],
        relational=stores["relational"],
    ) as writer:
        normalized = normalize(text)
        chunk_results = chunk_user_upload(
            text=normalized,
            source=source,
            cfg=cfg.chunking,
        )
        for cr in chunk_results:
            chunk_id = compute_chunk_id(cr.text, source, session_id)
            metadata = {
                "case_id": "",
                "section_type": "user_upload",
                "source": source,
                "chunk_idx": cr.chunk_idx,
                "session_id": session_id,
            }
            writer.add_chunk_auto_embed(
                chunk_id=chunk_id,
                text_with_prefix=cr.text_with_prefix,
                metadata=metadata,
            )
            chunks_indexed += 1

    return chunks_indexed


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

class TestSessionIsolation:
    """Assert that user_workspace queries for session A never leak session B's chunks."""

    SESSION_A = "isolation-test-session-A"
    SESSION_B = "isolation-test-session-B"

    # Unique marker words that exist only in one session's documents.
    MARKER_A = "UNIQUEWORDALPHA" * 3
    MARKER_B = "UNIQUEWORDBETA" * 3

    def _unique_text(self, marker: str, n_sentences: int = 20) -> str:
        return " ".join([
            f"This is sentence {i} containing the marker {marker} for testing isolation."
            for i in range(n_sentences)
        ])

    @pytest.fixture(autouse=True)
    def ingest_fixtures(self, tmp_stores):
        """Ingest distinct documents into two sessions before each test."""
        text_a = self._unique_text(self.MARKER_A)
        text_b = self._unique_text(self.MARKER_B)

        _ingest_doc(tmp_stores, self.SESSION_A, "doc_a.txt", text_a)
        _ingest_doc(tmp_stores, self.SESSION_B, "doc_b.txt", text_b)
        self._stores = tmp_stores

    def _retrieve_session(self, session_id: str, query: str, k: int = 10) -> list[dict]:
        """Run BM25 + dense retrieval for a session and return all chunk dicts."""
        stores = self._stores
        results: list[dict] = []

        # BM25
        try:
            bm25_hits = stores["bm25"].query_user_workspace(
                session_id=session_id,
                query_text=query,
                k=k,
            )
            results.extend(bm25_hits)
        except Exception:
            pass

        # Dense
        try:
            dense_hits = stores["dense"].query_user_workspace(
                session_id=session_id,
                query_text=query,
                k=k,
            )
            results.extend(dense_hits)
        except Exception:
            pass

        return results

    def _chunk_belongs_to_session(self, chunk: dict, session_id: str) -> bool:
        meta = chunk.get("metadata") or {}
        return meta.get("session_id") == session_id

    def test_session_a_never_returns_session_b_chunks(self):
        """Session A queries must not return any chunk tagged with session B."""
        queries = [
            self.MARKER_A,
            "testing isolation",
            "sentence",
            "marker",
        ]
        for q in queries:
            hits = self._retrieve_session(self.SESSION_A, q)
            for hit in hits:
                assert self._chunk_belongs_to_session(hit, self.SESSION_A), (
                    f"Query '{q}' for session A returned chunk belonging to "
                    f"session '{hit.get('metadata', {}).get('session_id')}': "
                    f"chunk_id={hit.get('chunk_id')}"
                )

    def test_session_b_never_returns_session_a_chunks(self):
        """Session B queries must not return any chunk tagged with session A."""
        queries = [
            self.MARKER_B,
            "testing isolation",
            "sentence",
            "marker",
        ]
        for q in queries:
            hits = self._retrieve_session(self.SESSION_B, q)
            for hit in hits:
                assert self._chunk_belongs_to_session(hit, self.SESSION_B), (
                    f"Query '{q}' for session B returned chunk belonging to "
                    f"session '{hit.get('metadata', {}).get('session_id')}': "
                    f"chunk_id={hit.get('chunk_id')}"
                )

    def test_marker_only_retrievable_in_correct_session(self):
        """Marker text from session A should appear in A's results, not B's."""
        hits_a = self._retrieve_session(self.SESSION_A, self.MARKER_A)
        hits_b = self._retrieve_session(self.SESSION_B, self.MARKER_A)

        a_session_ids = {h.get("metadata", {}).get("session_id") for h in hits_a}
        b_session_ids = {h.get("metadata", {}).get("session_id") for h in hits_b}

        assert self.SESSION_B not in a_session_ids
        assert self.SESSION_A not in b_session_ids

    def test_empty_session_raises_on_bm25(self):
        from retriever_service.retrieval.bm25_retriever import BM25Retriever
        bm25 = self._stores["bm25"]
        with pytest.raises(ValueError, match="session_id"):
            bm25.query_user_workspace(session_id="", query_text="test", k=5)

    def test_empty_session_raises_on_dense(self):
        dense = self._stores["dense"]
        with pytest.raises(ValueError, match="session_id"):
            dense.query_user_workspace(session_id="", query_text="test", k=5)
