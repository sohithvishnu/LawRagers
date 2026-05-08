"""Unit tests for RelationalStore using an in-memory SQLite database (spec §4.3)."""

import pytest
import sqlite3
from unittest.mock import patch

from retriever_service.stores.relational_store import RelationalStore, ensure_tables


@pytest.fixture
def tmp_db(tmp_path) -> str:
    db_path = str(tmp_path / "test.db")
    ensure_tables(db_path)
    return db_path


@pytest.fixture
def store(tmp_db) -> RelationalStore:
    return RelationalStore(tmp_db)


class TestEnsureTables:
    def test_tables_created(self, tmp_db):
        conn = sqlite3.connect(tmp_db)
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        conn.close()
        assert "citations" in tables
        assert "retrieval_log" in tables
        assert "ingestion_log" in tables
        assert "documents" in tables

    def test_idempotent(self, tmp_db):
        ensure_tables(tmp_db)  # second call must not raise
        ensure_tables(tmp_db)


class TestCitationEdges:
    def test_add_and_get_out_edges(self, store):
        store.add_citation_edges(1, [2, 3, 4])
        result = store.get_edges(1, direction="out")
        assert set(result["out"]) == {2, 3, 4}
        assert result["in"] == []

    def test_add_and_get_in_edges(self, store):
        store.add_citation_edges(10, [99])
        result = store.get_edges(99, direction="in")
        assert result["in"] == [10]
        assert result["out"] == []

    def test_both_direction(self, store):
        store.add_citation_edges(1, [2])
        store.add_citation_edges(3, [1])
        result = store.get_edges(1, direction="both")
        assert 2 in result["out"]
        assert 3 in result["in"]

    def test_duplicate_edges_ignored(self, store):
        n1 = store.add_citation_edges(1, [2])
        n2 = store.add_citation_edges(1, [2])  # duplicate
        result = store.get_edges(1, direction="out")
        assert len(result["out"]) == 1

    def test_empty_cited_ids_returns_zero(self, store):
        count = store.add_citation_edges(1, [])
        assert count == 0

    def test_invalid_direction_raises(self, store):
        with pytest.raises(ValueError):
            store.get_edges(1, direction="sideways")

    def test_limit_respected(self, store):
        store.add_citation_edges(1, list(range(10)))
        result = store.get_edges(1, direction="out", limit=3)
        assert len(result["out"]) <= 3


class TestSubgraphEdges:
    def test_internal_edges_only(self, store):
        store.add_citation_edges(1, [2, 3])
        store.add_citation_edges(2, [99])  # 99 is outside seed set
        edges = store.get_subgraph_edges([1, 2, 3])
        edge_pairs = set(edges)
        assert (1, 2) in edge_pairs
        assert (1, 3) in edge_pairs
        assert (2, 99) not in edge_pairs

    def test_empty_input(self, store):
        assert store.get_subgraph_edges([]) == []


class TestRetrievalLog:
    def test_log_and_anchors(self, store):
        # Two separate queries each returning case_id=100 once.
        store.log_retrieval(
            "sess-A", "hash1",
            [("chunk-1", 100, 1, 0.9)],
        )
        store.log_retrieval(
            "sess-A", "hash2",
            [("chunk-1", 100, 1, 0.85)],
        )
        anchors = store.anchors("sess-A", min_hits=2, limit=10)
        assert len(anchors) == 1
        assert anchors[0]["case_id"] == 100
        assert anchors[0]["hits"] == 2

    def test_anchors_min_hits_filter(self, store):
        store.log_retrieval("sess-B", "h1", [("c1", 10, 1, None)])
        anchors = store.anchors("sess-B", min_hits=2)
        assert anchors == []

    def test_delete_retrieval_log(self, store):
        store.log_retrieval("sess-C", "h1", [("c1", 10, 1, None)])
        deleted = store.delete_retrieval_log("sess-C")
        assert deleted > 0
        anchors = store.anchors("sess-C", min_hits=1)
        assert anchors == []

    def test_prune_old_rows(self, tmp_db):
        # Insert a row with a very old timestamp directly
        conn = sqlite3.connect(tmp_db)
        conn.execute(
            "INSERT INTO retrieval_log (session_id, query_hash, chunk_id, rank, retrieved_at) "
            "VALUES ('s', 'h', 'c', 1, datetime('now', '-400 days'))"
        )
        conn.commit()
        conn.close()

        store = RelationalStore(tmp_db)
        deleted = store.prune_retrieval_log(older_than_days=365)
        assert deleted >= 1


class TestIngestionLog:
    def test_start_done_cycle(self, store):
        row_id = store.log_ingestion_start("corpus", "chunk", "target-1")
        assert row_id > 0
        store.log_ingestion_done(row_id)
        assert store.is_ingested("corpus", "target-1")

    def test_is_ingested_false_before_done(self, store):
        store.log_ingestion_start("corpus", "chunk", "target-2")
        assert not store.is_ingested("corpus", "target-2")

    def test_failed_not_ingested(self, store):
        row_id = store.log_ingestion_start("corpus", "chunk", "target-3")
        store.log_ingestion_failed(row_id, "boom")
        assert not store.is_ingested("corpus", "target-3")


class TestHealth:
    def test_health_ok(self, store):
        assert store.health() == "ok"


class TestDocuments:
    def test_upsert_list_and_delete_documents(self, store):
        store.upsert_document(
            session_id="sess-1",
            source="brief.pdf",
            chunk_count=4,
            size_bytes=1024,
        )
        store.upsert_document(
            session_id="sess-1",
            source="notes.txt",
            chunk_count=2,
            size_bytes=128,
        )

        documents = store.list_documents("sess-1")

        assert documents == [
            {
                "source": "brief.pdf",
                "chunk_count": 4,
                "size_bytes": 1024,
                "ingested_at": documents[0]["ingested_at"],
            },
            {
                "source": "notes.txt",
                "chunk_count": 2,
                "size_bytes": 128,
                "ingested_at": documents[1]["ingested_at"],
            },
        ]

        deleted = store.delete_document("sess-1", "brief.pdf")
        assert deleted == 1
        assert [row["source"] for row in store.list_documents("sess-1")] == ["notes.txt"]
