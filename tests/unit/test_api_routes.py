"""Unit tests for retriever_service API route contracts."""

from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from retriever_service.api.routes import router


class _CaseMetaStore:
    def get_case(self, case_id):
        return {
            "case_id": case_id,
            "case_name": "Test Case",
            "citation_official": "1 N.Y. 1",
            "decision_date": "2020-01-01",
            "court_name": "Court",
            "jurisdiction": "NY",
            "pagerank_percentile": 0.8,
            "ocr_confidence": 0.7,
            "good_law": True,
        }


class _CountingTantivy:
    def __init__(self) -> None:
        self.calls = []

    def count_by_case_id(self, index, case_id):
        self.calls.append((index, case_id))
        return 7

    def search(self, *args, **kwargs):
        raise AssertionError("route should use count_by_case_id instead of materializing hits")


def _make_client():
    app = FastAPI()
    app.include_router(router)
    app.state.case_meta = _CaseMetaStore()
    app.state.tantivy = _CountingTantivy()
    app.state.relational = SimpleNamespace(
        list_documents=lambda session_id: [
            {
                "source": "brief.pdf",
                "chunk_count": 4,
                "size_bytes": 1024,
                "ingested_at": "2026-05-04 10:00:00",
            }
        ]
    )
    return TestClient(app), app.state.tantivy


class TestCaseRoutes:
    def test_get_case_uses_count_by_case_id(self):
        client, tantivy = _make_client()

        response = client.get("/cases/1117516")

        assert response.status_code == 200
        assert response.json()["chunk_count"] == 7
        assert tantivy.calls == [("ny_case_law", 1117516)]

    def test_case_edges_serialize_alias_field_names(self, monkeypatch):
        client, _ = _make_client()

        monkeypatch.setattr(
            "retriever_service.graph.edges.get_edges",
            lambda **kwargs: {
                "case_id": 123,
                "in": [{"case_id": 1, "case_name": "Inbound", "pagerank_percentile": 0.4}],
                "out": [{"case_id": 2, "case_name": "Outbound", "pagerank_percentile": 0.6}],
            },
        )

        response = client.get("/cases/123/edges")

        assert response.status_code == 200
        body = response.json()
        assert "in" in body
        assert "in_" not in body

    def test_subgraph_serializes_from_alias(self, monkeypatch):
        client, _ = _make_client()

        monkeypatch.setattr(
            "retriever_service.graph.subgraph.build_subgraph",
            lambda **kwargs: {
                "nodes": [{"case_id": 1, "case_name": "Seed", "pagerank_percentile": 0.5, "is_seed": True}],
                "edges": [{"from": 1, "to": 2}],
            },
        )

        response = client.post("/graph/subgraph", json={"seed_case_ids": [1]})

        assert response.status_code == 200
        body = response.json()
        assert body["edges"][0]["from"] == 1
        assert "from_" not in body["edges"][0]

    def test_list_documents_uses_relational_store(self):
        client, _ = _make_client()

        response = client.get("/sessions/sess-1/documents")

        assert response.status_code == 200
        body = response.json()
        assert body["documents"] == [
            {
                "source": "brief.pdf",
                "chunks": 4,
                "ingested_at": "2026-05-04 10:00:00",
                "size_bytes": 1024,
            }
        ]
        assert body["totals"] == {"documents": 1, "chunks": 4, "size_bytes": 1024}
