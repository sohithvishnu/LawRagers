"""Unit tests for dense retrieval filter composition."""

from retriever_service.retrieval.dense_retriever import _build_chroma_where


class TestBuildChromaWhere:
    def test_extra_clauses_merge_into_and_expression(self):
        where = _build_chroma_where(
            {
                "min_ocr_confidence": 0.9,
                "exclude_section_types": ["head_matter"],
            },
            extra_clauses=[{"session_id": {"$eq": "sess-1"}}],
        )

        assert "$and" in where
        assert {"session_id": {"$eq": "sess-1"}} in where["$and"]
        assert {"ocr_confidence": {"$gte": 0.9}} in where["$and"]
        assert {"section_type": {"$ne": "head_matter"}} in where["$and"]

    def test_single_extra_clause_returns_plain_clause(self):
        where = _build_chroma_where({}, extra_clauses=[{"session_id": {"$eq": "sess-1"}}])

        assert where == {"session_id": {"$eq": "sess-1"}}
