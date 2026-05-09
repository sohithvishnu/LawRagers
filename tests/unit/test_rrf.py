"""Unit tests for retriever_service.retrieval.rrf (spec §7.1)."""

import pytest
from retriever_service.retrieval.rrf import fuse


def _make_result(chunk_id: str, text: str = "text") -> dict:
    return {"chunk_id": chunk_id, "text": text, "metadata": {}}


class TestFuseBasics:
    def test_empty_inputs_return_empty(self):
        assert fuse([], []) == []

    def test_bm25_only(self):
        bm25 = [_make_result("a"), _make_result("b")]
        result = fuse(bm25, [])
        ids = [r["chunk_id"] for r in result]
        assert "a" in ids and "b" in ids

    def test_dense_only(self):
        dense = [_make_result("x"), _make_result("y")]
        result = fuse([], dense)
        ids = [r["chunk_id"] for r in result]
        assert "x" in ids and "y" in ids

    def test_union_of_both_lists(self):
        bm25 = [_make_result("a"), _make_result("b")]
        dense = [_make_result("b"), _make_result("c")]
        result = fuse(bm25, dense)
        ids = {r["chunk_id"] for r in result}
        assert ids == {"a", "b", "c"}

    def test_overlap_scores_higher(self):
        # "b" appears in both lists at rank 1 → highest RRF score
        bm25 = [_make_result("b"), _make_result("a")]
        dense = [_make_result("b"), _make_result("c")]
        result = fuse(bm25, dense)
        assert result[0]["chunk_id"] == "b"


class TestRRFFormula:
    def test_rrf_score_annotation(self):
        bm25 = [_make_result("a")]
        dense = [_make_result("a")]
        result = fuse(bm25, dense, k=60)
        # score(a) = 1/(60+1) + 1/(60+1) = 2/61
        expected = 2.0 / 61.0
        assert abs(result[0]["rrf_score"] - expected) < 1e-9

    def test_single_list_rank1_score(self):
        bm25 = [_make_result("a")]
        result = fuse(bm25, [], k=60)
        expected = 1.0 / 61.0
        assert abs(result[0]["rrf_score"] - expected) < 1e-9


class TestRankAnnotations:
    def test_bm25_rank_annotated(self):
        bm25 = [_make_result("a"), _make_result("b")]
        result = fuse(bm25, [])
        rank_map = {r["chunk_id"]: r["bm25_rank"] for r in result}
        assert rank_map["a"] == 1
        assert rank_map["b"] == 2

    def test_dense_rank_annotated(self):
        dense = [_make_result("x"), _make_result("y")]
        result = fuse([], dense)
        rank_map = {r["chunk_id"]: r["dense_rank"] for r in result}
        assert rank_map["x"] == 1
        assert rank_map["y"] == 2

    def test_bm25_rank_none_for_dense_only_chunk(self):
        dense = [_make_result("x")]
        result = fuse([], dense)
        assert result[0]["bm25_rank"] is None

    def test_dense_rank_none_for_bm25_only_chunk(self):
        bm25 = [_make_result("a")]
        result = fuse(bm25, [])
        assert result[0]["dense_rank"] is None


class TestTopN:
    def test_top_n_truncates(self):
        bm25 = [_make_result(f"b{i}") for i in range(10)]
        dense = [_make_result(f"d{i}") for i in range(10)]
        result = fuse(bm25, dense, top_n=5)
        assert len(result) == 5

    def test_top_n_none_returns_all(self):
        bm25 = [_make_result(f"b{i}") for i in range(5)]
        dense = [_make_result(f"d{i}") for i in range(5)]
        result = fuse(bm25, dense, top_n=None)
        assert len(result) == 10

    def test_sorted_descending(self):
        bm25 = [_make_result("a"), _make_result("b"), _make_result("c")]
        dense = [_make_result("c"), _make_result("b")]
        result = fuse(bm25, dense)
        scores = [r["rrf_score"] for r in result]
        assert scores == sorted(scores, reverse=True)


class TestDataCarryOver:
    def test_text_preserved_from_first_seen_list(self):
        bm25 = [{"chunk_id": "a", "text": "from bm25", "metadata": {}}]
        dense = [{"chunk_id": "a", "text": "from dense", "metadata": {}}]
        result = fuse(bm25, dense)
        # BM25 was processed first; its text should be preserved.
        assert result[0]["text"] == "from bm25"
