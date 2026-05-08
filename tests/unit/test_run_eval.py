"""Unit tests for eval/run_eval.py helpers."""

from eval.run_eval import (
    METRICS,
    build_retriever_payload,
    evaluate_variant,
    extract_ranked_case_ids,
)


class TestExtractRankedCaseIds:
    def test_current_retriever_contract(self):
        data = {
            "results": [
                {"case_id": 10},
                {"case_id": 10},
                {"case_id": 11},
                {"case_id": None},
            ]
        }

        assert extract_ranked_case_ids(data) == ["10", "11"]

    def test_legacy_cases_contract(self):
        data = {
            "cases": [
                {"case_id": 10},
                {"case_id": 10},
                {"case_id": 11},
            ]
        }

        assert extract_ranked_case_ids(data) == ["10", "11"]


class TestEvaluateVariant:
    def test_returns_zero_metrics_for_empty_run(self):
        qrels = {"q1": {"10": 1}}
        run = {"q1": {}}

        result = evaluate_variant(qrels, run, METRICS)

        assert result == {metric: 0.0 for metric in METRICS}


class TestBuildRetrieverPayload:
    def test_defaults_to_no_rerank_for_phase4_eval(self):
        payload = build_retriever_payload(
            session_id="eval-session",
            query="test query",
            corpora=["ny_case_law"],
            rerank=False,
        )

        assert payload["rerank"] is False
        assert payload["corpora"] == ["ny_case_law"]
