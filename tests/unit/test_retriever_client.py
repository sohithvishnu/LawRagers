"""Unit tests for api.py retriever delegation helpers."""

from retriever_service.retriever_client import build_search_response


class TestBuildSearchResponse:
    def test_translates_retriever_response_to_legacy_search_shape(self):
        payload = {
            "results": [
                {
                    "chunk_id": "chunk-1",
                    "case_id": 111,
                    "text": "Case chunk",
                    "score": 0.9,
                    "source": {
                        "case_name": "Case A",
                        "decision_date": "2020-01-01",
                        "corpus": "ny_case_law",
                    },
                },
                {
                    "chunk_id": "chunk-2",
                    "case_id": None,
                    "text": "Workspace chunk",
                    "score": 0.4,
                    "source": {
                        "corpus": "user_workspace",
                    },
                },
            ]
        }

        result = build_search_response(payload)

        assert result["cases"] == [
            {
                "id": "Case A",
                "case_id": 111,
                "date": "2020-01-01",
                "text": "Case chunk",
                "distance": 0.9,
            }
        ]
        assert "--- BINDING PRECEDENT (NY CASE LAW) ---" in result["context_text"]
        assert "--- USER UPLOADED DOCUMENTS ---" in result["context_text"]
        assert "Workspace chunk" in result["context_text"]
