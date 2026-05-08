"""Unit tests for retrieval pipeline response formatting."""

from retriever_service.retrieval.pipeline import _format_result


class TestFormatResult:
    def test_strips_contextual_prefix_from_chunk_text(self):
        result = _format_result(
            {
                "chunk_id": "chunk-1",
                "text": "[Foo v. Bar | opinion:majority]\nBody text here.",
                "metadata": {
                    "case_id": 123,
                    "case_name": "Foo v. Bar",
                },
                "rrf_score": 0.5,
            },
            return_debug=False,
        )

        assert result["text"] == "Body text here."
