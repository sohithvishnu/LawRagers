"""Unit tests for case metadata lookup behavior."""

from retriever_service.stores.case_metadata import CaseMetadataStore


class _DummyTantivy:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int, int]] = []

    def get_by_case_id(self, index: str, case_id: int, k: int = 1):
        self.calls.append((index, case_id, k))
        return [
            {
                "metadata": {
                    "case_id": case_id,
                    "case_name": "Test Case",
                    "citation_official": "1 N.Y. 1",
                }
            }
        ]

    def search(self, *args, **kwargs):
        raise AssertionError("string query search should not be used for case_id lookup")


class _DummyChroma:
    _collections = {}


class _DummyCacheConfig:
    max_entries = 16
    ttl_seconds = 10


class _DummyConfig:
    case_metadata_cache = _DummyCacheConfig()


class TestCaseMetadataStore:
    def test_uses_numeric_tantivy_lookup_for_case_id(self):
        tantivy = _DummyTantivy()
        store = CaseMetadataStore(
            tantivy_store=tantivy,
            chroma_store=_DummyChroma(),
            cfg=_DummyConfig(),
        )

        result = store.get_case(1117516)

        assert result is not None
        assert result["case_id"] == 1117516
        assert tantivy.calls == [("ny_case_law", 1117516, 1)]
