"""Unit tests for TantivyStore."""

from retriever_service.config import RetrieverSettings
from retriever_service.stores.tantivy_store import TantivyStore, _escape_query_text


class TestTantivyStore:
    def test_user_workspace_docs_do_not_require_numeric_case_id(self, tmp_path):
        store = TantivyStore(base_path=str(tmp_path / "tantivy"), cfg=RetrieverSettings())

        store.add(
            "user_workspace",
            [
                {
                    "chunk_id": "chunk-1",
                    "text": "Body text",
                    "case_id": "",
                    "section_type": "user_upload",
                    "source": "source.txt",
                    "chunk_idx": 0,
                    "session_id": "sess-1",
                }
            ],
        )

        hits = store.search(
            "user_workspace",
            query_text="Body",
            k=5,
            filters={"session_id": "sess-1"},
        )
        assert len(hits) == 1

    def test_search_escapes_query_parser_syntax(self, tmp_path):
        store = TantivyStore(base_path=str(tmp_path / "tantivy"), cfg=RetrieverSettings())

        store.add(
            "user_workspace",
            [
                {
                    "chunk_id": "chunk-1",
                    "text": "Lawless Hackett citation token",
                    "case_id": "",
                    "section_type": "user_upload",
                    "source": "source.txt",
                    "chunk_idx": 0,
                    "session_id": "sess-1",
                }
            ],
        )

        hits = store.search(
            "user_workspace",
            query_text="[CASE_Lawless_v_Hackett], [CITATION]: token",
            k=5,
            filters={"session_id": "sess-1"},
        )
        assert isinstance(hits, list)

    def test_search_escapes_apostrophes_in_query_text(self, tmp_path):
        store = TantivyStore(base_path=str(tmp_path / "tantivy"), cfg=RetrieverSettings())

        store.add(
            "user_workspace",
            [
                {
                    "chunk_id": "chunk-1",
                    "text": "The plaintiff's papers supported the motion",
                    "case_id": "",
                    "section_type": "user_upload",
                    "source": "source.txt",
                    "chunk_idx": 0,
                    "session_id": "sess-1",
                }
            ],
        )

        hits = store.search(
            "user_workspace",
            query_text="plaintiff's papers",
            k=5,
            filters={"session_id": "sess-1"},
        )
        assert isinstance(hits, list)


class TestEscapeQueryText:
    def test_escapes_reserved_query_parser_characters(self):
        escaped = _escape_query_text('[CASE_Foo_v_Bar]: "quoted"? test')

        assert r"\[CASE_Foo_v_Bar\]\: \"quoted\"\? test" == escaped

    def test_escapes_apostrophes(self):
        escaped = _escape_query_text("plaintiff's papers")

        assert r"plaintiff\'s papers" == escaped
