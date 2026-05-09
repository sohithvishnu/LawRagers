"""Unit tests for DualIndexWriter buffering behavior."""

from retriever_service.ingestion.dual_writer import DualIndexWriter


class _DummyEmbedder:
    def encode(self, texts, batch_size=32, show_progress_bar=False):
        raise AssertionError("embedder should not run when embeddings are precomputed")


class _DummyChroma:
    def __init__(self) -> None:
        self.calls = []

    def add(self, collection, ids, texts, embeddings, metadatas):
        self.calls.append(
            {
                "collection": collection,
                "ids": ids,
                "texts": texts,
                "embeddings": embeddings,
                "metadatas": metadatas,
            }
        )


class _DummyTantivy:
    def __init__(self) -> None:
        self.calls = []

    def add(self, index, docs):
        self.calls.append({"index": index, "docs": docs})


class _DummyRelational:
    def add_citation_edges(self, citing_case_id, cited_case_ids):
        return 0


class TestDualIndexWriter:
    def test_preserves_precomputed_embeddings_and_does_not_store_text_bare_metadata(self):
        chroma = _DummyChroma()
        tantivy = _DummyTantivy()

        writer = DualIndexWriter(
            corpus="user_workspace",
            embedder=_DummyEmbedder(),
            chroma=chroma,
            tantivy=tantivy,
            relational=_DummyRelational(),
        )

        with writer:
            writer.add_chunk(
                chunk_id="chunk-1",
                text_with_prefix="[source.txt]\nBody text",
                embedding=[0.1, 0.2, 0.3],
                metadata={"case_id": "", "source": "source.txt", "session_id": "sess-1"},
            )

        assert chroma.calls[0]["embeddings"] == [[0.1, 0.2, 0.3]]
        assert "_text_bare" not in chroma.calls[0]["metadatas"][0]
