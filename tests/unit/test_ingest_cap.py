"""Unit tests for scripts/ingest_cap.py idempotency boundaries."""

from types import SimpleNamespace

from scripts.ingest_cap import ingest_case


class _FakeWriter:
    def __init__(self):
        self.chunks = []
        self.citation_edges = []

    def add_chunk_auto_embed(self, **kwargs):
        self.chunks.append(kwargs)

    def add_citation_edges(self, case_id, cited_ids):
        self.citation_edges.append((case_id, cited_ids))


class _FakeRelational:
    def __init__(self):
        self.is_ingested_calls = []
        self.started = []
        self.done = []
        self.failed = []

    def is_ingested(self, corpus, target_id):
        self.is_ingested_calls.append((corpus, target_id))
        return False

    def log_ingestion_start(self, corpus, kind, target_id):
        self.started.append((corpus, kind, target_id))
        return 1

    def log_ingestion_done(self, row_id):
        self.done.append(row_id)

    def log_ingestion_failed(self, row_id, error):
        self.failed.append((row_id, error))


class _ChunkResult:
    def __init__(self, chunk_idx, text, text_with_prefix):
        self.chunk_idx = chunk_idx
        self.text = text
        self.text_with_prefix = text_with_prefix


class TestIngestCase:
    def test_uses_only_case_level_ingestion_sentinel(self):
        relational = _FakeRelational()
        writer = _FakeWriter()
        case = {
            "id": 1117516,
            "name_abbreviation": "Abbott v. Case",
            "decision_date": "1868-01-01",
            "court": {"name": "Court"},
            "jurisdiction": {"name": "NY"},
            "citations": [{"type": "official", "cite": "4 Abb. Ct. App. 1"}],
            "cites_to": [{"case_ids": [2, 3]}],
            "casebody": {
                "head_matter": "Head matter text.",
                "opinions": [],
            },
        }

        ingest_case(
            case=case,
            writer=writer,
            embedder=None,
            relational=relational,
            normalize_fn=lambda text: text,
            chunk_fn=lambda **kwargs: [_ChunkResult(0, "Chunk text", "[prefix]\nChunk text")],
            chunk_id_fn=lambda text, source, session_id: "chunk-1",
            cfg=SimpleNamespace(chunking=SimpleNamespace()),
            dry_run=False,
        )

        assert relational.is_ingested_calls == [("ny_case_law", "1117516")]
        assert relational.started == [("ny_case_law", "chunk", "1117516")]
        assert relational.done == [1]
