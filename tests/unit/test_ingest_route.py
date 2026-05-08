"""Unit tests for /ingest route bookkeeping."""

from contextlib import nullcontext
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from retriever_service.api.routes import router


class _FakeDualIndexWriter:
    def __init__(self, **kwargs):
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def add_chunk_auto_embed(self, chunk_id, text_with_prefix, metadata):
        self.calls.append((chunk_id, text_with_prefix, metadata))


class _FakeRelational:
    def __init__(self) -> None:
        self.started = []
        self.done = []
        self.failed = []
        self.documents = []
        self.jobs = {}

    def log_ingestion_start(self, corpus, kind, target_id):
        self.started.append((corpus, kind, target_id))
        return len(self.started)

    def log_ingestion_done(self, row_id):
        self.done.append(row_id)

    def log_ingestion_failed(self, row_id, error):
        self.failed.append((row_id, error))

    def is_ingested(self, corpus, target_id):
        return False

    def upsert_document(self, *, session_id, source, chunk_count, size_bytes):
        self.documents.append((session_id, source, chunk_count, size_bytes))

    def create_job(self):
        job_id = f"job-{len(self.jobs) + 1}"
        self.jobs[job_id] = {"job_id": job_id, "status": "processing", "error": None}
        return job_id

    def update_job(self, job_id, *, status, error=None):
        self.jobs[job_id] = {"job_id": job_id, "status": status, "error": error}

    def get_job(self, job_id):
        return self.jobs.get(job_id)


def _make_client(monkeypatch):
    app = FastAPI()
    app.include_router(router)
    app.state.embedder = object()
    app.state.chroma = SimpleNamespace(_collections={"user_workspace": SimpleNamespace(get=lambda **kwargs: {"ids": [], "metadatas": []})})
    app.state.tantivy = object()
    app.state.relational = _FakeRelational()
    app.state.cfg = SimpleNamespace(chunking=SimpleNamespace())

    monkeypatch.setattr("retriever_service.ingestion.dual_writer.DualIndexWriter", _FakeDualIndexWriter)
    monkeypatch.setattr(
        "retriever_service.ingestion.chunker.chunk_user_upload",
        lambda text, source, cfg: [SimpleNamespace(text="Body text", text_with_prefix="[source.txt]\nBody text", chunk_idx=0)],
    )
    monkeypatch.setattr("retriever_service.normalize.normalize", lambda text: text)
    monkeypatch.setattr("retriever_service.stores.chunk_ids.compute_chunk_id", lambda text, source, session_id: "chunk-1")
    return TestClient(app), app.state.relational


class TestIngestRoute:
    def test_ingest_logs_start_and_done_per_document(self, monkeypatch):
        client, relational = _make_client(monkeypatch)

        response = client.post(
            "/ingest",
            json={
                "corpus": "user_workspace",
                "session_id": "sess-1",
                "documents": [{"source": "source.txt", "text": "Body text", "metadata": {}}],
            },
        )

        assert response.status_code == 200
        assert relational.started == [("user_workspace", "chunk", "source.txt")]
        assert relational.done == [1]
        assert relational.failed == []

    def test_large_ingest_returns_background_job(self, monkeypatch):
        client, relational = _make_client(monkeypatch)

        response = client.post(
            "/ingest",
            json={
                "corpus": "user_workspace",
                "session_id": "sess-1",
                "documents": [{"source": "source.txt", "text": "x" * (1024 * 1024 + 1), "metadata": {}}],
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "processing"
        assert body["job_id"] == "job-1"
        assert relational.jobs["job-1"]["status"] in {"processing", "done"}

    def test_job_status_endpoint(self, monkeypatch):
        client, relational = _make_client(monkeypatch)
        relational.jobs["job-1"] = {"job_id": "job-1", "status": "done", "error": None}

        response = client.get("/ingest/jobs/job-1")

        assert response.status_code == 200
        assert response.json() == {"job_id": "job-1", "status": "done", "error": None}
