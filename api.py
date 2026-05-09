import os
from contextlib import asynccontextmanager

# --- 🛑 STOP TENSORFLOW INTERFERENCE ---
os.environ["USE_TF"] = "NO"
os.environ["USE_TORCH"] = "YES"

import httpx
import io
import json
import sqlite3
import uuid

import chromadb
import ollama
import PyPDF2

from fastapi import FastAPI, HTTPException, Path, Query, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from httpx import HTTPStatusError
from pydantic import BaseModel
from typing import Any, List, Optional

from indexer import get_device, make_embedding_function, chunk_text, doc_id

from retriever_service.retriever_client import (
    build_search_response,
    get_retriever_url,
    ingest_user_document,
    retrieve_for_search,
)


# ---------------------------------------------------------------------------
# Shared HTTP client (lifespan)
# ---------------------------------------------------------------------------

_http: httpx.AsyncClient


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _http
    _http = httpx.AsyncClient(base_url=get_retriever_url(), timeout=30.0)
    yield
    await _http.aclose()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# ChromaDB client (Library Manager — firm libraries only)
# ---------------------------------------------------------------------------

_device = get_device()
_ef = make_embedding_function(_device)
_chroma = chromadb.PersistentClient(path="./chroma_db")
_SYSTEM_COLLECTIONS = {"user_workspace"}


# ---------------------------------------------------------------------------
# SQLite setup
# ---------------------------------------------------------------------------

db_conn = sqlite3.connect("legal_sessions.db", check_same_thread=False)
cursor = db_conn.cursor()

cursor.execute("PRAGMA journal_mode=WAL")

cursor.execute("""
    CREATE TABLE IF NOT EXISTS sessions (
        id TEXT PRIMARY KEY,
        name TEXT,
        description TEXT,
        databases TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
""")

cursor.execute("""
    CREATE TABLE IF NOT EXISTS messages (
        id TEXT PRIMARY KEY,
        session_id TEXT,
        role TEXT,
        content TEXT,
        graph_state TEXT DEFAULT '[]',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(session_id) REFERENCES sessions(id)
    )
""")

try:
    cursor.execute("ALTER TABLE messages ADD COLUMN graph_state TEXT DEFAULT '[]'")
except sqlite3.OperationalError:
    pass

# Retriever service tables (spec §4.3) — created here so they exist before
# the first request regardless of whether the retriever service has started.
cursor.executescript("""
    CREATE TABLE IF NOT EXISTS citations (
        citing_case_id INTEGER NOT NULL,
        cited_case_id  INTEGER NOT NULL,
        PRIMARY KEY (citing_case_id, cited_case_id)
    ) WITHOUT ROWID;

    CREATE INDEX IF NOT EXISTS idx_citations_cited
        ON citations(cited_case_id);

    CREATE TABLE IF NOT EXISTS retrieval_log (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id   TEXT NOT NULL,
        query_hash   TEXT NOT NULL,
        chunk_id     TEXT NOT NULL,
        case_id      INTEGER,
        rank         INTEGER NOT NULL,
        rerank_score REAL,
        retrieved_at TEXT NOT NULL DEFAULT (datetime('now'))
    );

    CREATE INDEX IF NOT EXISTS idx_retrieval_session_case
        ON retrieval_log(session_id, case_id);

    CREATE INDEX IF NOT EXISTS idx_retrieval_session_time
        ON retrieval_log(session_id, retrieved_at);

    CREATE TABLE IF NOT EXISTS ingestion_log (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        corpus       TEXT NOT NULL,
        kind         TEXT NOT NULL,
        target_id    TEXT NOT NULL,
        status       TEXT NOT NULL,
        started_at   TEXT NOT NULL DEFAULT (datetime('now')),
        finished_at  TEXT,
        error        TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_ingestion_target ON ingestion_log(corpus, target_id);
""")

db_conn.commit()


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class SessionCreate(BaseModel):
    name: str
    description: str
    databases: str


class LegalArgument(BaseModel):
    session_id: str
    argument: str


class GenerateRequest(BaseModel):
    session_id: str
    argument: str
    context_text: str


class ChatMessage(BaseModel):
    session_id: str
    role: str
    content: str
    graph_state: Optional[str] = "[]"


class SubgraphProxyRequest(BaseModel):
    seed_case_ids: list[int]
    depth: int = 1
    include_external_neighbors: bool = False
    max_neighbors_per_seed: int = 20


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_session(session_id: str) -> None:
    """Raise 404 if session_id does not exist in the local sessions table."""
    cursor.execute("SELECT 1 FROM sessions WHERE id = ?", (session_id,))
    if cursor.fetchone() is None:
        raise HTTPException(
            status_code=404,
            detail={"error": "session_not_found", "session_id": session_id},
        )


async def _proxy(
    method: str,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    json: Any = None,
) -> Any:
    """Forward a request to the retriever service and surface errors clearly."""
    try:
        r = await _http.request(method, path, params=params, json=json)
    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail={"error": "retriever_unavailable", "hint": "Is the retriever service running on port 8001?"},
        )
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail={"error": "retriever_timeout"})

    if r.status_code in (404, 422):
        raise HTTPException(status_code=r.status_code, detail=r.json())
    if 400 <= r.status_code < 500:
        raise HTTPException(status_code=r.status_code, detail=r.json())
    if r.status_code >= 500:
        raise HTTPException(
            status_code=502,
            detail={"error": "retriever_error", "upstream_status": r.status_code},
        )

    return r.json()


# ---------------------------------------------------------------------------
# Sessions & chat memory
# ---------------------------------------------------------------------------

@app.post("/sessions")
def create_session(req: SessionCreate):
    session_id = str(uuid.uuid4())
    cursor.execute(
        "INSERT INTO sessions (id, name, description, databases) VALUES (?, ?, ?, ?)",
        (session_id, req.name, req.description, req.databases),
    )
    db_conn.commit()
    return {"id": session_id, "name": req.name}


@app.get("/sessions")
def get_sessions():
    cursor.execute(
        "SELECT id, name, description, databases, created_at FROM sessions ORDER BY created_at DESC"
    )
    rows = cursor.fetchall()
    return {
        "sessions": [
            {"id": r[0], "name": r[1], "description": r[2], "databases": r[3], "created_at": r[4]}
            for r in rows
        ]
    }


@app.get("/sessions/{session_id}/messages")
def get_session_messages(session_id: str):
    _require_session(session_id)
    cursor.execute(
        "SELECT id, role, content, graph_state FROM messages WHERE session_id = ? ORDER BY created_at ASC",
        (session_id,),
    )
    rows = cursor.fetchall()
    return {
        "messages": [
            {"id": r[0], "role": r[1], "text": r[2], "graph_state": r[3]}
            for r in rows
        ]
    }


@app.post("/messages")
def save_message(req: ChatMessage):
    _require_session(req.session_id)
    msg_id = str(uuid.uuid4())
    cursor.execute(
        "INSERT INTO messages (id, session_id, role, content, graph_state) VALUES (?, ?, ?, ?, ?)",
        (msg_id, req.session_id, req.role, req.content, req.graph_state),
    )
    db_conn.commit()
    return {"status": "success", "id": msg_id}


# ---------------------------------------------------------------------------
# File upload
# ---------------------------------------------------------------------------

@app.post("/upload")
async def upload_document(session_id: str = Form(...), file: UploadFile = File(...)):
    _require_session(session_id)
    contents = await file.read()

    if file.filename.endswith(".pdf"):
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(contents))
        text = "".join(page.extract_text() + "\n" for page in pdf_reader.pages)
    else:
        text = contents.decode("utf-8")

    try:
        ingest_result = await ingest_user_document(
            session_id=session_id,
            source=file.filename,
            text=text,
        )
    except HTTPStatusError as exc:
        return {"status": "error", "filename": file.filename, "detail": exc.response.text}

    return {
        "status": "success",
        "filename": file.filename,
        "chunks_indexed": ingest_result.get("chunks_indexed", 0),
        "job_id": ingest_result.get("job_id"),
    }


# ---------------------------------------------------------------------------
# Search & generation
# ---------------------------------------------------------------------------

@app.post("/search")
def search_cases(req: LegalArgument):
    _require_session(req.session_id)
    try:
        payload = retrieve_for_search(req.argument, req.session_id)
    except HTTPStatusError as exc:
        return {"cases": [], "context_text": "", "detail": exc.response.text}
    return build_search_response(payload)


@app.post("/generate")
def generate_memo(req: GenerateRequest):
    _require_session(req.session_id)
    cursor.execute(
        "SELECT role, content FROM messages WHERE session_id = ? ORDER BY created_at ASC",
        (req.session_id,),
    )
    history = cursor.fetchall()

    system_prompt = (
        "You are a legal AI workspace. Analyze the user's argument or question based on "
        "the provided USER UPLOADED DOCUMENTS and BINDING PRECEDENT. Synthesize the facts "
        "of the upload with the rules of the precedent. Use IRAC format."
    )
    messages = [{"role": "system", "content": system_prompt}]
    for row in history:
        messages.append({"role": row[0], "content": row[1]})
    messages.append({"role": "user", "content": f"USER QUERY:\n{req.argument}\n\n{req.context_text}"})

    def stream_generator():
        for chunk in ollama.chat(model="llama3", messages=messages, stream=True):
            yield chunk["message"]["content"]

    return StreamingResponse(stream_generator(), media_type="text/plain")


# ---------------------------------------------------------------------------
# Proxy: ingest job polling
# ---------------------------------------------------------------------------

@app.get("/ingest/jobs/{job_id}")
async def get_ingest_job(job_id: str):
    return await _proxy("GET", f"/ingest/jobs/{job_id}")


# ---------------------------------------------------------------------------
# Proxy: session document management
# ---------------------------------------------------------------------------

@app.get("/sessions/{session_id}/documents")
async def list_session_documents(session_id: str = Path(...)):
    _require_session(session_id)
    return await _proxy("GET", f"/sessions/{session_id}/documents")


@app.delete("/sessions/{session_id}/documents")
async def delete_session_documents(session_id: str = Path(...)):
    """Remove all uploaded documents for a session (preserves chat history)."""
    _require_session(session_id)
    return await _proxy("DELETE", f"/sessions/{session_id}/documents")


@app.delete("/sessions/{session_id}/documents/{source:path}")
async def delete_session_document(
    session_id: str = Path(...),
    source: str = Path(...),
):
    _require_session(session_id)
    return await _proxy("DELETE", f"/sessions/{session_id}/documents/{source}")


# ---------------------------------------------------------------------------
# Proxy: cases
# ---------------------------------------------------------------------------

@app.get("/cases/{case_id}")
async def get_case(case_id: int = Path(...)):
    return await _proxy("GET", f"/cases/{case_id}")


@app.get("/cases/{case_id}/edges")
async def get_case_edges(
    case_id: int = Path(...),
    direction: str = Query(default="both", pattern="^(out|in|both)$"),
    limit: int = Query(default=50, ge=1, le=500),
):
    return await _proxy(
        "GET",
        f"/cases/{case_id}/edges",
        params={"direction": direction, "limit": limit},
    )


# ---------------------------------------------------------------------------
# Proxy: graph
# ---------------------------------------------------------------------------

@app.post("/graph/subgraph")
async def post_subgraph(body: SubgraphProxyRequest):
    return await _proxy("POST", "/graph/subgraph", json=body.model_dump())


# ---------------------------------------------------------------------------
# Proxy: anchors
# ---------------------------------------------------------------------------

@app.get("/sessions/{session_id}/anchors")
async def get_anchors(
    session_id: str = Path(...),
    min_hits: int = Query(default=2, ge=1),
    limit: int = Query(default=20, ge=1, le=100),
    weight_by_pagerank: bool = Query(default=True),
):
    _require_session(session_id)
    return await _proxy(
        "GET",
        f"/sessions/{session_id}/anchors",
        params={"min_hits": min_hits, "limit": limit, "weight_by_pagerank": weight_by_pagerank},
    )


# ---------------------------------------------------------------------------
# Library Manager — firm-library collection management
# ---------------------------------------------------------------------------

@app.get("/databases")
def list_databases():
    names = [c.name for c in _chroma.list_collections()
             if c.name not in _SYSTEM_COLLECTIONS]
    return {"databases": names}


@app.post("/databases/create")
async def create_database(
    db_name: str = Form(...),
    files: List[UploadFile] = File(...),
):
    if db_name in _SYSTEM_COLLECTIONS:
        return {"error": f"'{db_name}' is a reserved collection name."}

    parsed: list[tuple[str, bytes]] = []
    for f in files:
        parsed.append((f.filename or "upload", await f.read()))

    def generate():
        collection = _chroma.get_or_create_collection(name=db_name, embedding_function=_ef)
        total = len(parsed)

        for i, (filename, content) in enumerate(parsed):
            text = ""
            if filename.lower().endswith(".pdf"):
                try:
                    pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
                    for page in pdf_reader.pages:
                        extracted = page.extract_text()
                        if extracted:
                            text += extracted + "\n"
                except Exception:
                    pass
            else:
                try:
                    text = content.decode("utf-8")
                except UnicodeDecodeError:
                    text = content.decode("latin-1", errors="ignore")

            chunks = chunk_text(text)
            if chunks:
                ids       = [doc_id(db_name, filename, c) for c in chunks]
                metadatas = [{"source": filename, "db_name": db_name} for _ in chunks]
                collection.upsert(documents=chunks, metadatas=metadatas, ids=ids)

            progress = round(((i + 1) / total) * 100)
            yield json.dumps({"progress": progress, "status": f"Indexed {filename}"}) + "\n"

        yield json.dumps({"progress": 100, "status": "complete"}) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")
