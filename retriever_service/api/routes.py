"""API route definitions for the retriever service (spec §8).

All endpoints except GET /health require the stores and pipeline to be
initialised in main.py's lifespan and attached to app.state.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, Path, Query, Request
from fastapi.responses import JSONResponse

from retriever_service.api.schemas import (
    AnchorsResponse,
    CaseEdgesResponse,
    CaseResponse,
    DeleteDocumentResponse,
    DeleteSessionDocumentsResponse,
    HealthResponse,
    ComponentStatus,
    IngestJobResponse,
    IngestRequest,
    IngestResponse,
    ListDocumentsResponse,
    DocumentInfo,
    SessionDocumentTotals,
    RetrieveRequest,
    RetrieveResponse,
    RetrievalStatsResponse,
    SubgraphRequest,
    SubgraphResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()

VERSION = "0.2.0"
_BACKGROUND_INGEST_THRESHOLD_BYTES = 1 * 1024 * 1024


def _ingest_documents(
    *,
    state: Any,
    body: IngestRequest,
) -> tuple[int, int]:
    embedder = getattr(state, "embedder", None)
    chroma = getattr(state, "chroma", None)
    tantivy = getattr(state, "tantivy", None)
    relational = getattr(state, "relational", None)

    if not all([embedder, chroma, tantivy, relational]):
        raise HTTPException(status_code=503, detail={"error": "stores_not_initialized"})

    from retriever_service.ingestion.dual_writer import DualIndexWriter
    from retriever_service.ingestion.chunker import chunk_user_upload
    from retriever_service.normalize import normalize
    from retriever_service.stores.chunk_ids import compute_chunk_id
    from retriever_service.stores.chroma_store import COLLECTION_USER_WORKSPACE
    from retriever_service.stores.tantivy_store import INDEX_USER_WORKSPACE

    corpus = body.corpus
    session_id = body.session_id or ""

    _MAX_FILE_BYTES = 50 * 1024 * 1024
    _MAX_FILES = 200
    _MAX_CHUNKS = 20_000

    existing_sources: set[str] = set()
    existing_chunk_count = 0
    if session_id:
        try:
            col = chroma._collections[COLLECTION_USER_WORKSPACE]
            existing = col.get(
                where={"session_id": {"$eq": session_id}},
                include=["metadatas"],
            )
            for metadata in (existing.get("metadatas") or []):
                source = metadata.get("source")
                if source:
                    existing_sources.add(source)
            existing_chunk_count = len(existing.get("ids") or [])
        except Exception:
            pass

    chunks_indexed = 0
    duplicates_skipped = 0

    for doc in body.documents:
        log_row_id = relational.log_ingestion_start(corpus, "chunk", doc.source)
        doc_bytes = len(doc.text.encode("utf-8"))
        if doc_bytes > _MAX_FILE_BYTES:
            relational.log_ingestion_failed(
                log_row_id,
                f"payload_too_large:{doc.source}:{doc_bytes}",
            )
            raise HTTPException(
                status_code=413,
                detail={
                    "error": "payload_too_large",
                    "source": doc.source,
                    "size_bytes": doc_bytes,
                    "limit_bytes": _MAX_FILE_BYTES,
                },
            )

        is_new_source = doc.source not in existing_sources
        if session_id and is_new_source and len(existing_sources) >= _MAX_FILES:
            relational.log_ingestion_failed(
                log_row_id,
                f"session_files_limit_reached:{doc.source}",
            )
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "session_files_limit_reached",
                    "limit": _MAX_FILES,
                    "current": len(existing_sources),
                },
            )

        try:
            if session_id and not is_new_source:
                logger.info(
                    "Prefix-delete: removing stale chunks for source=%s session=%s",
                    doc.source, session_id,
                )
                chroma.delete_by_filter(
                    COLLECTION_USER_WORKSPACE,
                    {"$and": [
                        {"session_id": {"$eq": session_id}},
                        {"source": {"$eq": doc.source}},
                    ]},
                )
                tantivy.delete_by_filter(
                    INDEX_USER_WORKSPACE,
                    {"session_id": session_id, "source": doc.source},
                )

            normalized_text = normalize(doc.text)
            chunk_results = chunk_user_upload(
                text=normalized_text,
                source=doc.source,
                cfg=state.cfg.chunking,
            )

            document_chunk_count = 0
            with DualIndexWriter(
                corpus=corpus,
                embedder=embedder,
                chroma=chroma,
                tantivy=tantivy,
                relational=relational,
            ) as writer:
                for cr in chunk_results:
                    if session_id and (existing_chunk_count + chunks_indexed) >= _MAX_CHUNKS:
                        raise HTTPException(
                            status_code=409,
                            detail={
                                "error": "session_chunks_limit_reached",
                                "limit": _MAX_CHUNKS,
                                "current": existing_chunk_count + chunks_indexed,
                            },
                        )

                    chunk_id = compute_chunk_id(cr.text, doc.source, session_id)
                    if relational.is_ingested(corpus, chunk_id):
                        duplicates_skipped += 1
                        continue

                    metadata: dict[str, Any] = {
                        "case_id": "",
                        "section_type": "user_upload",
                        "source": doc.source,
                        "chunk_idx": cr.chunk_idx,
                        **doc.metadata,
                    }
                    if session_id:
                        metadata["session_id"] = session_id

                    writer.add_chunk_auto_embed(
                        chunk_id=chunk_id,
                        text_with_prefix=cr.text_with_prefix,
                        metadata=metadata,
                    )
                    chunks_indexed += 1
                    document_chunk_count += 1
            if session_id:
                relational.upsert_document(
                    session_id=session_id,
                    source=doc.source,
                    chunk_count=document_chunk_count,
                    size_bytes=doc_bytes,
                )
            relational.log_ingestion_done(log_row_id)
        except Exception as exc:
            relational.log_ingestion_failed(log_row_id, str(exc))
            raise

        if is_new_source:
            existing_sources.add(doc.source)

    return chunks_indexed, duplicates_skipped


def _run_ingest_job(*, state: Any, body: IngestRequest, job_id: str) -> None:
    try:
        _ingest_documents(state=state, body=body)
        state.relational.update_job(job_id, status="done")
    except Exception as exc:
        logger.exception("Background ingest job failed: %s", job_id)
        state.relational.update_job(job_id, status="failed", error=str(exc))


# ---------------------------------------------------------------------------
# Health (§8.8)
# ---------------------------------------------------------------------------

@router.get("/health", response_model=HealthResponse, tags=["ops"])
async def health(request: Request) -> HealthResponse:
    """Return service health and per-component status."""
    state = request.app.state

    def _probe(name: str) -> str:
        try:
            store = getattr(state, name, None)
            if store is None:
                return "not_initialized"
            return store.health()
        except Exception as exc:
            logger.warning("Health probe failed for %s: %s", name, exc)
            return "error"

    chroma_status = _probe("chroma")
    tantivy_status = _probe("tantivy")
    relational_status = _probe("relational")
    reranker_status = _probe("reranker")

    overall = (
        "ok"
        if all(s in ("ok", "loaded") for s in (chroma_status, tantivy_status, relational_status))
        else "degraded"
    )

    return HealthResponse(
        status=overall,
        components=ComponentStatus(
            chroma=chroma_status,
            tantivy=tantivy_status,
            relational=relational_status,
            reranker=reranker_status,
            normalizer="ok",
        ),
        version=VERSION,
    )


# ---------------------------------------------------------------------------
# Retrieve (§8.1)
# ---------------------------------------------------------------------------

@router.post("/retrieve", response_model=RetrieveResponse, tags=["retrieval"])
async def retrieve(body: RetrieveRequest, request: Request) -> RetrieveResponse:
    """Hybrid BM25+dense retrieval with optional cross-encoder reranking."""
    from retriever_service.retrieval.pipeline import RetrievalError

    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(status_code=503, detail={"error": "retrieval_pipeline_not_initialized"})

    try:
        result = await pipeline.retrieve(
            query=body.query,
            session_id=body.session_id,
            k=body.k,
            corpora=body.corpora,
            filters=body.filters.to_dict(),
            rerank=body.rerank,
            return_debug=body.return_debug,
            retrievers=body.retrievers,
        )
    except RetrievalError as exc:
        raise HTTPException(status_code=503, detail={"error": "all_retrievers_unavailable"})
    except ValueError as exc:
        raise HTTPException(status_code=400, detail={"error": "validation", "details": [str(exc)]})

    return RetrieveResponse(**result)


# ---------------------------------------------------------------------------
# Ingest (§8.2)
# ---------------------------------------------------------------------------

@router.post("/ingest", response_model=IngestResponse, tags=["ingestion"])
async def ingest(
    body: IngestRequest,
    request: Request,
    background_tasks: BackgroundTasks,
) -> IngestResponse:
    """Ingest documents into the specified corpus.

    Enforces per-session limits (spec §4.5.4):
      - 50 MB max per document (413)
      - 200 max files per session (409)
      - 20 000 max chunks per session (409)

    Prefix-delete (spec §4.5.1 Bug B-5): if a source filename already exists
    in the session, its old chunks are deleted from both indexes before
    re-indexing, so stale chunks never linger.
    """
    corpus = body.corpus
    if any(len(doc.text.encode("utf-8")) > _BACKGROUND_INGEST_THRESHOLD_BYTES for doc in body.documents):
        job_id = request.app.state.relational.create_job()
        background_tasks.add_task(
            _run_ingest_job,
            state=request.app.state,
            body=body,
            job_id=job_id,
        )
        return IngestResponse(
            chunks_indexed=0,
            duplicates_skipped=0,
            corpus=corpus,
            status="processing",
            job_id=job_id,
        )

    chunks_indexed, duplicates_skipped = _ingest_documents(state=request.app.state, body=body)
    return IngestResponse(
        chunks_indexed=chunks_indexed,
        duplicates_skipped=duplicates_skipped,
        corpus=corpus,
    )


@router.get("/ingest/jobs/{job_id}", response_model=IngestJobResponse, tags=["ingestion"])
async def get_ingest_job(job_id: str, request: Request) -> IngestJobResponse:
    job = request.app.state.relational.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail={"error": "not_found", "job_id": job_id})
    return IngestJobResponse(**job)


# ---------------------------------------------------------------------------
# Session document management (§8.3)
# ---------------------------------------------------------------------------

@router.delete(
    "/sessions/{session_id}/documents",
    response_model=DeleteSessionDocumentsResponse,
    tags=["sessions"],
)
async def delete_session_documents(
    session_id: str = Path(..., description="Session ID"),
    request: Request = None,
) -> DeleteSessionDocumentsResponse:
    """Remove all chunks and retrieval log rows for a session (spec §8.3.1)."""
    state = request.app.state
    chroma = state.chroma
    tantivy = state.tantivy
    relational = state.relational
    from retriever_service.stores.chroma_store import COLLECTION_USER_WORKSPACE
    from retriever_service.stores.tantivy_store import INDEX_USER_WORKSPACE

    chroma_deleted = chroma.delete_by_filter(
        COLLECTION_USER_WORKSPACE, {"session_id": {"$eq": session_id}}
    )
    tantivy_deleted = tantivy.delete_by_filter(
        INDEX_USER_WORKSPACE, {"session_id": session_id}
    )
    log_rows = relational.delete_retrieval_log(session_id)
    relational.delete_documents_for_session(session_id)

    return DeleteSessionDocumentsResponse(
        chunks_deleted=max(chroma_deleted, tantivy_deleted),
        log_rows_deleted=log_rows,
    )


@router.delete(
    "/sessions/{session_id}/documents/{source:path}",
    response_model=DeleteDocumentResponse,
    tags=["sessions"],
)
async def delete_session_document(
    session_id: str = Path(...),
    source: str = Path(...),
    request: Request = None,
) -> DeleteDocumentResponse:
    """Remove a single uploaded document from a session (spec §8.3.2)."""
    state = request.app.state
    from retriever_service.stores.chroma_store import COLLECTION_USER_WORKSPACE
    from retriever_service.stores.tantivy_store import INDEX_USER_WORKSPACE

    chroma_deleted = state.chroma.delete_by_filter(
        COLLECTION_USER_WORKSPACE,
        {"$and": [{"session_id": {"$eq": session_id}}, {"source": {"$eq": source}}]},
    )
    tantivy_deleted = state.tantivy.delete_by_filter(
        INDEX_USER_WORKSPACE, {"session_id": session_id, "source": source}
    )
    state.relational.delete_document(session_id, source)

    deleted = max(chroma_deleted, tantivy_deleted)
    if deleted == 0:
        raise HTTPException(status_code=404, detail={"error": "not_found", "source": source})

    return DeleteDocumentResponse(chunks_deleted=deleted, source=source)


@router.get(
    "/sessions/{session_id}/documents",
    response_model=ListDocumentsResponse,
    tags=["sessions"],
)
async def list_session_documents(
    session_id: str = Path(...),
    request: Request = None,
) -> ListDocumentsResponse:
    """List distinct uploaded documents for a session (spec §8.3.3)."""
    rows = request.app.state.relational.list_documents(session_id)
    documents = [
        DocumentInfo(
            source=row["source"],
            chunks=int(row["chunk_count"]),
            ingested_at=row.get("ingested_at"),
            size_bytes=int(row.get("size_bytes", 0)),
        )
        for row in rows
    ]
    total_chunks = sum(d.chunks for d in documents)
    total_size_bytes = sum((d.size_bytes or 0) for d in documents)

    return ListDocumentsResponse(
        session_id=session_id,
        documents=documents,
        totals=SessionDocumentTotals(
            documents=len(documents),
            chunks=total_chunks,
            size_bytes=total_size_bytes,
        ),
    )


# ---------------------------------------------------------------------------
# Cases (§8.4)
# ---------------------------------------------------------------------------

@router.get("/cases/{case_id}", response_model=CaseResponse, tags=["cases"])
async def get_case(
    case_id: int = Path(..., description="CAP case ID"),
    request: Request = None,
) -> CaseResponse:
    """Return case-level metadata derived from the chunks index (spec §8.4)."""
    state = request.app.state
    meta = state.case_meta.get_case(case_id)
    if meta is None:
        raise HTTPException(status_code=404, detail={"error": "not_found", "case_id": case_id})

    try:
        chunk_count = state.tantivy.count_by_case_id("ny_case_law", case_id)
    except Exception:
        chunk_count = 0

    return CaseResponse(
        case_id=case_id,
        case_name=meta.get("case_name"),
        citation_official=meta.get("citation_official"),
        decision_date=meta.get("decision_date"),
        court_name=meta.get("court_name"),
        jurisdiction=meta.get("jurisdiction"),
        pagerank_percentile=meta.get("pagerank_percentile"),
        ocr_confidence=meta.get("ocr_confidence"),
        good_law=bool(meta.get("good_law", True)),
        corpus="ny_case_law",
        chunk_count=chunk_count,
    )


# ---------------------------------------------------------------------------
# Case edges (§8.5)
# ---------------------------------------------------------------------------

@router.get(
    "/cases/{case_id}/edges",
    response_model=CaseEdgesResponse,
    response_model_by_alias=True,
    tags=["graph"],
)
async def get_case_edges(
    case_id: int = Path(...),
    direction: str = Query(default="both", pattern="^(out|in|both)$"),
    limit: int = Query(default=50, ge=1, le=500),
    request: Request = None,
) -> CaseEdgesResponse:
    """Citation graph edges for a case (spec §8.5)."""
    from retriever_service.graph.edges import get_edges

    result = get_edges(
        case_id=case_id,
        relational=request.app.state.relational,
        case_meta=request.app.state.case_meta,
        direction=direction,
        limit=limit,
    )

    return CaseEdgesResponse(
        case_id=result["case_id"],
        **{"in": result["in"], "out": result["out"]},
    )


# ---------------------------------------------------------------------------
# Subgraph (§8.6)
# ---------------------------------------------------------------------------

@router.post(
    "/graph/subgraph",
    response_model=SubgraphResponse,
    response_model_by_alias=True,
    tags=["graph"],
)
async def post_subgraph(body: SubgraphRequest, request: Request) -> SubgraphResponse:
    """Return a subgraph for visualization (spec §8.6)."""
    from retriever_service.graph.subgraph import build_subgraph

    result = build_subgraph(
        seed_case_ids=body.seed_case_ids,
        relational=request.app.state.relational,
        case_meta=request.app.state.case_meta,
        depth=body.depth,
        include_external_neighbors=body.include_external_neighbors,
        max_neighbors_per_seed=body.max_neighbors_per_seed,
    )

    nodes = [
        {"case_id": n["case_id"], "case_name": n["case_name"],
         "pagerank_percentile": n["pagerank_percentile"], "is_seed": n["is_seed"]}
        for n in result["nodes"]
    ]
    edges = [{"from": e["from"], "to": e["to"]} for e in result["edges"]]

    return SubgraphResponse(
        nodes=nodes,
        edges=edges,
    )


# ---------------------------------------------------------------------------
# Anchors (§8.7)
# ---------------------------------------------------------------------------

@router.get(
    "/sessions/{session_id}/anchors",
    response_model=AnchorsResponse,
    tags=["sessions"],
)
async def get_anchors(
    session_id: str = Path(...),
    min_hits: int = Query(default=2, ge=1),
    limit: int = Query(default=20, ge=1, le=100),
    weight_by_pagerank: bool = Query(default=True),
    request: Request = None,
) -> AnchorsResponse:
    """Return anchor cases for a session (spec §8.7)."""
    from retriever_service.graph.anchors import get_anchors as _get_anchors

    result = _get_anchors(
        session_id=session_id,
        relational=request.app.state.relational,
        case_meta=request.app.state.case_meta,
        min_hits=min_hits,
        limit=limit,
        weight_by_pagerank=weight_by_pagerank,
    )

    return AnchorsResponse(**result)


# ---------------------------------------------------------------------------
# Admin stats (§8.8 / Phase 7.2)
# ---------------------------------------------------------------------------

@router.get(
    "/admin/retrieval_stats",
    response_model=RetrievalStatsResponse,
    tags=["admin"],
)
async def get_retrieval_stats(
    since: str = Query(
        default=None,
        description="ISO date (YYYY-MM-DD) lower bound. Defaults to last 30 days.",
        pattern=r"^\d{4}-\d{2}-\d{2}$",
    ),
    request: Request = None,
) -> RetrievalStatsResponse:
    """Per-day retrieval volume stats for operational monitoring (spec §Phase 7.2).

    Returns one row per calendar day with: query_count, chunk_count,
    unique_cases, session count, and top-rank-1 ratio (a rough precision proxy).
    """
    relational = request.app.state.relational
    rows = relational.retrieval_stats(since=since)
    return RetrievalStatsResponse(since=since, rows=rows)
