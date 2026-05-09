"""Retriever service entry point.

Starts a FastAPI application on port 8001 (configurable via config.yaml).
Phase 2: all retrieval, ingest, graph, and session-management endpoints.

Run with:
    uvicorn retriever_service.main:app --port 8001 --reload

All heavy initialisation (index loading, model loading) is done in the
lifespan context manager so startup errors surface immediately rather than
on the first request.
"""

from __future__ import annotations

import logging
import os
import uuid
import warnings
from contextlib import asynccontextmanager
from importlib import import_module
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from retriever_service.config import load_settings
from retriever_service.api.routes import router
from retriever_service.logging_config import configure_logging, request_id_var

logger = logging.getLogger(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Attach a unique request_id to every request for structured log correlation."""

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        token = request_id_var.set(request_id)
        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            request_id_var.reset(token)


def warn_if_missing_pysbd(import_module=import_module) -> None:
    try:
        import_module("pysbd")
    except ImportError:
        warnings.warn(
            "pysbd not installed; chunker using regex fallback (lower recall).",
            stacklevel=2,
        )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialise and tear down shared resources."""
    cfg = load_settings()

    # Configure structured logging before any logger calls.
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    json_logs = os.environ.get("LOG_FORMAT", "json").lower() != "console"
    configure_logging(level=log_level, json_logs=json_logs)

    warn_if_missing_pysbd()

    chroma_path = os.path.abspath(cfg.service.chroma_path)
    tantivy_path = os.path.abspath(cfg.service.tantivy_path)
    db_path = os.path.abspath(cfg.relational_store.database_path)

    logger.info(
        "Retriever service starting",
        extra={
            "chroma_path": chroma_path,
            "tantivy_path": tantivy_path,
            "db_path": db_path,
        },
    )

    # --- Stores -----------------------------------------------------------
    from retriever_service.stores.chroma_store import ChromaStore
    from retriever_service.stores.tantivy_store import TantivyStore
    from retriever_service.stores.relational_store import RelationalStore
    from retriever_service.stores.case_metadata import CaseMetadataStore

    chroma = ChromaStore(chroma_path=chroma_path, cfg=cfg)
    tantivy = TantivyStore(base_path=tantivy_path, cfg=cfg)
    relational = RelationalStore(db_path=db_path)
    case_meta = CaseMetadataStore(tantivy_store=tantivy, chroma_store=chroma, cfg=cfg)

    # --- Embedder ---------------------------------------------------------
    from retriever_service.retrieval.dense_retriever import Embedder, DenseRetriever

    embedder = Embedder(model_name=cfg.chunking.tokenizer_model)
    dense = DenseRetriever(chroma_store=chroma, embedder=embedder)

    # --- Reranker ---------------------------------------------------------
    from retriever_service.retrieval.reranker import Reranker

    reranker = Reranker(cfg=cfg.reranker)

    # --- BM25 retriever ---------------------------------------------------
    from retriever_service.retrieval.bm25_retriever import BM25Retriever

    bm25 = BM25Retriever(tantivy_store=tantivy)

    # --- Pipeline ---------------------------------------------------------
    from retriever_service.retrieval.pipeline import RetrievalPipeline

    pipeline = RetrievalPipeline(
        bm25=bm25,
        dense=dense,
        reranker=reranker,
        relational=relational,
        rrf_cfg=cfg.rrf,
        reranker_cfg=cfg.reranker,
        timeouts_cfg=cfg.timeouts_ms,
        circuit_breaker_cfg=cfg.circuit_breaker,
    )

    # Attach everything to app.state so routes can access via request.app.state.
    app.state.chroma = chroma
    app.state.tantivy = tantivy
    app.state.relational = relational
    app.state.case_meta = case_meta
    app.state.embedder = embedder
    app.state.dense = dense
    app.state.reranker = reranker
    app.state.bm25 = bm25
    app.state.pipeline = pipeline
    app.state.cfg = cfg

    logger.info("All stores and models initialised — service ready")

    yield

    # --- Graceful shutdown ------------------------------------------------
    logger.info("Retriever service shutting down — flushing Tantivy writers")
    try:
        tantivy.commit_all()
    except Exception:
        logger.exception("Error committing Tantivy writers on shutdown")

    logger.info("Running SQLite WAL checkpoint")
    try:
        import sqlite3 as _sqlite3
        with _sqlite3.connect(db_path) as _conn:
            _conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    except Exception:
        logger.exception("Error checkpointing SQLite WAL on shutdown")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Legal Scribe — Retriever Service",
        description="Hybrid BM25 + dense retrieval microservice (spec §3).",
        version="0.2.0",
        lifespan=lifespan,
    )

    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)
    return app


app = create_app()
