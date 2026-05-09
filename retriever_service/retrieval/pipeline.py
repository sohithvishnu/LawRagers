"""Retrieval pipeline orchestrator (spec §7, §9).

Wires together: normalize → parallel BM25+dense → RRF → reranker → log.

    query
      → normalize()
      → parallel:
           ├── Tantivy BM25 top-100 (filtered)   [timeout §9, circuit-breaker §9]
           └── ChromaDB HNSW top-100 (filtered)  [timeout §9, circuit-breaker §9]
      → RRF fuse (k=60) → top-50
      → cross-encoder rerank → top-k (default 10) [timeout §9]
      → fire-and-forget log
      → return

Degradation (spec §8.1):
  - Circuit open / timeout / exception on one retriever → use the other,
    set degraded=True.
  - Both retrievers fail → raise RetrievalError (→ 503).
  - Reranker fails → return RRF-ordered results, degraded=True.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from retriever_service.retrieval.bm25_retriever import BM25Retriever
    from retriever_service.retrieval.dense_retriever import DenseRetriever
    from retriever_service.retrieval.reranker import Reranker
    from retriever_service.stores.relational_store import RelationalStore

from retriever_service.normalize import normalize_query
from retriever_service.retrieval import rrf as rrf_module
from retriever_service.retrieval.retrieval_logger import log_retrieval_async
from retriever_service.retrieval.circuit_breaker import CircuitBreaker
from retriever_service.ingestion.dual_writer import _strip_prefix
from retriever_service.config import (
    CircuitBreakerConfig,
    RRFConfig,
    RerankerConfig,
    TimeoutsConfig,
    settings as default_settings,
)

logger = logging.getLogger(__name__)


class RetrievalError(Exception):
    """Raised when all retrievers are unavailable (→ 503)."""


def _ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


class RetrievalPipeline:
    """Full hybrid retrieval pipeline with per-call timeouts and circuit breakers."""

    def __init__(
        self,
        bm25: "BM25Retriever",
        dense: "DenseRetriever",
        reranker: "Reranker",
        relational: "RelationalStore",
        rrf_cfg: Optional[RRFConfig] = None,
        reranker_cfg: Optional[RerankerConfig] = None,
        timeouts_cfg: Optional[TimeoutsConfig] = None,
        circuit_breaker_cfg: Optional[CircuitBreakerConfig] = None,
    ) -> None:
        self._bm25 = bm25
        self._dense = dense
        self._reranker = reranker
        self._relational = relational
        self._rrf_cfg = rrf_cfg or default_settings.rrf
        self._reranker_cfg = reranker_cfg or default_settings.reranker
        self._timeouts = timeouts_cfg or default_settings.timeouts_ms
        cb_cfg = circuit_breaker_cfg or default_settings.circuit_breaker

        self._cb_bm25 = CircuitBreaker(
            "bm25",
            consecutive_failures=cb_cfg.consecutive_failures,
            cooldown_seconds=cb_cfg.cooldown_seconds,
        )
        self._cb_dense = CircuitBreaker(
            "dense",
            consecutive_failures=cb_cfg.consecutive_failures,
            cooldown_seconds=cb_cfg.cooldown_seconds,
        )
        self._cb_reranker = CircuitBreaker(
            "reranker",
            consecutive_failures=cb_cfg.consecutive_failures,
            cooldown_seconds=cb_cfg.cooldown_seconds,
        )

    async def retrieve(
        self,
        query: str,
        session_id: Optional[str],
        k: int,
        corpora: list[str],
        filters: dict[str, Any],
        rerank: bool = True,
        return_debug: bool = False,
        retrievers: list[str] | None = None,
    ) -> dict[str, Any]:
        """Execute the full retrieval pipeline.

        Returns a dict matching the /retrieve 200 response schema (spec §8.1).
        """
        t_total = time.perf_counter()
        latency: dict[str, int] = {}
        degraded = False
        retrievers_used: list[str] = []

        normalized_query = normalize_query(query)
        candidate_pool = self._rrf_cfg.candidate_pool

        # Default OCR filter always applied (spec §7.1.1)
        effective_filters = dict(filters)
        if "min_ocr_confidence" not in effective_filters:
            effective_filters["min_ocr_confidence"] = (
                default_settings.filters.default_min_ocr_confidence
            )

        active = set(retrievers) if retrievers is not None else {"bm25", "dense"}
        loop = asyncio.get_running_loop()

        async def _leg_disabled() -> tuple[list, bool, int, bool]:
            return [], False, 0, False

        bm25_task = asyncio.create_task(
            self._run_bm25_guarded(
                loop=loop,
                normalized_query=normalized_query,
                raw_query=query,
                session_id=session_id,
                corpora=corpora,
                candidate_pool=candidate_pool,
                filters=effective_filters,
            ) if "bm25" in active else _leg_disabled()
        )
        dense_task = asyncio.create_task(
            self._run_dense_guarded(
                loop=loop,
                normalized_query=normalized_query,
                raw_query=query,
                session_id=session_id,
                corpora=corpora,
                candidate_pool=candidate_pool,
                filters=effective_filters,
            ) if "dense" in active else _leg_disabled()
        )

        bm25_outcome, dense_outcome = await asyncio.gather(bm25_task, dense_task)
        bm25_results, bm25_used, bm25_latency, bm25_degraded = bm25_outcome
        dense_results, dense_used, dense_latency, dense_degraded = dense_outcome
        latency["bm25"] = bm25_latency
        latency["dense"] = dense_latency
        degraded = degraded or bm25_degraded or dense_degraded
        if bm25_used:
            retrievers_used.append("bm25")
        if dense_used:
            retrievers_used.append("dense")

        if not bm25_results and not dense_results and (bm25_degraded or dense_degraded):
            raise RetrievalError("All retrievers unavailable.")

        # --- RRF ---
        t0 = time.perf_counter()
        fused = rrf_module.fuse(
            bm25_results=bm25_results,
            dense_results=dense_results,
            k=self._rrf_cfg.k,
            top_n=self._reranker_cfg.top_n_input,
        )
        latency["rrf"] = _ms(t0)

        # --- Reranker ---
        if rerank and self._reranker_cfg.enabled_default and fused:
            t0 = time.perf_counter()
            if self._cb_reranker.is_open():
                logger.warning("Reranker circuit open; skipping rerank, using RRF order.")
                fused = fused[:k]
                degraded = True
            else:
                try:
                    fused = await asyncio.wait_for(
                        loop.run_in_executor(
                            None,
                            lambda: self._reranker.rerank(
                                query=normalized_query,
                                candidates=fused,
                                top_n=k,
                            ),
                        ),
                        timeout=self._timeouts.rerank / 1000.0,
                    )
                    self._cb_reranker.record_success()
                    retrievers_used.append("rerank")
                except asyncio.TimeoutError:
                    logger.warning(
                        "Reranker timed out after %dms; falling back to RRF order.",
                        self._timeouts.rerank,
                    )
                    self._cb_reranker.record_failure()
                    fused = fused[:k]
                    degraded = True
                except Exception:
                    logger.exception("Reranker failed; falling back to RRF order.")
                    self._cb_reranker.record_failure()
                    fused = fused[:k]
                    degraded = True
            latency["rerank"] = _ms(t0)
        else:
            fused = fused[:k]

        latency["total"] = _ms(t_total)

        # --- Build response ---
        results = [_format_result(r, return_debug) for r in fused]

        # --- Fire-and-forget log ---
        if session_id:
            log_retrieval_async(self._relational, session_id, query, fused)

        response: dict[str, Any] = {
            "results": results,
            "degraded": degraded,
            "retrievers_used": retrievers_used,
        }
        if return_debug:
            response["latency_ms"] = latency

        return response

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_bm25(
        self,
        query: str,
        session_id: Optional[str],
        corpora: list[str],
        k: int,
        filters: dict[str, Any],
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        if "ny_case_law" in corpora:
            results.extend(self._bm25.query_case_law(query, k, filters))
        if "user_workspace" in corpora and session_id:
            results.extend(self._bm25.query_user_workspace(session_id, query, k, filters))
        return results

    async def _run_bm25_guarded(
        self,
        *,
        loop: asyncio.AbstractEventLoop,
        normalized_query: str,
        raw_query: str,
        session_id: Optional[str],
        corpora: list[str],
        candidate_pool: int,
        filters: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], bool, int, bool]:
        t0 = time.perf_counter()
        degraded = False
        if self._cb_bm25.is_open():
            logger.warning("BM25 circuit breaker open; skipping.")
            return [], False, _ms(t0), True

        try:
            results = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: self._run_bm25(
                        normalized_query,
                        session_id,
                        corpora,
                        candidate_pool,
                        filters,
                    ),
                ),
                timeout=self._timeouts.bm25 / 1000.0,
            )
            self._cb_bm25.record_success()
            return results, True, _ms(t0), degraded
        except asyncio.TimeoutError:
            logger.warning(
                "BM25 retriever timed out after %dms for query=%s",
                self._timeouts.bm25,
                raw_query[:80],
            )
            self._cb_bm25.record_failure()
            degraded = True
        except Exception:
            logger.exception("BM25 retriever failed for query=%s", raw_query[:80])
            self._cb_bm25.record_failure()
            degraded = True
        return [], False, _ms(t0), degraded

    def _run_dense(
        self,
        query: str,
        session_id: Optional[str],
        corpora: list[str],
        k: int,
        filters: dict[str, Any],
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        if "ny_case_law" in corpora:
            results.extend(self._dense.query_case_law(query, k, filters))
        if "user_workspace" in corpora and session_id:
            results.extend(self._dense.query_user_workspace(session_id, query, k, filters))
        return results

    async def _run_dense_guarded(
        self,
        *,
        loop: asyncio.AbstractEventLoop,
        normalized_query: str,
        raw_query: str,
        session_id: Optional[str],
        corpora: list[str],
        candidate_pool: int,
        filters: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], bool, int, bool]:
        t0 = time.perf_counter()
        degraded = False
        if self._cb_dense.is_open():
            logger.warning("Dense circuit breaker open; skipping.")
            return [], False, _ms(t0), True

        try:
            results = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: self._run_dense(
                        normalized_query,
                        session_id,
                        corpora,
                        candidate_pool,
                        filters,
                    ),
                ),
                timeout=self._timeouts.dense / 1000.0,
            )
            self._cb_dense.record_success()
            return results, True, _ms(t0), degraded
        except asyncio.TimeoutError:
            logger.warning(
                "Dense retriever timed out after %dms for query=%s",
                self._timeouts.dense,
                raw_query[:80],
            )
            self._cb_dense.record_failure()
            degraded = True
        except Exception:
            logger.exception("Dense retriever failed for query=%s", raw_query[:80])
            self._cb_dense.record_failure()
            degraded = True
        return [], False, _ms(t0), degraded


def _format_result(chunk: dict[str, Any], return_debug: bool) -> dict[str, Any]:
    """Map an internal chunk dict to the /retrieve response schema (spec §8.1)."""
    meta = chunk.get("metadata") or {}
    text = _strip_prefix(chunk.get("text", ""))

    raw_case_id = meta.get("case_id")
    case_id_int: int | None = None
    if raw_case_id is not None and str(raw_case_id).strip().isdigit():
        case_id_int = int(raw_case_id)

    result: dict[str, Any] = {
        "chunk_id": chunk["chunk_id"],
        "case_id": case_id_int,
        "text": text,
        "score": chunk.get("reranker_score") or chunk.get("rrf_score") or 0.0,
        "source": {
            "case_name": meta.get("case_name"),
            "citation_official": meta.get("citation_official"),
            "decision_date": meta.get("decision_date"),
            "court_name": meta.get("court_name"),
            "corpus": meta.get("corpus", "ny_case_law"),
            "section_type": meta.get("section_type"),
            "opinion_type": meta.get("opinion_type"),
            "opinion_author": meta.get("opinion_author"),
            "pagerank_percentile": meta.get("pagerank_percentile"),
            "ocr_confidence": meta.get("ocr_confidence"),
        },
    }

    if return_debug:
        result["ranks"] = {
            "bm25": chunk.get("bm25_rank"),
            "dense": chunk.get("dense_rank"),
            "rrf": None,
            "rerank": chunk.get("rerank_rank"),
        }
        result["score_components"] = chunk.get("score_components")

    return result
