"""Async fire-and-forget retrieval logger (spec §4.3.2 + §4.5.3).

Every /retrieve call writes one row per returned chunk to retrieval_log.
This is done via asyncio.create_task so log failures never block the response.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from retriever_service.stores.relational_store import RelationalStore

logger = logging.getLogger(__name__)


def log_retrieval_async(
    relational: "RelationalStore",
    session_id: str,
    query: str,
    results: list[dict[str, Any]],
) -> None:
    """Schedule a fire-and-forget log write via asyncio.create_task.

    Failures are logged but never propagate to the caller (spec §4.5.3).
    """
    asyncio.create_task(
        _write_log(relational, session_id, query, results)
    )


async def _write_log(
    relational: "RelationalStore",
    session_id: str,
    query: str,
    results: list[dict[str, Any]],
) -> None:
    query_hash = relational.hash_query(query)
    entries: list[tuple[str, Optional[int], int, Optional[float]]] = []
    for result in results:
        chunk_id: str = result["chunk_id"]
        meta = result.get("metadata") or {}
        case_id_raw = meta.get("case_id")
        case_id: Optional[int] = int(case_id_raw) if case_id_raw else None
        rank: int = result.get("rerank_rank") or result.get("rank") or 0
        rerank_score: Optional[float] = (
            result.get("reranker_score")
            or result.get("rrf_score")
        )
        entries.append((chunk_id, case_id, rank, rerank_score))

    try:
        relational.log_retrieval(session_id, query_hash, entries)
    except Exception:
        logger.exception(
            "retrieval_logger: failed to write log (session=%s, n=%d)",
            session_id, len(entries),
        )
