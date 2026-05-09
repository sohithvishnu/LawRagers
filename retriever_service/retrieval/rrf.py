"""Reciprocal Rank Fusion (spec §7.1).

Canonical Cormack/Clarke/Buettcher formula:
    score(d) = Σ_{r ∈ retrievers} 1 / (k + rank_r(d))

with k=60 (configurable).

Input: results from two or more retrievers, each as a list of dicts with
  at least {'chunk_id': str, 'text': str, 'metadata': dict}.

Output: a single fused list of dicts ordered by descending RRF score, each
  annotated with:
    - rrf_score
    - bm25_rank (or None)
    - dense_rank (or None)
    - text, metadata (from whichever retriever surfaced the chunk first)
"""

from __future__ import annotations

from typing import Any, Optional


def fuse(
    bm25_results: list[dict[str, Any]],
    dense_results: list[dict[str, Any]],
    k: int = 60,
    top_n: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Fuse two ranked lists via Reciprocal Rank Fusion.

    Args:
        bm25_results:  Ranked list from BM25 (position 0 = rank 1).
        dense_results: Ranked list from dense HNSW (position 0 = rank 1).
        k:             RRF constant (default 60 per spec).
        top_n:         If set, return only the top_n results.

    Returns:
        Fused list sorted by descending rrf_score with rank annotations.
    """
    scores: dict[str, float] = {}
    # Carry text+metadata from whichever result set saw the chunk first.
    chunk_data: dict[str, dict[str, Any]] = {}

    for rank, result in enumerate(bm25_results, start=1):
        cid = result["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
        if cid not in chunk_data:
            chunk_data[cid] = result

    for rank, result in enumerate(dense_results, start=1):
        cid = result["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
        if cid not in chunk_data:
            chunk_data[cid] = result

    # Build rank maps for annotation
    bm25_rank_map = {r["chunk_id"]: i + 1 for i, r in enumerate(bm25_results)}
    dense_rank_map = {r["chunk_id"]: i + 1 for i, r in enumerate(dense_results)}

    fused = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)
    if top_n is not None:
        fused = fused[:top_n]

    return [
        {
            **chunk_data[cid],
            "rrf_score": scores[cid],
            "bm25_rank": bm25_rank_map.get(cid),
            "dense_rank": dense_rank_map.get(cid),
        }
        for cid in fused
    ]
