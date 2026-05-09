"""Anchor-case detection (spec §8.7).

Serves GET /sessions/{session_id}/anchors.

Anchor score formula (spec §8.7):
    anchor_score = hits * (1 + pagerank_percentile)  when weight_by_pagerank=True
    anchor_score = hits                               otherwise
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from retriever_service.stores.relational_store import RelationalStore
    from retriever_service.stores.case_metadata import CaseMetadataStore

from retriever_service.config import AnchorsConfig, settings as default_settings


def get_anchors(
    session_id: str,
    relational: "RelationalStore",
    case_meta: "CaseMetadataStore",
    min_hits: int = 2,
    limit: int = 20,
    weight_by_pagerank: bool = True,
) -> dict[str, Any]:
    """Return anchor cases for a session.

    Returns:
        {
          "session_id": str,
          "anchors": [
            {
              "case_id", "case_name", "hits",
              "first_retrieved_at", "last_retrieved_at",
              "pagerank_percentile", "anchor_score"
            }, ...
          ],
        }
    """
    raw = relational.anchors(
        session_id=session_id,
        min_hits=min_hits,
        limit=limit,
        weight_by_pagerank=weight_by_pagerank,
    )

    if not raw:
        return {"session_id": session_id, "anchors": []}

    case_ids = [r["case_id"] for r in raw]
    meta_map = case_meta.get_many(case_ids)

    anchors = []
    for row in raw:
        cid = row["case_id"]
        meta = meta_map.get(cid, {})
        pagerank = meta.get("pagerank_percentile") or 0.0
        hits = row["hits"]
        anchor_score = hits * (1 + pagerank) if weight_by_pagerank else float(hits)
        anchors.append({
            "case_id": cid,
            "case_name": meta.get("case_name"),
            "hits": hits,
            "first_retrieved_at": row["first_retrieved_at"],
            "last_retrieved_at": row["last_retrieved_at"],
            "pagerank_percentile": pagerank,
            "anchor_score": round(anchor_score, 4),
        })

    # Re-sort by anchor_score descending (SQL ordered by raw hits; pagerank re-sorts)
    if weight_by_pagerank:
        anchors.sort(key=lambda a: a["anchor_score"], reverse=True)

    return {"session_id": session_id, "anchors": anchors}
