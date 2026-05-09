"""Citation graph edge retrieval (spec §8.5).

Serves GET /cases/{case_id}/edges.  Uses RelationalStore for edge lookup
and CaseMetadataStore for per-case enrichment (name, pagerank_percentile).

Edges within each direction are ordered by cited/citing case pagerank_percentile
descending — most authoritative neighbors first (spec §8.5).
"""

from __future__ import annotations

from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from retriever_service.stores.relational_store import RelationalStore
    from retriever_service.stores.case_metadata import CaseMetadataStore


def get_edges(
    case_id: int,
    relational: "RelationalStore",
    case_meta: "CaseMetadataStore",
    direction: str = "both",
    limit: int = 50,
) -> dict[str, Any]:
    """Return citation edges for a case with enriched neighbor metadata.

    Returns:
        {
          "case_id": int,
          "out": [{"case_id", "case_name", "pagerank_percentile"}, ...],
          "in":  [{"case_id", "case_name", "pagerank_percentile"}, ...],
        }
    """
    if direction not in ("out", "in", "both"):
        raise ValueError(f"Invalid direction '{direction}'.")

    raw = relational.get_edges(case_id, direction=direction, limit=limit)

    all_neighbor_ids = raw["out"] + raw["in"]
    meta_map = case_meta.get_many(all_neighbor_ids) if all_neighbor_ids else {}

    def _enrich(neighbor_ids: list[int]) -> list[dict[str, Any]]:
        enriched = []
        for nid in neighbor_ids:
            meta = meta_map.get(nid, {})
            enriched.append({
                "case_id": nid,
                "case_name": meta.get("case_name"),
                "pagerank_percentile": meta.get("pagerank_percentile"),
            })
        # Sort by pagerank_percentile descending (None sorts last)
        enriched.sort(
            key=lambda x: x.get("pagerank_percentile") or -1.0,
            reverse=True,
        )
        return enriched

    return {
        "case_id": case_id,
        "out": _enrich(raw["out"]),
        "in": _enrich(raw["in"]),
    }
