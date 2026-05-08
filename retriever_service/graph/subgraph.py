"""Subgraph builder for visualization (spec §8.6).

Serves POST /graph/subgraph.

Two modes:
  - include_external_neighbors=False (default): internal edges only — edges
    among the seed set itself.
  - include_external_neighbors=True: add top-N most-cited neighbors of seeds
    (ranked by pagerank_percentile).

Depth >1 is supported but capped at 2 server-side (spec §8.6).
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from retriever_service.stores.relational_store import RelationalStore
    from retriever_service.stores.case_metadata import CaseMetadataStore

MAX_DEPTH = 2


def build_subgraph(
    seed_case_ids: list[int],
    relational: "RelationalStore",
    case_meta: "CaseMetadataStore",
    depth: int = 1,
    include_external_neighbors: bool = False,
    max_neighbors_per_seed: int = 20,
) -> dict[str, Any]:
    """Build a subgraph for a set of seed case IDs.

    Returns:
        {
          "nodes": [{"case_id", "case_name", "pagerank_percentile", "is_seed"}, ...],
          "edges": [{"from": int, "to": int}, ...],
        }
    """
    depth = min(depth, MAX_DEPTH)

    node_ids: set[int] = set(seed_case_ids)

    # Expand to depth-N neighbors
    frontier = set(seed_case_ids)
    for _ in range(depth):
        if not frontier:
            break
        neighbor_batch: set[int] = set()
        for case_id in frontier:
            raw = relational.get_edges(case_id, direction="both", limit=max_neighbors_per_seed)
            neighbor_batch.update(raw["out"])
            neighbor_batch.update(raw["in"])
        new_nodes = neighbor_batch - node_ids
        node_ids.update(new_nodes)
        frontier = new_nodes

    # Add external high-pagerank neighbors if requested
    if include_external_neighbors:
        external_ids = relational.get_top_cited_neighbors(
            seed_case_ids, max_per_seed=max_neighbors_per_seed
        )
        node_ids.update(external_ids)

    # Fetch edges among all nodes in the final set
    all_ids = list(node_ids)
    internal_edges = relational.get_subgraph_edges(all_ids)

    # Enrich node metadata
    meta_map = case_meta.get_many(all_ids)

    nodes = []
    for nid in all_ids:
        meta = meta_map.get(nid, {})
        nodes.append({
            "case_id": nid,
            "case_name": meta.get("case_name"),
            "pagerank_percentile": meta.get("pagerank_percentile"),
            "is_seed": nid in seed_case_ids,
        })

    edges = [{"from": src, "to": dst} for src, dst in internal_edges]

    return {"nodes": nodes, "edges": edges}
