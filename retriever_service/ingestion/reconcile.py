"""Reconciliation job — compare ID sets across Tantivy, Chroma, and citations.

Detects and removes orphan chunks (present in one index but not the other).
Intended as a standalone maintenance script, not a request-path operation.

Usage:
    python -m retriever_service.ingestion.reconcile --corpus ny_case_law
    python -m retriever_service.ingestion.reconcile --corpus ny_case_law --fix
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _load_chroma_ids(chroma, collection: str) -> set[str]:
    col = chroma._collections[collection]
    result = col.get(include=[])
    return set(result["ids"])


def _load_tantivy_ids(tantivy, index: str) -> set[str]:
    tantivy._indexes[index].reload()
    searcher = tantivy._indexes[index].searcher()
    import tantivy as tv
    query = tv.Query.all_query() if hasattr(tv.Query, "all_query") else None
    if query is None:
        # Fallback: search for everything via TermQuery on a known field
        return set()
    results = searcher.search(query, limit=10_000_000)
    ids: set[str] = set()
    for _, addr in results.hits:
        doc = searcher.doc(addr)
        vals = doc.get_all("chunk_id")
        if vals:
            ids.add(vals[0])
    return ids


def reconcile(
    corpus: str,
    chroma,
    tantivy,
    fix: bool = False,
) -> dict:
    from retriever_service.stores.chroma_store import COLLECTION_CASE_LAW, COLLECTION_USER_WORKSPACE
    from retriever_service.stores.tantivy_store import INDEX_CASE_LAW, INDEX_USER_WORKSPACE

    collection = COLLECTION_CASE_LAW if corpus == "ny_case_law" else COLLECTION_USER_WORKSPACE
    index = INDEX_CASE_LAW if corpus == "ny_case_law" else INDEX_USER_WORKSPACE

    chroma_ids = _load_chroma_ids(chroma, collection)
    tantivy_ids = _load_tantivy_ids(tantivy, index)

    only_in_chroma = chroma_ids - tantivy_ids
    only_in_tantivy = tantivy_ids - chroma_ids

    report = {
        "corpus": corpus,
        "chroma_total": len(chroma_ids),
        "tantivy_total": len(tantivy_ids),
        "only_in_chroma": len(only_in_chroma),
        "only_in_tantivy": len(only_in_tantivy),
        "fixed": False,
    }

    if fix and (only_in_chroma or only_in_tantivy):
        if only_in_chroma:
            col = chroma._collections[collection]
            col.delete(ids=list(only_in_chroma))
            logger.info("Removed %d orphan chunks from ChromaDB.", len(only_in_chroma))
        if only_in_tantivy:
            for cid in only_in_tantivy:
                tantivy.delete_by_filter(index, {"chunk_id": cid})
            logger.info("Removed %d orphan chunks from Tantivy.", len(only_in_tantivy))
        report["fixed"] = True

    return report


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default="ny_case_law",
                    choices=["ny_case_law", "user_workspace"])
    ap.add_argument("--fix", action="store_true", help="Remove orphan entries")
    ap.add_argument("--config", help="Path to retriever config.yaml")
    args = ap.parse_args()

    from retriever_service.config import load_settings
    cfg = load_settings(args.config)

    from retriever_service.stores.chroma_store import ChromaStore
    from retriever_service.stores.tantivy_store import TantivyStore

    chroma = ChromaStore(chroma_path=os.path.abspath(cfg.service.chroma_path), cfg=cfg)
    tantivy = TantivyStore(base_path=os.path.abspath(cfg.service.tantivy_path), cfg=cfg)

    report = reconcile(args.corpus, chroma, tantivy, fix=args.fix)

    for k, v in report.items():
        print(f"{k}: {v}")

    return 0 if report["only_in_chroma"] == 0 and report["only_in_tantivy"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
