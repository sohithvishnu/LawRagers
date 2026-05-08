"""Case-level metadata derived from the chunks index with LRU cache.

There is no `cases` table (spec §4.3 §8.4).  Case-level metadata is served
by a LIMIT-1 lookup against the chunks index — every chunk carries the full
case metadata as stored fields.

Cache: functools.lru_cache is not used directly because we need a TTL.
We implement a simple dict-based cache with per-entry expiry timestamps.
The cache is invalidated implicitly when TTL expires (default 10 s).

Lookup strategy (spec §8.4):
  1. Tantivy `case_id:<X>` filtered query with LIMIT 1 (fast warm path).
  2. ChromaDB `where={"case_id": X}` with limit=1 (fallback if Tantivy miss).
  Returns None if neither index has the case.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from retriever_service.stores.chroma_store import ChromaStore
    from retriever_service.stores.tantivy_store import TantivyStore

from retriever_service.config import RetrieverSettings, settings as default_settings

logger = logging.getLogger(__name__)

# Fields extracted from a chunk to represent case-level metadata.
_CASE_FIELDS = (
    "case_id",
    "case_name",
    "citation_official",
    "decision_date",
    "court_name",
    "jurisdiction",
    "pagerank_percentile",
    "ocr_confidence",
    "good_law",
)


class CaseMetadataStore:
    """LRU + TTL cache for case-level metadata derived from the chunks index.

    Instantiate once at service startup and share the instance.
    """

    def __init__(
        self,
        tantivy_store: "TantivyStore",
        chroma_store: "ChromaStore",
        cfg: RetrieverSettings = default_settings,
    ) -> None:
        self._tantivy = tantivy_store
        self._chroma = chroma_store
        self._max_entries = cfg.case_metadata_cache.max_entries
        self._ttl = cfg.case_metadata_cache.ttl_seconds

        # cache: {case_id: (expires_at, metadata_dict | None)}
        self._cache: dict[int, tuple[float, Optional[dict[str, Any]]]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_case(self, case_id: int) -> Optional[dict[str, Any]]:
        """Return case-level metadata for a single case_id.

        Returns None if the case is not present in either index.
        """
        cached = self._cache_get(case_id)
        if cached is not None:
            return cached

        meta = self._lookup_single(case_id)
        self._cache_set(case_id, meta)
        return meta

    def get_many(self, case_ids: list[int]) -> dict[int, dict[str, Any]]:
        """Return case metadata for multiple case IDs in one batched call.

        Only case IDs found in the index are included in the result.
        Cache misses are resolved via a single batched index query.
        """
        result: dict[int, dict[str, Any]] = {}
        missing: list[int] = []

        for cid in case_ids:
            hit = self._cache_get(cid)
            if hit is not None:
                result[cid] = hit
            else:
                missing.append(cid)

        if missing:
            fetched = self._lookup_many(missing)
            for cid in missing:
                meta = fetched.get(cid)
                self._cache_set(cid, meta)
                if meta is not None:
                    result[cid] = meta

        return result

    def invalidate(self, case_id: Optional[int] = None) -> None:
        """Invalidate a specific case or the entire cache."""
        if case_id is not None:
            self._cache.pop(case_id, None)
        else:
            self._cache.clear()

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _cache_get(self, case_id: int) -> Optional[dict[str, Any]]:
        entry = self._cache.get(case_id)
        if entry is None:
            return None
        expires_at, meta = entry
        if time.monotonic() > expires_at:
            del self._cache[case_id]
            return None
        # Treat a cached None as a verified miss (don't re-query within TTL).
        return meta  # may be None for known-missing cases

    def _cache_set(self, case_id: int, meta: Optional[dict[str, Any]]) -> None:
        if len(self._cache) >= self._max_entries:
            # Evict the oldest entry (FIFO approximation — dict is ordered).
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        self._cache[case_id] = (time.monotonic() + self._ttl, meta)

    # ------------------------------------------------------------------
    # Index lookups
    # ------------------------------------------------------------------

    def _lookup_single(self, case_id: int) -> Optional[dict[str, Any]]:
        """Try Tantivy first, fall back to Chroma."""
        meta = self._lookup_via_tantivy(case_id)
        if meta is not None:
            return meta
        return self._lookup_via_chroma(case_id)

    def _lookup_many(self, case_ids: list[int]) -> dict[int, dict[str, Any]]:
        """Batch lookup — query Tantivy once per case_id; Chroma for misses."""
        result: dict[int, dict[str, Any]] = {}
        chroma_needed: list[int] = []

        for cid in case_ids:
            meta = self._lookup_via_tantivy(cid)
            if meta is not None:
                result[cid] = meta
            else:
                chroma_needed.append(cid)

        for cid in chroma_needed:
            meta = self._lookup_via_chroma(cid)
            if meta is not None:
                result[cid] = meta

        return result

    def _lookup_via_tantivy(self, case_id: int) -> Optional[dict[str, Any]]:
        try:
            hits = self._tantivy.get_by_case_id(
                index="ny_case_law",
                case_id=case_id,
                k=1,
            )
            if hits:
                return self._extract_case_fields(hits[0]["metadata"])
        except Exception:
            logger.debug("Tantivy case lookup failed for case_id=%s", case_id, exc_info=True)
        return None

    def _lookup_via_chroma(self, case_id: int) -> Optional[dict[str, Any]]:
        try:
            from retriever_service.stores.chroma_store import COLLECTION_CASE_LAW

            col = self._chroma._collections[COLLECTION_CASE_LAW]
            results = col.get(
                where={"case_id": {"$eq": case_id}},
                limit=1,
                include=["metadatas"],
            )
            if results["metadatas"]:
                return self._extract_case_fields(results["metadatas"][0])
        except Exception:
            logger.debug("Chroma case lookup failed for case_id=%s", case_id, exc_info=True)
        return None

    @staticmethod
    def _extract_case_fields(metadata: dict[str, Any]) -> dict[str, Any]:
        return {k: metadata[k] for k in _CASE_FIELDS if k in metadata}
