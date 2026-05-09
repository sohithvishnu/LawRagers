"""ChromaDB wrapper for dense (HNSW) vector storage.

Two collections are managed here (spec §4.1):
  - ny_case_law   — all binding-precedent chunks across the corpus (global).
  - user_workspace — all user-uploaded chunks, isolated logically by session_id.

HNSW parameters are set explicitly on collection creation; ChromaDB defaults
to L2 distance which would silently break cosine-similarity search (Bug B-3
in the gap map).

Carry-over from Phase 0: add() validates that every metadata dict contains
a `case_id` key.  User-upload chunks use case_id="" to satisfy this contract
while signalling no associated case.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from retriever_service.config import RetrieverSettings, settings as default_settings

logger = logging.getLogger(__name__)

# Canonical collection names — single source of truth.
COLLECTION_CASE_LAW = "ny_case_law"
COLLECTION_USER_WORKSPACE = "user_workspace"

_VALID_COLLECTIONS = frozenset([COLLECTION_CASE_LAW, COLLECTION_USER_WORKSPACE])


class ChromaStore:
    """Thin wrapper around ChromaDB providing the two shared collections.

    Instantiate once at service startup and share the instance.
    """

    def __init__(
        self,
        chroma_path: str,
        cfg: RetrieverSettings = default_settings,
    ) -> None:
        self._cfg = cfg
        self._client = chromadb.PersistentClient(
            path=chroma_path,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collections: dict[str, Any] = {}
        self._init_collections()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _hnsw_metadata(self) -> dict[str, Any]:
        """Build ChromaDB collection metadata for HNSW configuration."""
        h = self._cfg.hnsw
        return {
            "hnsw:space": h.space,
            "hnsw:M": h.M,
            "hnsw:construction_ef": h.construction_ef,
            "hnsw:search_ef": h.search_ef,
        }

    def _init_collections(self) -> None:
        meta = self._hnsw_metadata()
        for name in _VALID_COLLECTIONS:
            self._collections[name] = self._client.get_or_create_collection(
                name=name,
                metadata=meta,
            )
        logger.info(
            "ChromaDB collections ready",
            extra={"collections": list(self._collections.keys()), "hnsw": meta},
        )

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_collection(collection: str) -> None:
        if collection not in _VALID_COLLECTIONS:
            raise ValueError(
                f"Unknown collection '{collection}'. "
                f"Valid options: {sorted(_VALID_COLLECTIONS)}"
            )

    @staticmethod
    def _sanitize_metadata(meta: dict[str, Any]) -> dict[str, Any]:
        """Coerce metadata into ChromaDB-compatible scalars.

        ChromaDB rejects empty lists and non-primitive list elements. Lists
        are JSON-serialised to text (mirroring Tantivy's cites_to_case_ids
        storage); None values are dropped so they don't pollute filters.
        """
        out: dict[str, Any] = {}
        for k, v in meta.items():
            if v is None:
                continue
            if isinstance(v, list):
                out[k] = json.dumps(v)
            else:
                out[k] = v
        return out

    @staticmethod
    def _validate_case_ids(metadatas: list[dict[str, Any]]) -> None:
        """Reject batches where any metadata dict is missing 'case_id'.

        Phase 0 carry-over: case_id is load-bearing in the eval pipeline and
        must be present in every chunk record.  User-upload chunks must pass
        an empty string explicitly rather than omitting the field.
        """
        missing = [i for i, m in enumerate(metadatas) if "case_id" not in m]
        if missing:
            raise ValueError(
                f"Metadata at indices {missing} is missing required 'case_id' field. "
                "User-upload chunks must set case_id='' explicitly."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(
        self,
        collection: str,
        ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Upsert chunks into the named collection.

        Raises ValueError if any metadata entry lacks 'case_id'.
        """
        self._validate_collection(collection)
        if not ids:
            return
        self._validate_case_ids(metadatas)

        sanitized = [self._sanitize_metadata(m) for m in metadatas]

        # ChromaDB enforces a server-side max batch size (5461 in current
        # versions). Split large upserts to stay safely under it.
        batch_size = 5000
        col = self._collections[collection]
        for start in range(0, len(ids), batch_size):
            end = start + batch_size
            col.upsert(
                ids=ids[start:end],
                documents=texts[start:end],
                embeddings=embeddings[start:end],
                metadatas=sanitized[start:end],
            )
        logger.debug("ChromaDB upsert", extra={"collection": collection, "count": len(ids)})

    def query(
        self,
        collection: str,
        embedding: list[float],
        k: int,
        where: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """Return up to k nearest neighbours in the collection.

        Returns a list of dicts with keys: chunk_id, text, metadata, distance.
        """
        self._validate_collection(collection)
        kwargs: dict[str, Any] = {
            "query_embeddings": [embedding],
            "n_results": k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = self._collections[collection].query(**kwargs)
        out: list[dict[str, Any]] = []
        if not results["ids"] or not results["ids"][0]:
            return out
        for chunk_id, doc, meta, dist in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            out.append(
                {
                    "chunk_id": chunk_id,
                    "text": doc,
                    "metadata": meta,
                    "distance": dist,
                }
            )
        return out

    def delete_by_filter(
        self,
        collection: str,
        where: dict[str, Any],
    ) -> int:
        """Delete all documents in the collection matching the filter.

        Returns the number of documents deleted.
        """
        self._validate_collection(collection)
        col = self._collections[collection]
        # ChromaDB does not return a count from delete; we fetch IDs first.
        existing = col.get(where=where, include=[])
        ids_to_delete = existing["ids"]
        if ids_to_delete:
            col.delete(ids=ids_to_delete)
        logger.debug(
            "ChromaDB delete",
            extra={"collection": collection, "deleted": len(ids_to_delete)},
        )
        return len(ids_to_delete)

    def count(self, collection: str) -> int:
        self._validate_collection(collection)
        return self._collections[collection].count()

    def health(self) -> str:
        """Return 'ok' if all collections are reachable, else raise."""
        for name, col in self._collections.items():
            col.count()  # lightweight round-trip
        return "ok"
