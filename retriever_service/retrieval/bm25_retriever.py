"""BM25 retriever wrapping TantivyStore (spec §7).

Single chokepoint for all sparse retrieval.  Enforces mandatory session_id
filter on user_workspace queries (spec §4.5.2).
"""

from __future__ import annotations

import logging
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from retriever_service.stores.tantivy_store import TantivyStore

from retriever_service.stores.tantivy_store import INDEX_CASE_LAW, INDEX_USER_WORKSPACE

logger = logging.getLogger(__name__)


class BM25Retriever:
    """Sparse BM25 retriever backed by TantivyStore."""

    def __init__(self, tantivy_store: "TantivyStore") -> None:
        self._tantivy = tantivy_store

    def query_case_law(
        self,
        query_text: str,
        k: int,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """Query the global ny_case_law index.

        Returns list of {chunk_id, text, metadata, score, rank}.
        """
        return self._tantivy.search(
            index=INDEX_CASE_LAW,
            query_text=query_text,
            k=k,
            filters=filters or {},
        )

    def query_user_workspace(
        self,
        session_id: str,
        query_text: str,
        k: int,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """Query user_workspace, enforcing mandatory session_id isolation.

        Raises ValueError if session_id is empty (spec §4.5.2 rule 3).
        """
        if not session_id:
            raise ValueError(
                "session_id is required and must be non-empty for user_workspace queries."
            )
        combined = dict(filters or {})
        combined["session_id"] = session_id
        return self._tantivy.search(
            index=INDEX_USER_WORKSPACE,
            query_text=query_text,
            k=k,
            filters=combined,
        )
