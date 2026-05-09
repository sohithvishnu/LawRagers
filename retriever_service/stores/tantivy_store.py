"""Tantivy (sparse BM25) index wrapper.

Two named indexes mirror the ChromaDB collections (spec §4.2):
  - ny_case_law      — all case-law chunks (global, no session_id filter).
  - user_workspace   — all user-uploaded chunks, isolated by mandatory
                       session_id filter on every query.

Schema matches spec §4.2 exactly.  Field notes:
  - `chunk_id`, `session_id`, `case_name`, `citation_official`, `court_name`,
    `jurisdiction`, `section_type`, `opinion_type`, `opinion_author`, `source`
    — raw tokenizer (no stemming; exact-match / filter fields).
  - `text` — English stemming tokenizer (BM25 search body).
  - `decision_date` — stored as ISO-8601 text for portability; Tantivy's date
    field representation in the Python bindings is version-dependent.
  - `cites_to_case_ids` — JSON-encoded list of u64s; queried by the graph
    layer at ingest time, not directly searched by BM25.
  - `good_law` — stored as u64 (0/1); boolean field support varies across
    tantivy-py versions.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Optional

import tantivy

from retriever_service.config import RetrieverSettings, settings as default_settings

logger = logging.getLogger(__name__)

INDEX_CASE_LAW = "ny_case_law"
INDEX_USER_WORKSPACE = "user_workspace"
_VALID_INDEXES = frozenset([INDEX_CASE_LAW, INDEX_USER_WORKSPACE])

# Fields that use raw (non-stemmed) tokenizer — exact-match / filter fields.
_RAW_TEXT_FIELDS = {
    "chunk_id",
    "session_id",
    "case_name",
    "citation_official",
    "court_name",
    "jurisdiction",
    "section_type",
    "opinion_type",
    "opinion_author",
    "source",
}

# Numeric u64 fields (spec §4.2). Stored on disk as u64 — must match here or
# tantivy.Index.open() will reuse the on-disk schema and writer.commit() will
# fail with "Expected a U64 for field …".
_U64_FIELDS = ("case_id", "opinion_index", "chunk_idx", "good_law")
_QUERY_SPECIAL_CHARS_RE = re.compile(r"([+\-!<>(){}\[\]^\"'~*?:\\\\/])")


def _build_schema() -> tantivy.Schema:
    """Construct the full Tantivy schema per spec §4.2."""
    sb = tantivy.SchemaBuilder()

    # --- Text fields -------------------------------------------------------
    sb.add_text_field("text", stored=True, tokenizer_name="en_stem")
    for field in sorted(_RAW_TEXT_FIELDS):
        sb.add_text_field(field, stored=True, tokenizer_name="raw")

    # --- Unsigned integer fields (spec §4.2) -------------------------------
    for field in _U64_FIELDS:
        sb.add_unsigned_field(field, stored=True, indexed=True)

    # --- Float fields ------------------------------------------------------
    sb.add_float_field("pagerank_percentile", stored=True, indexed=True)
    sb.add_float_field("ocr_confidence", stored=True, indexed=True)

    # --- Date as ISO text --------------------------------------------------
    # Stored as text (YYYY-MM-DD) so range queries use lexicographic ordering
    # which is correct for zero-padded ISO dates.
    sb.add_text_field("decision_date", stored=True, tokenizer_name="raw")

    # --- Multi-value / serialised fields -----------------------------------
    # cites_to_case_ids: JSON-encoded list of ints, stored only (not indexed
    # for BM25; the graph layer reads this field directly from stored documents).
    sb.add_text_field("cites_to_case_ids", stored=True, tokenizer_name="raw")

    return sb.build()


def _escape_query_text(query_text: str) -> str:
    """Escape Tantivy query parser control characters in free-text queries."""
    flattened = " ".join(query_text.split())
    return _QUERY_SPECIAL_CHARS_RE.sub(r"\\\1", flattened)


class TantivyStore:
    """Sparse BM25 index wrapper managing ny_case_law and user_workspace indexes.

    Instantiate once at service startup and share the instance.
    """

    def __init__(
        self,
        base_path: str,
        cfg: RetrieverSettings = default_settings,
    ) -> None:
        self._cfg = cfg
        self._base_path = Path(base_path)
        self._schema = _build_schema()
        self._indexes: dict[str, tantivy.Index] = {}
        self._writers: dict[str, tantivy.IndexWriter] = {}
        self._init_indexes()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_indexes(self) -> None:
        for name in sorted(_VALID_INDEXES):
            idx_path = self._base_path / name
            idx_path.mkdir(parents=True, exist_ok=True)
            try:
                index = tantivy.Index.open(str(idx_path))
            except Exception:
                index = tantivy.Index(self._schema, path=str(idx_path))
            self._indexes[name] = index
        logger.info("Tantivy indexes ready", extra={"indexes": list(self._indexes.keys())})

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_index(index: str) -> None:
        if index not in _VALID_INDEXES:
            raise ValueError(
                f"Unknown index '{index}'. Valid options: {sorted(_VALID_INDEXES)}"
            )

    # ------------------------------------------------------------------
    # Writer helpers
    # ------------------------------------------------------------------

    def _get_writer(self, index: str) -> tantivy.IndexWriter:
        if index not in self._writers:
            self._writers[index] = self._indexes[index].writer()
        return self._writers[index]

    def _commit(self, index: str) -> None:
        if index in self._writers:
            self._writers[index].commit()
            # Reload the writer to avoid holding stale state.
            del self._writers[index]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, index: str, docs: list[dict[str, Any]]) -> None:
        """Index a batch of chunk documents.

        Each dict must contain all metadata fields.  Missing optional fields
        are skipped; required fields (chunk_id, text) raise if absent.
        """
        self._validate_index(index)
        if not docs:
            return

        writer = self._get_writer(index)
        for doc in docs:
            if "chunk_id" not in doc or "text" not in doc:
                raise ValueError("Each document must have 'chunk_id' and 'text' fields.")
            writer.add_document(self._build_tantivy_doc(doc))

        writer.commit()
        del self._writers[index]
        self._indexes[index].reload()
        logger.debug("Tantivy add", extra={"index": index, "count": len(docs)})

    def _build_tantivy_doc(self, doc: dict[str, Any]) -> tantivy.Document:
        """Convert a metadata dict to a tantivy.Document.

        Uses explicit typed add_* methods because tantivy.Document(**kwargs)
        adds every value as text regardless of Python type, which fails
        commit-time validation against u64/f64 fields.
        """
        td = tantivy.Document()

        # Text fields
        for field in ("text", *_RAW_TEXT_FIELDS, "decision_date"):
            val = doc.get(field)
            if val is not None:
                td.add_text(field, str(val))

        # u64 fields
        for field in ("case_id", "opinion_index", "chunk_idx"):
            val = doc.get(field)
            if val not in (None, ""):
                td.add_unsigned(field, int(val))

        good_law_raw = doc.get("good_law", True)
        td.add_unsigned("good_law", 1 if good_law_raw else 0)

        # Float fields
        for field in ("pagerank_percentile", "ocr_confidence"):
            val = doc.get(field)
            if val is not None:
                td.add_float(field, float(val))

        # Serialised multi-value field (stored as text)
        cites = doc.get("cites_to_case_ids", [])
        td.add_text(
            "cites_to_case_ids",
            json.dumps([int(c) for c in cites] if cites else []),
        )

        return td

    def search(
        self,
        index: str,
        query_text: str,
        k: int,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """BM25 search with optional pre-filter.

        Supported filter keys (all pushed down before ranking):
          min_ocr_confidence, min_pagerank_percentile,
          exclude_section_types, exclude_opinion_types,
          decision_date_gte, decision_date_lte,
          court_name (list[str]), good_law (bool),
          session_id (str — mandatory for user_workspace).

        Returns list of dicts: {chunk_id, text, metadata, score, rank}.
        """
        self._validate_index(index)
        self._indexes[index].reload()
        searcher = self._indexes[index].searcher()

        query = self._build_query(index, query_text, filters or {})
        results = searcher.search(query, limit=k)

        out: list[dict[str, Any]] = []
        for rank, (score, addr) in enumerate(results.hits, start=1):
            stored = searcher.doc(addr)
            out.append(
                {
                    "chunk_id": self._get_field(stored, "chunk_id"),
                    "text": self._get_field(stored, "text"),
                    "metadata": self._doc_to_metadata(stored),
                    "score": score,
                    "rank": rank,
                }
            )
        return out

    def get_by_case_id(
        self,
        index: str,
        case_id: int,
        k: int = 1,
    ) -> list[dict[str, Any]]:
        """Fetch stored documents by numeric case_id using a term query."""
        self._validate_index(index)
        self._indexes[index].reload()
        searcher = self._indexes[index].searcher()
        cid = int(case_id)
        query = self._indexes[index].parse_query(f"case_id:[{cid} TO {cid}]")
        results = searcher.search(query, limit=k)

        out: list[dict[str, Any]] = []
        for rank, (score, addr) in enumerate(results.hits, start=1):
            stored = searcher.doc(addr)
            out.append(
                {
                    "chunk_id": self._get_field(stored, "chunk_id"),
                    "text": self._get_field(stored, "text"),
                    "metadata": self._doc_to_metadata(stored),
                    "score": score,
                    "rank": rank,
                }
            )
        return out

    def count_by_case_id(self, index: str, case_id: int) -> int:
        """Count documents by numeric case_id without materializing stored fields."""
        self._validate_index(index)
        self._indexes[index].reload()
        searcher = self._indexes[index].searcher()
        cid = int(case_id)
        query = self._indexes[index].parse_query(f"case_id:[{cid} TO {cid}]")
        return int(searcher.search(query, limit=1).count)

    def _build_query(
        self,
        index: str,
        query_text: str,
        filters: dict[str, Any],
    ) -> Any:
        """Compose a Tantivy query from query text + filter primitives.

        Strategy: parse the user query against the `text` field, then wrap it
        in a BooleanQuery alongside MUST filter clauses for each active filter.
        """
        idx = self._indexes[index]
        text_query = idx.parse_query(
            _escape_query_text(query_text),
            default_field_names=["text"],
        )

        must_clauses: list[tuple[tantivy.Occur, Any]] = [(tantivy.Occur.Must, text_query)]

        # --- Numeric range filters ----------------------------------------
        if "min_ocr_confidence" in filters:
            val = float(filters["min_ocr_confidence"])
            must_clauses.append(
                (tantivy.Occur.Must, idx.parse_query(f"ocr_confidence:[{val} TO *]"))
            )

        if "min_pagerank_percentile" in filters:
            val = float(filters["min_pagerank_percentile"])
            must_clauses.append(
                (tantivy.Occur.Must, idx.parse_query(f"pagerank_percentile:[{val} TO *]"))
            )

        # --- Date range filters -------------------------------------------
        if "decision_date_gte" in filters:
            gte = filters["decision_date_gte"]
            must_clauses.append(
                (tantivy.Occur.Must, idx.parse_query(f"decision_date:[{gte} TO *]"))
            )
        if "decision_date_lte" in filters:
            lte = filters["decision_date_lte"]
            must_clauses.append(
                (tantivy.Occur.Must, idx.parse_query(f"decision_date:[* TO {lte}]"))
            )

        # --- Exact-match filters ------------------------------------------
        if "session_id" in filters:
            sid = filters["session_id"]
            must_clauses.append(
                (tantivy.Occur.Must, idx.parse_query(f'session_id:"{sid}"'))
            )

        # --- good_law filter -----------------------------------------------
        if "good_law" in filters:
            val = 1 if filters["good_law"] else 0
            must_clauses.append(
                (tantivy.Occur.Must, idx.parse_query(f"good_law:[{val} TO {val}]"))
            )

        # --- Exclusion filters (must_not) ----------------------------------
        for section_type in filters.get("exclude_section_types", []):
            must_clauses.append(
                (tantivy.Occur.MustNot, idx.parse_query(f'section_type:"{section_type}"'))
            )
        for opinion_type in filters.get("exclude_opinion_types", []):
            must_clauses.append(
                (tantivy.Occur.MustNot, idx.parse_query(f'opinion_type:"{opinion_type}"'))
            )

        # --- court_name inclusion filter ----------------------------------
        court_names: list[str] = filters.get("court_name", [])
        if court_names:
            # OR across names, then require the OR result (SHOULD → wrap in MUST)
            should_clauses = [
                (tantivy.Occur.Should, idx.parse_query(f'court_name:"{c}"'))
                for c in court_names
            ]
            must_clauses.append(
                (tantivy.Occur.Must, tantivy.Query.boolean_query(should_clauses))
            )

        if len(must_clauses) == 1:
            return text_query
        return tantivy.Query.boolean_query(must_clauses)

    def delete_by_filter(
        self,
        index: str,
        filters: dict[str, Any],
    ) -> int:
        """Delete documents matching all provided exact-match filters.

        Supported keys: session_id, source.
        Returns approximate count deleted (Tantivy does not give exact counts
        synchronously; we return the number of matching docs before deletion).
        """
        self._validate_index(index)
        self._indexes[index].reload()
        searcher = self._indexes[index].searcher()

        must_clauses: list[tuple[tantivy.Occur, Any]] = []
        idx = self._indexes[index]

        for key in ("session_id", "source"):
            if key in filters:
                must_clauses.append(
                    (tantivy.Occur.Must, idx.parse_query(f'{key}:"{filters[key]}"'))
                )

        if not must_clauses:
            raise ValueError("delete_by_filter requires at least one filter key.")

        query = (
            tantivy.Query.boolean_query(must_clauses)
            if len(must_clauses) > 1
            else must_clauses[0][1]
        )

        # Collect matching chunk_ids, then delete by term query on chunk_id.
        results = searcher.search(query, limit=100_000)
        chunk_ids = [
            self._get_field(searcher.doc(addr), "chunk_id")
            for _, addr in results.hits
        ]
        count = len(chunk_ids)

        if chunk_ids:
            writer = self._get_writer(index)
            for cid in chunk_ids:
                writer.delete_documents("chunk_id", cid)
            writer.commit()
            del self._writers[index]
            self._indexes[index].reload()

        logger.debug("Tantivy delete", extra={"index": index, "deleted": count})
        return count

    def commit_all(self) -> None:
        """Flush all pending writers (called at service shutdown)."""
        for name in list(self._writers.keys()):
            self._commit(name)

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def health(self) -> str:
        for name, idx in self._indexes.items():
            idx.searcher()  # lightweight check
        return "ok"

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_field(doc: tantivy.Document, field: str) -> Any:
        """Extract the first value of a field from a stored Document."""
        values = doc.get_all(field)
        return values[0] if values else None

    def _doc_to_metadata(self, doc: tantivy.Document) -> dict[str, Any]:
        """Convert a stored Tantivy document back to a metadata dict."""
        meta: dict[str, Any] = {}
        text_fields = list(_RAW_TEXT_FIELDS) + ["decision_date"]
        for field in text_fields:
            val = self._get_field(doc, field)
            if val is not None:
                meta[field] = val

        for field in ("case_id", "opinion_index", "chunk_idx"):
            val = self._get_field(doc, field)
            if val not in (None, ""):
                meta[field] = int(val)

        for field in ("pagerank_percentile", "ocr_confidence"):
            val = self._get_field(doc, field)
            if val is not None:
                meta[field] = float(val)

        good_law_raw = self._get_field(doc, "good_law")
        if good_law_raw is not None:
            meta["good_law"] = bool(int(good_law_raw))

        cites_raw = self._get_field(doc, "cites_to_case_ids")
        if cites_raw:
            try:
                meta["cites_to_case_ids"] = json.loads(cites_raw)
            except (json.JSONDecodeError, TypeError):
                meta["cites_to_case_ids"] = []

        return meta
