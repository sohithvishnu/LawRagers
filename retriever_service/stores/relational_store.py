"""SQLite wrapper for relational tables in the shared legal_sessions.db.

Tables owned here (spec §4.3):
  - citations       — case-citation graph edges (spec §4.3.1)
  - retrieval_log   — per-query audit log + anchor-detection foundation (§4.3.2)
  - ingestion_log   — ingestion status for reconciliation (spec §5)

Connection model (spec §4.3.3):
  - `legal_sessions.db` is shared with api.py; the retriever opens its own
    connections to the same file.
  - WAL mode is set once on first connection; SQLite persists this setting.
  - Each public method opens (or borrows from a small thread-local pool) a
    connection and releases it on completion — never a module-level cursor.

DDL is co-located here as the canonical source.  api.py calls
`ensure_tables(db_path)` at startup so the tables are created before any
request arrives.
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
from contextlib import contextmanager
from typing import Generator, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

_DDL_CITATIONS = """
CREATE TABLE IF NOT EXISTS citations (
    citing_case_id INTEGER NOT NULL,
    cited_case_id  INTEGER NOT NULL,
    PRIMARY KEY (citing_case_id, cited_case_id)
) WITHOUT ROWID;

CREATE INDEX IF NOT EXISTS idx_citations_cited
    ON citations(cited_case_id);
"""

_DDL_RETRIEVAL_LOG = """
CREATE TABLE IF NOT EXISTS retrieval_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   TEXT NOT NULL,
    query_hash   TEXT NOT NULL,
    chunk_id     TEXT NOT NULL,
    case_id      INTEGER,
    rank         INTEGER NOT NULL,
    rerank_score REAL,
    retrieved_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_retrieval_session_case
    ON retrieval_log(session_id, case_id);

CREATE INDEX IF NOT EXISTS idx_retrieval_session_time
    ON retrieval_log(session_id, retrieved_at);
"""

_DDL_INGESTION_LOG = """
CREATE TABLE IF NOT EXISTS ingestion_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    corpus       TEXT NOT NULL,
    kind         TEXT NOT NULL,
    target_id    TEXT NOT NULL,
    status       TEXT NOT NULL,
    started_at   TEXT NOT NULL DEFAULT (datetime('now')),
    finished_at  TEXT,
    error        TEXT
);

CREATE INDEX IF NOT EXISTS idx_ingestion_target ON ingestion_log(corpus, target_id);
"""

_DDL_DOCUMENTS = """
CREATE TABLE IF NOT EXISTS documents (
    session_id   TEXT NOT NULL,
    source       TEXT NOT NULL,
    chunk_count  INTEGER NOT NULL,
    size_bytes   INTEGER NOT NULL,
    ingested_at  TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (session_id, source)
);

CREATE INDEX IF NOT EXISTS idx_documents_session
    ON documents(session_id, ingested_at);
"""

_DDL_JOBS = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id        TEXT PRIMARY KEY,
    status        TEXT NOT NULL,
    created_at    TEXT NOT NULL DEFAULT (datetime('now')),
    completed_at  TEXT,
    error         TEXT
);
"""


def ensure_tables(db_path: str) -> None:
    """Create all retriever-owned tables in the shared SQLite database.

    Idempotent — safe to call on every startup.
    Called by api.py at startup (spec §4.3.3).
    """
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        for ddl in (
            _DDL_CITATIONS,
            _DDL_RETRIEVAL_LOG,
            _DDL_INGESTION_LOG,
            _DDL_DOCUMENTS,
            _DDL_JOBS,
        ):
            conn.executescript(ddl)
    logger.info("Relational tables verified", extra={"db_path": db_path})


# ---------------------------------------------------------------------------
# Store class
# ---------------------------------------------------------------------------

class RelationalStore:
    """SQLite wrapper for citations, retrieval_log, and ingestion_log.

    Each method opens a fresh connection, performs its work in a transaction,
    and closes the connection.  This satisfies the spec requirement of no
    shared module-level cursor (Bug B-2 anti-pattern) while keeping connections
    cheap via SQLite's local file access.
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        # Verify tables exist (will raise if db is inaccessible).
        ensure_tables(db_path)

    # ------------------------------------------------------------------
    # Connection helper
    # ------------------------------------------------------------------

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Citations
    # ------------------------------------------------------------------

    def add_citation_edges(
        self,
        citing_case_id: int,
        cited_case_ids: list[int],
    ) -> int:
        """Insert citation edges for a case.  Duplicate edges are silently ignored.

        Returns the number of new rows inserted.
        """
        if not cited_case_ids:
            return 0
        rows = [(citing_case_id, cid) for cid in cited_case_ids]
        with self._conn() as conn:
            conn.executemany(
                "INSERT OR IGNORE INTO citations (citing_case_id, cited_case_id) VALUES (?, ?)",
                rows,
            )
            inserted: int = conn.execute("SELECT changes()").fetchone()[0]
        return inserted

    def get_edges(
        self,
        case_id: int,
        direction: str = "both",
        limit: int = 50,
    ) -> dict[str, list[int]]:
        """Return citation edges for a case.

        direction: 'out' | 'in' | 'both'
        Returns {'out': [...], 'in': [...]}.  Empty lists when no edges exist.
        """
        if direction not in ("out", "in", "both"):
            raise ValueError(f"Invalid direction '{direction}'.")

        result: dict[str, list[int]] = {"out": [], "in": []}
        with self._conn() as conn:
            if direction in ("out", "both"):
                rows = conn.execute(
                    "SELECT cited_case_id FROM citations WHERE citing_case_id = ? LIMIT ?",
                    (case_id, limit),
                ).fetchall()
                result["out"] = [r[0] for r in rows]

            if direction in ("in", "both"):
                rows = conn.execute(
                    "SELECT citing_case_id FROM citations WHERE cited_case_id = ? LIMIT ?",
                    (case_id, limit),
                ).fetchall()
                result["in"] = [r[0] for r in rows]

        return result

    def get_subgraph_edges(
        self,
        case_ids: list[int],
    ) -> list[tuple[int, int]]:
        """Return all citation edges among a set of case IDs.

        Used by the subgraph endpoint for internal-edges-only mode.
        """
        if not case_ids:
            return []
        placeholders = ",".join("?" * len(case_ids))
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT citing_case_id, cited_case_id FROM citations "
                f"WHERE citing_case_id IN ({placeholders}) "
                f"  AND cited_case_id IN ({placeholders})",
                case_ids + case_ids,
            ).fetchall()
        return [(r[0], r[1]) for r in rows]

    def get_top_cited_neighbors(
        self,
        seed_case_ids: list[int],
        max_per_seed: int = 20,
    ) -> list[int]:
        """Return the most-cited outgoing neighbors of seed cases (union).

        Used by subgraph when include_external_neighbors=true.
        """
        if not seed_case_ids:
            return []
        placeholders = ",".join("?" * len(seed_case_ids))
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT cited_case_id, COUNT(*) AS c FROM citations "
                f"WHERE citing_case_id IN ({placeholders}) "
                f"GROUP BY cited_case_id ORDER BY c DESC LIMIT ?",
                seed_case_ids + [max_per_seed],
            ).fetchall()
        return [r[0] for r in rows]

    # ------------------------------------------------------------------
    # Retrieval log
    # ------------------------------------------------------------------

    @staticmethod
    def hash_query(query: str) -> str:
        return hashlib.md5(query.encode("utf-8")).hexdigest()

    def log_retrieval(
        self,
        session_id: str,
        query_hash: str,
        entries: list[tuple[str, Optional[int], int, Optional[float]]],
    ) -> None:
        """Write retrieval log rows.

        entries: list of (chunk_id, case_id, rank, rerank_score).
        Written synchronously per spec §4.3.2 — the cost is negligible vs
        the reranker pass.
        """
        rows = [
            (session_id, query_hash, chunk_id, case_id, rank, rerank_score)
            for chunk_id, case_id, rank, rerank_score in entries
        ]
        with self._conn() as conn:
            conn.executemany(
                "INSERT INTO retrieval_log "
                "(session_id, query_hash, chunk_id, case_id, rank, rerank_score) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                rows,
            )

    def anchors(
        self,
        session_id: str,
        min_hits: int = 2,
        limit: int = 20,
        weight_by_pagerank: bool = True,
    ) -> list[dict]:
        """Return anchor cases for a session (spec §8.7).

        Returns list of dicts: {case_id, hits, first_retrieved_at,
        last_retrieved_at}.  Enrichment with case_name + pagerank_percentile
        is done by the caller (graph/anchors.py) via case_metadata.get_many.
        """
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT case_id, COUNT(*) AS hits, "
                "       MIN(retrieved_at) AS first_retrieved_at, "
                "       MAX(retrieved_at) AS last_retrieved_at "
                "FROM retrieval_log "
                "WHERE session_id = ? AND case_id IS NOT NULL "
                "GROUP BY case_id "
                "HAVING hits >= ? "
                "ORDER BY hits DESC "
                "LIMIT ?",
                (session_id, min_hits, limit),
            ).fetchall()
        return [
            {
                "case_id": r["case_id"],
                "hits": r["hits"],
                "first_retrieved_at": r["first_retrieved_at"],
                "last_retrieved_at": r["last_retrieved_at"],
            }
            for r in rows
        ]

    def delete_retrieval_log(self, session_id: str) -> int:
        """Delete all retrieval log rows for a session.

        Called when a session is deleted (spec §4.5.1).
        Returns the number of rows deleted.
        """
        with self._conn() as conn:
            conn.execute(
                "DELETE FROM retrieval_log WHERE session_id = ?",
                (session_id,),
            )
            deleted: int = conn.execute("SELECT changes()").fetchone()[0]
        return deleted

    def prune_retrieval_log(self, older_than_days: int) -> int:
        """Delete retrieval log rows older than N days across all sessions.

        Called by the daily maintenance cron (spec §4.5.4).
        """
        with self._conn() as conn:
            conn.execute(
                "DELETE FROM retrieval_log "
                "WHERE retrieved_at < datetime('now', ?)",
                (f"-{older_than_days} days",),
            )
            deleted: int = conn.execute("SELECT changes()").fetchone()[0]
        logger.info("Pruned retrieval log", extra={"deleted": deleted, "days": older_than_days})
        return deleted

    # ------------------------------------------------------------------
    # Ingestion log
    # ------------------------------------------------------------------

    def log_ingestion_start(
        self,
        corpus: str,
        kind: str,
        target_id: str,
    ) -> int:
        """Record the start of an ingestion operation.  Returns the row id."""
        with self._conn() as conn:
            cursor = conn.execute(
                "INSERT INTO ingestion_log (corpus, kind, target_id, status) "
                "VALUES (?, ?, ?, 'started')",
                (corpus, kind, target_id),
            )
            return cursor.lastrowid  # type: ignore[return-value]

    def log_ingestion_done(self, row_id: int) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE ingestion_log SET status='done', finished_at=datetime('now') WHERE id=?",
                (row_id,),
            )

    def log_ingestion_failed(self, row_id: int, error: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE ingestion_log "
                "SET status='failed', finished_at=datetime('now'), error=? "
                "WHERE id=?",
                (error, row_id),
            )

    def is_ingested(self, corpus: str, target_id: str) -> bool:
        """Return True if target_id has a 'done' record (idempotency check)."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM ingestion_log "
                "WHERE corpus=? AND target_id=? AND status='done' LIMIT 1",
                (corpus, target_id),
            ).fetchone()
        return row is not None

    # ------------------------------------------------------------------
    # Session document inventory
    # ------------------------------------------------------------------

    def upsert_document(
        self,
        *,
        session_id: str,
        source: str,
        chunk_count: int,
        size_bytes: int,
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO documents (session_id, source, chunk_count, size_bytes, ingested_at) "
                "VALUES (?, ?, ?, ?, datetime('now')) "
                "ON CONFLICT(session_id, source) DO UPDATE SET "
                "chunk_count=excluded.chunk_count, "
                "size_bytes=excluded.size_bytes, "
                "ingested_at=datetime('now')",
                (session_id, source, chunk_count, size_bytes),
            )

    def list_documents(self, session_id: str) -> list[dict[str, object]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT source, chunk_count, size_bytes, ingested_at "
                "FROM documents WHERE session_id = ? ORDER BY ingested_at ASC, source ASC",
                (session_id,),
            ).fetchall()
        return [
            {
                "source": row["source"],
                "chunk_count": row["chunk_count"],
                "size_bytes": row["size_bytes"],
                "ingested_at": row["ingested_at"],
            }
            for row in rows
        ]

    def delete_document(self, session_id: str, source: str) -> int:
        with self._conn() as conn:
            conn.execute(
                "DELETE FROM documents WHERE session_id = ? AND source = ?",
                (session_id, source),
            )
            deleted: int = conn.execute("SELECT changes()").fetchone()[0]
        return deleted

    def delete_documents_for_session(self, session_id: str) -> int:
        with self._conn() as conn:
            conn.execute(
                "DELETE FROM documents WHERE session_id = ?",
                (session_id,),
            )
            deleted: int = conn.execute("SELECT changes()").fetchone()[0]
        return deleted

    # ------------------------------------------------------------------
    # Background ingest jobs
    # ------------------------------------------------------------------

    def create_job(self) -> str:
        import uuid

        job_id = str(uuid.uuid4())
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO jobs (job_id, status) VALUES (?, 'processing')",
                (job_id,),
            )
        return job_id

    def update_job(self, job_id: str, *, status: str, error: str | None = None) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE jobs SET status = ?, completed_at = CASE WHEN ? IN ('done', 'failed') THEN datetime('now') ELSE completed_at END, error = ? WHERE job_id = ?",
                (status, status, error, job_id),
            )

    def get_job(self, job_id: str) -> dict[str, object] | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT job_id, status, error FROM jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "job_id": row["job_id"],
            "status": row["status"],
            "error": row["error"],
        }

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Session limits (spec §4.5.4)
    # ------------------------------------------------------------------

    def count_session_chunks(self, session_id: str) -> int:
        """Return the number of chunks indexed for a session in the retrieval log.

        Used to enforce max_chunks_per_session at ingest time.  Because we
        count by retrieval_log writes it's an approximation; the authoritative
        source of truth is the index itself, but this is fast and avoids a
        cross-process Chroma query on every upload.

        A separate, accurate path counts via the Chroma store directly from
        the ingest route; this helper exists as a lightweight DB-only path.
        """
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COUNT(DISTINCT chunk_id) FROM retrieval_log WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return int(row[0]) if row else 0

    def list_session_sources(self, session_id: str) -> list[str]:
        """Return distinct source filenames recorded in the ingestion log for a session.

        Used to enforce max_files_per_session at ingest time.
        """
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT DISTINCT target_id FROM ingestion_log "
                "WHERE corpus LIKE ? AND status='done'",
                (f"%{session_id}%",),
            ).fetchall()
        return [r[0] for r in rows]

    # ------------------------------------------------------------------
    # Admin stats (spec §Phase 7.2)
    # ------------------------------------------------------------------

    def retrieval_stats(self, since: str | None = None) -> list[dict]:
        """Return per-day retrieval stats for the admin endpoint.

        Aggregates: queries (distinct query_hash), chunks_returned, p50/p95
        latency are unavailable from retrieval_log alone (no duration column),
        so we report row-level counts. degraded% requires the pipeline, so
        this focuses on volume and rank distribution.

        Args:
            since: ISO date string (YYYY-MM-DD). Defaults to last 30 days.

        Returns:
            List of dicts: {date, query_count, chunk_count, unique_cases,
                            top_rank_1_ratio, sessions}.
        """
        since_clause = since or "date('now', '-30 days')"
        # Build the date filter — if caller provided a date string, use it as
        # a literal; otherwise fall back to the 30-day window expression.
        if since:
            date_filter = ("retrieved_at >= ?", (since,))
        else:
            date_filter = ("retrieved_at >= date('now', '-30 days')", ())

        query = (
            "SELECT date(retrieved_at) AS day, "
            "       COUNT(DISTINCT query_hash) AS query_count, "
            "       COUNT(*) AS chunk_count, "
            "       COUNT(DISTINCT case_id) AS unique_cases, "
            "       COUNT(DISTINCT session_id) AS sessions, "
            "       SUM(CASE WHEN rank = 1 THEN 1 ELSE 0 END) AS rank1_count "
            f"FROM retrieval_log WHERE {date_filter[0]} "
            "GROUP BY day ORDER BY day DESC"
        )
        with self._conn() as conn:
            rows = conn.execute(query, date_filter[1]).fetchall()
        results = []
        for r in rows:
            chunk_count = r["chunk_count"] or 0
            rank1 = r["rank1_count"] or 0
            results.append({
                "date": r["day"],
                "query_count": r["query_count"],
                "chunk_count": chunk_count,
                "unique_cases": r["unique_cases"],
                "sessions": r["sessions"],
                "top_rank_1_ratio": round(rank1 / chunk_count, 4) if chunk_count else 0.0,
            })
        return results

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def health(self) -> str:
        with self._conn() as conn:
            conn.execute("SELECT 1").fetchone()
        return "ok"
