# Hybrid Retrieval Service — Design Specification

**Date:** 2026-05-01
**Status:** Draft for review
**Scope:** Backend retrieval subsystem of Legal Scribe

---

## 1. Goals

Build an isolated, fault-tolerant retrieval microservice that returns top-k legally relevant chunks for a query, combining sparse (BM25) and dense (HNSW) retrieval via Reciprocal Rank Fusion (RRF), with a cross-encoder reranking step. The service must scale to the full U.S. case-law corpus once the download pipeline is in place.

## 2. Non-Goals

- LLM generation (remains in `api.py` `/generate`).
- Frontend changes.
- Replacing the existing user-document ingestion path (this design adds a layer; the existing path is refactored to call it).
- Implementing Shepardizing / "good law" data acquisition (the schema reserves the field; population is out of scope).

## 3. Service Boundaries

A new Python service `retriever_service` exposed via FastAPI on a dedicated port (default 8001). The existing `api.py` calls it over HTTP. The retriever owns:

- Chunk store (source of truth for chunk text and metadata).
- Sparse index (BM25, Tantivy).
- Dense index (HNSW, ChromaDB).
- Cross-encoder reranker.
- Legal text normalization layer.

`api.py` retains: session/message persistence, file upload handling (delegates chunking to the retriever), LLM generation, citation post-processing.

## 4. Data Stores

### 4.1 Chunk store + dense index — ChromaDB

Two collections, both shared:

- `ny_case_law` — all binding-precedent chunks across the corpus.
- `user_workspace` — all user-uploaded chunks across **all** matters; isolated *logically* by a mandatory `session_id` metadata filter on every query.

Logical isolation is the team's accepted trade-off: simpler ops, fewer collections, no per-matter resource overhead. The cost is that a single bug dropping the `session_id` filter would leak data across matters. Mitigations are spelled out in §4.5.

HNSW parameters (explicit, not defaults):

- `hnsw:space = "cosine"` (default in ChromaDB is L2 — must be set explicitly; current code does not, see Bug B-3).
- `hnsw:M = 16`
- `hnsw:construction_ef = 200`
- `hnsw:search_ef = 100`

### 4.2 Sparse index — Tantivy via `tantivy-py`

Two Tantivy indexes, mirroring the ChromaDB collections one-to-one:

- `ny_case_law` index — all case-law chunks.
- `user_workspace` index — all user-uploaded chunks, isolated by mandatory `session_id` filter on every query.

Schema mirrors the per-chunk metadata in §6.5; both indexes store the same fields so any field can be a filter on either side. The `session_id` field is added to the schema below for the `user_workspace` use.

| Field | Type | Indexed | Stored | Notes |
|---|---|---|---|---|
| `chunk_id` | str | yes | yes | Primary key; matches ChromaDB ID. |
| `session_id` | str | yes (raw) | yes | Set on user_workspace chunks; empty on case_law chunks. **Mandatory filter** on every user_workspace query. |
| `text` | text | yes (English analyzer + legal token filter) | yes | Normalized body. |
| `case_id` | u64 | yes | yes | CAP `id`. |
| `case_name` | str | yes (raw) | yes | |
| `citation_official` | str | yes (raw) | yes | E.g. `"4 Abb. Ct. App. 1"`. |
| `decision_date` | date | yes | yes | |
| `court_name` | str | yes | yes | |
| `jurisdiction` | str | yes | yes | |
| `section_type` | str | yes | yes | `"head_matter"` \| `"opinion"`. |
| `opinion_type` | str | yes | yes | `"majority"` \| `"dissent"` \| `"concurrence"` \| empty. |
| `opinion_author` | str | yes | yes | Nullable. |
| `opinion_index` | u64 | yes | yes | Section addressing. |
| `chunk_idx` | u64 | yes | yes | Position within section. |
| `pagerank_percentile` | f64 | yes | yes | Authority signal. |
| `ocr_confidence` | f64 | yes | yes | Quality signal. |
| `cites_to_case_ids` | u64 (multi) | yes | yes | Outgoing citation graph edges. |
| `good_law` | bool | yes | yes | Reserved; default `true`. |
| `source` | str | yes | yes | Original filename / URL. |

For user-uploaded chunks, case-law-specific fields (`case_id`, `pagerank_percentile`, `cites_to_case_ids`, etc.) are unset; `section_type` is `"user_upload"`; `session_id`, `source`, `text`, `chunk_idx` are populated; an additional `page_number` field is set when the parser exposes it.

### 4.3 Relational tables — added to existing `legal_sessions.db`

The retriever needs two relational tables. Both are added to the SQLite database that `api.py` already owns (`legal_sessions.db`) — no separate `retriever_service.db` file. The retriever owns the *write* responsibility for these tables; `api.py` owns connection management and migrations.

There is **no `cases` table.** Case-level metadata (`case_name`, `decision_date`, `court_name`, `pagerank_percentile`, `ocr_confidence`, etc.) is already attached to every chunk in Tantivy + Chroma. `GET /cases/{case_id}` is served by a `LIMIT 1` lookup against the chunks index — no duplication, no separate maintenance.

#### 4.3.1 `citations` (case-citation graph)

```sql
CREATE TABLE IF NOT EXISTS citations (
    citing_case_id INTEGER NOT NULL,
    cited_case_id  INTEGER NOT NULL,
    PRIMARY KEY (citing_case_id, cited_case_id)
) WITHOUT ROWID;

CREATE INDEX IF NOT EXISTS idx_citations_cited
    ON citations(cited_case_id);   -- powers reverse lookup (cited_by)
```

- Populated at ingest time from `cites_to[].case_ids` in the CAP JSON. CAP entries with empty `case_ids` (cited case not in coverage) are skipped.
- `WITHOUT ROWID` saves space on a pure-key table; lookups remain O(log n).
- Estimated size: ~16 bytes/edge × ~50M edges (worst case full national CAP) ≈ 800 MB. NY-only is ~10× smaller.
- Irreplaceable from REST: no public API answers `cited_by` for CAP IDs (see prior discussion). Storing edges is the only viable design.

#### 4.3.2 `retrieval_log` (audit + anchor-detection foundation)

```sql
CREATE TABLE IF NOT EXISTS retrieval_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   TEXT NOT NULL,
    query_hash   TEXT NOT NULL,        -- md5 of normalized query
    chunk_id     TEXT NOT NULL,
    case_id      INTEGER,              -- NULL for user-upload chunks
    rank         INTEGER NOT NULL,
    rerank_score REAL,
    retrieved_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_retrieval_session_case
    ON retrieval_log(session_id, case_id);

CREATE INDEX IF NOT EXISTS idx_retrieval_session_time
    ON retrieval_log(session_id, retrieved_at);
```

- One row per (chunk returned to caller, query). For `k=10`, ten rows per `/retrieve`.
- **Written synchronously to disk** — not buffered in memory. Single INSERT in WAL mode is ~10–50μs; the cost is a rounding error against the 200ms reranker pass. In-memory + periodic flush was considered and rejected: crash loss between flushes silently decays the anchor feature, and SQLite's local write cost makes the optimization unnecessary.
- Retention: 365 days per session by default; daily cron prunes older rows (§4.5.4).
- Foundation for **anchor-case detection** (§8.7): `GROUP BY case_id` over rows filtered by `session_id`, optionally weighted by the `pagerank_percentile` looked up from a chunk in the chunks index.

#### 4.3.3 Connection ownership

- `legal_sessions.db` is owned by `api.py`. Schema migrations (including these two tables) live in `api.py`'s startup.
- The retriever opens its own connection(s) to the same file. SQLite WAL mode supports multiple writers across processes. Each request opens (or borrows from a small pool) a connection; no shared module-level cursor (the bug B-2 pattern is not repeated).
- Backups are simple: one file (`legal_sessions.db`) covers sessions, messages, citations, and retrieval log.

### 4.4 Chunk identifier

All indexes use the same deterministic ID:

```
chunk_id = md5(normalized_text + "|" + source + "|" + (session_id or "")).hexdigest()
```

Re-ingesting the same chunk produces the same ID, providing inherent deduplication. Including `session_id` in the hash ensures the same paragraph uploaded to two different matters produces two distinct IDs (they're conceptually different chunks, even if textually identical).

### 4.5 Session Documents Management

The `user_workspace` indexes (Chroma + Tantivy) hold chunks for *all* matters. This section defines how those chunks are created, scoped, deleted, and bounded — and how the system maintains the isolation guarantee at the code layer rather than the storage layer.

#### 4.5.1 Lifecycle

| Trigger | Effect |
|---|---|
| **Session created** (`POST /sessions` in `api.py`) | No retriever-side action. The session has no documents yet; nothing to provision. The `session_id` simply becomes a label that future writes will carry. |
| **Document uploaded** (`POST /ingest` with `session_id`) | Parse → normalize → chunk → embed → write to both indexes with `session_id` set. Idempotent: re-uploading the same file produces identical IDs and silently overwrites. |
| **Document re-uploaded with same filename, different content** | Old chunks orphan because the content hash changes. The ingest path runs a **prefix delete** first: `DELETE WHERE session_id=X AND source=filename` on both indexes, then re-inserts. (Fixes Bug B-5 from the gap map.) |
| **Single document removed** (`DELETE /sessions/{id}/documents/{source}`) | Filtered delete on both indexes by `(session_id, source)`. Returns count of chunks removed. |
| **Session deleted** (`DELETE /sessions/{id}`) | Filtered delete on both indexes by `session_id` alone. Also deletes `retrieval_log` rows for the session. Cases and citations tables are untouched (they hold case-law data, not user data). |
| **Session inactive ≥90 days** (no upload, no `/retrieve`, no message) | Marked `archived` in `api.py`'s sessions table. Chunks are NOT deleted automatically — only on explicit user action. Archive flag is advisory; the session can be reactivated at any time without re-ingestion. |

#### 4.5.2 Isolation enforcement (code-layer)

Since storage is shared, isolation depends on every retrieval and delete path applying the `session_id` filter. The discipline:

1. **Single chokepoint.** All `user_workspace` queries go through `dense_retriever.query_user_workspace(session_id, ...)` and `bm25_retriever.query_user_workspace(session_id, ...)`. Neither method's signature allows omitting `session_id`. There is no public method that queries `user_workspace` without it.
2. **Mandatory pydantic validation.** The `/retrieve` request schema makes `session_id` required when `corpora` includes `user_workspace`. Validation rejects requests that ask for `user_workspace` without it (400, not 200-with-empty-results).
3. **Server-side defense in depth.** Even if a caller passes `session_id=""`, the retriever rejects empty strings explicitly. Falsy `session_id` values never reach the index.
4. **Audit log.** Every user_workspace query writes `(session_id, query_hash, returned_chunk_ids)` to `retrieval_log`. Periodic cross-checks: `SELECT chunk_id FROM retrieval_log r JOIN chunks c ON r.chunk_id=c.chunk_id WHERE r.session_id != c.session_id` should always return zero rows.
5. **Integration test.** A test in the suite uploads two distinct documents to two sessions and asserts `/retrieve` for session A never returns session B's chunks, across 100 randomized queries.

#### 4.5.3 Lazy patterns

These are deliberately lazy / deferred to keep request latency low:

| Operation | Pattern | Rationale |
|---|---|---|
| PDF text extraction | Async background task; `POST /ingest` returns `{job_id, status: "processing"}` immediately. `GET /ingest/jobs/{id}` polled by client. | Large PDFs block the request thread for seconds. Fixes the synchronous-upload concern in the gap map. |
| Embedding generation | Batched per upload (batch size 32) inside the background task. | Sentence-transformer batching is ~5× faster than one-at-a-time. |
| Tantivy segment merging | Native background merger; never forced. | Forcing merges blocks writes; let Tantivy decide. |
| `retrieval_log` writes | Fire-and-forget via `asyncio.create_task`; failures logged but don't block the response. | Audit data; eventual consistency is fine. |
| Stale-session cleanup | Cron-style job (daily); not request-path. | Inactive matters don't need real-time reaping. |

#### 4.5.4 Thresholds

Hard limits to prevent abuse and bound storage:

| Limit | Default | Enforcement |
|---|---|---|
| Max upload size per file | 50 MB | `POST /ingest` rejects with 413 (Payload Too Large). |
| Max files per session | 200 | `POST /ingest` rejects with 409 (Conflict, body explains). |
| Max chunks per session | 20,000 | Same. |
| Max total storage per session | 500 MB | Computed from on-disk chunk size; rejects on overrun. |
| Retrieval log retention per session | 365 days | Daily cron prunes older rows; anchor detection only needs recent history. |
| Inactive-archive threshold | 90 days since last activity | Sets `archived` flag; chunks remain. |
| Hard-delete threshold | None by default | User-initiated only; we never auto-delete chunks. Configurable per deployment for stricter retention regimes. |

All thresholds are config-overridable per deployment (a firm with stricter compliance can lower them; a heavy-use enterprise can raise them).

#### 4.5.5 What is *not* per-session

To make the model unambiguous:

- **`ny_case_law` is global.** Never filtered by session.
- **`cases`, `citations` tables are global.** Case-law metadata is shared.
- **`retrieval_log` is per-session.** Drives anchor detection; pruned on session delete.
- **Reranker model and embedder are global.** Loaded once at service startup, shared across all requests.

## 5. Atomic Triple-Index Ingestion

A `DualIndexWriter` context manager wraps writes to **three** targets: ChromaDB (chunks), Tantivy (chunks), and the SQLite `citations` table (graph edges). Per-chunk metadata carries everything else; there is no separate cases table to write.

```python
with DualIndexWriter(corpus="ny_case_law") as w:
    for case in cases:
        for chunk in chunk_cap_case(case):
            w.add_chunk(chunk_id, text, embedding, metadata)   # → Tantivy + Chroma (buffered)
        w.add_citation_edges(case.id, case.cites_to_ids)       # → SQLite (buffered)
    # commit() on __exit__:
    #   1. SQLite transaction (citations) — atomic by design.
    #   2. Tantivy commit.
    #   3. ChromaDB upsert.
    #   4. On any exception: rollback SQLite, delete by ID from whichever index committed.
```

A small SQLite `ingestion_log` table (also in `legal_sessions.db`) records `(corpus, kind, target_id, status, started_at, finished_at, error)` for reconciliation:

```sql
CREATE TABLE IF NOT EXISTS ingestion_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    corpus       TEXT NOT NULL,
    kind         TEXT NOT NULL,        -- "chunk" | "citation_batch"
    target_id    TEXT NOT NULL,        -- chunk_id or citing_case_id
    status       TEXT NOT NULL,        -- "started" | "done" | "failed"
    started_at   TEXT NOT NULL DEFAULT (datetime('now')),
    finished_at  TEXT,
    error        TEXT
);
CREATE INDEX IF NOT EXISTS idx_ingestion_target ON ingestion_log(corpus, target_id);
```

A `reconcile.py` job compares ID sets across Tantivy, Chroma, and (for case-law) `citations` per corpus, removes orphans, and reports discrepancies.

## 6. Legal-Aware Normalization Layer

A single `normalize(text: str) -> str` function applied **identically** at ingestion and query time. Symmetry is mandatory; any divergence silently hurts recall.

Pipeline:

1. **Unicode normalization:** NFKC, ligature unfolding, smart quote / em-dash folding.
2. **Citation canonicalization** via `eyecite`: every detected citation is replaced with a stable token: `[CITE_<reporter>_<volume>_<page>]` (e.g. `[CITE_NY3D_15_42]`). Survives BM25 tokenization and is treated as a single embedding token.
3. **Section markers:** `§` → ` section `, `§§` → ` sections `, `¶` → ` paragraph `.
4. **Case names:** regex `\b([A-Z][A-Za-z\.\-']+)\s+v\.\s+([A-Z][A-Za-z\.\-']+)\b` → `[CASE_<P1>_v_<P2>]`.
5. **Whitespace collapse.**

Tantivy receives normalized text and applies its English analyzer on top (lowercasing, stemming, stopword removal). The bracketed `[CITE_*]` / `[CASE_*]` tokens are added to the analyzer's protected-token list so they pass through unmodified.

## 6.5 Chunking Strategy (CAP case-law)

CAP cases arrive as JSON with the structure:

```
{
  id, name_abbreviation, decision_date, citations[], court, jurisdiction,
  cites_to[], analysis: {pagerank, ocr_confidence, ...},
  casebody: {
    head_matter: "...",
    opinions: [{text, type, author}, ...]
  }
}
```

Chunking treats `head_matter` and each opinion as **independent sections**. They are *not* concatenated and *not* one-chunk-per-section. Each section is split independently with a recursive paragraph splitter; every chunk inherits all case-level metadata plus section-level metadata.

### 6.5.1 Algorithm (high level)

For each case JSON:

1. Pre-filter: drop case if `analysis.ocr_confidence < 0.3` (rare; mostly very old scans).
2. Build the section list: `[("head_matter", head_matter_text, None, None)] + [("opinion", op["text"], op["type"], op["author"]) for op in opinions]`.
3. For each section:
   1. Run the legal normalizer (§6) on the section text.
   2. Apply **structure-first hierarchical splitting** (full technique in §6.5.5).
   3. For each surviving chunk, emit a record with the metadata table in §6.5.2.

### 6.5.2 Per-chunk metadata (CAP case-law)

| Field | Source in JSON | Purpose |
|---|---|---|
| `case_id` | `id` | Join key. |
| `case_name` | `name_abbreviation` | Display + BM25. |
| `citation_official` | `citations[?type=official].cite` | Stable citation string. |
| `decision_date` | `decision_date` | Recency filter / boost. |
| `court_name` | `court.name` | Binding-vs-persuasive routing. |
| `jurisdiction` | `jurisdiction.name` | Filter. |
| `section_type` | derived (`"head_matter"` \| `"opinion"`) | **Critical** — system prompt prefers `opinion` chunks for rule citation. |
| `opinion_type` | `opinions[i].type` | Allows filtering out dissents/concurrences when searching for binding rules. |
| `opinion_author` | `opinions[i].author` | Display. |
| `opinion_index` | array index in `opinions` | Reconstruction. |
| `chunk_idx` | counter within section | Reconstruction + dedup safety. |
| `pagerank_percentile` | `analysis.pagerank.percentile` | Authority signal — used as a tiebreaker / soft boost in reranking. |
| `ocr_confidence` | `analysis.ocr_confidence` | Quality signal — drop <0.3 globally; downweight low values in scoring. |
| `cites_to_case_ids` | flattened from `cites_to[].case_ids` | Citation-graph edges; populates the knowledge-graph backend with no NLP work. |
| `good_law` | reserved, default `true` | Future Shepardizing. |
| `source` | filename / URL | Provenance. |

### 6.5.3 Why these structural choices

- **Separating `head_matter` from opinions** prevents the LLM from citing the editor's syllabus paraphrase as if it were the binding holding. The `section_type` field is exposed in retrieval results so the system prompt can say: *"cite rules from `opinion` chunks with `opinion_type=majority`; cite facts from any chunk."*
- **One chunk per opinion-section-piece, not per opinion** — opinions routinely run 10k–50k chars; embedding context (~2k chars for MiniLM) and reranker context (512 tokens) both demand sub-section chunking.
- **`cites_to` is gold** — Harvard already extracted the case-to-case citation graph and ID-linked it. Storing it per-chunk lets the knowledge-graph feature populate connections without running eyecite over the corpus. (eyecite is still needed for *user-uploaded* documents, which lack this metadata.)
- **`pagerank.percentile`** — pre-computed authority. Plumbed through retrieval results so the reranker (or a downstream weighted RRF variant) can use it as a soft boost.

### 6.5.4 Chunking for user uploads

For user-uploaded PDFs/text the same sentence-aware splitter from §6.5.5 step 4 is used (target 256 tokens, overlap 40, min 30), but:

- `section_type = "user_upload"`.
- No structural pre-split (no Held-points or section-header detection); user uploads are treated as one flat section.
- All case-law-specific fields are unset.
- `source` = filename; an additional `page_number` field is set when the parser exposes it (PyPDF2 does).
- The contextual prefix (§6.5.5 step 5) is `[<filename>]` — provenance only, no case identity.

The current ad-hoc `\n\n` split with `len > 50` (`api.py:149`) is replaced by this unified path.

### 6.5.5 Chunking technique (Approach B — structure-first hierarchical splitting)

The splitter applied per section in §6.5.1 step 3.2. Heuristic-driven, designed to degrade gracefully when structural cues are absent.

#### Pipeline per section

```
section_text  (already normalized: eyecite + § + case-name preservation)
   │
   ▼ STEP 1  structural pre-split (section-type-aware)
   │
   ▼ STEP 2  for each structural unit:
   │           STEP 3  sentence-tokenize via pysbd (legal-aware)
   │           STEP 4  pack sentences into token-bounded chunks with sentence-aligned overlap
   │           STEP 5  prepend contextual prefix (config-toggleable)
   │
   ▼ chunks → DualIndexWriter
```

#### Step 1 — Structural pre-split

`head_matter`: detect numbered Held-points and the attorneys block:

| Pattern | Regex (illustrative) | Action |
|---|---|---|
| Held points | `r"(?:^|\n)\s*Held[,:]?\s*[0-9]+\.?\s+"` | Each Held point becomes its own structural unit, regardless of size (down to the 30-token floor — units below it merge with the next Held point). |
| Attorneys block | `r"(?:^|\n)[A-Z][^\n]+,\s+attorney for [^\n]+"` | Attorneys block becomes its own unit (often retrieves poorly anyway; isolating it prevents pollution of fact-pattern chunks). |
| Residual | everything else | One unit, falls through to step 2. |

`opinion`: detect section markers:

| Pattern | Regex (illustrative) | Action |
|---|---|---|
| Roman-numeral headers | `r"(?:^|\n)\s*[IVX]+\.\s+"` | Split on these; each section is a structural unit. |
| Letter sub-headers | `r"(?:^|\n)\s*[A-Z]\.\s+"` | Same, used only when Roman headers are absent. |
| Named headers | `r"(?:^|\n)\s*(Background|Discussion|Held|Conclusion|Analysis|Facts|Procedural History)\s*[:.]?\s*$"` | Same. |
| No structure detected | — | Whole section is one unit. Proceeds to step 2 unchanged. |

The detectors are deliberately conservative: false positives (treating a normal sentence as a header) are worse than false negatives (missing a header). When structural detection fails, the splitter falls back to a single unit and the sentence-packing step (4) still produces well-formed chunks.

#### Step 2 — Per-unit processing

Each structural unit ≥ 30 tokens proceeds to step 3. Units < 30 tokens merge into the next unit (or, for the last unit, the previous one).

#### Step 3 — Sentence tokenization

Library: **`pysbd`** (Python Sentence Boundary Disambiguation), language `"en"`. Designed for sentence segmentation with abbreviation handling; pure-Python; ~10× faster than spaCy on this task.

Custom abbreviation augmentation: extend pysbd's English abbreviation list with legal-specific tokens that survived the normalizer: `Inc.`, `Co.`, `Corp.`, `Ltd.`, `LLC.`, `Mr.`, `Mrs.`, `Dr.`, `J.` (judge initial), `C.J.`, `P.`, `Mass.`, `R.I.`, `Conn.`, plus common reporter abbreviations not already absorbed by `[CITE_*]` tokens.

Sentence boundaries are *never* placed inside a `[CITE_*]`, `[CASE_*]`, or quoted span (`"…"`, `'…'`, or block quote indented by ≥ 4 spaces). Implementation: pre-mask these spans with sentinel tokens before pysbd, restore after.

#### Step 4 — Token-bounded packing with sentence-aligned overlap

Length unit: tokens, measured by the embedder's tokenizer (`AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")`). Char-count is never used.

Targets:

| Parameter | Default | Source of constraint |
|---|---|---|
| `target_tokens` | 256 | Half of MiniLM's 512-token limit; leaves room for the cross-encoder pair `(query, chunk)` |
| `overlap_tokens` | 40 | ~15% of target; aligned to sentence boundaries (overlap is N preceding sentences whose total tokens ≥ 40) |
| `min_chunk_tokens` | 30 | Below this the chunk has no retrieval value; merged into adjacent chunk |
| `max_chunk_tokens` | 320 | Hard cap — a single very long sentence that would bust this is split at the nearest comma; truly atomic spans (a single quoted statute) are emitted oversized with a warning logged |

Packing algorithm: greedy. Accumulate sentences until adding the next sentence would exceed `target_tokens`; emit the chunk; rewind to the sentence-boundary that puts the cursor `overlap_tokens` back. Repeat to the end of the unit.

#### Step 5 — Contextual prefix (config-toggleable)

When `chunking.contextual_prefix.enabled = true` (default), prepend a single bracketed line to the chunk text *before* it goes to Tantivy and Chroma:

```
[Ranger v. Goodrich | opinion:majority]
<chunk body>
```

Format: `[<case_name> | <section_type>:<opinion_type or 'head_matter'>]`. For user uploads: `[<source_filename>]`.

Why: the metadata fields stored on the chunk are filter-only on the dense channel and a separate field on the sparse channel; they do not enter the embedding vector or the cross-encoder input. The prefix injects the case identity into both the dense embedding and the BM25 body channel without requiring multi-field query construction. Cost: ~10 tokens (~4% of target). When eval shows no measurable lift, set `enabled: false`.

The prefix is **stripped from the response payload** in `/retrieve` — the frontend and the LLM see only the chunk body. The prefix is an indexing-time artifact.

#### Step 6 — Atomicity guarantees

Across all steps, the splitter never:

- Splits inside a `[CITE_*]` or `[CASE_*]` token (preserved as sentinels through pysbd).
- Splits inside a quoted span (`"…"`, `'…'`, block quote).
- Drops content silently. Anything not emitted as a chunk because of size constraints is logged with the case_id for inspection.

#### Step 7 — Heuristic guardrails (logged warnings, not errors)

| Condition | Action |
|---|---|
| Section produces zero chunks (e.g., entire section is one citation) | Log warning with case_id + section info; emit a single chunk with the raw text capped at `max_chunk_tokens`. |
| Single sentence > `max_chunk_tokens` | Split at the nearest comma; if no comma, emit oversized and log. |
| Held-point detection finds > 20 points | Likely false positive on a list-heavy passage. Fall back to no structural split for that section. |
| Roman-numeral header detection finds > 15 sections | Same fallback. |

These guardrails make the heuristic-based detectors safe to deploy without per-case tuning. Warnings accumulate in the ingestion log and inform regex refinement over time.

#### Worked example — §6.5 example case

Applied to `https://static.case.law/abb-ct-app/4/cases/0001-01.json` (`Ranger v. Goodrich`):

- `head_matter`: structural detector finds two Held points (`Held, 1` and `3.` — point 2 missing, OCR artifact). Each becomes its own structural unit. Residual block (parties summary, factual summary, attorneys) becomes a third unit. Sentence-packed result: ~4 chunks.
- `opinion[0]` (majority by J. M. Parker): no Roman-numeral headers detected; one structural unit. Sentence-packed result: ~2 chunks (it's a short opinion).

Total: ~6 chunks for the case, each with a `[Ranger v. Goodrich | opinion:majority]` or `[Ranger v. Goodrich | head_matter]` prefix.

---

## 7. Retrieval Pipeline

```
query
  → normalize()
  → parallel:
       ├── Tantivy BM25 top-100 (filtered)
       └── ChromaDB HNSW top-100 (filtered)
  → RRF fuse (k=60) → top-50
  → cross-encoder rerank → top-k (default 10)
  → return
```

### 7.1 RRF

Canonical Cormack/Clarke/Buettcher formula:

```
score(d) = Σ_{r ∈ retrievers} 1 / (k + rank_r(d))
```

with `k = 60`. No score normalization, no per-retriever weights in v1 (config-tunable for future experiments).

Candidate pool: top-100 per retriever → fused list of up to 200 unique IDs → top-50 to reranker.

### 7.1.1 Filter primitives

The `/retrieve` `filters` block exposes the data-quality and authority signals as first-class query controls. Filters are pushed down into both indexes (Chroma `where` clauses, Tantivy term/range queries) — never post-filtered, which would break top-k semantics.

| Filter | Type | Default | Effect |
|---|---|---|---|
| `min_ocr_confidence` | float | 0.3 | Excludes low-quality OCR chunks. |
| `min_pagerank_percentile` | float | 0.0 | Restricts to authoritative cases. |
| `exclude_section_types` | list[str] | `[]` | E.g. `["head_matter"]` for rule-only retrieval. |
| `exclude_opinion_types` | list[str] | `[]` | E.g. `["dissent", "concurrence"]` for binding-only retrieval. |
| `decision_date_gte` / `_lte` | ISO date | none | Recency window. |
| `court_name` | list[str] | none | Jurisdictional restriction. |
| `good_law` | bool | none | Will become `true` by default once Shepardizing exists. |
| `corpora` | list[str] | session default | Which corpora to query. |

The default `min_ocr_confidence=0.3` is enforced by the service even when the client omits the filter — bad-OCR results are never silently surfaced.

### 7.2 Reranker

- Model: `cross-encoder/ms-marco-MiniLM-L-6-v2` (HuggingFace, MIT, 22M params).
- Loaded once at service startup.
- Config: `max_length=512`, `batch_size=16`, raw logits as score, MPS device with CPU fallback.
- Input: `(query, chunk_text)` pairs for the top-50 fused candidates.
- Output: top-10 by reranker score (configurable per request).

Latency budget: ~200ms on MPS for 50 pairs × 512 tokens. Total p95 target: ≤500ms end-to-end.

Future swap path: `BAAI/bge-reranker-base` (278M, MIT) — higher quality, 5–8× slower.

### 7.2.1 Pagerank-aware final score (optional)

When `reranker.pagerank_boost_weight` (config) is non-zero, the final score for each candidate is:

```
final_score = reranker_logit + alpha * pagerank_percentile
```

with `alpha` defaulting to `0.0` (off) and a recommended starting value of `0.3` once the eval set exists to validate it. The breakdown is exposed in the response under `score_components` when `return_debug=true`, so the eval harness can ablate the boost without code changes. For user-uploaded chunks (no `pagerank_percentile`), the boost term is treated as zero.

## 8. Service API

### 8.1 `POST /retrieve`

**Request:**

```json
{
  "query": "string",
  "k": 10,
  "corpora": ["ny_case_law", "user_workspace_<session_id>"],
  "filters": {
    "good_law": true,
    "decision_date_gte": "2010-01-01",
    "court_level": ["appellate", "supreme"]
  },
  "rerank": true,
  "return_debug": false
}
```

**Response 200:**

```json
{
  "results": [
    {
      "chunk_id": "a3f...",
      "case_id": 1117516,
      "text": "...",
      "score": 0.87,
      "source": {
        "case_name": "Smith v. Jones",
        "citation_official": "15 N.Y.3d 42",
        "decision_date": "2014-06-01",
        "court_name": "New York Court of Appeals",
        "corpus": "ny_case_law",
        "section_type": "opinion",
        "opinion_type": "majority",
        "opinion_author": "Smith, J.",
        "pagerank_percentile": 0.83,
        "ocr_confidence": 0.97
      },
      "ranks": { "bm25": 3, "dense": 7, "rrf": 1, "rerank": 1 },
      "score_components": { "reranker_logit": 0.71, "pagerank_boost": 0.16 }
    }
  ],
  "degraded": false,
  "retrievers_used": ["bm25", "dense", "rerank"],
  "latency_ms": { "bm25": 12, "dense": 18, "rrf": 1, "rerank": 180, "total": 215 }
}
```

`ranks` and `latency_ms` only populated when `return_debug=true`.

**Error responses:**

| Code | When | Body |
|---|---|---|
| 400 | Validation failure | `{"error": "validation", "details": [...]}` |
| 404 | Corpus does not exist | `{"error": "corpus_not_found", "corpus": "..."}` |
| 503 | All retrievers down | `{"error": "all_retrievers_unavailable"}` |

**Degradation policy:** if exactly one of {BM25, dense} fails, return the other's results with `degraded: true` in body and `X-Degraded: true` response header. If reranker fails, return RRF-only ordering with `degraded: true`. Only 503 if no results can be produced at all.

### 8.2 `POST /ingest`

Used by `api.py` upload handler and the future case-law download pipeline.

```json
{
  "corpus": "user_workspace_<session_id>",
  "documents": [
    {"text": "...", "source": "deposition.pdf", "metadata": {...}}
  ]
}
```

Returns:

```json
{"chunks_indexed": 42, "duplicates_skipped": 3, "corpus": "..."}
```

### 8.3 Session document deletion

With logical isolation, there is no "drop the corpus" operation for user data — both `user_workspace` indexes are shared. Deletion is done via filtered removal:

#### 8.3.1 `DELETE /sessions/{session_id}/documents`

Removes **all** chunks for a session from both Tantivy and Chroma `user_workspace` indexes. Also deletes `retrieval_log` rows for the session. Used on session deletion (called by `api.py`).

Response:
```json
{"chunks_deleted": 142, "log_rows_deleted": 87}
```

#### 8.3.2 `DELETE /sessions/{session_id}/documents/{source}`

Removes chunks for a single uploaded document (matched by `source` filename) within the session. Used when the user removes a document from the matter via the dashboard.

Response:
```json
{"chunks_deleted": 18, "source": "deposition.pdf"}
```

`404` if no chunks match.

#### 8.3.3 `GET /sessions/{session_id}/documents`

Lists distinct uploaded documents for a session, with chunk counts and ingestion timestamps. Backs the dashboard's "files in this matter" pane.

```json
{
  "session_id": "...",
  "documents": [
    {"source": "deposition.pdf", "chunks": 18, "ingested_at": "2026-05-01T14:22:11Z", "size_bytes": 245312}
  ],
  "totals": {"documents": 3, "chunks": 47, "size_bytes": 612480}
}
```

> Note: there is no endpoint that drops the global `ny_case_law` index via the API. Case-law re-ingestion is operator-level work via `scripts/ingest_cap.py`.

### 8.4 `GET /cases/{case_id}`

Returns case-level metadata. There is no `cases` table; the response is derived on demand from the **first chunk** matching `case_id` in the chunks index (every chunk carries the full case metadata as part of its document fields).

```json
{
  "case_id": 1117516,
  "case_name": "Ranger v. Goodrich",
  "citation_official": "4 Abb. Ct. App. 1",
  "decision_date": "1867-09",
  "court_name": "New York Court of Appeals",
  "jurisdiction": "N.Y.",
  "pagerank_percentile": 0.27,
  "ocr_confidence": 0.50,
  "good_law": true,
  "corpus": "ny_case_law",
  "chunk_count": 7
}
```

Implementation: a single Tantivy `case_id:1117516` filter with `limit=1`, plus a count query for `chunk_count`. Both indexes have `case_id` indexed; sub-millisecond on a warm index.

`404` if no chunks match.

A small in-process LRU cache (default 1024 entries, ~10s TTL) absorbs the hot tail — the same case is often hit multiple times during a graph render. Cache invalidation is trivial (TTL only); ingest does not need to invalidate because case metadata is monotonic — it doesn't mutate after ingest, only re-ingestion would change it, and re-ingestion is rare and operator-driven.

### 8.5 `GET /cases/{case_id}/edges`

Citation graph edges for a single case. Query parameters:

- `direction`: `out` | `in` | `both` (default `both`)
- `limit`: int (default 50)

Response:

```json
{
  "case_id": 1117516,
  "out": [
    {"case_id": 2021229, "case_name": "Mickles v. Townsend", "pagerank_percentile": 0.71}
  ],
  "in": [
    {"case_id": 999123, "case_name": "Later v. Authority", "pagerank_percentile": 0.55}
  ]
}
```

**Implementation:**
- `out`: `SELECT cited_case_id FROM citations WHERE citing_case_id = ? LIMIT ?` (PK index).
- `in`: `SELECT citing_case_id FROM citations WHERE cited_case_id = ? LIMIT ?` (the `idx_citations_cited` index).
- For each returned `case_id`, `case_name` and `pagerank_percentile` are looked up via the cached `GET /cases/{id}` path (§8.4). Edges to cases not present in the local corpus (CAP `cites_to` entries with empty `case_ids`) are skipped at ingest, so this lookup never 404s.
- Sub-50 ms for `limit=50` on warm caches.

**Pagerank ordering:** when both `direction=in` and `direction=out` are returned, edges within each side are ordered by the cited/citing case's `pagerank_percentile` descending — most authoritative neighbors first.

### 8.6 `POST /graph/subgraph`

Returns a subgraph for visualization. Typical input: the set of `case_id`s retrieved for a query.

**Request:**

```json
{
  "seed_case_ids": [1117516, 2021229, 1982830],
  "depth": 1,
  "include_external_neighbors": false,
  "max_neighbors_per_seed": 20
}
```

**Response:**

```json
{
  "nodes": [
    {"case_id": 1117516, "case_name": "...", "pagerank_percentile": 0.27, "is_seed": true}
  ],
  "edges": [
    {"from": 1117516, "to": 2021229}
  ]
}
```

When `include_external_neighbors=false` (default), only edges *among* the seed set are returned — the "internal" subgraph. When `true`, the top-N most-cited neighbors of the seeds (ranked by `pagerank_percentile`) are added as nodes. Depth >1 is allowed but capped server-side at 2.

This endpoint is what the frontend's "Semantic Network" pane consumes. The graph is built from real CAP citation edges, not LLM extraction.

### 8.7 `GET /sessions/{session_id}/anchors`

Returns the cases most repeatedly retrieved across a session's history — the "Anchor Cases" feature from CLAUDE.md.

Query parameters:

- `min_hits`: int (default 2) — minimum retrieval count to qualify
- `limit`: int (default 20)
- `weight_by_pagerank`: bool (default true) — if true, ranks by `hits * (1 + pagerank_percentile)` rather than raw hit count

Response:

```json
{
  "session_id": "...",
  "anchors": [
    {
      "case_id": 1117516,
      "case_name": "Ranger v. Goodrich",
      "hits": 5,
      "first_retrieved_at": "2026-05-01T14:22:11Z",
      "last_retrieved_at": "2026-05-03T09:11:42Z",
      "pagerank_percentile": 0.27,
      "anchor_score": 6.35
    }
  ]
}
```

Computed in two steps: (1) `SELECT case_id, COUNT(*) AS hits, MIN(retrieved_at), MAX(retrieved_at) FROM retrieval_log WHERE session_id = ? AND case_id IS NOT NULL GROUP BY case_id ORDER BY hits DESC LIMIT ?` — a single indexed query over the `(session_id, case_id)` index. (2) For each result, the chunks index is consulted (via the same LRU-cached path as `GET /cases/{id}`) to fetch `case_name` and `pagerank_percentile`. Total cost: one SQL query + N cached metadata lookups; sub-100 ms for typical limits.

### 8.8 `GET /health`

```json
{
  "status": "ok",
  "components": {
    "chroma": "ok",
    "tantivy": "ok",
    "reranker": "loaded",
    "normalizer": "ok"
  },
  "version": "0.1.0"
}
```

## 9. Fault Tolerance

- Each retriever wrapped in a per-call timeout (default 2s) and try/except.
- A failing retriever increments a circuit-breaker counter; after N consecutive failures it is marked degraded for a cooldown window and skipped.
- The reranker is non-critical; failure falls back to RRF order.
- Chunk store and sparse index are local files (no network); failures are I/O errors and surface as 500 only if both fail.

## 10. Evaluation Harness

`eval/` directory:

- `eval/queries.jsonl` — `{query, relevant_chunk_ids: [...], notes: "..."}` written manually by the lawyer in the loop.
- `eval/run_eval.py` — for each query calls `/retrieve` with `k=10`, computes via **`ranx`**:
  - `precision@{1,5,10}`
  - `recall@{5,10}`
  - `MRR`
  - `NDCG@10`
- Latency: per-query `time.perf_counter()`, p50/p95/p99 via `numpy.percentile`.
- Output: markdown report at `eval/reports/YYYY-MM-DD.md` with a comparison row vs the previous report.

### 10.1 MVP pass thresholds (calibrate after first baseline)

| Metric | Threshold |
|---|---|
| `recall@10` | ≥ 0.70 |
| `MRR` | ≥ 0.50 |
| `p95_latency_ms` | ≤ 500 |

## 11. Configuration

`retriever_service/config.yaml`:

```yaml
hnsw:
  M: 16
  construction_ef: 200
  search_ef: 100
  space: cosine

bm25:
  k1: 1.2
  b: 0.75

rrf:
  k: 60
  candidate_pool: 100

reranker:
  model: cross-encoder/ms-marco-MiniLM-L-6-v2
  max_length: 512
  batch_size: 16
  enabled_default: true
  top_n_input: 50
  top_n_output: 10
  pagerank_boost_weight: 0.0    # alpha in §7.2.1; recommend 0.3 once eval validates

filters:
  default_min_ocr_confidence: 0.3

graph:
  subgraph_max_depth: 2
  subgraph_max_neighbors: 50

anchors:
  default_min_hits: 2
  default_weight_by_pagerank: true

case_metadata_cache:
  max_entries: 1024
  ttl_seconds: 10

chunking:
  target_tokens: 256
  overlap_tokens: 40
  min_chunk_tokens: 30
  max_chunk_tokens: 320
  tokenizer_model: sentence-transformers/all-MiniLM-L6-v2
  sentence_splitter: pysbd
  contextual_prefix:
    enabled: true                          # ablate via eval; disable if no measurable lift
  structural_split:
    head_matter:
      detect_held_points: true
      detect_attorneys_block: true
      held_points_max: 20                  # fallback threshold (false-positive guard)
    opinion:
      detect_roman_headers: true
      detect_letter_subheaders: true
      detect_named_headers: true
      headers_max: 15                      # fallback threshold

relational_store:
  database_path: ./legal_sessions.db    # shared with api.py
  retrieval_log_retention_days: 365

timeouts_ms:
  bm25: 2000
  dense: 2000
  rerank: 3000

circuit_breaker:
  consecutive_failures: 5
  cooldown_seconds: 60
```

## 12. Module Layout

```
retriever_service/
├── __init__.py
├── main.py                  # FastAPI app
├── config.py                # pydantic-settings
├── normalize.py             # legal text normalizer
├── stores/
│   ├── chunk_ids.py         # ID computation
│   ├── chroma_store.py      # dense index wrapper
│   ├── tantivy_store.py     # sparse index wrapper
│   ├── relational_store.py  # SQLite wrapper for citations + retrieval_log + ingestion_log (in legal_sessions.db)
│   └── case_metadata.py     # derives case-level metadata from chunks index, with LRU cache
├── retrieval/
│   ├── bm25_retriever.py
│   ├── dense_retriever.py
│   ├── rrf.py
│   ├── reranker.py
│   └── retrieval_logger.py  # writes to retrieval_log on every /retrieve
├── graph/
│   ├── edges.py             # /cases/{id}/edges
│   ├── subgraph.py          # /graph/subgraph
│   └── anchors.py           # /sessions/{id}/anchors
├── ingestion/
│   ├── dual_writer.py       # DualIndexWriter (chunks + cases + citations)
│   ├── chunker.py           # corpus-aware chunking
│   └── reconcile.py         # standalone reconciliation job
└── api/
    ├── routes.py
    └── schemas.py           # pydantic models

eval/
├── queries.jsonl
├── run_eval.py
└── reports/
```

## 13. Open Items (post-MVP)

- Reranker swap to `BAAI/bge-reranker-base` once eval shows L-6 is the bottleneck.
- Tantivy index backup/restore strategy.
- Per-tenant rate limiting on `/retrieve`.
- Sharded corpora once ny_case_law exceeds ~5M chunks.
- "Good law" data acquisition pipeline (separate spec).
