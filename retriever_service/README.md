# Legal Scribe — Retriever Service

Hybrid BM25 + dense retrieval microservice for Legal Scribe.  
Runs on **port 8001**; the main app (`api.py`, port 8000) proxies all retrieval and graph endpoints to this service.

---

## Architecture

```
query
  → normalize (lowercase, strip punctuation, expand abbreviations)
  → parallel:
       ├── Tantivy BM25   top-100 candidates   [timeout + circuit breaker]
       └── ChromaDB HNSW  top-100 candidates   [timeout + circuit breaker]
  → RRF fusion (k=60) → top-50
  → cross-encoder rerank → top-k=10            [disabled by default; enable in config.yaml]
  → fire-and-forget retrieval_log write
  → return JSON
```

Two corpora are supported:
- **`ny_case_law`** — NY Court of Appeals full-text opinions (read-only at runtime).
- **`user_workspace`** — Per-session uploaded documents (read/write, session-isolated).

---

## Prerequisites

- Python 3.12+ with the project virtualenv (`.venv`)
- ChromaDB and Tantivy indexes built via `scripts/ingest_cap.py` (for `ny_case_law`)
- `legal_sessions.db` created by `api.py` on first run (shared SQLite database)

Install Python dependencies from the project root:

```bash
.venv/bin/pip install -r requirements.txt
```

---

## Running

### Development (auto-reload)

```bash
.venv/bin/uvicorn retriever_service.main:app --port 8001 --reload
```

### Production

```bash
.venv/bin/uvicorn retriever_service.main:app --port 8001 --workers 1
```

> Use a single worker — Tantivy writers are not safe to share across OS processes.

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `LOG_LEVEL` | `INFO` | Root log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `LOG_FORMAT` | `json` | Set to `console` for human-readable dev output |

Paths (chroma, tantivy, SQLite) are configured in `retriever_service/config.yaml`.

---

## Configuration

Edit `retriever_service/config.yaml`:

```yaml
service:
  chroma_path: chroma_db        # ChromaDB storage directory
  tantivy_path: tantivy_index   # Tantivy index directory

reranker:
  enabled_default: false        # Set true to enable cross-encoder reranking
  model: cross-encoder/ms-marco-MiniLM-L-6-v2

rrf:
  k: 60                         # RRF smoothing constant
  candidate_pool: 100           # Candidates fetched from each leg before fusion

timeouts_ms:
  bm25: 2000
  dense: 5000
  rerank: 8000
```

---

## Running the Eval Harness

The eval harness replays a frozen CLERC-style testset against the live service and reports IR metrics (recall@10, MRR, nDCG@10) and latency.

### Full hybrid eval (both legs, no rerank)

```bash
.venv/bin/python eval/run_eval.py --label hybrid
```

### Isolate a single retrieval leg

```bash
# BM25 only
.venv/bin/python eval/run_eval.py --label bm25-only --retrievers bm25

# Dense only
.venv/bin/python eval/run_eval.py --label dense-only --retrievers dense
```

### Enable reranking

```bash
.venv/bin/python eval/run_eval.py --label hybrid-rerank --rerank
```

### Quick iteration (limit queries)

```bash
.venv/bin/python eval/run_eval.py --label debug --limit 20
```

Reports are written to `eval/reports/YYYY-MM-DD-<label>.md`.

### Pass/fail thresholds (spec §10.3, `all-removed` variant)

| Metric | Threshold |
|---|---|
| recall@10 | ≥ 0.70 |
| mrr | ≥ 0.50 |
| p95 latency | ≤ 500 ms |

---

## Key Endpoints

Full OpenAPI docs at `http://localhost:8001/docs`.

### Retrieval

```
POST /retrieve
```

```json
{
  "query": "what is the storm in progress doctrine",
  "k": 10,
  "session_id": "matter-abc",
  "corpora": ["ny_case_law", "user_workspace"],
  "rerank": false,
  "retrievers": ["bm25", "dense"]
}
```

Response includes `results` (ranked chunks), `degraded` flag, `retrievers_used`.  
Add `"return_debug": true` to get per-leg ranks and latency breakdown.

### Ingest (user workspace)

```
POST /ingest
```

Uploads chunk into the user workspace for a given session. Large payloads (>1 MB) return a `job_id` for async polling.

```
GET /ingest/jobs/{job_id}
```

### Session document management

```
GET    /sessions/{session_id}/documents
DELETE /sessions/{session_id}/documents
DELETE /sessions/{session_id}/documents/{source}
```

### Graph endpoints

```
GET  /cases/{case_id}
GET  /cases/{case_id}/edges?direction=out&limit=50
POST /graph/subgraph
GET  /sessions/{session_id}/anchors?min_hits=2&limit=20&weight_by_pagerank=true
```

**Subgraph request:**

```json
{
  "seed_case_ids": [1117516, 2048931],
  "depth": 1,
  "include_external_neighbors": false,
  "max_neighbors_per_seed": 20
}
```

### Admin / monitoring

```
GET /admin/retrieval_stats?since=2026-04-01
```

Returns per-day retrieval volume (query count, chunk count, unique cases, session count, rank-1 ratio).

### Health

```
GET /health
```

Returns component status: `chroma`, `tantivy`, `relational`, `reranker`, `normalizer`.

---

## Logging

All logs are structured JSON by default (set `LOG_FORMAT=console` for dev).  
Each request includes an `X-Request-ID` header (generated if not provided by the caller) for log correlation.

Example log line:

```json
{
  "event": "retrieve",
  "request_id": "a1b2c3d4-...",
  "query_len": 48,
  "retrievers_used": ["bm25", "dense"],
  "degraded": false,
  "latency_ms": {"bm25": 12, "dense": 38, "rrf": 1, "total": 51},
  "level": "info",
  "timestamp": "2026-05-08T10:32:17.405Z"
}
```

---

## Running Tests

```bash
# Unit tests
.venv/bin/pytest tests/unit -q

# Integration tests (requires index files)
.venv/bin/pytest tests/integration -q

# All
.venv/bin/pytest tests/ -q
```
