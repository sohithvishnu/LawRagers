# Eval report — phase4-hybrid-no-rerank

- date: 2026-05-06
- endpoint: `http://localhost:8001/retrieve`
- corpus: 201 cases (sha256 `04000671e265…`)
- queries: 252 (126 single-removed, 126 all-removed)
- k: 10

## Latency (across all queries)

| metric | ms | threshold | pass |
|---|---|---|---|
| p50 | 40.8 | — | — |
| p95 | 61.7 | ≤500 | ✓ |
| p99 | 255.9 | — | — |

## Retrieval — `single-removed`

| metric | value | threshold | pass |
|---|---|---|---|
| precision@1 | 0.0000 | — | — |
| precision@5 | 0.0937 | — | — |
| precision@10 | 0.0468 | — | — |
| recall@5 | 0.4683 | — | — |
| recall@10 | 0.4683 | — | — |
| mrr | 0.1853 | — | — |
| ndcg@10 | 0.2565 | — | — |

## Retrieval — `all-removed` (headline — thresholds apply)

| metric | value | threshold | pass |
|---|---|---|---|
| precision@1 | 0.0000 | — | — |
| precision@5 | 0.0952 | — | — |
| precision@10 | 0.0476 | — | — |
| recall@5 | 0.4762 | — | — |
| recall@10 | 0.4762 | ≥0.7 | ✗ |
| mrr | 0.1923 | ≥0.5 | ✗ |
| ndcg@10 | 0.2640 | — | — |
