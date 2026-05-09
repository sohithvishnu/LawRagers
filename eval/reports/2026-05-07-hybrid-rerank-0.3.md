# Eval report — hybrid-rerank-0.3

- date: 2026-05-07
- endpoint: `http://localhost:8001/retrieve`
- corpus: 201 cases (sha256 `04000671e265…`)
- queries: 252 (126 single-removed, 126 all-removed)
- k: 10

## Latency (across all queries)

| metric | ms | threshold | pass |
|---|---|---|---|
| p50 | 1139.4 | — | — |
| p95 | 1398.1 | ≤500 | ✗ |
| p99 | 1506.8 | — | — |

## Retrieval — `single-removed`

| metric | value | threshold | pass |
|---|---|---|---|
| precision@1 | 0.0000 | — | — |
| precision@5 | 0.0635 | — | — |
| precision@10 | 0.0341 | — | — |
| recall@5 | 0.3175 | — | — |
| recall@10 | 0.3413 | — | — |
| mrr | 0.1242 | — | — |
| ndcg@10 | 0.1781 | — | — |

## Retrieval — `all-removed` (headline — thresholds apply)

| metric | value | threshold | pass |
|---|---|---|---|
| precision@1 | 0.0000 | — | — |
| precision@5 | 0.0714 | — | — |
| precision@10 | 0.0381 | — | — |
| recall@5 | 0.3571 | — | — |
| recall@10 | 0.3810 | ≥0.7 | ✗ |
| mrr | 0.1390 | ≥0.5 | ✗ |
| ndcg@10 | 0.1991 | — | — |
