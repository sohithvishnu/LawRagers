"""Replay frozen testset against the retriever; compute IR metrics + latency.

Per spec §10.2. Posts each query to the retriever /retrieve endpoint, collapses
returned chunks → case_id ranked list (first occurrence wins), scores with
`ranx` split by variant (single-removed / all-removed), reports p50/p95/p99
latency.

Usage:
    # Against the new retriever service (default):
    python eval/run_eval.py --label hybrid

    # Against the legacy api.py /search endpoint:
    python eval/run_eval.py --endpoint http://localhost:8000/search --legacy --label baseline

    # Limit queries for quick iteration:
    python eval/run_eval.py --label debug --limit 20
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

import httpx
import numpy as np
from ranx import Qrels, Run, evaluate

TESTSET = Path(__file__).parent / "testset"
REPORTS = Path(__file__).parent / "reports"

# Default endpoint: the retriever service (spec §10.2, spec §8.1)
DEFAULT_ENDPOINT = "http://localhost:8001/retrieve"
DEFAULT_SESSION = "eval-session"
K = 10
METRICS = [
    "precision@1", "precision@5", "precision@10",
    "recall@5", "recall@10",
    "mrr", "ndcg@10",
]


# ---------------------------------------------------------------------------
# Testset loading
# ---------------------------------------------------------------------------

def load_queries(testset: Path) -> list[dict]:
    with (testset / "queries.jsonl").open() as f:
        return [json.loads(line) for line in f if line.strip()]


def load_qrels(testset: Path) -> dict[str, dict[str, int]]:
    """Returns {query_id: {case_id_str: relevance}}."""
    qrels: dict[str, dict[str, int]] = {}
    with (testset / "qrels.tsv").open() as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 4:
                continue
            qid, _, doc, rel = parts
            qrels.setdefault(qid, {})[doc] = int(rel)
    return qrels


# ---------------------------------------------------------------------------
# Endpoint adapters
# ---------------------------------------------------------------------------

def query_retriever(
    client: httpx.Client,
    endpoint: str,
    session_id: str,
    query: str,
    corpora: list[str] | None = None,
    rerank: bool = False,
    retrievers: list[str] | None = None,
) -> tuple[list[str], float]:
    """Query the retriever service /retrieve endpoint (spec §8.1).

    Returns (ranked case_id list, latency_seconds).
    Collapses chunk results to case_ids preserving order (first occurrence wins).
    """
    payload = build_retriever_payload(
        session_id=session_id,
        query=query,
        corpora=corpora or ["ny_case_law"],
        rerank=rerank,
        retrievers=retrievers,
    )
    t0 = time.perf_counter()
    resp = client.post(endpoint, json=payload, timeout=60.0)
    elapsed = time.perf_counter() - t0
    resp.raise_for_status()

    return extract_ranked_case_ids(resp.json()), elapsed


def build_retriever_payload(
    *,
    session_id: str,
    query: str,
    corpora: list[str],
    rerank: bool,
    retrievers: list[str] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "query": query,
        "k": K,
        "session_id": session_id,
        "corpora": corpora,
        "rerank": rerank,
        "return_debug": False,
    }
    if retrievers is not None:
        payload["retrievers"] = retrievers
    return payload


def extract_ranked_case_ids(data: dict[str, Any]) -> list[str]:
    """Collapse chunk/case responses to a unique ranked case-id list.

    Supports:
      - current retriever contract: {"results": [{"case_id": ...}, ...]}
      - legacy api.py contract: {"cases": [{"case_id": ...}, ...]}
    """
    if "results" in data:
        items = data.get("results") or []
    elif "cases" in data:
        items = data.get("cases") or []
    else:
        raise ValueError(
            "Unsupported endpoint response contract: expected 'results' or 'cases'."
        )

    seen: list[str] = []
    for item in items:
        case_id = item.get("case_id")
        if case_id is None:
            continue
        sid = str(case_id)
        if sid not in seen:
            seen.append(sid)
    return seen


def query_legacy(
    client: httpx.Client,
    endpoint: str,
    session_id: str,
    query: str,
) -> tuple[list[str], float]:
    """Query the legacy api.py /search endpoint (old format)."""
    payload = {"session_id": session_id, "argument": query}
    t0 = time.perf_counter()
    resp = client.post(endpoint, json=payload, timeout=60.0)
    elapsed = time.perf_counter() - t0
    resp.raise_for_status()
    cases = resp.json().get("cases", [])
    seen: list[str] = []
    for c in cases:
        cid = c.get("case_id")
        if cid is None:
            continue
        sid = str(cid)
        if sid not in seen:
            seen.append(sid)
    return seen, elapsed


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def build_run(
    queries: list[dict],
    rankings: dict[str, list[str]],
) -> dict[str, dict[str, float]]:
    """ranx Run format: {qid: {doc_id: score}}. Score = 1/rank."""
    run: dict[str, dict[str, float]] = {}
    for q in queries:
        qid = q["query_id"]
        ranked = rankings.get(qid, [])
        run[qid] = {doc: 1.0 / (i + 1) for i, doc in enumerate(ranked[:K])}
    return run


def evaluate_variant(
    qrels: dict[str, dict[str, int]],
    run: dict[str, dict[str, float]],
    metrics: list[str],
) -> dict[str, float]:
    """Evaluate one variant, returning zeros if no documents were retrieved.

    ranx Run(...) raises on an all-empty run. For checkpointing, an all-empty
    retrieval result is a legitimate zero-recall outcome and should be reported
    as such rather than crashing the harness.
    """
    if not any(scores for scores in run.values()):
        return {metric: 0.0 for metric in metrics}
    return evaluate(Qrels(qrels), Run(run), metrics)


def filter_by_variant(
    d: dict,
    queries: list[dict],
    variant: str,
) -> dict:
    keep = {q["query_id"] for q in queries if q["variant"] == variant}
    return {k: v for k, v in d.items() if k in keep}


def percentile_ms(latencies: list[float], p: float) -> float:
    return float(np.percentile(latencies, p) * 1000) if latencies else 0.0


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

def render_report(
    label: str,
    endpoint: str,
    manifest: dict,
    per_variant: dict[str, dict],
    latencies: list[float],
    thresholds: dict | None = None,
    legs: str = "bm25+dense",
) -> str:
    """Render a markdown eval report.

    Thresholds (optional): spec §10.3 pass/fail markers.
    """
    thresholds = thresholds or {
        "recall@10": 0.70,
        "mrr": 0.50,
        "p95_latency_ms": 500,
    }

    lines = [
        f"# Eval report — {label}",
        "",
        f"- date: {dt.date.today().isoformat()}",
        f"- endpoint: `{endpoint}`",
        f"- retrievers: {legs}",
        f"- corpus: {manifest['corpus_size']} cases "
        f"(sha256 `{manifest['corpus_sha256'][:12]}…`)",
        f"- queries: {manifest['queries_total']} "
        f"({manifest['queries_single_removed']} single-removed, "
        f"{manifest['queries_all_removed']} all-removed)",
        f"- k: {K}",
        "",
        "## Latency (across all queries)",
        "",
        "| metric | ms | threshold | pass |",
        "|---|---|---|---|",
    ]
    p50 = percentile_ms(latencies, 50)
    p95 = percentile_ms(latencies, 95)
    p99 = percentile_ms(latencies, 99)
    p95_pass = "✓" if p95 <= thresholds["p95_latency_ms"] else "✗"
    lines += [
        f"| p50 | {p50:.1f} | — | — |",
        f"| p95 | {p95:.1f} | ≤{thresholds['p95_latency_ms']} | {p95_pass} |",
        f"| p99 | {p99:.1f} | — | — |",
        "",
    ]

    for variant, metrics in per_variant.items():
        headline = variant == "all-removed"  # spec §10.3: headline on all-removed
        lines += [
            f"## Retrieval — `{variant}`"
            + (" (headline — thresholds apply)" if headline else ""),
            "",
            "| metric | value | threshold | pass |",
            "|---|---|---|---|",
        ]
        for m, v in metrics.items():
            thresh = thresholds.get(m) if headline else None
            if thresh is not None:
                passes = "✓" if v >= thresh else "✗"
                lines.append(f"| {m} | {v:.4f} | ≥{thresh} | {passes} |")
            else:
                lines.append(f"| {m} | {v:.4f} | — | — |")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--testset", type=Path, default=TESTSET,
        help="Path to testset directory (default: eval/testset/).",
    )
    ap.add_argument(
        "--endpoint", default=DEFAULT_ENDPOINT,
        help=f"Retriever endpoint (default: {DEFAULT_ENDPOINT}).",
    )
    ap.add_argument(
        "--session-id", default=DEFAULT_SESSION,
        help="Session ID used for user_workspace queries (default: eval-session).",
    )
    ap.add_argument(
        "--label", default="baseline",
        help="Label for the report filename (default: baseline).",
    )
    ap.add_argument(
        "--legacy", action="store_true",
        help="Use legacy api.py /search payload format instead of /retrieve.",
    )
    ap.add_argument(
        "--corpora", nargs="+", default=["ny_case_law"],
        help="Corpora to query (default: ny_case_law).",
    )
    ap.add_argument(
        "--rerank", action="store_true",
        help="Enable reranking when querying /retrieve. Default is disabled for the Phase 4 checkpoint.",
    )
    ap.add_argument(
        "--retrievers", nargs="+", choices=["bm25", "dense"], default=None,
        metavar="LEG",
        help="Retrieval legs to activate (bm25, dense, or both). Default: both.",
    )
    ap.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of queries (for quick iteration).",
    )
    return ap


def main() -> int:
    ap = build_arg_parser()
    args = ap.parse_args()

    queries = load_queries(args.testset)
    qrels_raw = load_qrels(args.testset)
    manifest = json.loads((args.testset / "manifest.json").read_text())

    if args.limit:
        queries = queries[: args.limit]

    mode = "legacy" if args.legacy else "retriever"
    legs_desc = "+".join(args.retrievers) if args.retrievers else "bm25+dense"
    print(
        f"Running {len(queries)} queries against {args.endpoint} "
        f"[mode={mode}, corpora={args.corpora}, retrievers={legs_desc}]…"
    )

    rankings: dict[str, list[str]] = {}
    latencies: list[float] = []
    failures = 0

    with httpx.Client() as client:
        for i, q in enumerate(queries):
            try:
                if args.legacy:
                    ranked, elapsed = query_legacy(
                        client, args.endpoint, args.session_id, q["query_text"]
                    )
                else:
                    ranked, elapsed = query_retriever(
                        client, args.endpoint, args.session_id,
                        q["query_text"], args.corpora, args.rerank,
                        args.retrievers,
                    )
            except Exception as exc:
                failures += 1
                print(f"  [{q['query_id']}] FAIL: {exc}", file=sys.stderr)
                rankings[q["query_id"]] = []
                continue

            rankings[q["query_id"]] = ranked
            latencies.append(elapsed)

            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(queries)}")

    print(f"Done. failures={failures}")

    if failures == len(queries):
        print(
            "All eval queries failed before scoring. Check that the endpoint is running "
            "and that its response contract matches the selected mode.",
            file=sys.stderr,
        )
        return 2

    run_dict = build_run(queries, rankings)

    per_variant: dict[str, dict] = {}
    for variant in ("single-removed", "all-removed"):
        v_qrels = filter_by_variant(qrels_raw, queries, variant)
        v_run = filter_by_variant(run_dict, queries, variant)
        if not v_qrels:
            continue
        per_variant[variant] = evaluate_variant(v_qrels, v_run, METRICS)

    report = render_report(
        label=args.label,
        endpoint=args.endpoint,
        manifest=manifest,
        per_variant=per_variant,
        latencies=latencies,
        legs=legs_desc,
    )

    REPORTS.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS / f"{dt.date.today().isoformat()}-{args.label}.md"
    out_path.write_text(report)
    print(f"\nWrote {out_path}")
    print()
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
