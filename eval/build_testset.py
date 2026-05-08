"""CLERC-style testset builder. Derives queries + qrels from CAP `cites_to` graph.

Per spec §10.1. For each citing opinion in eval/dataset/, finds citation spans
with eyecite, matches them to `cites_to[].case_ids` whose target is also in
our corpus, builds ±150-word windows around the citation, and emits two
variants (single-removed, all-removed) per citation with the cite string
replaced by `[CITATION]`. Self-citations and out-of-corpus targets are dropped.

Qrels are at case_id granularity (eval runner collapses retrieved chunks →
case_ids before scoring).

Outputs:
    eval/testset/queries.jsonl   — {query_id, citing_case_id, variant, query_text, gold_case_id}
    eval/testset/qrels.tsv       — TREC: query_id 0 case_id 1
    eval/testset/manifest.json   — corpus hash + counts
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

from eyecite import get_citations

DATASET_DIR = Path(__file__).parent / "dataset"
OUT_DIR = Path(__file__).parent / "testset"
WINDOW_WORDS = 150
MAX_QUERIES_PER_CASE = 5


def load_corpus(dataset_dir: Path) -> dict[int, dict]:
    """Returns {case_id: case_json}."""
    corpus = {}
    for path in sorted(dataset_dir.glob("*.json")):
        try:
            doc = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        cid = doc.get("id")
        if cid is None:
            continue
        corpus[cid] = doc
    return corpus


def opinion_text(case: dict) -> str:
    parts = []
    for op in case.get("casebody", {}).get("opinions", []):
        t = op.get("text", "")
        if t:
            parts.append(t)
    return "\n\n".join(parts)


def normalize_cite(s: str) -> str:
    return re.sub(r"\s+", "", s).lower()


def word_window(text: str, char_start: int, char_end: int, n_words: int) -> tuple[str, int, int]:
    """Returns (window_text, window_char_start, window_char_end)."""
    before = text[:char_start]
    after = text[char_end:]
    before_words = before.split()
    after_words = after.split()
    pre = before_words[-n_words:] if before_words else []
    post = after_words[:n_words] if after_words else []
    pre_str = " ".join(pre)
    post_str = " ".join(post)
    citation = text[char_start:char_end]
    window = (pre_str + (" " if pre_str else "") + citation + (" " if post_str else "") + post_str).strip()
    # absolute char offsets for the window in the source text (approximate; pre/post split on whitespace)
    win_start = char_start - len(pre_str) - (1 if pre_str else 0)
    win_end = char_end + len(post_str) + (1 if post_str else 0)
    return window, max(0, win_start), min(len(text), win_end)


def build_queries_for_case(
    citing_id: int,
    case: dict,
    corpus_ids: set[int],
) -> list[dict]:
    text = opinion_text(case)
    if not text:
        return []
    cites_to = case.get("cites_to", [])
    # Map normalized CAP cite string -> set of in-corpus case_ids it points to
    cap_cite_to_targets: dict[str, set[int]] = {}
    for entry in cites_to:
        ids = entry.get("case_ids") or []
        in_corpus = {i for i in ids if i in corpus_ids and i != citing_id}
        if not in_corpus:
            continue
        # CAP entries may have just `cite`; some have `parallel_cites` etc. — keep simple.
        for cite_str in [entry.get("cite")]:
            if cite_str:
                cap_cite_to_targets.setdefault(normalize_cite(cite_str), set()).update(in_corpus)
    if not cap_cite_to_targets:
        return []

    # Extract all citations from the opinion with eyecite
    try:
        eyecites = get_citations(text)
    except Exception:
        return []

    queries = []
    # All citation spans (for all-removed variant masking)
    all_spans = []
    for c in eyecites:
        try:
            span = c.span()
            all_spans.append(span)
        except Exception:
            continue

    matched = []
    for c in eyecites:
        try:
            cstr = c.matched_text()
            span = c.span()
        except Exception:
            continue
        targets = cap_cite_to_targets.get(normalize_cite(cstr))
        if not targets:
            continue
        for tgt in targets:
            matched.append((span, cstr, tgt))

    # Cap per case
    matched = matched[:MAX_QUERIES_PER_CASE]

    for idx, (span, cstr, gold_id) in enumerate(matched):
        cstart, cend = span
        window, wstart, wend = word_window(text, cstart, cend, WINDOW_WORDS)

        # single-removed: replace ONLY the matched cite string in the window
        single = window.replace(cstr, "[CITATION]", 1)
        queries.append({
            "query_id": f"{citing_id}_{idx}_single",
            "citing_case_id": citing_id,
            "variant": "single-removed",
            "query_text": single,
            "gold_case_id": gold_id,
        })

        # all-removed: mask every citation span that falls within the window
        chars = list(text[wstart:wend])
        for s, e in all_spans:
            if s >= wstart and e <= wend:
                # blank out the cite chars; we'll collapse later
                for i in range(s - wstart, e - wstart):
                    chars[i] = "\x00"
        masked = "".join(chars)
        # collapse runs of NULs into [CITATION]
        masked = re.sub(r"\x00+", "[CITATION]", masked)
        queries.append({
            "query_id": f"{citing_id}_{idx}_all",
            "citing_case_id": citing_id,
            "variant": "all-removed",
            "query_text": masked.strip(),
            "gold_case_id": gold_id,
        })

    return queries


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, default=DATASET_DIR)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    corpus = load_corpus(args.dataset)
    corpus_ids = set(corpus.keys())
    print(f"corpus: {len(corpus)} cases")

    all_queries = []
    cases_with_queries = 0
    for cid, case in corpus.items():
        qs = build_queries_for_case(cid, case, corpus_ids)
        if qs:
            cases_with_queries += 1
        all_queries.extend(qs)

    args.out.mkdir(parents=True, exist_ok=True)

    with (args.out / "queries.jsonl").open("w") as f:
        for q in all_queries:
            f.write(json.dumps(q) + "\n")

    with (args.out / "qrels.tsv").open("w") as f:
        for q in all_queries:
            f.write(f"{q['query_id']}\t0\t{q['gold_case_id']}\t1\n")

    # corpus hash for reproducibility
    h = hashlib.sha256()
    for path in sorted(args.dataset.glob("*.json")):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    n_single = sum(1 for q in all_queries if q["variant"] == "single-removed")
    n_all = sum(1 for q in all_queries if q["variant"] == "all-removed")
    manifest = {
        "corpus_dir": str(args.dataset),
        "corpus_size": len(corpus),
        "corpus_sha256": h.hexdigest(),
        "queries_total": len(all_queries),
        "queries_single_removed": n_single,
        "queries_all_removed": n_all,
        "cases_with_queries": cases_with_queries,
        "window_words": WINDOW_WORDS,
        "max_queries_per_case": MAX_QUERIES_PER_CASE,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"queries: {len(all_queries)} ({n_single} single-removed, {n_all} all-removed)")
    print(f"cases with ≥1 query: {cases_with_queries} / {len(corpus)}")
    print(f"wrote {args.out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
