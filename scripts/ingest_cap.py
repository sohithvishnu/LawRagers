"""CAP case-law ingest pipeline — operator-level script (spec §5, §6.5).

Reads CAP JSON files from --input-dir, normalizes, chunks, embeds, and writes
to all three indexes (ChromaDB, Tantivy, SQLite citations) atomically via
DualIndexWriter.  Idempotent: cases already recorded as 'done' in ingestion_log
are skipped without re-processing.

Usage:
    python scripts/ingest_cap.py --input-dir eval/dataset/ [--config retriever_service/config.yaml] [--dry-run]

Options:
    --input-dir   Directory containing CAP *.json files (required).
    --config      Path to retriever_service config.yaml (default: retriever_service/config.yaml).
    --batch-size  Chunk batch size for embedding (default: 32).
    --dry-run     Parse and chunk but do not write to indexes.
    --verbose     DEBUG-level logging.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Iterable, Iterator, TypeVar

logger = logging.getLogger(__name__)
_T = TypeVar("_T")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_OCR_MIN = 0.3  # Pre-filter threshold (spec §6.5.1 step 1)
_CORPUS = "ny_case_law"


# ---------------------------------------------------------------------------
# CAP JSON helpers
# ---------------------------------------------------------------------------

def _load_case(path: Path) -> dict | None:
    try:
        return json.loads(path.read_bytes())
    except Exception as exc:
        logger.warning("Skipping %s: %s", path.name, exc)
        return None


def _ocr_confidence(case: dict) -> float:
    return float(case.get("analysis", {}).get("ocr_confidence") or 1.0)


def _pagerank_percentile(case: dict) -> float:
    pr = case.get("analysis", {}).get("pagerank", {})
    if isinstance(pr, dict):
        return float(pr.get("percentile") or 0.0)
    return 0.0


def _official_citation(case: dict) -> str:
    for c in case.get("citations", []):
        if c.get("type") == "official":
            return c.get("cite", "")
    cites = case.get("citations", [])
    return cites[0].get("cite", "") if cites else ""


def _cites_to_case_ids(case: dict) -> list[int]:
    ids: list[int] = []
    for entry in case.get("cites_to", []):
        for cid in entry.get("case_ids") or []:
            try:
                ids.append(int(cid))
            except (TypeError, ValueError):
                pass
    return ids


def _build_sections(case: dict) -> list[dict]:
    """Return a list of section dicts: {section_type, opinion_type, opinion_author, opinion_index, text}."""
    sections: list[dict] = []
    body = case.get("casebody", {})

    hm = body.get("head_matter") or ""
    if hm.strip():
        sections.append({
            "section_type": "head_matter",
            "opinion_type": "",
            "opinion_author": "",
            "opinion_index": 0,
            "text": hm,
        })

    for idx, op in enumerate(body.get("opinions", [])):
        text = op.get("text", "").strip()
        if not text:
            continue
        sections.append({
            "section_type": "opinion",
            "opinion_type": op.get("type", ""),
            "opinion_author": op.get("author", "") or "",
            "opinion_index": idx,
            "text": text,
        })

    return sections


def batched(items: Iterable[_T], size: int) -> Iterator[list[_T]]:
    if size <= 0:
        raise ValueError("batch size must be positive")

    batch: list[_T] = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


# ---------------------------------------------------------------------------
# Core ingest logic
# ---------------------------------------------------------------------------

def ingest_case(
    case: dict,
    writer,
    embedder,
    relational,
    normalize_fn,
    chunk_fn,
    chunk_id_fn,
    cfg,
    dry_run: bool,
) -> tuple[int, int]:
    """Ingest a single case.  Returns (chunks_indexed, chunks_skipped)."""
    case_id = int(case.get("id") or 0)
    if not case_id:
        logger.warning("Case has no id; skipping.")
        return 0, 0

    case_name = case.get("name_abbreviation", "Unknown")
    decision_date = case.get("decision_date", "")
    court_name = case.get("court", {}).get("name", "")
    jurisdiction = case.get("jurisdiction", {}).get("name", "")
    citation_official = _official_citation(case)
    pagerank = _pagerank_percentile(case)
    ocr_conf = _ocr_confidence(case)
    cites_ids = _cites_to_case_ids(case)

    # Idempotency: skip if all sections already ingested
    # We use the case_id as a sentinel target_id for the ingestion_log check.
    sentinel = str(case_id)
    if not dry_run and relational.is_ingested(_CORPUS, sentinel):
        logger.debug("Case %s already ingested; skipping.", case_id)
        return 0, 0

    sections = _build_sections(case)
    if not sections:
        logger.debug("Case %s has no sections; skipping.", case_id)
        return 0, 0

    indexed = 0
    skipped = 0

    log_row = None
    if not dry_run:
        log_row = relational.log_ingestion_start(_CORPUS, "chunk", sentinel)

    try:
        for section in sections:
            normalized = normalize_fn(section["text"])
            chunk_results = chunk_fn(
                section_text=normalized,
                section_type=section["section_type"],
                case_name=case_name,
                opinion_type=section["opinion_type"],
                cfg=cfg.chunking,
            )

            for cr in chunk_results:
                chunk_id = chunk_id_fn(cr.text, citation_official or case_name, "")

                metadata: dict = {
                    "case_id": case_id,
                    "case_name": case_name,
                    "citation_official": citation_official,
                    "decision_date": decision_date,
                    "court_name": court_name,
                    "jurisdiction": jurisdiction,
                    "section_type": section["section_type"],
                    "opinion_type": section["opinion_type"],
                    "opinion_author": section["opinion_author"],
                    "opinion_index": section["opinion_index"],
                    "chunk_idx": cr.chunk_idx,
                    "pagerank_percentile": pagerank,
                    "ocr_confidence": ocr_conf,
                    "cites_to_case_ids": cites_ids,
                    "good_law": True,
                    "source": f"cap/{case_id}",
                    "session_id": "",
                }

                if not dry_run:
                    writer.add_chunk_auto_embed(
                        chunk_id=chunk_id,
                        text_with_prefix=cr.text_with_prefix,
                        metadata=metadata,
                    )
                indexed += 1

        if not dry_run and cites_ids:
            writer.add_citation_edges(case_id, cites_ids)

        if not dry_run and log_row is not None:
            relational.log_ingestion_done(log_row)

    except Exception as exc:
        if not dry_run and log_row is not None:
            relational.log_ingestion_failed(log_row, str(exc))
        raise

    return indexed, skipped


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--input-dir",
        type=Path,
        required=False,
        default=Path(__file__).parent.parent / "eval" / "dataset/",
        help="Directory of CAP *.json files.",
    ), 
    ap.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent.parent / "retriever_service" / "config.yaml",
        help="Path to config.yaml.",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Embedding batch size (default 32).",
    )
    ap.add_argument(
        "--cases-per-batch",
        type=int,
        default=500,
        help="Number of cases to process per DualIndexWriter batch (default 500).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and chunk without writing to indexes.",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    return ap


def main() -> int:
    ap = build_arg_parser()
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    # --- Load config -------------------------------------------------------
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from retriever_service.config import load_settings
    cfg = load_settings(str(args.config) if args.config.exists() else None)

    # --- Imports after sys.path fix ----------------------------------------
    from retriever_service.normalize import normalize
    from retriever_service.ingestion.chunker import chunk_case_law
    from retriever_service.stores.chunk_ids import compute_chunk_id

    # --- Collect JSON files ------------------------------------------------
    input_dir = args.input_dir
    if not input_dir.is_dir():
        logger.error("--input-dir %s does not exist or is not a directory.", input_dir)
        return 1

    json_files = sorted(input_dir.glob("*.json"))
    logger.info("Found %d JSON files in %s", len(json_files), input_dir)
    if not json_files:
        logger.warning("No JSON files found; nothing to ingest.")
        return 0

    if args.dry_run:
        logger.info("DRY RUN — no writes to indexes.")

    relational = None
    embedder = None
    DualIndexWriter = None
    chroma = None
    tantivy = None

    if not args.dry_run:
        # --- Initialise stores -------------------------------------------------
        chroma_path = os.path.abspath(cfg.service.chroma_path)
        tantivy_path = os.path.abspath(cfg.service.tantivy_path)
        db_path = os.path.abspath(cfg.relational_store.database_path)

        from retriever_service.stores.chroma_store import ChromaStore
        from retriever_service.stores.tantivy_store import TantivyStore
        from retriever_service.stores.relational_store import RelationalStore

        chroma = ChromaStore(chroma_path=chroma_path, cfg=cfg)
        tantivy = TantivyStore(base_path=tantivy_path, cfg=cfg)
        relational = RelationalStore(db_path=db_path)

        from retriever_service.retrieval.dense_retriever import Embedder
        embedder = Embedder(model_name=cfg.chunking.tokenizer_model)

        from retriever_service.ingestion.dual_writer import DualIndexWriter

    # --- Ingest loop -------------------------------------------------------
    total_indexed = 0
    total_skipped = 0
    total_errors = 0
    total_filtered = 0

    try:
        from tqdm import tqdm
        file_iter = tqdm(json_files, desc="cases", unit="file")
    except ImportError:
        file_iter = json_files  # type: ignore[assignment]

    # Dry run: use a no-op writer context
    class _NullWriter:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def add_chunk_auto_embed(self, **kw): pass
        def add_citation_edges(self, *a): pass

    for path_batch in batched(file_iter, args.cases_per_batch):
        if args.dry_run:
            writer_cm = _NullWriter()
        else:
            writer_cm = DualIndexWriter(
                corpus=_CORPUS,
                embedder=embedder,
                chroma=chroma,
                tantivy=tantivy,
                relational=relational,
                batch_size=args.batch_size,
            )

        with writer_cm as writer:
            for path in path_batch:
                case = _load_case(path)
                if case is None:
                    total_errors += 1
                    continue

                if _ocr_confidence(case) < _OCR_MIN:
                    logger.debug("Filtered low-OCR case %s (conf=%.2f)", case.get("id"), _ocr_confidence(case))
                    total_filtered += 1
                    continue

                try:
                    indexed, skipped = ingest_case(
                        case=case,
                        writer=writer,
                        embedder=embedder,
                        relational=relational,
                        normalize_fn=normalize,
                        chunk_fn=chunk_case_law,
                        chunk_id_fn=compute_chunk_id,
                        cfg=cfg,
                        dry_run=args.dry_run,
                    )
                    total_indexed += indexed
                    total_skipped += skipped
                except Exception as exc:
                    logger.error("Failed to ingest case %s: %s", path.name, exc, exc_info=args.verbose)
                    total_errors += 1

    logger.info(
        "Ingest complete: indexed=%d  skipped=%d  filtered_low_ocr=%d  errors=%d",
        total_indexed, total_skipped, total_filtered, total_errors,
    )

    if not args.dry_run and tantivy is not None:
        try:
            tantivy.commit_all()
        except Exception:
            logger.warning("Error on final Tantivy commit_all.", exc_info=True)

    return 0 if total_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
