import os

# --- STOP TENSORFLOW INTERFERENCE ---
os.environ["USE_TF"] = "NO"
os.environ["USE_TORCH"] = "YES"

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, Optional

import chromadb
from chromadb.utils import embedding_functions
import torch


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

@dataclass
class IndexConfig:
    """All parameters that control a single indexing run."""
    collection_name: str
    data_dir: str
    batch_size: int = 500
    embed_device: Optional[str] = None
    rebuild: bool = True
    chunk_min_len: int = 100
    chroma_path: str = "./chroma_db"


# ---------------------------------------------------------------------------
# SHARED HELPERS  (imported by api.py so both use identical behaviour)
# ---------------------------------------------------------------------------

def get_device(override: Optional[str] = None) -> str:
    """Return the best available compute device."""
    if override:
        return override
    return "mps" if torch.backends.mps.is_available() else "cpu"


def make_embedding_function(device: str):
    """Build a SentenceTransformer embedding function pinned to the given device."""
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2",
        device=device,
    )


def chunk_text(text: str, min_len: int = 100) -> list[str]:
    """
    Split text at paragraph boundaries and discard short fragments.
    Used by both the offline indexer and the live upload handler so
    chunking is consistent across all collections.
    """
    return [p.strip() for p in text.split("\n") if len(p.strip()) >= min_len]


def doc_id(case_name: str, decision_date: str, chunk: str) -> str:
    """
    Deterministic, content-addressed document ID.
    Safe to re-index: upsert on the same ID is a no-op.
    """
    key = f"{case_name}|{decision_date}|{chunk[:128]}"
    return hashlib.sha256(key.encode()).hexdigest()[:20]


# ---------------------------------------------------------------------------
# INTERNAL UTILITIES
# ---------------------------------------------------------------------------

def _path_metadata(file_path: Path, data_dir: Path) -> dict:
    """
    Best-effort extraction of reporter and volume from the relative path.
    Handles two common layouts produced by the download scripts:
      <data_dir>/<reporter>/vol_N/<case>.json   →  reporter=ny3d, volume=5
      <data_dir>/vol_N/<case>.json              →  reporter=unknown, volume=5
    """
    try:
        parts = file_path.relative_to(data_dir).parts
        if len(parts) >= 3:
            return {
                "reporter": parts[0],
                "volume": parts[1].replace("vol_", ""),
            }
        if len(parts) == 2:
            return {
                "reporter": "unknown",
                "volume": parts[0].replace("vol_", ""),
            }
    except ValueError:
        pass
    return {"reporter": "unknown", "volume": "unknown"}


# ---------------------------------------------------------------------------
# CORE INDEXING PIPELINE
# ---------------------------------------------------------------------------

def stream_index(config: IndexConfig) -> Generator[dict, None, None]:
    """
    Index Harvard CAP-format JSON case files into a ChromaDB collection.

    Yields progress dicts so callers can stream updates to a UI or CLI:
        {"progress": 0-100, "total": N, "status": "..."}
    Final yield includes summary fields:
        {"progress": 100, ..., "status": "complete",
         "cases_processed": N, "chunks_indexed": N, "collection": "..."}
    """

    device = get_device(config.embed_device)
    yield {"progress": 0, "total": 0, "status": f"Initializing on {device.upper()}..."}

    ef = make_embedding_function(device)
    client = chromadb.PersistentClient(path=config.chroma_path)

    if config.rebuild:
        try:
            client.delete_collection(config.collection_name)
            yield {"progress": 0, "total": 0, "status": f"Cleared existing '{config.collection_name}'."}
        except Exception:
            pass
        collection = client.create_collection(
            name=config.collection_name,
            embedding_function=ef,
            metadata={
                "hnsw:space": "cosine",
                "hnsw:M": 16,
                "hnsw:construction_ef": 200,
                "hnsw:search_ef": 100,
            },
        )
    else:
        collection = client.get_or_create_collection(
            name=config.collection_name,
            embedding_function=ef,
        )

    # --- Discover files ---
    data_dir = Path(config.data_dir)
    json_files = [f for f in data_dir.rglob("*.json") if "Metadata" not in f.name]
    total_files = len(json_files)

    if total_files == 0:
        yield {"progress": 100, "total": 0, "status": "No JSON files found.", "complete": True}
        return

    yield {
        "progress": 0,
        "total": total_files,
        "status": f"Found {total_files} case files. Starting...",
    }

    # --- Batch state ---
    batch_docs: list[str] = []
    batch_meta: list[dict] = []
    batch_ids: list[str] = []
    cases_processed = 0
    chunks_total = 0
    files_done = 0

    def flush() -> None:
        nonlocal chunks_total
        if not batch_docs:
            return
        # Deduplicate by ID within the batch to avoid ChromaDB DuplicateIDError
        seen: dict[str, int] = {}
        for i, id_ in enumerate(batch_ids):
            if id_ not in seen:
                seen[id_] = i
        unique = list(seen.values())
        docs  = [batch_docs[i] for i in unique]
        metas = [batch_meta[i] for i in unique]
        ids   = [batch_ids[i]  for i in unique]
        collection.upsert(documents=docs, metadatas=metas, ids=ids)
        chunks_total += len(docs)
        batch_docs.clear()
        batch_meta.clear()
        batch_ids.clear()

    # --- Main loop: one file at a time ---
    for file_path in json_files:
        try:
            case_data = json.loads(file_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            files_done += 1
            continue

        case_id       = case_data.get("id")
        case_name     = case_data.get("name_abbreviation", "Unknown Case")
        decision_date = case_data.get("decision_date", "Unknown Date")
        opinions      = case_data.get("casebody", {}).get("opinions", [])

        if not opinions:
            files_done += 1
            continue

        cases_processed += 1
        path_meta = _path_metadata(file_path, data_dir)

        for opinion in opinions:
            opinion_type = opinion.get("type", "majority")
            for chunk in chunk_text(opinion.get("text", ""), config.chunk_min_len):
                batch_docs.append(chunk)
                batch_meta.append({
                    "case_id":       str(case_id) if case_id is not None else "",
                    "case_name":     case_name,
                    "decision_date": decision_date,
                    "source_file":   file_path.name,
                    "reporter":      path_meta["reporter"],
                    "volume":        path_meta["volume"],
                    "opinion_type":  opinion_type,
                })
                batch_ids.append(doc_id(case_name, decision_date, chunk))

                if len(batch_docs) >= config.batch_size:
                    flush()

        files_done += 1
        yield {
            "progress": round((files_done / total_files) * 100),
            "total": total_files,
            "status": f"{files_done}/{total_files} files — {chunks_total} chunks indexed",
        }

    flush()

    yield {
        "progress": 100,
        "total": total_files,
        "status": "complete",
        "cases_processed": cases_processed,
        "chunks_indexed": chunks_total,
        "collection": config.collection_name,
    }
