import os

# --- 🛑 STOP TENSORFLOW INTERFERENCE ---
os.environ["USE_TF"] = "NO"
os.environ["USE_TORCH"] = "YES"

import json
import chromadb
from chromadb.utils import embedding_functions
import torch
from pathlib import Path

# --- 1. Setup MPS (Apple Silicon GPU Acceleration) ---
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"⚡️ Using device for embeddings: {device.upper()}")

# Explicitly load the default Chroma model onto the Mac's GPU
mps_embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2",
    device=device
)

# --- 2. Setup ChromaDB ---
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Delete the old collection so we start with a completely clean slate
try:
    chroma_client.delete_collection("ny_case_law")
    print("🧹 Cleared old database collection.")
except Exception:
    pass

# Create collection WITH the hardware-accelerated embedding function
collection = chroma_client.create_collection(
    name="ny_case_law",
    embedding_function=mps_embedding_function,
    metadata={"hnsw:space": "cosine",
            "hnsw:M": 16,                                                      
            "hnsw:construction_ef": 200,                                                                                                                                                          
            "hnsw:search_ef": 100
            }
)

EXTRACT_DIR = os.environ.get("CASES_DIR", "eval/dataset")


def process_and_index_cases():
    print("⏳ Starting to index cases into ChromaDB...")

    json_files = list(Path(EXTRACT_DIR).rglob("*.json"))

    docs_to_insert = []
    metadatas_to_insert = []
    ids_to_insert = []

    doc_id_counter = 1
    cases_processed = 0

    for file_path in json_files:
        # Skip Harvard's volume metadata files, we only want the cases
        if "Metadata" in file_path.name:
            continue

        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                case_data = json.load(f)
            except json.JSONDecodeError:
                continue  # Skip if it's not valid JSON

            # Extract basic metadata
            case_id = case_data.get("id")
            case_name = case_data.get("name_abbreviation", "Unknown Case")
            decision_date = case_data.get("decision_date", "Unknown Date")

            # THE EXACT PATH based on your JSON snippet:
            opinions = case_data.get("casebody", {}).get("opinions", [])

            if not opinions:
                continue

            cases_processed += 1

            for opinion in opinions:
                opinion_text = opinion.get("text", "")

                # CHUNKING STRATEGY: Split by single newline (\n) based on your JSON format
                # We filter out very short chunks (like "OPINION OF THE COURT")
                paragraphs = [p.strip() for p in opinion_text.split("\n") if len(p.strip()) > 100]

                for para in paragraphs:
                    docs_to_insert.append(para)
                    metadatas_to_insert.append({
                        "case_id": case_id,
                        "case_name": case_name,
                        "decision_date": decision_date,
                        "source_file": str(file_path.name)
                    })
                    ids_to_insert.append(f"doc_{doc_id_counter}")
                    doc_id_counter += 1

    print(f"✅ Parsed {cases_processed} total cases.")

    # Batch insert into ChromaDB
    if docs_to_insert:
        batch_size = 5000
        for i in range(0, len(docs_to_insert), batch_size):
            collection.add(
                documents=docs_to_insert[i:i + batch_size],
                metadatas=metadatas_to_insert[i:i + batch_size],
                ids=ids_to_insert[i:i + batch_size]
            )
            print(f"💾 Indexed batch {i} to {i + batch_size} into database...")

    print(f"🎉 Complete! Successfully indexed {len(docs_to_insert)} case law paragraphs.")


if __name__ == "__main__":
    process_and_index_cases()