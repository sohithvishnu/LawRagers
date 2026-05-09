import os

# --- STOP TENSORFLOW INTERFERENCE ---
os.environ["USE_TF"] = "NO"
os.environ["USE_TORCH"] = "YES"

from indexer import IndexConfig, stream_index

# ---------------------------------------------------------------------------
# CLI entry point — configure the run here and execute with:
#   python build_index.py
#
# To add a new jurisdiction later, copy the config block, change
# collection_name and data_dir, and run again.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    config = IndexConfig(
        collection_name="ny_case_law",
        data_dir="data/extracted_json",
        batch_size=500,
        rebuild=True,
    )

    print(f"⚡️  Indexing '{config.collection_name}' from '{config.data_dir}' ...")
    print(f"     batch_size={config.batch_size}  rebuild={config.rebuild}\n")

    for update in stream_index(config):
        progress = update["progress"]
        status   = update["status"]

        if status == "complete":
            print(
                f"\n🎉  Done!  "
                f"{update['cases_processed']} cases → "
                f"{update['chunks_indexed']} chunks "
                f"in collection '{update['collection']}'"
            )
        else:
            print(f"[{progress:3d}%]  {status}")
