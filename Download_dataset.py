import os
import requests
import zipfile
import json
from pathlib import Path

# --- Configuration ---
# Replace this with the actual base URL where the CAP ny3d zips are hosted
BASE_URL = "https://static.case.law/ny3d/"
TOTAL_VOLUMES = 29

# Directories for our data pipeline
DOWNLOAD_DIR = "data/raw_zips"
EXTRACT_DIR = "data/extracted_json"


def setup_directories():
    """Creates the necessary folders if they don't exist."""
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    print(f"📁 Directories ready: {DOWNLOAD_DIR} and {EXTRACT_DIR}")


def download_zips():
    """Downloads the zip files from volume 1 to TOTAL_VOLUMES."""
    for vol in range(1, TOTAL_VOLUMES + 1):
        file_name = f"{vol}.zip"
        url = f"{BASE_URL}{file_name}"
        save_path = os.path.join(DOWNLOAD_DIR, file_name)

        # Skip if already downloaded
        if os.path.exists(save_path):
            print(f"⏩ {file_name} already exists. Skipping download.")
            continue

        print(f"⬇️ Downloading {file_name} from {url}...")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Check for HTTP errors

            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"✅ Successfully downloaded {file_name}")
        except requests.exceptions.RequestException as e:
            print(f"❌ Failed to download {file_name}: {e}")


def unzip_data():
    """Extracts all downloaded zip files into the extraction directory."""
    for vol in range(1, TOTAL_VOLUMES + 1):
        zip_path = os.path.join(DOWNLOAD_DIR, f"{vol}.zip")
        extract_path = os.path.join(EXTRACT_DIR, f"vol_{vol}")

        if not os.path.exists(zip_path):
            continue

        print(f"📦 Extracting {vol}.zip...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)

    print("✅ All volumes extracted successfully.")


def preview_case_data():
    """Loads one case to verify the structure for our RAG setup."""
    print("\n--- 🔍 Data Preview ---")

    # Find the first json file in the extracted folder (assumes vol_1 exists)
    vol_1_path = Path(EXTRACT_DIR) / "vol_1"

    # CAP data is usually inside a nested 'data/data.jsonl' or individual json files inside the zip
    # We will search for any .json file to preview
    json_files = list(vol_1_path.rglob("*.json"))

    if not json_files:
        print("No JSON files found to preview yet.")
        return

    sample_file = json_files[0]

    with open(sample_file, 'r', encoding='utf-8') as f:
        case_data = json.load(f)

        # Print out the keys to see the structure we are working with
        print(f"Successfully loaded a sample case from {sample_file.name}")
        print(f"Available Metadata Keys: {list(case_data.keys())}")

        if 'name_abbreviation' in case_data:
            print(f"Case Name: {case_data['name_abbreviation']}")
        if 'decision_date' in case_data:
            print(f"Date: {case_data['decision_date']}")


if __name__ == "__main__":
    setup_directories()
    download_zips()
    unzip_data()
    preview_case_data()