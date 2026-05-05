import os
import requests
import zipfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Enterprise Configuration: The Mega Payload ---
REPORTERS = {
    "ny3d": 29,             # New York 3rd
    "cal-4th": 63,          # California 4th
    "cal-app-4th": 248,     # California Appellate 4th
    "ill-2d": 50,           # Illinois 2nd
    "mass": 50,             # Massachusetts Reports
    "f3d": 999,             # Federal 3rd Circuit 
    "f2d": 999              # Federal 2nd Circuit 

}

BASE_URL = "https://static.case.law/"
DOWNLOAD_DIR = "data/raw_zips"
EXTRACT_DIR = "data/extracted_json"

def setup_directories():
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    print(f"📁 Directories ready: {DOWNLOAD_DIR} and {EXTRACT_DIR}")

def process_single_volume(reporter, vol):
    """
    The complete asynchronous lifecycle for a single volume:
    Check -> Download -> Extract -> Delete ZIP.
    """
    file_name = f"{reporter}_{vol}.zip"
    url = f"{BASE_URL}{reporter}/{vol}.zip"
    save_path = os.path.join(DOWNLOAD_DIR, file_name)
    extract_path = os.path.join(EXTRACT_DIR, reporter, f"vol_{vol}")

    # 1. Verification: Skip if already successfully extracted
    if os.path.exists(extract_path):
        # Check if the directory is actually populated with files
        try:
            if any(os.scandir(extract_path)):
                return f"⏩ SKIPPED: {reporter.upper()} Vol {vol} (Already fully extracted)"
        except FileNotFoundError:
            pass

    # 2. Download the ZIP
    try:
        response = requests.get(url, stream=True, timeout=20)
        if response.status_code == 404:
            return f"⚠️ NOT FOUND: {reporter.upper()} Vol {vol} (Skipping 404)"
        response.raise_for_status()

        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    except Exception as e:
        return f"❌ DOWNLOAD ERROR: {reporter.upper()} Vol {vol} - {str(e)}"

    # 3. Extract the ZIP
    try:
        os.makedirs(extract_path, exist_ok=True)
        with zipfile.ZipFile(save_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
    except zipfile.BadZipFile:
        # If corrupt, delete the bad zip and fail gracefully
        if os.path.exists(save_path):
            os.remove(save_path)
        return f"❌ CORRUPT ZIP: {reporter.upper()} Vol {vol} (Deleted bad file)"
    except Exception as e:
        return f"❌ EXTRACTION ERROR: {reporter.upper()} Vol {vol} - {str(e)}"

    # 4. Clean up: Delete the ZIP to save massive disk space
    try:
        if os.path.exists(save_path):
            os.remove(save_path)
    except Exception as e:
        return f"⚠️ CLEANUP WARNING: Could not delete {file_name} - {str(e)}"

    return f"✅ SUCCESS: {reporter.upper()} Vol {vol} (Downloaded ➔ Extracted ➔ Cleaned)"


def execute_pipeline_concurrently():
    """Uses ThreadPoolExecutor to run the assembly line on multiple files at once."""
    tasks = []
    for reporter, volumes in REPORTERS.items():
        for vol in range(1, volumes + 1):
            tasks.append((reporter, vol))
            
    print(f"🚀 Starting continuous assembly line for {len(tasks)} volumes...")
    print("   [Download -> Extract -> Delete] will happen simultaneously to save disk space.\n")

    # 8 workers will process 8 volumes start-to-finish concurrently.
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_single_volume, rep, vol): (rep, vol) for rep, vol in tasks}
        
        for future in as_completed(futures):
            print(future.result())

def count_extracted_documents():
    """Scans the hard drive and provides the final dataset length."""
    print("\n🧮 Tallying total documents across all jurisdictions...")
    total_cases = 0
    reporter_counts = {}

    for reporter_dir in Path(EXTRACT_DIR).iterdir():
        if reporter_dir.is_dir():
            reporter_name = reporter_dir.name
            # Count all .json cases, explicitly ignoring Harvard's metadata/volume markers
            count = sum(1 for _ in reporter_dir.rglob("*.json") if "Metadata" not in _.name)
            
            if count > 0:
                reporter_counts[reporter_name] = count
                total_cases += count

    print("\n===========================================")
    print("      📊 FINAL DATA PIPELINE SUMMARY       ")
    print("===========================================")
    for reporter, count in sorted(reporter_counts.items()):
        print(f" 📁 {reporter.upper():<15} : {count:>10,} cases")
    print("-------------------------------------------")
    print(f" 🏆 TOTAL DATASET : {total_cases:>10,} cases ready")
    print("===========================================\n")

if __name__ == "__main__":
    setup_directories()
    execute_pipeline_concurrently()
    count_extracted_documents()