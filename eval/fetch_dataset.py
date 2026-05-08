"""Fetch full CAP case JSONs from static.case.law into eval/dataset/.

Walks: /{reporter}/VolumesMetadata.json -> per-volume /{reporter}/{vol}/CasesMetadata.json
-> per-case /{reporter}/{vol}/cases/{file_name}.json. Saves canonical JSON
(including casebody, analysis, full cites_to with case_ids) to
eval/dataset/{reporter}_{vol}_{file_name}.json. Idempotent: skips files that
already exist unless --force is passed.

Usage:
    python eval/fetch_dataset.py --reporter abb-ct-app
    python eval/fetch_dataset.py --reporter abb-ct-app --volumes 1,2 --limit 50
    python eval/fetch_dataset.py --bootstrap   # close citation graph from existing dataset
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = "https://static.case.law"
DEST = Path(__file__).parent / "dataset"


UA = "LegalScribe-eval-fetcher/0.1 (+https://github.com/local)"


def fetch_json(url: str, retries: int = 3, backoff: float = 1.5) -> object:
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.load(resp)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(backoff ** attempt)
    raise RuntimeError(f"failed after {retries} attempts: {url}: {last_err}") from last_err


def list_volumes(reporter: str) -> list[str]:
    url = f"{BASE}/{reporter}/VolumesMetadata.json"
    data = fetch_json(url)
    # CAP volume entries carry volume_number; fall back to barcode-derived ordinal.
    vols = []
    for entry in data:
        v = entry.get("volume_number") or entry.get("volume_folder")
        if v is not None:
            vols.append(str(v))
    return vols


def list_cases(reporter: str, volume: str) -> list[dict]:
    url = f"{BASE}/{reporter}/{volume}/CasesMetadata.json"
    return fetch_json(url)


def case_url(reporter: str, volume: str, file_name: str) -> str:
    return f"{BASE}/{reporter}/{volume}/cases/{file_name}.json"


def parse_case_path(path: str) -> tuple[str, str, str] | None:
    """`/barb/10/0354-01` -> ('barb', '10', '0354-01'). None if malformed."""
    parts = path.strip("/").split("/")
    if len(parts) != 3:
        return None
    return parts[0], parts[1], parts[2]


def out_path_for(reporter: str, volume: str, file_name: str) -> Path:
    return DEST / f"{reporter}_{volume}_{file_name}.json"


def save_case(reporter: str, volume: str, file_name: str, force: bool) -> str:
    """Returns 'saved' | 'skipped' | 'failed'."""
    out = out_path_for(reporter, volume, file_name)
    if out.exists() and not force:
        return "skipped"
    try:
        doc = fetch_json(case_url(reporter, volume, file_name))
    except Exception as e:
        print(f"  [{reporter}/{volume}/{file_name}] FAIL: {e}", file=sys.stderr)
        return "failed"
    out.write_text(json.dumps(doc, indent=2))
    return "saved"


def collect_cited_paths(dataset_dir: Path) -> set[str]:
    """Read every JSON in dataset_dir, return all `cites_to[].case_paths`."""
    paths: set[str] = set()
    for f in sorted(dataset_dir.glob("*.json")):
        try:
            doc = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        for c in doc.get("cites_to", []) or []:
            for p in c.get("case_paths") or []:
                paths.add(p)
    return paths


def run_bootstrap(force: bool, sleep_s: float) -> tuple[int, int, int]:
    """Fetch every case path cited by the existing dataset. Returns (saved, skipped, failed)."""
    paths = collect_cited_paths(DEST)
    print(f"bootstrap: {len(paths)} cited case_paths discovered")
    saved = skipped = failed = 0
    for p in sorted(paths):
        parsed = parse_case_path(p)
        if parsed is None:
            print(f"  skip malformed path: {p}", file=sys.stderr)
            continue
        result = save_case(*parsed, force=force)
        if result == "saved":
            saved += 1
            time.sleep(sleep_s)
        elif result == "skipped":
            skipped += 1
        else:
            failed += 1
    return saved, skipped, failed


def run_walk(reporter: str, volumes: list[str] | None, limit: int | None, force: bool, sleep_s: float) -> tuple[int, int, int]:
    """Walk reporter -> volumes -> cases. Returns (saved, skipped, failed)."""
    if volumes is None:
        print(f"listing volumes for {reporter}...")
        volumes = list_volumes(reporter)
        print(f"  found {len(volumes)}: {volumes}")
    saved = skipped = failed = 0
    for vol in volumes:
        try:
            cases = list_cases(reporter, vol)
        except Exception as e:
            print(f"[vol {vol}] failed to list cases: {e}", file=sys.stderr)
            continue
        if limit:
            cases = cases[:limit]
        print(f"[vol {vol}] {len(cases)} cases")
        for c in cases:
            file_name = c.get("file_name")
            if not file_name:
                continue
            result = save_case(reporter, vol, file_name, force=force)
            if result == "saved":
                saved += 1
                time.sleep(sleep_s)
            elif result == "skipped":
                skipped += 1
            else:
                failed += 1
    return saved, skipped, failed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reporter", default="abb-ct-app")
    ap.add_argument("--volumes", help="comma-separated volume numbers; default all")
    ap.add_argument("--limit", type=int, help="max cases per volume (debug)")
    ap.add_argument("--force", action="store_true", help="overwrite existing files")
    ap.add_argument("--sleep", type=float, default=0.05, help="seconds between case fetches")
    ap.add_argument("--bootstrap", action="store_true",
                    help="instead of walking volumes, fetch every case path cited by the existing dataset")
    args = ap.parse_args()

    DEST.mkdir(parents=True, exist_ok=True)

    if args.bootstrap:
        saved, skipped, failed = run_bootstrap(force=args.force, sleep_s=args.sleep)
    else:
        volumes = [v.strip() for v in args.volumes.split(",")] if args.volumes else None
        saved, skipped, failed = run_walk(
            reporter=args.reporter, volumes=volumes, limit=args.limit,
            force=args.force, sleep_s=args.sleep,
        )

    print(f"\ndone. saved={saved} skipped={skipped} failed={failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
