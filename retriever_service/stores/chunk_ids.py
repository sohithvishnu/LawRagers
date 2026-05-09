"""Deterministic chunk identifier computation.

ID formula (spec §4.4):
    md5(normalized_text + "|" + source + "|" + session_id).hexdigest()

Including session_id ensures the same paragraph uploaded to two different
matters produces two distinct IDs, even when textually identical.
"""

import hashlib


def compute_chunk_id(text: str, source: str, session_id: str = "") -> str:
    """Return a deterministic hex digest identifying a chunk.

    Re-ingesting the same content (text + source + session_id) produces the
    same ID, providing inherent deduplication across ingest runs.
    """
    payload = f"{text}|{source}|{session_id}"
    return hashlib.md5(payload.encode("utf-8")).hexdigest()
