"""Legal text normalization (spec §6).

A single `normalize(text) -> str` function applied **identically** at
ingest time and query time.  Symmetry is mandatory — any divergence
silently hurts recall.

Pipeline order (spec §6):
  1. Unicode normalization: NFKC, ligature unfolding, smart-quote / em-dash folding.
  2. Citation canonicalization via eyecite → [CITE_<reporter>_<vol>_<page>] tokens.
  3. Section markers: § → "section", §§ → "sections", ¶ → "paragraph".
  4. Case names: Person v. Person → [CASE_<P1>_v_<P2>] tokens.
  5. Whitespace collapse.

[CITE_*] and [CASE_*] tokens are added to the Tantivy analyzer's protected-token
list so they pass through the English stemmer unmodified (see tantivy_store.py).
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional

# eyecite is an optional dependency; if absent, citation canonicalization is
# skipped gracefully (recall degrades slightly on citation-heavy queries).
try:
    from eyecite import get_citations
    from eyecite.models import FullCaseCitation
    _EYECITE_AVAILABLE = True
except ImportError:
    _EYECITE_AVAILABLE = False


# ---------------------------------------------------------------------------
# Step 1 — Unicode / typography normalisation
# ---------------------------------------------------------------------------

_SMART_QUOTE_MAP = str.maketrans({
    "‘": "'",   # LEFT SINGLE QUOTATION MARK
    "’": "'",   # RIGHT SINGLE QUOTATION MARK
    "“": '"',   # LEFT DOUBLE QUOTATION MARK
    "”": '"',   # RIGHT DOUBLE QUOTATION MARK
    "–": "-",   # EN DASH
    "—": " - ", # EM DASH
    "…": "...", # HORIZONTAL ELLIPSIS
    "­": "",    # SOFT HYPHEN (strip)
    "ﬁ": "fi",  # LATIN SMALL LIGATURE FI
    "ﬂ": "fl",  # LATIN SMALL LIGATURE FL
    "ﬃ": "ffi", # LATIN SMALL LIGATURE FFI
    "ﬄ": "ffl", # LATIN SMALL LIGATURE FFL
})


def _unicode_normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_SMART_QUOTE_MAP)
    return text


# ---------------------------------------------------------------------------
# Step 2 — Citation canonicalization
# ---------------------------------------------------------------------------

def _canonicalize_citations(text: str) -> str:
    """Replace eyecite-detected citations with stable [CITE_*] tokens.

    The token format is:  [CITE_<REPORTER>_<VOLUME>_<PAGE>]
    where REPORTER has spaces replaced with underscores and is uppercased.
    Unreported or partial citations (no reporter/volume/page) are left as-is.
    """
    if not _EYECITE_AVAILABLE:
        return text

    try:
        citations = get_citations(text)
    except Exception:
        return text

    # Sort by position descending so replacements don't shift offsets.
    # eyecite FullCaseCitation carries span info.
    resolved: list[tuple[int, int, str]] = []
    for citation in citations:
        if not isinstance(citation, FullCaseCitation):
            continue
        try:
            token = citation.token
            reporter = getattr(token, "reporter", None)
            volume = getattr(token, "volume", None)
            page = getattr(token, "page", None)
            if not (reporter and volume and page):
                continue
            safe_reporter = re.sub(r"[^A-Za-z0-9]", "_", str(reporter)).upper()
            cite_token = f"[CITE_{safe_reporter}_{volume}_{page}]"
            span = token.span()
            resolved.append((span[0], span[1], cite_token))
        except Exception:
            continue

    if not resolved:
        return text

    resolved.sort(key=lambda t: t[0], reverse=True)
    chars = list(text)
    for start, end, replacement in resolved:
        chars[start:end] = list(replacement)
    return "".join(chars)


# ---------------------------------------------------------------------------
# Step 3 — Section markers
# ---------------------------------------------------------------------------

_SECTION_DOUBLE_RE = re.compile(r"§§\s*")
_SECTION_SINGLE_RE = re.compile(r"§\s*")
_PILCROW_RE = re.compile(r"¶\s*")


def _normalize_section_markers(text: str) -> str:
    text = _SECTION_DOUBLE_RE.sub(" sections ", text)
    text = _SECTION_SINGLE_RE.sub(" section ", text)
    text = _PILCROW_RE.sub(" paragraph ", text)
    return text


# ---------------------------------------------------------------------------
# Step 4 — Case-name canonicalization
# ---------------------------------------------------------------------------

# Matches "Lastname v. Lastname" style patterns (spec §6 step 4).
# Allows hyphens, apostrophes, periods in name components (e.g. "O'Brien", "St. Paul").
_NAME_SEGMENT = r"[A-Z][A-Za-z\.\-']+"
_CASE_NAME_RE = re.compile(
    rf"\b({_NAME_SEGMENT}(?:\s+{_NAME_SEGMENT})*)\s+v\.\s+({_NAME_SEGMENT}(?:\s+{_NAME_SEGMENT})*)\b"
)


def _sanitize_name_part(name: str) -> str:
    """Convert a case party name to a safe token fragment."""
    return re.sub(r"[^A-Za-z0-9]", "_", name).strip("_")


def _canonicalize_case_names(text: str) -> str:
    def _replace(m: re.Match) -> str:
        p1 = _sanitize_name_part(m.group(1))
        p2 = _sanitize_name_part(m.group(2))
        return f"[CASE_{p1}_v_{p2}]"

    return _CASE_NAME_RE.sub(_replace, text)


# ---------------------------------------------------------------------------
# Step 5 — Whitespace collapse
# ---------------------------------------------------------------------------

_MULTI_WHITESPACE_RE = re.compile(r"[ \t]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


def _collapse_whitespace(text: str) -> str:
    text = _MULTI_WHITESPACE_RE.sub(" ", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def normalize(text: str) -> str:
    """Apply the full legal normalization pipeline to a text string.

    Must be called with identical arguments at ingest time and query time.
    """
    text = _unicode_normalize(text)
    text = _canonicalize_citations(text)
    text = _normalize_section_markers(text)
    text = _canonicalize_case_names(text)
    text = _collapse_whitespace(text)
    return text


def normalize_query(query: str) -> str:
    """Normalize a retrieval query.

    Identical to normalize() — a separate entry-point to make the call-site
    intent explicit and to allow query-specific overrides in future if needed.
    """
    return normalize(query)
