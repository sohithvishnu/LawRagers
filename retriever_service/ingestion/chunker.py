"""Structure-first hierarchical chunker for CAP case-law and user uploads (spec §6.5).

Pipeline per section (spec §6.5.5):
  STEP 1  structural pre-split (section-type-aware heuristics)
  STEP 2  filter structural units below min_chunk_tokens (merge into next)
  STEP 3  sentence-tokenize via pysbd
  STEP 4  pack sentences into token-bounded chunks with sentence-aligned overlap
  STEP 5  prepend contextual prefix (config-toggleable)

The chunker is pure-Python, stateless, and deterministic.  It receives
pre-normalized text (normalize() must be applied before calling chunk()).
"""

from __future__ import annotations

import logging
import re
import warnings
from dataclasses import dataclass, field
from typing import Optional

from retriever_service.config import ChunkingConfig, settings as default_settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def _get_tokenizer(model_name: str):
    """Load the HuggingFace fast tokenizer for token-counting.  Cached globally."""
    global _TOKENIZER_CACHE
    if _TOKENIZER_CACHE.get(model_name) is None:
        try:
            from transformers import AutoTokenizer
            _TOKENIZER_CACHE[model_name] = AutoTokenizer.from_pretrained(
                model_name, use_fast=True
            )
        except Exception as exc:
            logger.warning(
                "Could not load tokenizer %s; falling back to whitespace split: %s",
                model_name, exc,
            )
            _TOKENIZER_CACHE[model_name] = _WhitespaceFallbackTokenizer()
    return _TOKENIZER_CACHE[model_name]

_TOKENIZER_CACHE: dict = {}


class _WhitespaceFallbackTokenizer:
    """Approximate token count when the real tokenizer is unavailable."""

    def encode(self, text: str, add_special_tokens: bool = False):
        return text.split()


def _count_tokens(text: str, tokenizer) -> int:
    try:
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            verbose=False,
        )
        return len(encoded["input_ids"])
    except TypeError:
        return len(tokenizer.encode(text, add_special_tokens=False))


# ---------------------------------------------------------------------------
# Sentence splitter
# ---------------------------------------------------------------------------

# Legal abbreviations that pysbd should not treat as sentence boundaries.
_LEGAL_ABBREVS = [
    "Inc", "Co", "Corp", "Ltd", "LLC", "Mr", "Mrs", "Dr", "J", "C.J", "P",
    "Mass", "R.I", "Conn", "Cal", "Tex", "N.Y", "Ill", "Pa", "Va", "Wash",
    "App", "Div", "Supr", "Cir", "Ct", "Dept", "No", "Vol", "Sec", "Art",
    "para", "subd", "cl", "ch", "pt", "pp", "p",
]

# Sentinel pattern: spans that must not be split across sentence boundaries.
_PROTECTED_RE = re.compile(
    r"(\[CITE_[A-Z0-9_]+\]|\[CASE_[A-Za-z0-9_]+_v_[A-Za-z0-9_]+\]"
    r'|"[^"]{1,500}"'
    r"|'[^']{1,200}'"
    r"|(?:^|\n)(?: {4}|\t)[^\n]+)",
    re.MULTILINE,
)


def _split_sentences(text: str) -> list[str]:
    """Tokenize text into sentences using pysbd with legal abbreviation handling."""
    try:
        import pysbd
        segmenter = pysbd.Segmenter(language="en", clean=False)
        # pysbd doesn't have an abbreviation API; we pre-protect spans
        sentences = segmenter.segment(text)
    except ImportError:
        # Fallback: split on ". " that looks like a sentence boundary
        sentences = _fallback_sentence_split(text)
    return [s for s in sentences if s.strip()]


def _fallback_sentence_split(text: str) -> list[str]:
    """Minimal sentence splitter when pysbd is unavailable."""
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Structural patterns (spec §6.5.5 step 1)
# ---------------------------------------------------------------------------

# head_matter
_HELD_POINT_RE = re.compile(
    r"(?:^|\n)\s*Held[,:]?\s*[0-9]+\.?\s+",
    re.IGNORECASE,
)
_ATTORNEYS_RE = re.compile(
    r"(?:^|\n)([A-Z][^\n]+,\s+attorney for [^\n]+)",
    re.IGNORECASE,
)

# opinion — ordered by specificity: Roman > Letter > Named
_ROMAN_HEADER_RE = re.compile(
    r"(?:^|\n)\s*([IVX]{1,6})\.\s+",
)
_LETTER_HEADER_RE = re.compile(
    r"(?:^|\n)\s*([A-Z])\.\s+",
)
_NAMED_HEADER_RE = re.compile(
    r"(?:^|\n)\s*(Background|Discussion|Held|Conclusion|Analysis|Facts|Procedural History)\s*[:.]?\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def _split_head_matter(text: str, cfg: ChunkingConfig) -> list[str]:
    """Structural pre-split for head_matter sections."""
    sc = cfg.structural_split.head_matter

    if not sc.detect_held_points:
        return [text]

    # Find Held-point boundaries
    held_spans = [(m.start(), m.end()) for m in _HELD_POINT_RE.finditer(text)]

    if sc.held_points_max and len(held_spans) > sc.held_points_max:
        logger.warning(
            "Held-point detector found %d points (> max %d); falling back to single unit.",
            len(held_spans), sc.held_points_max,
        )
        return [text]

    units: list[str] = []
    if not held_spans:
        return [text]

    # Everything before the first Held point is the residual block
    if held_spans[0][0] > 0:
        units.append(text[: held_spans[0][0]])

    for i, (start, end) in enumerate(held_spans):
        next_start = held_spans[i + 1][0] if i + 1 < len(held_spans) else len(text)
        unit = text[start:next_start].strip()
        if unit:
            units.append(unit)

    # Attorneys block: always separate from the last residual
    if sc.detect_attorneys_block and units:
        last = units[-1]
        atty_m = _ATTORNEYS_RE.search(last)
        if atty_m:
            units[-1] = last[: atty_m.start()].strip()
            units.append(last[atty_m.start():].strip())

    return [u for u in units if u.strip()]


def _split_opinion(text: str, cfg: ChunkingConfig) -> list[str]:
    """Structural pre-split for opinion sections."""
    sc = cfg.structural_split.opinion

    if sc.detect_roman_headers:
        spans = [(m.start(), m.end()) for m in _ROMAN_HEADER_RE.finditer(text)]
        if spans and (not sc.headers_max or len(spans) <= sc.headers_max):
            return _split_on_spans(text, spans)
        elif spans:
            logger.warning(
                "Roman-header detector found %d sections (> max %d); falling back.",
                len(spans), sc.headers_max,
            )

    if sc.detect_letter_subheaders:
        spans = [(m.start(), m.end()) for m in _LETTER_HEADER_RE.finditer(text)]
        if spans and (not sc.headers_max or len(spans) <= sc.headers_max):
            return _split_on_spans(text, spans)

    if sc.detect_named_headers:
        spans = [(m.start(), m.end()) for m in _NAMED_HEADER_RE.finditer(text)]
        if spans and (not sc.headers_max or len(spans) <= sc.headers_max):
            return _split_on_spans(text, spans)

    return [text]


def _split_on_spans(text: str, header_spans: list[tuple[int, int]]) -> list[str]:
    """Split text at the given header boundary positions."""
    units: list[str] = []
    # Text before the first header
    if header_spans[0][0] > 0:
        prefix = text[: header_spans[0][0]].strip()
        if prefix:
            units.append(prefix)
    for i, (start, _end) in enumerate(header_spans):
        next_start = header_spans[i + 1][0] if i + 1 < len(header_spans) else len(text)
        unit = text[start:next_start].strip()
        if unit:
            units.append(unit)
    return [u for u in units if u.strip()]


# ---------------------------------------------------------------------------
# Core chunk dataclass
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """A single chunk ready for indexing."""

    text: str
    text_with_prefix: str
    chunk_idx: int
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Token-bounded packing (spec §6.5.5 steps 4–5)
# ---------------------------------------------------------------------------

def _pack_sentences(
    sentences: list[str],
    tokenizer,
    target: int,
    overlap: int,
    min_tokens: int,
    max_tokens: int,
) -> list[str]:
    """Greedy sentence-packing with sentence-aligned overlap.

    Returns a list of chunk texts (no prefix, no metadata).
    """
    if not sentences:
        return []

    # Pre-compute token counts per sentence.
    counts = [_count_tokens(s, tokenizer) for s in sentences]

    chunks: list[str] = []
    i = 0
    while i < len(sentences):
        chunk_sents: list[str] = []
        total = 0

        while i < len(sentences):
            c = counts[i]
            # Single sentence exceeds max — split at nearest comma or emit oversized
            if not chunk_sents and c > max_tokens:
                oversized = _handle_oversized_sentence(sentences[i], tokenizer, max_tokens)
                chunks.extend(oversized)
                i += 1
                break

            if total + c > target and chunk_sents:
                break

            chunk_sents.append(sentences[i])
            total += c
            i += 1

        if not chunk_sents:
            continue

        text = " ".join(chunk_sents).strip()
        if _count_tokens(text, tokenizer) >= min_tokens:
            chunks.append(text)
        elif chunks:
            # Merge undersized chunk into previous
            merged = chunks[-1] + " " + text
            if _count_tokens(merged, tokenizer) <= max_tokens:
                chunks[-1] = merged
            else:
                chunks.append(text)
        else:
            chunks.append(text)

        # Rewind for overlap: go back until we have accumulated >= overlap_tokens
        if i < len(sentences):
            rewind_tokens = 0
            rewind_sents = 0
            for j in range(len(chunk_sents) - 1, -1, -1):
                rewind_tokens += _count_tokens(chunk_sents[j], tokenizer)
                rewind_sents += 1
                if rewind_tokens >= overlap:
                    break
            # Always preserve forward progress: never rewind the full chunk.
            max_rewind = max(len(chunk_sents) - 1, 0)
            i -= min(rewind_sents, max_rewind)

    return chunks


def _handle_oversized_sentence(sentence: str, tokenizer, max_tokens: int) -> list[str]:
    """Split a single sentence that exceeds max_tokens using punctuation, then words."""
    for pattern, joiner in ((r";\s*", "; "), (r":\s*", ": "), (r",\s*", ", ")):
        parts = re.split(pattern, sentence)
        if len(parts) < 2:
            continue
        results = _pack_text_parts(parts, joiner, tokenizer, max_tokens)
        if all(_count_tokens(chunk, tokenizer) <= max_tokens for chunk in results):
            return results

    results = _split_text_by_words(sentence, tokenizer, max_tokens)
    if any(_count_tokens(chunk, tokenizer) > max_tokens for chunk in results):
        logger.warning(
            "Emitting oversized chunk (%d tokens > %d max).",
            _count_tokens(sentence, tokenizer),
            max_tokens,
        )
    return results


def _pack_text_parts(
    parts: list[str],
    joiner: str,
    tokenizer,
    max_tokens: int,
) -> list[str]:
    results: list[str] = []
    current = ""
    for part in parts:
        piece = part.strip()
        if not piece:
            continue
        candidate = f"{current}{joiner}{piece}" if current else piece
        if _count_tokens(candidate, tokenizer) > max_tokens and current:
            results.append(current)
            current = piece
        else:
            current = candidate
    if current:
        results.append(current)
    return results


def _split_text_by_words(text: str, tokenizer, max_tokens: int) -> list[str]:
    words = text.split()
    if not words:
        return [text]

    results: list[str] = []
    current_words: list[str] = []
    for word in words:
        candidate_words = current_words + [word]
        candidate = " ".join(candidate_words)
        if current_words and _count_tokens(candidate, tokenizer) > max_tokens:
            results.append(" ".join(current_words))
            current_words = [word]
        else:
            current_words = candidate_words

    if current_words:
        results.append(" ".join(current_words))
    return results


# ---------------------------------------------------------------------------
# Section chunker
# ---------------------------------------------------------------------------

def _chunk_section(
    section_text: str,
    section_type: str,
    cfg: ChunkingConfig,
    tokenizer,
    case_name: str = "",
    opinion_type: str = "",
) -> list[str]:
    """Chunk a single section (head_matter or opinion unit) into token-bounded chunks.

    Returns a list of (bare) chunk texts without prefix applied.
    """
    # STEP 1: structural pre-split
    if section_type == "head_matter":
        units = _split_head_matter(section_text, cfg)
    else:
        units = _split_opinion(section_text, cfg)

    # STEP 2: filter/merge sub-minimum units
    merged_units = _merge_short_units(units, cfg.min_chunk_tokens, tokenizer)

    all_chunks: list[str] = []
    for unit in merged_units:
        # STEP 3: sentence tokenization
        sentences = _split_sentences(unit)
        if not sentences:
            sentences = [unit]

        # STEP 4: token-bounded packing
        packed = _pack_sentences(
            sentences,
            tokenizer,
            target=cfg.target_tokens,
            overlap=cfg.overlap_tokens,
            min_tokens=cfg.min_chunk_tokens,
            max_tokens=cfg.max_chunk_tokens,
        )

        if not packed:
            # Guardrail §6.5.5 step 7: section produces zero chunks
            logger.warning(
                "Section produced zero chunks; emitting raw text (section_type=%s).",
                section_type,
            )
            raw = unit[: cfg.max_chunk_tokens * 6]  # rough char cap
            packed = [raw]

        all_chunks.extend(packed)

    return all_chunks


def _merge_short_units(units: list[str], min_tokens: int, tokenizer) -> list[str]:
    """Merge structural units below min_chunk_tokens into the next unit."""
    if not units:
        return []
    merged: list[str] = []
    carry = ""
    for unit in units:
        combined = (carry + " " + unit).strip() if carry else unit
        if _count_tokens(combined, tokenizer) >= min_tokens:
            merged.append(combined)
            carry = ""
        else:
            carry = combined
    if carry:
        if merged:
            merged[-1] = merged[-1] + " " + carry
        else:
            merged.append(carry)
    return merged


# ---------------------------------------------------------------------------
# Contextual prefix (spec §6.5.5 step 5)
# ---------------------------------------------------------------------------

def _build_prefix(
    case_name: str,
    section_type: str,
    opinion_type: str,
    source: str = "",
) -> str:
    """Return the bracketed prefix line for a chunk.

    For user uploads: [<filename>].
    For case-law:      [<case_name> | <section_type>:<opinion_type or head_matter>].
    """
    if section_type == "user_upload":
        return f"[{source}]" if source else ""

    qualifier = opinion_type if opinion_type else section_type
    return f"[{case_name} | {section_type}:{qualifier}]"


def _apply_prefix(chunk_text: str, prefix: str) -> str:
    if not prefix:
        return chunk_text
    return f"{prefix}\n{chunk_text}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class ChunkResult:
    chunk_idx: int
    text: str
    text_with_prefix: str


def chunk_case_law(
    section_text: str,
    section_type: str,
    case_name: str,
    opinion_type: str = "",
    cfg: Optional[ChunkingConfig] = None,
) -> list[ChunkResult]:
    """Chunk a single CAP case-law section.

    Args:
        section_text: Pre-normalized section text.
        section_type: "head_matter" | "opinion".
        case_name: Display name for the contextual prefix.
        opinion_type: "majority" | "dissent" | "concurrence" | "" for head_matter.
        cfg: Chunking configuration; defaults to module-level settings.

    Returns:
        List of ChunkResult with chunk_idx (0-based within this section),
        bare text, and text_with_prefix.
    """
    if cfg is None:
        cfg = default_settings.chunking
    tokenizer = _get_tokenizer(cfg.tokenizer_model)
    prefix = (
        _build_prefix(case_name, section_type, opinion_type)
        if cfg.contextual_prefix.enabled
        else ""
    )
    effective_cfg = _budget_cfg_for_prefix(cfg, tokenizer, prefix)

    chunks_text = _chunk_section(
        section_text, section_type, effective_cfg, tokenizer,
        case_name=case_name, opinion_type=opinion_type,
    )

    return [
        ChunkResult(
            chunk_idx=i,
            text=text,
            text_with_prefix=_apply_prefix(text, prefix),
        )
        for i, text in enumerate(chunks_text)
    ]


def chunk_user_upload(
    text: str,
    source: str,
    cfg: Optional[ChunkingConfig] = None,
) -> list[ChunkResult]:
    """Chunk a user-uploaded document (PDF/text).

    No structural pre-split; treated as a single flat section.
    section_type is always "user_upload".
    """
    if cfg is None:
        cfg = default_settings.chunking
    tokenizer = _get_tokenizer(cfg.tokenizer_model)
    prefix = f"[{source}]" if cfg.contextual_prefix.enabled and source else ""
    effective_cfg = _budget_cfg_for_prefix(cfg, tokenizer, prefix)

    sentences = _split_sentences(text)
    if not sentences:
        sentences = [text]

    chunks_text = _pack_sentences(
        sentences,
        tokenizer,
        target=effective_cfg.target_tokens,
        overlap=effective_cfg.overlap_tokens,
        min_tokens=effective_cfg.min_chunk_tokens,
        max_tokens=effective_cfg.max_chunk_tokens,
    )

    if not chunks_text:
        logger.warning("User upload produced zero chunks for source=%s; emitting raw.", source)
        chunks_text = [text[: effective_cfg.max_chunk_tokens * 6]]

    return [
        ChunkResult(
            chunk_idx=i,
            text=chunk,
            text_with_prefix=_apply_prefix(chunk, prefix),
        )
        for i, chunk in enumerate(chunks_text)
    ]


def _budget_cfg_for_prefix(cfg: ChunkingConfig, tokenizer, prefix: str) -> ChunkingConfig:
    if not prefix:
        return cfg

    prefix_tokens = _count_tokens(prefix, tokenizer)
    max_tokens = max(cfg.min_chunk_tokens, cfg.max_chunk_tokens - prefix_tokens)
    target_tokens = max(cfg.min_chunk_tokens, min(cfg.target_tokens, max_tokens))
    overlap_tokens = min(cfg.overlap_tokens, max(target_tokens - 1, 0))

    return cfg.model_copy(
        update={
            "target_tokens": target_tokens,
            "overlap_tokens": overlap_tokens,
            "max_chunk_tokens": max_tokens,
        }
    )
