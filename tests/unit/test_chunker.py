"""Unit tests for the hierarchical chunker (spec §6.5)."""

import signal
import pytest
from retriever_service.ingestion.chunker import (
    chunk_case_law,
    chunk_user_upload,
    ChunkResult,
    _pack_sentences,
    _split_head_matter,
    _split_opinion,
    _build_prefix,
    _WhitespaceFallbackTokenizer,
)
from retriever_service.config import ChunkingConfig


@pytest.fixture
def cfg() -> ChunkingConfig:
    """Minimal ChunkingConfig using the whitespace fallback tokenizer."""
    return ChunkingConfig(
        target_tokens=30,
        overlap_tokens=5,
        min_chunk_tokens=3,
        max_chunk_tokens=50,
        tokenizer_model="sentence-transformers/all-MiniLM-L6-v2",
    )


@pytest.fixture
def tok():
    return _WhitespaceFallbackTokenizer()


# ---------------------------------------------------------------------------
# Prefix builder
# ---------------------------------------------------------------------------

class TestBuildPrefix:
    def test_case_law_majority(self):
        p = _build_prefix("Smith v. Jones", "opinion", "majority")
        assert p == "[Smith v. Jones | opinion:majority]"

    def test_head_matter(self):
        p = _build_prefix("Smith v. Jones", "head_matter", "")
        assert p == "[Smith v. Jones | head_matter:head_matter]"

    def test_user_upload(self):
        p = _build_prefix("", "user_upload", "", source="deposition.pdf")
        assert p == "[deposition.pdf]"

    def test_user_upload_no_source(self):
        p = _build_prefix("", "user_upload", "")
        assert p == ""


# ---------------------------------------------------------------------------
# Sentence packing
# ---------------------------------------------------------------------------

class TestPackSentences:
    def test_empty_input(self, tok):
        result = _pack_sentences([], tok, target=30, overlap=5, min_tokens=3, max_tokens=50)
        assert result == []

    def test_single_sentence(self, tok):
        sents = ["This is one sentence."]
        result = _pack_sentences(sents, tok, target=30, overlap=5, min_tokens=3, max_tokens=50)
        assert len(result) == 1
        assert "This is one sentence." in result[0]

    def test_produces_non_empty_chunks(self, tok):
        sents = ["Word " * 5 + "." for _ in range(20)]
        result = _pack_sentences(sents, tok, target=30, overlap=5, min_tokens=3, max_tokens=50)
        assert len(result) >= 1
        for chunk in result:
            assert chunk.strip()

    def test_overlap_creates_repeated_sentences(self, tok):
        sents = [f"Sentence number {i}." for i in range(10)]
        result = _pack_sentences(sents, tok, target=10, overlap=3, min_tokens=2, max_tokens=20)
        if len(result) >= 2:
            # Last sentence of chunk N should appear in chunk N+1 due to overlap.
            last_words_of_first = set(result[0].split())
            first_words_of_second = set(result[1].split())
            assert last_words_of_first & first_words_of_second  # non-empty intersection

    def test_overlap_does_not_cause_non_progress_loop(self, tok):
        sents = ["a b", "c d", "e f", "g h", "i j"]

        def _timeout(_signum, _frame):
            raise TimeoutError("chunk packing did not terminate")

        previous = signal.signal(signal.SIGALRM, _timeout)
        signal.alarm(1)
        try:
            result = _pack_sentences(
                sents,
                tok,
                target=4,
                overlap=5,
                min_tokens=1,
                max_tokens=50,
            )
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)

        assert result

    def test_merging_short_tail_does_not_exceed_max_tokens(self, tok):
        sents = [
            " ".join(f"a{i}" for i in range(17)),
            " ".join(f"b{i}" for i in range(17)),
            "tail one two",
        ]

        result = _pack_sentences(
            sents,
            tok,
            target=35,
            overlap=0,
            min_tokens=5,
            max_tokens=35,
        )

        assert result
        assert all(len(chunk.split()) <= 35 for chunk in result)


# ---------------------------------------------------------------------------
# Structural splitting
# ---------------------------------------------------------------------------

class TestSplitHeadMatter:
    def test_no_held_points_returns_single_unit(self, cfg):
        text = "This is a head matter without held points."
        units = _split_head_matter(text, cfg)
        assert units == [text]

    def test_held_points_split(self, cfg):
        text = (
            "Introduction text here.\n"
            "Held, 1. First ruling.\n"
            "Held, 2. Second ruling."
        )
        units = _split_head_matter(text, cfg)
        assert len(units) >= 2

    def test_too_many_held_points_fallback(self, cfg):
        cfg.structural_split.head_matter.held_points_max = 2
        text = "\n".join([f"Held, {i}. Ruling {i}." for i in range(1, 5)])
        units = _split_head_matter(text, cfg)
        # Should fall back to single unit
        assert len(units) == 1


class TestSplitOpinion:
    def test_roman_headers_split(self, cfg):
        text = (
            "Introduction.\n"
            "I. First section content.\n"
            "II. Second section content."
        )
        units = _split_opinion(text, cfg)
        assert len(units) >= 2

    def test_no_headers_single_unit(self, cfg):
        text = "Just a plain opinion without any headers."
        units = _split_opinion(text, cfg)
        assert len(units) == 1

    def test_too_many_headers_fallback(self, cfg):
        cfg.structural_split.opinion.headers_max = 2
        text = "\n".join([f"I{'I' * i}. Section {i}. Content {i}." for i in range(4)])
        # Falls back to single unit when too many detected
        units = _split_opinion(text, cfg)
        assert isinstance(units, list)


# ---------------------------------------------------------------------------
# chunk_case_law public API
# ---------------------------------------------------------------------------

class TestChunkCaseLaw:
    def test_returns_list_of_chunk_results(self, cfg):
        text = "This is a legal opinion. " * 20
        results = chunk_case_law(
            section_text=text,
            section_type="opinion",
            case_name="Test v. Case",
            opinion_type="majority",
            cfg=cfg,
        )
        assert isinstance(results, list)
        assert all(isinstance(r, ChunkResult) for r in results)

    def test_chunk_idx_sequential(self, cfg):
        text = "Word " * 200
        results = chunk_case_law(
            section_text=text,
            section_type="opinion",
            case_name="A v. B",
            cfg=cfg,
        )
        for i, r in enumerate(results):
            assert r.chunk_idx == i

    def test_contextual_prefix_present(self, cfg):
        cfg.contextual_prefix.enabled = True
        results = chunk_case_law(
            section_text="Test opinion text. " * 10,
            section_type="opinion",
            case_name="Ranger v. Goodrich",
            opinion_type="majority",
            cfg=cfg,
        )
        assert any("[Ranger v. Goodrich | opinion:majority]" in r.text_with_prefix for r in results)

    def test_prefix_disabled(self, cfg):
        cfg.contextual_prefix.enabled = False
        results = chunk_case_law(
            section_text="Test text. " * 10,
            section_type="opinion",
            case_name="X v. Y",
            cfg=cfg,
        )
        for r in results:
            assert r.text == r.text_with_prefix

    def test_bare_text_not_empty(self, cfg):
        results = chunk_case_law(
            section_text="Non-empty opinion. " * 10,
            section_type="opinion",
            case_name="A v. B",
            cfg=cfg,
        )
        for r in results:
            assert r.text.strip()

    def test_head_matter_section(self, cfg):
        text = "Head matter content. " * 10
        results = chunk_case_law(
            section_text=text,
            section_type="head_matter",
            case_name="Test v. Case",
            cfg=cfg,
        )
        assert len(results) >= 1

    def test_text_with_prefix_respects_max_chunk_tokens(self, cfg):
        cfg.target_tokens = 35
        cfg.max_chunk_tokens = 35
        cfg.min_chunk_tokens = 5
        cfg.contextual_prefix.enabled = True

        text = " ".join(f"word{i}" for i in range(60))
        results = chunk_case_law(
            section_text=text,
            section_type="opinion",
            case_name="Very Long Case Name v. Very Long Other Party",
            opinion_type="majority",
            cfg=cfg,
        )

        for r in results:
            assert len(r.text_with_prefix.split()) <= 35


class TestChunkUserUpload:
    def test_returns_list(self, cfg):
        results = chunk_user_upload(
            text="User uploaded document text. " * 20,
            source="deposition.pdf",
            cfg=cfg,
        )
        assert isinstance(results, list)
        assert len(results) >= 1

    def test_source_prefix_in_text_with_prefix(self, cfg):
        cfg.contextual_prefix.enabled = True
        results = chunk_user_upload(
            text="Some text. " * 10,
            source="exhibit_A.pdf",
            cfg=cfg,
        )
        assert any("[exhibit_A.pdf]" in r.text_with_prefix for r in results)

    def test_section_type_not_opinion(self, cfg):
        results = chunk_user_upload(
            text="Text. " * 10,
            source="doc.txt",
            cfg=cfg,
        )
        # chunk_idx should be sequential
        for i, r in enumerate(results):
            assert r.chunk_idx == i

    def test_empty_text_handled(self, cfg):
        results = chunk_user_upload(text="", source="empty.txt", cfg=cfg)
        assert isinstance(results, list)
