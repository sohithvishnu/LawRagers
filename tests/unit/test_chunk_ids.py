"""Unit tests for retriever_service.stores.chunk_ids (spec §4.4)."""

import pytest
from retriever_service.stores.chunk_ids import compute_chunk_id


class TestDeterminism:
    def test_same_inputs_same_id(self):
        a = compute_chunk_id("hello world", "file.txt", "session-1")
        b = compute_chunk_id("hello world", "file.txt", "session-1")
        assert a == b

    def test_different_text_different_id(self):
        a = compute_chunk_id("hello", "file.txt", "s1")
        b = compute_chunk_id("world", "file.txt", "s1")
        assert a != b

    def test_different_source_different_id(self):
        a = compute_chunk_id("hello", "file_a.txt", "s1")
        b = compute_chunk_id("hello", "file_b.txt", "s1")
        assert a != b

    def test_different_session_different_id(self):
        a = compute_chunk_id("hello", "file.txt", "session-A")
        b = compute_chunk_id("hello", "file.txt", "session-B")
        assert a != b


class TestFormat:
    def test_returns_hex_string(self):
        result = compute_chunk_id("text", "source", "session")
        assert isinstance(result, str)
        assert all(c in "0123456789abcdef" for c in result)

    def test_md5_length(self):
        result = compute_chunk_id("text", "source", "session")
        assert len(result) == 32


class TestEmptySession:
    def test_empty_session_allowed(self):
        result = compute_chunk_id("text", "source", "")
        assert isinstance(result, str)
        assert len(result) == 32

    def test_empty_session_differs_from_nonempty(self):
        a = compute_chunk_id("text", "source", "")
        b = compute_chunk_id("text", "source", "s1")
        assert a != b
