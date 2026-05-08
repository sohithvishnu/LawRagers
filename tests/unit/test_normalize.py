"""Unit tests for retriever_service.normalize (spec §6)."""

import pytest
from retriever_service.normalize import normalize, normalize_query


class TestUnicodeNormalization:
    def test_smart_quotes_folded(self):
        result = normalize("“Hello”")
        assert '"Hello"' in result

    def test_em_dash_folded(self):
        result = normalize("A—B")
        assert " - " in result

    def test_ligature_fi_unfolded(self):
        result = normalize("ﬁrst")
        assert "first" in result

    def test_ligature_fl_unfolded(self):
        result = normalize("ﬂoor")
        assert "floor" in result

    def test_nfkc_applied(self):
        # NFKC: full-width digit → ASCII
        result = normalize("０")  # FULLWIDTH DIGIT ZERO
        assert "0" in result


class TestSectionMarkers:
    def test_single_section_marker(self):
        result = normalize("See § 100 of the code.")
        assert "section 100" in result

    def test_double_section_marker(self):
        result = normalize("See §§ 100-102.")
        assert "sections 100" in result

    def test_pilcrow_marker(self):
        result = normalize("See ¶ 5.")
        assert "paragraph 5" in result


class TestCaseNameCanonicalisation:
    def test_simple_v_pattern(self):
        result = normalize("Smith v. Jones")
        assert "[CASE_Smith_v_Jones]" in result

    def test_multiword_party(self):
        result = normalize("State of New York v. American Corp")
        assert "[CASE_" in result
        assert "_v_" in result

    def test_no_false_positive_lowercase(self):
        # "smith v. jones" should NOT match (starts lowercase)
        result = normalize("smith v. jones")
        assert "[CASE_" not in result


class TestWhitespaceCollapse:
    def test_trailing_whitespace_stripped(self):
        result = normalize("  hello world  ")
        assert result == result.strip()

    def test_multiple_spaces_collapsed(self):
        result = normalize("hello   world")
        assert "  " not in result

    def test_excessive_newlines_collapsed(self):
        result = normalize("a\n\n\n\nb")
        assert "\n\n\n" not in result


class TestNormalizeQueryAlias:
    def test_normalize_query_identical_to_normalize(self):
        text = "Smith v. Jones § 100"
        assert normalize_query(text) == normalize(text)


class TestPipeline:
    def test_empty_string(self):
        result = normalize("")
        assert isinstance(result, str)

    def test_idempotent(self):
        text = "Smith v. Jones, §§ 1-3"
        once = normalize(text)
        twice = normalize(once)
        assert once == twice
