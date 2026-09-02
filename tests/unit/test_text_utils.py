"""
tests/unit/test_text_utils.py
=============================

Unit tests for the text helpers. Each test checks ONE small behavior.
Run: pytest

A test PASSES if all its `assert` statements are true.
"""

from app.utils.text_utils import clean_text, chunk_text


def test_clean_text_strips_edges():
    # Leading/trailing spaces should be removed.
    assert clean_text("   hello world   ") == "hello world"


def test_clean_text_collapses_blank_lines():
    messy = "line1\n\n\n\n\nline2"
    result = clean_text(messy)
    # Both words survive, but the big gap is reduced (no triple newline).
    assert "line1" in result
    assert "line2" in result
    assert "\n\n\n" not in result


def test_chunk_text_returns_list_of_strings():
    chunks = chunk_text("word " * 100, chunk_size=100)
    assert isinstance(chunks, list)
    assert len(chunks) >= 1
    assert all(isinstance(c, str) for c in chunks)


def test_chunk_text_respects_size():
    chunks = chunk_text("a b c d e f g h " * 50, chunk_size=100)
    # No chunk should be much larger than the requested size.
    assert all(len(c) <= 120 for c in chunks)


def test_chunk_text_empty_input():
    # Empty text should give an empty list, not crash.
    assert chunk_text("", chunk_size=100) == []
