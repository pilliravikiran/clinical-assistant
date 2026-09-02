"""
tests/unit/test_guardrails.py
=============================

Unit tests for the safety guardrails.
"""

from app.services.guardrails_service import redact_phi, passes_relevance


def test_redact_phi_masks_email_ssn_and_id():
    text = "contact a@b.com, SSN 123-45-6789, record DS-001"
    out = redact_phi(text)
    # The sensitive values must be gone, replaced by placeholders.
    assert "a@b.com" not in out
    assert "123-45-6789" not in out
    assert "DS-001" not in out
    assert "[EMAIL]" in out and "[SSN]" in out and "[ID]" in out


def test_passes_relevance_true_above_threshold():
    chunks = [{"rerank_score": 5.0}, {"rerank_score": 1.0}]
    assert passes_relevance(chunks, threshold=0.0) is True


def test_passes_relevance_false_below_threshold():
    chunks = [{"rerank_score": -3.0}, {"rerank_score": -8.0}]
    assert passes_relevance(chunks, threshold=0.0) is False


def test_passes_relevance_empty_is_false():
    assert passes_relevance([]) is False
