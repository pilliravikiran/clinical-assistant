"""
app/services/guardrails_service.py
==================================

Safety guardrails for the RAG system.

Two guardrails:
  1. passes_relevance() - is the best retrieved chunk actually relevant enough?
     If not, the pipeline should refuse ("I don't know") instead of guessing.
  2. redact_phi() - mask Protected Health Information (PHI) / PII before we
     log anything. In healthcare you must NEVER log raw patient data.
"""

import re

import app.config as config


def passes_relevance(top_chunks, threshold=None):
    """
    Decide whether the retrieved chunks are relevant enough to answer.

    Input:
        top_chunks -> the re-ranked chunks (each has a "rerank_score")
        threshold  -> minimum score for the BEST chunk (default from config)
    Output:
        True  -> good enough, go ahead and answer
        False -> too weak, the pipeline should refuse

    Called by: rag_service.answer_question().
    """
    if threshold is None:
        threshold = config.RELEVANCE_THRESHOLD

    if not top_chunks:
        return False

    # The best chunk is the highest re-rank score.
    best_score = max(chunk.get("rerank_score", 0.0) for chunk in top_chunks)
    return best_score >= threshold


# Patterns for common PHI/PII. This is a simple, illustrative set - a real
# system would use a dedicated PII detector. Order matters (more specific first).
_REDACTION_PATTERNS = [
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN]"),                # 123-45-6789
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), "[EMAIL]"),       # a@b.com
    (re.compile(r"\b\d{4}-\d{2}-\d{2}\b"), "[DATE]"),               # 2025-04-10
    (re.compile(r"\b[A-Z]{2,3}-\d{2,}\b"), "[ID]"),                 # PN-001, DS-001
    (re.compile(r"\b\+?\d[\d\-\s]{7,}\d\b"), "[PHONE]"),            # long digit runs
]


# Patterns that suggest a prompt-injection / jailbreak attempt.
_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions", re.I),
    re.compile(r"disregard\s+(the\s+)?(previous|above|system)", re.I),
    re.compile(r"forget\s+(your|all)\s+(instructions|rules)", re.I),
    re.compile(r"(reveal|show|print|repeat)\s+(your\s+)?(system\s+)?prompt", re.I),
    re.compile(r"you\s+are\s+now\s+", re.I),
    re.compile(r"\bact\s+as\b", re.I),
    re.compile(r"override|bypass|jailbreak", re.I),
]


def detect_prompt_injection(text):
    """
    Return True if the text looks like a prompt-injection / jailbreak attempt
    (e.g. "ignore your instructions", "reveal your system prompt").

    This is a simple pattern-based check; production systems also use a
    classifier (e.g. Llama Guard). Called by the RAG pipeline to BLOCK such
    requests before they reach the LLM.
    """
    return any(pattern.search(text) for pattern in _INJECTION_PATTERNS)


def redact_phi(text):
    """
    Replace obvious PHI/PII in text with safe placeholders, for logging.

    Input:  text -> any string (e.g. the user's question)
    Output: the same text with sensitive patterns masked.

    Example:
        "Email a@b.com about SSN 123-45-6789" ->
        "Email [EMAIL] about SSN [SSN]"

    Called by: logging code (added later) before writing logs.
    """
    redacted = text
    for pattern, replacement in _REDACTION_PATTERNS:
        redacted = pattern.sub(replacement, redacted)
    return redacted
