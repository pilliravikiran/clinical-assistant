"""
app/services/query_service.py
=============================

Query Transformation (multi-query).

We take the user's question and produce a few VARIATIONS of it, so that
searching with all of them catches documents that a single phrasing would miss.

Two versions (same provider-abstraction pattern as llm_service):
  - MOCK   : offline, makes simple deterministic variations (keeps things running)
  - CLAUDE : asks the LLM to write real paraphrases (much better)
"""

import re

import app.config as config


# Very small stopword list used only by the offline mock variation.
_STOPWORDS = {
    "what", "which", "who", "when", "where", "why", "how", "is", "are", "was",
    "were", "the", "a", "an", "of", "to", "for", "and", "or", "in", "on", "did",
    "do", "does", "recommended", "patient",
}


def generate_query_variations(question, n=3):
    """
    Produce a list of query variations (including the original question).

    Input:
        question -> the user's question (string)
        n        -> how many variations to aim for
    Output:
        a list of strings (the original + variations, duplicates removed).

    Called by: retrieval_service.multi_query_search().
    """
    if config.APP_MODE.lower() == "online" and config.LLM_API_KEY:
        variations = _variations_with_claude(question, n)
    else:
        variations = _variations_with_mock(question)

    # Always include the original, and remove duplicates while keeping order.
    all_queries = [question] + variations
    seen = set()
    unique = []
    for q in all_queries:
        key = q.strip().lower()
        if key and key not in seen:
            seen.add(key)
            unique.append(q.strip())
    return unique


def _variations_with_mock(question):
    """
    Offline stand-in: makes simple variations without a real LLM.
      1. a keyword-only version (stopwords removed)
      2. a "tell me about ..." version
    Not as good as real paraphrases, but keeps the pipeline runnable offline.
    """
    words = re.findall(r"[A-Za-z0-9]+", question)
    keywords = [w for w in words if w.lower() not in _STOPWORDS]

    variations = []
    if keywords:
        variations.append(" ".join(keywords))                 # keyword-only
        variations.append("information about " + " ".join(keywords))  # rephrase
    return variations


def _variations_with_claude(question, n):
    """
    Ask Claude to rewrite the question into n alternative phrasings.
    Returns a list of strings (one per line of the reply).
    """
    import anthropic

    client = anthropic.Anthropic(api_key=config.LLM_API_KEY)
    prompt = (
        f"Rewrite the following question in {n} different ways that mean the "
        f"same thing, using different words and synonyms. "
        f"Return ONLY the {n} rewrites, one per line, no numbering.\n\n"
        f"Question: {question}"
    )

    try:
        response = client.messages.create(
            model=config.LLM_MODEL,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
    except anthropic.APIStatusError:
        # If the rewrite call fails, fall back to just the original question.
        return []

    text = ""
    for block in response.content:
        if block.type == "text":
            text += block.text

    # Split the reply into individual lines -> variations.
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines
