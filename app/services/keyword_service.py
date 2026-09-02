"""
app/services/keyword_service.py
===============================

Keyword (a.k.a. "sparse" or "lexical") search using BM25.

Why: semantic search matches MEANING but can miss EXACT terms (drug names,
codes, IDs). BM25 is a smart keyword ranker that catches those. Later we
combine this with semantic search ("hybrid search").

BM25 scores a document using three ideas:
  1. term frequency        - how often query words appear in the document
  2. inverse doc frequency - rare words count more than common words
  3. length normalization  - long documents get no unfair advantage

Same shape as vector_service: add_documents(...) then search(...).
"""

import re

from rank_bm25 import BM25Okapi


# Our stored documents and the built BM25 index.
_docs = []       # list of {"text": ..., "source": ...}
_bm25 = None     # the BM25 index object (built from the docs)


def _tokenize(text):
    """
    Turn text into a list of lowercase word-tokens.

    Why: BM25 works on individual words, so we split text into words first.
    Example: "Start Lisinopril 10mg" -> ["start", "lisinopril", "10mg"]
    """
    # Lowercase, then grab runs of letters/numbers as words.
    return re.findall(r"[a-z0-9]+", text.lower())


def reset():
    """Empty the keyword store (before re-adding docs or in tests)."""
    global _docs, _bm25
    _docs = []
    _bm25 = None


def add_documents(texts, sources, parent_ids=None):
    """
    Remember document pieces and (re)build the BM25 keyword index.

    Input:
        texts      -> list of document pieces (strings)
        sources    -> list of file names (strings), one per piece
        parent_ids -> optional list of parent ids (for parent-document retrieval)
    Output:
        None. Builds the index in memory.
    """
    global _docs, _bm25

    if parent_ids is None:
        parent_ids = [None] * len(texts)

    # Save the pieces together with their source file and parent id.
    for text, source, parent_id in zip(texts, sources, parent_ids):
        _docs.append({"text": text, "source": source, "parent_id": parent_id})

    # Build the BM25 index from the tokenized text of every stored piece.
    tokenized_corpus = [_tokenize(doc["text"]) for doc in _docs]
    _bm25 = BM25Okapi(tokenized_corpus)


def search(query, top_k=5):
    """
    Find the document pieces whose KEYWORDS best match the query.

    Input:
        query -> the user's question / search text (string)
        top_k -> how many results to return
    Output:
        list of {"text": ..., "source": ..., "score": <bm25 score>}
        sorted best-first. Higher score = better keyword match.
    """
    if _bm25 is None or len(_docs) == 0:
        return []

    # BM25 scores every stored document against the tokenized query.
    query_tokens = _tokenize(query)
    scores = _bm25.get_scores(query_tokens)

    # Pair each score with its document, then sort best-first.
    results = []
    for doc, score in zip(_docs, scores):
        results.append({
            "text": doc["text"],
            "source": doc["source"],
            "parent_id": doc.get("parent_id"),
            "score": float(score),
        })
    results.sort(key=lambda r: r["score"], reverse=True)

    return results[:top_k]
