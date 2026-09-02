"""
app/services/compression_service.py
===================================

Contextual Compression (post-retrieval).

After we retrieve a chunk, it may contain sentences that are NOT relevant to the
question. This service keeps only the sentences whose meaning is close to the
question, so the LLM sees focused context (fewer tokens, less noise).

How: split a chunk into sentences, embed each sentence and the question, keep
the sentences with high similarity (always keep at least the best one).
"""

import re

import numpy as np

import app.config as config
from app.services.embedding_service import embed_query, embed_documents


def _split_sentences(text):
    """Split text into sentences on . ! ? boundaries."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p.strip()]


def compress_chunk(question, text, margin=None):
    """
    Trim one chunk to only its question-relevant sentences.

    Input:
        question -> the user's question
        text     -> the chunk text
        margin   -> keep sentences within this similarity margin of the BEST
                    sentence (default from config). Relative, not absolute.
    Output:
        the trimmed text (relevant sentences only, original order).
    """
    if margin is None:
        margin = config.COMPRESSION_MARGIN

    sentences = _split_sentences(text)
    if len(sentences) <= 1:
        return text                       # nothing to trim

    q_vec = embed_query(question)
    s_vecs = embed_documents(sentences)
    sims = [float(np.dot(q_vec, v)) for v in s_vecs]

    # Keep sentences whose similarity is within `margin` of the best sentence.
    best = max(sims)
    threshold = best - margin
    keep = {i for i, s in enumerate(sims) if s >= threshold}

    kept = [sentences[i] for i in range(len(sentences)) if i in keep]
    return " ".join(kept)


def compress_chunks(question, chunks):
    """
    Apply compress_chunk to every retrieved chunk.

    Input:  question, chunks (list of dicts with "text")
    Output: new list of chunks with trimmed "text".
    """
    compressed = []
    for chunk in chunks:
        new_chunk = dict(chunk)
        new_chunk["text"] = compress_chunk(question, chunk["text"])
        compressed.append(new_chunk)
    return compressed
