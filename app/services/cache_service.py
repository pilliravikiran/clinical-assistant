"""
app/services/cache_service.py
=============================

Semantic cache for answers.

If a new question is very similar (by meaning) to one we already answered, we
return the cached answer instantly - skipping retrieval + the LLM. This saves
cost and latency for repeated / rephrased questions.

Unlike an exact-match cache, this matches by MEANING (embedding similarity), so
"what follow-up care?" and "what aftercare was advised?" can share a cache hit.
"""

import numpy as np

import app.config as config
from app.services.embedding_service import embed_query


# list of {"vector": <question embedding>, "result": <answer dict>}
_cache = []


def reset():
    """Clear the cache (e.g. after re-ingesting documents)."""
    global _cache
    _cache = []


def get(question):
    """
    Return a cached result for a semantically similar question, or None.

    Uses the highest-similarity cached question; returns it only if the
    similarity clears CACHE_SIM_THRESHOLD (high, so only near-identical reuse).
    """
    if not _cache:
        return None

    q_vec = embed_query(question)
    best_item, best_sim = None, -1.0
    for item in _cache:
        sim = float(np.dot(q_vec, item["vector"]))
        if sim > best_sim:
            best_sim, best_item = sim, item

    if best_sim >= config.CACHE_SIM_THRESHOLD:
        return best_item["result"]
    return None


def put(question, result):
    """
    Store an answer keyed by the question's embedding.
    Caps the cache at CACHE_MAX_SIZE (drops the oldest entry) so it can't grow
    without bound - a real deployment would use Redis with a TTL/eviction policy.
    """
    _cache.append({"vector": embed_query(question), "result": result})
    if len(_cache) > config.CACHE_MAX_SIZE:
        _cache.pop(0)   # evict the oldest (simple FIFO)
