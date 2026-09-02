"""
app/services/rerank_service.py
==============================

Cross-encoder RE-RANKING (stage 2 of two-stage retrieval).

The first stage (hybrid_search) is fast but only "roughly" right. This stage
takes those candidates and has a smarter model (a CROSS-ENCODER) read each
(question, chunk) pair TOGETHER to score true relevance, then re-orders them.

Bi-encoder vs cross-encoder:
  - bi-encoder (our embedding model): encodes question and chunk SEPARATELY,
    then compares vectors. Fast, scales to millions, less precise.
  - cross-encoder (here): reads question and chunk TOGETHER in one pass, so it
    can directly compare them. Slow, but much more accurate. Used only on the
    small candidate set from stage 1.

We load the model once and reuse it (same caching pattern as embeddings).
"""

from sentence_transformers import CrossEncoder

import app.config as config


_reranker = None


def get_reranker():
    """Load the cross-encoder re-ranker once, then reuse it."""
    global _reranker
    if _reranker is None:
        print(f"[rerank_service] Loading reranker: {config.RERANKER_MODEL}")
        _reranker = CrossEncoder(config.RERANKER_MODEL)
    return _reranker


def rerank(question, candidates, top_k=None):
    """
    Re-order candidate chunks by TRUE relevance to the question.

    Input:
        question   -> the user's question (string)
        candidates -> list of result dicts (from hybrid_search), each with
                      at least "text" and "source"
        top_k      -> how many to keep after re-ranking (default from config)

    Output:
        the candidates re-ordered best-first, each with a new
        "rerank_score" (higher = more relevant). Only top_k are returned.

    Called by: rag_service (the full pipeline), built next.
    """
    if top_k is None:
        top_k = config.TOP_K

    # Nothing to do if there are no candidates.
    if not candidates:
        return []

    model = get_reranker()

    # Build (question, chunk) PAIRS - the cross-encoder scores each pair.
    pairs = [(question, item["text"]) for item in candidates]

    # One relevance score per pair. Higher = more relevant.
    scores = model.predict(pairs)

    # Attach each score to its candidate.
    reranked = []
    for item, score in zip(candidates, scores):
        reranked.append({
            "text": item["text"],
            "source": item["source"],
            "parent_id": item.get("parent_id"),
            "rerank_score": float(score),
        })

    # Best relevance first, then keep only the top_k.
    reranked.sort(key=lambda r: r["rerank_score"], reverse=True)
    return reranked[:top_k]
