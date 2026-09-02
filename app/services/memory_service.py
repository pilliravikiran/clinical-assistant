"""
app/services/memory_service.py
==============================

Semantic long-term memory (advanced conversational RAG).

Unlike the recent-turns buffer, this stores EVERY past turn as an embedding and
can RECALL the most relevant past messages by meaning - even old ones that fell
out of the recent buffer. This is "retrieve relevant past turns", i.e. a small
vector store over the conversation itself.
"""

import numpy as np

import app.config as config
from app.services.embedding_service import embed_documents, embed_query


# session_id -> list of {"text": <past message>, "vector": <embedding>}
_memory = {}


def reset(session_id=None):
    """Forget one session's long-term memory, or all sessions."""
    if session_id is None:
        _memory.clear()
    else:
        _memory.pop(session_id, None)


def add(session_id, text):
    """Store a past message (with its embedding) for future semantic recall."""
    vector = embed_documents([text])[0]
    _memory.setdefault(session_id, []).append({"text": text, "vector": vector})


def recall(session_id, query, top_k=None, min_sim=None):
    """
    Return the most RELEVANT past messages for this query (by meaning).

    Input:
        session_id -> which conversation
        query      -> the new message to find relevant history for
        top_k      -> how many to recall (default config)
        min_sim    -> only recall messages at least this similar (default config)
    Output:
        a list of past message strings (most relevant first), possibly empty.
    """
    if top_k is None:
        top_k = config.MEMORY_RECALL_K
    if min_sim is None:
        min_sim = config.MEMORY_RECALL_MIN_SIM

    items = _memory.get(session_id, [])
    if not items:
        return []

    q_vec = embed_query(query)
    scored = [(float(np.dot(q_vec, it["vector"])), it["text"]) for it in items]
    scored.sort(key=lambda x: x[0], reverse=True)

    # keep only the top_k that clear the relevance floor
    return [text for sim, text in scored[:top_k] if sim >= min_sim]
