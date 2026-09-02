"""
app/services/vector_service.py
==============================

The "vector store": remembers document pieces + their meaning-numbers, and
finds the pieces closest in meaning to a question.

TWO backends behind the SAME functions (chosen by config.VECTOR_BACKEND):
  - "local"    : a simple in-memory Python list (free, no key; rebuilt each run)
  - "pinecone" : a managed CLOUD vector database (persists; needs an API key)

The rest of the app (retrieval_service, rag_service) calls add_documents() and
search() and never needs to know which backend is active.
"""

import uuid

import numpy as np

import app.config as config


# ---- local backend state ----
_store = []          # list of {"text","source","vector"}
# ---- pinecone backend state ----
_pc_index = None     # cached Pinecone index handle


def _use_pinecone():
    """True if config says to use the Pinecone cloud backend."""
    return config.VECTOR_BACKEND.lower() == "pinecone"


def _get_pinecone_index():
    """
    Connect to Pinecone (once), creating the index if it does not exist.
    Cached so we connect only one time.
    """
    global _pc_index
    if _pc_index is None:
        import time
        from pinecone import Pinecone, ServerlessSpec
        from app.services.embedding_service import get_dimension

        pc = Pinecone(api_key=config.PINECONE_API_KEY)
        existing = [ix["name"] for ix in pc.list_indexes()]

        if config.PINECONE_INDEX_NAME not in existing:
            print(f"[vector_service] Creating Pinecone index '{config.PINECONE_INDEX_NAME}'...")
            pc.create_index(
                name=config.PINECONE_INDEX_NAME,
                dimension=get_dimension(),      # 384 for our model
                metric="cosine",                # match how we compare meaning
                spec=ServerlessSpec(cloud=config.PINECONE_CLOUD, region=config.PINECONE_REGION),
            )
            # Wait until the new index is ready before using it.
            while not pc.describe_index(config.PINECONE_INDEX_NAME).status["ready"]:
                time.sleep(1)

        _pc_index = pc.Index(config.PINECONE_INDEX_NAME)
    return _pc_index


def reset():
    """Empty the store (local: clear list; pinecone: delete all vectors)."""
    global _store, _pc_index
    if _use_pinecone():
        index = _get_pinecone_index()
        try:
            index.delete(delete_all=True)
        except Exception:
            pass   # empty index raises on delete_all; safe to ignore
    else:
        _store = []


def add_documents(texts, sources, vectors, parent_ids=None):
    """
    Store document pieces together with their meaning-vectors.

    Input:
        texts      -> list of chunk strings
        sources    -> list of file names
        vectors    -> list of embedding vectors (one per chunk)
        parent_ids -> optional list of parent ids (for parent-document retrieval)
    """
    if parent_ids is None:
        parent_ids = [None] * len(texts)

    if _use_pinecone():
        index = _get_pinecone_index()
        items = []
        for text, source, vector, parent_id in zip(texts, sources, vectors, parent_ids):
            items.append({
                "id": uuid.uuid4().hex,                        # unique id per chunk
                "values": [float(x) for x in vector],          # the embedding
                "metadata": {                                   # stored alongside
                    "text": text, "source": source, "parent_id": parent_id or "",
                },
            })
        # Upsert in batches (Pinecone likes reasonable batch sizes).
        for i in range(0, len(items), 100):
            index.upsert(vectors=items[i:i + 100])
    else:
        for text, source, vector, parent_id in zip(texts, sources, vectors, parent_ids):
            _store.append({"text": text, "source": source, "vector": vector, "parent_id": parent_id})


def search(query_vector, top_k=5):
    """
    Find the stored pieces closest in meaning to the query vector.

    Output: list of {"text","source","score"} sorted best-first.
    """
    if _use_pinecone():
        index = _get_pinecone_index()
        res = index.query(
            vector=[float(x) for x in query_vector],
            top_k=top_k,
            include_metadata=True,
        )
        matches = res["matches"] if "matches" in res else res.matches
        results = []
        for m in matches:
            score = m["score"] if isinstance(m, dict) else m.score
            md = (m["metadata"] if isinstance(m, dict) else m.metadata) or {}
            results.append({
                "text": md.get("text", ""),
                "source": md.get("source", ""),
                "parent_id": md.get("parent_id", ""),
                "score": float(score),
            })
        return results
    else:
        if len(_store) == 0:
            return []
        results = []
        for item in _store:
            closeness = float(np.dot(query_vector, item["vector"]))
            results.append({
                "text": item["text"],
                "source": item["source"],
                "parent_id": item.get("parent_id"),
                "score": closeness,
            })
        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:top_k]


def count():
    """How many vectors are stored."""
    if _use_pinecone():
        stats = _get_pinecone_index().describe_index_stats()
        return int(stats.get("total_vector_count", 0) or 0)
    return len(_store)
