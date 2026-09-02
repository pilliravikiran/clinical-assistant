"""
app/services/embedding_service.py
=================================

Turns text into embeddings: lists of numbers that capture MEANING.

Industry techniques used here:
  1. Asymmetric embeddings - the question and the documents are embedded a
     little differently. We add a short instruction to the QUESTION only,
     because the BGE model retrieves better that way.
  2. Normalized vectors - each vector is scaled to length 1, so a fast
     dot-product equals cosine similarity (what vector databases use).

We load the model ONCE and reuse it (loading is slow; reusing is instant).
"""

from sentence_transformers import SentenceTransformer

import app.config as config


# Module-level cache. Filled the first time we load the model, then reused.
_model = None


def get_model():
    """
    Load the embedding model the first time it is needed, then reuse it.

    Output: a ready-to-use SentenceTransformer model.
    Called by: embed_documents() and embed_query().
    """
    global _model
    if _model is None:
        print(f"[embedding_service] Loading model: {config.EMBEDDING_MODEL}")
        _model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _model


def embed_documents(texts):
    """
    Turn a LIST of document chunks into a list of embedding vectors.

    Input:  texts -> a list of strings (the chunks)
    Output: a NumPy array of vectors (one per chunk), each length-1 normalized

    Why a list: encoding many chunks together (batching) is much faster
    than one at a time.

    Called by: the ingestion pipeline (when we store documents).
    """
    model = get_model()
    # normalize_embeddings=True -> vectors have length 1 (for cosine via dot-product)
    vectors = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return vectors


def embed_query(question):
    """
    Turn the USER'S QUESTION into a single embedding vector.

    Input:  question -> a string (what the user asked)
    Output: one NumPy vector, length-1 normalized

    Advanced detail (asymmetric embeddings):
        We prepend config.QUERY_PREFIX to the question. The BGE model was
        trained so that adding this instruction to the QUERY (not the
        documents) improves search accuracy.

    Called by: the search / RAG pipeline (when the user asks something).
    """
    model = get_model()
    text = config.QUERY_PREFIX + question
    vector = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
    return vector


def get_dimension():
    """
    How many numbers are in each vector (e.g. 384 for bge-small).
    The vector database index must be created with this same number.
    """
    return get_model().get_sentence_embedding_dimension()
