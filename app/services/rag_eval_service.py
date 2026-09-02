"""
app/services/rag_eval_service.py
================================

RAGAS-style evaluation of GENERATION quality (not just retrieval).

Two core RAGAS metrics:
  - faithfulness      : is the answer supported by the retrieved context?
                        (low = hallucination)
  - answer_relevancy  : does the answer address the question?

NOTE: the real RAGAS library uses an LLM as a judge. Here we approximate these
with EMBEDDING similarity so it runs offline and free. The concept and the
interpretation are the same; production would use LLM-graded RAGAS.
"""

import numpy as np

from app.services.embedding_service import embed_query, embed_documents


def faithfulness(answer, context_pieces):
    """
    How well is the answer supported by the retrieved context (0..1).
    Approximation: the best similarity between the answer and any context piece.
    """
    if not answer or not context_pieces:
        return 0.0
    a_vec = embed_query(answer)
    c_vecs = embed_documents(context_pieces)
    return float(max(np.dot(a_vec, c) for c in c_vecs))


def answer_relevancy(answer, question):
    """
    How well the answer addresses the question (0..1).
    Approximation: similarity between the answer and the question.
    """
    if not answer or not question:
        return 0.0
    return float(np.dot(embed_query(answer), embed_query(question)))


def evaluate_generation(question, answer, context_pieces):
    """
    Return both RAGAS-style metrics for one answer.

    Output: {"faithfulness": float, "answer_relevancy": float}
    """
    return {
        "faithfulness": round(faithfulness(answer, context_pieces), 3),
        "answer_relevancy": round(answer_relevancy(answer, question), 3),
    }
