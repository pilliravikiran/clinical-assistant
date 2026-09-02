"""
app/services/retrieval_service.py
=================================

The RETRIEVAL orchestrator. It combines our two searches into one:

  dense retrieval (semantic / vectors)  +  sparse retrieval (keyword / BM25)
                         |
                         v
          Reciprocal Rank Fusion (RRF)  -> one merged ranked list

Terminology:
  - dense retrieval  : search by meaning (vector_service)
  - sparse retrieval : search by keywords (keyword_service, BM25)
  - hybrid search    : run both, then fuse
  - fusion (RRF)     : merge ranked lists by RANK, not by raw score
                       (raw scores from the two searches are on different
                        scales, so we use position instead)

Later steps add a cross-encoder RE-RANKER on top of this.
"""

import app.config as config
from app.services.embedding_service import embed_query
from app.services import vector_service, keyword_service, query_service


def reciprocal_rank_fusion(ranked_lists, k=60, top_k=5):
    """
    Merge several ranked lists into one, using Reciprocal Rank Fusion.

    Input:
        ranked_lists -> a list of ranked result-lists. Each result is a dict
                        with at least "text" and "source".
        k            -> RRF constant (standard value 60). Larger k = ranks
                        matter less; smaller k = top ranks dominate.
        top_k        -> how many fused results to return.

    Output:
        one fused, sorted list of results. Each item's "score" is its RRF score.

    How:
        For every list, a document at position `rank` (starting at 1) earns
        1 / (k + rank). We add these up across all lists. A document that
        ranks high in MANY lists gets the highest total.
    """
    fused_scores = {}   # text -> summed RRF score
    items_by_text = {}  # text -> the original result dict (to return later)

    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, start=1):   # rank starts at 1
            key = item["text"]                            # identify the doc by its text
            fused_scores[key] = fused_scores.get(key, 0.0) + 1.0 / (k + rank)
            items_by_text[key] = item

    # Build the merged list with the new RRF score attached.
    fused = []
    for key, score in fused_scores.items():
        merged_item = {
            "text": items_by_text[key]["text"],
            "source": items_by_text[key]["source"],
            "parent_id": items_by_text[key].get("parent_id"),
            "score": score,   # this is now the RRF score
        }
        fused.append(merged_item)

    # Best RRF score first.
    fused.sort(key=lambda r: r["score"], reverse=True)
    return fused[:top_k]


def hybrid_search(question, top_k=None, candidate_k=10, rrf_k=60):
    """
    Run BOTH searches and fuse them with RRF.

    Input:
        question    -> the user's question (string)
        top_k       -> how many final results to return (default from config)
        candidate_k -> how many candidates to pull from EACH search before
                       fusing (the "first-stage recall" pool)
        rrf_k       -> the RRF constant

    Output:
        a fused, ranked list of results (dicts with text/source/score).

    Called by: rag_service (the full pipeline), built next.
    """
    if top_k is None:
        top_k = config.TOP_K

    # Dense (semantic) candidates. embed_query adds the BGE query prefix.
    dense_results = vector_service.search(embed_query(question), top_k=candidate_k)

    # Sparse (keyword/BM25) candidates.
    sparse_results = keyword_service.search(question, top_k=candidate_k)

    # Merge the two ranked lists into one.
    return reciprocal_rank_fusion([dense_results, sparse_results], k=rrf_k, top_k=top_k)


def multi_query_search(question, top_k=None, candidate_k=10, rrf_k=60):
    """
    Multi-query retrieval: rewrite the question into several variations, run a
    hybrid search for EACH variation, then fuse all the lists with RRF.

    Why: different phrasings surface different relevant documents. Fusing them
    improves recall (we find more of the right chunks).

    Input:
        question    -> the user's question (string)
        top_k       -> how many final results to return (default from config)
        candidate_k -> candidates per variation before fusing
        rrf_k       -> the RRF constant

    Output:
        a fused, ranked list of results (dicts with text/source/score).

    Called by: rag_service when multi-query is enabled.
    """
    if top_k is None:
        top_k = config.TOP_K

    # 1. Turn the question into several variations (original + rewrites).
    variations = query_service.generate_query_variations(question)

    # 2. Run a hybrid search for each variation -> a ranked list per variation.
    ranked_lists = []
    for variation in variations:
        ranked_lists.append(hybrid_search(variation, top_k=candidate_k, candidate_k=candidate_k))

    # 3. Fuse ALL the lists together with RRF.
    return reciprocal_rank_fusion(ranked_lists, k=rrf_k, top_k=top_k)
