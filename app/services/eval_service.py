"""
app/services/eval_service.py
============================

Evaluation for RETRIEVAL quality. We use a small "golden set" of questions,
each paired with the document that SHOULD be retrieved, and compute:

  - Hit-rate@k : did the correct document appear in the top-k?  (recall)
  - MRR        : how high did the correct document rank?         (ordering)

Higher is better for both (max 1.0).
"""


def _ordered_unique_sources(results):
    """
    From a ranked list of chunk-results, return the source file names in order,
    without repeats. (Several top chunks may come from the same document.)
    """
    sources = []
    for r in results:
        if r["source"] not in sources:
            sources.append(r["source"])
    return sources


def reciprocal_rank(results, expected_source):
    """
    1 / (rank of the first correct document). Returns 0.0 if not found.

    Example: correct doc is 2nd -> 1/2 = 0.5.
    """
    for rank, source in enumerate(_ordered_unique_sources(results), start=1):
        if source == expected_source:
            return 1.0 / rank
    return 0.0


def hit_at_k(results, expected_source, k):
    """
    True if the correct document is among the top-k sources.
    """
    return expected_source in _ordered_unique_sources(results)[:k]


def evaluate(golden, retrieve_fn, k=3):
    """
    Run every golden question through retrieve_fn and compute the metrics.

    Input:
        golden      -> list of {"question": ..., "expected_source": ...}
        retrieve_fn -> a function: question -> ranked list of chunk results
        k           -> the k for hit-rate@k

    Output:
        {"hit_rate": ..., "mrr": ..., "details": [per-question rows]}
    """
    total_rr = 0.0
    hits = 0
    details = []

    for item in golden:
        results = retrieve_fn(item["question"])
        rr = reciprocal_rank(results, item["expected_source"])
        hit = hit_at_k(results, item["expected_source"], k)

        total_rr += rr
        hits += 1 if hit else 0

        details.append({
            "question": item["question"],
            "expected": item["expected_source"],
            "got_order": _ordered_unique_sources(results),
            "reciprocal_rank": rr,
            "hit": hit,
        })

    n = len(golden)
    return {
        "hit_rate": hits / n if n else 0.0,
        "mrr": total_rr / n if n else 0.0,
        "details": details,
    }
