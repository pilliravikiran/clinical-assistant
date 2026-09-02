"""
scripts/evaluate_retrieval.py
=============================

Measure how good our retrieval is, using a small golden set.

Run:  python scripts/evaluate_retrieval.py
"""

# --- Make sure Python can find the "app" package ---
# When you run a script inside scripts/, Python only searches scripts/ by
# default. These two lines add the PROJECT ROOT to the search path so
# "import app..." works.
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services import rag_service, retrieval_service, rerank_service
from app.services import eval_service


# GOLDEN SET: each question + the document that SHOULD be retrieved for it.
# (We wrote the sample docs, so we know the right answers.)
GOLDEN = [
    {"question": "What follow-up care was recommended?", "expected_source": "discharge_summary.txt"},
    {"question": "high blood pressure",                  "expected_source": "physician_note.txt"},
    {"question": "cholesterol and glucose results",      "expected_source": "lab_report.txt"},
    {"question": "antibiotics after pneumonia",          "expected_source": "discharge_summary.txt"},
    {"question": "diabetes medication plan",             "expected_source": "physician_note.txt"},
]


def retrieve_basic(question):
    """First-stage only: hybrid search, NO re-ranking."""
    return retrieval_service.hybrid_search(question, top_k=3, candidate_k=6)


def retrieve_advanced(question):
    """Full: hybrid search + cross-encoder re-ranking."""
    return rag_service.retrieve_chunks(question, top_k=3, candidate_k=6)


if __name__ == "__main__":
    # Load the documents into the search stores first.
    rag_service.ingest_folder("data", chunk_size=200)

    print("Evaluating on", len(GOLDEN), "golden questions (k=3)\n")

    for name, fn in [("BASIC (hybrid only)", retrieve_basic),
                     ("ADVANCED (hybrid + rerank)", retrieve_advanced)]:
        report = eval_service.evaluate(GOLDEN, fn, k=3)
        print(f"=== {name} ===")
        print(f"  Hit-rate@3 : {report['hit_rate']:.2f}")
        print(f"  MRR        : {report['mrr']:.2f}")
        for d in report["details"]:
            mark = "OK " if d["hit"] else "MISS"
            print(f"    [{mark}] rr={d['reciprocal_rank']:.2f}  '{d['question']}' -> expected {d['expected']}, got {d['got_order']}")
        print()
