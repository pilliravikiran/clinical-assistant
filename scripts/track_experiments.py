"""
scripts/track_experiments.py
============================

Run several RAG retrieval experiments with different settings and log each one
to MLflow, so we can compare them and pick the best configuration.

Run:  python scripts/track_experiments.py
Then: mlflow ui    (and open http://127.0.0.1:5000 to see the runs)
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlflow

from app.services import rag_service, eval_service


# Small golden set (question -> the document that SHOULD be retrieved).
GOLDEN = [
    {"question": "What follow-up care was recommended?", "expected_source": "discharge_summary.txt"},
    {"question": "high blood pressure",                  "expected_source": "physician_note.txt"},
    {"question": "cholesterol and glucose results",      "expected_source": "lab_report.txt"},
    {"question": "antibiotics after pneumonia",          "expected_source": "discharge_summary.txt"},
    {"question": "diabetes medication plan",             "expected_source": "physician_note.txt"},
]


# The different settings we want to try. Each dict is one experiment.
EXPERIMENTS = [
    {"chunk_size": 120, "top_k": 3, "candidate_k": 6},
    {"chunk_size": 200, "top_k": 3, "candidate_k": 6},
    {"chunk_size": 300, "top_k": 3, "candidate_k": 6},
]


def make_retriever(top_k, candidate_k):
    """Return a function question -> retrieved chunks, using these settings."""
    def retrieve(question):
        return rag_service.retrieve_chunks(question, top_k=top_k, candidate_k=candidate_k)
    return retrieve


if __name__ == "__main__":
    # All runs go under this experiment name in MLflow.
    mlflow.set_experiment("tenet-rag-retrieval")

    print("Running", len(EXPERIMENTS), "experiments and logging to MLflow...\n")

    for cfg in EXPERIMENTS:
        # Re-ingest with this chunk size (rebuilds the stores).
        rag_service.ingest_folder("data", chunk_size=cfg["chunk_size"])

        # Evaluate retrieval quality with this config.
        report = eval_service.evaluate(
            GOLDEN, make_retriever(cfg["top_k"], cfg["candidate_k"]), k=cfg["top_k"]
        )

        # Log ONE MLflow run: its parameters (settings) + metrics (results).
        with mlflow.start_run():
            mlflow.log_params(cfg)                          # the settings
            mlflow.log_metric("hit_rate", report["hit_rate"])
            mlflow.log_metric("mrr", report["mrr"])

        print(f"  chunk_size={cfg['chunk_size']:>3}  ->  "
              f"hit_rate={report['hit_rate']:.2f}  mrr={report['mrr']:.2f}")

    print("\nDone. Runs logged to MLflow (folder: mlruns/).")
    print("View them with:  mlflow ui   then open http://127.0.0.1:5000")
