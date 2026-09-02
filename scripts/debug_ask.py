"""
scripts/debug_ask.py
====================

A simple entry point for DEBUGGING the whole RAG pipeline.

How to debug (VS Code):
  1. Open this file.
  2. Click to the LEFT of a line number to set a red breakpoint - a good first
     one is the "result = rag_service.answer_question(...)" line below.
  3. Press F5 and choose "Debug: ask a question".
  4. When it stops at your breakpoint, press F11 (Step Into) to go INSIDE
     answer_question, then keep stepping to watch every stage run.

Good places to set breakpoints (the pipeline stages):
  - rag_service.answer_question      (the orchestrator)
  - rag_service.retrieve_chunks      (retrieval)
  - retrieval_service.hybrid_search  (dense + BM25 + RRF)
  - rerank_service.rerank            (cross-encoder)
  - guardrails_service.passes_relevance
  - llm_service.generate_answer      (the answer)
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services import rag_service


# --- This is the "ingest at startup" step (loads + embeds the documents once) ---
print("Ingesting documents...")
chunk_count = rag_service.ingest_folder("data")
print(f"Ingested {chunk_count} chunks.\n")

# --- Ask one question. Put a breakpoint on the next line and Step Into (F11). ---
question = "What follow-up care was recommended?"
result = rag_service.answer_question(question, top_k=3)

print("QUESTION:", question)
print("ANSWER  :", result["answer"])
print("SOURCES :", result["sources"])
