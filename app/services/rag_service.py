"""
app/services/rag_service.py
===========================

The ORCHESTRATOR - it ties the whole RAG pipeline together.

Two jobs:
  1. ingest_folder()  -> load documents into the search stores (the "remember" side)
  2. answer_question() -> the full pipeline (the "answer" side):

        question
          -> hybrid_search   (dense + BM25 + RRF)      [stage 1: recall]
          -> rerank          (cross-encoder)           [stage 2: precision]
          -> guardrail       (refuse if nothing relevant)
          -> build context   (best chunks, best-first)
          -> generate_answer (grounded LLM / mock)
          -> return answer + citations (sources)

This is the file you point to in an interview and say "this is my RAG pipeline".
"""

import os
import time
import uuid

import app.config as config
from app.utils.text_utils import clean_text, chunk_text, extract_metadata
from app.utils.logging_utils import get_logger
from app.utils.tracing import traceable

# One logger for this module. Used to record events (never raw PHI).
logger = get_logger("rag_service")
from app.services.embedding_service import embed_documents
from app.services import (
    vector_service,
    keyword_service,
    retrieval_service,
    rerank_service,
    llm_service,
    guardrails_service,
    compression_service,
    parent_service,
    cache_service,
    monitoring_service,
)


def ingest_folder(folder="data", chunk_size=None):
    """
    Load every .txt document in a folder into BOTH search stores.

    Steps per document: read -> clean -> chunk -> embed -> store (dense + BM25).

    Input:
        folder     -> where the .txt documents live (default "data")
        chunk_size -> characters per chunk (default from config)
    Output:
        the total number of chunks stored (an integer).

    Called by: startup / an /ingest API endpoint (later).
    """
    # Resolve a bare folder name (like "data") relative to the PROJECT ROOT so
    # it works regardless of the current working directory (terminal/debugger).
    if not os.path.isabs(folder):
        _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        folder = os.path.join(_root, folder)

    # Start fresh so we don't double-store on repeated calls.
    vector_service.reset()
    keyword_service.reset()
    parent_service.reset()
    cache_service.reset()   # documents changed -> old cached answers are stale

    total_chunks = 0
    for filename in sorted(os.listdir(folder)):
        if not filename.endswith(".txt"):
            continue

        # Read and clean the document.
        with open(os.path.join(folder, filename), encoding="utf-8") as f:
            clean = clean_text(f.read())

        # Read its metadata (type / id / date) from the header.
        meta = extract_metadata(clean)
        prefix = f"Document type: {meta['document_type']}. Source: {filename}. "

        # Build the list of CHILD chunks to index, each tagged with a parent_id.
        children = []
        child_parent_ids = []
        if config.USE_PARENT_DOCUMENT:
            # Split into big PARENT sections, then small CHILD chunks per parent.
            parents = chunk_text(clean, chunk_size=config.PARENT_CHUNK_SIZE)
            for parent in parents:
                parent_id = uuid.uuid4().hex
                parent_service.put(parent_id, parent)        # store parent in the docstore
                for child in chunk_text(parent, chunk_size=config.CHILD_CHUNK_SIZE):
                    children.append(child)
                    child_parent_ids.append(parent_id)       # tag child with its parent id
        else:
            # Plain chunking: children are the document chunks (no parent ids).
            children = chunk_text(clean, chunk_size=(chunk_size or config.CHILD_CHUNK_SIZE))
            child_parent_ids = [None] * len(children)

        sources = [filename] * len(children)

        # CONTEXTUAL RETRIEVAL: embed each child WITH a document-context prefix
        # (keeps meaning), but STORE the plain child (used for the answer/citation).
        if config.USE_CONTEXTUAL_RETRIEVAL:
            to_embed = [prefix + c for c in children]
        else:
            to_embed = children

        vector_service.add_documents(children, sources, embed_documents(to_embed), parent_ids=child_parent_ids)
        keyword_service.add_documents(children, sources, parent_ids=child_parent_ids)

        total_chunks += len(children)

    return total_chunks


def retrieve_chunks(question, top_k=None, candidate_k=10):
    """
    The RETRIEVAL half only (no LLM): hybrid search -> re-rank.

    Kept separate so evaluation can measure retrieval quality without paying
    for answer generation.

    Output: the top re-ranked chunks (list of dicts with text/source/rerank_score).
    """
    if top_k is None:
        top_k = config.TOP_K

    # STAGE 1 - hybrid recall: pull a pool of candidates (dense + BM25 + RRF).
    candidates = retrieval_service.hybrid_search(
        question, top_k=candidate_k, candidate_k=candidate_k
    )

    # STAGE 2 - precision: cross-encoder re-ranks the pool down to the best top_k.
    return rerank_service.rerank(question, candidates, top_k=top_k)


@traceable(name="rag.answer_question")
def answer_question(question, top_k=None, candidate_k=10):
    """
    Run the full RAG pipeline for one question.

    Input:
        question    -> the user's question (string)
        top_k       -> how many chunks to keep after re-ranking (default config)
        candidate_k -> how many candidates to pull in stage 1 (recall pool)

    Output:
        a dictionary:
        {
          "answer":  <the written answer>,
          "sources": <list of source file names used as citations>,
          "chunks":  <the top chunks used, with their rerank scores>
        }

    Called by: the /ask API endpoint (built next step).
    """
    start = time.perf_counter()

    # SEMANTIC CACHE - if a very similar question was answered, reuse it.
    if config.USE_SEMANTIC_CACHE:
        cached = cache_service.get(question)
        if cached is not None:
            logger.info("Cache hit - returning cached answer")
            monitoring_service.record_request(time.perf_counter() - start, cache_hit=True)
            return cached

    ctx = _build_context(question, top_k=top_k, candidate_k=candidate_k)
    if not ctx["ok"]:
        monitoring_service.record_request(time.perf_counter() - start, refused=True)
        return {"answer": ctx["refusal"], "sources": [], "chunks": []}

    # GENERATE - grounded answer (mock offline, Claude online).
    answer = llm_service.generate_answer(question, ctx["context_pieces"])

    logger.info("Answer generated with %d source(s): %s", len(ctx["sources"]), ctx["sources"])
    result = {"answer": answer, "sources": ctx["sources"], "chunks": ctx["chunks"]}

    # Store in the semantic cache for next time.
    if config.USE_SEMANTIC_CACHE:
        cache_service.put(question, result)

    monitoring_service.record_request(time.perf_counter() - start)
    return result


def _build_context(question, top_k=None, candidate_k=10):
    """
    Shared retrieval half used by answer_question, the structured variant, and
    the streaming variant: retrieve -> guardrail -> parent-expand -> compress.

    Output dict:
        ok             -> True if we have relevant context, False if we should refuse
        refusal        -> the refusal message (when ok is False)
        context_pieces -> list of context strings for the LLM
        sources        -> citation source file names
        chunks         -> the retrieved chunks (for debugging)
    """
    # Log the request - but REDACT the question first (never log raw PHI).
    logger.info("Question received (redacted): %s", guardrails_service.redact_phi(question))

    # GUARDRAIL - block prompt-injection / jailbreak attempts before the LLM.
    if config.USE_PROMPT_INJECTION_GUARD and guardrails_service.detect_prompt_injection(question):
        logger.warning("Blocked: potential prompt injection")
        return {
            "ok": False,
            "refusal": "This request can't be processed. Please ask a clinical question.",
            "context_pieces": [], "sources": [], "chunks": [],
        }

    top_chunks = retrieve_chunks(question, top_k=top_k, candidate_k=candidate_k)
    logger.info("Retrieved %d chunks after re-ranking", len(top_chunks))

    # GUARDRAIL - refuse if nothing relevant enough was retrieved.
    if not guardrails_service.passes_relevance(top_chunks):
        logger.warning("Refused: no sufficiently relevant context found")
        return {
            "ok": False,
            "refusal": "I could not find relevant information in the documents to answer that.",
            "context_pieces": [], "sources": [], "chunks": [],
        }

    # PARENT-DOCUMENT expansion, then CONTEXTUAL COMPRESSION.
    expanded = parent_service.expand_to_parents(top_chunks) if config.USE_PARENT_DOCUMENT else top_chunks
    context_chunks = compression_service.compress_chunks(question, expanded) if config.USE_COMPRESSION else expanded
    context_pieces = [chunk["text"] for chunk in context_chunks]

    # CITATIONS - unique source documents, in order.
    sources = []
    for chunk in top_chunks:
        if chunk["source"] not in sources:
            sources.append(chunk["source"])

    return {"ok": True, "refusal": "", "context_pieces": context_pieces,
            "sources": sources, "chunks": top_chunks}


def answer_question_structured(question, top_k=None, candidate_k=10):
    """
    Like answer_question, but returns STRUCTURED output: an answer plus a
    confidence score (and sources). Uses llm_service.generate_structured.
    """
    ctx = _build_context(question, top_k=top_k, candidate_k=candidate_k)
    if not ctx["ok"]:
        return {"answer": ctx["refusal"], "confidence": 0.0, "sources": []}

    structured = llm_service.generate_structured(question, ctx["context_pieces"])
    return {
        "answer": structured["answer"],
        "confidence": structured["confidence"],
        "sources": ctx["sources"],
    }


def stream_answer(question, top_k=None, candidate_k=10):
    """
    Generator that STREAMS the answer token-by-token (for chat-like UX).
    Yields text chunks. If nothing relevant is found, yields the refusal once.
    """
    ctx = _build_context(question, top_k=top_k, candidate_k=candidate_k)
    if not ctx["ok"]:
        yield ctx["refusal"]
        return
    yield from llm_service.stream_answer(question, ctx["context_pieces"])
