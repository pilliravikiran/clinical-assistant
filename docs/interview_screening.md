# Recruiter Screening - Answers (Clinical RAG Assistant)

Personalize the [FILL IN] parts (salary, location, work authorization).

## Project one-liner
Internal Clinical Knowledge Assistant - a RAG system that lets clinicians ask
natural-language questions and get answers grounded strictly in source documents
(notes, discharge summaries, labs), with citations and a safe "I don't know".

## Technologies
Python, FastAPI, Pydantic, uvicorn | LangChain, LangGraph, LangSmith, Anthropic
Claude | RAG: hybrid (dense + BM25) + RRF + cross-encoder rerank + multi-query +
contextual retrieval/compression + parent-document | Hugging Face,
sentence-transformers (BGE), PyTorch, cross-encoder reranker, LoRA/PEFT |
Pinecone, BM25, SQLite docstore | MLflow, logging, monitoring (/metrics), pytest
| Guardrails: grounded refusal, PHI redaction, prompt-injection, semantic cache.

## 1. Salary
"I'm flexible and focused on the right role. Based on the market for an AI/ML
engineer in [location] and my experience, I'm targeting ~$[X]-$[Y], open to the
full package." (Research market first; give a range; deflect if too early.)

## 2. Location
"I'm in [City, State], open to [remote/hybrid/relocation]."

## 3. Walk me through an AI solution you designed and deployed
See the RAG pipeline: ingestion (clean -> recursive chunk to child+parent ->
BGE embed -> Pinecone upsert, parents in docstore) -> query (hybrid dense+BM25 ->
RRF -> cross-encoder rerank -> parent expand -> compression) -> grounded Claude
answer with citations -> guardrails (relevance refusal, PHI redaction,
prompt-injection) -> FastAPI (search/ask/stream/chat/agent). Eval: Hit-rate/MRR
+ RAGAS-style; MLflow; LangSmith tracing.

## 4. What I personally owned
The AI/ML service end-to-end: ingestion, embeddings + vector search, hybrid +
re-ranking retrieval, RAG orchestration + prompting, guardrails, FastAPI
endpoints, evaluation, monitoring. Frontend, auth, and cloud networking were
other teams; I integrated with the existing auth layer.

## 5. Tradeoffs
- BGE-small embeddings (speed/cost vs a bit of accuracy).
- ms-marco cross-encoder reranker (speed) vs bge-reranker/Cohere (accuracy).
- Hybrid (dense + BM25) vs pure vector - keyword catches exact terms.
- Custom pipeline vs full LangChain - control/understanding; LangChain where it
  clearly helped (splitter, LangGraph agent).
- RAG (pretrained) vs fine-tuning - facts change, so RAG for knowledge.
- Pinecone (managed) with local fallback for dev.

## 6. How the business used it
Clinicians ask a question, get a grounded, cited answer in seconds instead of
manually reading many documents; refusal + citations give a safety guarantee a
plain chatbot can't.

## 7. Rebuild differently
- Move state (memory/cache/session) to Redis for horizontal scale.
- LLM-graded RAGAS instead of embedding proxy.
- Contextual retrieval (LLM chunk context) + fine-tuned domain embeddings.
- Ingestion as a separate offline job with incremental upserts + versioning.
- Full observability (LangSmith + Prometheus/Grafana), stronger reranker.
- Guardrails: Llama Guard classifier + Presidio PII detector.

## 8. Work authorization
"[U.S. Citizen / Green Card / EAD awaiting GC / X visa], authorized to work
[with/without sponsorship]." (Answer factually in one line.)
