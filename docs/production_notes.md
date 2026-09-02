# Production Notes - What's Production-Grade vs Simplified

Honest audit of every component: what is already production-shaped, what is a
dev simplification, and the production path for each. Goal: no hidden "toy"
shortcuts, and a clear upgrade story for interviews.

Legend: PROD = production-ready as-is; FIXED = just upgraded; DEV = simplified
for local dev (production path noted).

## Retrieval / storage
- Vector DB: **Pinecone** (serverless, cosine, metadata). PROD. Local in-memory
  backend exists only as a free fallback.
- Parent-document store: **SQLite docstore**, children carry `parent_id` in
  vector metadata, parents fetched on demand. FIXED (was in-memory dict).
  Production at large scale: Redis or a document DB instead of SQLite.
- BM25 keyword index: in-memory (rank-bm25), rebuilt on ingest. DEV.
  Production: Elasticsearch/OpenSearch, or Pinecone sparse vectors, for a
  persistent, scalable, incrementally-updatable lexical index.
- Embeddings / reranker: loaded once per process. PROD-ish. At scale: a
  dedicated embedding/rerank service (TGI, Cohere Rerank) with batching.

## Caching / memory / state
- Semantic cache: in-memory, now **size-capped (FIFO)**. FIXED (was unbounded).
  Production: Redis with TTL + eviction (or GPTCache), shared across instances.
- Conversation buffer + semantic long-term memory: per-process in-memory dicts.
  DEV. Production: Redis or a database keyed by session_id, shared across
  instances and surviving restarts.
- Monitoring counters: in-process. DEV. Production: Prometheus client exposing
  /metrics in Prometheus format, scraped by Prometheus, dashboarded in Grafana.

## Ingestion
- Runs at startup and rebuilds everything. DEV. Production: a SEPARATE offline
  ingestion job that upserts into Pinecone incrementally (with dedup and
  document versioning); the API only connects and queries - no re-embed on
  restart.

## API / ops
- Pinecone index auto-created in code if missing. DEV convenience. Production:
  provision the index via Infrastructure-as-Code (Terraform) ahead of time.
- Secrets in .env (git-ignored). PROD-ish for a single box. Production: a secret
  manager (AWS Secrets Manager / Vault) injected at deploy.
- Thread-safety: monitoring + parent docstore use locks. Cache/memory lists are
  not individually locked - fine single-instance; moving them to Redis makes
  this moot.

## Guardrails / safety
- Prompt-injection: pattern rules. DEV. Production: add a classifier (Llama
  Guard) in addition to patterns.
- PHI redaction: regex. DEV. Production: a dedicated PII detector (Presidio /
  NER) for names and free-text PII.

## Evaluation
- Retrieval: Hit-rate / MRR. PROD.
- Generation: RAGAS approximated with embeddings offline. DEV. Production:
  LLM-graded RAGAS (faithfulness, answer-relevancy) for sharp scores.

## Summary
The RAG core (hybrid + RRF + cross-encoder rerank + parent-document via docstore
+ contextual retrieval/compression + Pinecone + grounded generation + guardrails
+ eval + observability hooks) is production-shaped. The remaining DEV items are
in-memory state (cache/conversation/memory/monitoring) and the offline
ingestion job - all of which move to Redis / a DB / Prometheus / a scheduled
job in a real deployment, with the interfaces already in place to swap them.
