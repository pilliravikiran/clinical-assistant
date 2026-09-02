# Production-Grade RAG — Complete Techniques Guide

The full landscape of RAG techniques, organized by pipeline stage, from naive
to state-of-the-art. Use this as your interview study reference. Each item has:
what it is / why it helps / [status in this project].

Status legend: [BUILD] = we implement it here, [DOC] = explained here for
interviews but not fully implemented (needs infra or is a variant).

---

## The RAG maturity ladder
- **Naive RAG:** embed -> vector search -> stuff top-K into prompt -> generate.
- **Advanced RAG:** add pre-retrieval (query transforms) and post-retrieval
  (re-ranking, compression) to raise retrieval quality.
- **Modular / Agentic RAG:** the model decides when/what to retrieve, grades
  results, and self-corrects (CRAG, Self-RAG, GraphRAG, multi-hop).

---

## STAGE 1 — Indexing / Ingestion (offline: prepare documents)

### Chunking strategies
- Fixed-size — cut every N chars. Simple, cuts mid-sentence. [naive]
- **Recursive / boundary-aware** — split on paragraph/sentence boundaries. [BUILD - done]
- **Semantic chunking** — start a new chunk when meaning shifts (embedding
  distance between sentences spikes). Better coherence. [DOC]
- **Sentence-window** — index single sentences but return a window of
  neighbors as context. [DOC]
- **Parent-document (small-to-big)** — search small chunks, but feed the LLM
  the larger parent section they belong to. [BUILD - later]
- Layout-aware — respect tables, headings, lists (for PDFs). [DOC]

### Embedding techniques
- Strong open models: **BGE**, GTE, E5, Nomic. [BUILD - BGE done]
- Asymmetric query/document embedding. [BUILD - done]
- L2-normalized vectors (cosine via dot product). [BUILD - done]
- Fine-tuned / domain-adapted embeddings (train on your own pairs). [DOC]
- Multi-vector (ColBERT-style late interaction) — token-level match. [DOC]
- Matryoshka embeddings — truncate dimensions to trade speed/quality. [DOC]

### Enrichment at index time
- **Metadata** — attach type/date/source for filtering + citations. [BUILD]
- **Contextual Retrieval (Anthropic)** — prepend an LLM-written 1-2 sentence
  context to each chunk before embedding, so isolated chunks keep meaning.
  Big accuracy win. [BUILD - later]
- Multi-representation — also index an LLM summary or hypothetical questions
  per chunk. [DOC]
- Deduplication + incremental/versioned indexing. [DOC]

---

## STAGE 2 — Pre-retrieval (transform the query before searching)

- **Query rewriting** — clean/expand a messy question. [BUILD]
- **Multi-query** — generate several paraphrases, search all, merge results.
  Improves recall. [BUILD]
- **HyDE (Hypothetical Document Embeddings)** — ask the LLM to draft a fake
  answer, embed THAT, and search with it (answers look like documents). [DOC/BUILD-optional]
- **Step-back prompting** — ask a broader question first for background. [DOC]
- **Query decomposition** — split a complex question into sub-questions
  (multi-hop). [DOC]
- **Routing** — classify the query and send it to the right index/tool
  (adaptive RAG). [DOC]
- **Self-query / metadata extraction** — pull filters from the query
  ("2025 discharge summaries" -> filter type=discharge, year=2025). [DOC]

---

## STAGE 3 — Retrieval (find candidate chunks)

- **Dense retrieval** — vector similarity. [BUILD - done]
- **Sparse / lexical retrieval (BM25)** — keyword scoring; great for exact
  terms, codes, names. [BUILD]
- **Hybrid search** — combine dense + sparse. [BUILD]
- **Fusion: Reciprocal Rank Fusion (RRF)** — merge two ranked lists by rank,
  no score-scaling needed. Standard hybrid method. [BUILD]
- **MMR (Maximal Marginal Relevance)** — pick results that are relevant AND
  diverse (avoid near-duplicates). [BUILD - optional]
- Metadata filtering — restrict by type/date/permissions. [BUILD]

---

## STAGE 4 — Post-retrieval (refine the candidates)

- **Re-ranking (cross-encoder)** — a second model reads (query, chunk) TOGETHER
  and scores true relevance; reorder top candidates. Biggest quality lever.
  Models: bge-reranker, Cohere Rerank. [BUILD]
- **Contextual compression** — trim each chunk to only the relevant sentences
  before sending to the LLM. Saves tokens, reduces noise. [DOC/BUILD-optional]
- **Context ordering ("lost in the middle")** — LLMs attend best to the start
  and end; place the most relevant chunks there. [BUILD]
- Deduplicate / cap total context length (token budget). [BUILD]

---

## STAGE 5 — Generation (write the grounded answer)

- **Grounded system prompt** — answer only from context; refuse if absent. [BUILD - done]
- **Citations / source attribution** — return which chunk/document each claim
  came from. [BUILD]
- **Structured output** — JSON with {answer, sources, confidence}. [BUILD - optional]
- **Guardrails** — block unsafe output; PHI/PII handling. [DOC]
- Streaming responses for UX. [DOC]

---

## STAGE 6 — Evaluation (prove it works)

- **Retrieval metrics:** Hit-rate, MRR, Recall@k, nDCG. [BUILD]
- **RAGAS metrics:** faithfulness (answer supported by context), answer
  relevancy, context precision, context recall. [BUILD - later]
- **LLM-as-judge** — an LLM scores answer quality against a rubric. [DOC]
- Golden/eval datasets of question->expected answer. [BUILD - small one]

---

## STAGE 7 — Ops & Observability (run it in production)

- **Tracing** — LangSmith / OpenTelemetry to see every step + latency. [DOC/BUILD-later]
- **Caching** — embedding cache; semantic cache for repeat questions. [DOC]
- **Monitoring** — latency, error rate, retrieval score distribution, drift. [BUILD - later]
- **Feedback loop** — thumbs up/down -> improve retrieval / fine-tune. [DOC]
- **Security** — per-user metadata filters (row-level access), audit logs. [DOC]

---

## Advanced architectures (know these names for interviews)
- **Agentic RAG** — an agent decides when to retrieve, which tool/index, and
  can loop. (We build a LangGraph agent later.) [BUILD - later]
- **Corrective RAG (CRAG)** — grade retrieved docs; if weak, fall back to web
  search or re-query. [DOC]
- **Self-RAG** — model emits reflection tokens to decide retrieve/critique. [DOC]
- **GraphRAG** — build a knowledge graph from documents; retrieve subgraphs for
  global/multi-hop questions. [DOC]
- **Multi-hop RAG** — chain retrievals to answer questions needing several
  linked facts. [DOC]

---

## One-line interview soundbites
- "Naive RAG is embed-search-stuff-generate. Production RAG adds pre-retrieval
  query transforms and post-retrieval re-ranking."
- "Hybrid search = dense (meaning) + BM25 (keywords), fused with Reciprocal
  Rank Fusion."
- "A cross-encoder re-ranker reads the query and chunk together, so it scores
  relevance far more accurately than the bi-encoder used for first-stage recall."
- "Contextual Retrieval prepends an LLM-written context to each chunk before
  embedding, which fixes the 'isolated chunk lost its meaning' problem."
- "I evaluate retrieval with hit-rate/MRR and generation with RAGAS
  faithfulness so I can prove changes actually help."
