# Resume Points — Tenet Clinical AI (Project 1)

A living list of **honest, interview-ready** resume bullets, added as we build.
Scope is realistic for an **AI/ML Engineer** — it does not claim the frontend,
cloud networking, or authentication that other teams own.

> How to use: pick the bullets that match the level of the job, and be ready
> to explain the "why" behind each one (the notes under each bullet).

---

## Project summary line (top of resume)
- Built an internal **Clinical Knowledge Assistant** (Retrieval-Augmented
  Generation) that lets clinicians ask natural-language questions and get
  answers grounded strictly in source documents, with safe "no answer found"
  behavior — using Python, Hugging Face embeddings, a vector database,
  LangChain, and FastAPI.

---

## Component bullets (added as we build)

### Configuration & secrets management
- Implemented 12-factor-style configuration with environment variables and a
  `.env` template, keeping API keys and secrets out of source control.
  - *Why it matters:* shows security awareness — secrets never reach GitHub.

### Synthetic data generation (privacy-safe development)
- Created a synthetic clinical-document generator to enable development and
  testing **without any real PHI (patient data)**.
  - *Why it matters:* demonstrates awareness of healthcare data-privacy rules.

### Document chunking (production technique)
- Engineered a boundary-aware document chunking pipeline using **LangChain's
  RecursiveCharacterTextSplitter** with configurable size and overlap, splitting
  on paragraph/sentence boundaries to preserve semantic coherence and improve
  retrieval quality.
  - *Why it matters:* chunk quality drives retrieval and answer quality; this is
    the real production approach, not naive fixed-size cutting.
  - *Interview soundbite:* "If a chunk cuts a sentence in half, the model sees a
    broken idea. Recursive, boundary-aware splitting keeps each chunk a complete
    unit of meaning, which reduces hallucination."

### Embeddings / semantic representation
- Implemented a semantic embedding service using the **BAAI/bge-small-en-v1.5**
  model (Hugging Face / sentence-transformers), applying production techniques:
  **asymmetric query/document embedding** and **L2-normalized vectors** for
  cosine-via-dot-product similarity.
  - *Why it matters:* enables meaning-based search (e.g. "high blood pressure"
    matches "hypertension" at 0.85 similarity with zero shared words).
  - *Interview soundbite:* "I normalize embeddings so similarity is a fast dot
    product, and I embed queries with an instruction prefix because BGE was
    trained for asymmetric retrieval — both are standard production details."

### Semantic search / vector retrieval
- Built a semantic retrieval layer that stores document embeddings and returns
  top-K nearest chunks by cosine similarity, enabling meaning-based search
  (e.g. a query for "high blood pressure" retrieves a chunk mentioning
  "hypertension"). Designed for a swappable backend (in-memory now,
  **FAISS/Pinecone** in production).
  - *Interview soundbite:* "A vector store just keeps embeddings and returns the
    nearest ones by cosine similarity; FAISS and Pinecone are optimized
    implementations of that same idea."

### LLM integration & grounded prompting
- Integrated an LLM answer layer (Anthropic **Claude** via the official SDK)
  behind a **provider-abstraction interface** with an offline mock fallback,
  using a grounding **system prompt** that constrains answers to retrieved
  context and enforces "say you don't know" behavior to reduce hallucination.
  Included typed error handling (rate-limit / connection / API errors).
  - *Why it matters:* grounding + refusal behavior is the core safety
    requirement for clinical/regulated RAG.
  - *Interview soundbite:* "The rest of the app calls one generate_answer()
    function; swapping the LLM provider or mocking it for tests is a config
    change, not a rewrite."

### Advanced retrieval (hybrid + fusion + re-ranking)
- Built a production-grade two-stage retriever: **hybrid search** combining
  dense (BGE embeddings) and sparse (**BM25**) retrieval, merged with
  **Reciprocal Rank Fusion (RRF)**, followed by **cross-encoder re-ranking**
  (sentence-transformers CrossEncoder) to maximize top-result precision.
  - *Why it matters:* dense+sparse cover opposite failure modes; the
    cross-encoder reads (query, chunk) jointly for far better ordering than the
    first-stage bi-encoder.
  - *Interview soundbite:* "First stage uses a bi-encoder for cheap recall over
    everything; second stage uses a cross-encoder to re-rank the shortlist for
    precision. RRF merges the dense and sparse lists by rank so their
    incompatible score scales don't matter."

### End-to-end RAG orchestration with citations
- Orchestrated a full RAG pipeline (`answer_question`): hybrid retrieval ->
  cross-encoder re-ranking -> grounded generation, returning the answer with
  **source citations** and a guardrail for missing information.
  - *Why it matters:* citations + grounded refusal are trust/safety
    requirements for clinical use.

### Query transformation & retrieval evaluation
- Added **multi-query** transformation (LLM-generated query variations fused
  with RRF) to improve recall, and built a **retrieval evaluation harness**
  with a golden set measuring **Hit-rate@k** and **MRR** to validate retrieval
  changes with numbers rather than intuition.
  - *Interview soundbite:* "I don't guess whether a retrieval change helps - I
    measure it against a golden set with Hit-rate and MRR."

### Safety guardrails (grounded refusal + PHI redaction)
- Added guardrails: a **relevance-threshold refusal** (returns "I don't know"
  when the best re-ranked chunk is below a confidence cutoff) and **PHI/PII
  redaction** before logging, addressing healthcare data-privacy requirements.
  - *Interview soundbite:* "In regulated domains I refuse low-confidence answers
    and redact PHI before logging - a wrong confident answer is worse than a
    safe 'I don't know'."

### REST API (FastAPI)
- Exposed the RAG system as a **FastAPI** REST service (`/health`, `/ingest`,
  `/search`, `/ask`) with **Pydantic** request/response validation, typed error
  handling, and correct HTTP status codes (422 validation, 503 AI-unavailable),
  plus auto-generated OpenAPI/Swagger docs.
  - *Interview soundbite:* "Pydantic schemas give me automatic request
    validation and self-documenting APIs - invalid input returns a 422 without
    any manual checks."

### Structured, PHI-safe logging
- Implemented structured logging (timestamp | level | module | message) with
  **PHI/PII redaction** applied to user input before it is logged, and
  severity levels (INFO/WARNING) for events vs. refusals.
  - *Interview soundbite:* "Logs never contain raw patient data - I redact PHI
    before logging and log events/counts, not sensitive content."

### Testing (unit + integration + mocking)
- Wrote a **pytest** suite: unit tests for text utils and guardrails, plus
  integration tests for the FastAPI endpoints using **mocking** (monkeypatch) to
  isolate the API layer from heavy models and the LLM.
  - *Interview soundbite:* "I mock the RAG pipeline and model loading in API
    tests so they run in seconds and test one layer at a time."

### Experiment tracking (MLflow)
- Used **MLflow Tracking** to log RAG retrieval experiments (chunk size, top-K
  as params; Hit-rate and MRR as metrics) and compare runs to choose the best
  configuration; documented the Model Registry (Staging -> Production) workflow.
  - *Interview soundbite:* "MLflow logs params and metrics per run so I compare
    configs objectively and reproduce results, and the registry versions models
    from staging to production."

### Transformer NLP + supervised classification
- Built a clinical note-type classifier framed as **supervised multi-class
  classification**: transformer **embeddings as features**, a scikit-learn
  **Logistic Regression** model, evaluated with a **train/test split** and
  **accuracy**; class probabilities via softmax. (Fine-tuning with LoRA/QLoRA
  demonstrated separately on Colab GPU.)
  - *Interview soundbite:* "I used embeddings as features and a logistic
    regression head - a cheap, strong baseline - and measured generalization
    with a held-out test split before considering full fine-tuning."

### Transformer fine-tuning with LoRA/PEFT
- Fine-tuned a transformer (DistilBERT) for clinical-note classification using
  **LoRA (PEFT)** on GPU, training only ~1% of parameters (adapter weights)
  instead of the full model, with Hugging Face `transformers` + `peft`.
  Understood epochs, learning rate, loss, transfer learning, and QLoRA (4-bit).
  - *Interview soundbite:* "LoRA freezes the base model and trains small adapter
    matrices - comparable accuracy at a fraction of compute, and the adapter is
    a few MB to ship."

### Managed vector database (Pinecone)
- Integrated **Pinecone** (serverless managed vector DB) behind a swappable
  backend interface, with automatic index creation (384-dim, cosine),
  batched upserts with metadata, and top-K query - persistent storage so
  ingestion is decoupled from serving (no re-embedding on restart). Keeps a
  free in-memory backend for local dev via one config switch.
  - *Interview soundbite:* "Ingestion upserts embeddings into Pinecone once;
    the API just queries it - retrieval scales independently of the app, and
    adding documents is an upsert, not a redeploy."

### Retrieval robustness (contextual retrieval + calibration)
- Implemented **Contextual Retrieval**: enriched each chunk with document
  context (type/source) before embedding so isolated chunks retain meaning and
  match more query phrasings, and **calibrated the relevance threshold** on
  observed reranker scores to eliminate false refusals of valid questions while
  still blocking off-topic queries.
  - *Interview soundbite:* "I calibrate the refusal threshold on a validation
    set - too strict rejects valid questions, too loose hallucinates on
    off-topic ones."

### Contextual compression (post-retrieval)
- Added **contextual compression**: each retrieved chunk is trimmed to only the
  sentences within a relative similarity margin of the query, cutting tokens and
  noise before generation while preserving relevant content.

### Parent-document (small-to-big) retrieval
- Implemented **parent-document retrieval**: index small child chunks for
  precise matching but expand each hit to its larger parent section for the LLM,
  giving precise recall with full-context generation.

### Conversational RAG (multi-turn memory)
- Added a **conversational layer**: per-session buffer memory (windowed) plus
  **follow-up condensing** (rewrite a follow-up into a standalone question using
  history) before retrieval, exposed via a stateful `/chat` endpoint alongside
  the stateless `/ask`.
  - *Interview soundbite:* "I make RAG multi-turn by keeping per-session history
    and condensing each follow-up into a standalone query before retrieval - so
    'and her dosage?' resolves against the conversation."
- Added **semantic long-term memory**: every turn is embedded and stored, and
  the most relevant past turns are recalled by meaning (a vector store over the
  conversation) - so context beyond the recent buffer window is not lost.
  - *Interview soundbite:* "Beyond a recent-turns buffer, I keep a vector store
    of the whole conversation and semantically recall relevant old turns - not
    just the last N messages."

### Agentic AI (LangGraph)
- Built a **LangGraph agent** that routes each question to the right tool
  (search via RAG / summarize / list) using a state graph with nodes,
  conditional edges, and shared state; exposed via an `/agent` endpoint.
  Offline uses rule-based routing; online swaps in LLM tool-calling with the
  same graph.
  - *Interview soundbite:* "The agent is a LangGraph state machine - a router
    node picks a tool via a conditional edge, the tool runs, then a respond node
    finalizes. Plain RAG is a fixed pipeline; the agent decides."

### Structured output + streaming
- Added **structured output** (LLM returns JSON `{answer, confidence}`, parsed
  and validated) and **streaming responses** (token-by-token via a FastAPI
  StreamingResponse) - both behind the provider abstraction (mock offline,
  Claude `messages.stream` online).
  - *Interview soundbite:* "Structured output gives machine-readable JSON with a
    confidence field; streaming yields token deltas for a typing UX - I refactored
    retrieval into a shared context builder so /ask, /ask/structured and
    /ask/stream reuse the same pipeline."

### Semantic caching + prompt-injection defense
- Added a **semantic cache** (embedding-similarity keyed) that returns answers
  for near-identical questions ~30x faster, skipping retrieval and the LLM; and
  a **prompt-injection guardrail** that pattern-detects jailbreak attempts and
  blocks them before the LLM.
  - *Interview soundbite:* "Semantic caching keys on the question embedding, so
    rephrasings hit the cache, not just exact matches. I block prompt-injection
    with pattern rules before the model - production would add a classifier like
    Llama Guard."

### RAG generation evaluation (RAGAS) + observability (LangSmith)
- Added **RAGAS-style generation metrics** (faithfulness, answer-relevancy) to
  measure hallucination and on-topic-ness, and integrated **LangSmith** tracing
  via `@traceable` (env-var enabled) to observe every RAG/agent step end-to-end.
  - *Interview soundbite:* "Retrieval I measure with Hit-rate/MRR; generation with
    RAGAS faithfulness and answer-relevancy. Offline I approximate RAGAS with
    embeddings; production uses an LLM judge. LangSmith gives per-step traces
    and latency."

### Monitoring / metrics
- Instrumented the pipeline with metrics (request count, refusal rate, cache
  hit-rate, error count, average latency) exposed on a `/metrics` endpoint for
  Prometheus scraping / Grafana dashboards and alerting.

### Web UI (Streamlit)
- Built a **Streamlit** chat UI over the conversational RAG (session memory,
  grounded answers with sources, live monitoring metrics in the sidebar).

### Scalable parent-document store (production pattern)
- Refactored parent-document retrieval to the production pattern: children store
  a `parent_id` in vector-DB metadata and parents live in a **persistent SQLite
  docstore**, fetched by id on demand - removing in-memory parent storage so it
  scales to large corpora. Also size-capped the semantic cache.
  - *Interview soundbite:* "Children carry a parent_id in Pinecone metadata;
    parents sit in a docstore (SQLite here, Redis/DB at scale) fetched on
    retrieval - parents never sit in app RAM."

### Containerization + CI/CD
- Containerized the service with a layered **Dockerfile** (CPU-only PyTorch,
  cached dependency layer, `.dockerignore` keeping secrets/artifacts out) and a
  **GitHub Actions** CI pipeline (install -> lint with ruff -> pytest) that runs
  on every push/PR.
  - *Interview soundbite:* "CI runs the test suite on every push; the Docker
    image installs CPU torch and caches the deps layer so code changes don't
    reinstall everything. Secrets stay out of the image and are passed via
    --env-file at runtime."
