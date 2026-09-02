# Method Flow & Terminology Map (Project 1 — Tenet Clinical AI)

Trace an example through EVERY file and method, with the interview
TERMINOLOGY labeled at each step.

Example question: "What follow-up care was recommended?"

---

## FLOW 1 — INGESTION (offline: prepare documents)  [Technique: Document Ingestion Pipeline]

```
main.py: lifespan()  (on server startup)   OR   routes.py: ingest()  (POST /ingest)
   |
   v
rag_service.py: ingest_folder("data")
   |-- reads each .txt file
   |-- text_utils.py: clean_text()      [Technique: Text Normalization / Cleaning]
   |-- text_utils.py: chunk_text()      [Technique: Recursive/boundary-aware Chunking + overlap]
   |-- embedding_service.py: embed_documents()   [Technique: Dense Embeddings (bi-encoder), L2-normalized]
   |-- vector_service.py: add_documents()        [Technique: Dense Vector Index]
   |-- keyword_service.py: add_documents()       [Technique: Sparse/BM25 Keyword Index]
   v
6 chunks stored in BOTH a dense (vector) index and a sparse (BM25) index.
```

Data at each step:
- raw file text -> clean_text -> tidy text
- chunk_text -> ["Follow-up Care Recommended: Finish the full 7-day...", ...]
- embed_documents -> list of 384-number vectors (one per chunk)
- add_documents (vector) -> stored {text, source, vector}
- add_documents (keyword) -> BM25 index built from tokenized chunks

---

## FLOW 2 — QUERY / ANSWER (online: RAG)  [Technique: Retrieval-Augmented Generation]

```
POST /ask  {"question": "What follow-up care was recommended?"}
   |
   v
routes.py: ask()                         [Technique: REST API endpoint; Pydantic validation]
   |
   v
rag_service.py: answer_question()
   |
   |-- guardrails_service.py: redact_phi(question)   [Technique: PHI/PII redaction for logging]
   |-- logging_utils.py: get_logger()->info()        [Technique: Structured logging]
   |
   |-- rag_service.py: retrieve_chunks()
   |      |
   |      |-- retrieval_service.py: hybrid_search()   [Technique: Hybrid Search]
   |      |      |-- embedding_service.py: embed_query()   [Technique: Asymmetric query embedding]
   |      |      |-- vector_service.py: search()           [Technique: Dense retrieval, cosine/top-K]
   |      |      |-- keyword_service.py: search()          [Technique: Sparse retrieval (BM25)]
   |      |      |-- retrieval_service.py: reciprocal_rank_fusion()  [Technique: RRF fusion]
   |      |
   |      |-- rerank_service.py: rerank()             [Technique: Cross-encoder Re-ranking (stage 2)]
   |
   |-- guardrails_service.py: passes_relevance()      [Technique: Grounded refusal / relevance threshold]
   |
   |-- llm_service.py: generate_answer()
   |      |-- llm_service.py: build_user_message()    [Technique: Prompt engineering / grounding]
   |      |-- _answer_with_mock()  OR  _answer_with_claude()   [Technique: LLM inference; provider abstraction]
   |
   v
returns {answer, sources}   [Technique: Citations / source attribution]
   |
   v
routes.py: ask() -> AskResponse -> JSON to user
```

Data at each step (debugging view):
- question = "What follow-up care was recommended?"
- redact_phi -> same (no PHI here)
- embed_query -> one 384-number vector (with BGE query prefix)
- vector_service.search -> [discharge chunk (0.76), ...]  (dense candidates)
- keyword_service.search -> [discharge chunk (2.11), ...]  (sparse candidates)
- reciprocal_rank_fusion -> merged ranked list
- rerank -> discharge chunk on top (rerank_score +7.11)
- passes_relevance -> True (7.11 >= threshold 0.0)
- build_user_message -> "Context:\n<chunks>\n\nQuestion: ..."
- generate_answer -> answer text
- sources -> ["discharge_summary.txt", ...]

---

## FLOW 3 — EVALUATION (offline: measure quality)  [Technique: Retrieval Evaluation]

```
scripts/evaluate_retrieval.py
   |-- rag_service.py: ingest_folder()
   |-- rag_service.py: retrieve_chunks()   (per golden question)
   |-- eval_service.py: evaluate()
          |-- reciprocal_rank()   [Technique: MRR]
          |-- hit_at_k()          [Technique: Hit-rate@k]
```

```
scripts/track_experiments.py -> mlflow.log_params/log_metric  [Technique: Experiment tracking (MLflow)]
```

---

## TERMINOLOGY CHEAT-SHEET (name -> what it is -> where in our code)

| Terminology | What it means | Our file: method |
|---|---|---|
| Configuration management (12-factor, env vars) | Settings/secrets outside code | config.py |
| Synthetic data generation | Fake data instead of real PHI | scripts/generate_sample_data.py |
| Document ingestion pipeline | Prepare docs for retrieval (offline) | rag_service.py: ingest_folder |
| Text normalization / cleaning | Tidy raw text | text_utils.py: clean_text |
| Chunking (recursive, overlap) | Split docs into coherent pieces | text_utils.py: chunk_text |
| Dense embeddings (bi-encoder) | Text -> meaning vector | embedding_service.py: embed_documents/embed_query |
| Asymmetric embedding | Query embedded differently than docs | embedding_service.py: embed_query |
| Vector store / dense retrieval | Search by meaning (cosine, top-K) | vector_service.py: search |
| Sparse retrieval / BM25 | Search by keywords | keyword_service.py: search |
| Hybrid search | Combine dense + sparse | retrieval_service.py: hybrid_search |
| Reciprocal Rank Fusion (RRF) | Merge ranked lists by rank | retrieval_service.py: reciprocal_rank_fusion |
| Query transformation / multi-query | Rewrite question into variations | query_service.py + retrieval_service.py: multi_query_search |
| Cross-encoder re-ranking | Re-score (query,chunk) together | rerank_service.py: rerank |
| Two-stage retrieval | Fast recall then precise re-rank | retrieve_chunks (hybrid -> rerank) |
| RAG (retrieve-augment-generate) | Ground the LLM in retrieved text | rag_service.py: answer_question |
| Prompt engineering / grounding | System prompt rules the LLM | llm_service.py: SYSTEM_PROMPT/build_user_message |
| LLM inference | Generate the answer | llm_service.py: generate_answer |
| Provider abstraction | Swap LLM (mock/Claude) | llm_service.py: mock vs claude |
| Guardrail: grounded refusal | Refuse if not confident | guardrails_service.py: passes_relevance |
| Guardrail: PHI/PII redaction | Mask patient data | guardrails_service.py: redact_phi |
| Citations / source attribution | Return sources | rag_service.py: answer_question (sources) |
| Evaluation: Hit-rate, MRR | Measure retrieval quality | eval_service.py |
| Experiment tracking (MLflow) | Log params + metrics | scripts/track_experiments.py |
| REST API (FastAPI) | Expose over HTTP | routes.py, main.py |
| Schema validation (Pydantic) | Validate request/response | schemas/models.py |
| Structured logging (PHI-safe) | Record events safely | logging_utils.py |
| Testing (unit/integration/mocking) | Prove code works | tests/ |
```
