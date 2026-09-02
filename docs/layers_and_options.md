# System Layers, Technology Options, and What We Used

Every layer of the RAG/GenAI system, the full menu of options (with
terminology), and what THIS project uses (marked USED).

Layer order:
1 Data/Ingestion -> 2 Chunking -> 3 Embeddings -> 4 Vector store -> 5 Keyword
-> 6 Pre-retrieval -> 7 Retrieval/Fusion -> 8 Post-retrieval -> 9 Generation
-> 10 Guardrails -> 11 Conversation -> 12 Orchestration/Agents -> 13 API
-> 14 Evaluation -> 15 MLOps -> 16 Observability -> 17 Fine-tuning -> 18 Testing
-> 19 Deployment -> 20 Config/Secrets

## 1. Data / Ingestion
- Sources: txt, PDF, DOCX, HTML, DB, S3 | USED: txt
- Parsing: PyPDF, PyMuPDF, pdfplumber, Unstructured, LlamaParse, OCR (Tesseract, AWS Textract) | USED: plain read
- Cleaning: regex, ftfy, unidecode | USED: regex clean_text
- Metadata: header parse, NER | USED: header extract_metadata
- Data: real (de-identified), synthetic | USED: synthetic generator

## 2. Chunking
- Strategy: fixed, recursive, token-based, semantic, sentence-window, parent-document, layout-aware | USED: recursive + parent-document
- Params: size, overlap | USED: 800 parent / 200 child, overlap 50
- Library: LangChain, LlamaIndex | USED: LangChain

## 3. Embeddings
- Open: BGE, GTE, E5, Nomic, MiniLM, MPNet | USED: BAAI/bge-small-en-v1.5
- Commercial: OpenAI text-embedding-3, Cohere, Voyage
- Advanced: ColBERT (multi-vector), matryoshka, fine-tuned
- Techniques: bi-encoder, asymmetric, L2 normalization | USED: all three

## 4. Vector storage
- DBs: Pinecone, Weaviate, Milvus, Qdrant, Chroma, FAISS, pgvector | USED: Pinecone + local
- Index (ANN): HNSW, IVF, IVF-PQ, flat | USED: Pinecone-managed
- Metric: cosine, dot, euclidean | USED: cosine

## 5. Keyword / sparse
- Method: BM25, TF-IDF, Elasticsearch/OpenSearch, SPLADE | USED: BM25
- Library: rank-bm25, Elasticsearch | USED: rank-bm25

## 6. Pre-retrieval (query transform)
- Techniques: rewriting, multi-query, HyDE, step-back, decomposition, routing, self-query | USED: multi-query

## 7. Retrieval / fusion
- Retrieval: dense, sparse, hybrid | USED: hybrid
- Fusion: RRF, weighted, MMR | USED: RRF

## 8. Post-retrieval
- Rerank: cross-encoder (bge-reranker, ms-marco, Cohere Rerank), LLM rerank | USED: ms-marco cross-encoder
- Compression: contextual compression, LLMChainExtractor | USED: contextual compression
- Expansion: parent-document, sentence-window | USED: parent-document
- Ordering: lost-in-the-middle handling | USED: best-first

## 9. Generation (LLM)
- Models: Claude, GPT, Gemini, Llama, Mistral | USED: Claude + mock
- Self-host serving: Ollama, vLLM, TGI
- Params: temperature, max_tokens, context window | USED: temp 0, max_tokens
- Prompting: zero/few-shot, chain-of-thought, grounding | USED: grounded system prompt
- Design: provider abstraction | USED: yes

## 10. Guardrails / safety
- Grounding: relevance-threshold refusal | USED: yes (calibrated)
- PII/PHI: regex, Presidio, NER | USED: regex redaction
- Prompt-injection: Llama Guard, NeMo Guardrails, Guardrails AI | PLANNED
- Output: schema validation, moderation | PLANNED

## 11. Conversation / memory
- Memory: buffer, window, summary, entity | PLANNED
- Follow-up: query condensing/contextualization | PLANNED
- Session store: in-memory, Redis | PLANNED

## 12. Orchestration / agents
- Build: custom pipeline, LangChain LCEL, LlamaIndex | USED: custom + LangChain splitter
- Agents: LangGraph, CrewAI, AutoGen | PLANNED (LangGraph P1, CrewAI P2)
- Mechanism: tool/function calling, agent loop | PLANNED

## 13. API
- Framework: FastAPI, Flask, Django | USED: FastAPI
- Protocol: REST, GraphQL, gRPC | USED: REST
- Validation: Pydantic | USED: Pydantic
- Server: uvicorn, gunicorn | USED: uvicorn

## 14. Evaluation
- Retrieval: Hit-rate, MRR, nDCG, Recall@k | USED: Hit-rate, MRR
- Generation: RAGAS (faithfulness, answer-relevancy, context precision/recall), BLEU/ROUGE/BERTScore, LLM-as-judge | PLANNED: RAGAS

## 15. MLOps / experiment tracking
- Tracking: MLflow, Weights & Biases | USED: MLflow
- Versioning: model registry, DVC | USED: MLflow registry (concept)

## 16. Observability / monitoring
- LLM tracing: LangSmith, LangFuse, Phoenix/Arize | PLANNED: LangSmith
- Metrics: Prometheus+Grafana, CloudWatch | PLANNED
- Logging: structured, PHI-safe | USED

## 17. Fine-tuning / ML training
- LLM: full, LoRA, QLoRA, PEFT, Unsloth | USED: LoRA/PEFT (Colab)
- Classifiers: Logistic Regression, MLP, SVM, trees | USED: LogReg + MLP demo

## 18. Testing
- Framework: pytest, unittest | USED: pytest
- Types: unit, integration, e2e | USED: unit + integration
- Isolation: mocking (monkeypatch), fixtures | USED: monkeypatch

## 19. Deployment / infra
- Container: Docker, docker-compose | PLANNED
- Orchestration: Kubernetes, ECS | PLANNED (concept)
- IaC: Terraform | PLANNED (concept)
- CI/CD: GitHub Actions, Jenkins | PLANNED
- Cloud: AWS (S3, SageMaker, Lambda, ECR), Azure, GCP | PLANNED (concept) + Pinecone cloud

## 20. Config / secrets
- Config: env vars/.env, YAML | USED: .env + python-dotenv
- Secrets: Vault, cloud secret managers, .gitignore | USED: .env git-ignored
