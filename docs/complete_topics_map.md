# Complete AI/ML Interview Topics — Coverage Map

Goal: after the 3 projects, be able to answer ANY common AI/ML interview
question because you actually built something that uses the concept.

Legend: P1 = Tenet (Healthcare, GenAI/RAG), P2 = Morgan Stanley (Finance,
NLP + multi-agent + classic ML), P3 = Meijer (Retail, end-to-end classic ML).

---

## 1. Programming & Data
| Topic | Where covered |
|---|---|
| Python (clean, typed, tested) | P1, P2, P3 |
| Pandas / NumPy | P1, P2, P3 |
| SQL (SELECT, JOIN, GROUP BY, indexes) | P2 (PostgreSQL), P3 |
| Data validation (Pydantic, Great Expectations) | P1 (Pydantic), P3 (GE concept) |

## 2. Classic Machine Learning
| Topic | Where |
|---|---|
| Regression (demand forecasting) | P3 |
| Classification (fraud, purchase prediction) | P2, P3 |
| Clustering (customer segmentation) | P2 |
| Feature engineering | P2, P3 |
| Train/test split, cross-validation | P2, P3 |
| Hyperparameter tuning (GridSearchCV/Optuna) | P3 |
| Metrics: accuracy, precision, recall, F1, ROC-AUC, RMSE, MAE | P2, P3 |
| Imbalanced data, threshold tuning, PR curves | P2 |
| Overfitting, regularization, bias-variance | P2, P3 |

## 3. Deep Learning & NLP
| Topic | Where |
|---|---|
| PyTorch basics (tensors, inference) | P1 |
| Transformers & attention (concept) | P1, P2 |
| Tokenization | P1, P2 |
| Embeddings (sentence-transformers, BGE) | P1, P2 |
| BERT fine-tuning (document classification) | P2 |
| Named Entity Recognition (NER) | P2 |
| Summarization | P1 |
| Transfer learning / fine-tuning concept | P1, P2 |

## 4. Generative AI / LLM / RAG
| Topic | Where |
|---|---|
| LLM prompting (system/user, temperature, tokens) | P1 |
| Retrieval-Augmented Generation (RAG) | P1, P2 |
| Vector databases: Pinecone, FAISS (+ Chroma/Weaviate concept) | P1 (Pinecone), P2 (FAISS) |
| Semantic search, cosine similarity, top-K | P1, P2 |
| Asymmetric embeddings (query vs document) | P1 |
| Hybrid search (keyword + vector) | P1 |
| Cross-encoder re-ranking | P1 |
| Prompt versioning & guardrails | P1 |
| Hallucination mitigation / grounding | P1 |
| RAG evaluation (RAGAS, faithfulness, hit-rate) | P1 |
| LangChain (build) | P1, P2 |
| LangGraph (stateful agent workflow) | P1 |
| LangSmith (tracing + evaluation) | P1 |
| CrewAI (multi-agent) | P2 |
| Agents, tool/function calling, agent loop | P1 (LangGraph), P2 (CrewAI) |
| Provider abstraction (Claude + mock, swappable) | P1 |

## 5. MLOps
| Topic | Where |
|---|---|
| Experiment tracking (MLflow) | P1, P2, P3 |
| Model registry / versioning (MLflow) | P3 |
| Data/version control (DVC concept) | P3 |
| Drift monitoring (Evidently) | P3 |
| Metrics/monitoring (Prometheus + Grafana concept) | P1 |
| Structured logging | P1, P2, P3 |
| Automated retraining pipeline | P3 |
| Orchestration (Airflow/Prefect concept) | P3 |

## 6. Software Engineering & Testing
| Topic | Where |
|---|---|
| REST APIs (FastAPI) | P1, P2, P3 |
| Request/response schemas (Pydantic) | P1, P2, P3 |
| Error handling & HTTP status codes | P1, P2, P3 |
| Unit + integration tests (pytest, mocking) | P1, P2, P3 |
| Git / GitHub / PR workflow | all |
| RBAC / auth concepts | P2 |

## 7. Deployment & Cloud (production-grade)
| Topic | Where |
|---|---|
| Docker (multi-stage builds) | P1, P2, P3 |
| docker-compose (app + db + vector store) | P2 |
| Kubernetes manifests (deployment/service) | P1 (concept + yaml) |
| CI/CD (GitHub Actions: test -> lint -> build) | all |
| Model serving (FastAPI; BentoML/vLLM/TorchServe concept) | P1 |
| AWS: S3, ECR, SageMaker, Lambda | P1, P2 |
| Infra as Code (Terraform concept) | P1 |
| Secrets management (.env, cloud secret managers) | all |

## 8. Security & Privacy
| Topic | Where |
|---|---|
| PHI / healthcare data privacy | P1 |
| Financial data confidentiality | P2 |
| Secrets out of source control | all |
| What to log vs. never log | all |

---

Note: "concept" means we implement a realistic local version and clearly
document how the production/cloud version works — so it is truthful on a
resume and you can explain it, without requiring paid cloud to run locally.
