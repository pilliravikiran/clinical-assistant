# AI/ML Frameworks Guide — What each one is and when to use it

The single most important idea: **LangChain, LangGraph, LangSmith, and CrewAI
are mostly NOT competitors — they are different LAYERS of a stack.**

Analogy: running a restaurant kitchen.

| Tool        | What it is                                   | Kitchen analogy                    | Use it when |
|-------------|----------------------------------------------|------------------------------------|-------------|
| LangChain   | Toolkit to BUILD LLM apps (prompts, retrievers, splitters, integrations) | The tools & recipes | Building RAG, chains, any LLM feature. The default foundation. |
| LangGraph   | Build STATEFUL multi-step workflows/agents as a graph (nodes, edges, branching, loops) | The process flowchart | The agent needs real control flow: branching, retries, memory, human-in-the-loop |
| LangSmith   | OBSERVE / debug / evaluate / monitor LLM apps | CCTV + quality inspector | You want to see why an answer was wrong and measure quality over time |
| CrewAI      | MULTI-AGENT framework: role-playing agents collaborate | A team of specialist chefs | Multiple agents (researcher + writer + reviewer) working together |

Also good to know:
- LlamaIndex — RAG-focused alternative/complement to LangChain (strong at data indexing).
- AutoGen — Microsoft multi-agent framework (alternative to CrewAI).
- DSPy — automatic prompt optimization (advanced).

## How they layer together
Build with LangChain -> Orchestrate complex agents with LangGraph ->
Watch & evaluate with LangSmith -> Multi-agent collaboration with CrewAI.

## Coverage plan across the three projects

### Project 1 — Tenet (Healthcare) : GenAI / RAG / Agent showcase
- LangChain    : RAG pipeline (splitters, retriever, prompt, LLM)
- LangGraph    : the clinical agent workflow (validate -> retrieve -> answer)
- LangSmith    : tracing + RAG evaluation
- Also: Claude (LLM), Hugging Face (embeddings/transformers), Pinecone (vector DB), MLflow

### Project 2 — Morgan Stanley (Finance) : NLP + Multi-agent + Classic ML
- LangChain    : RAG over financial policies
- CrewAI       : multi-agent (research agent + compliance agent + summarizer)
- FAISS        : vector search
- BERT + NER   : document classification and entity extraction
- scikit-learn : fraud classification, customer clustering
- PostgreSQL   : structured data

### Project 3 — Meijer (Retail) : End-to-end classic ML / MLOps
- scikit-learn : demand forecasting (regression), purchase prediction (classification)
- Recommenders : collaborative + content-based filtering
- MLflow       : experiment tracking, model registry
- (optional)   : a small recommendation-explainer agent

## After all three you can honestly claim experience with:
LangChain, LangGraph, LangSmith, CrewAI, Hugging Face, PyTorch, Pinecone,
FAISS, scikit-learn, MLflow, FastAPI, Docker, PostgreSQL.
