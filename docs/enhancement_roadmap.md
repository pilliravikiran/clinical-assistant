# Project 1 Enhancement Roadmap - Complete Advanced GenAI

Additions to make Project 1 a complete advanced GenAI engineering portfolio.
Status: [x] done, [ ] planned.

## Retrieval robustness (before + after retrieval)
- [x] Contextual Retrieval - enrich chunks with document context before embedding
- [x] Metadata extraction (type / id / date)
- [x] Relevance-threshold calibration (no false refusals, still blocks off-topic)
- [x] Contextual compression - trim retrieved chunks to only relevant sentences (after retrieval)
- [x] Parent-document / small-to-big retrieval (search small, return larger parent)
- [ ] Metadata filtering (e.g. "only lab reports", "only 2025")
- [ ] HyDE (hypothetical document embeddings) as an optional query transform

## Conversational RAG (memory / multi-turn)
- [x] Chat history / memory per session
- [x] Follow-up condensing (history + follow-up -> standalone question)
- [x] Session management (session_id)
- [x] /chat endpoint (stateful) alongside /ask (stateless)
- [x] Memory types: buffer (recent turns, windowed) [summary memory optional]
- [x] Semantic long-term memory (L4): recall RELEVANT past turns by meaning (vector memory)
- [ ] Summary memory (L3) - optional
- [ ] Entity memory / persistent store (Redis) - optional

## Advanced GenAI features
- [x] LangGraph agent + tool calling (search / summarize / list) + /agent endpoint
- [x] Structured output (JSON: answer + confidence + sources) + /ask/structured
- [x] Streaming responses (token-by-token) + /ask/stream
- [x] Semantic caching (cache answers for similar questions) - ~30x faster on repeats
- [x] Prompt-injection guardrail (block "ignore your instructions" attacks)
- [ ] Prompt versioning (v1/v2/v3 compare)

## Evaluation & observability
- [x] RAGAS-style evaluation (faithfulness, answer relevancy) - offline embedding proxy
- [x] LangSmith tracing hook (@traceable; enable via env vars)
- [x] Monitoring metrics (latency, error rate, refusal rate, cache hit-rate) + /metrics

## Shipping
- [x] Streamlit UI (chat page with live metrics + memory)
- [x] Docker (Dockerfile + .dockerignore)
- [x] CI/CD (GitHub Actions: install -> lint -> pytest)
- [ ] README + interview_notes
- [ ] Push to GitHub

## Build order (recommended)
1. Contextual compression + parent-document (finish retrieval robustness)
2. LangGraph agent + tool calling + structured output
3. Streaming + semantic caching + prompt-injection guardrail
4. RAGAS + LangSmith + monitoring
5. Streamlit UI
6. Docker + CI/CD + README + push
