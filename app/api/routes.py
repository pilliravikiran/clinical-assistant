"""
app/api/routes.py
=================

The API endpoints (the "menu"). Each function below handles one endpoint.

Flow for a request:
    HTTP request -> FastAPI validates it against a schema -> the function here
    runs -> it calls rag_service -> we return a response object -> FastAPI turns
    it into JSON.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

import app.config as config
from app.services import (
    rag_service, retrieval_service, conversation_service, agent_service,
    monitoring_service,
)
from app.schemas.models import (
    AskRequest, AskResponse,
    SearchRequest, SearchResponse, SearchResultItem,
    IngestResponse, HealthResponse,
    ChatRequest, ChatResponse,
    AgentRequest, AgentResponse,
    StructuredAskResponse,
)

# A router groups related endpoints. main.py plugs this into the app.
router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health():
    """
    GET /health - a simple "is the service alive?" check.
    Used by load balancers / monitoring. Always fast, no heavy work.
    """
    return HealthResponse(status="ok", mode=config.APP_MODE)


@router.post("/ingest", response_model=IngestResponse)
def ingest():
    """
    POST /ingest - load the documents in the data folder into the pipeline.
    Returns how many chunks were stored.
    """
    try:
        count = rag_service.ingest_folder("data")
    except Exception as error:
        # If ingestion breaks, return a 500 with a clear message.
        raise HTTPException(status_code=500, detail="Failed to ingest documents.") from error
    return IngestResponse(chunks_ingested=count)


@router.post("/search", response_model=SearchResponse)
def search(request: SearchRequest):
    """
    POST /search - hybrid search over the documents (no LLM answer).

    Request body: {"query": "...", "top_k": 5}
    Response: a list of matching chunks with scores.
    """
    results = retrieval_service.hybrid_search(request.query, top_k=request.top_k)

    # Convert plain dicts into typed response items.
    items = [
        SearchResultItem(text=r["text"], source=r["source"], score=r["score"])
        for r in results
    ]
    return SearchResponse(results=items)


@router.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    """
    POST /ask - the full RAG pipeline: retrieve -> rerank -> guardrail -> answer.

    Request body: {"question": "...", "top_k": 3}
    Response: {"answer": "...", "sources": [...]}
    """
    try:
        result = rag_service.answer_question(request.question, top_k=request.top_k)
    except RuntimeError as error:
        # llm_service raises RuntimeError for AI-service problems -> 503.
        monitoring_service.record_error()
        raise HTTPException(status_code=503, detail=str(error)) from error

    return AskResponse(answer=result["answer"], sources=result["sources"])


@router.get("/metrics")
def metrics():
    """GET /metrics - counters, rates, and average latency for monitoring."""
    return monitoring_service.get_metrics()


@router.post("/ask/structured", response_model=StructuredAskResponse)
def ask_structured(request: AskRequest):
    """POST /ask/structured - returns {answer, confidence, sources} as JSON."""
    try:
        result = rag_service.answer_question_structured(request.question, top_k=request.top_k)
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return StructuredAskResponse(
        answer=result["answer"], confidence=result["confidence"], sources=result["sources"]
    )


@router.post("/ask/stream")
def ask_stream(request: AskRequest):
    """POST /ask/stream - streams the answer token-by-token (plain text)."""
    def generate():
        for piece in rag_service.stream_answer(request.question, top_k=request.top_k):
            yield piece
    return StreamingResponse(generate(), media_type="text/plain")


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    POST /chat - conversational RAG WITH memory.

    Request: {"session_id": "abc", "message": "..."}
    Keeps per-session history and condenses follow-ups into standalone questions.
    Response: {answer, sources, standalone_question, turns}
    """
    try:
        result = conversation_service.chat(request.session_id, request.message)
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    return ChatResponse(
        answer=result["answer"],
        sources=result["sources"],
        standalone_question=result["standalone_question"],
        turns=result["turns"],
    )


@router.post("/agent", response_model=AgentResponse)
def agent(request: AgentRequest):
    """
    POST /agent - the tool-choosing agent (LangGraph).

    The agent decides which tool to use (search / summarize / list) for the
    question, runs it, and returns the answer plus which tool it chose.
    """
    try:
        result = agent_service.run_agent(request.question)
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    return AgentResponse(
        answer=result["answer"],
        tool_used=result["tool_used"],
        sources=result["sources"],
    )
