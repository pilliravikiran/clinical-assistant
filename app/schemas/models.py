"""
app/schemas/models.py
=====================

The "shapes" of our API requests and responses, using Pydantic.

Why: FastAPI uses these classes to (1) automatically CHECK that incoming data
is valid (e.g. a question is a non-empty string), and (2) describe the JSON
that goes out. If a request has the wrong shape, FastAPI returns a clear error
automatically - we don't have to write that checking ourselves.
"""

from pydantic import BaseModel, Field


# ---------- Requests (data coming IN) ----------

class AskRequest(BaseModel):
    """Body for POST /ask."""
    # Field(...) means required. min_length=1 rejects empty questions.
    question: str = Field(..., min_length=1, description="The clinical question to answer")
    top_k: int = Field(3, ge=1, le=20, description="How many chunks to use")


class SearchRequest(BaseModel):
    """Body for POST /search."""
    query: str = Field(..., min_length=1, description="Text to search for")
    top_k: int = Field(5, ge=1, le=20, description="How many results to return")


# ---------- Responses (data going OUT) ----------

class HealthResponse(BaseModel):
    """Body returned by GET /health."""
    status: str
    mode: str            # "offline" or "online"


class IngestResponse(BaseModel):
    """Body returned by POST /ingest."""
    chunks_ingested: int


class SearchResultItem(BaseModel):
    """One search result."""
    text: str
    source: str
    score: float


class SearchResponse(BaseModel):
    """Body returned by POST /search."""
    results: list[SearchResultItem]


class AskResponse(BaseModel):
    """Body returned by POST /ask."""
    answer: str
    sources: list[str]   # citations


class ChatRequest(BaseModel):
    """Body for POST /chat (conversational, with memory)."""
    session_id: str = Field(..., min_length=1, description="Conversation id (keeps history)")
    message: str = Field(..., min_length=1, description="The user's message")


class ChatResponse(BaseModel):
    """Body returned by POST /chat."""
    answer: str
    sources: list[str]
    standalone_question: str   # the condensed follow-up we actually searched
    turns: int                 # how many turns in this conversation so far


class StructuredAskResponse(BaseModel):
    """Body returned by POST /ask/structured."""
    answer: str
    confidence: float          # 0..1, how confident the answer is
    sources: list[str]


class AgentRequest(BaseModel):
    """Body for POST /agent (the tool-choosing agent)."""
    question: str = Field(..., min_length=1)


class AgentResponse(BaseModel):
    """Body returned by POST /agent."""
    answer: str
    tool_used: str             # which tool the agent chose
    sources: list[str]
