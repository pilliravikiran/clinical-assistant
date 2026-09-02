"""
app/main.py
===========

The FastAPI application entry point. It:
  1. Creates the app.
  2. Plugs in our endpoints (the router from api/routes.py).
  3. Loads the documents once when the server starts.

Run the server with:
    uvicorn app.main:app --reload

Then open http://127.0.0.1:8000/docs for an interactive API page.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.services import rag_service


# The "lifespan" handler is the modern FastAPI way to run startup/shutdown code.
# Everything BEFORE `yield` runs at startup; everything after runs at shutdown.
@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP: load the documents once so /ask and /search work immediately.
    count = rag_service.ingest_folder("data")
    print(f"[startup] Ingested {count} chunks. API is ready.")
    yield
    # SHUTDOWN: (nothing to clean up for now)


# Create the application. title/description show up on the /docs page.
app = FastAPI(
    title="Tenet Clinical AI",
    description="A grounded clinical question-answering API (RAG).",
    version="0.1.0",
    lifespan=lifespan,
)

# Attach all our endpoints from routes.py.
app.include_router(router)


@app.get("/")
def root():
    """A friendly landing message at the base URL."""
    return {"message": "Tenet Clinical AI is running. See /docs for the API."}
