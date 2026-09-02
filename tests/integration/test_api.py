"""
tests/integration/test_api.py
=============================

Integration tests for the FastAPI endpoints, using FastAPI's TestClient.
The heavy RAG pipeline is mocked so tests run fast and don't need real models.
"""

from fastapi.testclient import TestClient

import app.services.rag_service as rag_service
from app.main import app


def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


def test_ask_rejects_empty_question():
    # Pydantic should auto-reject an empty question with 422 (validation error).
    with TestClient(app) as client:
        response = client.post("/ask", json={"question": ""})
        assert response.status_code == 422


def test_ask_returns_answer_with_mocked_rag(monkeypatch):
    # MOCK the RAG pipeline: replace answer_question with a fake that returns a
    # fixed result. Now the test checks ONLY the API layer, fast and isolated.
    def fake_answer(question, top_k=3, candidate_k=10):
        return {"answer": "TEST ANSWER", "sources": ["doc.txt"], "chunks": []}

    monkeypatch.setattr(rag_service, "answer_question", fake_answer)

    with TestClient(app) as client:
        response = client.post("/ask", json={"question": "any question"})
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "TEST ANSWER"
        assert data["sources"] == ["doc.txt"]
