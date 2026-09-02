"""
tests/integration/conftest.py
=============================

Shared setup for integration tests.

The API's startup event loads documents + AI models (slow). For fast, isolated
tests we MOCK that step so it does nothing. This lets us test the API layer
without loading real models.
"""

import pytest

import app.services.rag_service as rag_service


@pytest.fixture(autouse=True)
def skip_heavy_ingest(monkeypatch):
    """
    Runs automatically before EACH integration test.
    Replaces ingest_folder with a no-op so startup is instant.
    """
    monkeypatch.setattr(rag_service, "ingest_folder", lambda *args, **kwargs: 0)
