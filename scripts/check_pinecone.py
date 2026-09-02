"""
scripts/check_pinecone.py
=========================

Verify the Pinecone connection end-to-end. Run this AFTER you set your key.

Setup (in your .env file):
    VECTOR_BACKEND=pinecone
    PINECONE_API_KEY=your-real-key

Run:
    python scripts/check_pinecone.py
"""

import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app.config as config
from app.services import rag_service, vector_service

if config.VECTOR_BACKEND.lower() != "pinecone":
    print("VECTOR_BACKEND is not 'pinecone'. Set it in your .env file first.")
    sys.exit(1)

if not config.PINECONE_API_KEY:
    print("PINECONE_API_KEY is empty. Add it to your .env file.")
    sys.exit(1)

print("Ingesting documents into Pinecone...")
n = rag_service.ingest_folder("data", chunk_size=200)
time.sleep(3)   # give Pinecone a moment to index the new vectors
print(f"Ingested {n} chunks. Pinecone now holds {vector_service.count()} vectors.")

print("\nAsking a question (served from Pinecone)...")
result = rag_service.answer_question("What follow-up care was recommended?", top_k=3)
print("SOURCES:", result["sources"])
print("ANSWER :", result["answer"].splitlines()[0])
print("\nPinecone is connected and working.")
