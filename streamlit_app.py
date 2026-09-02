"""
streamlit_app.py
================

The UI. It works TWO ways automatically:
  - If the FastAPI backend is running (local 2-terminal setup), it calls the API.
  - If not (e.g. a single Hugging Face Space), it runs the RAG in-process.

Local:            uvicorn app.main:app --reload    +    streamlit run streamlit_app.py
Hugging Face:     just this file runs (in-process). Set secrets: APP_MODE,
                  LLM_API_KEY, PINECONE_API_KEY, VECTOR_BACKEND.
"""

import os
import uuid

import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

# --- Decide: call the API, or run in-process? ---
USE_API = False
try:
    import httpx
    httpx.get(f"{API_URL}/health", timeout=3)
    USE_API = True
except Exception:
    USE_API = False

# In-process mode: import the services and load documents once.
if not USE_API:
    from app.services import rag_service, conversation_service, monitoring_service

    @st.cache_resource(show_spinner="Loading models and documents (first run takes ~30s)...")
    def _load():
        return rag_service.ingest_folder("data")

    _load()

st.set_page_config(page_title="Clinical Assistant", page_icon="🏥")
st.title("🏥 Clinical Assistant")
st.caption("Grounded clinical Q&A (RAG). " + ("Using backend API." if USE_API else "Running in-process."))

# --- Per-browser conversation state ---
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex
    st.session_state.messages = []

# --- Sidebar: metrics ---
with st.sidebar:
    st.header("Metrics")
    try:
        if USE_API:
            m = httpx.get(f"{API_URL}/metrics", timeout=5).json()
        else:
            m = monitoring_service.get_metrics()
        st.metric("Requests", m["requests_total"])
        st.metric("Avg latency (ms)", m["avg_latency_ms"])
        st.metric("Cache hit rate", m["cache_hit_rate"])
        st.metric("Refusal rate", m["refusal_rate"])
    except Exception:
        st.caption("(metrics unavailable)")
    if st.button("New conversation"):
        st.session_state.session_id = uuid.uuid4().hex
        st.session_state.messages = []
        st.rerun()
    st.caption("Try: *What follow-up care was recommended?* or *What were the lab results?*")

# --- Show the conversation ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources"):
            st.caption("Sources: " + ", ".join(msg["sources"]))

# --- Chat input ---
question = st.chat_input("Ask a clinical question...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                if USE_API:
                    resp = httpx.post(
                        f"{API_URL}/chat",
                        json={"session_id": st.session_state.session_id, "message": question},
                        timeout=120,
                    ).json()
                    answer, sources = resp.get("answer", ""), resp.get("sources", [])
                else:
                    result = conversation_service.chat(st.session_state.session_id, question)
                    answer, sources = result["answer"], result["sources"]
            except Exception as e:
                answer, sources = f"Error: {e}", []
        st.write(answer)
        if sources:
            st.caption("Sources: " + ", ".join(sources))

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
