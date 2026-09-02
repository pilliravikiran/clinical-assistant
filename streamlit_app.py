"""
streamlit_app.py
================

The FRONTEND UI. It calls the BACKEND API (FastAPI) over HTTP - the real
architecture: UI -> API -> RAG services.

RUN BOTH (two terminals), from the project folder with the venv active:

  Terminal 1 (backend):   uvicorn app.main:app --reload
  Terminal 2 (UI):        streamlit run streamlit_app.py

The UI talks to the backend at API_URL (default http://127.0.0.1:8000).
"""

import os
import uuid

import httpx
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Tenet Clinical AI", page_icon="🏥")
st.title("🏥 Tenet Clinical AI")
st.caption(f"Frontend UI - calls the backend API at {API_URL}")

# ---- Check the backend is running ----
try:
    health = httpx.get(f"{API_URL}/health", timeout=5).json()
    st.success(f"Backend connected · mode: {health.get('mode')}")
except Exception:
    st.error(
        "Backend is not running. Start it in another terminal:\n\n"
        "    uvicorn app.main:app --reload\n\n"
        "then refresh this page."
    )
    st.stop()

# ---- Per-browser conversation state ----
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex
    st.session_state.messages = []

# ---- Sidebar: live metrics from the backend (GET /metrics) ----
with st.sidebar:
    st.header("Backend metrics")
    try:
        m = httpx.get(f"{API_URL}/metrics", timeout=5).json()
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

# ---- Show the conversation so far ----
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources"):
            st.caption("Sources: " + ", ".join(msg["sources"]))

# ---- Chat input -> call the backend /chat endpoint ----
question = st.chat_input("Ask a clinical question...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                resp = httpx.post(
                    f"{API_URL}/chat",
                    json={"session_id": st.session_state.session_id, "message": question},
                    timeout=120,
                ).json()
                answer = resp.get("answer", "(no answer)")
                sources = resp.get("sources", [])
            except Exception as e:
                answer, sources = f"Error calling backend: {e}", []
        st.write(answer)
        if sources:
            st.caption("Sources: " + ", ".join(sources))

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
