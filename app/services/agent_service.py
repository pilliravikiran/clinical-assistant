"""
app/services/agent_service.py
=============================

A small AGENT built with LangGraph.

Unlike plain RAG (which always does the same thing), the agent DECIDES which
TOOL to use for each question:
  - search_documents   -> our RAG pipeline (question answering)
  - summarize_document -> summarize the most relevant document
  - list_documents     -> list available documents and their types

LangGraph terminology shown here:
  - State           : a dict passed between steps (question, tool, result, ...)
  - Node            : one step (route, a tool, respond)
  - Edge            : a connection between nodes
  - Conditional edge: branch to a different node based on the state (the routing)
  - Graph execution : run the nodes following the edges

Offline, the router uses simple keyword rules. Online (Claude), it would use
LLM tool-calling to choose - the graph structure is identical either way.
"""

import os
from typing import TypedDict

from langgraph.graph import StateGraph, END

import app.config as config
from app.services import rag_service, parent_service
from app.utils.text_utils import clean_text, extract_metadata


# ---- The State that flows through the graph ----
class AgentState(TypedDict, total=False):
    question: str
    tool: str
    tool_result: str
    sources: list
    answer: str


def _pick_tool(question):
    """Choose a tool for the question (keyword rules for the offline demo)."""
    q = question.lower()
    if "summar" in q:
        return "summarize"
    if "list" in q or "what documents" in q or "which documents" in q or "available" in q:
        return "list"
    return "search"


# ---- Nodes ----
def route_node(state):
    """Node: decide which tool to use."""
    return {"tool": _pick_tool(state["question"])}


def search_node(state):
    """Tool: answer the question with the RAG pipeline."""
    result = rag_service.answer_question(state["question"])
    return {"tool_result": result["answer"], "sources": result["sources"]}


def summarize_node(state):
    """Tool: summarize the most relevant document (naive offline summary)."""
    chunks = rag_service.retrieve_chunks(state["question"], top_k=1)
    if not chunks:
        return {"tool_result": "No relevant document found to summarize.", "sources": []}
    text = parent_service.get_parent(chunks[0]["text"])
    # naive summary: first two sentences
    sentences = text.replace("\n", " ").split(". ")
    summary = ". ".join(sentences[:2]).strip()
    return {"tool_result": "Summary: " + summary, "sources": [chunks[0]["source"]]}


def list_node(state):
    """Tool: list available documents and their types."""
    items = []
    for filename in sorted(os.listdir("data")):
        if filename.endswith(".txt"):
            with open(os.path.join("data", filename), encoding="utf-8") as f:
                meta = extract_metadata(clean_text(f.read()))
            items.append(f"{filename} ({meta['document_type']})")
    return {"tool_result": "Available documents: " + "; ".join(items), "sources": []}


def respond_node(state):
    """Node: produce the final answer from the tool result."""
    return {"answer": state.get("tool_result", "")}


# ---- Build the graph once ----
def _build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("route", route_node)
    graph.add_node("search", search_node)
    graph.add_node("summarize", summarize_node)
    graph.add_node("list", list_node)
    graph.add_node("respond", respond_node)

    graph.set_entry_point("route")
    # CONDITIONAL EDGE: go to the tool named in state["tool"].
    graph.add_conditional_edges("route", lambda s: s["tool"],
                                {"search": "search", "summarize": "summarize", "list": "list"})
    # Each tool then goes to respond, and respond ends.
    graph.add_edge("search", "respond")
    graph.add_edge("summarize", "respond")
    graph.add_edge("list", "respond")
    graph.add_edge("respond", END)
    return graph.compile()


_app = _build_graph()


def run_agent(question):
    """
    Run the agent on a question.

    Output: {"answer", "tool_used", "sources"}
    Called by: the /agent endpoint.
    """
    result = _app.invoke({"question": question})
    return {
        "answer": result.get("answer", ""),
        "tool_used": result.get("tool", ""),
        "sources": result.get("sources", []),
    }
