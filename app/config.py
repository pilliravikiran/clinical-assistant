"""
app/config.py
=============

This file is our "control panel."
It reads settings from the .env file and gives them simple names,
so the rest of our code can use them easily.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the PROJECT ROOT (this file's parent's parent), by ABSOLUTE
# path, so it works no matter which folder you launch from - the terminal, the
# VS Code debugger, or uvicorn. (Plain load_dotenv() depends on the current
# working directory and silently loads nothing if .env isn't there.)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


# --- Run mode ---
APP_MODE = os.getenv("APP_MODE", "offline")

# --- Claude (LLM) secret key. Blank in offline mode - that's fine. ---
LLM_API_KEY = os.getenv("LLM_API_KEY", "")

# Which Claude model to use when online. (Only used if APP_MODE=online.)
LLM_MODEL = os.getenv("LLM_MODEL", "claude-opus-5")

# Longest answer the LLM may write, measured in tokens (word-pieces).
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))

# Only needed if your Anthropic key is "identity-linked" and asks for a
# workspace id. Standard API keys can leave this blank.
ANTHROPIC_WORKSPACE_ID = os.getenv("ANTHROPIC_WORKSPACE_ID", "")

# SSL for the Claude call.
# - LLM_CA_BUNDLE: path to a corporate CA .pem (for networks that do SSL
#   inspection). Preferred fix on corporate networks/VPNs.
# - LLM_VERIFY_SSL: set "false" to DISABLE verification (INSECURE - local dev
#   only, never in production). Use only to unblock testing.
LLM_CA_BUNDLE = os.getenv("LLM_CA_BUNDLE", "")
if LLM_CA_BUNDLE and not os.path.isabs(LLM_CA_BUNDLE):
    LLM_CA_BUNDLE = str(_PROJECT_ROOT / LLM_CA_BUNDLE)
LLM_VERIFY_SSL = os.getenv("LLM_VERIFY_SSL", "true").lower() == "true"

# --- Chunking settings ---
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
TOP_K = int(os.getenv("TOP_K", "5"))

# --- Embedding model ---
# BAAI/bge-small-en-v1.5 is a modern, benchmark-leading open model that is
# still small enough to run fast on CPU. Swap this name to try another model.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

# BGE models retrieve better when the QUESTION (not the documents) is prefixed
# with a short instruction. This is the model author's recommended prefix.
# We keep it in config so it is easy to change or clear for other models.
QUERY_PREFIX = os.getenv(
    "QUERY_PREFIX",
    "Represent this sentence for searching relevant passages: ",
)

# --- Cross-encoder RE-RANKER model ---
# Reads (question, chunk) together and scores true relevance. This one is a
# small, fast, classic reranker. Stronger production options:
# "BAAI/bge-reranker-v2-m3" (open) or the Cohere Rerank API.
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# --- Semantic cache ---
# Return a cached answer when a new question is very similar to a past one.
USE_SEMANTIC_CACHE = os.getenv("USE_SEMANTIC_CACHE", "true").lower() == "true"
CACHE_SIM_THRESHOLD = float(os.getenv("CACHE_SIM_THRESHOLD", "0.95"))  # high = only near-identical
CACHE_MAX_SIZE = int(os.getenv("CACHE_MAX_SIZE", "1000"))              # cap entries (avoid unbounded growth)

# --- Prompt-injection guardrail ---
USE_PROMPT_INJECTION_GUARD = os.getenv("USE_PROMPT_INJECTION_GUARD", "true").lower() == "true"

# --- Conversation memory ---
# How many recent turns (user+assistant pairs) to keep per chat session (buffer).
MEMORY_WINDOW = int(os.getenv("MEMORY_WINDOW", "5"))

# Semantic long-term memory: recall the most RELEVANT past turns by meaning
# (works even for old turns outside the recent buffer).
MEMORY_RECALL_K = int(os.getenv("MEMORY_RECALL_K", "2"))       # how many to recall
MEMORY_RECALL_MIN_SIM = float(os.getenv("MEMORY_RECALL_MIN_SIM", "0.35"))  # relevance floor

# --- Parent-document retrieval ---
# Index small "child" chunks for precise matching, but feed the LLM the larger
# "parent" section they belong to (more context). Set False for plain chunking.
USE_PARENT_DOCUMENT = os.getenv("USE_PARENT_DOCUMENT", "true").lower() == "true"
PARENT_CHUNK_SIZE = int(os.getenv("PARENT_CHUNK_SIZE", "800"))   # big (context)
CHILD_CHUNK_SIZE = int(os.getenv("CHILD_CHUNK_SIZE", "200"))     # small (search)

# --- Contextual Compression (post-retrieval) ---
# When True, each retrieved chunk is trimmed to only the sentences relevant to
# the question (saves tokens, reduces noise) before going to the LLM.
USE_COMPRESSION = os.getenv("USE_COMPRESSION", "true").lower() == "true"
# Keep a sentence if its similarity is within this margin of the BEST sentence.
# Relative (not absolute) because baseline similarity varies by text.
COMPRESSION_MARGIN = float(os.getenv("COMPRESSION_MARGIN", "0.2"))

# --- Contextual Retrieval ---
# When True, each chunk is enriched with its document context (type, source)
# BEFORE embedding, so isolated chunks keep their meaning and match more
# questions. We still STORE the original chunk for the answer/citation.
USE_CONTEXTUAL_RETRIEVAL = os.getenv("USE_CONTEXTUAL_RETRIEVAL", "true").lower() == "true"

# --- Vector database backend ---
# "local"    = the free in-memory store (default, no key)
# "pinecone" = the managed cloud vector DB (persists, needs a key below)
VECTOR_BACKEND = os.getenv("VECTOR_BACKEND", "pinecone")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "tenet-clinical")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

# --- Guardrail: relevance threshold ---
# If the best re-ranked chunk scores BELOW this, we refuse ("I don't know")
# instead of guessing. The scale depends on the reranker; for the ms-marco
# model, positive scores mean relevant, so 0.0 is a sensible cutoff.
# Calibrated on observed rerank scores: valid questions score >= ~0 while
# clearly off-topic questions score around -11, so -6 cleanly separates them.
# NOTE: re-calibrate this if you change the reranker model or corpus.
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "-6.0"))
