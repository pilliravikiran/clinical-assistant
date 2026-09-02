"""
app/services/parent_service.py
==============================

Parent-document DOCSTORE (production pattern).

Children (small chunks) are stored in the vector DB with a `parent_id` in their
metadata. The full parent sections are stored HERE, in a persistent SQLite
key-value store keyed by parent_id. When a child is retrieved, we fetch ONLY
its parent by id (on demand) - so parents never sit in application RAM.

Why SQLite: persistent (survives restarts), no server to run, scales far beyond
an in-memory dict. In a larger deployment this would be Redis or a document DB.
"""

import os
import sqlite3
import threading

# Store the DB under data/processed, resolved from the project root so it works
# regardless of the current working directory.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DB_PATH = os.path.join(_PROJECT_ROOT, "data", "processed", "parent_store.db")

_lock = threading.Lock()
_conn = None


def _get_conn():
    """Open the SQLite connection once (creating the table if needed)."""
    global _conn
    if _conn is None:
        os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
        # check_same_thread=False: FastAPI/Streamlit use multiple threads.
        _conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
        _conn.execute("CREATE TABLE IF NOT EXISTS parents (id TEXT PRIMARY KEY, text TEXT)")
        _conn.commit()
    return _conn


def reset():
    """Delete all parents (before re-ingesting)."""
    conn = _get_conn()
    with _lock:
        conn.execute("DELETE FROM parents")
        conn.commit()


def put(parent_id, text):
    """Store one parent section, keyed by its id."""
    conn = _get_conn()
    with _lock:
        conn.execute("INSERT OR REPLACE INTO parents (id, text) VALUES (?, ?)", (parent_id, text))
        conn.commit()


def get(parent_id):
    """Fetch one parent's text by id (or None if not found)."""
    if not parent_id:
        return None
    conn = _get_conn()
    cur = conn.execute("SELECT text FROM parents WHERE id = ?", (parent_id,))
    row = cur.fetchone()
    return row[0] if row else None


def expand_to_parents(chunks):
    """
    Replace each retrieved child's text with its PARENT section (fetched from the
    docstore by parent_id). Deduplicates by parent_id, keeping best-ranked order.

    Input:  chunks -> retrieved children (each may carry "parent_id")
    Output: chunks with "text" set to the parent section, deduped.
    """
    seen = set()
    expanded = []
    for chunk in chunks:
        parent_id = chunk.get("parent_id")
        parent_text = get(parent_id)
        text = parent_text if parent_text else chunk["text"]

        key = parent_id or chunk["text"]
        if key in seen:
            continue
        seen.add(key)

        new_chunk = dict(chunk)
        new_chunk["text"] = text
        expanded.append(new_chunk)
    return expanded
