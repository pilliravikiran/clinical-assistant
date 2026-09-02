"""
app/services/conversation_service.py
====================================

Conversational RAG - adds MEMORY so follow-up questions work.

Two ideas:
  1. Per-session chat history (buffer memory, trimmed to a window).
  2. Follow-up condensing: rewrite a follow-up ("and when?") into a STANDALONE
     question using the history, BEFORE running retrieval.

The core RAG pipeline (rag_service.answer_question) is unchanged - we just
wrap it with memory + condensing.
"""

import app.config as config
from app.services import rag_service, memory_service


# session_id -> list of {"role": "user"/"assistant", "content": "..."}
_sessions = {}


def reset(session_id=None):
    """Clear one session's history (buffer + long-term memory), or all."""
    if session_id is None:
        _sessions.clear()
    else:
        _sessions.pop(session_id, None)
    memory_service.reset(session_id)


def _get_history(session_id):
    """Return (creating if needed) the message list for a session."""
    return _sessions.setdefault(session_id, [])


def _format_history(history):
    """Turn the history into text for the condensing prompt."""
    return "\n".join(f"{m['role']}: {m['content']}" for m in history)


def condense_question(history, question):
    """
    Rewrite a follow-up question into a STANDALONE question using the history.

    If there is no history, the question is already standalone.
    Online (Claude) does a real rewrite; offline uses a simple heuristic.
    """
    if not history:
        return question
    if config.APP_MODE.lower() == "online" and config.LLM_API_KEY:
        return _condense_with_claude(history, question)
    return _condense_with_mock(history, question)


def _condense_with_mock(history, question):
    """
    Offline heuristic: if the question looks like a follow-up (short, or uses a
    pronoun like 'they'/'it'), attach the previous user question as context.
    """
    pronouns = {"it", "they", "them", "he", "she", "her", "his", "that",
                "this", "those", "these", "their"}
    words = set(question.lower().replace("?", "").split())
    is_followup = len(question.split()) <= 6 or bool(words & pronouns)

    last_user = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
    if is_followup and last_user:
        return f"{question} (context: {last_user})"
    return question


def _condense_with_claude(history, question):
    """Ask Claude to rewrite the follow-up as a standalone question."""
    import anthropic

    client = anthropic.Anthropic(api_key=config.LLM_API_KEY)
    prompt = (
        "Given the conversation history and a follow-up question, rewrite the "
        "follow-up as a STANDALONE question that makes sense without the history. "
        "Return ONLY the rewritten question.\n\n"
        f"History:\n{_format_history(history)}\n\n"
        f"Follow-up: {question}\n\nStandalone question:"
    )
    try:
        resp = client.messages.create(
            model=config.LLM_MODEL, max_tokens=128,
            messages=[{"role": "user", "content": prompt}],
        )
    except anthropic.APIStatusError:
        return question
    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    return text or question


def chat(session_id, message):
    """
    Handle one chat turn WITH memory.

    Steps:
      1. load this session's history
      2. condense the follow-up into a standalone question
      3. run the RAG pipeline on the standalone question
      4. save the turn to history (trimmed to the memory window)
      5. return the answer + the standalone question used

    Called by: the /chat endpoint.
    """
    history = _get_history(session_id)

    # 2. condense follow-up -> standalone question (using the RECENT buffer)
    standalone = condense_question(history, message)

    # 2b. SEMANTIC LONG-TERM MEMORY: recall RELEVANT past turns by meaning
    # (even old ones outside the recent buffer) and add them as extra context.
    recalled = memory_service.recall(session_id, message)
    if recalled:
        standalone = f"{standalone} (relevant earlier context: {' '.join(recalled)})"

    # 3. run the normal RAG pipeline
    result = rag_service.answer_question(standalone)

    # 4a. save this turn to the recent buffer, trimmed to MEMORY_WINDOW pairs
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": result["answer"]})
    max_messages = config.MEMORY_WINDOW * 2
    if len(history) > max_messages:
        del history[: len(history) - max_messages]

    # 4b. store this message in long-term memory for future semantic recall
    memory_service.add(session_id, message)

    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "standalone_question": standalone,
        "recalled": recalled,
        "turns": len(history) // 2,
    }
