"""
app/services/llm_service.py
===========================

The "answer writer". It takes the user's question plus the document pieces we
retrieved, and produces a written answer.

Two versions behind ONE function (provider abstraction):
  - MOCK   : offline, free, no API key. Used while learning.
  - CLAUDE : the real LLM. Used when APP_MODE=online and a key is set.

The rest of the app only calls generate_answer(...) and never needs to know
which version ran.

Grounding (prompt engineering):
  We give the LLM strict rules in a "system prompt": answer ONLY from the
  provided context, and say "I don't know" if the answer is not there. This
  is what stops the model from making up medical information (hallucinating).
"""

import app.config as config


# The RULES we give the LLM. This is the heart of "grounding".
SYSTEM_PROMPT = (
    "You are a careful clinical assistant. "
    "Answer the question using ONLY the provided context. "
    "If the answer is not in the context, clearly say you do not know. "
    "Never invent or guess medical information."
)


def _anthropic_client():
    """
    Build the Anthropic client.
      - ANTHROPIC_WORKSPACE_ID -> sent as a header (for identity-linked keys).
      - LLM_CA_BUNDLE / LLM_VERIFY_SSL -> control SSL verification for networks
        that do SSL inspection (corporate/VPN). Disabling verification is
        INSECURE and for local dev only.
    """
    import anthropic

    headers = None
    if config.ANTHROPIC_WORKSPACE_ID:
        headers = {"anthropic-workspace-id": config.ANTHROPIC_WORKSPACE_ID}

    # Decide the SSL verify setting: a CA bundle path, or True/False.
    if config.LLM_CA_BUNDLE:
        verify = config.LLM_CA_BUNDLE          # trust this corporate CA
    elif not config.LLM_VERIFY_SSL:
        verify = False                          # INSECURE - dev only
    else:
        verify = True                           # normal, secure default

    http_client = None
    if verify is not True:
        import httpx
        http_client = httpx.Client(verify=verify)

    return anthropic.Anthropic(
        api_key=config.LLM_API_KEY,
        default_headers=headers,
        http_client=http_client,
    )


def build_user_message(question, context_pieces):
    """
    Glue the retrieved pieces and the question into one message for the LLM.

    Input:
        question       -> the user's question (string)
        context_pieces -> list of retrieved document pieces (strings)
    Output:
        one string containing the context followed by the question.
    """
    context_text = "\n\n".join(context_pieces)
    return f"Context:\n{context_text}\n\nQuestion: {question}"


def generate_answer(question, context_pieces):
    """
    Produce a written answer for the question, using the retrieved context.

    This is the ONE function the rest of the app calls. It picks the right
    backend (mock or Claude) based on config.

    Input:
        question       -> the user's question (string)
        context_pieces -> list of retrieved document pieces (strings)
    Output:
        a written answer (string).

    Called by: rag_service (next step).
    """
    # Use the real LLM only if we are online AND have a key. Otherwise mock.
    if config.APP_MODE.lower() == "online" and config.LLM_API_KEY:
        return _answer_with_claude(question, context_pieces)
    return _answer_with_mock(question, context_pieces)


def _answer_with_mock(question, context_pieces):
    """
    Offline stand-in for the LLM. It does NOT truly reason - it simply shows
    the retrieved context that a real LLM would turn into a written answer.
    This keeps the whole app runnable for free while learning.
    """
    if not context_pieces:
        return "I could not find information about that in the documents."

    joined = "\n".join(f"- {piece}" for piece in context_pieces)
    return (
        "(Offline mock answer - not a real LLM. It shows the retrieved context "
        "a real LLM would use to write the final answer.)\n\n"
        f"Most relevant information found:\n{joined}"
    )


def _answer_with_claude(question, context_pieces):
    """
    Real answer written by Claude.

    Uses the official Anthropic SDK. We import it INSIDE this function so the
    app does not require the 'anthropic' package in offline mode.
    """
    import anthropic

    client = _anthropic_client()
    user_message = build_user_message(question, context_pieces)

    try:
        response = client.messages.create(
            model=config.LLM_MODEL,          # which Claude model
            max_tokens=config.LLM_MAX_TOKENS,  # longest answer allowed
            system=SYSTEM_PROMPT,             # the grounding RULES
            messages=[{"role": "user", "content": user_message}],
        )
    except anthropic.RateLimitError as error:
        # Too many requests too fast.
        raise RuntimeError("AI service is busy. Please try again shortly.") from error
    except anthropic.APIConnectionError as error:
        # Could not reach the service (network problem).
        raise RuntimeError("Could not reach the AI service.") from error
    except anthropic.APIStatusError as error:
        # Any other error the API returned.
        raise RuntimeError("AI service returned an error.") from error

    # The response can contain several blocks; we keep only the text ones.
    answer = ""
    for block in response.content:
        if block.type == "text":
            answer += block.text
    return answer


# =====================================================================
# STRUCTURED OUTPUT - return {answer, confidence} instead of free text
# =====================================================================

def generate_structured(question, context_pieces):
    """
    Produce a STRUCTURED answer: {"answer": str, "confidence": float 0..1}.
    Offline: mock. Online: Claude asked to return JSON.
    """
    if config.APP_MODE.lower() == "online" and config.LLM_API_KEY:
        return _structured_with_claude(question, context_pieces)
    return _structured_with_mock(question, context_pieces)


def _structured_with_mock(question, context_pieces):
    """Offline: reuse the mock answer, attach a naive confidence."""
    if not context_pieces:
        return {"answer": "I do not know.", "confidence": 0.0}
    return {"answer": _answer_with_mock(question, context_pieces), "confidence": 0.8}


def _structured_with_claude(question, context_pieces):
    """Online: ask Claude to return JSON, then parse it."""
    import json
    import anthropic

    client = _anthropic_client()
    user_message = build_user_message(question, context_pieces) + (
        '\n\nReturn ONLY JSON: {"answer": string, "confidence": number between 0 and 1}.'
    )
    try:
        response = client.messages.create(
            model=config.LLM_MODEL, max_tokens=config.LLM_MAX_TOKENS,
            system=SYSTEM_PROMPT, messages=[{"role": "user", "content": user_message}],
        )
    except anthropic.APIStatusError as error:
        raise RuntimeError("AI service returned an error.") from error

    text = "".join(b.text for b in response.content if b.type == "text")
    try:
        data = json.loads(text)
        return {"answer": data.get("answer", ""), "confidence": float(data.get("confidence", 0.0))}
    except Exception:
        # If the model didn't return valid JSON, fall back gracefully.
        return {"answer": text, "confidence": 0.5}


# =====================================================================
# STREAMING - yield the answer token-by-token
# =====================================================================

def stream_answer(question, context_pieces):
    """
    Generator that yields the answer in pieces (for a typing/chat UX).
    Offline: yields the mock answer word-by-word. Online: streams from Claude.
    """
    if config.APP_MODE.lower() == "online" and config.LLM_API_KEY:
        yield from _stream_with_claude(question, context_pieces)
    else:
        yield from _stream_with_mock(question, context_pieces)


def _stream_with_mock(question, context_pieces):
    """Offline: yield the mock answer one word at a time."""
    answer = _answer_with_mock(question, context_pieces)
    for word in answer.split(" "):
        yield word + " "


def _stream_with_claude(question, context_pieces):
    """Online: stream text deltas from Claude as they arrive."""
    import anthropic

    client = _anthropic_client()
    user_message = build_user_message(question, context_pieces)
    with client.messages.stream(
        model=config.LLM_MODEL, max_tokens=config.LLM_MAX_TOKENS,
        system=SYSTEM_PROMPT, messages=[{"role": "user", "content": user_message}],
    ) as stream:
        for text in stream.text_stream:
            yield text
