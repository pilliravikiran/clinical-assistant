# Debugging Guide - Step Through the Whole App

Goal: understand the COMPLETE process by pausing at every important line, from
starting the app -> loading data -> answering a question.

## Setup (once)
1. Open the `tenet-clinical-ai` folder in VS Code.
2. Select the `.venv` interpreter (bottom-right, or Ctrl+Shift+P -> Python: Select Interpreter).
3. The debug config already exists (.vscode/launch.json).

## The entry point for debugging
Open `scripts/debug_ask.py`. It does BOTH flows:
  - ingest_folder("data")        -> the DATA-LOADING flow
  - answer_question("...")       -> the ANSWERING flow

Press F5 -> choose "Debug: ask a question".

## Stepping keys
- F9         : toggle a breakpoint on the current line (red dot)
- F5         : start / continue to next breakpoint
- F10        : step over (run this line, don't go inside)
- F11        : step into (go inside the function on this line)
- Shift+F11  : step out (finish this function, go back up)
- Hover / Variables panel / Watch : inspect values

Tip: press F11 to go INTO our functions; press F10 to skip over library calls
(sentence-transformers, pinecone) so you don't get lost in library code.

===================================================================
FLOW 1 - LOADING DATA (ingestion)   set a breakpoint in ingest_folder
===================================================================
| # | File -> function | What to watch |
|---|------------------|---------------|
| 1 | rag_service.py -> ingest_folder | filename (each .txt), clean (cleaned text) |
| 2 | text_utils.py -> clean_text | text before/after cleanup |
| 3 | text_utils.py -> extract_metadata | metadata dict {document_type, id, date} |
| 4 | text_utils.py -> chunk_text | parents (big), children (small) lists |
| 5 | parent_service.py -> register | child_text -> parent_text mapping grows |
| 6 | embedding_service.py -> embed_documents | to_embed (context-prefixed), vectors (384 numbers each) |
| 7 | vector_service.py -> add_documents | branches to pinecone or local; items upserted |
| 8 | keyword_service.py -> add_documents | BM25 index built from tokenized children |
Result: 6 child chunks stored in dense + BM25 indexes; parent map built.

===================================================================
FLOW 2 - ANSWERING A QUESTION (RAG)   set a breakpoint in answer_question
===================================================================
| # | File -> function | What to watch |
|---|------------------|---------------|
| 1 | debug_ask.py | question = "What follow-up care..." |
| 2 | rag_service.py -> answer_question | start timer; cache check |
| 3 | cache_service.py -> get | cache miss first time (returns None) |
| 4 | rag_service.py -> _build_context | the shared retrieval half begins |
| 5 | guardrails_service.py -> detect_prompt_injection | False for a normal question |
| 6 | rag_service.py -> retrieve_chunks | calls stage 1 then stage 2 |
| 7 | retrieval_service.py -> hybrid_search | dense + sparse + fusion |
| 8 | embedding_service.py -> embed_query | query vector (with BGE prefix) |
| 9 | vector_service.py -> search | dense candidates (by cosine) |
| 10| keyword_service.py -> search | sparse candidates (BM25 scores) |
| 11| retrieval_service.py -> reciprocal_rank_fusion | fused_scores per doc, merged list |
| 12| rerank_service.py -> rerank | pairs (question, chunk); rerank_score per chunk |
| 13| guardrails_service.py -> passes_relevance | True if best score >= threshold (-6) |
| 14| parent_service.py -> expand_to_parents | child text -> parent text (dedup) |
| 15| compression_service.py -> compress_chunks | trimmed context sentences |
| 16| llm_service.py -> generate_answer -> _answer_with_mock | the answer string |
| 17| rag_service.py -> answer_question (back) | build result; cache_service.put |
| 18| monitoring_service.py -> record_request | counters + latency updated |
| 19| debug_ask.py | result = {answer, sources, chunks} |

===================================================================
OTHER FLOWS (optional)
===================================================================
- Conversation: breakpoint in conversation_service.chat -> condense_question,
  memory_service.recall, then answer_question, then memory_service.add.
- Agent: breakpoint in agent_service.run_agent -> route_node (_pick_tool) ->
  the chosen tool node -> respond_node.
- API: run "Debug: FastAPI server", then call an endpoint (curl / /docs) and the
  breakpoints in routes.py -> the service fire.

## Suggested first session
1. Breakpoint on `rag_service.ingest_folder` line 1 -> F5 -> step with F10/F11 to
   watch data being loaded (Flow 1).
2. Breakpoint on `rag_service.answer_question` -> continue (F5) -> F11 into each
   stage (Flow 2). Inspect the variable named in the table at each stop.
