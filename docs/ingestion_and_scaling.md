# Ingestion & Handling Huge Documents / Scale

## What is ingestion?
Ingestion is the OFFLINE process of preparing documents before any question is
asked - the "remember" side of RAG. Analogy: stocking library shelves before
opening. It runs once (or on a schedule) and is separate from answering.

Pipeline: raw document -> load -> clean -> chunk -> embed -> store (index) + metadata.
Our implementation: rag_service.ingest_folder().

Retrieval (answering) is the fast ONLINE side that happens per question.

---

## Handling ONE huge document (e.g. a 500-page PDF)

| Challenge | Technique / Tool |
|---|---|
| Extract text | PyMuPDF, pdfplumber, Unstructured; OCR (Tesseract) for scans |
| Tables & layout | Layout-aware parsing (Unstructured, LlamaParse) |
| Too big for memory | Stream / lazy read, process page-by-page |
| Chunk keeps context | Parent-document (small-to-big): index small chunks, feed the LLM the parent section |
| Isolated chunk loses meaning | Contextual Retrieval: prepend an LLM-written 1-2 sentence context to each chunk before embedding |
| Whole-document tasks (summarize all) | Map-reduce or refine summarization (summarize chunks, then combine) |
| Very long single answers | Long-context models OR retrieve-then-synthesize |

### Chunking choices for large docs
- Recursive/boundary-aware (default) - respects paragraphs/sentences.
- Semantic chunking - split where meaning shifts (embedding distance spikes).
- Sentence-window - index single sentences, return neighbors as context.
- Parent-document - small child chunks for search, big parent for context.
- Token-aware sizing - respect the embedding model's max token length.

---

## Handling MILLIONS of documents (scale)

| Challenge | Technique |
|---|---|
| Keep search fast | ANN (Approximate Nearest Neighbor) indexes: HNSW, IVF, IVF-PQ. Used by FAISS, Pinecone, Milvus, Weaviate, Qdrant. Exact search (our in-memory list) is only fine for small scale. |
| Embed millions of chunks | Batch embedding; parallel/distributed jobs; GPU |
| Continuous new docs | Incremental indexing / upsert; deduplication; document versioning |
| Storage & latency | Metadata partitioning / sharding; pre-filter by metadata before vector search |
| Per-user security | Metadata filters (row-level access control); audit logs |
| Cost | Cache embeddings; semantic cache for repeated questions; smaller/quantized models |

### ANN in one line
Exact nearest-neighbor compares the query to EVERY vector (slow at scale).
ANN indexes (HNSW = a navigable small-world graph; IVF = cluster-then-search;
PQ = product quantization to compress vectors) trade a tiny bit of accuracy for
massive speed - the standard at production scale.

---

## The RAG vs long-context question (common interview trap)
"Why not just paste the whole document into a long-context model?"
- Long context is simple but: expensive per token, slower, and accuracy drops
  in the middle of very long inputs ("lost in the middle").
- RAG retrieves only the relevant pieces: cheaper, faster, and easier to cite
  and update (just re-index, no retraining). They can also be combined
  (retrieve, then use a long-context model on the retrieved set).

## Interview soundbite
"For huge documents I use parent-document and contextual retrieval so chunks
stay small but keep meaning. For scale I use ANN indexes like HNSW in a vector
DB instead of exact search, with batched embedding and incremental upserts.
RAG beats dumping everything into long context on cost, latency, freshness,
and citability."
