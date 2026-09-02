"""
app/utils/text_utils.py
=======================

Two small tools for preparing a document:
  1. clean_text  -> tidy up messy spacing
  2. chunk_text  -> cut a document into smart, boundary-aware pieces ("chunks")

For chunking we use LangChain's RecursiveCharacterTextSplitter, which is the
approach real RAG systems use in production. Instead of blindly cutting every
N characters, it tries to split on natural boundaries (paragraphs, then
sentences, then words) so each piece holds a complete thought.
"""

import re

# Industry-standard splitter used across production RAG systems.
from langchain_text_splitters import RecursiveCharacterTextSplitter

import app.config as config


def clean_text(text):
    """
    Tidy up a document's spacing, without changing the words.

    Input:  text -> the raw document (a string)
    Output: a cleaned string (same words, neater spacing)
    """
    text = re.sub(r"\n{3,}", "\n\n", text)   # squash big gaps to one blank line
    text = re.sub(r"[ \t]+\n", "\n", text)   # drop trailing spaces on each line
    text = text.strip()                       # trim the whole document
    return text


def chunk_text(text, chunk_size=None):
    """
    Cut a document into smart, overlapping pieces ("chunks").

    Input:
        text       -> the cleaned document (a string)
        chunk_size -> target characters per piece
                      (if not given, use the value from config.py)

    Output:
        a list of strings (the chunks)

    How it works (the "advanced" part):
        RecursiveCharacterTextSplitter is given a PRIORITY LIST of places it
        is allowed to split:
            1. "\n\n"  -> paragraph breaks   (best place to split)
            2. "\n"    -> line breaks
            3. ". "    -> sentence ends
            4. " "     -> word gaps
            5. ""      -> as a last resort, any character
        It always prefers the highest-priority boundary that keeps a chunk
        under the target size. This keeps whole thoughts together instead of
        cutting sentences in half.
    """
    if chunk_size is None:
        chunk_size = config.CHUNK_SIZE

    # Build the splitter once, with our size + overlap settings from config.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=config.CHUNK_OVERLAP,
        # The priority list of boundaries to split on (best first).
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    # Do the split. Returns a clean list of chunk strings.
    return splitter.split_text(text)


def extract_metadata(text):
    """
    Read the small header at the top of a document and return its metadata.

    Input:  text -> the cleaned document text, whose top lines look like:
                "Document Type: Physician Note"
                "Document ID: PN-001"
                "Date: 2025-03-14"
    Output: a dict {"document_type","document_id","date"} (missing -> "unknown").

    Why: metadata lets us enrich chunks (contextual retrieval), filter searches,
    and cite sources more precisely.
    """
    metadata = {"document_type": "unknown", "document_id": "unknown", "date": "unknown"}

    for line in text.splitlines()[:10]:      # only scan the header area
        if ":" not in line:
            continue
        key, value = line.split(":", 1)      # split on the FIRST colon only
        key = key.strip().lower()
        value = value.strip()
        if key == "document type":
            metadata["document_type"] = value
        elif key == "document id":
            metadata["document_id"] = value
        elif key == "date":
            metadata["date"] = value

    return metadata
