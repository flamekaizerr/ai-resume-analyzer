"""Small RAG module for asking questions over resumes and job descriptions.

Usage from the backend folder:
    python -m rag ingest rag/sample_docs
    python -m rag ask "Which projects match the backend JD?"
"""

from .loaders import Chunk, chunk_text, load_documents
from .store import FaissStore, SearchResult

__all__ = ["Chunk", "chunk_text", "load_documents", "FaissStore", "SearchResult"]
