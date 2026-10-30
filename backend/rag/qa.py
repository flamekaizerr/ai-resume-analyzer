"""Ingest and ask. Answers come from a GPT model on OpenRouter if OPENROUTER_API_KEY
is set, otherwise we just return the best matching passages."""

import logging
import os
from pathlib import Path
from typing import Iterable, List, Optional

from .loaders import Chunk, chunk_documents, load_documents
from .store import FaissStore, SearchResult

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "openai/gpt-4o-mini"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_INDEX_DIR = Path(__file__).resolve().parent.parent / "rag_index"
SAMPLE_DIR = Path(__file__).resolve().parent / "sample_docs"


def index_dir() -> Path:
    return Path(os.environ.get("RAG_INDEX_DIR", DEFAULT_INDEX_DIR))


def ingest(paths: Iterable[str], embedder, folder=None, chunk_size=500, overlap=100) -> FaissStore:
    chunks = load_documents(paths, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        raise ValueError("No text found in the given paths (supported: .txt, .md, .pdf, .docx)")
    store = build_store(chunks, embedder)
    store.save(folder or index_dir())
    return store


def build_store(chunks: List[Chunk], embedder) -> FaissStore:
    """In-memory index, nothing is saved."""
    if not chunks:
        raise ValueError("No text to index")
    vectors = embedder.embed([c.text for c in chunks])
    store = FaissStore(vectors.shape[1])
    store.add(chunks, vectors)
    return store


def store_from_texts(docs, embedder, chunk_size=500, overlap=100) -> FaissStore:
    return build_store(chunk_documents(docs, chunk_size, overlap), embedder)


def retrieve(store: FaissStore, question: str, embedder, k: int = 4) -> List[SearchResult]:
    return store.search(embedder.embed([question]), k=k)


def cite(r: SearchResult) -> str:
    return f"[{r.chunk.source} #{r.chunk.chunk_id}]"


def extractive_answer(question: str, results: List[SearchResult]) -> str:
    if not results:
        return "Nothing relevant found in the index."
    lines = ["No OPENROUTER_API_KEY set, so here are the closest passages:", ""]
    for r in results:
        lines.append(f"{cite(r)} (score {r.score:.3f})")
        lines.append(r.chunk.text)
        lines.append("")
    return "\n".join(lines).strip()


def build_prompt(question: str, results: List[SearchResult]) -> str:
    context = "\n\n".join(f"{cite(r)}\n{r.chunk.text}" for r in results)
    return (
        "Answer the question using only the context below. The context comes from a resume "
        "and job descriptions. After each claim, cite the source in the same "
        "[file #chunk] form. If the context does not have the answer, say so.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )


def model_answer(question: str, results: List[SearchResult], model: Optional[str] = None) -> str:
    # OpenRouter speaks the OpenAI API, so the openai client works with a different base_url
    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url=os.environ.get("RAG_BASE_URL", DEFAULT_BASE_URL),
        timeout=60,
    )
    resp = client.chat.completions.create(
        model=model or os.environ.get("RAG_MODEL", DEFAULT_MODEL),
        messages=[
            {"role": "system", "content": "You help a job seeker compare their resume with job descriptions."},
            {"role": "user", "content": build_prompt(question, results)},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


def answer(question: str, store: FaissStore, embedder, k: int = 4) -> dict:
    results = retrieve(store, question, embedder, k=k)
    mode = "extractive"
    text = None
    if os.environ.get("OPENROUTER_API_KEY") and results:
        try:
            text = model_answer(question, results)
            mode = "model"
        except Exception as e:
            # fall back instead of failing the whole request
            logger.warning("Model call failed, using extractive answer: %s", type(e).__name__)
    if text is None:
        text = extractive_answer(question, results)
    return {
        "answer": text,
        "mode": mode,
        "sources": [
            {"source": r.chunk.source, "chunk_id": r.chunk.chunk_id, "score": round(r.score, 4), "text": r.chunk.text}
            for r in results
        ],
    }
