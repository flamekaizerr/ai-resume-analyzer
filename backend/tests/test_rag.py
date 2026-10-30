"""Tests for the rag module. They use a fake embedder so nothing gets downloaded."""

import hashlib
import re
import sys
from pathlib import Path

import numpy as np
import pytest

backend_dir = str(Path(__file__).resolve().parent.parent)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from rag import qa
from rag.loaders import Chunk, chunk_text, load_documents
from rag.store import FaissStore

SAMPLE_DIR = Path(backend_dir) / "rag" / "sample_docs"


class FakeEmbedder:
    """Bag of words hashed into a small vector. Same text always gives the same vector."""

    def __init__(self, dim=64):
        self.dim = dim

    def embed(self, texts):
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for i, t in enumerate(texts):
            for word in re.findall(r"[a-z0-9]+", t.lower()):
                h = int(hashlib.md5(word.encode()).hexdigest(), 16)
                out[i, h % self.dim] += 1.0
        return out


@pytest.fixture
def docs(tmp_path):
    (tmp_path / "python.txt").write_text("Python FastAPI backend with PostgreSQL and Docker.")
    (tmp_path / "cooking.md").write_text("Recipe for tomato soup with basil and garlic.")
    (tmp_path / "music.txt").write_text("Guitar chords and piano scales for beginners.")
    (tmp_path / "ignore.csv").write_text("this,should,be,skipped")
    return tmp_path


def test_chunk_text_overlap():
    text = " ".join(f"w{i}" for i in range(300))
    chunks = chunk_text(text, chunk_size=100, overlap=30)
    assert len(chunks) > 1
    assert all(len(c) <= 100 for c in chunks)
    # end of one chunk should show up again at the start of the next
    for a, b in zip(chunks, chunks[1:]):
        assert a.split()[-1] in b.split()
    # nothing lost
    words = set()
    for c in chunks:
        words.update(c.split())
    assert words == set(text.split())


def test_chunk_text_short_and_empty():
    assert chunk_text("") == []
    assert chunk_text("   \n ") == []
    assert chunk_text("hello world", chunk_size=100, overlap=10) == ["hello world"]
    with pytest.raises(ValueError):
        chunk_text("abc", chunk_size=10, overlap=10)


def test_load_documents_skips_unsupported(docs):
    chunks = load_documents([str(docs)])
    assert sorted({c.source for c in chunks}) == ["cooking.md", "music.txt", "python.txt"]


def test_load_pdf(tmp_path):
    pypdf = pytest.importorskip("pypdf")
    # blank page pdf has no text, should give no chunks and not crash
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=200, height=200)
    path = tmp_path / "blank.pdf"
    with open(path, "wb") as f:
        writer.write(f)
    assert load_documents([str(path)]) == []


def test_store_save_load(tmp_path):
    emb = FakeEmbedder()
    chunks = [Chunk("python backend", "a.txt", 0), Chunk("tomato soup", "b.txt", 0)]
    store = FaissStore(emb.dim)
    store.add(chunks, emb.embed([c.text for c in chunks]))
    store.save(tmp_path / "idx")

    loaded = FaissStore.load(tmp_path / "idx")
    assert len(loaded) == 2
    assert loaded.chunks == chunks
    q = emb.embed(["python"])
    assert [r.chunk for r in loaded.search(q, k=2)] == [r.chunk for r in store.search(q, k=2)]


def test_load_missing_index(tmp_path):
    with pytest.raises(FileNotFoundError):
        FaissStore.load(tmp_path / "nope")


def test_retrieval_ranking(docs, tmp_path):
    emb = FakeEmbedder()
    store = qa.ingest([str(docs)], emb, folder=tmp_path / "idx")
    results = qa.retrieve(store, "python backend docker", emb, k=3)
    assert results[0].chunk.source == "python.txt"
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    assert len(qa.retrieve(store, "soup", emb, k=10)) == 3


def test_fallback_answer_without_key(docs, tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    emb = FakeEmbedder()
    store = qa.ingest([str(docs)], emb, folder=tmp_path / "idx")
    result = qa.answer("tomato soup recipe", store, emb, k=2)
    assert result["mode"] == "extractive"
    assert result["sources"][0]["source"] == "cooking.md"
    assert "[cooking.md #0]" in result["answer"]


def test_model_error_falls_back(docs, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    def boom(*args, **kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr(qa, "model_answer", boom)
    emb = FakeEmbedder()
    store = qa.ingest([str(docs)], emb, folder=tmp_path / "idx")
    result = qa.answer("guitar", store, emb)
    assert result["mode"] == "extractive"
    assert "music.txt" in result["answer"]


def test_model_answer_used_when_key_set(docs, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(qa, "model_answer", lambda q, r: "You know Python [python.txt #0]")
    emb = FakeEmbedder()
    store = qa.ingest([str(docs)], emb, folder=tmp_path / "idx")
    result = qa.answer("python", store, emb)
    assert result["mode"] == "model"
    assert result["answer"] == "You know Python [python.txt #0]"


def test_cli_ingest_and_ask(docs, tmp_path, monkeypatch, capsys):
    import rag.embeddings
    from rag.__main__ import main

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(rag.embeddings, "get_embedder", lambda: FakeEmbedder())
    idx = str(tmp_path / "idx")
    assert main(["--index-dir", idx, "ingest", str(docs)]) == 0
    assert main(["--index-dir", idx, "ask", "piano scales"]) == 0
    out = capsys.readouterr().out
    assert "Indexed 3 chunks" in out
    assert "[music.txt #0]" in out


def _model_available():
    try:
        from huggingface_hub import snapshot_download

        # only checks the local cache, no download
        snapshot_download("sentence-transformers/all-MiniLM-L6-v2", local_files_only=True)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _model_available(), reason="all-MiniLM-L6-v2 not downloaded")
def test_real_model_on_sample_docs(tmp_path, monkeypatch):
    from rag.embeddings import SentenceTransformerEmbedder

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    emb = SentenceTransformerEmbedder("all-MiniLM-L6-v2")
    store = qa.ingest([str(SAMPLE_DIR)], emb, folder=tmp_path / "idx")
    results = qa.retrieve(store, "dashboards in Power BI or Tableau", emb, k=1)
    assert results[0].chunk.source == "jd_data_analyst.txt"


def test_ask_route(docs, tmp_path, monkeypatch):
    # call the route function directly, TestClient breaks with the pinned starlette
    from fastapi import HTTPException

    import rag.embeddings
    from rag.router import AskRequest, ask

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("RAG_INDEX_DIR", str(tmp_path / "idx"))
    monkeypatch.setattr(rag.embeddings, "get_embedder", lambda: FakeEmbedder())

    # no index yet
    with pytest.raises(HTTPException) as err:
        ask(AskRequest(question="python"))
    assert err.value.status_code == 404

    qa.ingest([str(docs)], FakeEmbedder(), folder=tmp_path / "idx")
    body = ask(AskRequest(question="python docker", k=2))
    assert body["mode"] == "extractive"
    assert body["sources"][0]["source"] == "python.txt"

    with pytest.raises(HTTPException) as err:
        ask(AskRequest(question="  "))
    assert err.value.status_code == 400


def test_model_answer_uses_openrouter(monkeypatch):
    import openai

    seen = {}

    class FakeClient:
        def __init__(self, **kwargs):
            seen.update(kwargs)
            self.chat = self
            self.completions = self

        def create(self, **kwargs):
            seen["model"] = kwargs["model"]
            msg = type("M", (), {"content": " ok [a.txt #0] "})
            choice = type("C", (), {"message": msg})
            return type("R", (), {"choices": [choice]})

    monkeypatch.setattr(openai, "OpenAI", FakeClient)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.delenv("RAG_BASE_URL", raising=False)
    monkeypatch.delenv("RAG_MODEL", raising=False)

    from rag.store import SearchResult

    out = qa.model_answer("q", [SearchResult(Chunk("text", "a.txt", 0), 0.9)])
    assert out == "ok [a.txt #0]"
    assert seen["base_url"] == "https://openrouter.ai/api/v1"
    assert seen["api_key"] == "test-key"
    assert seen["model"] == "openai/gpt-4o-mini"
