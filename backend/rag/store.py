"""FAISS index plus the chunk metadata, saved to a local folder."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import faiss
import numpy as np

from .loaders import Chunk

INDEX_FILE = "index.faiss"
META_FILE = "chunks.json"


@dataclass
class SearchResult:
    chunk: Chunk
    score: float


def _normalize(vecs: np.ndarray) -> np.ndarray:
    vecs = np.asarray(vecs, dtype="float32")
    if vecs.ndim == 1:
        vecs = vecs.reshape(1, -1)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


class FaissStore:
    """Inner product index on normalized vectors, so scores are cosine similarity."""

    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.chunks: List[Chunk] = []

    def __len__(self):
        return len(self.chunks)

    def add(self, chunks: List[Chunk], vectors: np.ndarray):
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        if not chunks:
            return
        self.index.add(_normalize(vectors))
        self.chunks.extend(chunks)

    def search(self, query_vec: np.ndarray, k: int = 4) -> List[SearchResult]:
        if not self.chunks:
            return []
        k = min(k, len(self.chunks))
        scores, ids = self.index.search(_normalize(query_vec), k)
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:
                continue
            results.append(SearchResult(chunk=self.chunks[idx], score=float(score)))
        return results

    def save(self, folder):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(folder / INDEX_FILE))
        meta = {"dim": self.dim, "chunks": [c.to_dict() for c in self.chunks]}
        (folder / META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, folder):
        folder = Path(folder)
        if not (folder / INDEX_FILE).exists():
            raise FileNotFoundError(f"No index found in {folder}. Run ingest first.")
        meta = json.loads((folder / META_FILE).read_text(encoding="utf-8"))
        store = cls(meta["dim"])
        store.index = faiss.read_index(str(folder / INDEX_FILE))
        store.chunks = [Chunk(**c) for c in meta["chunks"]]
        return store
