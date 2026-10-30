"""Embedding models. Uses the same sentence-transformers model as the rest of the backend."""

import os
from typing import List

import numpy as np

DEFAULT_EMBED_MODEL = "all-MiniLM-L6-v2"


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = None):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name or os.environ.get("RAG_EMBED_MODEL", DEFAULT_EMBED_MODEL)
        self.model = SentenceTransformer(self.model_name)

    def embed(self, texts: List[str]) -> np.ndarray:
        vecs = self.model.encode(list(texts), convert_to_numpy=True, normalize_embeddings=True)
        return np.asarray(vecs, dtype="float32")


_embedder = None


def get_embedder():
    # load once, the model takes a few seconds
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformerEmbedder()
    return _embedder
