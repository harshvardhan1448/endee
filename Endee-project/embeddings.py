"""
Embedding module — wraps a HuggingFace sentence-transformers model.

Uses `all-MiniLM-L6-v2` by default (384-dim, ~80 MB, very fast on CPU).
"""

from __future__ import annotations

from typing import List

from sentence_transformers import SentenceTransformer

import config


class Embedder:
    """Thin wrapper around a sentence-transformers model."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or config.EMBEDDING_MODEL
        self._model: SentenceTransformer | None = None

    # Lazy-load so import is cheap
    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    # ── public API ───────────────────────────────────────────────────────
    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string and return a list of floats."""
        vec = self.model.encode(text, normalize_embeddings=True)
        return vec.tolist()

    def embed_texts(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        """Embed a batch of texts and return a list of float lists."""
        vecs = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        return vecs.tolist()


# Module-level singleton for convenience
_default_embedder: Embedder | None = None


def get_embedder() -> Embedder:
    global _default_embedder
    if _default_embedder is None:
        _default_embedder = Embedder()
    return _default_embedder
