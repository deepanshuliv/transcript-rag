"""Local sentence-transformers embedding wrapper for Phase 2."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .config import settings


class EmbeddingDimensionError(ValueError):
    """Raised when the configured embedding model returns the wrong shape."""


class SentenceTransformerEmbedder:
    """Generate normalized BAAI/bge-m3 embeddings with a fixed dimension.

    The model is loaded lazily so parsing, JSONL tests, and index restarts do
    not load a multi-gigabyte model unless embeddings are actually requested.
    A model-like object can be injected in tests; it must expose ``encode``.
    """

    def __init__(
        self,
        model_name: str | None = None,
        dimension: int | None = None,
        model: Any | None = None,
    ) -> None:
        self.model_name = model_name or settings.embedding_model
        self.dimension = dimension or settings.embedding_dimension
        self._model = model

    @property
    def model(self) -> Any:
        """Load and cache the configured sentence-transformers model."""

        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Encode text into normalized vectors and verify the configured size."""

        if not texts:
            return []

        values = self.model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        vectors = np.asarray(values, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if vectors.ndim != 2 or vectors.shape != (len(texts), self.dimension):
            actual_shape = tuple(vectors.shape)
            raise EmbeddingDimensionError(
                f"Embedding model '{self.model_name}' returned shape "
                f"{actual_shape}; expected ({len(texts)}, {self.dimension})"
            )
        return vectors.astype(float).tolist()


# Short alias for callers that prefer the plan's general terminology.
EmbeddingModel = SentenceTransformerEmbedder
