"""Embedding providers used by transcript indexing and semantic search."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np
from openai import OpenAI

from .config import settings


class EmbeddingDimensionError(ValueError):
    """Raised when the configured embedding model returns the wrong shape."""


class EmbeddingConfigurationError(ValueError):
    """Raised when an embedding provider cannot be configured."""


class EmbeddingModel(Protocol):
    """Common interface for document and query embedding providers."""

    provider: str
    model_name: str
    dimension: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class OpenRouterEmbedder:
    """Create normalized embeddings through OpenRouter's embeddings API."""

    provider = "openrouter"

    def __init__(
        self,
        model_name: str | None = None,
        dimension: int | None = None,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        client: Any | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.model_name = model_name or settings.embedding_model
        self.dimension = dimension or settings.embedding_dimension
        key = api_key if api_key is not None else settings.openrouter_api_key
        if client is not None:
            self._client = client
        else:
            if not key:
                raise EmbeddingConfigurationError(
                    "OPENROUTER_API_KEY is required for OpenRouter embeddings"
                )
            self._client = OpenAI(
                api_key=key,
                base_url=base_url or settings.openrouter_base_url,
                timeout=timeout,
            )

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed an ordered batch, normalize it, and enforce the index dimension."""

        if not texts:
            return []

        response = self._client.embeddings.create(
            model=self.model_name,
            input=list(texts),
            encoding_format="float",
        )
        data = sorted(response.data, key=lambda item: item.index)
        vectors = np.asarray([item.embedding for item in data], dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape != (len(texts), self.dimension):
            actual_shape = tuple(vectors.shape)
            raise EmbeddingDimensionError(
                f"OpenRouter model '{self.model_name}' returned shape "
                f"{actual_shape}; expected ({len(texts)}, {self.dimension})"
            )

        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("OpenRouter returned a zero-length embedding vector")
        return (vectors / norms).astype(float).tolist()


class SentenceTransformerEmbedder:
    """Generate normalized BAAI/bge-m3 embeddings with a fixed dimension.

    The model is loaded lazily so parsing, JSONL tests, and index restarts do
    not load a multi-gigabyte model unless embeddings are actually requested.
    A model-like object can be injected in tests; it must expose ``encode``.
    """

    provider = "local"

    def __init__(
        self,
        model_name: str | None = None,
        dimension: int | None = None,
        model: Any | None = None,
    ) -> None:
        self.model_name = model_name or "BAAI/bge-m3"
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


def create_embedder(
    *,
    provider: str | None = None,
    model_name: str | None = None,
    dimension: int | None = None,
) -> EmbeddingModel:
    """Build the configured embedding provider or one named by an index."""

    selected_provider = provider or settings.embedding_provider
    if selected_provider == "openrouter":
        return OpenRouterEmbedder(model_name=model_name, dimension=dimension)
    if selected_provider == "local":
        return SentenceTransformerEmbedder(model_name=model_name, dimension=dimension)
    raise EmbeddingConfigurationError(
        f"Unsupported embedding provider '{selected_provider}'"
    )
