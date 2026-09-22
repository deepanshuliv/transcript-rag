"""Persistent BM25 index for Phase 2."""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from .schemas import TranscriptChunk


TOKEN_RE = re.compile(r"[\w]+", flags=re.UNICODE)
BM25_FORMAT_VERSION = 1


def tokenize(text: str) -> list[str]:
    """Tokenize consistently for both BM25 construction and later queries."""

    return TOKEN_RE.findall(text.casefold())


class BM25Index:
    """A rebuildable BM25 index whose document order matches chunk IDs."""

    def __init__(self, chunk_ids: list[str], documents: list[str]) -> None:
        if len(chunk_ids) != len(documents):
            raise ValueError("BM25 chunk IDs and documents must have equal length")
        if not chunk_ids:
            raise ValueError("Cannot create a BM25 index without chunks")
        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError("BM25 chunk IDs must be unique")
        self.chunk_ids = list(chunk_ids)
        self.documents = list(documents)
        self.tokenized_documents = [tokenize(document) for document in documents]
        self._bm25 = BM25Okapi(self.tokenized_documents)

    @classmethod
    def from_chunks(cls, chunks: list[TranscriptChunk]) -> "BM25Index":
        """Build the lexical index from the exact retrieval text persisted in JSONL."""

        return cls(
            chunk_ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.retrieval_text for chunk in chunks],
        )

    def save(self, path: str | Path) -> None:
        """Persist a versioned, local-only BM25 artifact."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": BM25_FORMAT_VERSION,
            "chunk_ids": self.chunk_ids,
            "documents": self.documents,
            "tokenized_documents": self.tokenized_documents,
        }
        with output_path.open("wb") as file_handle:
            pickle.dump(payload, file_handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "BM25Index":
        """Load and validate a persisted BM25 artifact."""

        with Path(path).open("rb") as file_handle:
            payload: dict[str, Any] = pickle.load(file_handle)
        if payload.get("format_version") != BM25_FORMAT_VERSION:
            raise ValueError("Unsupported BM25 index format version")
        index = cls(
            chunk_ids=list(payload["chunk_ids"]),
            documents=list(payload["documents"]),
        )
        if index.tokenized_documents != payload.get("tokenized_documents"):
            raise ValueError("Persisted BM25 tokenization does not match the documents")
        return index

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Return stable ``(chunk_id, score)`` pairs for a lexical query."""

        if top_k <= 0:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        ranked_indices = sorted(
            range(len(self.chunk_ids)),
            key=lambda index: (-float(scores[index]), index),
        )
        return [
            (self.chunk_ids[index], float(scores[index]))
            for index in ranked_indices[:top_k]
        ]
