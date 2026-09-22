"""Persistent Chroma vector-store wrapper for Phase 2."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import chromadb

from .config import DEFAULT_COLLECTION_NAME
from .schemas import EvidenceItem, TranscriptChunk

if TYPE_CHECKING:
    from .ingest import LocalIndex


def metadata_for_chunk(chunk: TranscriptChunk) -> dict[str, Any]:
    """Return all citation-relevant chunk fields in Chroma-safe metadata."""

    metadata: dict[str, Any] = {
        "evidence_id": chunk.chunk_id,
        "source_file": chunk.source_file,
        "source_hash": chunk.source_hash,
        "country": chunk.country,
        "expert": chunk.expert,
        "speaker": chunk.speaker,
        "question_text": chunk.question_text,
        "answer_text": chunk.answer_text,
        "question_timestamp": chunk.question_timestamp,
        "answer_timestamp": chunk.answer_timestamp,
    }
    if chunk.char_start is not None:
        metadata["char_start"] = chunk.char_start
    if chunk.char_end is not None:
        metadata["char_end"] = chunk.char_end
    if chunk.parent_chunk_id is not None:
        metadata["parent_chunk_id"] = chunk.parent_chunk_id
    return metadata


class ChromaVectorStore:
    """Own a persistent Chroma collection for transcript chunks."""

    def __init__(
        self,
        persist_directory: str | Path,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        client: Any | None = None,
    ) -> None:
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.client = client or chromadb.PersistentClient(
            path=str(self.persist_directory)
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def replace(
        self,
        chunks: list[TranscriptChunk],
        embeddings: list[list[float]],
    ) -> None:
        """Replace the collection contents with one complete index build."""

        if len(chunks) != len(embeddings):
            raise ValueError("Chroma chunks and embeddings must have equal length")
        if not chunks:
            raise ValueError("Cannot persist an empty Chroma collection")

        existing_ids = self.collection.get()["ids"]
        if existing_ids:
            self.collection.delete(ids=existing_ids)

        self.collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            embeddings=embeddings,
            documents=[chunk.retrieval_text for chunk in chunks],
            metadatas=[metadata_for_chunk(chunk) for chunk in chunks],
        )

    def get(self, evidence_id: str) -> dict[str, Any] | None:
        """Return one persisted document, metadata, and embedding by evidence ID."""

        result = self.collection.get(
            ids=[evidence_id],
            include=["documents", "metadatas", "embeddings"],
        )
        if not result["ids"]:
            return None
        return {
            "id": result["ids"][0],
            "document": result["documents"][0],
            "metadata": result["metadatas"][0],
            "embedding": result["embeddings"][0],
        }

    def search(
        self,
        query_embedding: list[float],
        top_k: int,
    ) -> list[tuple[str, float]]:
        """Return Chroma-ranked evidence IDs and cosine-similarity scores."""

        if top_k <= 0 or self.count == 0:
            return []
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.count),
            include=["distances"],
        )
        ids = result.get("ids", [[]])[0]
        distances = result.get("distances", [[]])[0]
        return [
            (evidence_id, 1.0 - float(distance))
            for evidence_id, distance in zip(ids, distances, strict=True)
        ]

    @property
    def count(self) -> int:
        """Return the number of persisted vector records."""

        return self.collection.count()


def dense_search(
    query: str,
    top_k: int,
    *,
    index: "LocalIndex",
) -> list[EvidenceItem]:
    """Embed a query locally and retrieve the nearest Chroma chunks."""

    if top_k <= 0:
        return []
    query_embedding = index.embedder.embed([query])[0]
    ranked = index.vector_store.search(query_embedding, top_k)
    results: list[EvidenceItem] = []
    for evidence_id, score in ranked:
        chunk = index.get_chunk(evidence_id)
        if chunk is not None:
            results.append(
                EvidenceItem.from_chunk(chunk, retrieval_score=score)
            )
    return results
