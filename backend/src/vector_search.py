"""Persistent Chroma vector-store wrapper for Phase 2."""

from __future__ import annotations

from collections.abc import Sequence
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
        *,
        countries: list[str] | None = None,
        experts: list[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Return Chroma-ranked evidence IDs and cosine-similarity scores."""

        if top_k <= 0 or self.count == 0:
            return []
        filters: list[dict[str, Any]] = []
        if countries:
            filters.append({"country": {"$in": countries}})
        if experts:
            filters.append({"expert": {"$in": experts}})
        where = None
        if len(filters) == 1:
            where = filters[0]
        elif filters:
            where = {"$and": filters}

        available = self.count
        if countries or experts:
            available = len(self.collection.get(where=where, include=["metadatas"])["ids"])
        if available == 0:
            return []
        query_options = {
            "query_embeddings": [query_embedding],
            "n_results": min(top_k, available),
            "include": ["distances"],
        }
        if where is not None:
            query_options["where"] = where
        result = self.collection.query(
            **query_options,
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
    countries: list[str] | None = None,
    experts: list[str] | None = None,
) -> list[EvidenceItem]:
    """Embed one query and retrieve the nearest Chroma chunks."""

    return dense_search_many(
        [query],
        top_k,
        index=index,
        countries=countries,
        experts=experts,
    )[0]


def dense_search_many(
    queries: Sequence[str],
    top_k: int,
    *,
    index: "LocalIndex",
    countries: list[str] | None = None,
    experts: list[str] | None = None,
) -> list[list[EvidenceItem]]:
    """Embed all planned queries in one request, then search each vector."""

    if not queries:
        return []
    if top_k <= 0:
        return [[] for _query in queries]

    query_embeddings = index.embedder.embed(queries)
    results_by_query: list[list[EvidenceItem]] = []
    for query_embedding in query_embeddings:
        ranked = index.vector_store.search(
            query_embedding,
            top_k,
            countries=countries,
            experts=experts,
        )
        results: list[EvidenceItem] = []
        for evidence_id, score in ranked:
            chunk = index.get_chunk(evidence_id)
            if chunk is not None:
                results.append(EvidenceItem.from_chunk(chunk, retrieval_score=score))
        results_by_query.append(results)
    return results_by_query


def dense_search_many_by_country(
    queries: Sequence[str],
    top_k: int,
    *,
    index: "LocalIndex",
    countries: Sequence[str],
    experts: list[str] | None = None,
) -> dict[str, list[list[EvidenceItem]]]:
    """Retrieve each query independently inside each requested country."""

    if not queries or top_k <= 0:
        return {country: [[] for _query in queries] for country in countries}
    query_embeddings = index.embedder.embed(queries)
    grouped: dict[str, list[list[EvidenceItem]]] = {}
    for country in countries:
        country_results: list[list[EvidenceItem]] = []
        for query_embedding in query_embeddings:
            ranked = index.vector_store.search(
                query_embedding,
                top_k,
                countries=[country],
                experts=experts,
            )
            country_results.append(
                [
                    EvidenceItem.from_chunk(chunk, retrieval_score=score)
                    for evidence_id, score in ranked
                    if (chunk := index.get_chunk(evidence_id)) is not None
                ]
            )
        grouped[country] = country_results
    return grouped
