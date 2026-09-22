"""Phase 2 local ingestion and persistence orchestration."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

from .config import (
    CHROMA_DIR,
    DATA_DIR,
    DEFAULT_COLLECTION_NAME,
    INDEXES_DIR,
    PARSED_DIR,
    settings,
)
from .embeddings import EmbeddingModel, create_embedder
from .lexical_search import BM25Index
from .parser import parse_transcripts
from .schemas import IndexManifest, TranscriptChunk
from .vector_search import ChromaVectorStore


CHUNKS_FILENAME = "chunks.jsonl"
BM25_FILENAME = "bm25.pkl"
MANIFEST_FILENAME = "manifest.json"


def _atomic_write_text(path: Path, text: str) -> None:
    """Write a text artifact and replace the destination atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(text, encoding="utf-8")
    os.replace(temporary_path, path)


def write_chunks_jsonl(
    chunks: Iterable[TranscriptChunk],
    path: str | Path,
) -> list[TranscriptChunk]:
    """Persist validated chunks as one Pydantic JSON object per line."""

    materialized = list(chunks)
    if not materialized:
        raise ValueError("Cannot persist an empty chunk set")
    ids = [chunk.chunk_id for chunk in materialized]
    if len(set(ids)) != len(ids):
        raise ValueError("Chunk IDs must be unique before persistence")

    jsonl = "".join(f"{chunk.model_dump_json()}\n" for chunk in materialized)
    _atomic_write_text(Path(path), jsonl)
    return materialized


def read_chunks_jsonl(path: str | Path) -> list[TranscriptChunk]:
    """Read and validate a persisted JSONL chunk artifact."""

    chunks: list[TranscriptChunk] = []
    seen_ids: set[str] = set()
    with Path(path).open("r", encoding="utf-8", newline="") as file_handle:
        for line_number, line in enumerate(file_handle, start=1):
            if not line.strip():
                continue
            try:
                chunk = TranscriptChunk.model_validate_json(line)
            except Exception as exc:
                raise ValueError(
                    f"Invalid TranscriptChunk JSON on line {line_number}: {exc}"
                ) from exc
            if chunk.chunk_id in seen_ids:
                raise ValueError(f"Duplicate chunk ID in JSONL: {chunk.chunk_id}")
            seen_ids.add(chunk.chunk_id)
            chunks.append(chunk)
    if not chunks:
        raise ValueError("Chunk JSONL contains no chunks")
    return chunks


class LocalIndex:
    """The three restartable Phase 2 artifacts and their validated manifest."""

    def __init__(
        self,
        *,
        chunks: list[TranscriptChunk],
        manifest: IndexManifest,
        bm25: BM25Index,
        vector_store: ChromaVectorStore,
        embedder: EmbeddingModel,
    ) -> None:
        self.chunks = chunks
        self.manifest = manifest
        self.bm25 = bm25
        self.vector_store = vector_store
        self.embedder = embedder
        self._chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}

    def get_chunk(self, evidence_id: str) -> TranscriptChunk | None:
        """Return the JSONL-backed canonical chunk for citation validation."""

        return self._chunks_by_id.get(evidence_id)

    @classmethod
    def build(
        cls,
        transcript_paths: Iterable[str | Path],
        *,
        data_dir: str | Path = DATA_DIR,
        embedder: EmbeddingModel | None = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ) -> "LocalIndex":
        """Parse, embed, and persist a complete local index."""

        paths = [Path(path) for path in transcript_paths]
        if not paths:
            raise ValueError("At least one transcript path is required")

        chunks = parse_transcripts(paths)
        embedding_model = embedder or create_embedder()
        embeddings = embedding_model.embed(
            [chunk.retrieval_text for chunk in chunks]
        )

        root = Path(data_dir)
        parsed_dir = root / PARSED_DIR.name
        chroma_dir = root / CHROMA_DIR.name
        indexes_dir = root / INDEXES_DIR.name
        chunks_path = parsed_dir / CHUNKS_FILENAME
        bm25_path = indexes_dir / BM25_FILENAME
        manifest_path = indexes_dir / MANIFEST_FILENAME

        persisted_chunks = write_chunks_jsonl(chunks, chunks_path)
        bm25 = BM25Index.from_chunks(persisted_chunks)
        bm25.save(bm25_path)

        vector_store = ChromaVectorStore(
            persist_directory=chroma_dir,
            collection_name=collection_name,
        )
        vector_store.replace(persisted_chunks, embeddings)

        source_hashes = {
            chunk.source_file: chunk.source_hash for chunk in persisted_chunks
        }
        manifest = IndexManifest(
            embedding_provider=getattr(embedding_model, "provider", "local"),
            embedding_model=embedding_model.model_name,
            embedding_dimension=embedding_model.dimension,
            chroma_collection=collection_name,
            chunks_path=str(chunks_path.relative_to(root)),
            bm25_index_path=str(bm25_path.relative_to(root)),
            source_hashes=source_hashes,
            chunk_ids=[chunk.chunk_id for chunk in persisted_chunks],
            chunk_count=len(persisted_chunks),
        )
        _atomic_write_text(
            manifest_path,
            manifest.model_dump_json(indent=2) + "\n",
        )
        return cls(
            chunks=persisted_chunks,
            manifest=manifest,
            bm25=bm25,
            vector_store=vector_store,
            embedder=embedding_model,
        )

    @classmethod
    def load(
        cls,
        *,
        data_dir: str | Path = DATA_DIR,
        collection_name: str | None = None,
        embedder: EmbeddingModel | None = None,
    ) -> "LocalIndex":
        """Reload all Phase 2 artifacts without reparsing or embedding."""

        root = Path(data_dir)
        manifest_path = root / INDEXES_DIR.name / MANIFEST_FILENAME
        if not manifest_path.exists():
            raise FileNotFoundError(f"Index manifest not found: {manifest_path}")
        manifest = IndexManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
        if collection_name is not None and collection_name != manifest.chroma_collection:
            raise ValueError(
                f"Requested collection '{collection_name}' does not match the manifest "
                f"collection '{manifest.chroma_collection}'"
            )

        chunks = read_chunks_jsonl(root / manifest.chunks_path)
        actual_ids = [chunk.chunk_id for chunk in chunks]
        if actual_ids != manifest.chunk_ids:
            raise ValueError("Chunk JSONL IDs do not match the index manifest")
        if len(chunks) != manifest.chunk_count:
            raise ValueError("Chunk JSONL count does not match the index manifest")
        actual_hashes = {
            chunk.source_file: chunk.source_hash for chunk in chunks
        }
        if actual_hashes != manifest.source_hashes:
            raise ValueError("Chunk source hashes do not match the index manifest")

        bm25 = BM25Index.load(root / manifest.bm25_index_path)
        if bm25.chunk_ids != manifest.chunk_ids:
            raise ValueError("BM25 chunk IDs do not match the index manifest")

        vector_store = ChromaVectorStore(
            persist_directory=root / CHROMA_DIR.name,
            collection_name=manifest.chroma_collection,
        )
        if vector_store.count != manifest.chunk_count:
            raise ValueError("Chroma count does not match the index manifest")

        if embedder is not None:
            embedding_model = embedder
        else:
            embedding_model = create_embedder(
                provider=manifest.embedding_provider,
                model_name=manifest.embedding_model,
                dimension=manifest.embedding_dimension,
            )
        if getattr(embedding_model, "provider", "local") != manifest.embedding_provider:
            raise ValueError("Embedding provider does not match the index manifest")
        if embedding_model.model_name != manifest.embedding_model:
            raise ValueError("Embedding model does not match the index manifest")
        if embedding_model.dimension != manifest.embedding_dimension:
            raise ValueError("Embedding dimension does not match the index manifest")

        return cls(
            chunks=chunks,
            manifest=manifest,
            bm25=bm25,
            vector_store=vector_store,
            embedder=embedding_model,
        )


def build_index(
    transcript_paths: Iterable[str | Path],
    *,
    data_dir: str | Path = DATA_DIR,
    embedder: EmbeddingModel | None = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> LocalIndex:
    """Convenience wrapper for Phase 2 index construction."""

    return LocalIndex.build(
        transcript_paths,
        data_dir=data_dir,
        embedder=embedder,
        collection_name=collection_name,
    )


def load_index(
    *,
    data_dir: str | Path = DATA_DIR,
    collection_name: str | None = None,
    embedder: EmbeddingModel | None = None,
) -> LocalIndex:
    """Convenience wrapper for restarting from persisted Phase 2 artifacts."""

    return LocalIndex.load(
        data_dir=data_dir,
        collection_name=collection_name,
        embedder=embedder,
    )
