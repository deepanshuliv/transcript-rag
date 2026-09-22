"""Phase 2 persistence, metadata, and restart tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.embeddings import EmbeddingDimensionError, SentenceTransformerEmbedder
from src.ingest import build_index, load_index, read_chunks_jsonl


ROOT = Path(__file__).resolve().parents[2]
TRANSCRIPTS = [
    ROOT / "Transcript_1_France.txt",
    ROOT / "Transcript_2_Germany.txt",
    ROOT / "Transcript_3_UK.txt",
]


class FakeEmbeddingModel:
    """Deterministic test double with the BAAI/bge-m3 output dimension."""

    model_name = "BAAI/bge-m3"
    dimension = 1024

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for index, _text in enumerate(texts):
            vector = np.zeros(self.dimension, dtype=float)
            vector[index % self.dimension] = 1.0
            vectors.append(vector.tolist())
        return vectors


class FakeSentenceTransformer:
    def __init__(self, dimension: int = 1024) -> None:
        self.dimension = dimension
        self.last_kwargs: dict[str, object] = {}

    def encode(self, texts: list[str], **kwargs: object) -> np.ndarray:
        self.last_kwargs = kwargs
        return np.ones((len(texts), self.dimension), dtype=np.float32)


def test_sentence_transformer_wrapper_uses_bge_m3_and_1024_dimensions() -> None:
    model = FakeSentenceTransformer()
    embedder = SentenceTransformerEmbedder(model=model)

    vectors = embedder.embed(["one", "two"])

    assert embedder.model_name == "BAAI/bge-m3"
    assert embedder.dimension == 1024
    assert len(vectors) == 2
    assert len(vectors[0]) == 1024
    assert model.last_kwargs["normalize_embeddings"] is True


def test_sentence_transformer_wrapper_rejects_wrong_embedding_dimension() -> None:
    embedder = SentenceTransformerEmbedder(model=FakeSentenceTransformer(3))

    with pytest.raises(EmbeddingDimensionError, match=r"expected \(1, 1024\)"):
        embedder.embed(["wrong shape"])


def test_build_persists_jsonl_bm25_chroma_and_citation_metadata(tmp_path: Path) -> None:
    index = build_index(TRANSCRIPTS, data_dir=tmp_path, embedder=FakeEmbeddingModel())

    chunks_path = tmp_path / "parsed" / "chunks.jsonl"
    bm25_path = tmp_path / "indexes" / "bm25.pkl"
    manifest_path = tmp_path / "indexes" / "manifest.json"
    assert chunks_path.exists()
    assert bm25_path.exists()
    assert manifest_path.exists()
    assert index.manifest.chunk_count == 21
    assert index.vector_store.count == 21

    persisted_chunks = read_chunks_jsonl(chunks_path)
    assert [chunk.chunk_id for chunk in persisted_chunks] == index.manifest.chunk_ids

    item = index.vector_store.get("france_00_18")
    assert item is not None
    assert item["id"] == "france_00_18"
    assert len(item["embedding"]) == 1024
    assert item["document"] == persisted_chunks[0].retrieval_text
    assert item["metadata"] == {
        "evidence_id": "france_00_18",
        "source_file": "Transcript_1_France.txt",
        "source_hash": persisted_chunks[0].source_hash,
        "country": "France",
        "expert": "Dr. Jean Martin",
        "speaker": "Dr. Martin",
        "question_text": persisted_chunks[0].question_text,
        "answer_text": persisted_chunks[0].answer_text,
        "question_timestamp": "00:00",
        "answer_timestamp": "00:18",
        "char_start": persisted_chunks[0].char_start,
        "char_end": persisted_chunks[0].char_end,
    }

    lexical_hits = index.bm25.search("capital budget approval", top_k=1)
    assert lexical_hits[0][0] == "france_01_20"


def test_restart_loads_without_embedding_or_reparsing(tmp_path: Path) -> None:
    first = build_index(TRANSCRIPTS, data_dir=tmp_path, embedder=FakeEmbeddingModel())
    restarted = load_index(data_dir=tmp_path)

    assert restarted.manifest == first.manifest
    assert restarted.vector_store.count == first.vector_store.count == 21
    assert restarted.bm25.search("maintenance service contracts", top_k=1)[0][0] == (
        "germany_02_08"
    )
    assert restarted.get_chunk("united_kingdom_06_04").answer_text.startswith(
        "The key point"
    )


def test_restart_rejects_manifest_chunk_mismatch(tmp_path: Path) -> None:
    build_index(TRANSCRIPTS, data_dir=tmp_path, embedder=FakeEmbeddingModel())
    chunks_path = tmp_path / "parsed" / "chunks.jsonl"
    lines = chunks_path.read_text(encoding="utf-8").splitlines()
    chunks_path.write_text("\n".join(reversed(lines)) + "\n", encoding="utf-8")

    import pytest

    with pytest.raises(ValueError, match="IDs do not match"):
        load_index(data_dir=tmp_path)
