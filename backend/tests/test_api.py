from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from src import api
from src.schemas import (
    Citation,
    IndexManifest,
    TranscriptChunk,
    VerifiedAnswer,
)


def make_chunk() -> TranscriptChunk:
    return TranscriptChunk(
        chunk_id="france_01_20",
        source_file="Transcript_1_France.txt",
        source_hash="hash",
        country="France",
        expert="Dr. Martin",
        speaker="Dr. Martin",
        question_text="What are the barriers?",
        answer_text="Funding is the main barrier.",
        retrieval_text=(
            "Interviewer: What are the barriers?\n"
            "Dr. Martin: Funding is the main barrier."
        ),
        question_timestamp="00:18",
        answer_timestamp="01:20",
    )


def make_index(chunk: TranscriptChunk | None = None) -> SimpleNamespace:
    item = chunk or make_chunk()
    manifest = IndexManifest(
        embedding_model="fake",
        embedding_dimension=3,
        chroma_collection="test",
        chunks_path="parsed/chunks.jsonl",
        bm25_index_path="indexes/bm25.pkl",
        source_hashes={item.source_file: item.source_hash},
        chunk_ids=[item.chunk_id],
        chunk_count=1,
    )
    return SimpleNamespace(
        chunks=[item],
        manifest=manifest,
        get_chunk=lambda evidence_id: item if evidence_id == item.chunk_id else None,
    )


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(api.state, "load_existing", lambda: None)
    api.state.index = None
    with TestClient(api.app) as test_client:
        yield test_client
    api.state.index = None


def test_status_reports_an_empty_backend_before_ingest(client: TestClient) -> None:
    response = client.get("/api/status")

    assert response.status_code == 200
    assert response.json() == {
        "ready": False,
        "chunk_count": 0,
        "source_files": [],
        "last_indexed_at": None,
    }


def test_ask_requires_a_loaded_index(client: TestClient) -> None:
    response = client.post("/api/ask", json={"question": "What are the barriers?"})

    assert response.status_code == 503
    assert "No index is loaded" in response.json()["detail"]


def test_ask_returns_the_verified_answer_from_the_orchestrator(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    item = make_chunk()
    api.state.index = make_index(item)
    expected = VerifiedAnswer(
        valid=True,
        answer="Funding is the main barrier.",
        claims=[],
        citations=[
            Citation(
                evidence_id=item.chunk_id,
                quote=item.answer_text,
                country=item.country,
                expert=item.expert,
                source_file=item.source_file,
                timestamp=item.answer_timestamp,
            )
        ],
        abstain=False,
    )
    calls: list[tuple[str, object]] = []

    def fake_answer(question: str, index: object) -> VerifiedAnswer:
        calls.append((question, index))
        return expected

    monkeypatch.setattr(api, "answer_question", fake_answer)
    response = client.post("/api/ask", json={"question": "  What are the barriers?  "})

    assert response.status_code == 200
    assert response.json() == expected.model_dump()
    assert calls == [("What are the barriers?", api.state.index)]


def test_evidence_returns_canonical_metadata_and_404s_unknown_ids(client: TestClient) -> None:
    item = make_chunk()
    api.state.index = make_index(item)

    response = client.get(f"/api/evidence/{item.chunk_id}")
    missing = client.get("/api/evidence/missing")

    assert response.status_code == 200
    assert response.json() == {
        "evidence_id": item.chunk_id,
        "text": item.retrieval_text,
        "country": item.country,
        "expert": item.expert,
        "source_file": item.source_file,
        "timestamp": item.answer_timestamp,
        "retrieval_score": None,
        "rerank_score": None,
    }
    assert missing.status_code == 404


def test_ingest_accepts_transcript_filenames_and_replaces_state(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_index = make_index()
    received_paths: list[Path] = []

    def fake_build(paths: list[Path], *, data_dir: Path):
        received_paths.extend(paths)
        assert data_dir == api.DATA_DIR
        return fake_index

    monkeypatch.setattr(api, "build_index", fake_build)
    response = client.post(
        "/api/ingest",
        json={"source_files": ["Transcript_1_France.txt"]},
    )

    assert response.status_code == 200
    assert response.json()["indexed"] is True
    assert response.json()["ready"] is True
    assert response.json()["chunk_count"] == 1
    assert received_paths == [api.RAW_DIR / "Transcript_1_France.txt"]


def test_ingest_rejects_paths_outside_the_raw_transcript_directory(client: TestClient) -> None:
    response = client.post("/api/ingest", json={"source_files": ["../secrets.txt"]})

    assert response.status_code == 400
    assert "transcript filenames" in response.json()["detail"]
