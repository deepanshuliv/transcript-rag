from __future__ import annotations

import json
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
    api._answer_cache.clear()
    with TestClient(api.app) as test_client:
        yield test_client
    api.state.index = None


def test_identical_query_reuses_the_same_verified_answer(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    item = make_chunk()
    api.state.index = make_index(item)
    answer = VerifiedAnswer(
        valid=True,
        answer="The verified market summary.",
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
    calls = 0

    def fake_answer(*args: object, **kwargs: object) -> VerifiedAnswer:
        nonlocal calls
        calls += 1
        return answer

    monkeypatch.setattr(api, "answer_question", fake_answer)
    first = client.post("/api/ask/stream", json={"question": "Summarize adoption"})
    second = client.post("/api/ask/stream", json={"question": "Summarize adoption"})

    assert calls == 1
    assert '"stage": "cache_hit"' in second.text
    assert answer.answer in first.text
    assert answer.answer in second.text

    def events(response_text: str) -> list[dict[str, object]]:
        return [
            json.loads(line.removeprefix("data: "))
            for line in response_text.splitlines()
            if line.startswith("data: ")
        ]

    first_events = events(first.text)
    second_events = events(second.text)
    first_done = next(event for event in first_events if event["type"] == "done")
    second_done = next(event for event in second_events if event["type"] == "done")
    first_citations = next(event for event in first_events if event["type"] == "citations")
    second_citations = next(event for event in second_events if event["type"] == "citations")
    assert first_done["answer"] == second_done["answer"]
    assert first_citations["citations"] == second_citations["citations"]


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


def test_stream_ask_emits_progress_citations_deltas_and_done(
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

    def fake_answer(
        question: str,
        index: object,
        *,
        request_id: str | None = None,
        trace=None,
        on_answer_delta=None,
        on_answer_reset=None,
    ) -> VerifiedAnswer:
        if on_answer_delta is not None:
            on_answer_delta("Funding is the main barrier.")
        if trace is not None:
            trace("query_planning", 1.25, {"search_queries": 1})
            trace("total", 2.5, {})
        return expected

    monkeypatch.setattr(api, "answer_question", fake_answer)
    response = client.post("/api/ask/stream", json={"question": "What are the barriers?"})

    assert response.status_code == 200
    assert "\"type\": \"status\"" in response.text
    assert "\"type\": \"timing\"" in response.text
    assert "answer_stream" in response.text
    assert "\"type\": \"citations\"" in response.text
    assert "\"type\": \"delta\"" in response.text
    assert "\"type\": \"done\"" in response.text
    assert expected.answer in response.text
    assert response.text.index('"type": "delta"') < response.text.index('"type": "citations"')


def test_stream_ask_logs_internal_errors_but_only_sends_safe_message(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    api.state.index = make_index()

    def failing_answer(*args: object, **kwargs: object) -> VerifiedAnswer:
        raise RuntimeError("provider payload must stay in backend logs")

    monkeypatch.setattr(api, "answer_question", failing_answer)
    response = client.post("/api/ask/stream", json={"question": "Question?"})

    assert response.status_code == 200
    assert "The request could not be completed. Please try again." in response.text
    assert "provider payload must stay in backend logs" not in response.text
    assert "provider payload must stay in backend logs" in caplog.text


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
