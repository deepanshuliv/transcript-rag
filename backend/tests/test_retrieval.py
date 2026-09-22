"""Phase 4 dense, BM25, fusion, and reranking tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.fusion import reciprocal_rank_fusion
from src.ingest import build_index
from src.lexical_search import bm25_search
from src.reranker import NoOpReranker, OpenRouterCohereReranker
from src.schemas import EvidenceItem
from src.vector_search import dense_search


ROOT = Path(__file__).resolve().parents[2]
TRANSCRIPTS = [
    ROOT / "Transcript_1_France.txt",
    ROOT / "Transcript_2_Germany.txt",
    ROOT / "Transcript_3_UK.txt",
]


class FakeEmbedder:
    model_name = "BAAI/bge-m3"
    dimension = 4

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            lower = text.casefold()
            vectors.append(
                [
                    1.0 if "capital" in lower or "funding" in lower else 0.0,
                    1.0 if "training" in lower else 0.0,
                    1.0 if "timeline" in lower or "months" in lower else 0.0,
                    1.0 if "outlook" in lower or "growth" in lower else 0.0,
                ]
            )
        return vectors


def test_dense_and_bm25_search_return_citation_ready_items(tmp_path: Path) -> None:
    index = build_index(TRANSCRIPTS, data_dir=tmp_path, embedder=FakeEmbedder())

    dense = dense_search("funding barriers", 3, index=index)
    lexical = bm25_search("capital budget approval", 3, index=index)

    assert dense
    assert lexical
    for item in [*dense, *lexical]:
        assert item.evidence_id
        assert item.text
        assert item.country
        assert item.expert
        assert item.source_file.endswith(".txt")
        assert item.timestamp
    assert lexical[0].evidence_id == "france_01_20"


def evidence(evidence_id: str) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=evidence_id,
        text=f"text for {evidence_id}",
        country="France",
        expert="Expert",
        source_file="source.txt",
        timestamp="00:00",
    )


def test_rrf_deduplicates_and_rewards_cross_retriever_agreement() -> None:
    first = [evidence("a"), evidence("b"), evidence("c")]
    second = [evidence("c"), evidence("a"), evidence("d")]

    fused = reciprocal_rank_fusion([first, second], rank_constant=1)

    assert [item.evidence_id for item in fused] == ["a", "c", "b", "d"]
    assert len({item.evidence_id for item in fused}) == 4
    assert fused[0].retrieval_score > fused[2].retrieval_score


class FakeHttpClient:
    def __init__(self, response: object | None = None, error: Exception | None = None):
        self.response = response
        self.error = error
        self.endpoint: str | None = None
        self.request: dict[str, object] | None = None

    def post(self, endpoint: str, **kwargs: object) -> object:
        self.endpoint = endpoint
        self.request = kwargs
        if self.error:
            raise self.error
        return self.response


class FakeResponse:
    def __init__(self, payload: object):
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> object:
        return self.payload


def test_cohere_reranker_maps_indices_and_preserves_metadata() -> None:
    items = [evidence("a"), evidence("b"), evidence("c")]
    http_client = FakeHttpClient(
        response=FakeResponse(
            {
                "results": [
                    {"index": 2, "relevance_score": 0.91},
                    {"index": 0, "relevance_score": 0.71},
                    {"index": 1, "relevance_score": 0.42},
                ]
            }
        )
    )
    reranker = OpenRouterCohereReranker(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        model="cohere/rerank-v3.5:free",
        http_client=http_client,
    )

    reranked = reranker.rerank("find relevant evidence", items)

    assert [item.evidence_id for item in reranked] == ["c", "a", "b"]
    assert reranked[0].country == "France"
    assert reranked[0].expert == "Expert"
    assert reranked[0].timestamp == "00:00"
    assert reranked[0].rerank_score == 0.91
    assert http_client.endpoint == "https://openrouter.ai/api/v1/rerank"
    assert http_client.request["json"] == {
        "model": "cohere/rerank-v3.5:free",
        "query": "find relevant evidence",
        "documents": [item.text for item in items],
        "top_n": 3,
    }


def test_reranker_provider_error_keeps_rrf_order() -> None:
    items = [evidence("a"), evidence("b")]
    http_client = FakeHttpClient(error=RuntimeError("rate limited"))
    reranker = OpenRouterCohereReranker(
        api_key="test-key",
        http_client=http_client,
    )

    assert reranker.rerank("query", items) == items
    assert isinstance(reranker.last_error, RuntimeError)
    assert NoOpReranker().rerank("query", items) == items
