"""Phase 5–7 evidence bundle, DeepSeek generation, and validation tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.answer import ANSWER_SYSTEM_PROMPT, answer_question, generate_answer
from src.evidence import create_evidence_bundle, evidence_is_sufficient
from src.ingest import build_index
from src.reranker import NoOpReranker
from src.schemas import EvidenceItem, LLMAnswer, QueryPlan
from src.validator import validate_answer


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
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]


class FakeAnswerClient:
    def __init__(self, answers: list[dict[str, Any]]) -> None:
        self.answers = [json.dumps(answer) for answer in answers]
        self.calls: list[dict[str, object]] = []

    def complete_json(self, messages: list[dict[str, str]], **kwargs: object) -> str:
        self.calls.append({"messages": messages, **kwargs})
        answer = self.answers.pop(0)
        on_content_delta = kwargs.get("on_content_delta")
        if callable(on_content_delta):
            for offset in range(0, len(answer), 11):
                on_content_delta(answer[offset : offset + 11])
        return answer


def plan_for(*search_queries: str, intent: str = "fact_lookup") -> QueryPlan:
    return QueryPlan(
        original_query="question",
        normalized_query="question",
        search_queries=list(search_queries),
        countries=[],
        experts=[],
        intent=intent,  # type: ignore[arg-type]
        requires_all_countries=False,
    )


def item(
    evidence_id: str,
    text: str,
    *,
    country: str = "France",
    timestamp: str = "01:20",
) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=evidence_id,
        text=text,
        country=country,
        expert="Dr. Jean Martin",
        source_file="Transcript_1_France.txt",
        timestamp=timestamp,
    )


def test_evidence_bundle_preserves_reranked_order_and_ids() -> None:
    bundle = create_evidence_bundle(
        "question",
        [item("b", "second"), item("a", "first"), item("b", "duplicate")],
        request_id="request-1",
    )

    assert bundle.request_id == "request-1"
    assert bundle.ordered_ids == ["b", "a"]
    assert list(bundle.items) == ["b", "a"]
    assert bundle.items["b"].text == "second"


def test_evidence_gate_rejects_missing_comparison_country() -> None:
    plan = QueryPlan(
        original_query="Compare France and Germany",
        normalized_query="compare France and Germany",
        search_queries=["France", "Germany"],
        countries=["France", "Germany"],
        experts=[],
        intent="comparison",
        requires_all_countries=True,
    )

    sufficient, reason = evidence_is_sufficient(
        plan.original_query,
        [item("fr", "capital budget approval", country="France")],
        query_plan=plan,
    )

    assert sufficient is False
    assert "Germany" in reason


def test_generate_answer_sends_only_bundle_evidence_and_fixed_schema() -> None:
    evidence_bundle = create_evidence_bundle(
        "What is holding adoption back?",
        [item("france_01_20", "Question: What is holding adoption back? Answer: Capital budget approval.")],
    )
    client = FakeAnswerClient(
        [
            {
                "answer": "Capital budget approval is the biggest issue.",
                "claims": [
                    {
                        "claim": "Capital budget approval is the biggest issue.",
                        "evidence_ids": ["france_01_20"],
                    }
                ],
                "abstain": False,
                "abstain_reason": None,
            }
        ]
    )

    streamed: list[str] = []
    draft = generate_answer(
        evidence_bundle.query,
        evidence_bundle,
        client=client,
        on_delta=streamed.append,
    )

    assert draft.answer.startswith("Capital budget")
    assert client.calls[0]["schema_name"] == "llm_answer"
    assert client.calls[0]["reasoning_effort"] == "minimal"
    assert "".join(streamed) == draft.answer
    response_schema = client.calls[0]["json_schema"]
    assert response_schema["required"] == list(response_schema["properties"])
    assert "default" not in response_schema["properties"]["abstain_reason"]
    assert response_schema["properties"]["abstain_reason"]["anyOf"] == [
        {"type": "string"},
        {"type": "null"},
    ]
    assert ANSWER_SYSTEM_PROMPT in client.calls[0]["messages"][0]["content"]
    evidence_message = client.calls[0]["messages"][1]["content"]
    assert "france_01_20" in evidence_message
    assert "Capital budget approval" in evidence_message


def test_validator_reconstructs_citations_from_bundle_metadata() -> None:
    evidence_bundle = create_evidence_bundle(
        "What is holding adoption back?",
        [item("france_01_20", "Capital budget approval is the biggest issue.")],
    )
    answer = LLMAnswer(
        answer='"Capital budget approval is the biggest issue."',
        claims=[
            {
                "claim": '"Capital budget approval is the biggest issue."',
                "evidence_ids": ["france_01_20"],
            }
        ],
        abstain=False,
    )

    verified = validate_answer(answer, evidence_bundle)

    assert verified.valid is True
    assert verified.abstain is False
    assert verified.citations[0].evidence_id == "france_01_20"
    assert verified.citations[0].timestamp == "01:20"
    assert verified.citations[0].source_file == "Transcript_1_France.txt"
    assert verified.citations[0].quote == evidence_bundle.items["france_01_20"].text


def test_validator_rejects_unknown_id_unsupported_number_and_bad_quote() -> None:
    evidence_bundle = create_evidence_bundle(
        "question",
        [item("france_01_20", "Capital budget approval is the biggest issue.")],
    )

    unknown_id = validate_answer(
        LLMAnswer(
            answer="The evidence says this.",
            claims=[{"claim": "The evidence says this.", "evidence_ids": ["missing"]}],
            abstain=False,
        ),
        evidence_bundle,
    )
    unsupported_number = validate_answer(
        LLMAnswer(
            answer="There were 99 systems.",
            claims=[
                {
                    "claim": "There were 99 systems.",
                    "evidence_ids": ["france_01_20"],
                }
            ],
            abstain=False,
        ),
        evidence_bundle,
    )
    bad_quote = validate_answer(
        LLMAnswer(
            answer='"An invented quote."',
            claims=[
                {
                    "claim": '"An invented quote."',
                    "evidence_ids": ["france_01_20"],
                }
            ],
            abstain=False,
        ),
        evidence_bundle,
    )

    assert unknown_id.valid is False
    assert unsupported_number.valid is False
    assert bad_quote.valid is False


def test_validator_labels_contradictory_evidence_as_disagreement() -> None:
    evidence_bundle = create_evidence_bundle(
        "Is finance the only decision factor?",
        [
            item(
                "france_04_08",
                "Finance alone decides the purchase.",
                country="France",
            ),
            item(
                "uk_03_10",
                "Finance alone does not decide the purchase; clinical strategy is balanced.",
                country="United Kingdom",
                timestamp="03:10",
            ),
        ],
    )
    answer = LLMAnswer(
        answer="Experts disagree about whether finance alone decides the purchase.",
        claims=[
            {
                "claim": "Experts disagree about whether finance alone decides the purchase.",
                "evidence_ids": ["france_04_08", "uk_03_10"],
            }
        ],
        abstain=False,
    )

    verified = validate_answer(answer, evidence_bundle)

    assert verified.valid is True
    assert verified.claims[0].disagreement is True


def test_answer_question_retries_invalid_draft_then_returns_verified_answer(
    tmp_path: Path,
) -> None:
    index = build_index(TRANSCRIPTS, data_dir=tmp_path, embedder=FakeEmbedder())
    client = FakeAnswerClient(
        [
            {
                "answer": "France approved 99 systems.",
                "claims": [
                    {
                        "claim": "France approved 99 systems.",
                        "evidence_ids": ["france_01_20"],
                    }
                ],
                "abstain": False,
            },
            {
                "answer": "Capital budget approval is the biggest issue.",
                "claims": [
                    {
                        "claim": "Capital budget approval is the biggest issue.",
                        "evidence_ids": ["france_01_20"],
                    }
                ],
                "abstain": False,
            },
        ]
    )

    verified = answer_question(
        "What is holding adoption back?",
        index,
        plan_builder=lambda _query: plan_for("capital budget approval"),
        reranker=NoOpReranker(),
        llm_client=client,
        request_id="request-1",
        country_scope=["France"],
        max_evidence_items=1,
    )

    assert verified.valid is True
    assert verified.abstain is False
    assert verified.citations[0].evidence_id == "france_01_20"
    assert len(client.calls) == 2
    assert "failed deterministic validation" in client.calls[1]["messages"][1]["content"]
