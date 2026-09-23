"""Deterministic cross-market and citation coverage tests."""

from src.evidence import create_evidence_bundle
from src.schemas import AnswerClaim, EvidenceItem, LLMAnswer
from src.validator import validate_answer


def evidence(evidence_id: str, country: str, text: str) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=evidence_id,
        text=f"Question: What matters?\nAnswer: {text}",
        country=country,
        expert=f"Expert {country}",
        source_file=f"{country}.txt",
        timestamp="01:20",
    )


def test_answer_requires_every_selected_item_and_market_to_be_cited() -> None:
    bundle = create_evidence_bundle(
        "Summarize the markets",
        [
            evidence("fr", "France", "Capital budgets shape hospital adoption."),
            evidence("de", "Germany", "Staff training shapes hospital adoption."),
            evidence("uk", "United Kingdom", "Clinical evidence shapes hospital adoption."),
        ],
    )
    answer = LLMAnswer(
        answer=(
            "Capital budgets shape hospital adoption. Staff training shapes hospital adoption. "
            "Clinical evidence shapes hospital adoption."
        ),
        claims=[
            AnswerClaim(
                claim="Capital budgets shape hospital adoption in France.",
                evidence_ids=["fr"],
            ),
            AnswerClaim(
                claim="Staff training shapes hospital adoption in Germany.",
                evidence_ids=["de"],
            ),
            AnswerClaim(
                claim="Clinical evidence shapes hospital adoption in the United Kingdom.",
                evidence_ids=["uk"],
            ),
        ],
        abstain=False,
    )

    verified = validate_answer(
        answer,
        bundle,
        required_countries={"france", "germany", "united kingdom"},
        required_evidence_ids=set(bundle.ordered_ids),
    )

    assert verified.valid
    assert [citation.evidence_id for citation in verified.citations] == ["fr", "de", "uk"]


def test_answer_rejects_missing_market_or_selected_citation() -> None:
    bundle = create_evidence_bundle(
        "Summarize the markets",
        [
            evidence("fr", "France", "Capital budgets shape hospital adoption."),
            evidence("de", "Germany", "Staff training shapes hospital adoption."),
        ],
    )
    answer = LLMAnswer(
        answer="Capital budgets shape hospital adoption.",
        claims=[
            AnswerClaim(
                claim="Capital budgets shape hospital adoption in France.",
                evidence_ids=["fr"],
            )
        ],
        abstain=False,
    )

    verified = validate_answer(
        answer,
        bundle,
        required_countries={"france", "germany"},
        required_evidence_ids=set(bundle.ordered_ids),
    )

    assert not verified.valid
    assert "germany" in (verified.abstain_reason or "").casefold()
