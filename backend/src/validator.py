"""Phase 7 deterministic claim, quote, timestamp, and citation validation."""

from __future__ import annotations

import re
from collections.abc import Iterable

from .schemas import (
    Citation,
    EvidenceBundle,
    EvidenceItem,
    LLMAnswer,
    VerifiedAnswer,
    VerifiedClaim,
)


WORD_RE = re.compile(r"[\w]+", flags=re.UNICODE)
QUOTE_PATTERNS = (
    re.compile(r'"([^"\n]+)"'),
    re.compile(r"“([^”\n]+)”"),
)
TIMESTAMP_RE = re.compile(r"\b\d{2}:\d{2}\b")
NUMBER_RE = re.compile(r"(?<!\w)\d+(?:[.,]\d+)?\s*%?")

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by",
    "for", "from", "has", "have", "in", "into", "is", "it", "its", "of",
    "on", "or", "that", "the", "their", "this", "to", "was", "were", "with",
}
NEGATION_WORDS = {"no", "not", "never", "neither", "nor", "without"}
ANTONYM_PAIRS = {
    ("accelerate", "slow"),
    ("accelerates", "slows"),
    ("accelerating", "slowing"),
    ("increase", "decrease"),
    ("increasing", "decreasing"),
    ("growth", "decline"),
    ("growing", "declining"),
    ("higher", "lower"),
    ("high", "low"),
    ("longer", "shorter"),
    ("more", "less"),
    ("positive", "negative"),
}


def normalize_text(text: str) -> str:
    """Normalize whitespace and case for exact quote containment checks."""

    return " ".join(text.casefold().split())


def _tokens(text: str) -> set[str]:
    return {token for token in WORD_RE.findall(text.casefold()) if token not in STOPWORDS}


def _source_quote(item: EvidenceItem) -> str:
    """Return only the expert's verbatim answer from a stored Q&A chunk."""

    marker = "\nAnswer:"
    if marker in item.text:
        return item.text.split(marker, 1)[1].strip()
    if item.text.startswith("Answer:"):
        return item.text[len("Answer:") :].strip()
    return item.text.strip()


def _support_text(items: Iterable[EvidenceItem]) -> str:
    return " ".join(
        " ".join(
            [
                _source_quote(item),
                item.country,
                item.expert,
            ]
        )
        for item in items
    )


def _extract_quotes(text: str) -> list[str]:
    quotes: list[str] = []
    for pattern in QUOTE_PATTERNS:
        quotes.extend(pattern.findall(text))
    return quotes


def _unsupported_quotes(text: str, support_text: str) -> list[str]:
    normalized_support = normalize_text(support_text)
    return [
        quote
        for quote in _extract_quotes(text)
        if normalize_text(quote) not in normalized_support
    ]


def _numbers_supported(claim: str, support_text: str) -> bool:
    claim_numbers = {normalize_text(number) for number in NUMBER_RE.findall(claim)}
    support_numbers = {
        normalize_text(number) for number in NUMBER_RE.findall(support_text)
    }
    return claim_numbers.issubset(support_numbers)


def _claim_supported(claim: str, support_text: str) -> bool:
    claim_tokens = _tokens(claim)
    if not claim_tokens:
        return bool(claim.strip())
    support_tokens = _tokens(support_text)
    overlap = claim_tokens & support_tokens
    # This is a conservative deterministic entailment proxy. Numeric and
    # timestamp checks below are stricter and run independently.
    return len(overlap) / len(claim_tokens) >= 0.5


def _timestamps_supported(text: str, items: list[EvidenceItem]) -> bool:
    allowed = {item.timestamp for item in items}
    return set(TIMESTAMP_RE.findall(text)).issubset(allowed)


def _has_negation(text: str) -> bool:
    return bool(_tokens(text) & NEGATION_WORDS)


def _contradicts(first: str, second: str) -> bool:
    first_tokens = _tokens(first)
    second_tokens = _tokens(second)
    shared = first_tokens & second_tokens
    if len(shared) >= 2 and _has_negation(first) != _has_negation(second):
        return True
    for left, right in ANTONYM_PAIRS:
        if (left in first_tokens and right in second_tokens) or (
            right in first_tokens and left in second_tokens
        ):
            return True
    return False


def _detect_disagreement(items: list[EvidenceItem]) -> bool:
    return any(
        _contradicts(first.text, second.text)
        for index, first in enumerate(items)
        for second in items[index + 1 :]
    )


def _citation(item: EvidenceItem) -> Citation:
    """Build citation fields only from the bundle's stored metadata."""

    return Citation(
        evidence_id=item.evidence_id,
        quote=_source_quote(item),
        country=item.country,
        expert=item.expert,
        source_file=item.source_file,
        timestamp=item.timestamp,
    )


def _supporting_quote(claim: str, item: EvidenceItem) -> str:
    """Choose the shortest verbatim answer passage with sufficient word overlap."""

    answer = _source_quote(item)
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", answer)
        if sentence.strip()
    ]
    claim_tokens = _tokens(claim)
    if not claim_tokens or not sentences:
        return answer

    best_quote = answer
    best_length = len(answer)
    for start in range(len(sentences)):
        quote_parts: list[str] = []
        for end in range(start, len(sentences)):
            quote_parts.append(sentences[end])
            quote = " ".join(quote_parts)
            overlap = len(claim_tokens & _tokens(quote)) / len(claim_tokens)
            if overlap >= 0.5 and len(quote) < best_length:
                best_quote = quote
                best_length = len(quote)
                break
    return best_quote


def _invalid(reason: str, answer: str = "") -> VerifiedAnswer:
    return VerifiedAnswer(
        valid=False,
        answer=answer,
        claims=[],
        citations=[],
        abstain=True,
        abstain_reason=reason,
    )


def validate_answer(
    llm_answer: LLMAnswer,
    evidence_bundle: EvidenceBundle,
    *,
    required_countries: set[str] | None = None,
    required_evidence_ids: set[str] | None = None,
) -> VerifiedAnswer:
    """Validate a model answer against only the current in-memory bundle."""

    bundle_ids = set(evidence_bundle.items)
    if (
        len(evidence_bundle.ordered_ids) != len(bundle_ids)
        or set(evidence_bundle.ordered_ids) != bundle_ids
        or any(
            evidence_id != item.evidence_id
            for evidence_id, item in evidence_bundle.items.items()
        )
    ):
        return _invalid("EvidenceBundle IDs and ordering are inconsistent.")

    if llm_answer.abstain:
        return VerifiedAnswer(
            valid=True,
            answer="",
            claims=[],
            citations=[],
            abstain=True,
            abstain_reason=llm_answer.abstain_reason or "The model abstained.",
        )
    if not llm_answer.claims:
        return _invalid("The answer contains no evidence-backed claims.")

    verified_claims: list[VerifiedClaim] = []
    all_citations: dict[str, Citation] = {}
    for claim in llm_answer.claims:
        if not claim.claim.strip():
            return _invalid("An answer claim is empty.", llm_answer.answer)
        if not claim.evidence_ids:
            return _invalid(
                f"Claim has no evidence IDs: {claim.claim}", llm_answer.answer
            )
        if len(set(claim.evidence_ids)) != len(claim.evidence_ids):
            return _invalid(
                f"Claim repeats an evidence ID: {claim.claim}", llm_answer.answer
            )
        unknown_ids = [
            evidence_id
            for evidence_id in claim.evidence_ids
            if evidence_id not in bundle_ids
        ]
        if unknown_ids:
            return _invalid(
                f"Claim cites unknown evidence IDs: {', '.join(unknown_ids)}",
                llm_answer.answer,
            )

        cited_items = [evidence_bundle.items[evidence_id] for evidence_id in claim.evidence_ids]
        support_text = _support_text(cited_items)
        if not _claim_supported(claim.claim, support_text):
            return _invalid(
                f"Claim is not supported by its cited evidence: {claim.claim}",
                llm_answer.answer,
            )
        if not _numbers_supported(claim.claim, support_text):
            return _invalid(
                f"Claim contains unsupported numbers: {claim.claim}",
                llm_answer.answer,
            )
        if not _timestamps_supported(claim.claim, cited_items):
            return _invalid(
                f"Claim contains an uncited timestamp: {claim.claim}",
                llm_answer.answer,
            )
        unsupported_quotes = _unsupported_quotes(claim.claim, support_text)
        if unsupported_quotes:
            return _invalid(
                f"Claim contains a quote absent from evidence: {unsupported_quotes[0]}",
                llm_answer.answer,
            )

        citations = [
            _citation(item).model_copy(
                update={"quote": _supporting_quote(claim.claim, item)}
            )
            for item in cited_items
        ]
        verified_claims.append(
            VerifiedClaim(
                claim=claim.claim,
                evidence_ids=list(claim.evidence_ids),
                citations=citations,
                disagreement=_detect_disagreement(cited_items),
            )
        )
        for citation in citations:
            all_citations[citation.evidence_id] = citation

    if required_countries:
        cited_countries = {
            citation.country.casefold() for citation in all_citations.values()
        }
        missing_countries = {
            country.casefold() for country in required_countries
        } - cited_countries
        if missing_countries:
            return _invalid(
                "Answer citations do not cover every required market: "
                + ", ".join(sorted(missing_countries)),
                llm_answer.answer,
            )

    if required_evidence_ids is not None:
        missing_ids = [
            evidence_id
            for evidence_id in evidence_bundle.ordered_ids
            if evidence_id in required_evidence_ids
            and evidence_id not in all_citations
        ]
        if missing_ids:
            return _invalid(
                "The answer omitted required evidence citations: "
                + ", ".join(missing_ids),
                llm_answer.answer,
            )

    ordered_citations = [
        all_citations[evidence_id]
        for evidence_id in evidence_bundle.ordered_ids
        if evidence_id in all_citations
    ]
    return VerifiedAnswer(
        valid=True,
        answer="\n\n".join(claim.claim for claim in verified_claims),
        claims=verified_claims,
        citations=ordered_citations,
        abstain=False,
    )
