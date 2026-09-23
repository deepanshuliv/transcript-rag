"""Phase 5 in-memory evidence selection and sufficiency checks."""

from __future__ import annotations

from collections.abc import Iterable
from uuid import uuid4

from .schemas import EvidenceBundle, EvidenceItem, QueryPlan


DEFAULT_MAX_EVIDENCE_ITEMS = 7
DEFAULT_RELEVANCE_THRESHOLD = 0.0


def create_request_id() -> str:
    """Create an opaque request ID for one evidence bundle."""

    return uuid4().hex


def _score(item: EvidenceItem) -> float | None:
    return item.rerank_score if item.rerank_score is not None else item.retrieval_score


def _unique_in_order(items: Iterable[EvidenceItem]) -> list[EvidenceItem]:
    unique: list[EvidenceItem] = []
    seen: set[str] = set()
    for item in items:
        if item.evidence_id in seen:
            continue
        seen.add(item.evidence_id)
        unique.append(item)
    return unique


def evidence_is_sufficient(
    user_query: str,
    items: list[EvidenceItem],
    *,
    query_plan: QueryPlan | None = None,
    relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
) -> tuple[bool, str | None]:
    """Apply deterministic pre-generation checks to selected candidates."""

    if not items:
        return False, "The transcripts do not contain any retrieved evidence."

    scored = [score for item in items if (score := _score(item)) is not None]
    if scored and all(score < relevance_threshold for score in scored):
        return False, "The retrieved evidence is below the relevance threshold."

    if query_plan is not None and query_plan.requires_all_countries:
        required = {country.casefold() for country in query_plan.countries}
        present = {item.country.casefold() for item in items}
        display_names = {
            country.casefold(): country for country in query_plan.countries
        }
        missing = sorted(
            (display_names.get(country, country) for country in required - present),
            key=str.casefold,
        )
        if missing:
            return False, f"Evidence is missing required countries: {', '.join(missing)}."

    if query_plan is not None and query_plan.intent == "comparison":
        if len({item.country.casefold() for item in items}) < 2:
            return False, "A comparison requires evidence from at least two countries."

    if query_plan is not None and query_plan.intent == "multi_part":
        evidence_text = " ".join(item.text.casefold() for item in items)
        uncovered_parts = [
            search_query
            for search_query in query_plan.search_queries
            if not any(
                token in evidence_text
                for token in search_query.casefold().split()
                if len(token) > 2
            )
        ]
        if uncovered_parts:
            return False, "The retrieved evidence does not cover every part of the question."

    return True, None


def select_evidence(
    candidates: Iterable[EvidenceItem],
    *,
    query_plan: QueryPlan | None = None,
    max_items: int = DEFAULT_MAX_EVIDENCE_ITEMS,
) -> list[EvidenceItem]:
    """Select a bounded, order-preserving evidence set after reranking."""

    if max_items <= 0:
        return []
    unique = _unique_in_order(candidates)

    # Preserve country coverage for comparison plans whenever the candidate
    # pool contains a suitable item and the size limit allows it.
    required_countries: set[str] = set()
    if query_plan is not None and query_plan.requires_all_countries:
        required_countries = {country.casefold() for country in query_plan.countries}
    elif query_plan is not None and query_plan.intent == "comparison":
        required_countries = {item.country.casefold() for item in unique}

    if not required_countries:
        return unique[:max_items]

    # Allocate evidence evenly by country before filling remaining slots by
    # rank. This prevents a large or unusually similar market from dominating.
    country_order = (
        [country.casefold() for country in query_plan.countries]
        if query_plan is not None and query_plan.requires_all_countries
        else list(dict.fromkeys(item.country.casefold() for item in unique))
    )
    country_order = [country for country in country_order if country in required_countries]
    buckets = {
        country: [item for item in unique if item.country.casefold() == country]
        for country in country_order
    }
    base_quota, extra_quota = divmod(max_items, max(1, len(country_order)))
    selected: list[EvidenceItem] = []
    for index, country in enumerate(country_order):
        quota = base_quota + (1 if index < extra_quota else 0)
        selected.extend(buckets[country][:quota])
    selected_ids = {item.evidence_id for item in selected}
    selected.extend(
        item for item in unique if item.evidence_id not in selected_ids
    )
    return selected[:max_items]


def create_evidence_bundle(
    query: str,
    items: Iterable[EvidenceItem],
    *,
    request_id: str | None = None,
) -> EvidenceBundle:
    """Materialize the exact reranked candidates used for generation/validation."""

    ordered_items = _unique_in_order(items)
    if not ordered_items:
        raise ValueError("Cannot create an EvidenceBundle without evidence items")
    return EvidenceBundle(
        request_id=request_id or create_request_id(),
        query=query,
        items={item.evidence_id: item for item in ordered_items},
        ordered_ids=[item.evidence_id for item in ordered_items],
    )
