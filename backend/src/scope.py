"""Deterministic country and expert scoping for transcript retrieval."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .ingest import LocalIndex


COUNTRY_ALIASES: dict[str, tuple[str, ...]] = {
    "France": (
        "france",
        "frankreich",
        "french market",
        "french expert",
        "french interview",
        "french transcript",
    ),
    "Germany": (
        "germany",
        "deutschland",
        "allemagne",
        "german market",
        "german expert",
        "german interview",
        "german transcript",
        "german hospital",
    ),
    "United Kingdom": (
        "united kingdom",
        "vereinigtes königreich",
        "royaume-uni",
        "royaume uni",
        "großbritannien",
        "grossbritannien",
        "uk",
        "u.k.",
        "britain",
        "british market",
        "british expert",
        "british interview",
        "british transcript",
        "england",
        "english market",
    ),
}

ALL_MARKETS_PATTERNS = (
    r"\ball\s+(?:three|3)\s+(?:markets|countries|experts|interviews|transcripts)\b",
    r"\bcompare\s+all\s+(?:three|3)\b",
    r"\bevery\s+(?:market|country|expert|interview|transcript)\b",
    r"\bacross\s+europe\b",
)
RESPONSE_LANGUAGE_PATTERNS = (
    r"\b(?:answer|respond|reply|write|translate)\s+in\s+(?:german|french|english)\b",
    r"^\s*in\s+(?:german|french|english)\b",
)


def available_countries(index: "LocalIndex") -> list[str]:
    return sorted({chunk.country for chunk in index.chunks}, key=str.casefold)


def _contains_phrase(text: str, phrase: str) -> bool:
    escaped = re.escape(phrase.casefold())
    return re.search(rf"(?<!\w){escaped}(?!\w)", text.casefold()) is not None


def canonicalize_countries(
    values: Iterable[str],
    available: list[str],
) -> list[str]:
    """Map planner or UI country values to the transcript's canonical labels."""

    canonical: list[str] = []
    for value in values:
        normalized = value.strip().casefold()
        direct = next((country for country in available if country.casefold() == normalized), None)
        if direct:
            if direct not in canonical:
                canonical.append(direct)
            continue
        aliases = next(
            (
                aliases
                for country, aliases in COUNTRY_ALIASES.items()
                if normalized == country.casefold() or normalized in aliases
            ),
            (),
        )
        match = next(
            (
                country
                for country in available
                if any(_contains_phrase(country, alias) for alias in aliases)
                or any(
                    _contains_phrase(alias, country.casefold())
                    for alias in aliases
                )
            ),
            None,
        )
        if match and match not in canonical:
            canonical.append(match)
    return canonical


def infer_country_scope(query: str, index: "LocalIndex") -> list[str] | None:
    """Infer a country scope from explicit market names or expert names."""

    available = available_countries(index)
    lowered = query.casefold()
    if any(re.search(pattern, lowered) for pattern in ALL_MARKETS_PATTERNS):
        return available

    selected: list[str] = []
    for canonical_name, aliases in COUNTRY_ALIASES.items():
        if canonical_name not in available:
            continue
        if any(_contains_phrase(lowered, alias) for alias in aliases):
            selected.append(canonical_name)

    for expert, country in {
        (chunk.expert, chunk.country) for chunk in index.chunks
    }:
        name_parts = [
            part.casefold().strip(".,")
            for part in expert.split()
            if part.casefold().strip(".,") not in {"dr", "doctor"}
        ]
        if any(part and _contains_phrase(lowered, part) for part in name_parts):
            selected.append(country)

    unique = list(dict.fromkeys(selected))
    return unique or None


def resolve_country_scope(
    index: "LocalIndex",
    *,
    requested: list[str] | None,
    query: str,
    planned: list[str] | None = None,
    require_all: bool = False,
) -> list[str] | None:
    """Resolve explicit UI scope, query mentions, and planner hints safely."""

    available = available_countries(index)
    if requested is not None:
        resolved = canonicalize_countries(requested, available)
        if len(resolved) != len(set(requested)):
            raise ValueError(
                "Country scope must contain only markets present in the transcript index."
            )
        return resolved

    explicit = infer_country_scope(query, index)
    if explicit:
        return explicit

    # The caller's all-market policy outranks a planner's best-guess scope.
    # Planners often choose one country even when the user asked a broad
    # question; allowing that hint through silently recreates UK-only answers.
    if require_all:
        return available

    if any(re.search(pattern, query.casefold()) for pattern in RESPONSE_LANGUAGE_PATTERNS):
        return None

    planned_scope = canonicalize_countries(planned or [], available)
    if planned_scope:
        return planned_scope

    return None


def infer_expert_scope(query: str, index: "LocalIndex") -> list[str] | None:
    """Restrict retrieval when a full name or distinctive name part is used."""

    lowered = query.casefold()
    selected = []
    for expert in dict.fromkeys(chunk.expert for chunk in index.chunks):
        name_parts = [
            part.casefold().strip(".,")
            for part in expert.split()
            if part.casefold().strip(".,") not in {"dr", "doctor"}
        ]
        if any(part and _contains_phrase(lowered, part) for part in name_parts):
            selected.append(expert)
    return selected or None
