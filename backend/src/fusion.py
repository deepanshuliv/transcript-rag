"""Reciprocal rank fusion for dense and lexical retrieval results."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from .schemas import EvidenceItem


def reciprocal_rank_fusion(
    result_lists: Sequence[Iterable[EvidenceItem]],
    *,
    rank_constant: int = 60,
) -> list[EvidenceItem]:
    """Fuse ranked lists, deduplicating by evidence ID.

    Ties retain first-seen order, making the output deterministic when dense
    and BM25 scores are equal or unavailable.
    """

    if rank_constant <= 0:
        raise ValueError("rank_constant must be positive")

    scores: dict[str, float] = {}
    first_items: dict[str, EvidenceItem] = {}
    first_seen: dict[str, int] = {}
    next_seen = 0
    for ranked_items in result_lists:
        for rank, item in enumerate(ranked_items, start=1):
            evidence_id = item.evidence_id
            if evidence_id not in first_items:
                first_items[evidence_id] = item
                first_seen[evidence_id] = next_seen
                next_seen += 1
            scores[evidence_id] = scores.get(evidence_id, 0.0) + (
                1.0 / (rank_constant + rank)
            )

    ordered_ids = sorted(
        scores,
        key=lambda evidence_id: (-scores[evidence_id], first_seen[evidence_id]),
    )
    return [
        first_items[evidence_id].model_copy(
            update={"retrieval_score": scores[evidence_id]}
        )
        for evidence_id in ordered_ids
    ]
